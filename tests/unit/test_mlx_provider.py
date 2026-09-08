# tests/unit/test_mlx_provider.py
"""
Unit tests for MLXProvider -- the core Apple Silicon provider.

All tests use the mock_mlx / mock_mlx_provider fixtures from conftest.py
so they run on any platform without MLX installed.
"""
import importlib
import sys
import threading
import time

import pytest
from heylook_llm.samplers import GLOBAL_SAMPLER_FLOOR

from heylook_llm.config import ChatMessage, ChatRequest
from helpers.mlx_mock import create_mock_model, create_mock_processor, create_mock_vlm_model


@pytest.mark.unit
class TestGenerationStreamThreadLocal:
    """The module-level generation stream must be thread-local.

    Generation runs on FastAPI's thread pool (asyncio.to_thread /
    run_in_executor), not the import thread. MLX streams are thread-local:
    a stream from mx.new_stream() is bound to the thread that created it, so
    synchronizing it from a pool worker raises
    'There is no Stream(gpu, 0) in current thread.' -- every VLM/text request
    fails. mx.new_thread_local_stream() materializes the stream per-thread
    (this is what mlx_lm.generate uses), so it is valid on any worker.
    """

    def test_module_uses_thread_local_stream(self, mock_mlx):  # noqa: ARG002
        # Force a fresh import so module-level stream creation runs under the mock.
        mx = sys.modules["mlx.core"]
        mx.new_thread_local_stream.reset_mock()  # ignore any earlier import
        mx.new_stream.reset_mock()
        sys.modules.pop("heylook_llm.providers.mlx_provider", None)
        importlib.import_module("heylook_llm.providers.mlx_provider")

        mx.new_thread_local_stream.assert_called_once_with(mx.default_device.return_value)
        mx.new_stream.assert_not_called()


@pytest.mark.unit
class TestMLXProviderInit:
    def test_init_sets_model_id(self, mock_mlx_provider):
        assert mock_mlx_provider.model_id == "test-model"

    def test_init_defaults(self, mock_mlx_provider):
        assert mock_mlx_provider._active_generations == 0
        assert mock_mlx_provider.model is None
        assert mock_mlx_provider.processor is None
        assert mock_mlx_provider.draft_model is None

    def test_init_text_only_not_vlm(self, mock_mlx_provider):
        assert mock_mlx_provider.is_vlm is False

    def test_init_vlm_flag(self, mock_vlm_provider):
        assert mock_vlm_provider.is_vlm is True

    def test_init_strategies_empty_before_load(self, mock_mlx_provider):
        assert mock_mlx_provider._strategies == {}

    def test_init_with_config_values(self, mock_mlx):  # noqa: ARG001
        from heylook_llm.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_id="custom",
            config={
                "model_path": "/my/model",
                "vision": True,
                "enable_thinking": True,
                "max_tokens": 2048,
            },
            verbose=True,
        )
        assert provider.model_id == "custom"
        assert provider.is_vlm is True
        assert provider.verbose is True
        assert provider.config["enable_thinking"] is True


@pytest.mark.unit
class TestStrategyCompilation:
    def test_text_only_strategy_compiled(self, mock_mlx_provider):
        """After _compile_strategies, text-only provider has 'text' strategy."""
        mock_mlx_provider._compile_strategies()
        assert "text" in mock_mlx_provider._strategies

    def test_vlm_strategies_compiled(self, mock_vlm_provider):
        """VLM provider has 'text' and 'vision' strategies."""
        mock_vlm_provider._compile_strategies()
        assert "text" in mock_vlm_provider._strategies
        assert "vision" in mock_vlm_provider._strategies

    def test_text_only_no_vision_strategy(self, mock_mlx_provider):
        mock_mlx_provider._compile_strategies()
        assert "vision" not in mock_mlx_provider._strategies

    def test_text_strategy_is_vlm_flag(self, mock_vlm_provider):
        """VLM provider's text strategy should have is_vlm=True."""
        mock_vlm_provider._compile_strategies()
        assert mock_vlm_provider._strategies['text'].is_vlm is True

    def test_text_only_strategy_not_vlm(self, mock_mlx_provider):
        """Text-only provider's text strategy should have is_vlm=False."""
        mock_mlx_provider._compile_strategies()
        assert mock_mlx_provider._strategies['text'].is_vlm is False

    def test_no_diffusion_strategy_by_default(self, mock_vlm_provider):
        """An ordinary VLM must not get the denoising path."""
        mock_vlm_provider._compile_strategies()
        assert "diffusion" not in mock_vlm_provider._strategies

    def test_diffusion_strategy_compiled_when_detected(self, mock_vlm_provider):
        """A diffusion checkpoint registers 'diffusion' alongside 'text'.

        'text' stays registered even for diffusion models: warmup resolves its
        generation model through UnifiedTextStrategy._get_generation_model.
        """
        mock_vlm_provider.is_diffusion = True
        mock_vlm_provider._compile_strategies()
        assert "diffusion" in mock_vlm_provider._strategies
        assert "text" in mock_vlm_provider._strategies


@pytest.mark.unit
class TestDiffusionDetection:
    """Diffusion routing decisions.

    A masked-diffusion checkpoint driven by mlx-lm's autoregressive
    stream_generate emits ZERO tokens (it samples one meaningless token from
    the last prompt position, which lands on EOS), so these assertions guard a
    silent empty-response bug, not a crash.
    """

    def test_defaults_to_autoregressive(self, mock_mlx_provider):
        assert mock_mlx_provider.is_diffusion is False

    def test_detect_returns_false_when_predicate_unavailable(self, mock_mlx_provider):
        """Detection is best-effort: a predicate failure degrades to the AR path."""
        mock_mlx_provider.model = object()  # no config, no language_model
        assert mock_mlx_provider._detect_diffusion() is False

    def test_diffusion_wins_over_vision_routing(self, mock_vlm_provider):
        """Diffusion takes images inline -- there is no separate vision split."""
        mock_vlm_provider.is_diffusion = True
        mock_vlm_provider._compile_strategies()
        # The route picks 'diffusion' before the is_vlm/has_images branch, so a
        # diffusion provider must never be able to fall through to 'vision'.
        assert mock_vlm_provider._strategies["diffusion"] is not mock_vlm_provider._strategies.get("vision")


@pytest.mark.unit
class TestDetectImages:
    def test_no_images_text_content(self, mock_mlx_provider):
        messages = [ChatMessage(role="user", content="Hello")]
        assert mock_mlx_provider._detect_images_optimized(messages) is False

    def test_images_detected(self, mock_mlx_provider, sample_multimodal_request):
        assert mock_mlx_provider._detect_images_optimized(
            sample_multimodal_request.messages
        ) is True

    def test_text_only_multipart_no_images(self, mock_mlx_provider):
        """Multipart content with only text parts should not detect images."""
        from heylook_llm.config import TextContentPart

        messages = [
            ChatMessage(
                role="user",
                content=[TextContentPart(type="text", text="just text")],
            )
        ]
        assert mock_mlx_provider._detect_images_optimized(messages) is False


@pytest.mark.unit
class TestApplyModelDefaults:
    def test_defaults_applied(self, mock_mlx_provider):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        effective = mock_mlx_provider._apply_model_defaults(req)
        assert effective["temperature"] == GLOBAL_SAMPLER_FLOOR["temperature"]  # global floor
        assert effective["max_tokens"] == 4096

    def test_request_overrides_defaults(self, mock_mlx_provider):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.8,
            max_tokens=1024,
        )
        effective = mock_mlx_provider._apply_model_defaults(req)
        assert effective["temperature"] == 0.8
        assert effective["max_tokens"] == 1024

    def test_thinking_mode_defaults(self, mock_mlx):  # noqa: ARG001
        """Model-config thinking sets the switch and nothing else.

        Decode tuning comes from the vendor layer or the floor -- never a
        hardcode tuned for one family and wrong for the rest (gemma wants
        1.0/64, not Qwen's 0.6/20). Since v2.0.32 that includes
        presence_penalty, which the overlay used to add here.
        """
        from heylook_llm.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_id="think-model",
            config={
                "model_path": "/fake",
                "vision": False,
                "enable_thinking": True,
            },
            verbose=False,
        )
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="think about this")],
        )
        effective = provider._apply_model_defaults(req)
        assert effective["enable_thinking"] is True
        assert effective["presence_penalty"] == GLOBAL_SAMPLER_FLOOR["presence_penalty"]
        assert effective["temperature"] == GLOBAL_SAMPLER_FLOOR["temperature"]  # floor, NOT Qwen's 0.6

    def test_request_thinking_reaches_the_prompt_without_a_sampler_change(self, mock_mlx):  # noqa: ARG001
        """A request flipping thinking ON resolves the switch even when the
        model config never declares it -- keying the layer on model config
        alone made it dead code, since nothing sets it. What it must NOT do
        any more is change sampling on the way (v2.0.32).
        """
        from heylook_llm.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_id="think-model",
            config={"model_path": "/fake", "vision": False},
            verbose=False,
        )
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            enable_thinking=True,
        )
        effective = provider._apply_model_defaults(req)
        assert effective["enable_thinking"] is True
        assert effective["presence_penalty"] == GLOBAL_SAMPLER_FLOOR["presence_penalty"]

    def test_request_thinking_false_suppresses_overlay(self, mock_mlx):  # noqa: ARG001
        """Claim: request enable_thinking=False beats a thinking-on model
        config -- no loop penalty rides a non-thinking generation."""
        from heylook_llm.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_id="think-model",
            config={
                "model_path": "/fake",
                "vision": False,
                "enable_thinking": True,
            },
            verbose=False,
        )
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            enable_thinking=False,
        )
        effective = provider._apply_model_defaults(req)
        assert effective["presence_penalty"] == 0.0
        assert effective["enable_thinking"] is False

    def test_vendor_generation_config_layer(self, mock_mlx, tmp_path):  # noqa: ARG001
        """Claim: the model dir's generation_config.json supplies per-model
        decode tuning above the floor (gemma 1.0/64/0.95 vs the floor);
        without it every model runs one-size floor sampling."""
        import json

        from heylook_llm.providers.mlx_provider import MLXProvider

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"temperature": 1.0, "top_k": 64, "top_p": 0.95,
                        "do_sample": True, "eos_token_id": [1, 106]})
        )
        provider = MLXProvider(
            model_id="vendor-model",
            config={"model_path": str(tmp_path), "vision": False},
            verbose=False,
        )
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        effective = provider._apply_model_defaults(req)
        assert effective["temperature"] == 1.0
        assert effective["top_k"] == 64
        assert effective["top_p"] == 0.95

    def test_models_toml_overrides_vendor(self, mock_mlx, tmp_path):  # noqa: ARG001
        """Claim: operator fields in models.toml stay above the vendor layer."""
        import json

        from heylook_llm.providers.mlx_provider import MLXProvider

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"temperature": 1.0})
        )
        provider = MLXProvider(
            model_id="vendor-model",
            config={"model_path": str(tmp_path), "vision": False,
                    "temperature": 0.3},
            verbose=False,
        )
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        effective = provider._apply_model_defaults(req)
        assert effective["temperature"] == 0.3

    def test_config_overrides_thinking_defaults(self, mock_mlx):  # noqa: ARG001
        from heylook_llm.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_id="think-model",
            config={
                "model_path": "/fake",
                "vision": False,
                "enable_thinking": True,
                "temperature": 0.3,  # override thinking default
            },
            verbose=False,
        )
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        effective = provider._apply_model_defaults(req)
        assert effective["temperature"] == 0.3


@pytest.mark.unit
class TestResolveAddGenerationPrompt:
    """Prefill convention: trailing assistant message -> continue it (no new
    generation prompt); anything else -> open a fresh assistant turn."""

    def test_empty_messages_true(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import resolve_add_generation_prompt

        assert resolve_add_generation_prompt([]) is True

    def test_last_message_user_true(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import resolve_add_generation_prompt

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        assert resolve_add_generation_prompt(messages) is True

    def test_last_message_assistant_false(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import resolve_add_generation_prompt

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "partial answer..."},
        ]
        assert resolve_add_generation_prompt(messages) is False

    def test_single_assistant_message_false(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import resolve_add_generation_prompt

        messages = [{"role": "assistant", "content": "prefill"}]
        assert resolve_add_generation_prompt(messages) is False


@pytest.mark.unit
class TestContinuationTemplate:
    """continue_final_message reaches the template call with the right shape:
    add_generation_prompt=False + continue_final_message=True (transformers
    refuses True/True). Suppressing the generation prompt alone was NOT
    continuation -- the turn still rendered CLOSED, so the model saw a
    finished message and nothing to continue; that half-state is the bug this
    class exists to keep dead."""

    class _Tok:
        def __init__(self, reject=()):
            self.calls = []
            self.reject = set(reject)

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            if self.reject & set(kwargs):
                raise TypeError("unexpected keyword argument")
            return "PROMPT"

        def encode(self, s):
            return [1, 2, 3]

    def _apply(self, tok, continuing, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(model_id="m")
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "he"}]
        # What generation does with the render: encode a string, pass a
        # token list through (build_prompt + generate in the provider).
        prompt = strategy._render_template(
            messages, tok, None, None, {"enable_thinking": False},
            continuing=continuing)
        return tok.encode(prompt) if isinstance(prompt, str) else prompt

    def test_continuing_leaves_the_turn_open(self, mock_mlx):
        tok = self._Tok()
        self._apply(tok, True, mock_mlx)
        kwargs = tok.calls[-1]
        assert kwargs["continue_final_message"] is True
        assert kwargs["add_generation_prompt"] is False

    def test_not_continuing_never_passes_the_kwarg(self, mock_mlx):
        # continuing=False here is the resolved EXPLICIT opt-out
        # (continue_final_message=false): the trailing assistant turn renders
        # closed and a FRESH generation prompt opens -- "reply to it", the
        # only meaning "never continue" can coherently have. (Auto mode never
        # reaches this branch with a trailing assistant message.)
        tok = self._Tok()
        self._apply(tok, False, mock_mlx)
        kwargs = tok.calls[-1]
        assert "continue_final_message" not in kwargs
        assert kwargs["add_generation_prompt"] is True

    def test_enable_thinking_fallback_keeps_continuation(self, mock_mlx):
        # A wrapper that rejects enable_thinking must retry WITHOUT it but
        # WITH continue_final_message -- dropping both would silently render
        # a closed turn.
        tok = self._Tok(reject={"enable_thinking"})
        self._apply(tok, True, mock_mlx)
        kwargs = tok.calls[-1]
        assert kwargs["continue_final_message"] is True
        assert "enable_thinking" not in kwargs

    def test_unsupported_continuation_refuses_loudly(self, mock_mlx):
        from heylook_llm.providers.base import InvalidGenerationRequest

        tok = self._Tok(reject={"continue_final_message"})
        with pytest.raises(InvalidGenerationRequest, match="cannot continue"):
            self._apply(tok, True, mock_mlx)


@pytest.mark.unit
class TestMLXPromptSideMatchesReportedThinking:
    """The MLX half of the cross-surface property.

    ``test_thinking_capability.py`` pins it through the gguf provider (which
    runs without the MLX module mocks); this pins the MLX path specifically,
    because MLX's prompt side reads the effective request through its own
    helper AND its cascade call passes a vendor layer that the reported flag's
    does not. Vendor sampling is numeric-only, so the two agree -- if that
    ever stops being true, the prompt and the parser diverge again and this
    is where it surfaces.
    """

    def test_prompt_helper_matches_effective_thinking(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import (
            MLXProvider, _resolve_enable_thinking,
        )

        cases = [{}, {"enable_thinking": True}, {"enable_thinking": False},
                 {"sampler": "thinking"}, {"sampler": "deterministic"}]
        for config in ({"model_path": "/fake", "vision": False},
                       {"model_path": "/fake", "vision": False, "enable_thinking": True},
                       {"model_path": "/fake", "vision": False, "default_sampler": "thinking"}):
            provider = MLXProvider(model_id="m", config=dict(config), verbose=False)
            for kw in cases:
                req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], **kw)
                prompt_side = _resolve_enable_thinking(provider._apply_model_defaults(req))
                assert prompt_side == provider.effective_thinking(req), (config, kw)


@pytest.mark.unit
class TestGetMetrics:
    def test_metrics_with_no_model(self, mock_mlx_provider):
        """get_metrics should still return something even without a loaded model."""
        # model is None, so mx.metal calls will use the mock
        metrics = mock_mlx_provider.get_metrics()
        assert metrics is not None
        assert metrics.requests_active == 0

    def test_metrics_active_requests(self, mock_mlx_provider):
        mock_mlx_provider._active_generations = 3
        try:
            metrics = mock_mlx_provider.get_metrics()
            assert metrics.requests_active == 3
        finally:
            # PUT THE COUNTER BACK. BaseProvider.__del__ calls unload(), whose
            # drain loop polls `time.sleep(0.1)` until the active count reaches
            # zero or a 30s cap expires -- and nothing here is generating, so a
            # counter left at 3 burns the whole cap at garbage-collection time.
            # That single leak was 29s of a 63s suite (289 sleeps, measured),
            # and `--durations` cannot see one second of it: the stall happens
            # in __del__ during GC, outside every phase pytest times. The file
            # reported 2.5s across its 80 duration entries while taking 33.5s.
            mock_mlx_provider._active_generations = 0


@pytest.mark.unit
class TestCollectionDoesNotBlock:
    """A destructor must not wait, and must not tear down live GPU state.

    `BaseProvider.__del__` used to run the full `unload()`, whose drain loop
    polls for up to 30s. Anything collected while its active counter was
    non-zero therefore stalled whatever thread the GC fired on -- in the suite
    that was ~29s of a 65s run from one leaked counter, and in the server it
    would land on whichever thread GC chose, including one delivering tokens.

    Nothing here asserts the counter is ever non-zero in production; a running
    generation holds a reference, so it should not be. This pins what happens
    if that assumption is ever wrong, which is the case a destructor cannot
    afford to get wrong.
    """

    def test_collection_with_traffic_returns_at_once(self, mock_mlx_provider):
        import time as _time
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider._active_generations = 3
        try:
            started = _time.perf_counter()
            mock_mlx_provider.__del__()
            elapsed = _time.perf_counter() - started
            # Two orders of magnitude under the 30s cap: this is asserting
            # "did not wait", not a latency budget.
            assert elapsed < 1.0, (
                f"__del__ blocked for {elapsed:.1f}s with generations in flight -- "
                "the drain loop is running in a destructor again"
            )
            # And it declined to tear down rather than freeing weights
            # mid-decode, which is the fault the drain loop exists to prevent.
            # `unload()` ends in `del self.model`, so the attribute surviving
            # is the observable. `_strategies` is NOT: it is empty on a
            # provider that never loaded, so asserting on it passes whether or
            # not anything was torn down -- which is how the first version of
            # this check failed for the wrong reason.
            assert hasattr(mock_mlx_provider, "model"), (
                "__del__ tore down a provider that still had generations in "
                "flight -- skipping the wait must mean leaving it alone, not "
                "releasing resources out from under a live decode"
            )
        finally:
            mock_mlx_provider._active_generations = 0

    def test_a_quiet_provider_still_unloads_on_collection(self, mock_mlx_provider):
        """The refusal is conditional. With nothing in flight -- the normal
        case -- collection must still release the model, or the change trades
        a stall for a leak."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider._active_generations = 0
        mock_mlx_provider.__del__()
        assert not hasattr(mock_mlx_provider, "model")


@pytest.mark.unit
class TestClearCache:
    def test_clear_cache_calls_manager(self, mock_mlx_provider):
        mock_mlx_provider.model = create_mock_model()
        result = mock_mlx_provider.clear_cache()
        assert result is True


@pytest.mark.unit
class TestUnload:
    def test_unload_clears_state(self, mock_mlx_provider):
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()
        mock_mlx_provider._compile_strategies()

        mock_mlx_provider.unload()

        assert not hasattr(mock_mlx_provider, "model")
        assert not hasattr(mock_mlx_provider, "processor")
        assert mock_mlx_provider._strategies == {}

    def test_unload_immediate_when_idle(self, mock_mlx_provider):
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()

        start = time.time()
        mock_mlx_provider.unload()
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_unload_waits_for_active_generations(self, mock_mlx_provider):
        """unload() should wait for active generations to finish."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()

        # Simulate an active generation
        gen_started = threading.Event()

        def hold_generation():
            with mock_mlx_provider._active_lock:
                mock_mlx_provider._active_generations += 1
            gen_started.set()
            time.sleep(0.3)
            with mock_mlx_provider._active_lock:
                mock_mlx_provider._active_generations -= 1

        t = threading.Thread(target=hold_generation)
        t.start()
        gen_started.wait()

        # Unload should wait for generation to complete
        mock_mlx_provider.unload()
        t.join()

        assert not hasattr(mock_mlx_provider, "model")


@pytest.mark.unit
class TestCreateChatCompletion:
    def test_no_model_loaded_raises(self, mock_mlx_provider):
        """If model is not loaded, the generator raises GenerationFailed
        (typed exceptions, not error-text chunks -- every consumer fails
        loudly by default)."""
        from heylook_llm.providers.base import GenerationFailed
        mock_mlx_provider._compile_strategies()
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        with pytest.raises(GenerationFailed, match="processor not loaded"):
            list(mock_mlx_provider.create_chat_completion(req))

    def test_text_model_rejects_images(self, mock_mlx_provider):
        """Text-only model should reject image inputs."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()
        mock_mlx_provider._compile_strategies()

        from heylook_llm.config import ImageContentPart, ImageUrl, TextContentPart

        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        TextContentPart(type="text", text="describe"),
                        ImageContentPart(
                            type="image_url",
                            image_url=ImageUrl(url="data:image/png;base64,abc"),
                        ),
                    ],
                )
            ],
        )
        from heylook_llm.providers.base import InvalidGenerationRequest
        # Client error (their request can never succeed here) -> the 400-class
        # exception, distinct from server-side GenerationFailed.
        with pytest.raises(InvalidGenerationRequest, match="text-only"):
            list(mock_mlx_provider.create_chat_completion(req))

    def test_generation_gate_released_after_error(self, mock_mlx_provider):
        """The generation gate must release even after errors, so the next
        queued request can run instead of deadlocking."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()
        mock_mlx_provider._compile_strategies()

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )

        # Whether the mocked generation completes or raises, the generator's
        # finally must run: gate released, no deadlock for the next request.
        from heylook_llm.providers.base import GenerationFailed
        try:
            list(mock_mlx_provider.create_chat_completion(req))
        except GenerationFailed:
            pass

        # Slot should be free, and capacity available again.
        assert mock_mlx_provider._gen_gate.busy is False
        mock_mlx_provider.check_capacity()  # no raise

    def test_active_generation_counter_decremented(self, mock_mlx_provider):
        """_active_generations should return to 0 after generation."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()
        mock_mlx_provider._compile_strategies()

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        from heylook_llm.providers.base import GenerationFailed
        try:
            list(mock_mlx_provider.create_chat_completion(req))
        except GenerationFailed:
            pass  # mock strategy may error; the counter must reset either way
        assert mock_mlx_provider._active_generations == 0


@pytest.mark.unit
class TestQueueWaitTagging:
    """create_chat_completion tags each chunk with the FIFO queue-wait time and
    still propagates close() to the inner strategy generator (so the gate
    releases promptly on client disconnect)."""

    class _Chunk:
        def __init__(self, text):
            self.text = text

    def _inject_strategy(self, provider, gen_factory):
        provider.processor = create_mock_processor()
        provider.is_vlm = False

        class _FakeStrategy:
            def generate(self, *a, **k):
                yield from gen_factory()

        provider._strategies = {"text": _FakeStrategy()}

    def test_chunks_tagged_with_queue_wait_ms(self, mock_mlx_provider):
        self._inject_strategy(
            mock_mlx_provider,
            lambda: iter([self._Chunk("hello"), self._Chunk("world")]),
        )
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        chunks = list(mock_mlx_provider.create_chat_completion(req))

        assert [c.text for c in chunks] == ["hello", "world"]
        # Tagged once on the first chunk (the route carries it forward).
        assert hasattr(chunks[0], "queue_wait_ms")
        assert chunks[0].queue_wait_ms >= 0.0
        assert not hasattr(chunks[1], "queue_wait_ms")

    def test_inner_generator_closed_and_gate_released_on_outer_close(self, mock_mlx_provider):
        closed = {"v": False}

        def gen_factory():
            try:
                yield self._Chunk("a")
                yield self._Chunk("b")
            finally:
                closed["v"] = True

        self._inject_strategy(mock_mlx_provider, gen_factory)
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])

        gen = mock_mlx_provider.create_chat_completion(req)
        assert next(gen).text == "a"          # start generation (acquires gate)
        gen.close()                            # client disconnect / early close

        assert closed["v"] is True             # inner strategy generator closed
        assert mock_mlx_provider._gen_gate.busy is False  # gate released


@pytest.mark.unit
class TestPerRequestAbortEvent:
    """Each request must use its OWN abort event, not a shared provider-level
    one -- otherwise one client's disconnect aborts a different client's
    in-flight generation (the FIFO concurrency cross-contamination bug)."""

    class _Chunk:
        def __init__(self, text):
            self.text = text

    def _inject_capturing_strategy(self, provider, sink):
        provider.processor = create_mock_processor()
        provider.is_vlm = False

        class _FakeStrategy:
            def generate(self, *a, abort_event=None, **k):
                sink.append(abort_event)
                yield TestPerRequestAbortEvent._Chunk("hi")

        provider._strategies = {"text": _FakeStrategy()}

    def test_strategy_receives_the_passed_abort_event(self, mock_mlx_provider):
        from heylook_llm.providers.abort import AbortEvent

        seen = []
        self._inject_capturing_strategy(mock_mlx_provider, seen)
        ev = AbortEvent()
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        list(mock_mlx_provider.create_chat_completion(req, abort_event=ev))
        assert seen == [ev]

    def test_each_call_gets_a_distinct_default_event_no_shared_state(self, mock_mlx_provider):
        seen = []
        self._inject_capturing_strategy(mock_mlx_provider, seen)
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        list(mock_mlx_provider.create_chat_completion(req))
        list(mock_mlx_provider.create_chat_completion(req))

        assert seen[0] is not None and seen[1] is not None
        assert seen[0] is not seen[1]  # per-request, not one shared event
        # The shared provider-level abort event must be gone.
        assert not hasattr(mock_mlx_provider, "_abort_event")

    def test_disconnect_of_one_request_does_not_abort_another(self, mock_mlx_provider):
        """A's event being set must not be visible through B's event."""
        from heylook_llm.providers.abort import AbortEvent

        seen = []
        self._inject_capturing_strategy(mock_mlx_provider, seen)
        ev_a, ev_b = AbortEvent(), AbortEvent()
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        list(mock_mlx_provider.create_chat_completion(req, abort_event=ev_a))
        list(mock_mlx_provider.create_chat_completion(req, abort_event=ev_b))

        seen[1].set()  # B aborts
        assert seen[0].is_set() is False  # A unaffected


@pytest.mark.unit
class TestCheckCapacity:
    """check_capacity() applies backpressure (503) via the generation gate."""

    def test_idle_provider_has_capacity(self, mock_mlx_provider):
        mock_mlx_provider.check_capacity()  # no raise

    def test_raises_model_busy_when_queue_full(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import MLXProvider
        from heylook_llm.providers.common.generation_gate import (
            GenerationGate, ModelBusyError,
        )

        provider = MLXProvider(
            model_id="busy",
            config={"model_path": "/fake", "vision": False},
            verbose=False,
        )
        # Inject an isolated single-flight gate (the real gate is a process
        # singleton shared across providers; isolate it for a deterministic test).
        provider._gen_gate = GenerationGate(max_waiting=0)
        provider._gen_gate.acquire()  # simulate an in-flight generation
        try:
            with pytest.raises(ModelBusyError) as exc:
                provider.check_capacity()
            assert "MODEL_BUSY" in str(exc.value)
        finally:
            provider._gen_gate.release()

    def test_config_sets_queue_depth(self, mock_mlx):  # noqa: ARG002
        import heylook_llm.providers.mlx_provider as mp

        # The gate is a process-global singleton: max_queue_depth is read from
        # the FIRST provider created. Reset it so this provider is that first one.
        from heylook_llm.providers.common.generation_gate import reset_process_gate
        reset_process_gate()
        provider = mp.MLXProvider(
            model_id="d",
            config={"model_path": "/fake", "vision": False, "max_queue_depth": 3},
            verbose=False,
        )
        assert provider._gen_gate.max_waiting == 3


@pytest.mark.unit
class TestApplyModelDefaultsGetattr:
    """Verify _apply_model_defaults uses getattr instead of model_dump()."""

    def test_seed_from_request(self, mock_mlx_provider):
        """Seed should be extracted from request via getattr."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            seed=42,
        )
        effective = mock_mlx_provider._apply_model_defaults(req)
        assert effective["seed"] == 42

    def test_none_fields_excluded(self, mock_mlx_provider):
        """Fields that are None on the request should not override defaults."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        effective = mock_mlx_provider._apply_model_defaults(req)
        # temperature is not set on request, so default should apply
        assert effective["temperature"] == GLOBAL_SAMPLER_FLOOR["temperature"]

    def test_all_scalar_fields_extracted(self, mock_mlx_provider):
        """All 9 scalar fields should be extractable from request."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5,
            top_p=0.9,
            top_k=10,
            min_p=0.05,
            max_tokens=256,
            repetition_penalty=1.2,
            presence_penalty=0.5,
            enable_thinking=True,
            seed=123,
        )
        effective = mock_mlx_provider._apply_model_defaults(req)
        assert effective["temperature"] == 0.5
        assert effective["top_p"] == 0.9
        assert effective["top_k"] == 10
        assert effective["min_p"] == 0.05
        assert effective["max_tokens"] == 256
        assert effective["repetition_penalty"] == 1.2
        assert effective["presence_penalty"] == 0.5
        assert effective["enable_thinking"] is True
        assert effective["seed"] == 123


@pytest.mark.unit
class TestNoContentCache:
    """Verify _content_cache has been removed."""

    def test_no_content_cache_attribute(self, mock_mlx_provider):
        """MLXProvider should no longer have _content_cache."""
        assert not hasattr(mock_mlx_provider, "_content_cache")

    def test_detect_images_no_caching(self, mock_mlx_provider):
        """_detect_images_optimized should work without caching."""
        messages = [ChatMessage(role="user", content="hello")]
        # Call twice -- should work fine without cache
        assert mock_mlx_provider._detect_images_optimized(messages) is False
        assert mock_mlx_provider._detect_images_optimized(messages) is False


@pytest.mark.unit
class TestUnifiedTextStrategy:
    """Verify UnifiedTextStrategy for both text-only and VLM text paths."""

    def test_cached_wrapper_none_initially(self, mock_mlx):  # noqa: ARG001
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy
        strategy = UnifiedTextStrategy(draft_model=None, model_id="test-vlm", is_vlm=True)
        assert strategy._cached_wrapper is None  # None until generate() called

    def test_has_cache_manager(self, mock_mlx):  # noqa: ARG001
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy
        strategy = UnifiedTextStrategy(draft_model=None, model_id="test-vlm")
        assert strategy.cache_manager is not None

    def test_no_cached_generator(self, mock_mlx):  # noqa: ARG001
        """UnifiedTextStrategy should not have _cached_generator (only VLMVisionStrategy uses it)."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy
        strategy = UnifiedTextStrategy(draft_model=None, model_id="test-vlm")
        assert not hasattr(strategy, '_cached_generator')

    def test_text_only_mode(self, mock_mlx):  # noqa: ARG001
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy
        strategy = UnifiedTextStrategy(draft_model=None, model_id="test", is_vlm=False)
        assert strategy.is_vlm is False
        assert strategy._cached_wrapper is None

    def test_vlm_mode(self, mock_mlx):  # noqa: ARG001
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy
        strategy = UnifiedTextStrategy(draft_model=None, model_id="test", is_vlm=True)
        assert strategy.is_vlm is True


@pytest.mark.unit
class TestWarmupModelWrapping:
    """Regression: warmup must wrap VLM models the same way real requests do.

    Bug: warmup() passed the full VLM model directly to generate_text. A VLM's
    forward pass returns a LanguageModelOutput, but mlx-lm's generate_step does
    `logits = logits[:, -1, :]`, raising 'LanguageModelOutput' object is not
    subscriptable. Real requests avoid this by wrapping model.language_model in
    LanguageModelLogitsWrapper (via wrap_language_model); warmup must do the same
    so VLMs are actually JIT-primed instead of silently failing warmup.
    """

    def _capture_warmup_model(self, provider):
        """Run warmup with generate_text patched; return the model it received."""
        from unittest.mock import patch

        provider._compile_strategies()
        captured = {}

        def fake_generate_text(model, *args, **kwargs):  # noqa: ARG001
            captured['model'] = model
            return iter(())

        with patch(
            'heylook_llm.providers.common.generation_core.generate_text',
            side_effect=fake_generate_text,
        ):
            provider.warmup()
        return captured

    def test_vlm_warmup_wraps_language_model(self, mock_vlm_provider):
        """VLM warmup passes a wrapped model, not the raw VLM.

        The raw VLM is what triggers the LanguageModelOutput subscript crash, so
        the regression guard is simply: warmup must not hand generate_text the
        raw model. (Under mocked MLX, nn.Module is a MagicMock, so isinstance
        against the real wrapper class can't be used here.)
        """
        provider = mock_vlm_provider
        provider.model = create_mock_vlm_model()
        provider.processor = create_mock_processor()

        captured = self._capture_warmup_model(provider)

        assert 'model' in captured, "warmup never reached generate_text"
        gen_model = captured['model']
        assert gen_model is not provider.model  # wrapped, not the raw VLM
        assert gen_model is not provider.model.language_model  # wrapped, not the bare LM

    def test_text_only_warmup_uses_raw_model(self, mock_mlx_provider):
        """Text-only warmup passes the raw model unchanged (no wrapper)."""
        provider = mock_mlx_provider
        provider.model = create_mock_model()
        provider.processor = create_mock_processor()

        captured = self._capture_warmup_model(provider)

        assert 'model' in captured, "warmup never reached generate_text"
        assert captured['model'] is provider.model
