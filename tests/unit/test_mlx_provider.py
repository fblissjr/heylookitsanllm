# tests/unit/test_mlx_provider.py
"""
Unit tests for MLXProvider -- the core Apple Silicon provider.

All tests use the mock_mlx / mock_mlx_provider fixtures from conftest.py
so they run on any platform without MLX installed.
"""
import sys
import threading
import time

import pytest
from heylook_llm.samplers import GLOBAL_SAMPLER_FLOOR

from heylook_llm.config import ChatMessage, ChatRequest
from helpers.mlx_mock import create_mock_model, create_mock_processor


def _fixture(name):
    """A row builder that returns the named conftest fixture's value."""
    return lambda request: request.getfixturevalue(name)


def _custom_provider(request):
    request.getfixturevalue("mock_mlx")
    from heylook_llm.providers.mlx_provider import MLXProvider

    return MLXProvider(
        model_id="custom",
        config={
            "model_path": "/my/model",
            "vision": True,
            "enable_thinking": True,
            "max_tokens": 2048,
        },
        verbose=True,
    )


def _read(provider, path):
    """``attr`` or ``config.key`` off a provider."""
    if path.startswith("config."):
        return provider.config[path.split(".", 1)[1]]
    return getattr(provider, path)


@pytest.mark.unit
class TestMLXProviderInit:
    # One row per constructor fact. bool/None expectations are identity checks.
    @pytest.mark.parametrize("build, expected", [
        (_fixture("mock_mlx_provider"), {"model_id": "test-model"}),
        (_fixture("mock_mlx_provider"),
         {"_active_generations": 0, "model": None, "processor": None}),
        (_fixture("mock_mlx_provider"), {"is_vlm": False}),
        (_fixture("mock_vlm_provider"), {"is_vlm": True}),
        (_fixture("mock_mlx_provider"), {"_strategies": {}}),
        (_custom_provider, {"model_id": "custom", "is_vlm": True, "verbose": True,
                            "config.enable_thinking": True}),
    ], ids=["init_sets_model_id", "init_defaults", "init_text_only_not_vlm",
            "init_vlm_flag", "init_strategies_empty_before_load",
            "init_with_config_values"])
    def test_constructor_state(self, request, build, expected):
        provider = build(request)
        for path, want in expected.items():
            got = _read(provider, path)
            if want is None or isinstance(want, bool):
                assert got is want, path
            else:
                assert got == want, path


@pytest.mark.unit
class TestStrategyCompilation:
    """Which strategies _compile_strategies registers, by is_vlm x is_diffusion.

    Diffusion rows guard a silent empty-response bug, not a crash: a
    masked-diffusion checkpoint driven by mlx-lm's autoregressive
    stream_generate emits ZERO tokens (it samples one meaningless token from
    the last prompt position, which lands on EOS).
    """

    @pytest.mark.parametrize("provider_fixture, diffusion, compile_, check", [
        # text-only provider has a 'text' strategy
        ("mock_mlx_provider", None, True, lambda s, p: "text" in s),
        # VLM provider has 'text' and 'vision'
        ("mock_vlm_provider", None, True, lambda s, p: "text" in s and "vision" in s),
        ("mock_mlx_provider", None, True, lambda s, p: "vision" not in s),
        # VLM provider's text strategy carries is_vlm=True ...
        ("mock_vlm_provider", None, True, lambda s, p: s["text"].is_vlm is True),
        # ... and a text-only provider's is_vlm=False
        ("mock_mlx_provider", None, True, lambda s, p: s["text"].is_vlm is False),
        # an ordinary VLM must not get the denoising path
        ("mock_vlm_provider", None, True, lambda s, p: "diffusion" not in s),
        # a diffusion checkpoint registers 'diffusion' alongside 'text'; 'text'
        # stays because warmup resolves its generation model through
        # UnifiedTextStrategy._get_generation_model
        ("mock_vlm_provider", True, True, lambda s, p: "diffusion" in s and "text" in s),
        # detection defaults to the autoregressive path
        ("mock_mlx_provider", None, False, lambda s, p: p.is_diffusion is False),
        # diffusion takes images inline: the route picks 'diffusion' before the
        # is_vlm/has_images branch, so it must never fall through to 'vision'
        ("mock_vlm_provider", True, True,
         lambda s, p: s["diffusion"] is not s.get("vision")),
    ], ids=["text_only_strategy_compiled", "vlm_strategies_compiled",
            "text_only_no_vision_strategy", "text_strategy_is_vlm_flag",
            "text_only_strategy_not_vlm", "no_diffusion_strategy_by_default",
            "diffusion_strategy_compiled_when_detected", "defaults_to_autoregressive",
            "diffusion_wins_over_vision_routing"])
    def test_compiled_strategies(self, request, provider_fixture, diffusion, compile_, check):
        provider = request.getfixturevalue(provider_fixture)
        if diffusion is not None:
            provider.is_diffusion = diffusion
        if compile_:
            provider._compile_strategies()
        assert check(provider._strategies, provider)


@pytest.mark.unit
class TestDiffusionDetection:
    def test_detect_returns_false_when_predicate_unavailable(self, mock_mlx_provider):
        """Detection is best-effort: a predicate failure degrades to the AR path
        (see TestStrategyCompilation for why that path matters)."""
        mock_mlx_provider.model = object()  # no config, no language_model
        assert mock_mlx_provider._detect_diffusion() is False


def _text_parts_only(request):
    from heylook_llm.config import TextContentPart

    return [ChatMessage(role="user",
                        content=[TextContentPart(type="text", text="just text")])]


@pytest.mark.unit
class TestDetectImages:
    @pytest.mark.parametrize("build_messages, expected, calls", [
        (lambda r: [ChatMessage(role="user", content="Hello")], False, 1),
        (lambda r: r.getfixturevalue("sample_multimodal_request").messages, True, 1),
        # multipart content with only text parts is not an image
        (_text_parts_only, False, 1),
        # asked twice: the answer holds with no content cache (_content_cache
        # was removed)
        (lambda r: [ChatMessage(role="user", content="hello")], False, 2),
    ], ids=["no_images_text_content", "images_detected",
            "text_only_multipart_no_images", "detect_images_no_caching"])
    def test_detect_images(self, request, mock_mlx_provider, build_messages, expected, calls):
        messages = build_messages(request)
        for _ in range(calls):
            assert mock_mlx_provider._detect_images_optimized(messages) is expected


_FLOOR = GLOBAL_SAMPLER_FLOOR
_THINK_ON = {"model_path": "/fake", "vision": False, "enable_thinking": True}


@pytest.mark.unit
class TestApplyModelDefaults:
    # (model_id, config) builds a provider; None uses the mock_mlx_provider
    # fixture. vendor, when set, is written as the model dir's
    # generation_config.json (model_path then points at that dir).
    # bool expectations are identity checks.
    @pytest.mark.parametrize("provider_spec, vendor, request_kwargs, expected", [
        (None, None, {},
         {"temperature": _FLOOR["temperature"], "max_tokens": _FLOOR["max_tokens"]}),
        (None, None, {"temperature": 0.8, "max_tokens": 1024},
         {"temperature": 0.8, "max_tokens": 1024}),
        # model-config thinking sets the switch and nothing else: decode tuning
        # comes from the vendor layer or the floor, never a hardcode tuned for
        # one family (gemma wants 1.0/64, not Qwen's 0.6/20); since v2.0.32 that
        # includes the presence_penalty the overlay used to add
        (("think-model", _THINK_ON), None, {},
         {"enable_thinking": True, "presence_penalty": _FLOOR["presence_penalty"],
          "temperature": _FLOOR["temperature"]}),
        # a request flipping thinking ON resolves the switch though the model
        # config never declares it (keying on model config alone made the layer
        # dead code) and changes no sampling on the way (v2.0.32)
        (("think-model", {"model_path": "/fake", "vision": False}), None,
         {"enable_thinking": True},
         {"enable_thinking": True, "presence_penalty": _FLOOR["presence_penalty"]}),
        # request enable_thinking=False beats a thinking-on model config: no
        # loop penalty rides a non-thinking generation
        (("think-model", _THINK_ON), None, {"enable_thinking": False},
         {"presence_penalty": 0.0, "enable_thinking": False}),
        # operator fields in models.toml stay above the vendor layer
        (("vendor-model", {"vision": False, "temperature": 0.3}), {"temperature": 1.0},
         {}, {"temperature": 0.3}),
        (("think-model", {**_THINK_ON, "temperature": 0.3}), None, {},
         {"temperature": 0.3}),
        # fields are read off the request with getattr, not model_dump()
        (None, None, {"seed": 42}, {"seed": 42}),
        # a None request field does not override the default
        (None, None, {}, {"temperature": _FLOOR["temperature"]}),
        # every scalar sampler field is extractable from the request
        (None, None,
         {"temperature": 0.5, "top_p": 0.9, "top_k": 10, "min_p": 0.05,
          "max_tokens": 256, "repetition_penalty": 1.2, "presence_penalty": 0.5,
          "enable_thinking": True, "seed": 123},
         {"temperature": 0.5, "top_p": 0.9, "top_k": 10, "min_p": 0.05,
          "max_tokens": 256, "repetition_penalty": 1.2, "presence_penalty": 0.5,
          "enable_thinking": True, "seed": 123}),
    ], ids=["defaults_applied", "request_overrides_defaults", "thinking_mode_defaults",
            "request_thinking_reaches_the_prompt_without_a_sampler_change",
            "request_thinking_false_suppresses_overlay", "models_toml_overrides_vendor",
            "config_overrides_thinking_defaults", "seed_from_request",
            "none_fields_excluded", "all_scalar_fields_extracted"])
    def test_effective_values(self, request, tmp_path, provider_spec, vendor,
                              request_kwargs, expected):
        if provider_spec is None:
            provider = request.getfixturevalue("mock_mlx_provider")
        else:
            request.getfixturevalue("mock_mlx")
            from heylook_llm.providers.mlx_provider import MLXProvider

            model_id, config = provider_spec
            config = dict(config)
            if vendor is not None:
                import json

                (tmp_path / "generation_config.json").write_text(json.dumps(vendor))
                config["model_path"] = str(tmp_path)
            provider = MLXProvider(model_id=model_id, config=config, verbose=False)
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")],
                          **request_kwargs)
        effective = provider._apply_model_defaults(req)
        for key, want in expected.items():
            if isinstance(want, bool):
                assert effective[key] is want, key
            else:
                assert effective[key] == want, key

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

    _ABSENT = object()

    @pytest.mark.parametrize("reject, continuing, expected", [
        ((), True, {"continue_final_message": True, "add_generation_prompt": False}),
        # continuing=False is the resolved EXPLICIT opt-out
        # (continue_final_message=false): the trailing assistant turn renders
        # closed and a FRESH generation prompt opens -- "reply to it", the only
        # meaning "never continue" can coherently have. (Auto mode never reaches
        # this branch with a trailing assistant message.)
        ((), False, {"continue_final_message": _ABSENT, "add_generation_prompt": True}),
        # a wrapper that rejects enable_thinking must retry WITHOUT it but WITH
        # continue_final_message -- dropping both silently renders a closed turn
        (("enable_thinking",), True,
         {"continue_final_message": True, "enable_thinking": _ABSENT}),
        # a stack that cannot continue refuses loudly
        (("continue_final_message",), True, "refuses"),
    ], ids=["continuing_leaves_the_turn_open", "not_continuing_never_passes_the_kwarg",
            "enable_thinking_fallback_keeps_continuation",
            "unsupported_continuation_refuses_loudly"])
    def test_template_kwargs(self, mock_mlx, reject, continuing, expected):
        tok = self._Tok(reject=reject)
        if expected == "refuses":
            from heylook_llm.providers.base import InvalidGenerationRequest

            with pytest.raises(InvalidGenerationRequest, match="cannot continue"):
                self._apply(tok, continuing, mock_mlx)
            return
        self._apply(tok, continuing, mock_mlx)
        kwargs = tok.calls[-1]
        for key, want in expected.items():
            if want is self._ABSENT:
                assert key not in kwargs, key
            else:
                assert kwargs[key] is want, key


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

        cases = [{}, {"enable_thinking": True}, {"enable_thinking": False}]
        for config in ({"model_path": "/fake", "vision": False},
                       {"model_path": "/fake", "vision": False, "enable_thinking": True},
                       {"model_path": "/fake", "vision": False, "enable_thinking": False}):
            provider = MLXProvider(model_id="m", config=dict(config), verbose=False)
            for kw in cases:
                req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], **kw)
                prompt_side = _resolve_enable_thinking(provider._apply_model_defaults(req))
                assert prompt_side == provider.effective_thinking(req), (config, kw)




@pytest.mark.unit
class TestCollectionDoesNotBlock:
    """A destructor must not wait.

    `BaseProvider.__del__` used to run the full `unload()`, whose drain loop
    polls for up to 30s. Anything collected while its active counter was
    non-zero therefore stalled whatever thread the GC fired on -- in the suite
    that was ~29s of a 65s run from one leaked counter, and in the server it
    would land on whichever thread GC chose, including one delivering tokens.

    WHAT THESE CANNOT SAY: they call `__del__()` as a method while the
    fixture, the local name and pytest's frame all still hold references, so
    they pin the BRANCH, not collection. In particular they no longer assert
    the provider keeps its model. v2.0.28 claimed the branch "leaves it
    loaded" and this class asserted it via `hasattr` -- which is true by
    construction here and false in the only case that matters: a destructor
    that returns early retains NOTHING, because `__del__` runs during
    deallocation. Corrected in v2.0.34.

    The two assertions in the first check pin DIFFERENT claims, and neither
    covers the other: `time.sleep` going uncalled pins that `__del__` still
    passes `drain=False` (mutate it to call `unload()` and the poll spins),
    while the warning pins that the branch fires at all (the poll is guarded
    by `while drain:`, so no drain=False call can reach it whatever the branch
    above does). Both also pass against 2.0.28 -- they guard the fix that
    release made, and their green says nothing about the work in v2.0.34.
    """

    def test_collection_with_traffic_never_enters_the_drain_poll(
        self, mock_mlx_provider, monkeypatch, caplog
    ):
        """The invariant is "the polling loop was not entered", so that is
        what is observed. The wall-clock form this replaces (`elapsed < 1.0`)
        passes for any implementation that happens to be fast and reds on a
        loaded machine with no behaviour change at all."""
        import logging as _logging

        import heylook_llm.providers.mlx_provider as _mod

        slept: list = []
        monkeypatch.setattr(_mod.time, "sleep", lambda s: slept.append(s))
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider._active_generations = 3
        try:
            with caplog.at_level(_logging.WARNING):
                mock_mlx_provider.__del__()
            assert slept == [], (
                f"__del__ entered the drain poll ({len(slept)} sleeps) with "
                "generations in flight -- the loop is running in a destructor again"
            )
            # The branch's whole remaining value is this warning. The weights
            # go regardless, so what the branch buys a reader is the name of
            # the model whose reference was dropped.
            said = [r.getMessage() for r in caplog.records]
            assert any("test-model" in m and "3 active" in m for m in said), (
                f"no warning naming the model and its active count: {said}"
            )
        finally:
            mock_mlx_provider._active_generations = 0

    def test_a_quiet_provider_still_unloads_on_collection(self, mock_mlx_provider):
        """The skip is conditional. With nothing in flight -- the normal
        case -- collection must still release the model, or the change trades
        a stall for a leak."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider._active_generations = 0
        mock_mlx_provider.__del__()
        assert not hasattr(mock_mlx_provider, "model")

    def test_the_destructor_makes_no_engine_calls(self, mock_mlx_provider, monkeypatch):
        """`drain=False` means "no wait AND no engine calls", and the second
        half is the one that was missing.

        Dropping the `waiting` read narrowed the guard to actives alone, which
        let the QUIET destructor path fall through to `gc.collect()` +
        `mx.clear_cache()`. That is the hazard .claude/rules/mlx.md states as "never gate
        teardown on actives alone" and `test_unload_waiter_safety.py` exists
        for: the active counter decrements BEFORE `gate.release()` admits the
        next waiter, so a woken waiter can be starting a decode exactly then --
        and `__del__` runs on whatever thread the GC chose. The deliberate
        path answers that by WAITING; a destructor cannot, so it declines to
        make the calls at all.
        """
        # The module the PROVIDER'S CLASS came from, not a fresh import of
        # that name: one test in this file pops and re-imports the module, so
        # `import ... as _mod` can hand back a different object than the one
        # the running `unload` reads `mx` from, and the patch lands on the
        # wrong module. Two things forced this shape, both order-dependent and
        # both green in isolation.
        #
        # And the count is kept HERE rather than read off `call_count`: on
        # Apple hardware `mlx_mocks` skips its patch because real MLX imports,
        # so `mx` is the real nanobind module and has no mock API at all.
        _mod = sys.modules[type(mock_mlx_provider).__module__]
        swept: list = []
        monkeypatch.setattr(_mod.mx, "clear_cache", lambda: swept.append(1))
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider._active_generations = 0

        mock_mlx_provider.__del__()

        assert not hasattr(mock_mlx_provider, "model"), (
            "the destructor stopped dropping its references"
        )
        assert swept == [], (
            "the destructor called mx.clear_cache() -- an engine call on "
            "whatever thread the GC fired on, possibly while a woken gate "
            "waiter is starting a decode"
        )

    def test_a_deliberate_unload_still_sweeps_the_engine(self, mock_mlx_provider, monkeypatch):
        """The other half: gating the engine calls on `drain` must not stop
        the real teardown paths sweeping, or the fix trades a hazard for an
        unswept Metal buffer cache on every eviction."""
        _mod = sys.modules[type(mock_mlx_provider).__module__]
        swept: list = []
        monkeypatch.setattr(_mod.mx, "clear_cache", lambda: swept.append(1))
        mock_mlx_provider.model = create_mock_model()

        mock_mlx_provider.unload()

        assert swept, "a deliberate unload stopped clearing the Metal buffer cache"

    def test_another_models_gate_waiters_do_not_suppress_teardown(self, mock_mlx_provider):
        """The generation gate is a PROCESS-GLOBAL singleton, so its `waiting`
        count can be entirely another model's traffic.

        The destructor branch read it anyway: model A collected while model B
        had queue waiters skipped A's cleanup and logged a warning naming A,
        telling the reader to go find A's leaked counter or dropped reference.
        Both halves were wrong, and `max_loaded_models=1` bounding it in
        practice is a default, not an invariant.
        """
        asked = []
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider._active_generations = 0

        def _stats():
            asked.append(True)
            return {"active": 0, "waiting": 4}

        mock_mlx_provider.generation_queue_stats = _stats
        mock_mlx_provider.__del__()
        # The claim is that the destructor does not CONSULT the shared gate --
        # asserted directly. A stub returning waiters is inert once the read is
        # gone, so a test that only checked the outcome would pass identically
        # with the stub deleted, and say nothing the quiet-provider test above
        # does not already say.
        assert asked == [], (
            "the destructor consulted the PROCESS-GLOBAL generation gate; its "
            "waiters can belong to another model entirely"
        )
        assert not hasattr(mock_mlx_provider, "model"), (
            "another model's gate waiters suppressed this provider's teardown"
        )




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

    def _check_gate_free(provider):
        # the next queued request can run instead of deadlocking
        assert provider._gen_gate.busy is False
        provider.check_capacity()  # no raise

    def _check_counter_zero(provider):
        assert provider._active_generations == 0

    @pytest.mark.parametrize("check", [_check_gate_free, _check_counter_zero],
                             ids=["generation_gate_released_after_error",
                                  "active_generation_counter_decremented"])
    def test_state_is_released_after_the_generation(self, mock_mlx_provider, check):
        """Whether the mocked generation completes or raises, the generator's
        finally must run: gate released and the active counter back to 0."""
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
            pass  # the mock strategy may error; the state must reset either way
        check(mock_mlx_provider)


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

    def _check_passed_through(provider, passed, seen):
        assert seen == passed

    def _check_distinct_defaults(provider, passed, seen):
        assert seen[0] is not None and seen[1] is not None
        assert seen[0] is not seen[1]  # per-request, not one shared event
        # the shared provider-level abort event must be gone
        assert not hasattr(provider, "_abort_event")

    def _check_isolated(provider, passed, seen):
        # A's event being set must not be visible through B's event
        seen[1].set()  # B aborts
        assert seen[0].is_set() is False  # A unaffected

    # own_events: per call, pass a fresh AbortEvent (True) or none (False)
    @pytest.mark.parametrize("own_events, check", [
        ((True,), _check_passed_through),
        ((False, False), _check_distinct_defaults),
        ((True, True), _check_isolated),
    ], ids=["strategy_receives_the_passed_abort_event",
            "each_call_gets_a_distinct_default_event_no_shared_state",
            "disconnect_of_one_request_does_not_abort_another"])
    def test_abort_event_is_per_request(self, mock_mlx_provider, own_events, check):
        from heylook_llm.providers.abort import AbortEvent

        seen = []
        self._inject_capturing_strategy(mock_mlx_provider, seen)
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        passed = []
        for own in own_events:
            if own:
                ev = AbortEvent()
                passed.append(ev)
                list(mock_mlx_provider.create_chat_completion(req, abort_event=ev))
            else:
                list(mock_mlx_provider.create_chat_completion(req))
        check(mock_mlx_provider, passed, seen)


@pytest.mark.unit
class TestCheckCapacity:
    """check_capacity() applies backpressure (503) via the generation gate."""

    @pytest.mark.parametrize("hold_the_gate", [False, True],
                             ids=["idle_provider_has_capacity",
                                  "raises_model_busy_when_queue_full"])
    def test_capacity_answers_through_the_gate(self, mock_mlx_provider, hold_the_gate):
        from heylook_llm.providers.common.generation_gate import (
            GenerationGate, ModelBusyError,
        )

        if not hold_the_gate:
            mock_mlx_provider.check_capacity()  # no raise
            return
        # Inject an isolated single-flight gate (the real gate is a process
        # singleton shared across providers; isolate it for a deterministic test).
        mock_mlx_provider._gen_gate = GenerationGate(max_waiting=0)
        mock_mlx_provider._gen_gate.acquire()  # simulate an in-flight generation
        try:
            with pytest.raises(ModelBusyError) as exc:
                mock_mlx_provider.check_capacity()
            assert "MODEL_BUSY" in str(exc.value)
        finally:
            mock_mlx_provider._gen_gate.release()

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
class TestThinkingBudgetCriteria:
    """Plan W7: when the MLX budget applies, when it is a no-op, and that a
    harmony model refuses it whatever the switch says (it always reasons, so
    dropping the budget would let it think uncapped)."""

    def _call(self, mock_mlx, info, budget, thinking):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import _thinking_budget_criteria

        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        eff = {"thinking_budget_tokens": budget, "enable_thinking": thinking}
        return _thinking_budget_criteria(req, eff, info, tokenizer=None)

    def test_no_op_and_refusal(self, mock_mlx):
        from heylook_llm.providers.base import InvalidGenerationRequest
        from heylook_llm.providers.common.template_info import ModelTemplateInfo

        think = ModelTemplateInfo(has_thinking_markers=True)
        assert self._call(mock_mlx, think, None, True) is None      # no budget asked
        assert self._call(mock_mlx, think, 64, False) is None       # thinking off
        assert self._call(mock_mlx, ModelTemplateInfo(), 64, True) is None  # no format
        harmony = ModelTemplateInfo(has_harmony_structure=True)
        for switch in (True, False):
            with pytest.raises(InvalidGenerationRequest, match="harmony"):
                self._call(mock_mlx, harmony, 64, switch)
