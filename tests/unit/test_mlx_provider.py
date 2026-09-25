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


def _engine_module(provider):
    """The module the PROVIDER'S CLASS came from (a test elsewhere may have
    re-imported the module, and the patch must land where the provider reads)."""
    return sys.modules[type(provider).__module__]


class _Chunk:
    def __init__(self, text):
        self.text = text


def _stand_in_engines(monkeypatch, provider, **generate):
    """Replace the engine behind each named path ("text", "vision",
    "diffusion") with a generator function, give the provider a model and a
    processor, and compile its paths the way load does. Unnamed paths answer
    with their own name, so a test can see which engine served a request."""
    mod = _engine_module(provider)
    classes = {"text": mod.UnifiedTextStrategy, "vision": mod.VLMVisionStrategy,
               "diffusion": mod.DiffusionStrategy}
    for name, cls in classes.items():
        fn = generate.get(name)
        if fn is None:
            def fn(self, *a, _name=name, **k):
                yield _Chunk(_name)
        monkeypatch.setattr(cls, "generate", fn)
    provider.model = create_mock_model()
    provider.processor = create_mock_processor()
    provider._compile_strategies()


def _until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "setup never reached its state"
        time.sleep(0.005)


@pytest.mark.unit
class TestRequestRouting:
    """Each request reaches the engine that can serve it, through
    create_chat_completion: text on any model -> the text engine, an image on
    a vision model -> the vision engine, and anything on a masked-diffusion
    checkpoint -> the denoising engine (images included: it takes them
    inline, so it must win over the vision split).

    The diffusion rows guard a silent empty-response bug, not a crash: a
    masked-diffusion checkpoint driven by the autoregressive loop emits ZERO
    tokens (it samples one meaningless token from the last prompt position,
    which lands on EOS). An image on a text-only model is a 400
    (TestCreateChatCompletion::test_text_model_rejects_images); the
    advertised vision capability matching ``is_vlm`` is
    test_vision_capability.py.
    """

    @pytest.mark.parametrize("provider_fixture, diffusion, with_image, served_by", [
        ("mock_mlx_provider", False, False, "text"),
        ("mock_vlm_provider", False, False, "text"),
        ("mock_vlm_provider", False, True, "vision"),
        ("mock_vlm_provider", True, True, "diffusion"),
        ("mock_vlm_provider", True, False, "diffusion"),
    ], ids=["text_model_text_request", "vision_model_text_request",
            "vision_model_image_request", "diffusion_wins_over_vision_routing",
            "diffusion_serves_text_too"])
    def test_the_request_reaches_its_engine(self, request, monkeypatch, provider_fixture,
                                            diffusion, with_image, served_by):
        provider = request.getfixturevalue(provider_fixture)
        if diffusion:
            provider.is_diffusion = True   # what load decides from the checkpoint
        _stand_in_engines(monkeypatch, provider)
        req = (request.getfixturevalue("sample_multimodal_request") if with_image
               else ChatRequest(messages=[ChatMessage(role="user", content="hi")]))

        chunks = list(provider.create_chat_completion(req))
        assert [c.text for c in chunks] == [served_by]


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
    @pytest.mark.parametrize("build_messages, expected", [
        (lambda r: [ChatMessage(role="user", content="Hello")], False),
        (lambda r: r.getfixturevalue("sample_multimodal_request").messages, True),
        # multipart content with only text parts is not an image
        (_text_parts_only, False),
    ], ids=["no_images_text_content", "images_detected",
            "text_only_multipart_no_images"])
    def test_detect_images(self, request, mock_mlx_provider, build_messages, expected):
        messages = build_messages(request)
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
    """How a narrow wrapper's TypeError is answered (_apply_chat_template,
    reached through UnifiedTextStrategy._render_template). The rendered shape
    of a continuation is TestContinuationRender's, through a real template.

    Template variables (enable_thinking, the depth variable) travel SEPARATELY
    from the base kwargs so the TypeError retry can drop them: moved into
    base_kwargs, every request to a model whose tokenizer wrapper has a
    narrow signature becomes a hard TypeError."""

    class _Tok:
        """A call carrying a rejected keyword raises TypeError, as a narrow
        wrapper does; any other call renders."""

        def __init__(self, reject=()):
            self.reject = set(reject)

        def apply_chat_template(self, messages, **kwargs):
            if self.reject & set(kwargs):
                raise TypeError("unexpected keyword argument")
            return "PROMPT"

    def _render(self, tok, request, continuing, mock_mlx):  # noqa: ARG002
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        # No template_info: a requested depth rides as `reasoning_effort`.
        strategy = UnifiedTextStrategy(model_id="m")
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "he"}]
        return strategy._render_template(
            messages, tok, None, None, request, continuing=continuing)

    # Rows: (keywords the wrapper rejects, effective request, continuing,
    # "renders" | "refuses" | TypeError).
    @pytest.mark.parametrize("reject, request_, continuing, expected", [
        # a stack that cannot continue refuses loudly
        (("continue_final_message",), {"enable_thinking": False}, True, "refuses"),
        # a wrapper that rejects the depth variable still renders after the
        # retry, without either template variable
        (("reasoning_effort",), {"enable_thinking": True, "reasoning_effort": "low"}, False,
         "renders"),
        # When `continue_final_message` itself is what the wrapper rejects,
        # the retry fails the same way. A continuation is then a 400;
        # rendering a closed turn would silently restart the message.
        (("continue_final_message",), {"enable_thinking": True}, True, "refuses"),
        # ...but without a continuation the second TypeError is not a refusal
        (("tokenize",), {"enable_thinking": True}, False, TypeError),
    ], ids=["unsupported_continuation_refuses_loudly",
            "a_narrow_wrapper_still_renders_after_the_retry",
            "a_stack_that_cannot_continue_is_refused_not_restarted",
            "a_retry_failure_without_continuation_stays_a_type_error"])
    def test_template_kwargs(self, mock_mlx, reject, request_, continuing, expected):
        tok = self._Tok(reject=reject)
        if expected == "refuses":
            from heylook_llm.providers.base import InvalidGenerationRequest

            with pytest.raises(InvalidGenerationRequest, match="cannot continue"):
                self._render(tok, request_, continuing, mock_mlx)
            return
        if expected is TypeError:
            with pytest.raises(TypeError):
                self._render(tok, request_, continuing, mock_mlx)
            return
        assert self._render(tok, request_, continuing, mock_mlx) == "PROMPT"


def _qwen_tokenizer():
    """A real transformers tokenizer (no weights) carrying the Qwen3.5 chat
    template fixture: transformers itself implements continue_final_message,
    so the continuation shape is the one a real model is fed."""
    from pathlib import Path

    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")))
    tok.chat_template = (Path(__file__).resolve().parents[1] / "fixtures"
                         / "chat_templates" / "qwen3_5.jinja").read_text()
    return tok


class _NarrowWrapper:
    """A tokenizer wrapper whose signature cannot take template variables
    (it raises TypeError on enable_thinking), over the real tokenizer."""

    def __init__(self, tok):
        self.tok = tok
        self.chat_template = tok.chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            continue_final_message=False):
        return self.tok.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message)


@pytest.mark.unit
class TestContinuationRender:
    """The prompt a continuation renders to, through the real template.

    Continuing leaves the assistant turn OPEN: the prompt ends with the
    partial reply, so the model's next token extends it. Suppressing the
    generation prompt alone was NOT continuation -- the turn still rendered
    CLOSED, so the model saw a finished message and nothing to continue.
    An explicit opt-out (continue_final_message=false) closes the turn and
    opens a FRESH one -- "reply to it". A wrapper that cannot take template
    variables must still continue: dropping continue_final_message along
    with them silently renders a closed turn."""

    @pytest.mark.parametrize("flag, narrow, leaves_open", [
        (None, False, True),
        (False, False, False),
        (None, True, True),
    ], ids=["continuing_leaves_the_turn_open", "not_continuing_opens_a_fresh_turn",
            "enable_thinking_fallback_keeps_continuation"])
    def test_the_turn_is_open_or_closed(self, flag, narrow, leaves_open):
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        tok = _qwen_tokenizer()
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi"),
                                    ChatMessage(role="assistant", content="The answer is")],
                          continue_final_message=flag)
        prompt = UnifiedTextStrategy(model_id="m").render_prompt(
            req, {"enable_thinking": False}, None, _NarrowWrapper(tok) if narrow else tok)
        if leaves_open:
            assert prompt.endswith("The answer is"), prompt
        else:
            assert "The answer is<|im_end|>" in prompt, prompt
            assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), prompt


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
    they pin the BRANCH, not collection. A real collection with generations
    in flight cannot be staged: a running generation holds a reference to its
    provider, so the destructor never runs while one is live. A destructor
    that drains again (the ~29s stall) makes the engine call the first check
    below forbids.
    """

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

    def test_another_models_gate_waiters_do_not_suppress_teardown(self, mock_mlx_provider, caplog):
        """The generation gate is a PROCESS-GLOBAL singleton, so its `waiting`
        count can be entirely another model's traffic.

        The destructor branch read it anyway: model A collected while model B
        had queue waiters skipped A's cleanup and logged a warning naming A,
        telling the reader to go find A's leaked counter or dropped reference.
        Both halves were wrong, and `max_loaded_models=1` bounding it in
        practice is a default, not an invariant. Staged with REAL waiters on
        the real process gate.
        """
        import logging as _logging

        from heylook_llm.providers.common.generation_gate import get_process_gate

        gate = get_process_gate(8)
        gate.acquire()                     # another model's generation
        waiter = threading.Thread(target=lambda: (gate.acquire(), gate.release()),
                                  daemon=True)
        waiter.start()                     # ...and a request queued behind it
        try:
            _until(lambda: mock_mlx_provider.generation_queue_stats()["waiting"] == 1)
            mock_mlx_provider.model = create_mock_model()
            with caplog.at_level(_logging.WARNING):
                mock_mlx_provider.__del__()
        finally:
            gate.release()
            waiter.join(5)
        assert not hasattr(mock_mlx_provider, "model"), (
            "another model's gate waiters suppressed this provider's teardown"
        )
        assert not any("test-model" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
class TestUnload:
    def test_unload_releases_the_weights_while_the_provider_lives(self, mock_mlx_provider):
        """After unload() the provider no longer keeps its model or processor
        alive, even while something (the router mid-evict, a stale handle)
        still holds the provider object. That the wait before it covers
        actives and gate waiters is test_unload_waiter_safety.py."""
        import gc
        import weakref

        model, processor = create_mock_model(), create_mock_processor()
        refs = [weakref.ref(model), weakref.ref(processor)]
        mock_mlx_provider.model, mock_mlx_provider.processor = model, processor
        mock_mlx_provider._compile_strategies()
        del model, processor

        mock_mlx_provider.unload()
        gc.collect()

        assert [r() is None for r in refs] == [True, True], (
            "unload() left the weights reachable from a live provider")


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

    @pytest.mark.parametrize("check", [_check_gate_free],
                             ids=["generation_gate_released_after_error"])
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
    in-flight generation (the FIFO concurrency cross-contamination bug).

    Two real requests through create_chat_completion, queued behind each
    other at the real gate: A carries the event the route would set on
    disconnect, B carries none (an internal caller). Setting A's stops A and
    leaves B to finish in full."""

    STREAM = 40

    def test_one_requests_abort_stops_it_and_not_the_other(self, mock_mlx_provider, monkeypatch):
        from heylook_llm.providers.abort import AbortEvent

        n = self.STREAM

        def engine(self, request, effective_request, model, processor, abort_event=None):
            for i in range(n):
                if abort_event is not None and abort_event.is_set():
                    return
                yield _Chunk(str(i))
                time.sleep(0.005)

        _stand_in_engines(monkeypatch, mock_mlx_provider, text=engine)
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        got = {"a": [], "b": []}
        ev_a = AbortEvent()

        def run(key, **kw):
            for chunk in mock_mlx_provider.create_chat_completion(req, **kw):
                got[key].append(chunk)

        a = threading.Thread(target=run, args=("a",), kwargs={"abort_event": ev_a}, daemon=True)
        a.start()
        _until(lambda: len(got["a"]) >= 3)
        b = threading.Thread(target=run, args=("b",), daemon=True)
        b.start()
        _until(lambda: mock_mlx_provider.generation_queue_stats()["waiting"] == 1)
        ev_a.set()                          # A's client goes away
        a.join(5)
        b.join(5)

        assert len(got["a"]) < n, "setting A's abort event did not stop A"
        assert len(got["b"]) == n, "A's abort reached B"


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
        """max_queue_depth = 3 admits three waiters behind the running
        generation and refuses the next, counted on the real gate."""
        import heylook_llm.providers.mlx_provider as mp
        from heylook_llm.providers.common.generation_gate import (
            ModelBusyError, get_process_gate, reset_process_gate,
        )

        # The gate is a process-global singleton: max_queue_depth is read from
        # the FIRST provider created. Reset it so this provider is that first one.
        reset_process_gate()
        provider = mp.MLXProvider(
            model_id="d",
            config={"model_path": "/fake", "vision": False, "max_queue_depth": 3},
            verbose=False,
        )
        gate = get_process_gate(8)
        gate.acquire()                      # the running generation
        waiters = []
        try:
            for queued in (1, 2, 3):
                provider.check_capacity()   # room for this one
                t = threading.Thread(target=lambda: (gate.acquire(), gate.release()),
                                     daemon=True)
                t.start()
                waiters.append(t)
                _until(lambda: provider.generation_queue_stats()["waiting"] == queued)
            with pytest.raises(ModelBusyError):
                provider.check_capacity()
        finally:
            gate.release()
            for t in waiters:
                t.join(5)


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
