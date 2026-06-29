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
        assert effective["temperature"] == 0.1  # global default
        assert effective["max_tokens"] == 512

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
        """When enable_thinking is set in config, thinking defaults apply."""
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
        assert effective["temperature"] == 0.6  # thinking default
        assert effective["top_p"] == 0.95
        assert effective["presence_penalty"] == 1.5

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
class TestApplyModelDefaultsPresetCascade:
    """Preset layer sits between model sampler fields and request explicit
    fields. These tests pin the resolution order so a refactor doesn't
    silently change precedence.
    """

    def test_preset_overrides_model_defaults(self, mock_mlx, monkeypatch):  # noqa: ARG002
        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        registry = PresetRegistry({"hot": {"temperature": 1.0, "top_p": 0.9}})
        reset_preset_registry_for_test(registry)
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False, "temperature": 0.3},
                verbose=False,
            )
            req = ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                preset="hot",
            )
            effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 1.0  # preset beat model default
            assert effective["top_p"] == 0.9
        finally:
            reset_preset_registry_for_test(None)

    def test_request_field_beats_preset(self, mock_mlx, monkeypatch):  # noqa: ARG002
        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        registry = PresetRegistry({"hot": {"temperature": 1.0}})
        reset_preset_registry_for_test(registry)
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False},
                verbose=False,
            )
            req = ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                preset="hot",
                temperature=0.4,
            )
            effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 0.4  # request explicit beat preset
        finally:
            reset_preset_registry_for_test(None)

    def test_preset_unset_fields_pass_through(self, mock_mlx):  # noqa: ARG002
        """A preset that only sets `temperature` must NOT clear other model
        defaults. Unset keys fall through."""
        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        registry = PresetRegistry({"temp_only": {"temperature": 0.8}})
        reset_preset_registry_for_test(registry)
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False, "max_tokens": 777, "top_k": 15},
                verbose=False,
            )
            req = ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                preset="temp_only",
            )
            effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 0.8
            assert effective["max_tokens"] == 777
            assert effective["top_k"] == 15
        finally:
            reset_preset_registry_for_test(None)

    def test_unknown_preset_raises_preset_not_found(self, mock_mlx):  # noqa: ARG002
        """Provider raises a domain exception, not an HTTP one. Route handlers
        translate to 400 -- the provider stays transport-agnostic."""
        from heylook_llm.presets import (
            PresetNotFound,
            PresetRegistry,
            reset_preset_registry_for_test,
        )
        from heylook_llm.providers.mlx_provider import MLXProvider

        reset_preset_registry_for_test(PresetRegistry({}))
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False},
                verbose=False,
            )
            req = ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                preset="does-not-exist",
            )
            with pytest.raises(PresetNotFound):
                provider._apply_model_defaults(req)
        finally:
            reset_preset_registry_for_test(None)

    def test_no_preset_cascade_unchanged(self, mock_mlx):  # noqa: ARG002
        """If ChatRequest.preset is None, cascade behaves exactly like before
        (back-compat gate)."""
        from heylook_llm.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_id="m",
            config={"model_path": "/fake", "vision": False, "temperature": 0.55},
            verbose=False,
        )
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        effective = provider._apply_model_defaults(req)
        assert effective["temperature"] == 0.55


@pytest.mark.unit
class TestApplyModelDefaultsDefaultPreset:
    """Model's ``default_preset`` applies as layer 3b, only when the request
    didn't pick one. Explicit request preset beats it. Request explicit fields
    beat both."""

    def test_default_preset_applies_without_request_preset(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        registry = PresetRegistry({"snappy": {"temperature": 0.3, "top_p": 0.8}})
        reset_preset_registry_for_test(registry)
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False, "default_preset": "snappy"},
                verbose=False,
            )
            req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
            effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 0.3
            assert effective["top_p"] == 0.8
        finally:
            reset_preset_registry_for_test(None)

    def test_request_preset_wins_over_default_preset(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        registry = PresetRegistry({
            "snappy": {"temperature": 0.3},
            "wild": {"temperature": 1.1},
        })
        reset_preset_registry_for_test(registry)
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False, "default_preset": "snappy"},
                verbose=False,
            )
            req = ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                preset="wild",
            )
            effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 1.1
        finally:
            reset_preset_registry_for_test(None)

    def test_request_explicit_field_wins_over_default_preset(self, mock_mlx):  # noqa: ARG002
        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        registry = PresetRegistry({"snappy": {"temperature": 0.3}})
        reset_preset_registry_for_test(registry)
        try:
            provider = MLXProvider(
                model_id="m",
                config={"model_path": "/fake", "vision": False, "default_preset": "snappy"},
                verbose=False,
            )
            req = ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                temperature=0.9,
            )
            effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 0.9
        finally:
            reset_preset_registry_for_test(None)

    def test_unknown_default_preset_logs_and_skips(self, mock_mlx, caplog):  # noqa: ARG002
        """Unknown ``default_preset`` name is non-fatal -- log a warning and
        fall through to the rest of the cascade. Models are validated at
        startup, so a miss here indicates registry drift, not a user typo."""
        import logging

        from heylook_llm.presets import PresetRegistry, reset_preset_registry_for_test
        from heylook_llm.providers.mlx_provider import MLXProvider

        reset_preset_registry_for_test(PresetRegistry({}))
        try:
            provider = MLXProvider(
                model_id="m",
                config={
                    "model_path": "/fake",
                    "vision": False,
                    "default_preset": "ghost",
                    "temperature": 0.44,
                },
                verbose=False,
            )
            req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
            with caplog.at_level(logging.WARNING):
                effective = provider._apply_model_defaults(req)
            assert effective["temperature"] == 0.44  # model field layer still applied
            assert any("default_preset" in r.message for r in caplog.records)
        finally:
            reset_preset_registry_for_test(None)


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
        metrics = mock_mlx_provider.get_metrics()
        assert metrics.requests_active == 3


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
    def test_no_model_loaded_yields_error(self, mock_mlx_provider):
        """If model is not loaded, should yield error chunk."""
        mock_mlx_provider._compile_strategies()
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        chunks = list(mock_mlx_provider.create_chat_completion(req))
        assert len(chunks) == 1
        assert "Error" in chunks[0].text

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
        chunks = list(mock_mlx_provider.create_chat_completion(req))
        assert any("text-only" in c.text for c in chunks)

    def test_generation_gate_released_after_error(self, mock_mlx_provider):
        """The generation gate must release even after errors, so the next
        queued request can run instead of deadlocking."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()
        mock_mlx_provider._compile_strategies()

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )

        # First call (triggers error from mock strategy)
        list(mock_mlx_provider.create_chat_completion(req))

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
        list(mock_mlx_provider.create_chat_completion(req))
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
        mp._GENERATION_GATE = None
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
        assert effective["temperature"] == 0.1

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
