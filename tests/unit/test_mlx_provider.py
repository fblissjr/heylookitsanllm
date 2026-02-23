# tests/unit/test_mlx_provider.py
"""
Unit tests for MLXProvider -- the core Apple Silicon provider.

All tests use the mock_mlx / mock_mlx_provider fixtures from conftest.py
so they run on any platform without MLX installed.
"""
import threading
import time

import pytest

from heylook_llm.config import ChatMessage, ChatRequest
from helpers.mlx_mock import create_mock_model, create_mock_processor


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

    def test_generation_lock_released_after_error(self, mock_mlx_provider):
        """Generation lock should be released even after errors."""
        mock_mlx_provider.model = create_mock_model()
        mock_mlx_provider.processor = create_mock_processor()
        mock_mlx_provider._compile_strategies()

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )

        # First call (triggers error from mock strategy)
        list(mock_mlx_provider.create_chat_completion(req))

        # Lock should be released so we can acquire it again
        acquired = mock_mlx_provider._generation_lock.acquire(blocking=False)
        assert acquired is True
        mock_mlx_provider._generation_lock.release()

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
