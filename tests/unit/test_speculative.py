# tests/unit/test_speculative.py
"""Tests for speculative decoding configuration and acceptance tracking.

Covers:
- num_draft_tokens default value and passthrough
- UnifiedTextStrategy model_config propagation
- cache_config plumbing from model defaults
"""

import pytest

from heylook_llm.config import MLXModelConfig


class TestNumDraftTokensDefault:
    """Verify num_draft_tokens defaults are correct."""

    def test_default_is_3(self):
        config = MLXModelConfig(model_path="/fake/model")
        assert config.num_draft_tokens == 3

    def test_explicit_override(self):
        config = MLXModelConfig(model_path="/fake/model", num_draft_tokens=6)
        assert config.num_draft_tokens == 6

    def test_none_allowed(self):
        config = MLXModelConfig(model_path="/fake/model", num_draft_tokens=None)
        assert config.num_draft_tokens is None


class TestApplyModelDefaults:
    """Verify _apply_model_defaults includes cache and speculative decoding config."""

    def test_cache_config_propagated(self, mock_mlx_provider):
        """Model config cache settings should appear in effective_request."""
        from heylook_llm.config import ChatMessage, ChatRequest

        provider = mock_mlx_provider
        provider.config['cache_type'] = 'quantized'
        provider.config['kv_bits'] = 8
        provider.config['num_draft_tokens'] = 5

        request = ChatRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=64,
        )

        effective = provider._apply_model_defaults(request)
        assert effective.get('cache_type') == 'quantized'
        assert effective.get('kv_bits') == 8
        assert effective.get('num_draft_tokens') == 5

    def test_request_overrides_model_config(self, mock_mlx_provider):
        """Request-level params should override model config."""
        from heylook_llm.config import ChatMessage, ChatRequest

        provider = mock_mlx_provider
        provider.config['temperature'] = 0.5

        request = ChatRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=64,
            temperature=0.9,
        )

        effective = provider._apply_model_defaults(request)
        assert effective['temperature'] == 0.9


class TestUnifiedTextStrategyModelConfig:
    """Verify UnifiedTextStrategy receives model_config."""

    def test_model_config_stored(self, mock_mlx):
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        config = {'cache_type': 'quantized', 'kv_bits': 8}
        strategy = UnifiedTextStrategy(
            draft_model=None,
            model_id="test-vlm",
            model_config=config,
            is_vlm=True,
        )
        assert strategy.model_config == config

    def test_model_config_defaults_to_empty(self, mock_mlx):
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(draft_model=None, model_id="test-vlm")
        assert strategy.model_config == {}

    def test_is_vlm_flag_stored(self, mock_mlx):
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(draft_model=None, model_id="test", is_vlm=True)
        assert strategy.is_vlm is True

        strategy2 = UnifiedTextStrategy(draft_model=None, model_id="test", is_vlm=False)
        assert strategy2.is_vlm is False

    def test_compile_strategies_passes_config(self, mock_vlm_provider):
        """_compile_strategies should pass self.config to UnifiedTextStrategy."""
        provider = mock_vlm_provider
        provider.config['cache_type'] = 'quantized'
        provider._compile_strategies()

        text_strategy = provider._strategies.get('text')
        assert text_strategy is not None
        assert text_strategy.model_config.get('cache_type') == 'quantized'
        assert text_strategy.is_vlm is True

    def test_compile_strategies_text_only(self, mock_mlx_provider):
        """Text-only provider should compile UnifiedTextStrategy with is_vlm=False."""
        provider = mock_mlx_provider
        provider._compile_strategies()

        text_strategy = provider._strategies.get('text')
        assert text_strategy is not None
        assert text_strategy.is_vlm is False


class TestCacheConfigPlumbing:
    """Verify cache_config construction uses effective_request values correctly."""

    def test_kv_bits_none_when_standard(self, mock_mlx_provider):
        """When cache_type is standard, kv_bits should be None (no fallback to 8)."""
        from heylook_llm.config import ChatMessage, ChatRequest

        provider = mock_mlx_provider
        # No cache_type or kv_bits in model config

        request = ChatRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=64,
        )

        effective = provider._apply_model_defaults(request)
        # kv_bits should not be set when using standard cache
        assert effective.get('kv_bits') is None

    def test_kv_bits_from_model_config(self, mock_mlx_provider):
        """kv_bits should come from model config when set."""
        from heylook_llm.config import ChatMessage, ChatRequest

        provider = mock_mlx_provider
        provider.config['cache_type'] = 'quantized'
        provider.config['kv_bits'] = 4

        request = ChatRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=64,
        )

        effective = provider._apply_model_defaults(request)
        assert effective['kv_bits'] == 4
