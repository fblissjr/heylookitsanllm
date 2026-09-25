"""POST /v1/cache/clear on MLX empties everything a request could reuse:
the prefix cache AND the vision features (the latter survived a clear until
2026-09-24, so a "cold" image check after one still skipped the tower)."""
import mlx.core as mx
import pytest


@pytest.mark.unit
def test_clearing_the_cache_empties_the_vision_features(mock_vlm_provider):
    """A real vision provider with its real vision strategy: a feature the
    strategy cached is gone after the provider's clear_cache()."""
    from helpers.mlx_mock import create_mock_vlm_model, create_mock_processor

    provider = mock_vlm_provider
    provider.model = create_mock_vlm_model()
    provider.processor = create_mock_processor()
    provider._compile_strategies()
    cache = provider._strategies["vision"]._vision_cache
    cache.put("k", mx.zeros((2, 2)))
    assert cache.get("k") is not None

    assert provider.clear_cache() is True
    assert cache.get("k") is None
