"""POST /v1/cache/clear on MLX empties everything a request could reuse:
the prefix cache AND the vision features (the latter survived a clear until
2026-09-24, so a "cold" image check after one still skipped the tower)."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache
from heylook_llm.providers.mlx_provider import MLXProvider


@pytest.mark.unit
def test_clearing_the_cache_empties_the_vision_features():
    cache = VisionFeatureCache()
    cache.put(["data:a"], mx.zeros((2, 2)))
    fake = SimpleNamespace(_apc=None, model_id="m",
                           _strategies={"vision": SimpleNamespace(_vision_cache=cache)})
    assert MLXProvider.clear_cache(fake) is True
    assert cache.get(["data:a"]) is None
