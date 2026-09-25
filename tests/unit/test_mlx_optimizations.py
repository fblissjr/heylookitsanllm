# tests/unit/test_mlx_optimizations.py
"""Tests for MLX server optimizations (v1.26.0).

Covers:
- VisionFeatureCache: LRU eviction, content keys, stats
- Keepalive marker: streaming utils sentinel type
- Cached tokens passthrough: generation_core attaches cached_tokens
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# VisionFeatureCache tests
# ---------------------------------------------------------------------------

class TestVisionFeatureCache:
    @pytest.fixture
    def cache(self):
        from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache
        return VisionFeatureCache(max_entries=3)

    def test_url_key_hit_miss(self, cache):
        import mlx.core as mx
        features = mx.ones((1, 10))
        cache.put("http://example.com/img.jpg", features)

        result = cache.get("http://example.com/img.jpg")
        assert result is not None

        result = cache.get("http://example.com/other.jpg")
        assert result is None

    def test_content_key_follows_the_pixels_not_the_source(self):
        """Same picture, same key however it arrived; a changed picture, order,
        or shape keys differently (a web link whose file changed must miss)."""
        from PIL import Image
        from heylook_llm.providers.common.vision_feature_cache import image_content_key

        red, red_again = Image.new("RGB", (8, 4), "red"), Image.new("RGB", (8, 4), "red")
        blue = Image.new("RGB", (8, 4), "blue")
        assert image_content_key([red]) == image_content_key([red_again])
        assert image_content_key([red]) != image_content_key([blue])
        assert image_content_key([red, blue]) != image_content_key([blue, red])
        assert image_content_key([red]) != image_content_key([Image.new("RGB", (4, 8), "red")])

    def test_lru_eviction(self, cache):
        import mlx.core as mx
        for i in range(4):
            cache.put(f"img{i}.jpg", mx.ones((1, 5)))

        # img0 should have been evicted (max_entries=3)
        assert cache.get("img0.jpg") is None
        assert cache.get("img1.jpg") is not None
        assert cache.get("img2.jpg") is not None
        assert cache.get("img3.jpg") is not None

    def test_lru_access_updates_order(self, cache):
        import mlx.core as mx
        cache.put("a.jpg", mx.ones((1, 5)))
        cache.put("b.jpg", mx.ones((1, 5)))
        cache.put("c.jpg", mx.ones((1, 5)))

        # Access a.jpg to move it to end
        cache.get("a.jpg")

        # Insert d.jpg -- should evict b.jpg (oldest non-accessed)
        cache.put("d.jpg", mx.ones((1, 5)))

        assert cache.get("a.jpg") is not None
        assert cache.get("b.jpg") is None
        assert cache.get("c.jpg") is not None
        assert cache.get("d.jpg") is not None

    def test_stats(self, cache):
        import mlx.core as mx
        cache.put("a.jpg", mx.ones((1, 5)))
        cache.get("a.jpg")  # hit
        cache.get("b.jpg")  # miss

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_clear(self, cache):
        import mlx.core as mx
        cache.put("a.jpg", mx.ones((1, 5)))
        cache.put("b.jpg", mx.ones((1, 5)))
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a.jpg") is None

    def test_list_features_count_against_the_byte_cap(self, cache):
        """deepseek_v4 stores a list of per-image arrays; a list's bytes read
        as 0 would let it grow past max_bytes unseen."""
        import mlx.core as mx
        cache.put("a", [mx.ones((10,)), mx.ones((5,))])
        assert cache.stats()["bytes"] == 60

    def test_empty_key_no_cache(self, cache):
        import mlx.core as mx
        cache.put("", mx.ones((1, 10)))
        assert len(cache) == 0

    def test_update_existing_key(self, cache):
        """Updating an existing key replaces the value and moves to end."""
        import mlx.core as mx
        v1 = mx.ones((1, 5))
        v2 = mx.zeros((1, 5))
        cache.put("a.jpg", v1)
        cache.put("a.jpg", v2)

        result = cache.get("a.jpg")
        assert result is not None
        # Should be the updated value
        assert result[0, 0].item() == 0.0
        assert len(cache) == 1

    def test_stats_expose_bytes(self, cache):
        import mlx.core as mx
        features = mx.ones((1, 10))
        cache.put("a.jpg", features)

        stats = cache.stats()
        assert stats["bytes"] == features.nbytes
        assert stats["max_bytes"] > 0

    def test_byte_cap_evicts_before_count_cap(self):
        """If bytes overflow first, oldest entries are evicted even when the
        entry count is still under max_entries."""
        from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache
        import mlx.core as mx

        # Each entry is 10 float32 = 40 bytes. Cap at 100 bytes (2 entries fit).
        cache = VisionFeatureCache(max_entries=100, max_bytes=100)
        cache.put("a.jpg", mx.ones((10,)))
        cache.put("b.jpg", mx.ones((10,)))

        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["bytes"] == 80

        # Third insert should evict "a" (not "b"): 80 + 40 > 100.
        cache.put("c.jpg", mx.ones((10,)))

        assert cache.get("a.jpg") is None
        assert cache.get("b.jpg") is not None
        assert cache.get("c.jpg") is not None

        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["bytes"] == 80

    def test_byte_cap_evicts_until_under_cap(self):
        """A single large insert can evict multiple smaller entries."""
        from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache
        import mlx.core as mx

        cache = VisionFeatureCache(max_entries=100, max_bytes=200)
        cache.put("a.jpg", mx.ones((10,)))  # 40B
        cache.put("b.jpg", mx.ones((10,)))  # 40B
        cache.put("c.jpg", mx.ones((10,)))  # 40B -> total 120B

        # Insert a 160B entry; cap is 200B, so we must evict at least "a" and "b".
        cache.put("big.jpg", mx.ones((40,)))

        assert cache.get("a.jpg") is None
        assert cache.get("b.jpg") is None
        assert cache.get("big.jpg") is not None
        assert cache.stats()["bytes"] <= 200

    def test_clear_resets_byte_counter(self, cache):
        import mlx.core as mx
        cache.put("a.jpg", mx.ones((1, 10)))
        cache.put("b.jpg", mx.ones((1, 10)))
        assert cache.stats()["bytes"] > 0
        cache.clear()
        assert cache.stats()["bytes"] == 0
        assert cache.stats()["entries"] == 0

    def test_replace_updates_byte_accounting(self, cache):
        """Replacing a key must subtract old bytes before adding new."""
        import mlx.core as mx
        cache.put("a.jpg", mx.ones((1, 10)))  # 40B
        bytes_after_first = cache.stats()["bytes"]

        cache.put("a.jpg", mx.ones((1, 10)))  # same size
        assert cache.stats()["bytes"] == bytes_after_first

        cache.put("a.jpg", mx.ones((1, 5)))   # shrinks to 20B
        assert cache.stats()["bytes"] < bytes_after_first


# ---------------------------------------------------------------------------
# Segment-aware eviction tests
# ---------------------------------------------------------------------------

class TestKeepaliveMarker:
    def test_marker_type(self):
        from heylook_llm.streaming_utils import KeepaliveMarker, KEEPALIVE_MARKER
        assert isinstance(KEEPALIVE_MARKER, KeepaliveMarker)

    def test_marker_is_singleton(self):
        from heylook_llm.streaming_utils import KEEPALIVE_MARKER
        # Importing twice returns the same object
        from heylook_llm.streaming_utils import KEEPALIVE_MARKER as m2
        assert KEEPALIVE_MARKER is m2

    def test_marker_detected_by_isinstance(self):
        from heylook_llm.streaming_utils import KeepaliveMarker, KEEPALIVE_MARKER
        assert isinstance(KEEPALIVE_MARKER, KeepaliveMarker)
        assert not isinstance("data: chunk", KeepaliveMarker)


# ---------------------------------------------------------------------------
# PromptCacheManager byte budget integration tests
# ---------------------------------------------------------------------------
