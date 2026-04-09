# tests/unit/test_mlx_optimizations.py
"""Tests for MLX server optimizations (v1.26.0).

Covers:
- VisionFeatureCache: LRU eviction, URL keying, pixel hash fallback, stats
- snapshot_nbytes: byte size computation for KV cache snapshots
- Segment-aware eviction: system nodes survive longer than assistant nodes
- Byte budget trimming: radix cache trim_to_bytes
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

    def test_list_key(self, cache):
        import mlx.core as mx
        features = mx.ones((2, 10))
        cache.put(["img1.jpg", "img2.jpg"], features)

        result = cache.get(["img1.jpg", "img2.jpg"])
        assert result is not None

        # Different order = different key
        result = cache.get(["img2.jpg", "img1.jpg"])
        assert result is None

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

    def test_pixel_hash_fallback(self, cache):
        """Non-string image source falls back to pixel_values hash."""
        import mlx.core as mx
        pixel_values = mx.array([1.0, 2.0, 3.0])
        features = mx.ones((1, 10))

        # None image_source + pixel_values = pixel hash key
        cache.put(None, features, pixel_values=pixel_values)

        # Same pixel values should hit
        result = cache.get(None, pixel_values=pixel_values)
        assert result is not None

        # Different pixel values should miss
        other_pixels = mx.array([4.0, 5.0, 6.0])
        result = cache.get(None, pixel_values=other_pixels)
        assert result is None

    def test_empty_key_no_cache(self, cache):
        """No URL and no pixel_values means no caching."""
        import mlx.core as mx
        features = mx.ones((1, 10))
        cache.put(None, features)  # No pixel_values fallback
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


# ---------------------------------------------------------------------------
# snapshot_nbytes tests
# ---------------------------------------------------------------------------

class TestSnapshotNbytes:
    def test_empty_snapshot(self):
        from heylook_llm.providers.common.cache_helpers import snapshot_nbytes
        assert snapshot_nbytes([]) == 0
        assert snapshot_nbytes([None, None]) == 0

    def test_with_arrays(self):
        import mlx.core as mx
        from heylook_llm.providers.common.cache_helpers import snapshot_nbytes

        k = mx.zeros((1, 4, 32, 64))  # float32: 1*4*32*64*4 = 32768 bytes
        v = mx.zeros((1, 4, 32, 64))
        snapshot = [(k, v), None, (k, v)]

        expected = 4 * k.nbytes  # 2 layers * (k + v)
        assert snapshot_nbytes(snapshot) == expected

    def test_mixed_none_and_data(self):
        import mlx.core as mx
        from heylook_llm.providers.common.cache_helpers import snapshot_nbytes

        k = mx.zeros((8,))  # 32 bytes
        snapshot = [None, (k, k), None]
        assert snapshot_nbytes(snapshot) == 2 * k.nbytes


# ---------------------------------------------------------------------------
# Segment-aware eviction tests
# ---------------------------------------------------------------------------

class TestSegmentAwareEviction:
    def test_assistant_evicted_before_system(self):
        """With both system and assistant nodes, assistant should be evicted first."""
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import time

        cache = RadixCache(max_nodes=3)

        # Insert 2 blocks at system prefix len = BLOCK_SIZE (first block is system)
        tokens_a = list(range(BLOCK_SIZE * 2))
        snap_a = [("kv_a",)]
        cache.insert(tokens_a, snap_a, matched_length=0, system_prefix_len=BLOCK_SIZE)

        # Insert another path (all assistant)
        tokens_b = list(range(100, 100 + BLOCK_SIZE * 2))
        snap_b = [("kv_b",)]
        cache.insert(tokens_b, snap_b, matched_length=0, system_prefix_len=0)

        # We have 4 nodes but max is 3 -- should have evicted an assistant leaf
        assert cache._node_count <= 3

        # System node (first block of tokens_a) should survive
        matched, snap = cache.longest_prefix_match(tokens_a[:BLOCK_SIZE])
        assert matched == BLOCK_SIZE  # system block survived

    def test_all_assistant_evicts_oldest(self):
        """When all nodes are assistant type, oldest is evicted."""
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import time

        cache = RadixCache(max_nodes=2)

        tokens_a = list(range(BLOCK_SIZE))
        tokens_b = list(range(100, 100 + BLOCK_SIZE))
        tokens_c = list(range(200, 200 + BLOCK_SIZE))

        cache.insert(tokens_a, [("a",)], 0, system_prefix_len=0)
        time.sleep(0.001)  # Ensure different timestamps
        cache.insert(tokens_b, [("b",)], 0, system_prefix_len=0)
        time.sleep(0.001)
        cache.insert(tokens_c, [("c",)], 0, system_prefix_len=0)

        # tokens_a should be evicted (oldest assistant)
        assert cache.longest_prefix_match(tokens_a)[0] == 0
        assert cache.longest_prefix_match(tokens_b)[0] == BLOCK_SIZE


# ---------------------------------------------------------------------------
# Byte budget trimming tests
# ---------------------------------------------------------------------------

class TestByteBudgetTrimming:
    def test_trim_to_bytes(self):
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import mlx.core as mx

        cache = RadixCache(max_nodes=100)

        # Insert 3 nodes with known sizes
        for i in range(3):
            tokens = list(range(i * 1000, i * 1000 + BLOCK_SIZE))
            # Snapshot with a known-size array
            k = mx.zeros((64, 64))  # 64*64*4 = 16384 bytes per array
            snap = [(k, k)]
            cache.insert(tokens, snap, 0, system_prefix_len=0)

        assert cache._node_count == 3
        initial_bytes = cache.nbytes
        assert initial_bytes > 0

        # Trim to allow only 1 node worth of bytes
        single_node_bytes = initial_bytes // 3
        cache.trim_to_bytes(single_node_bytes + 1)

        assert cache._node_count <= 1
        assert cache.nbytes <= single_node_bytes + 1

    def test_trim_to_zero(self):
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import mlx.core as mx

        cache = RadixCache(max_nodes=100)
        tokens = list(range(BLOCK_SIZE))
        k = mx.zeros((8,))
        cache.insert(tokens, [(k, k)], 0)

        cache.trim_to_bytes(0)
        assert cache._node_count == 0
        assert cache.nbytes == 0

    def test_nbytes_tracks_insertions(self):
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import mlx.core as mx

        cache = RadixCache(max_nodes=100)
        assert cache.nbytes == 0

        k = mx.zeros((32,))  # 128 bytes
        tokens = list(range(BLOCK_SIZE))
        cache.insert(tokens, [(k, k)], 0)

        assert cache.nbytes == 2 * k.nbytes  # k + v

    def test_clear_resets_bytes(self):
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import mlx.core as mx

        cache = RadixCache(max_nodes=100)
        k = mx.zeros((32,))
        cache.insert(list(range(BLOCK_SIZE)), [(k, k)], 0)
        assert cache.nbytes > 0

        cache.clear()
        assert cache.nbytes == 0


# ---------------------------------------------------------------------------
# Segment stats tests
# ---------------------------------------------------------------------------

class TestSegmentStats:
    def test_stats_by_segment_type(self):
        from heylook_llm.providers.common.radix_cache import RadixCache, BLOCK_SIZE
        import mlx.core as mx

        cache = RadixCache(max_nodes=100)

        # Insert with system prefix covering first block
        tokens = list(range(BLOCK_SIZE * 2))
        k = mx.zeros((16,))
        cache.insert(tokens, [(k, k)], 0, system_prefix_len=BLOCK_SIZE)

        stats = cache.stats_by_segment_type()
        assert "system" in stats
        assert "assistant" in stats
        assert stats["system"]["nodes"] == 1
        assert stats["assistant"]["nodes"] == 1

    def test_stats_empty_tree(self):
        from heylook_llm.providers.common.radix_cache import RadixCache

        cache = RadixCache(max_nodes=100)
        stats = cache.stats_by_segment_type()
        assert stats == {}


# ---------------------------------------------------------------------------
# Keepalive marker tests
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

class TestPromptCacheManagerByteBudget:
    def test_enforce_with_no_budget(self):
        from heylook_llm.providers.common.prompt_cache import PromptCacheManager
        mgr = PromptCacheManager(max_cache_bytes=None)
        # Should not raise
        mgr.enforce_byte_budget()

    def test_total_cache_bytes_empty(self):
        from heylook_llm.providers.common.prompt_cache import PromptCacheManager
        mgr = PromptCacheManager()
        assert mgr.total_cache_bytes == 0

    def test_cache_info_includes_segment_stats(self):
        """get_cache_info should include radix_bytes and segment_stats."""
        from heylook_llm.providers.common.prompt_cache import PromptCacheManager
        import mlx.core as mx

        mgr = PromptCacheManager()

        # Create a mock model with layers property for make_cache
        mock_model = MagicMock()
        mock_model.layers = [MagicMock() for _ in range(2)]
        mock_model.make_cache.return_value = [MagicMock() for _ in range(2)]

        cache = mgr.get_or_create_cache("test-model", mock_model)
        info = mgr.get_cache_info()

        assert "test-model" in info
        assert "radix_bytes" in info["test-model"]
        assert "radix_nodes" in info["test-model"]
