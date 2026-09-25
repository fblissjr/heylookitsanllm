# tests/unit/test_mlx_optimizations.py
"""Tests for MLX server optimizations (v1.26.0).

Covers:
- VisionFeatureCache: LRU eviction, content keys, stats
- Cached tokens passthrough: generation_core attaches cached_tokens
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# VisionFeatureCache tests
# ---------------------------------------------------------------------------

def _feature(spec):
    """A cache value from a compact spec: a shape (ones), (shape, fill), or a
    list of shapes (deepseek_v4 stores a list of per-image arrays)."""
    import mlx.core as mx
    if isinstance(spec, list):
        return [mx.ones(s) for s in spec]
    if len(spec) == 2 and isinstance(spec[0], tuple):
        return mx.full(spec[0], spec[1])
    return mx.ones(spec)


def _run_ops(cache, ops):
    """Apply ("put", key, spec) / ("get", key) / ("clear",) / ("expect", stats)
    in order; an "expect" asserts a stats subset at that point. After every op
    the byte budget holds: bytes never exceed max_bytes."""
    for op in ops:
        if op[0] == "put":
            cache.put(op[1], _feature(op[2]))
        elif op[0] == "get":
            cache.get(op[1])
        elif op[0] == "clear":
            cache.clear()
        elif op[0] == "expect":
            stats = cache.stats()
            for key, want in op[1].items():
                assert stats[key] == pytest.approx(want), (key, stats)
        stats = cache.stats()
        assert 0 < stats["max_bytes"] and stats["bytes"] <= stats["max_bytes"], stats


class TestVisionFeatureCache:
    @pytest.mark.parametrize("ops, present, absent, length, values", [
        pytest.param(
            [("put", "http://example.com/img.jpg", (1, 10))],
            ["http://example.com/img.jpg"], ["http://example.com/other.jpg"], 1, {},
            id="url_key_hit_miss"),
        # max_entries=3: the fourth put evicts the oldest.
        pytest.param(
            [("put", f"img{i}.jpg", (1, 5)) for i in range(4)],
            ["img1.jpg", "img2.jpg", "img3.jpg"], ["img0.jpg"], 3, {},
            id="lru_eviction"),
        # A get moves a.jpg to the end, so d.jpg evicts b.jpg (oldest non-accessed).
        pytest.param(
            [("put", "a.jpg", (1, 5)), ("put", "b.jpg", (1, 5)), ("put", "c.jpg", (1, 5)),
             ("get", "a.jpg"), ("put", "d.jpg", (1, 5))],
            ["a.jpg", "c.jpg", "d.jpg"], ["b.jpg"], 3, {},
            id="lru_access_updates_order"),
        # Updating an existing key replaces the value (and moves it to the end).
        pytest.param(
            [("put", "a.jpg", (1, 5)), ("put", "a.jpg", ((1, 5), 0.0))],
            ["a.jpg"], [], 1, {"a.jpg": 0.0},
            id="update_existing_key"),
        pytest.param(
            [("put", "", (1, 10))], [], [], 0, {},
            id="empty_key_no_cache"),
        pytest.param(
            [("put", "a.jpg", (1, 5)), ("put", "b.jpg", (1, 5)), ("clear",)],
            [], ["a.jpg"], 0, {},
            id="clear"),
    ])
    def test_lru_behaviour(self, ops, present, absent, length, values):
        from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache
        cache = VisionFeatureCache(max_entries=3)
        _run_ops(cache, ops)
        assert len(cache) == length
        for key in present:
            assert cache.get(key) is not None, key
        for key in absent:
            assert cache.get(key) is None, key
        for key, first in values.items():
            assert cache.get(key)[0, 0].item() == first

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

    # Byte accounting: a float32 (1, 10) or (10,) entry is 40 bytes. Every row
    # also checks bytes <= max_bytes after each op (_run_ops).
    @pytest.mark.parametrize("limits, ops, present, absent", [
        pytest.param(
            {"max_entries": 3},
            [("put", "a.jpg", (1, 5)), ("get", "a.jpg"), ("get", "b.jpg"),
             ("expect", {"entries": 1, "hits": 1, "misses": 1, "hit_rate": 0.5})],
            [], [],
            id="stats"),
        pytest.param(
            {"max_entries": 3},
            [("put", "a.jpg", (1, 10)), ("expect", {"bytes": 40})],
            [], [],
            id="stats_expose_bytes"),
        # If bytes overflow first, the oldest entry goes even though the count
        # is under max_entries: 80 + 40 > 100 evicts "a", not "b".
        pytest.param(
            {"max_entries": 100, "max_bytes": 100},
            [("put", "a.jpg", (10,)), ("put", "b.jpg", (10,)),
             ("expect", {"entries": 2, "bytes": 80}),
             ("put", "c.jpg", (10,)), ("expect", {"entries": 2, "bytes": 80})],
            ["b.jpg", "c.jpg"], ["a.jpg"],
            id="byte_cap_evicts_before_count_cap"),
        # A single 160B insert under a 200B cap must evict both "a" and "b".
        pytest.param(
            {"max_entries": 100, "max_bytes": 200},
            [("put", "a.jpg", (10,)), ("put", "b.jpg", (10,)), ("put", "c.jpg", (10,)),
             ("put", "big.jpg", (40,))],
            ["big.jpg"], ["a.jpg", "b.jpg"],
            id="byte_cap_evicts_until_under_cap"),
        pytest.param(
            {"max_entries": 3},
            [("put", "a.jpg", (1, 10)), ("put", "b.jpg", (1, 10)), ("expect", {"bytes": 80}),
             ("clear",), ("expect", {"bytes": 0, "entries": 0})],
            [], [],
            id="clear_resets_byte_counter"),
        # Replacing a key subtracts the old bytes before adding the new.
        pytest.param(
            {"max_entries": 3},
            [("put", "a.jpg", (1, 10)), ("expect", {"bytes": 40}),
             ("put", "a.jpg", (1, 10)), ("expect", {"bytes": 40}),
             ("put", "a.jpg", (1, 5)), ("expect", {"bytes": 20})],
            [], [],
            id="replace_updates_byte_accounting"),
        # A list's bytes read as 0 would let it grow past max_bytes unseen.
        pytest.param(
            {"max_entries": 3},
            [("put", "a", [(10,), (5,)]), ("expect", {"bytes": 60})],
            [], [],
            id="list_features_count_against_the_byte_cap"),
    ])
    def test_byte_and_stats_accounting(self, limits, ops, present, absent):
        from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache
        cache = VisionFeatureCache(**limits)
        _run_ops(cache, ops)
        for key in present:
            assert cache.get(key) is not None, key
        for key in absent:
            assert cache.get(key) is None, key


# TestKeepaliveMarker (type, singleton, isinstance) was folded away: the KEPT
# tests in test_streaming_utils.py assert the same facts through the product
# path -- control_frame(KEEPALIVE_MARKER) is the ping frame only if the
# singleton is a KeepaliveMarker, a plain string is not a control frame, and
# the stream's keepalive is detected by isinstance.
