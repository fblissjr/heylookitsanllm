# tests/unit/test_radix_cache.py
"""Tests for the radix-tree prefix cache.

Covers insert, match, eviction, block boundaries, thread safety,
and partial-match edge cases.
"""

import threading

from heylook_llm.providers.common.radix_cache import (
    BLOCK_SIZE,
    RadixCache,
)


def _make_tokens(n: int, offset: int = 0) -> list[int]:
    """Generate a deterministic token sequence of length n."""
    return list(range(offset, offset + n))


class TestRadixCacheBasics:
    """Core insert / match behavior."""

    def test_empty_cache_returns_no_match(self):
        cache = RadixCache(max_nodes=128)
        matched, snapshot = cache.longest_prefix_match([1, 2, 3])
        assert matched == 0
        assert snapshot is None

    def test_insert_then_exact_match(self):
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE)
        fake_kv = ["kv_state"]
        cache.insert(tokens, fake_kv, matched_length=0)

        matched, snapshot = cache.longest_prefix_match(tokens)
        assert matched == BLOCK_SIZE
        assert snapshot == fake_kv

    def test_insert_two_blocks_match_both(self):
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE * 2)
        fake_kv = ["kv_two_blocks"]
        cache.insert(tokens, fake_kv, matched_length=0)

        matched, snapshot = cache.longest_prefix_match(tokens)
        assert matched == BLOCK_SIZE * 2
        assert snapshot == fake_kv

    def test_partial_block_not_stored(self):
        """Tokens shorter than BLOCK_SIZE produce no snapshot."""
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE - 1)
        cache.insert(tokens, ["kv"], matched_length=0)

        matched, snapshot = cache.longest_prefix_match(tokens)
        # No full block was completed, so no match
        assert matched == 0
        assert snapshot is None

    def test_prefix_match_returns_deepest_snapshot(self):
        """Query longer than cached prefix returns the deepest snapshot."""
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE)
        cache.insert(tokens, ["kv_one"], matched_length=0)

        longer = _make_tokens(BLOCK_SIZE * 3)
        matched, snapshot = cache.longest_prefix_match(longer)
        assert matched == BLOCK_SIZE
        assert snapshot == ["kv_one"]

    def test_divergent_prefix_returns_common_ancestor(self):
        """Two sequences sharing a prefix should return the shared part."""
        cache = RadixCache(max_nodes=128)

        # Insert sequence A: [0..63]
        tokens_a = _make_tokens(BLOCK_SIZE * 2)
        cache.insert(tokens_a, ["kv_a"], matched_length=0)

        # Insert sequence B: same first block, different second block
        tokens_b = _make_tokens(BLOCK_SIZE) + _make_tokens(BLOCK_SIZE, offset=1000)
        cache.insert(tokens_b, ["kv_b"], matched_length=BLOCK_SIZE)

        # Query with sequence A should get kv_a
        matched_a, snap_a = cache.longest_prefix_match(tokens_a)
        assert matched_a == BLOCK_SIZE * 2
        assert snap_a == ["kv_a"]

        # Query with sequence B should get kv_b
        matched_b, snap_b = cache.longest_prefix_match(tokens_b)
        assert matched_b == BLOCK_SIZE * 2
        assert snap_b == ["kv_b"]

    def test_branching_at_block_boundary(self):
        """Two sequences that diverge at a block boundary create two children."""
        cache = RadixCache(max_nodes=128)

        shared = _make_tokens(BLOCK_SIZE)
        branch_a = shared + _make_tokens(BLOCK_SIZE, offset=500)
        branch_b = shared + _make_tokens(BLOCK_SIZE, offset=600)

        cache.insert(branch_a, ["kv_a"], matched_length=0)
        cache.insert(branch_b, ["kv_b"], matched_length=BLOCK_SIZE)

        # Verify branching
        _, snap_a = cache.longest_prefix_match(branch_a)
        _, snap_b = cache.longest_prefix_match(branch_b)
        assert snap_a == ["kv_a"]
        assert snap_b == ["kv_b"]


class TestRadixCacheBlockBoundaries:
    """Edge cases around BLOCK_SIZE quantization."""

    def test_exact_one_block(self):
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE)
        cache.insert(tokens, ["kv"], matched_length=0)
        matched, _ = cache.longest_prefix_match(tokens)
        assert matched == BLOCK_SIZE

    def test_one_token_past_block(self):
        """BLOCK_SIZE + 1 tokens: only one full block stored."""
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE + 1)
        cache.insert(tokens, ["kv"], matched_length=0)
        matched, _ = cache.longest_prefix_match(tokens)
        # Only one full block worth of tokens matched
        assert matched == BLOCK_SIZE

    def test_three_blocks_exact(self):
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE * 3)
        cache.insert(tokens, ["kv"], matched_length=0)
        matched, _ = cache.longest_prefix_match(tokens)
        assert matched == BLOCK_SIZE * 3

    def test_insert_from_midpoint(self):
        """Insert starting from matched_length > 0."""
        cache = RadixCache(max_nodes=128)

        # Pre-populate first block
        first_block = _make_tokens(BLOCK_SIZE)
        cache.insert(first_block, ["kv_first"], matched_length=0)

        # Extend by two more blocks
        full = first_block + _make_tokens(BLOCK_SIZE * 2, offset=100)
        cache.insert(full, ["kv_full"], matched_length=BLOCK_SIZE)

        matched, snap = cache.longest_prefix_match(full)
        assert matched == BLOCK_SIZE * 3
        assert snap == ["kv_full"]


class TestRadixCacheEviction:
    """LRU eviction under node pressure."""

    def test_eviction_respects_max_nodes(self):
        # Very small cache
        cache = RadixCache(max_nodes=3)

        # Insert 4 different single-block entries (each is 1 node)
        for i in range(4):
            tokens = _make_tokens(BLOCK_SIZE, offset=i * 1000)
            cache.insert(tokens, [f"kv_{i}"], matched_length=0)

        # Oldest entry should be evicted
        assert cache._node_count <= 3

    def test_recently_accessed_survives_eviction(self):
        cache = RadixCache(max_nodes=3)

        # Insert 3 entries
        entries = []
        for i in range(3):
            tokens = _make_tokens(BLOCK_SIZE, offset=i * 1000)
            cache.insert(tokens, [f"kv_{i}"], matched_length=0)
            entries.append(tokens)

        # Access entry 0 (touch it to make it recent)
        cache.longest_prefix_match(entries[0])

        # Insert entry 3 -- should evict entry 1 (oldest untouched)
        tokens_3 = _make_tokens(BLOCK_SIZE, offset=3000)
        cache.insert(tokens_3, ["kv_3"], matched_length=0)

        # Entry 0 should still be accessible (was touched)
        matched, snap = cache.longest_prefix_match(entries[0])
        assert snap == ["kv_0"]

    def test_evict_leaf_not_internal(self):
        """Eviction should remove leaves, not internal nodes with children."""
        cache = RadixCache(max_nodes=4)

        shared = _make_tokens(BLOCK_SIZE)
        branch_a = shared + _make_tokens(BLOCK_SIZE, offset=500)
        branch_b = shared + _make_tokens(BLOCK_SIZE, offset=600)

        # Insert shared + A (2 nodes: shared, A)
        cache.insert(branch_a, ["kv_a"], matched_length=0)
        # Insert B using shared prefix (1 new node: B)
        cache.insert(branch_b, ["kv_b"], matched_length=BLOCK_SIZE)

        # 3 nodes total. Insert a 4th different entry.
        other = _make_tokens(BLOCK_SIZE, offset=9000)
        cache.insert(other, ["kv_other"], matched_length=0)

        # All 4 nodes fit. Now add a 5th to trigger eviction.
        yet_another = _make_tokens(BLOCK_SIZE, offset=8000)
        cache.insert(yet_another, ["kv_yet"], matched_length=0)

        # Should still be able to reach at least one of the branched entries
        # (shared internal node should not be evicted while it has children)
        _, snap_a = cache.longest_prefix_match(branch_a)
        _, snap_b = cache.longest_prefix_match(branch_b)
        # At least the shared prefix should survive
        shared_matched, _ = cache.longest_prefix_match(shared)
        assert shared_matched == BLOCK_SIZE


class TestRadixCacheThreadSafety:
    """Concurrent access must not corrupt the tree."""

    def test_concurrent_inserts(self):
        cache = RadixCache(max_nodes=256)
        errors = []

        def inserter(thread_id: int):
            try:
                for i in range(10):
                    tokens = _make_tokens(BLOCK_SIZE, offset=thread_id * 10000 + i * 100)
                    cache.insert(tokens, [f"kv_{thread_id}_{i}"], matched_length=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inserter, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_reads_and_writes(self):
        cache = RadixCache(max_nodes=256)
        errors = []

        # Pre-populate
        base_tokens = _make_tokens(BLOCK_SIZE)
        cache.insert(base_tokens, ["base_kv"], matched_length=0)

        def reader():
            try:
                for _ in range(50):
                    cache.longest_prefix_match(base_tokens)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(50):
                    tokens = _make_tokens(BLOCK_SIZE, offset=(i + 1) * 1000)
                    cache.insert(tokens, [f"kv_{i}"], matched_length=0)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_inserts_and_evictions(self):
        """Small max_nodes forces eviction during concurrent inserts."""
        cache = RadixCache(max_nodes=5)
        errors = []

        def inserter(thread_id: int):
            try:
                for i in range(20):
                    tokens = _make_tokens(BLOCK_SIZE, offset=thread_id * 10000 + i * 100)
                    cache.insert(tokens, [f"kv_{thread_id}_{i}"], matched_length=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inserter, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Thread errors: {errors}"
        assert cache._node_count <= 5


class TestRadixCacheEmptyAndEdge:
    """Degenerate inputs."""

    def test_empty_token_list_match(self):
        cache = RadixCache(max_nodes=128)
        matched, snap = cache.longest_prefix_match([])
        assert matched == 0
        assert snap is None

    def test_empty_token_list_insert(self):
        """Inserting empty tokens is a no-op."""
        cache = RadixCache(max_nodes=128)
        cache.insert([], ["kv"], matched_length=0)
        assert cache._node_count == 0

    def test_clear(self):
        cache = RadixCache(max_nodes=128)
        tokens = _make_tokens(BLOCK_SIZE * 2)
        cache.insert(tokens, ["kv"], matched_length=0)
        assert cache._node_count > 0

        cache.clear()
        assert cache._node_count == 0
        matched, snap = cache.longest_prefix_match(tokens)
        assert matched == 0
        assert snap is None

    def test_node_count_tracking(self):
        cache = RadixCache(max_nodes=128)
        assert cache._node_count == 0

        tokens = _make_tokens(BLOCK_SIZE * 3)
        cache.insert(tokens, ["kv"], matched_length=0)
        assert cache._node_count == 3
