# src/heylook_llm/providers/common/radix_cache.py
"""Radix tree for multi-prefix KV cache reuse.

Supports multiple cached prefixes simultaneously so that editing earlier
messages, branching, or regenerating with different system prompts does not
invalidate the entire cache. Each tree node holds a block of BLOCK_SIZE
tokens and optionally a KV cache snapshot taken at the end of that block.

Thread-safe: all public methods are protected by a reentrant lock.

Known limitation -- hybrid models (e.g. Qwen3.5):

    Models with mixed cache types (KVCache for attention layers, ArraysCache
    for SSM/recurrent layers) have a fundamental mismatch with radix caching.

    KVCache is position-indexed: keys[i] corresponds to token at position i.
    Trimming to a prefix is a simple slice (keys[:N]).

    ArraysCache holds recurrent state (conv buffers, SSM state) that is a
    compressed summary of ALL tokens seen so far. There is no way to "rewind"
    it to an earlier position without reprocessing from scratch.

    When a snapshot is taken at end-of-generation (prompt + generated tokens)
    and later restored for a partial prefix match, the KVCache layers are
    trimmed to the matched prefix via restore_kv_from_snapshot(trim_to=N).
    ArraysCache layers keep whatever state the snapshot had -- which may
    reflect tokens beyond the matched prefix. This is technically incorrect
    but does not crash; the SSM layers will produce slightly different outputs
    than a fresh computation from just the prefix tokens.

    If strict correctness is needed for hybrid models, consider bypassing the
    radix cache entirely (detect mixed cache types in make_cache and skip the
    radix path in process_prompt_with_cache).

    See internal/bugs/radix_cache_vlm_crash.md for the full postmortem.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

BLOCK_SIZE = 32  # tokens per node -- configurable at module level

# Eviction priority order: assistant caches are evicted first (lowest value),
# system caches last (highest value). Matches mlx-lm's LRUPromptCache.CacheOrder.
SEGMENT_SYSTEM = "system"
SEGMENT_ASSISTANT = "assistant"

EVICTION_PRIORITY: dict[str, int] = {
    SEGMENT_ASSISTANT: 0,
    SEGMENT_SYSTEM: 1,
}


@dataclass
class RadixNode:
    """A single node in the radix tree."""
    token_block: tuple[int, ...]                     # immutable block of token IDs
    children: dict[int, RadixNode] = field(default_factory=dict)  # first token of next block -> child
    kv_snapshot: list[Any] | None = None             # KV cache state at END of this block
    last_access: float = field(default_factory=time.monotonic)
    depth: int = 0
    snapshot_bytes: int = 0                          # byte size of kv_snapshot for budget tracking
    segment_type: str = SEGMENT_ASSISTANT               # SEGMENT_SYSTEM or SEGMENT_ASSISTANT for eviction priority


class RadixCache:
    """Radix tree for multi-prefix KV cache reuse.

    Each node stores a fixed-size block of tokens. KV snapshots are only
    stored at block boundaries, quantizing reuse to BLOCK_SIZE granularity
    (on average ~16 tokens of wasted re-prefill, but 32x fewer snapshots).
    """

    def __init__(self, max_nodes: int = 128, memory_pressure_fn: callable | None = None):
        self.root = RadixNode(token_block=(), children={}, depth=0)
        self.max_nodes = max_nodes
        self._node_count = 0
        self._total_bytes = 0
        self._lock = threading.RLock()
        self._memory_pressure_fn = memory_pressure_fn

    @property
    def nbytes(self) -> int:
        """Total bytes of KV snapshots stored in this tree."""
        with self._lock:
            return self._total_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def longest_prefix_match(self, tokens: list[int]) -> tuple[int, list[Any] | None]:
        """Find the longest cached prefix for a token sequence.

        Returns (matched_length, kv_cache_snapshot) where kv_cache_snapshot
        is the deepest node's KV state, or None if no match.
        """
        if not tokens:
            return 0, None

        with self._lock:
            matched = 0
            best_snapshot = None
            node = self.root

            blocks = self._chunk_tokens(tokens)

            for block in blocks:
                if len(block) < BLOCK_SIZE:
                    # Partial trailing block -- can't match a full node
                    break

                first_token = block[0]
                child = node.children.get(first_token)
                if child is None:
                    break

                # Verify full block match (not just the first-token key)
                if child.token_block != block:
                    break

                # Match found -- advance
                child.last_access = time.monotonic()
                matched += BLOCK_SIZE
                if child.kv_snapshot is not None:
                    best_snapshot = child.kv_snapshot
                node = child

            return matched, best_snapshot

    def insert(self, tokens: list[int], kv_snapshot: list[Any], matched_length: int,
               system_prefix_len: int = 0) -> None:
        """Insert KV state for a token sequence from matched_length onward.

        Only creates nodes at block boundaries. The kv_snapshot is attached
        to the final complete block node.

        IMPORTANT: The snapshot typically contains KV state for the FULL
        sequence (prompt + generated tokens), not just up to the deepest
        block boundary. When a future request matches a shorter prefix,
        restore_kv_from_snapshot must trim KVCache layers to the matched
        length. Without trimming, the restored cache offset exceeds the
        prefix boundary, corrupting position computations in models that
        derive position IDs from cache.offset (e.g. Qwen3.5 mRoPE).
        See internal/bugs/radix_cache_vlm_crash.md.

        Args:
            tokens: Full token sequence (including the already-matched prefix).
            kv_snapshot: KV cache state to store (deep copy is caller's
                responsibility). MUST be materialized (see snapshot_kv in
                cache_helpers) -- lazy MLX arrays are pinned to the producing
                thread's GPU stream, and a consumer on another thread crashes
                with "There is no Stream(gpu, N) in current thread". Not
                asserted here: MLX has no cheap is-materialized predicate,
                and evaluating under self._lock would stall the tree.
            matched_length: Number of tokens already matched in the tree.
        """
        if not tokens:
            return

        with self._lock:
            blocks = self._chunk_tokens(tokens)

            # Walk to the node at matched_length
            node = self.root
            blocks_to_skip = matched_length // BLOCK_SIZE

            for i, block in enumerate(blocks):
                if len(block) < BLOCK_SIZE:
                    # Trailing partial block -- don't create a node
                    break

                if i < blocks_to_skip:
                    # Walk existing path
                    first_token = block[0]
                    child = node.children.get(first_token)
                    if child is None:
                        break
                    child.last_access = time.monotonic()
                    node = child
                    continue

                # Insert new block (or verify existing)
                first_token = block[0]
                child = node.children.get(first_token)

                if child is not None and child.token_block == block:
                    # Node already exists, just update access time
                    child.last_access = time.monotonic()
                    node = child
                    continue

                # Compute segment type once for both replace and insert paths
                block_end = (i + 1) * BLOCK_SIZE
                seg_type = SEGMENT_SYSTEM if block_end <= system_prefix_len else SEGMENT_ASSISTANT

                if child is not None:
                    # Different block with same first token -- replace.
                    # Subtract replaced subtree's bytes to prevent budget drift.
                    self._total_bytes -= self._subtree_bytes(child)
                    self._node_count -= self._subtree_count(child)
                else:
                    # New insertion -- evict if at capacity
                    if self._node_count >= self.max_nodes or self._check_memory_pressure():
                        self._evict_lru_unlocked()

                new_node = RadixNode(
                    token_block=block,
                    children={},
                    depth=i + 1,
                    segment_type=seg_type,
                )
                node.children[first_token] = new_node
                self._node_count += 1
                node = new_node

            # Attach snapshot to the deepest node we reached
            if node is not self.root:
                # Subtract old snapshot bytes if replacing
                if node.kv_snapshot is not None:
                    self._total_bytes -= node.snapshot_bytes

                node.kv_snapshot = kv_snapshot
                node.last_access = time.monotonic()

                # Track snapshot size for byte budget
                from .cache_helpers import snapshot_nbytes
                snap_bytes = snapshot_nbytes(kv_snapshot) if kv_snapshot else 0
                node.snapshot_bytes = snap_bytes
                self._total_bytes += snap_bytes

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self.root.children.clear()
            self._node_count = 0
            self._total_bytes = 0

    def stats_by_segment_type(self) -> dict[str, dict[str, int]]:
        """Return node count and byte totals grouped by segment type.

        Returns dict like:
            {"system": {"nodes": 3, "bytes": 1048576},
             "assistant": {"nodes": 12, "bytes": 8388608}}
        """
        with self._lock:
            result: dict[str, dict[str, int]] = {}
            self._collect_segment_stats(self.root, result)
            return result

    def _collect_segment_stats(self, node: RadixNode, result: dict[str, dict[str, int]]) -> None:
        """Recursively sum node counts and bytes by segment type."""
        for child in node.children.values():
            seg = child.segment_type
            if seg not in result:
                result[seg] = {"nodes": 0, "bytes": 0}
            result[seg]["nodes"] += 1
            result[seg]["bytes"] += child.snapshot_bytes
            self._collect_segment_stats(child, result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_memory_pressure(self) -> bool:
        """Check if external memory pressure callback signals eviction needed."""
        if self._memory_pressure_fn is None:
            return False
        try:
            return self._memory_pressure_fn()
        except Exception:
            return False

    @staticmethod
    def _subtree_bytes(node: RadixNode) -> int:
        """Total snapshot bytes in a subtree (inclusive)."""
        total = node.snapshot_bytes
        for child in node.children.values():
            total += RadixCache._subtree_bytes(child)
        return total

    @staticmethod
    def _subtree_count(node: RadixNode) -> int:
        """Total node count in a subtree (inclusive)."""
        total = 1
        for child in node.children.values():
            total += RadixCache._subtree_count(child)
        return total

    @staticmethod
    def _chunk_tokens(tokens: list[int]) -> list[tuple[int, ...]]:
        """Split tokens into BLOCK_SIZE tuples."""
        return [
            tuple(tokens[i:i + BLOCK_SIZE])
            for i in range(0, len(tokens), BLOCK_SIZE)
        ]

    def trim_to_bytes(self, max_bytes: int) -> None:
        """Evict LRU leaves until total bytes is at or below max_bytes.

        Used by PromptCacheManager to enforce a process-wide byte budget.
        """
        with self._lock:
            while self._total_bytes > max_bytes and self._node_count > 0:
                self._evict_lru_unlocked()

    def _evict_lru_unlocked(self) -> None:
        """Remove the least-recently-accessed leaf node. Must hold lock.

        Segment-aware eviction: assistant nodes are evicted first, user nodes
        next, system nodes last. Within the same segment type, oldest (by
        last_access) is evicted first. This keeps system prompt KV caches
        alive longer, matching mlx-lm's LRUPromptCache priority ordering.
        """
        # Collect all leaf nodes with their parents
        leaves: list[tuple[float, int, RadixNode, RadixNode]] = []
        self._collect_leaves(self.root, leaves)

        if not leaves:
            return

        # Sort by (segment_priority ASC, last_access ASC) -- assistant first, oldest first
        leaves.sort(key=lambda x: (
            EVICTION_PRIORITY.get(x[2].segment_type, 0),
            x[0],
        ))

        # Remove the highest-priority eviction candidate
        _, first_token_key, leaf, parent = leaves[0]
        del parent.children[first_token_key]
        self._node_count -= 1
        self._total_bytes -= leaf.snapshot_bytes
        logging.debug(
            f"Evicted radix cache leaf (depth={leaf.depth}, "
            f"bytes={leaf.snapshot_bytes}, "
            f"age={time.monotonic() - leaf.last_access:.1f}s)"
        )

    def _collect_leaves(
        self,
        node: RadixNode,
        result: list[tuple[float, int, RadixNode, RadixNode]],
    ) -> None:
        """Recursively collect (last_access, key, leaf, parent) for all leaf nodes."""
        for key, child in node.children.items():
            if not child.children:
                # Leaf node
                result.append((child.last_access, key, child, node))
            else:
                # Internal node -- recurse
                self._collect_leaves(child, result)
