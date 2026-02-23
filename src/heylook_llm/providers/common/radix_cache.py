# src/heylook_llm/providers/common/radix_cache.py
"""Radix tree for multi-prefix KV cache reuse.

Supports multiple cached prefixes simultaneously so that editing earlier
messages, branching, or regenerating with different system prompts does not
invalidate the entire cache. Each tree node holds a block of BLOCK_SIZE
tokens and optionally a KV cache snapshot taken at the end of that block.

Thread-safe: all public methods are protected by a reentrant lock.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

BLOCK_SIZE = 32  # tokens per node -- configurable at module level


@dataclass
class RadixNode:
    """A single node in the radix tree."""
    token_block: tuple[int, ...]                     # immutable block of token IDs
    children: dict[int, RadixNode] = field(default_factory=dict)  # first token of next block -> child
    kv_snapshot: list[Any] | None = None             # KV cache state at END of this block
    last_access: float = field(default_factory=time.monotonic)
    depth: int = 0


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
        self._lock = threading.RLock()
        self._memory_pressure_fn = memory_pressure_fn

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

    def insert(self, tokens: list[int], kv_snapshot: list[Any], matched_length: int) -> None:
        """Insert KV state for a token sequence from matched_length onward.

        Only creates nodes at block boundaries. The kv_snapshot is attached
        to the final complete block node.

        Args:
            tokens: Full token sequence (including the already-matched prefix).
            kv_snapshot: KV cache state to store (deep copy is caller's responsibility).
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
                elif child is not None and child.token_block != block:
                    # Different block with same first token -- replace
                    # (rare: two blocks share first token but differ later)
                    new_node = RadixNode(
                        token_block=block,
                        children={},
                        depth=i + 1,
                    )
                    node.children[first_token] = new_node
                    # Don't decrement _node_count for replaced node's subtree
                    # since eviction handles cleanup -- just replace the leaf
                    node = new_node
                else:
                    # Evict if needed before inserting (node count OR memory pressure)
                    if self._node_count >= self.max_nodes or self._check_memory_pressure():
                        self._evict_lru_unlocked()

                    new_node = RadixNode(
                        token_block=block,
                        children={},
                        depth=i + 1,
                    )
                    node.children[first_token] = new_node
                    self._node_count += 1
                    node = new_node

            # Attach snapshot to the deepest node we reached
            if node is not self.root:
                node.kv_snapshot = kv_snapshot
                node.last_access = time.monotonic()

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self.root.children.clear()
            self._node_count = 0

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
    def _chunk_tokens(tokens: list[int]) -> list[tuple[int, ...]]:
        """Split tokens into BLOCK_SIZE tuples."""
        return [
            tuple(tokens[i:i + BLOCK_SIZE])
            for i in range(0, len(tokens), BLOCK_SIZE)
        ]

    def _evict_lru_unlocked(self) -> None:
        """Remove the least-recently-accessed leaf node. Must hold lock."""
        # Collect all leaf nodes with their parents
        leaves: list[tuple[float, int, RadixNode, RadixNode]] = []
        self._collect_leaves(self.root, leaves)

        if not leaves:
            return

        # Sort by last_access (oldest first)
        leaves.sort(key=lambda x: x[0])

        # Remove the oldest leaf
        _, first_token_key, leaf, parent = leaves[0]
        del parent.children[first_token_key]
        self._node_count -= 1
        logging.debug(f"Evicted radix cache leaf (depth={leaf.depth}, age={time.monotonic() - leaf.last_access:.1f}s)")

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
