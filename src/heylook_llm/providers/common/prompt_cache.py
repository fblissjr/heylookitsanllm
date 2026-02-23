# src/heylook_llm/providers/common/prompt_cache.py
"""
Cross-request prompt cache management for MLX models.

Uses a radix tree per model for multi-prefix KV cache reuse. Editing earlier
messages, branching, or regenerating no longer invalidates the entire cache --
only the divergent suffix needs re-prefilling.
"""

import logging
import threading
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .cache_helpers import make_cache, snapshot_kv, restore_kv_from_snapshot
from .radix_cache import RadixCache


def _mlx_memory_pressure() -> bool:
    """Check if GPU memory exceeds 85% of recommended working set."""
    try:
        import mlx.core as mx
        active = mx.metal.get_active_memory()
        info = mx.metal.device_info()
        limit = info.get('max_recommended_working_set_size', float('inf'))
        return active > limit * 0.85
    except Exception:
        return False


@dataclass
class PromptCache:
    """Working cache for a single generation request.

    Holds the live KV cache objects that lm_stream_generate mutates,
    plus token tracking for context usage metrics. The radix tree
    is the persistent backing store; this is the ephemeral working copy.
    """
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)
    _radix_matched_len: int = 0  # How many tokens were restored from radix cache

    def __str__(self):
        return f"PromptCache(tokens={len(self.tokens)}, model={self.model_key[0]})"


class PromptCacheManager:
    """Manages per-model radix caches and working caches.

    Each model gets a RadixCache (persistent, multi-prefix) and a PromptCache
    (ephemeral, for the current generation). The public API is unchanged from
    the single-prefix implementation.

    Thread-safe: all public methods are protected by a reentrant lock.
    """

    def __init__(self, max_cache_entries: int = 10, max_radix_nodes: int = 128):
        self._radix_caches: dict[str, RadixCache] = {}
        self._working_caches: dict[str, PromptCache] = {}
        self._max_entries = max_cache_entries
        self._max_radix_nodes = max_radix_nodes
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def get_or_create_cache(self, model_id: str, model: Any, cache_config: dict = None) -> PromptCache:
        """Get working cache for a model (thread-safe).

        Creates the RadixCache for the model on first access. Returns a
        PromptCache with a fresh KV cache -- the radix tree lookup happens
        in process_prompt_with_cache().
        """
        with self._lock:
            cache_config = cache_config or {}

            # Ensure radix cache exists for this model
            if model_id not in self._radix_caches:
                self._radix_caches[model_id] = RadixCache(
                    max_nodes=self._max_radix_nodes,
                    memory_pressure_fn=_mlx_memory_pressure,
                )
                logging.debug(f"Created radix cache for {model_id} (max_nodes={self._max_radix_nodes})")

            self._update_lru_unlocked(model_id)

            # Create fresh working cache (radix tree handles persistence)
            cache = PromptCache(
                cache=make_cache(model, cache_config),
                model_key=(model_id, None),
                tokens=[],
            )
            self._working_caches[model_id] = cache
            return cache

    def get_radix_cache(self, model_id: str) -> Optional[RadixCache]:
        """Get the radix cache for a model (thread-safe)."""
        with self._lock:
            return self._radix_caches.get(model_id)

    def _update_lru_unlocked(self, model_id: str):
        """Update LRU order. Must be called with lock held."""
        if model_id in self._access_order:
            self._access_order.remove(model_id)
        self._access_order.append(model_id)

        while len(self._access_order) > self._max_entries:
            evicted = self._access_order.pop(0)
            self._radix_caches.pop(evicted, None)
            self._working_caches.pop(evicted, None)
            logging.debug(f"Evicted caches for {evicted} (LRU)")

    def invalidate_cache(self, model_id: str):
        """Invalidate all caches for a specific model (thread-safe)."""
        with self._lock:
            if model_id in self._radix_caches:
                self._radix_caches[model_id].clear()
            self._working_caches.pop(model_id, None)
            logging.debug(f"Invalidated caches for {model_id}")

    def clear_all(self):
        """Clear all caches (thread-safe)."""
        with self._lock:
            for rc in self._radix_caches.values():
                rc.clear()
            self._radix_caches.clear()
            self._working_caches.clear()
            self._access_order.clear()
            logging.debug("Cleared all prompt caches")

    def get_cache_info(self) -> dict:
        """Get information about cached prompts (thread-safe)."""
        with self._lock:
            info = {}
            for model_id, working in self._working_caches.items():
                radix = self._radix_caches.get(model_id)
                info[model_id] = {
                    "tokens_cached": len(working.tokens),
                    "cache_layers": len(working.cache),
                    "radix_nodes": radix._node_count if radix else 0,
                }
            return info

    def get_context_usage(self, model_id: str) -> int:
        """Thread-safe method to get context usage for a model."""
        with self._lock:
            working = self._working_caches.get(model_id)
            if working:
                return len(working.tokens)
            return 0


def find_common_prefix_length(cached_tokens: List[int], new_tokens: List[int]) -> int:
    """Find the length of the common prefix between cached and new tokens.

    Retained as a utility for callers that need linear prefix comparison.
    The main cache path now uses radix tree lookup instead.
    """
    min_len = min(len(cached_tokens), len(new_tokens))
    for i in range(min_len):
        if cached_tokens[i] != new_tokens[i]:
            return i
    return min_len


def process_prompt_with_cache(
    prompt_cache: PromptCache,
    new_tokens: List[int],
    model: Any,
    cache_config: dict = None,
) -> Tuple[List[int], PromptCache]:
    """Process a prompt using radix-tree cached KV states when available.

    Looks up the global radix cache for the longest matching prefix. If found,
    restores KV state from the snapshot and returns only the suffix tokens for
    processing. Otherwise, returns the full token sequence with a fresh cache.

    Args:
        prompt_cache: Working cache (from get_or_create_cache).
        new_tokens: Tokenized prompt for the new request.
        model: The model (for cache creation if needed).
        cache_config: Cache type configuration.

    Returns:
        Tuple of (tokens_to_process, updated_prompt_cache).
    """
    cache_config = cache_config or {}
    model_id = prompt_cache.model_key[0]

    # Get radix cache for this model
    manager = get_global_cache_manager()
    radix = manager.get_radix_cache(model_id)

    if radix is not None and new_tokens:
        matched_len, kv_snapshot = radix.longest_prefix_match(new_tokens)

        if kv_snapshot is not None and matched_len > 0:
            # Restore KV state from radix tree snapshot
            prompt_cache.cache = restore_kv_from_snapshot(kv_snapshot, model, cache_config)
            prompt_cache._radix_matched_len = matched_len

            # Ensure at least one token to process (mlx-lm requirement)
            tokens_to_process = new_tokens[matched_len:]
            if not tokens_to_process:
                tokens_to_process = new_tokens[-1:]
                prompt_cache._radix_matched_len = len(new_tokens) - 1

            prompt_cache.tokens = new_tokens
            logging.info(
                f"Radix cache hit: reusing {prompt_cache._radix_matched_len}/{len(new_tokens)} tokens, "
                f"processing {len(tokens_to_process)} new"
            )
            return tokens_to_process, prompt_cache

    # No radix match -- process all tokens with fresh cache
    prompt_cache.cache = make_cache(model, cache_config)
    prompt_cache.tokens = new_tokens
    prompt_cache._radix_matched_len = 0
    logging.info(f"Radix cache miss: processing all {len(new_tokens)} tokens (model={model_id})")
    return new_tokens, prompt_cache


def store_generation_cache(
    prompt_cache: PromptCache,
    full_tokens: List[int],
    generation_cache: List[Any],
) -> None:
    """Store KV cache snapshot in the radix tree after generation completes.

    Called from strategy finally blocks. Snapshots the live KV cache and
    inserts it into the radix tree so future requests can reuse the prefix.

    Args:
        prompt_cache: The working cache for this generation.
        full_tokens: Complete token sequence (prompt + generated).
        generation_cache: The live KV cache objects (mutated by lm_stream_generate).
    """
    model_id = prompt_cache.model_key[0]
    manager = get_global_cache_manager()
    radix = manager.get_radix_cache(model_id)

    if radix is None or not generation_cache:
        return

    kv_snap = snapshot_kv(generation_cache)
    radix.insert(full_tokens, kv_snap, prompt_cache._radix_matched_len)
    prompt_cache.tokens = full_tokens

    logging.debug(
        f"Stored radix snapshot: {len(full_tokens)} tokens "
        f"(matched={prompt_cache._radix_matched_len}, new={len(full_tokens) - prompt_cache._radix_matched_len})"
    )


# Global cache manager instance
_global_cache_manager = PromptCacheManager()


def get_global_cache_manager() -> PromptCacheManager:
    """Get the global prompt cache manager instance."""
    return _global_cache_manager
