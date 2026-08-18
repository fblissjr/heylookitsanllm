# src/heylook_llm/providers/common/prompt_cache.py
"""
Cross-request prompt cache management for MLX models.

Single-slot cache per model (Q7, plan_2026-07.md: the radix tree, its
restore-time slicing, segment eviction and the system-boundary machinery are
DELETED -- mlx-lm-aligned: state round-trips through mlx-lm's own
state/meta_state properties and divergence uses its ``trim_prompt_cache``).
Each model keeps ONE slot: per-layer snapshots of the last completed
generation plus the token sequence the cache actually covers. A new request
reuses it in exactly two shapes:

    - EXTENSION (the multi-turn chat flow: new prompt = stored sequence +
      new turn): continue the cache as-is, process only the suffix. Valid
      for EVERY cache type -- including hybrid ArraysCache models (Qwen3.5
      GDN) and rotated sliding windows -- because nothing is sliced; this
      is just "keep generating".
    - TRIM to the common prefix (edit/regenerate: the new prompt diverges
      mid-sequence): drop the stored tail via mlx-lm's trim_prompt_cache,
      which is per-layer-type honest -- can_trim_prompt_cache is False for
      ArraysCache (recurrent state is a running summary, not positional)
      and for rotated RotatingKVCache, and those become a full re-prefill
      instead of a silently-wrong restore.

    Switching conversations = re-prefill. Accepted trade (guardrail #5).

This CLOSES the hybrid-model hole the radix implementation documented and
shipped with ("technically incorrect but does not crash... slightly
different outputs"): its keys[..., :N, :] restore-slicing left ArraysCache
state describing tokens beyond the boundary. Under exact prefix matching
that latent wrongness became constant, live-verified garbage on
Qwen3.5-0.8B (greedy: fresh="Paris.", restored="\\n\\n") -- which is how the
snapshot/slice approach died in review before it was ever committed.

The slot stores per-layer (state, meta_state) SNAPSHOTS, never live cache
objects, and every reuse reconstructs fresh cache objects from them. This
is load-bearing, not style: MLX arrays are immutable but cache OBJECTS are
not, and this server quarantines a wedged generator's worker thread alive
("Generator close timed out; quarantining its worker") -- a zombie
generation keeps rebinding .keys/.offset on its cache objects. Sharing
those objects through the slot handed the next request state mutating
under it (live-verified 2026-08-18: process-poisoning garbage during the
eval bank, unreproducible single-threaded). Snapshot arrays are immune:
the zombie rebinds its own attributes; the captured arrays never change.

Thread-affinity invariant (see postmortems/radix_thread_affinity.md): every
generation runs on its own worker thread with thread-local GPU streams, so
the snapshot arrays are MATERIALIZED (mx.eval, on the generating thread) at
store time. Publishing lazy state would crash the NEXT request's thread.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

from .cache_helpers import make_cache


def _mlx_memory_pressure() -> bool:
    """Check if GPU memory exceeds 85% of recommended working set."""
    try:
        active = mx.get_active_memory()
        info = mx.device_info()
        limit = info.get('max_recommended_working_set_size', float('inf'))
        return active > limit * 0.85
    except Exception:
        return False


@dataclass
class PromptCache:
    """Working cache for a single generation request.

    Holds the live KV cache objects that lm_stream_generate mutates, plus
    token tracking for context usage metrics. The per-model slot is the
    persistent backing store; this is the ephemeral working handle.
    """
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)
    _radix_matched_len: int = 0   # tokens reused from the slot (historical name)
    _radix_eligible: bool = True  # set per request by process_prompt_with_cache

    def __str__(self):
        return f"PromptCache(tokens={len(self.tokens)}, model={self.model_key[0]})"


def _flat_arrays(x, out: list) -> None:
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _flat_arrays(y, out)


def _materialize(layers: List[Tuple[Any, Any]]) -> int:
    """mx.eval every array in the CAPTURED snapshots ON THE CALLING THREAD
    and return the total byte size. Takes the snapshot list itself, never
    the cache: each ``.state`` access builds FRESH lazy slice objects, so
    evaluating a second capture leaves the stored one lazy -- and a lazy
    array restored on another thread is the "There is no Stream(gpu, N)"
    crash (postmortems/radix_thread_affinity.md; re-caught live in the
    chat e2e when exactly that eval/store split shipped here)."""
    arrays: list = []
    for state, _meta in layers:
        _flat_arrays(state, arrays)
    if arrays:
        # The security hook flags mx.eval; this is MLX's graph materializer,
        # not Python's eval().
        mx.eval(arrays)
    return sum(a.nbytes for a in arrays)


@dataclass
class _Slot:
    """One model's persistent cache: immutable per-layer snapshots of the
    last completed generation. ``layers`` holds (state, meta_state) pairs in
    layer order -- every cache type mlx-lm ships round-trips through those
    two properties (its own save/load_prompt_cache contract)."""
    tokens: List[int]
    layers: List[Tuple[Any, Any]]
    nbytes: int


def _snapshot_layers(cache: List[Any]) -> List[Tuple[Any, Any]]:
    return [(layer.state, getattr(layer, 'meta_state', None)) for layer in cache]


def _restore_layers(slot: _Slot, model: Any, cache_config: dict) -> List[Any]:
    """Fresh cache objects seeded from the slot's snapshots."""
    cache = make_cache(model, cache_config)
    for layer, (state, meta) in zip(cache, slot.layers):
        arrays: list = []
        _flat_arrays(state, arrays)
        if arrays:  # an empty layer's state round-trips as no-op
            layer.state = state
        if meta:
            layer.meta_state = meta
    return cache


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PromptCacheManager:
    """Per-model single-slot cache store + per-model working handles.

    The public API is unchanged from the radix implementation (api.py's
    cache endpoints, server.py's byte budget, memory.py's byte readout and
    the provider's context-usage/clear hooks all keep working); only the
    persistence machinery behind it shrank.

    Thread-safe: all public methods are protected by a reentrant lock.
    (Generations are additionally serialized by the process-global FIFO
    gate, so slot takes/puts never actually race -- the lock is the belt.)
    """

    def __init__(self, max_cache_entries: int = 10,
                 max_cache_bytes: int | None = None):
        self._slots: dict[str, _Slot] = {}
        self._working_caches: dict[str, PromptCache] = {}
        self._max_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def set_byte_budget(self, max_bytes: int | None) -> None:
        """Set the maximum byte budget for all slots combined."""
        with self._lock:
            self._max_cache_bytes = max_bytes

    def get_or_create_cache(self, model_id: str, model: Any, cache_config: dict | None = None) -> PromptCache:
        """Get a fresh working handle for a model (thread-safe).

        The slot lookup happens in process_prompt_with_cache().
        """
        with self._lock:
            cache_config = cache_config or {}
            self._update_lru_unlocked(model_id)
            cache = PromptCache(
                cache=make_cache(model, cache_config),
                model_key=(model_id, None),
                tokens=[],
            )
            self._working_caches[model_id] = cache
            return cache

    # -- slot access (internal to this module's lookup/store helpers) -------

    def _get_slot(self, model_id: str) -> Optional[_Slot]:
        with self._lock:
            slot = self._slots.get(model_id)
            if slot is not None:
                self._update_lru_unlocked(model_id)
            return slot

    def _put_slot(self, model_id: str, tokens: List[int], layers: List[Tuple[Any, Any]], nbytes: int) -> None:
        with self._lock:
            # Under GPU memory pressure, other models' slots are the
            # reclaimable thing we own -- drop them before storing this one.
            if _mlx_memory_pressure():
                for other in [m for m in self._slots if m != model_id]:
                    self._slots.pop(other, None)
                logging.debug("Memory pressure: dropped other models' cache slots")
            self._slots[model_id] = _Slot(tokens=list(tokens), layers=layers, nbytes=nbytes)
            self._update_lru_unlocked(model_id)

    def _update_lru_unlocked(self, model_id: str):
        """Update LRU order. Must be called with lock held."""
        if model_id in self._access_order:
            self._access_order.remove(model_id)
        self._access_order.append(model_id)

        while len(self._access_order) > self._max_entries:
            evicted = self._access_order.pop(0)
            self._slots.pop(evicted, None)
            self._working_caches.pop(evicted, None)
            logging.debug(f"Evicted caches for {evicted} (LRU)")

    def invalidate_cache(self, model_id: str):
        """Invalidate all caches for a specific model (thread-safe)."""
        with self._lock:
            self._slots.pop(model_id, None)
            self._working_caches.pop(model_id, None)
            logging.debug(f"Invalidated caches for {model_id}")

    def clear_all(self):
        """Clear all caches (thread-safe)."""
        with self._lock:
            self._slots.clear()
            self._working_caches.clear()
            self._access_order.clear()
            logging.debug("Cleared all prompt caches")

    @property
    def total_cache_bytes(self) -> int:
        """Total bytes across all slots."""
        with self._lock:
            return sum(s.nbytes for s in self._slots.values())

    def enforce_byte_budget(self) -> None:
        """Drop least-recently-used slots until under the byte budget.

        A slot is all-or-nothing (unlike the radix's partial trims): the
        stored cache is one live object graph, so the only trim is eviction.
        """
        if self._max_cache_bytes is None:
            return
        with self._lock:
            while (self.total_cache_bytes > self._max_cache_bytes
                   and any(m in self._slots for m in self._access_order)):
                for candidate in list(self._access_order):
                    if candidate in self._slots:
                        dropped = self._slots.pop(candidate)
                        logging.debug(
                            f"Byte budget: dropped {dropped.nbytes / (1024**2):.1f} MB "
                            f"slot for {candidate}"
                        )
                        break

    def get_cache_info(self) -> dict:
        """Get information about cached prompts (thread-safe)."""
        with self._lock:
            info = {}
            for model_id, working in self._working_caches.items():
                slot = self._slots.get(model_id)
                info[model_id] = {
                    "tokens_cached": len(working.tokens),
                    "cache_layers": len(working.cache),
                    "slot_tokens": len(slot.tokens) if slot else 0,
                    "slot_bytes": slot.nbytes if slot else 0,
                }
            return info

    def get_context_usage(self, model_id: str) -> int:
        """Thread-safe method to get context usage for a model."""
        with self._lock:
            working = self._working_caches.get(model_id)
            if working:
                return len(working.tokens)
            return 0


def process_prompt_with_cache(
    prompt_cache: PromptCache,
    new_tokens: List[int],
    model: Any,
    cache_config: dict | None = None,
) -> Tuple[List[int], PromptCache]:
    """Process a prompt reusing the model's slot when it is safe to.

    EXTENSION (stored sequence is a prefix of the new prompt): continue the
    live cache, process the suffix -- valid for every cache type. DIVERGENCE:
    trim the stored tail to the common prefix via mlx-lm's trim_prompt_cache
    when the cache supports it; otherwise put the slot back untouched and
    re-prefill. A reuse TAKES the slot -- the generation mutates those cache
    objects, and the end-of-generation store re-registers them.
    """
    cache_config = cache_config or {}
    model_id = prompt_cache.model_key[0]

    # Config-level gate, kept exactly as the radix had it: quantized /
    # rotating / bounded-KV CONFIGS never enter the reuse path at all.
    # Extension-only reuse would in fact be sound for them; widening is a
    # separate decision with its own verification, not a ride-along.
    # (Historical name: the gate outlived the radix it was written for.)
    prompt_cache._radix_eligible = (
        cache_config.get("cache_type", "standard") == "standard"
        and not cache_config.get("max_kv_size")
    )

    manager = get_global_cache_manager()
    slot = manager._get_slot(model_id) if prompt_cache._radix_eligible else None

    if slot is not None and new_tokens:
        matched_len = _common_prefix_len(slot.tokens, new_tokens)
        # mlx-lm needs at least one prompt token to process, and the cache
        # boundary must sit BEFORE the token being processed -- an exact
        # full-length repeat steps back one token (making it a 1-token trim).
        if matched_len == len(new_tokens):
            matched_len -= 1

        if matched_len > 0:
            tail = len(slot.tokens) - matched_len
            # Reconstruct FRESH cache objects from the immutable snapshots
            # (never hand out shared objects -- see the module docstring),
            # then close the divergence with mlx-lm's own trim. A cache that
            # refuses the trim (hybrid ArraysCache, rotated window) is a
            # re-prefill, never a slice; the slot stays for future extensions.
            restored = _restore_layers(slot, model, cache_config)
            if tail == 0 or (can_trim_prompt_cache(restored)
                             and trim_prompt_cache(restored, tail) == tail):
                prompt_cache.cache = restored
                prompt_cache._radix_matched_len = matched_len
                prompt_cache.tokens = new_tokens
                tokens_to_process = new_tokens[matched_len:]
                logging.info(
                    f"Prompt cache hit: reusing {matched_len}/{len(new_tokens)} tokens "
                    f"({'extension' if tail == 0 else f'trimmed {tail}'}), "
                    f"processing {len(tokens_to_process)} new"
                )
                return tokens_to_process, prompt_cache

    # Re-prefill everything with a fresh cache
    prompt_cache.cache = make_cache(model, cache_config)
    prompt_cache.tokens = new_tokens
    prompt_cache._radix_matched_len = 0
    logging.info(f"Prompt cache miss: processing all {len(new_tokens)} tokens (model={model_id})")
    return new_tokens, prompt_cache


def store_generation_cache(
    prompt_cache: PromptCache,
    full_tokens: List[int],
    generation_cache: List[Any],
) -> None:
    """Snapshot this generation's cache state as the model's slot.

    Called from strategy finally blocks on success (a failed generation
    stores nothing and the previous slot stands -- snapshots are immutable,
    so it is still valid). The state is materialized here, on the
    generating thread, per the thread-affinity invariant.
    """
    # Mirror the lookup-side gate: never publish from non-standard caches.
    if not getattr(prompt_cache, "_radix_eligible", True):
        return

    model_id = prompt_cache.model_key[0]
    if not model_id or not generation_cache:
        return

    # The slot's token list must describe what the cache ACTUALLY contains.
    # The final sampled token is never fed back through the model (it is
    # sampled from the previous forward), so the cache holds one fewer
    # token than full_tokens -- read the truth off the KV offset rather
    # than assuming the arithmetic. Registering full_tokens against a
    # one-short cache would leave every later extension blind to the last
    # token of this reply (position-content misalignment, the silent-wrong
    # class this module exists to avoid).
    offsets = [c.offset for c in generation_cache
               if isinstance(getattr(c, 'offset', None), int)]
    processed = min(max(offsets), len(full_tokens)) if offsets else len(full_tokens)

    layers = _snapshot_layers(generation_cache)
    nbytes = _materialize(layers)  # eval the EXACT arrays being stored
    manager = get_global_cache_manager()
    manager._put_slot(model_id, full_tokens[:processed], layers, nbytes)
    prompt_cache.tokens = full_tokens

    logging.debug(
        f"Stored prompt-cache slot: {len(full_tokens)} tokens "
        f"(matched={prompt_cache._radix_matched_len}, new={len(full_tokens) - prompt_cache._radix_matched_len})"
    )

    manager.enforce_byte_budget()


# Global cache manager instance
_global_cache_manager = PromptCacheManager()


def get_global_cache_manager() -> PromptCacheManager:
    """Get the global prompt cache manager instance."""
    return _global_cache_manager
