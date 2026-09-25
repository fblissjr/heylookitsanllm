# src/heylook_llm/providers/common/vision_feature_cache.py
"""
LRU cache for VLM vision encoder outputs.

Caches projected image features keyed by the CONTENT of a request's images, so
multi-turn conversations discussing the same image skip the vision tower.

The key is ``image_content_key(images)``: a hash of each loaded image's pixels,
joined in order. Content, not the URL: a web link keeps its URL when the file
behind it changes, and a URL key then serves the old picture's features. The
key still covers the request's whole image list, so a turn that adds an image
misses (a per-image key is open in plan W10).

Follows the same pattern as mlx-vlm's VisionFeatureCache (vision_cache.py),
and answers the ``get(key)`` / ``put(key, features)`` calls a model makes when
handed the cache as mlx-vlm's ``vision_cache`` kwarg:
- LRU eviction when max_entries exceeded
- Cleared on model unload

Thread-safe via lock for concurrent request handling.
"""

import hashlib
import logging
import threading
from collections import OrderedDict

import mlx.core as mx


def image_content_key(images) -> str:
    """The cache key for a request's loaded images (PIL), in order.

    Hashes each image's mode, size and pixels, so the same picture keys the
    same whether it arrived as a data URL or a web link, and a changed file
    behind an unchanged link keys differently.
    """
    parts = []
    for im in images:
        h = hashlib.sha256(f"{im.mode}:{im.size}:".encode())
        h.update(im.tobytes())
        parts.append(h.hexdigest()[:32])
    return "|".join(parts)


class VisionFeatureCache:
    """LRU cache for vision encoder outputs.

    Stores projected image features (mx.array after vision_tower + projector)
    keyed by ``image_content_key``. Thread-safe via lock.

    Evicts on BOTH caps:
    - count cap (``max_entries``) protects against unbounded growth with tiny features
    - byte cap (``max_bytes``) protects against a few large-image entries consuming
      multiple GB (the documented leak vector the entry-count cap alone left open)

    Args:
        max_entries: Maximum number of cached image features. Default 20.
        max_bytes: Hard byte ceiling across all cached entries. Default 8 GB.
            Reading ``feature.nbytes`` at insert time is safe because the caller
            materializes the array first (heylook's encode_image branch, or the
            model's own ``mx.eval`` before its ``put``).
    """

    def __init__(self, max_entries: int = 20, max_bytes: int = 8_000_000_000):
        self._cache: OrderedDict[str, mx.array] = OrderedDict()
        self._entry_bytes: dict[str, int] = {}
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._n_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> mx.array | None:
        """Look up cached features. Returns None on a miss or an empty key."""
        if not key:
            return None

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                logging.debug(f"Vision feature cache HIT (total hits={self._hits})")
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, features: mx.array) -> None:
        """Store computed features, evicting LRU entries until both caps hold."""
        if not key:
            return

        try:
            new_bytes = int(features.nbytes)
        except Exception:
            new_bytes = 0

        with self._lock:
            if key in self._cache:
                # Replace: subtract the old entry's bytes, then overwrite.
                self._n_bytes -= self._entry_bytes.get(key, 0)
                self._cache.move_to_end(key)
            self._cache[key] = features
            self._entry_bytes[key] = new_bytes
            self._n_bytes += new_bytes
            self._evict_until_within_caps_locked()

    def _evict_until_within_caps_locked(self) -> None:
        """Evict oldest entries until both count and byte caps hold.

        Assumes ``self._lock`` is already held.
        """
        while self._cache and (
            len(self._cache) > self._max_entries or self._n_bytes > self._max_bytes
        ):
            evicted_key, _ = self._cache.popitem(last=False)
            evicted_bytes = self._entry_bytes.pop(evicted_key, 0)
            self._n_bytes -= evicted_bytes
            logging.debug(
                f"Vision feature cache: evicted LRU entry ({evicted_bytes} bytes)"
            )

    def clear(self) -> None:
        """Clear all cached features."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._entry_bytes.clear()
            self._n_bytes = 0
            if count > 0:
                logging.debug(f"Vision feature cache: cleared {count} entries")

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "bytes": self._n_bytes,
                "max_bytes": self._max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
