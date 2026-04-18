# src/heylook_llm/providers/common/vision_feature_cache.py
"""
LRU cache for VLM vision encoder outputs.

Caches projected image features keyed by image URL/path string so that
multi-turn conversations discussing the same image skip the expensive
vision tower forward pass (typically 200-500ms per image).

Two keying strategies:
- Primary: image URL/path string (fast, covers the common case)
- Fallback: SHA-256 of pixel_values bytes (handles base64/PIL images
  that lack a stable URL). Requires mx.eval() to materialize the array
  (~2ms for a 1024x1024 image, negligible vs. 200ms+ vision encoding).

Follows the same pattern as mlx-vlm's VisionFeatureCache (vision_cache.py):
- LRU eviction when max_entries exceeded
- Cleared on model unload

Thread-safe via lock for concurrent request handling.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any

import mlx.core as mx


def _hash_pixel_values(pixel_values: mx.array) -> str:
    """Compute SHA-256 hash of pixel values for content-based keying.

    Requires the array to be evaluated (materialized). Returns a hex
    digest string prefixed with 'px:' to distinguish from URL keys.

    Note: mx.eval is MLX's lazy graph materializer, not Python's eval().
    """
    # Ensure array is evaluated before reading bytes
    mx.async_eval(pixel_values)
    data = bytes(memoryview(pixel_values))
    return f"px:{hashlib.sha256(data).hexdigest()[:16]}"


class VisionFeatureCache:
    """LRU cache for vision encoder outputs.

    Stores projected image features (mx.array after vision_tower + projector)
    keyed by image source string or pixel content hash. Thread-safe via lock.

    Evicts on BOTH caps:
    - count cap (``max_entries``) protects against unbounded growth with tiny features
    - byte cap (``max_bytes``) protects against a few large-image entries consuming
      multiple GB (the documented leak vector the entry-count cap alone left open)

    Args:
        max_entries: Maximum number of cached image features. Default 20.
        max_bytes: Hard byte ceiling across all cached entries. Default 8 GB.
            Reading ``feature.nbytes`` at insert time is safe because the caller
            materializes the array first (mlx_provider.py:421, mlx-vlm generate.py:662).
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

    @staticmethod
    def _make_key(image_source: Any, pixel_values: mx.array | None = None) -> str:
        """Derive a cache key from an image source.

        For str: use directly (URL or file path).
        For list of str: join with | separator.
        Fallback: if image_source yields no key but pixel_values is
        provided, hash the pixel data for content-based keying.
        """
        if isinstance(image_source, str):
            return image_source
        if isinstance(image_source, list):
            parts = []
            for item in image_source:
                if isinstance(item, str):
                    parts.append(item)
                else:
                    # Non-string in list -- fall through to pixel hash
                    break
            else:
                # All items were strings
                return "|".join(parts)

        # Fallback: hash pixel values if available
        if pixel_values is not None:
            try:
                return _hash_pixel_values(pixel_values)
            except Exception:
                pass

        return ""

    def get(self, image_source: Any, pixel_values: mx.array | None = None) -> mx.array | None:
        """Look up cached features. Returns None on miss or uncacheable source."""
        key = self._make_key(image_source, pixel_values)
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

    def put(self, image_source: Any, features: mx.array,
            pixel_values: mx.array | None = None) -> None:
        """Store computed features, evicting LRU entries until both caps hold."""
        key = self._make_key(image_source, pixel_values)
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
