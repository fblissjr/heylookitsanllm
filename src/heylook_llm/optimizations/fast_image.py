# src/heylook_llm/optimizations/fast_image.py
"""
Image helpers for the multipart vision path.

Only ImageCache lives here now. The xxhash and PyTurboJPEG imports this module
carried were removed 2026-08-18: nothing ever called them. There was no hash
call and no decode call anywhere in the repo -- the TurboJPEG decoder was
constructed at import and never touched -- so both libraries (plus the system
libjpeg-turbo PyTurboJPEG needs) were being paid for a status flag. If the
multipart JPEG decode is ever really wired to TurboJPEG, add the dependency back
together with the call site, never ahead of it.
"""

import logging
from typing import Optional
from PIL import Image

try:
    from cachetools import TTLCache
    HAS_CACHETOOLS = True
except ImportError:
    from collections import OrderedDict
    HAS_CACHETOOLS = False


def get_status():
    """Get the status of image optimizations."""
    return {
        "cachetools_available": HAS_CACHETOOLS,
    }


def log_status():
    """Log the image optimization status."""
    if HAS_CACHETOOLS:
        logging.info("Image optimizations available: TTL cache")
    else:
        logging.info("No image optimization libraries available - using standard implementations")


class ImageCache:
    """Thread-safe image cache with TTL support."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        """Initialize cache with max size and TTL in seconds."""
        self._max_size = max_size
        self._ttl = ttl
        if HAS_CACHETOOLS:
            self._cache = TTLCache(maxsize=max_size, ttl=ttl)
        else:
            self._cache = OrderedDict()
        
    def get(self, key: str) -> Optional[Image.Image]:
        """Get image from cache by key."""
        if key in self._cache:
            return self._cache[key].copy()
        return None
    
    def set(self, key: str, image: Image.Image):
        """Store image in cache."""
        self._cache[key] = image.copy()
        
        # Manual eviction for basic cache
        if not HAS_CACHETOOLS and len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
    
    def clear(self):
        """Clear all cached images."""
        self._cache.clear()


