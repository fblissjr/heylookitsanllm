# src/heylook_llm/optimizations/status.py
"""Optimization status reporting.

Only orjson is left to report. The image entry went with fast_image.py
(2026-08-26): its ImageCache had exactly one construction site, in the
multipart endpoint, behind a `hasattr(app.state, 'image_cache')` that nothing
ever set -- so the cache was never built, and `cachetools_available` was a flag
describing a code path that could not run. The xxhash/turbojpeg flags beside it
had already gone the same way on 2026-08-18.
"""

import logging

from . import fast_json


def log_all_optimization_status():
    """Log optimization status at server startup."""
    fast_json.log_status()


def get_optimization_summary():
    """Summary of optimization statuses (mirrored by /v1/capabilities)."""
    return {"json": fast_json.get_status()}
