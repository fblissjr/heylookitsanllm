# src/heylook_llm/diagnostic_logger.py
"""
Structured diagnostic logger -- append-only JSONL file with size cap.

Events are written to logs/events.jsonl (configurable via HEYLOOK_DIAG_LOG env).
Uses orjson for fast serialization. Simple rotation: truncate first half when >50MB.
"""

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from heylook_llm.optimizations import fast_json as json

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

_log_path: Optional[Path] = None
_lock = Lock()
_logger = logging.getLogger(__name__)


def _get_log_path() -> Path:
    global _log_path
    if _log_path is None:
        env_path = os.environ.get("HEYLOOK_DIAG_LOG")
        if env_path:
            _log_path = Path(env_path)
        else:
            _log_path = Path("logs/events.jsonl")
        _log_path.parent.mkdir(parents=True, exist_ok=True)
    return _log_path


def _rotate_if_needed(path: Path) -> None:
    """Truncate first half of the file when it exceeds the size cap."""
    try:
        if path.exists() and path.stat().st_size > _MAX_FILE_BYTES:
            data = path.read_bytes()
            # Find midpoint newline
            mid = len(data) // 2
            newline_pos = data.index(b"\n", mid)
            path.write_bytes(data[newline_pos + 1 :])
            _logger.debug("Rotated diagnostic log (truncated first half)")
    except Exception:
        _logger.debug("Diagnostic log rotation failed", exc_info=True)


def diag_event(
    event_type: str,
    request_id: Optional[str] = None,
    level: str = "info",
    **data,
) -> None:
    """
    Write a structured diagnostic event to the JSONL log.

    Args:
        event_type: e.g. 'request_start', 'model_load', 'model_evict'
        request_id: correlation ID from X-Request-ID header
        level: 'error', 'warn', 'info', 'debug'
        **data: arbitrary key-value pairs to include
    """
    event = {
        "ts": time.time(),
        "level": level,
        "type": event_type,
    }
    if request_id:
        event["request_id"] = request_id
    if data:
        event["data"] = data

    try:
        line = json.dumps(event) + "\n"
        path = _get_log_path()

        with _lock:
            _rotate_if_needed(path)
            with open(path, "a") as f:
                f.write(line)
    except Exception:
        _logger.debug("Failed to write diagnostic event", exc_info=True)
