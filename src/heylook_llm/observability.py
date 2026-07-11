# src/heylook_llm/observability.py
"""Observability spine: one ingestion path -> level-gated, best-effort JSONL.

`record_event` is the single write path for telemetry + diagnostics (it replaces
the ad-hoc `diagnostic_logger.diag_event` and `memory.py` stream writers as they
migrate in). It appends one JSON line to the right stream and NEVER raises --
observability must never break inference (the `safe_mm_call` discipline).

Two streams, by tier:
- `metrics.jsonl` -- content-free, aggregatable time-series (safe to share).
- `events.jsonl`  -- correlated discrete records; may carry BOUNDED error text
  (type + message + cause chain), never prompts/responses/token-IDs.

Verbosity: `off < minimal < standard < debug`. Each event declares the level at
which it starts being recorded (`min_level`); a configured level records that
event iff it is >= the event's min_level (so `off` records nothing).

The configured level + log dir are cached in-process (set at startup and on a
settings change) so the hot path never touches the DB. Content-safety is the
caller's responsibility per event-type -- metrics-tier callers pass numeric /
metadata fields only (documented invariant; enforced by discipline, not schema).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

logger = logging.getLogger(__name__)

# Verbosity ordering. `off` = 0 sits below every event's min_level (>= 1), so it
# gates out everything without a special-case.
_LEVEL_ORDER: dict[str, int] = {"off": 0, "minimal": 1, "standard": 2, "debug": 3}

_STREAMS: dict[str, str] = {"metrics": "metrics.jsonl", "events": "events.jsonl"}

# In-process cached config. Defaults are safe (minimal, ./logs) so a call before
# configure() still behaves sanely rather than crashing.
_level: str = "minimal"
_log_dir: Path = Path("logs")
_retention_days: int = 30

# Rotation: roll the active stream past this size, hourly sweep on the tick.
_MAX_STREAM_BYTES = 50 * 1024 * 1024  # 50 MB
_ROTATE_INTERVAL_S = 3600.0
_last_rotate_ts: float = 0.0


def configure(level: str, log_dir: Path | str, retention_days: int = 30) -> None:
    """Set the active verbosity + log directory + retention (startup / on change)."""
    global _level, _log_dir, _retention_days
    _level = level if level in _LEVEL_ORDER else "minimal"
    _log_dir = Path(log_dir)
    _retention_days = int(retention_days)


def current_level() -> str:
    return _level


def _enabled(min_level: str) -> bool:
    return _LEVEL_ORDER.get(_level, 0) >= _LEVEL_ORDER.get(min_level, 99)


def record_event(
    event_type: str,
    *,
    tier: str,
    min_level: str,
    source: str = "backend",
    **fields: Any,
) -> None:
    """Append one telemetry/diagnostic line. Best-effort -- never raises.

    Args:
        event_type: e.g. "request_complete", "request_error", "model_load".
        tier: "metrics" (content-free) or "events" (may carry bounded error text).
        min_level: lowest configured level at which this event is recorded.
        source: "backend" (default) or "frontend-v3".
        **fields: numeric / metadata payload (metrics tier: no free text).
    """
    try:
        if not _enabled(min_level):
            return
        stream = _STREAMS.get(tier)
        if stream is None:
            logger.debug("record_event: unknown tier %r", tier)
            return
        ts = time.time()
        record = {
            "ts": ts,
            # local-time ISO 8601 w/ offset -- readable without converting epoch;
            # `ts` stays authoritative for sort/aggregation/correlation.
            "iso": datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds"),
            "type": event_type,
            "source": source,
            **fields,
        }
        path = _log_dir / stream
        path.parent.mkdir(parents=True, exist_ok=True)
        line = orjson.dumps(record) + b"\n"
        with open(path, "ab") as f:
            f.write(line)
    except Exception:
        logger.debug("record_event failed", exc_info=True)


def rotate_streams(
    retention_days: int,
    max_bytes: int = _MAX_STREAM_BYTES,
    now: float | None = None,
) -> None:
    """Size cap -> roll the active stream to a timestamped archive; age cap ->
    delete archives older than ``retention_days`` (0 keeps forever). Best-effort;
    the active ``<stream>.jsonl`` is never deleted, only rolled. Never raises."""
    now = time.time() if now is None else now
    cutoff = now - retention_days * 86400
    for stream in _STREAMS.values():
        base = _log_dir / stream                      # e.g. logs/events.jsonl
        stem = base.stem                              # "events"
        try:
            if base.exists() and base.stat().st_size > max_bytes:
                # roll: events.jsonl -> events.<ts>.jsonl (fresh file on next write)
                base.rename(_log_dir / f"{stem}.{int(now)}.jsonl")
        except Exception:
            logger.debug("rotate: roll failed for %s", base, exc_info=True)
        if retention_days > 0:
            for archive in _log_dir.glob(f"{stem}.*.jsonl"):
                try:
                    if archive.stat().st_mtime < cutoff:
                        archive.unlink()
                except Exception:
                    logger.debug("rotate: unlink failed for %s", archive, exc_info=True)


def maybe_rotate(now: float | None = None) -> bool:
    """Throttled rotation for the maintenance tick (~hourly). Uses the cached
    retention. Returns True if a sweep ran. Never raises."""
    global _last_rotate_ts
    try:
        now = time.time() if now is None else now
        if now - _last_rotate_ts < _ROTATE_INTERVAL_S:
            return False
        _last_rotate_ts = now
        rotate_streams(_retention_days, now=now)
        return True
    except Exception:
        logger.debug("maybe_rotate failed", exc_info=True)
        return False
