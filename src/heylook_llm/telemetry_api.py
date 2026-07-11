# src/heylook_llm/telemetry_api.py
"""Frontend telemetry ingestion: POST /v1/telemetry/events.

v3's js/telemetry.js batches client events (JS errors, fetch failures, stream
stalls, UX events) and POSTs them here; each is appended to the events stream
via the observability spine with source='frontend-v3', so backend + frontend
telemetry share one queryable surface (correlate by request_id).

Content-safety: clients must send metadata only -- never prompts/responses. The
endpoint bounds batch size and per-field string length as a backstop, but the
primary guarantee is the client's discipline (same as the backend content
invariant). Level-gated like everything else: at `off` nothing is recorded.
"""

import logging

from fastapi import APIRouter, Body

from heylook_llm import observability

logger = logging.getLogger(__name__)

telemetry_router = APIRouter(prefix="/v1/telemetry", tags=["Telemetry"])

_MAX_BATCH = 100
_MAX_FIELD_CHARS = 2000
# client-event severity -> spine verbosity gate (client_error surfaces at minimal)
_SEVERITY_MIN_LEVEL = {"error": "minimal", "warn": "minimal", "info": "standard", "debug": "debug"}
_RESERVED = {"type", "level"}


def _sanitize_fields(ev: dict) -> dict:
    """Drop reserved keys; bound accidental large string fields (backstop)."""
    out = {}
    for k, v in ev.items():
        if k in _RESERVED:
            continue
        if isinstance(v, str) and len(v) > _MAX_FIELD_CHARS:
            v = v[:_MAX_FIELD_CHARS]
        out[k] = v
    return out


@telemetry_router.post(
    "/events",
    summary="Ingest Client Telemetry",
    description="Append a batch of frontend client events to the observability events "
                "stream (source=frontend-v3). Body: {events: [{type, level?, ...}]}. "
                "Metadata only -- never prompts/responses. Best-effort; returns the count accepted.",
)
async def ingest_events(payload: dict = Body(...)):
    events = payload.get("events")
    if not isinstance(events, list):
        return {"accepted": 0, "error": "body must be {events: [...]}"}
    accepted = 0
    for ev in events[:_MAX_BATCH]:
        if not isinstance(ev, dict) or "type" not in ev:
            continue
        level = ev.get("level", "info")
        observability.record_event(
            str(ev["type"]),
            tier="events",
            min_level=_SEVERITY_MIN_LEVEL.get(level, "standard"),
            source="frontend-v3",
            level=level,
            **_sanitize_fields(ev),
        )
        accepted += 1
    return {"accepted": accepted}
