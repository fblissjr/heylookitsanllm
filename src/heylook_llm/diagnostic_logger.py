# src/heylook_llm/diagnostic_logger.py
"""
Diagnostic events -- call-site API that delegates to the observability spine.

``diag_event`` is retained as the API used across api.py/router.py; as of the
observability redesign (Phase 2) it delegates to ``observability.record_event``
(events tier), so there is ONE writer + ONE schema + ONE rotation for
``logs/events.jsonl``. ``exception_detail`` is unchanged (a pure helper).

Schema note: diag fields are now flattened onto the record (queryable top-level
keys) instead of nested under ``data``; ``request_id`` and the diag ``level``
(severity) are carried as fields. The diag ``level`` (error/warn/info/debug) is a
SEVERITY; it is mapped to the spine's verbosity gate below (errors/warnings
surface at ``minimal``).
"""

import traceback

from heylook_llm import observability

# diag severity -> spine verbosity gate. An error/warning is worth recording as
# soon as observability is on at all (minimal); info needs standard; debug needs
# debug. `off` still suppresses everything (off = zero telemetry, by design).
_SEVERITY_MIN_LEVEL = {
    "error": "minimal",
    "warn": "minimal",
    "info": "standard",
    "debug": "debug",
}


def diag_event(
    event_type: str,
    request_id: str | None = None,
    level: str = "info",
    **data,
) -> None:
    """Record a diagnostic event via the observability spine (events tier).

    Args:
        event_type: e.g. 'request_start', 'model_load', 'request_error'
        request_id: correlation ID from X-Request-ID header (carried as a field)
        level: SEVERITY -- 'error', 'warn', 'info', 'debug' -- carried as a field
               and mapped to the spine verbosity gate (see _SEVERITY_MIN_LEVEL).
        **data: key-value fields, flattened onto the record (queryable top-level).

    Best-effort: record_event never raises.
    """
    fields = dict(data)
    if request_id is not None:
        fields["request_id"] = request_id
    fields["level"] = level  # severity carried as a field
    observability.record_event(
        event_type,
        tier="events",
        min_level=_SEVERITY_MIN_LEVEL.get(level, "standard"),
        source="backend",
        fields=fields,
    )


def exception_detail(exc: BaseException) -> dict:
    """Render an exception into a JSON-safe dict for a diagnostic event.

    Captures the exception class, its message, and -- when the error was raised
    from another (``raise X from Y``, or an implicit ``__context__``) -- a
    bounded ``chain`` of the underlying causes. Each chain link is rendered with
    ``traceback.format_exception_only``, which emits only the type and message,
    NEVER frame locals. That keeps prompt/response text (which can live in a
    frame's locals) out of the log, so this is safe to write even though
    events.jsonl is not bound by the telemetry-stream content invariant.

    Shape: ``{"error_type": ..., "error": ..., "chain": [...]}``. ``chain`` is
    omitted unless there is a distinct underlying cause.
    """
    detail: dict = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    try:
        chain: list[str] = []
        # Walk __cause__ (explicit `from`) then fall back to __context__
        # (implicit "during handling of the above"). Cap the depth so a
        # pathological cycle or deep wrap can't bloat a single log line.
        cursor: BaseException | None = exc.__cause__ or exc.__context__
        seen: set[int] = {id(exc)}
        while cursor is not None and id(cursor) not in seen and len(chain) < 5:
            seen.add(id(cursor))
            for ln in traceback.format_exception_only(type(cursor), cursor):
                ln = ln.rstrip("\n")
                if ln.strip():
                    chain.append(ln)
            cursor = cursor.__cause__ or cursor.__context__
        if chain:
            detail["chain"] = chain
    except Exception:
        pass
    return detail
