# src/heylook_llm/diagnostic_logger.py
"""
Diagnostic events -- call-site API that delegates to the observability spine.

``diag_event`` is retained as the API used across api.py/router.py; as of the
observability redesign (Phase 2) it delegates to ``observability.record_event``
(events tier), so there is ONE writer + ONE schema + ONE rotation for
``logs/events.jsonl``.

Schema note: diag fields are now flattened onto the record (queryable top-level
keys) instead of nested under ``data``; ``request_id`` and the diag ``level``
(severity) are carried as fields. The diag ``level`` (error/warn/info/debug) is a
SEVERITY; it is mapped to the spine's verbosity gate below (errors/warnings
surface at ``minimal``).
"""

from heylook_llm import observability

# diag severity -> spine verbosity gate. The mapping itself lives on the spine
# (observability.SEVERITY_MIN_LEVEL) because telemetry_api gates client-submitted
# events on the same table; it was duplicated here until 2026-09-06.


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
               and mapped to the spine verbosity gate
               (observability.SEVERITY_MIN_LEVEL).
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
        min_level=observability.SEVERITY_MIN_LEVEL.get(level, "standard"),
        source="backend",
        fields=fields,
    )
