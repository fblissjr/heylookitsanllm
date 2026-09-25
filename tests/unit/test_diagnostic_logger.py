# tests/unit/test_diagnostic_logger.py
"""diag_event now delegates to the observability spine (redesign Phase 2).

One writer, one schema, one rotation for logs/events.jsonl. diag fields are
flattened onto the record (queryable top-level keys), request_id + the diag
`level` (severity) are carried as fields, and severity maps to the spine's
verbosity gate.
"""

import orjson
import pytest

from heylook_llm import diagnostic_logger as dl
from heylook_llm import observability as obs


@pytest.fixture
def events(tmp_path):
    """Spine at debug level -> everything records; temp dir per test."""
    obs.configure(level="debug", log_dir=tmp_path)
    return tmp_path


def _read_events(tmp_path):
    p = tmp_path / "events.jsonl"
    if not p.exists():
        return []
    return [orjson.loads(line) for line in p.read_bytes().splitlines() if line]


class TestDiagEventDelegates:
    # One diag_event -> one record on the spine. Each row: the call, the keys
    # the record must carry, and keys it must not have. Every row also checks
    # the spine's own ts (float) and iso stamp.
    @pytest.mark.parametrize(
        "event, kwargs, expected, absent",
        [
            # the spine's type/source plus the diag severity and request_id
            ("request_start", dict(request_id="req-1", level="info", model="m"),
             {"type": "request_start", "source": "backend", "level": "info",
              "request_id": "req-1"},
             ()),
            # diag fields are flattened onto the record (queryable top-level
            # keys), never nested under "data"
            ("request_error", dict(request_id="r", level="error", model="m",
                                   stage="streaming"),
             {"model": "m", "stage": "streaming"},
             ("data",)),
        ],
        ids=["writes_to_events_stream_with_ts_iso", "fields_are_flattened_not_nested"],
    )
    def test_record_shape(self, events, event, kwargs, expected, absent):
        dl.diag_event(event, **kwargs)
        (rec,) = _read_events(events)
        for key, value in expected.items():
            assert rec[key] == value, key
        for key in absent:
            assert key not in rec
        assert isinstance(rec["ts"], float)
        assert "T" in rec["iso"]

    # Severity maps to the spine's verbosity gate. Each row: spine level, the
    # (type, severity) events emitted, the types that must be recorded. An
    # empty expectation means the file is never even created.
    @pytest.mark.parametrize(
        "spine_level, emitted, recorded",
        [
            # at minimal: errors/warnings record; info (needs standard) is dropped
            ("minimal", [("err", "error"), ("info", "info")], ["err"]),
            # off suppresses everything, errors included
            ("off", [("err", "error")], []),
        ],
        ids=["severity_maps_to_verbosity_gate", "off_suppresses_everything"],
    )
    def test_level_gate(self, tmp_path, spine_level, emitted, recorded):
        obs.configure(level=spine_level, log_dir=tmp_path)
        for i, (event, severity) in enumerate(emitted, start=1):
            dl.diag_event(event, level=severity, a=i)
        if recorded:
            assert [r["type"] for r in _read_events(tmp_path)] == recorded
        else:
            assert not (tmp_path / "events.jsonl").exists()
