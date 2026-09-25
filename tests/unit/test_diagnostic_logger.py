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
    def test_writes_to_events_stream_with_ts_iso(self, events):
        dl.diag_event("request_start", request_id="req-1", level="info", model="m")
        (rec,) = _read_events(events)
        assert rec["type"] == "request_start"
        assert rec["source"] == "backend"
        assert rec["level"] == "info"
        assert rec["request_id"] == "req-1"
        assert isinstance(rec["ts"], float)
        assert "T" in rec["iso"]

    def test_fields_are_flattened_not_nested(self, events):
        dl.diag_event("request_error", request_id="r", level="error",
                      model="m", stage="streaming")
        (rec,) = _read_events(events)
        assert rec["model"] == "m"
        assert rec["stage"] == "streaming"
        assert "data" not in rec  # flattened top-level, not nested under "data"

    def test_severity_maps_to_verbosity_gate(self, tmp_path):
        # at minimal: errors/warnings record; info/debug are dropped
        obs.configure(level="minimal", log_dir=tmp_path)
        dl.diag_event("err", level="error", a=1)   # kept
        dl.diag_event("info", level="info", a=2)   # dropped (needs standard)
        recs = _read_events(tmp_path)
        assert [r["type"] for r in recs] == ["err"]

    def test_off_suppresses_everything(self, tmp_path):
        obs.configure(level="off", log_dir=tmp_path)
        dl.diag_event("err", level="error", a=1)
        assert not (tmp_path / "events.jsonl").exists()
