# tests/unit/test_diagnostic_logger.py
"""diag_event now delegates to the observability spine (redesign Phase 2).

One writer, one schema, one rotation for logs/events.jsonl. diag fields are
flattened onto the record (queryable top-level keys), request_id + the diag
`level` (severity) are carried as fields, and severity maps to the spine's
verbosity gate. exception_detail is a pure helper (type + message + bounded
cause chain, never frame locals).
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


class TestExceptionDetail:
    def test_captures_type_and_message(self):
        try:
            raise KeyError("missing_field")
        except KeyError as e:
            detail = dl.exception_detail(e)
        assert detail["error_type"] == "KeyError"
        assert "missing_field" in detail["error"]
        assert "chain" not in detail

    def test_surfaces_cause_chain(self):
        try:
            try:
                raise ValueError("kv cache overflow at token 4096")
            except ValueError as root:
                raise RuntimeError("generation failed") from root
        except RuntimeError as e:
            detail = dl.exception_detail(e)
        assert detail["error_type"] == "RuntimeError"
        assert detail["error"] == "generation failed"
        assert any("kv cache overflow at token 4096" in link for link in detail["chain"])

    def test_never_contains_frame_locals(self):
        secret_prompt = "SENSITIVE_PROMPT_TEXT_should_never_appear"  # noqa: F841
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            detail = dl.exception_detail(e)
        assert "SENSITIVE_PROMPT_TEXT_should_never_appear" not in orjson.dumps(detail).decode()

    def test_bounds_chain_depth(self):
        exc = None
        for i in range(20):
            try:
                raise RuntimeError(f"layer-{i}")
            except RuntimeError as e:
                if exc is not None:
                    try:
                        raise e from exc
                    except RuntimeError as e2:
                        exc = e2
                else:
                    exc = e
        detail = dl.exception_detail(exc)
        assert len(detail.get("chain", [])) <= 5


class TestDiagPlusExceptionDetail:
    def test_error_event_carries_exception_detail_flattened(self, events):
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            dl.diag_event("request_error", request_id="r", level="error",
                          model="m", stage="provider_get", **dl.exception_detail(e))
        (rec,) = _read_events(events)
        assert rec["error_type"] == "RuntimeError"
        assert rec["stage"] == "provider_get"
        assert rec["model"] == "m"
