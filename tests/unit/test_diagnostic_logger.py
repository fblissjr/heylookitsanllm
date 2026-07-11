# tests/unit/test_diagnostic_logger.py
"""Unit tests for the structured diagnostic logger (logs/events.jsonl).

Covers the two properties that make error records actually useful:
- every event carries both `ts` (epoch, for sort/math) and `iso` (human-
  readable local time with offset);
- `exception_detail` surfaces the exception type and, for wrapped errors, the
  underlying cause chain -- type+message only, never frame locals.
"""

import importlib
from datetime import datetime

import orjson
import pytest


@pytest.fixture
def diag(tmp_path, monkeypatch):
    """Fresh diagnostic_logger module pointed at an isolated log file.

    The module caches its log path in a process-global, so reload it under the
    env override to get a clean, per-test file.
    """
    log_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("HEYLOOK_DIAG_LOG", str(log_file))
    import heylook_llm.diagnostic_logger as dl
    dl = importlib.reload(dl)
    return dl, log_file


def _read(log_file):
    return [orjson.loads(line) for line in log_file.read_bytes().splitlines() if line]


def test_every_event_has_ts_and_iso(diag):
    dl, log_file = diag
    dl.diag_event("request_start", request_id="req-1", model="m", stream=True)

    (event,) = _read(log_file)
    assert isinstance(event["ts"], float)
    # iso must parse and round-trip to (approximately) the same instant as ts.
    parsed = datetime.fromisoformat(event["iso"])
    assert parsed.tzinfo is not None  # carries an offset, not naive
    assert abs(parsed.timestamp() - event["ts"]) < 1.0


def test_event_shape_preserves_type_level_and_data(diag):
    dl, log_file = diag
    dl.diag_event("request_error", request_id="req-1", level="error", model="m", stage="streaming")

    (event,) = _read(log_file)
    assert event["type"] == "request_error"
    assert event["level"] == "error"
    assert event["request_id"] == "req-1"
    assert event["data"] == {"model": "m", "stage": "streaming"}


def test_exception_detail_captures_type_and_message(diag):
    dl, _ = diag
    try:
        raise KeyError("missing_field")
    except KeyError as e:
        detail = dl.exception_detail(e)

    assert detail["error_type"] == "KeyError"
    assert "missing_field" in detail["error"]
    # No cause -> no chain key.
    assert "chain" not in detail


def test_exception_detail_surfaces_cause_chain(diag):
    dl, _ = diag
    try:
        try:
            raise ValueError("kv cache overflow at token 4096")
        except ValueError as root:
            raise RuntimeError("generation failed") from root
    except RuntimeError as e:
        detail = dl.exception_detail(e)

    assert detail["error_type"] == "RuntimeError"
    assert detail["error"] == "generation failed"
    # The root cause -- otherwise invisible behind the wrapping error -- is
    # recorded in the chain.
    assert any("kv cache overflow at token 4096" in link for link in detail["chain"])


def test_exception_detail_never_contains_frame_locals(diag):
    dl, _ = diag
    # A local holding "prompt content" must not leak into the detail: only
    # type+message are formatted, never frame locals.
    secret_prompt = "SENSITIVE_PROMPT_TEXT_should_never_appear"  # noqa: F841
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        detail = dl.exception_detail(e)

    assert "SENSITIVE_PROMPT_TEXT_should_never_appear" not in orjson.dumps(detail).decode()


def test_exception_detail_bounds_chain_depth(diag):
    dl, _ = diag
    # Build a deep implicit (__context__) chain; detail caps it so one log line
    # can't blow up.
    exc = None
    for i in range(20):
        try:
            raise RuntimeError(f"layer-{i}")
        except RuntimeError as e:
            if exc is not None:
                # chain them implicitly via __context__ by raising inside handling
                try:
                    raise e from exc
                except RuntimeError as e2:
                    exc = e2
            else:
                exc = e
    detail = dl.exception_detail(exc)
    assert len(detail.get("chain", [])) <= 5


def test_diag_event_unknown_kwargs_go_into_data(diag):
    dl, log_file = diag
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        dl.diag_event("request_error", request_id="r", level="error",
                      model="m", stage="provider_get", **dl.exception_detail(e))

    (event,) = _read(log_file)
    data = event["data"]
    assert data["error_type"] == "RuntimeError"
    assert data["stage"] == "provider_get"
    assert data["model"] == "m"
