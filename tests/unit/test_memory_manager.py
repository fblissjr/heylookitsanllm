"""Tests for heylook_llm.memory (S1.2 observability)."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest

from heylook_llm.memory import (
    MemoryManager,
    ModelMetadata,
    _parse_bool_env,
    sampler_summary_from_request,
)


def _read_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"{path} should exist"
    return [orjson.loads(line) for line in path.read_text().splitlines() if line]


def _make_app_config(**overrides):
    defaults = dict(
        baseline_log_interval_seconds=3600,
        request_log_enabled=True,
        model_event_log_enabled=True,
        max_loaded_models=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_router():
    router = MagicMock()
    router.get_loaded_models.return_value = {}
    return router


@pytest.fixture
def mm(tmp_path: Path) -> MemoryManager:
    return MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(),
        log_dir=tmp_path,
    )


def test_parse_bool_env_defaults_when_unset():
    assert _parse_bool_env(None, True) is True
    assert _parse_bool_env(None, False) is False


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False), ("", False)],
)
def test_parse_bool_env_values(value: str, expected: bool):
    assert _parse_bool_env(value, not expected) is expected


def test_env_interval_overrides_app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS", "42")
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(baseline_log_interval_seconds=3600),
        log_dir=tmp_path,
    )
    assert manager.baseline_interval == 42


def test_env_interval_zero_disables_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS", "0")
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(),
        log_dir=tmp_path,
    )
    assert manager.baseline_interval == 0
    assert manager.maybe_log_baseline() is False
    assert not (tmp_path / "memory_baseline.jsonl").exists()


def test_register_model_load_writes_event(mm: MemoryManager, tmp_path: Path):
    metadata = ModelMetadata(
        model_id="test-model",
        path="/some/path",
        weights_bytes=1_024_000,
        architecture="test_arch",
        quantization="4bit",
        param_count=1_500_000_000,
        context_length=8192,
    )
    mm.register_model_load(metadata, load_duration_ms=1234.5)

    events = _read_jsonl(tmp_path / "model_events.jsonl")
    assert len(events) == 1
    record = events[0]
    assert record["event"] == "load"
    assert record["model_id"] == "test-model"
    assert record["weights_bytes"] == 1_024_000
    assert record["quantization"] == "4bit"
    assert record["param_count"] == 1_500_000_000
    assert record["load_duration_ms"] == 1234.5
    assert mm.model_metadata["test-model"] is metadata


def test_register_model_unload_emits_event_and_clears_metadata(mm: MemoryManager, tmp_path: Path):
    mm.model_metadata["to-evict"] = ModelMetadata(
        "to-evict", "/p", 0, "a", "none", 0, 0
    )
    mm.register_model_unload("to-evict", reason="lru_evict")

    events = _read_jsonl(tmp_path / "model_events.jsonl")
    assert len(events) == 1
    assert events[0]["event"] == "unload"
    assert events[0]["reason"] == "lru_evict"
    assert "to-evict" not in mm.model_metadata


def test_model_event_toggle_off_suppresses_writes(tmp_path: Path):
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(model_event_log_enabled=False),
        log_dir=tmp_path,
    )
    manager.register_model_load(
        ModelMetadata("m", "/p", 0, "a", "none", 0, 0),
        load_duration_ms=0.0,
    )
    manager.register_model_unload("m", reason="shutdown")
    assert not (tmp_path / "model_events.jsonl").exists()
    assert "m" not in manager.model_metadata


def test_request_event_toggle_off_suppresses_writes(tmp_path: Path):
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(request_log_enabled=False),
        log_dir=tmp_path,
    )
    manager.log_request_event({"ts": time.time(), "model": "m"})
    assert not (tmp_path / "request_events.jsonl").exists()


def test_request_event_writes(mm: MemoryManager, tmp_path: Path):
    mm.log_request_event({
        "ts": 1000.0,
        "model": "m",
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "sampler_summary": {"temperature": 0.7},
    })
    events = _read_jsonl(tmp_path / "request_events.jsonl")
    assert len(events) == 1
    assert events[0]["model"] == "m"
    assert events[0]["sampler_summary"]["temperature"] == 0.7


def test_snapshot_shape_and_content_invariant(mm: MemoryManager):
    mm.model_metadata["m1"] = ModelMetadata("m1", "/p", 123, "arch", "4bit", 7_000_000_000, 8192)
    snapshot = mm.snapshot()

    # Required numeric keys
    for key in (
        "ts", "rss_bytes", "available_ram_bytes", "cpu_percent",
        "mlx_active_bytes", "mlx_peak_bytes", "mlx_cache_bytes",
        "idle_seconds", "inflight_requests",
    ):
        assert key in snapshot, f"missing key {key}"

    assert snapshot["loaded_models"][0]["model_id"] == "m1"
    assert snapshot["mode"] == "background"

    # Content invariant: no field names that would carry message content.
    # prompt_cache / vision_cache are allowed -- they're counter dicts.
    forbidden_keys = {"prompt", "messages", "message", "completion", "response", "text"}

    def walk_keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in forbidden_keys, f"snapshot leaked content-bearing key: {key}"
                walk_keys(value)
        elif isinstance(node, list):
            for item in node:
                walk_keys(item)

    walk_keys(snapshot)


def test_maybe_log_baseline_respects_interval(tmp_path: Path):
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(baseline_log_interval_seconds=3600),
        log_dir=tmp_path,
    )
    assert manager.maybe_log_baseline() is True
    # Second call within the interval should not write
    assert manager.maybe_log_baseline() is False
    events = _read_jsonl(tmp_path / "memory_baseline.jsonl")
    assert len(events) == 1


def test_mark_request_tracks_inflight(mm: MemoryManager):
    mm.mark_request_start()
    mm.mark_request_start()
    snapshot = mm.snapshot()
    assert snapshot["inflight_requests"] == 2

    mm.mark_request_end()
    snapshot = mm.snapshot()
    assert snapshot["inflight_requests"] == 1

    mm.mark_request_end()
    mm.mark_request_end()  # safe below zero
    snapshot = mm.snapshot()
    assert snapshot["inflight_requests"] == 0


def test_log_startup_info_always_writes(tmp_path: Path):
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(
            request_log_enabled=False,
            model_event_log_enabled=False,
            baseline_log_interval_seconds=0,
        ),
        log_dir=tmp_path,
    )
    manager.log_startup_info()
    events = _read_jsonl(tmp_path / "baseline.jsonl")
    assert len(events) == 1
    assert events[0]["event"] == "startup"
    assert events[0]["baseline_interval_seconds"] == 0


def test_sampler_summary_drops_none_fields_and_messages():
    request = SimpleNamespace(
        temperature=0.7,
        top_p=None,
        top_k=40,
        min_p=None,
        repetition_penalty=None,
        presence_penalty=None,
        max_tokens=512,
        enable_thinking=True,
        seed=None,
        # Content fields that MUST NOT leak into the summary:
        messages=[{"role": "user", "content": "secret prompt"}],
        prompt="another secret",
    )
    summary = sampler_summary_from_request(request)
    assert summary == {
        "temperature": 0.7,
        "top_k": 40,
        "max_tokens": 512,
        "enable_thinking": True,
    }
    assert "messages" not in summary
    assert "prompt" not in summary


def test_append_jsonl_tolerates_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(),
        log_dir=tmp_path,
    )

    def boom(*_args, **_kwargs):
        raise IOError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    # Must not raise
    manager.log_request_event({"ts": 1.0})
