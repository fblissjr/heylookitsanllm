"""Tests for heylook_llm.memory (S1.2 observability)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest

from heylook_llm.memory import (
    MemoryManager,
    ModelMetadata,
    _normalize_path_for_log,
)

# Content-invariant forbidden-keys set. Any key we find with this name anywhere
# in a logged record indicates a content leak. Keep this list broad by design --
# tightening it below the audit is a regression risk.
FORBIDDEN_KEYS = {
    "prompt", "prompts", "messages", "message",
    "completion", "completions", "response", "responses",
    "text", "content", "contents",
    "tools", "tool_calls", "tool_call",
    "input", "inputs", "output", "outputs",
    "body", "raw",
}


def _assert_no_forbidden_keys(node, path=""):
    """Recursively walk a logged record; fail if a forbidden key appears."""
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in FORBIDDEN_KEYS, (
                f"content-bearing key {key!r} leaked at {path or '<root>'}"
            )
            _assert_no_forbidden_keys(value, path=f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _assert_no_forbidden_keys(item, path=f"{path}[{i}]")


def _read_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"{path} should exist"
    return [orjson.loads(line) for line in path.read_text().splitlines() if line]


def _make_app_config(**overrides):
    defaults = dict(max_loaded_models=2)
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


def test_off_level_silences_streams(mm: MemoryManager, tmp_path: Path):
    """observability_level=off is the master kill switch -- it silences the
    legacy memory.py streams too, so telemetry has a single 'turn it all off'."""
    from heylook_llm import observability
    observability.configure(level="off", log_dir=tmp_path)
    mm.register_model_load(
        ModelMetadata("m", "/p", 0, "a", "none", 0, 0), load_duration_ms=1.0
    )
    assert not (tmp_path / "model_events.jsonl").exists()
    assert mm.maybe_log_baseline() is False


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


def test_snapshot_shape_and_content_invariant(mm: MemoryManager):
    mm.model_metadata["m1"] = ModelMetadata("m1", "/p", 123, "arch", "4bit", 7_000_000_000, 8192)
    snapshot = mm.snapshot()

    # Required numeric keys
    for key in (
        "ts", "rss_bytes", "available_ram_bytes", "cpu_percent",
        "mlx_active_bytes", "mlx_peak_bytes", "mlx_cache_bytes",
    ):
        assert key in snapshot, f"missing key {key}"

    assert snapshot["loaded_models"][0]["model_id"] == "m1"
    assert snapshot["mode"] == "background"

    # Content invariant: recursive walk, broad forbidden-key set.
    # prompt_cache / vision_cache are allowed -- they're counter dicts.
    _assert_no_forbidden_keys(snapshot)


def test_model_event_content_invariant(mm: MemoryManager, tmp_path: Path):
    metadata = ModelMetadata(
        model_id="qwen3-4b",
        path="~/models/qwen3-4b-4bit",  # path-privacy: ignore (fixture: tests tilde normalization)
        weights_bytes=2_400_000_000,
        architecture="qwen3",
        quantization="4bit",
        param_count=4_000_000_000,
        context_length=32768,
    )
    mm.register_model_load(metadata, load_duration_ms=3500.0)
    mm.register_model_unload("qwen3-4b", reason="lru_evict")

    events = _read_jsonl(tmp_path / "model_events.jsonl")
    assert len(events) == 2
    for event in events:
        _assert_no_forbidden_keys(event)




def test_normalize_path_strips_home_prefix():
    home = str(Path.home())
    assert _normalize_path_for_log(f"{home}/models/foo") == "~/models/foo"  # path-privacy: ignore (the function under test strips the home prefix)
    assert _normalize_path_for_log(home) == "~"
    # Non-home paths are preserved
    assert _normalize_path_for_log("/opt/models/bar") == "/opt/models/bar"
    assert _normalize_path_for_log("") == ""


def test_maybe_log_baseline_respects_interval(tmp_path: Path):
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(),
        log_dir=tmp_path,
    )
    assert manager.maybe_log_baseline() is True
    # Second call within the interval should not write
    assert manager.maybe_log_baseline() is False
    events = _read_jsonl(tmp_path / "memory_baseline.jsonl")
    assert len(events) == 1


def test_log_startup_info_writes_when_level_raised(tmp_path: Path):
    # conftest configures level="minimal" per test.
    from heylook_llm.memory import BASELINE_INTERVAL_SECONDS
    manager = MemoryManager(router=_make_router(), app_config=_make_app_config(), log_dir=tmp_path)
    manager.log_startup_info()
    events = _read_jsonl(tmp_path / "baseline.jsonl")
    assert len(events) == 1
    assert events[0]["event"] == "startup"
    assert events[0]["baseline_interval_seconds"] == BASELINE_INTERVAL_SECONDS


def test_log_startup_info_gated_off_leaves_no_footprint(tmp_path: Path):
    """observability_level=off silences the startup record too, and the log
    dir is created lazily on first write -- so at 'off' it never appears."""
    from heylook_llm import observability
    log_dir = tmp_path / "logs"
    observability.configure(level="off", log_dir=log_dir)
    manager = MemoryManager(
        router=_make_router(),
        app_config=_make_app_config(),
        log_dir=log_dir,
    )
    manager.log_startup_info()
    assert not log_dir.exists()


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
    manager.register_model_load(
        ModelMetadata("m", "/p", 0, "a", "none", 0, 0), load_duration_ms=1.0)
