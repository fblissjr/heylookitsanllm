"""Tests for heylook_llm.memory (S1.2 observability)."""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest

from heylook_llm.memory import (
    MemoryManager,
    ModelMetadata,
    _normalize_path_for_log,
    parse_bool_env,
    sampler_summary_from_request,
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


def testparse_bool_env_defaults_when_unset():
    assert parse_bool_env(None, True) is True
    assert parse_bool_env(None, False) is False


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False), ("", False)],
)
def testparse_bool_env_values(value: str, expected: bool):
    assert parse_bool_env(value, not expected) is expected


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

    # Content invariant: recursive walk, broad forbidden-key set.
    # prompt_cache / vision_cache are allowed -- they're counter dicts.
    _assert_no_forbidden_keys(snapshot)


def test_request_event_content_invariant(mm: MemoryManager, tmp_path: Path):
    """A typical request-event record must carry no content-bearing keys.

    This covers the higher-risk path: per-request logs are the natural place
    for a future engineer to sneak in `messages`, `prompt`, or `tool_calls`.
    """
    record = {
        "timestamp": 100.0,
        "model": "test-model",
        "success": True,
        "total_ms": 1500.0,
        "queue_ms": 5.0,
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "tokens_per_second": 40.0,
        "had_images": True,
        "was_streaming": False,
        "sampler_summary": {"temperature": 0.7, "max_tokens": 512},
        "peak_memory_gb": 4.2,
        "kv_cache_bytes": 131072,
        "cached_tokens": 250,
        "thinking_tokens": 30,
        "content_tokens": 70,
        "stop_reason": "stop",
        "provider_type": "mlx",
        "image_count": 2,
        "cache_hit_rate": 0.5,
    }
    mm.log_request_event(record)
    written = _read_jsonl(tmp_path / "request_events.jsonl")
    assert len(written) == 1
    _assert_no_forbidden_keys(written[0])


def test_model_event_content_invariant(mm: MemoryManager, tmp_path: Path):
    metadata = ModelMetadata(
        model_id="qwen3-4b",
        path="~/models/qwen3-4b-4bit",
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


def test_request_event_dataclass_has_only_primitive_fields():
    """Guardrail: RequestEvent must only declare primitive field types.

    A future change that adds a list/dict field to RequestEvent (e.g.
    `messages_preview: list[str]`) would flow straight through asdict()
    into request_events.jsonl. This test fails fast if someone tries.
    """
    from heylook_llm.perf_collector import RequestEvent

    allowed_types = {int, float, bool, str}
    for field_info in fields(RequestEvent):
        ftype = field_info.type
        if isinstance(ftype, str):
            # Forward-ref; accept the common primitive names.
            assert ftype in {"int", "float", "bool", "str"}, (
                f"RequestEvent.{field_info.name} declares non-primitive "
                f"type {ftype!r}; request_events.jsonl would leak complex data"
            )
        else:
            assert ftype in allowed_types, (
                f"RequestEvent.{field_info.name} declares non-primitive "
                f"type {ftype!r}"
            )


def test_normalize_path_strips_home_prefix():
    home = str(Path.home())
    assert _normalize_path_for_log(f"{home}/models/foo") == "~/models/foo"
    assert _normalize_path_for_log(home) == "~"
    # Non-home paths are preserved
    assert _normalize_path_for_log("/opt/models/bar") == "/opt/models/bar"
    assert _normalize_path_for_log("") == ""


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
