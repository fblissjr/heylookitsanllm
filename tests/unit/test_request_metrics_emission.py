# tests/unit/test_request_metrics_emission.py
"""request_complete metrics emission into the observability spine.

Pins that _maybe_log_request_event mirrors the numeric request metrics into
logs/metrics.jsonl (content-free), and that the frozen §4.3 registry dims are
read null-safely -- an embedding provider (no effective_loader/is_vlm attrs)
yields null, not a crash.
"""

import orjson

from heylook_llm import observability as obs
from heylook_llm.api import _maybe_log_request_event
from heylook_llm.perf_collector import RequestEvent


class _StubMM:
    def mark_request_end(self):
        pass

    def log_request_event(self, record):
        pass


class _Provider:
    def __init__(self, provider, effective_loader=None, is_vlm=None):
        self.provider = provider
        if effective_loader is not None:
            self.effective_loader = effective_loader
        if is_vlm is not None:
            self.is_vlm = is_vlm


def _event(**kw):
    base = dict(
        timestamp=1.0, model="m", success=True, total_ms=100.0, queue_ms=5.0,
        model_load_ms=0.0, image_processing_ms=0.0, token_generation_ms=90.0,
        first_token_ms=20.0, prompt_tokens=10, completion_tokens=50,
        tokens_per_second=42.0, had_images=False, was_streaming=True,
    )
    base.update(kw)
    return RequestEvent(**base)


def _read(p):
    return [orjson.loads(line) for line in p.read_bytes().splitlines() if line]


def test_request_complete_emitted_with_metrics(tmp_path):
    obs.configure(level="minimal", log_dir=tmp_path)
    _maybe_log_request_event(
        {"memory_manager": _StubMM(), "image_count": 2},
        _event(),
        provider=_Provider("mlx", effective_loader="mlx-vlm", is_vlm=True),
        peak_memory_gb=3.5, kv_cache_bytes=1024, cached_tokens=4, stop_reason="stop",
    )
    (rec,) = _read(tmp_path / "metrics.jsonl")
    assert rec["type"] == "request_complete"
    assert rec["model"] == "m"
    assert rec["provider"] == "mlx"
    assert rec["effective_loader"] == "mlx-vlm"
    assert rec["is_vlm"] is True
    assert rec["completion_tokens"] == 50
    assert rec["generation_tps"] == 42.0
    assert rec["image_count"] == 2
    assert rec["peak_memory_gb"] == 3.5


def test_embedding_provider_yields_null_loader(tmp_path):
    obs.configure(level="minimal", log_dir=tmp_path)
    _maybe_log_request_event(
        {"memory_manager": _StubMM(), "image_count": 0},
        _event(model="emb"),
        provider=_Provider("mlx_embedding"),  # no effective_loader / is_vlm attrs
    )
    (rec,) = _read(tmp_path / "metrics.jsonl")
    assert rec["provider"] == "mlx_embedding"
    assert rec["effective_loader"] is None  # getattr default -> null, no throw
    assert rec["is_vlm"] is None


def test_off_level_suppresses_metrics(tmp_path):
    obs.configure(level="off", log_dir=tmp_path)
    _maybe_log_request_event(
        {"memory_manager": _StubMM(), "image_count": 0},
        _event(),
        provider=_Provider("mlx", effective_loader="mlx-lm", is_vlm=False),
    )
    assert not (tmp_path / "metrics.jsonl").exists()
