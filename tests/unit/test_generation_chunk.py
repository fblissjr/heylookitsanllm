# tests/unit/test_generation_chunk.py
#
# Contract tests for the owned GenerationChunk provider-output type, the
# telemetry latch over it, and the concrete providers as the router and the
# routes drive them (plan Phase 7a seam hardening).
#
# Claims (what breaks if a test is deleted):
# - slots test: the chunk type regresses to a non-slotted attr-bag
#   and silent runtime attr-patching (the old GenerationResponse mechanism)
#   can return.
# - from_engine test: the duck-conversion from engine chunk shapes (mlx-lm
#   GenerationResponse, mlx-vlm diffusion chunks) drifts from the fields the
#   API layer scrapes.
# - telemetry latch tests: ChunkTelemetry.absorb regresses to last-write-wins,
#   zeroing first-chunk-only telemetry (the cache report /
#   queue_wait_ms) now that every field exists on every chunk.
# - concrete-provider tests: a provider's declared engine drifts from its
#   config entry (loaded models read as stale, live config never reaches
#   them), or a provider stops honouring the per-request abort signal the
#   routes pass it.

from types import SimpleNamespace

import pytest

from heylook_llm.config import ChatRequest
from heylook_llm.providers.base import CacheReport, GenerationChunk, SpecReport
from heylook_llm.perf_collector import ChunkTelemetry

from _fake_chunk import fake_chunk as _chunk


# ---------------------------------------------------------------------------
# GenerationChunk shape
# ---------------------------------------------------------------------------

class TestGenerationChunkShape:
    """Slotted (no silent runtime attr-patching); from_engine copies every
    field an engine chunk carries and defaults the rest (diffusion /
    first-vision-token chunks carry only a subset)."""

    @pytest.mark.parametrize(
        "build, expected",
        [
            (lambda: GenerationChunk(text="hi"), dict(text="hi")),
            (lambda: GenerationChunk.from_engine(SimpleNamespace(
                text="tok", token=42, finish_reason="stop", prompt_tokens=10,
                generation_tokens=5, prompt_tps=100.0, generation_tps=50.0,
                peak_memory=1.5)),
             dict(text="tok", token=42, finish_reason="stop", prompt_tokens=10,
                  generation_tokens=5, prompt_tps=100.0, generation_tps=50.0,
                  peak_memory=1.5)),
            (lambda: GenerationChunk.from_engine(SimpleNamespace(text="x")),
             dict(text="x", token=None, prompt_tokens=0, finish_reason=None)),
        ],
        ids=["slotted-no-attr-patching", "from-engine-full", "from-engine-sparse"],
    )
    def test_shape(self, build, expected):
        c = build()
        for field, value in expected.items():
            assert getattr(c, field) == value, field
        with pytest.raises(AttributeError):
            c.surprise_field = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ChunkTelemetry latch semantics (fields now ALWAYS present on chunks)
# ---------------------------------------------------------------------------

_REPORT = CacheReport(prompt_tokens=10, cached_tokens=7, outcome="reused")


class TestTelemetryLatch:
    """ChunkTelemetry.absorb must not regress to last-write-wins now that every
    field exists on every chunk (the old getattr-absence trick no longer
    protects them). First-chunk fields (the cache report, queue_wait_ms) and
    non-zero rates survive later defaults -- the vision first-token chunk has
    no rates and must not wipe the engine's numbers; finish_reason arrives on
    the FINAL chunk and a trailing chunk without one must not erase it (the
    scrape lives in absorb(), one place, so the consume loops cannot drift
    apart: a response cut off by max_tokens must stay distinguishable from a
    natural stop); spec-decode reports are cumulative running totals, so the
    latest wins and a chunk without one does not reset them; peak memory is a
    max. Rows are absorb steps, each with what must hold after it; the
    finish_reason rows feed engine-shaped chunks (``_fake_chunk``)."""

    @pytest.mark.parametrize(
        "steps",
        [
            [(GenerationChunk(text="a", cache=_REPORT, queue_wait_ms=5.5, prompt_tokens=10,
                              generation_tokens=1, prompt_tps=100.0, generation_tps=50.0), {}),
             (GenerationChunk(text="b", prompt_tokens=10, generation_tokens=2,
                              prompt_tps=100.0, generation_tps=51.0),
              {"cache": _REPORT, "queue_wait_ms": 5.5, "completion_tokens": 2})],
            [(GenerationChunk(text="a", prompt_tps=120.0, generation_tps=80.0), {}),
             (GenerationChunk(text="b"), {"prompt_tps": 120.0, "generation_tps": 80.0})],
            [(GenerationChunk(text="a", finish_reason="length"), {}),
             (GenerationChunk(text=""), {"finish_reason": "length"})],
            [(GenerationChunk(peak_memory=2.0), {}),
             (GenerationChunk(peak_memory=1.0), {"peak_memory_gb": 2.0})],
            [(GenerationChunk(text="a", spec=SpecReport(accepted=4, emitted=10)), {}),
             (GenerationChunk(text="b", spec=SpecReport(accepted=9, emitted=20)), {}),
             (GenerationChunk(text=""), {"spec": SpecReport(accepted=9, emitted=20)})],
            [(_chunk("hi", finish_reason=None), {"finish_reason": None}),
             (_chunk("", finish_reason="length"), {"finish_reason": "length"})],
            [(_chunk("", finish_reason="length"), {}),
             (_chunk("", finish_reason=None), {"finish_reason": "length"})],
        ],
        ids=[
            "first-chunk-snapshot-fields-survive-later-zeros", "zero-tps-does-not-regress",
            "finish-reason-latches", "peak-memory-monotonic", "spec-report-latches-the-latest",
            "absorbs-engine-finish-reason", "later-none-does-not-clear-a-seen-reason",
        ],
    )
    def test_latch(self, steps):
        t = ChunkTelemetry()
        for chunk, expected in steps:
            t.absorb(chunk)
            for field, value in expected.items():
                if isinstance(value, CacheReport):
                    assert getattr(t, field) is value, field
                else:
                    assert getattr(t, field) == value, field

    def test_profile_weights_by_tokens_and_keeps_the_rates_apart(self):
        # Perf-page trends and the cache section (plan W5): token-weighted,
        # None (never 0) where no request reported the quantity, and the two
        # draft rates never merged -- MLX knows no drafted count.
        import time as _time

        from heylook_llm.perf_collector import PerfCollector, RequestEvent

        def event(**kw):
            base = dict(
                timestamp=_time.time(), model="m", success=True, total_ms=100.0,
                model_load_ms=0.0,
                token_generation_ms=90.0, prompt_tokens=10,
                completion_tokens=50, tokens_per_second=40.0, had_images=False,
                was_streaming=True,
            )
            base.update(kw)
            return RequestEvent(**base)

        def reports(cache, spec):
            t = ChunkTelemetry()
            t.cache, t.spec = cache, spec
            return RequestEvent.report_fields(t)

        c = PerfCollector(max_events=16)
        c.record_request(event(**reports(
            CacheReport(prompt_tokens=100, cached_tokens=90, outcome="reused"),
            SpecReport(accepted=60, drafted=100, emitted=80))))
        c.record_request(event(**reports(
            CacheReport(prompt_tokens=300, cached_tokens=0, outcome="miss", cause="cold"),
            SpecReport(accepted=20, emitted=50))))
        profile = c.build_profile("1h")
        (row,) = profile["trends"]
        assert row["cache_share"] == round(90 / 400, 3)
        assert row["draft_acceptance"] == 0.6          # gguf's alone: (60)/100
        assert row["draft_share"] == round(80 / 130, 3)
        (model,) = profile["cache"]
        assert model["outcomes"] == {"reused": 1, "miss": 1}
        assert model["causes"] == {"cold": 1}

        c2 = PerfCollector(max_events=16)
        c2.record_request(event())
        (row2,) = c2.build_profile("1h")["trends"]
        assert row2["cache_share"] is None and row2["draft_acceptance"] is None
        assert c2.build_profile("1h")["cache"] == []


# ---------------------------------------------------------------------------
# The concrete providers, driven the way the router and the routes drive them
# ---------------------------------------------------------------------------

def _mlx_class():
    from heylook_llm import router as router_mod

    if router_mod.MLXProvider is None:
        pytest.skip("mlx not importable")
    return router_mod.MLXProvider


def _gguf_class():
    from heylook_llm.providers.llama_server_provider import LlamaServerProvider

    return LlamaServerProvider


_ENGINES = [pytest.param("mlx", _mlx_class, id="mlx"),
            pytest.param("gguf", _gguf_class, id="gguf")]


def _toml(kind, temperature=None):
    lines = ["max_loaded_models = 1", "", "[[models]]", 'id = "m1"',
             f'provider = "{kind}"', "", "[models.config]",
             f'model_path = "/fake/m1{".gguf" if kind == "gguf" else ""}"']
    if temperature is not None:
        lines.append(f"temperature = {temperature}")
    return "\n".join(lines) + "\n"


class TestConcreteProviders:
    @pytest.mark.parametrize("kind, provider_class", _ENGINES)
    def test_a_loaded_model_is_not_stale_and_takes_live_config(
            self, kind, provider_class, tmp_path, monkeypatch):
        """The router matches a resident provider to its config entry by the
        class's declared engine: a mismatch reports a freshly loaded model as
        needing a reload ("provider") and skips the per_request refresh, so a
        PATCHed temperature never reaches the loaded model. Real provider
        classes, loaded through the router with only the weights load
        stubbed."""
        import logging

        from heylook_llm.router import ModelRouter

        cls = provider_class()
        monkeypatch.setattr(cls, "load_model", lambda self: None)
        monkeypatch.setattr(cls, "warmup", lambda self: None)
        monkeypatch.setattr(cls, "unload", lambda self, **kw: None)
        path = tmp_path / "heylook.toml"
        path.write_text(_toml(kind))
        router = ModelRouter(config_path=str(path), log_level=logging.INFO,
                             initial_model_id=None)
        provider = router.get_provider("m1")
        assert type(provider) is cls
        assert router.stale_reload_fields("m1") == []

        path.write_text(_toml(kind, temperature=0.9))
        router.reload_config()
        assert provider.config["temperature"] == 0.9

    @pytest.mark.parametrize("kind, provider_class", _ENGINES)
    def test_a_request_cancelled_while_queued_never_generates(
            self, kind, provider_class, monkeypatch):
        """The routes pass the per-request abort signal positionally
        (``create_chat_completion(request, abort_event)``, messages_api.py).
        A request whose client left while it queued behind another
        generation must leave the queue and yield nothing. If a provider
        stopped honouring the signal, it would take its turn once the other
        run releases the gate and fail on the unloaded model (MLX) or forward
        to llama-server (gguf)."""
        import threading

        from heylook_llm.providers.abort import AbortEvent
        from heylook_llm.providers.common.generation_gate import reset_process_gate

        reset_process_gate()
        cls = provider_class()
        path = "/fake/m1.gguf" if kind == "gguf" else "/fake/m1"
        provider = cls("m1", {"model_path": path}, False)
        if kind == "gguf":
            from heylook_llm.providers import llama_server_provider as llama_mod

            provider._base_url = "http://127.0.0.1:9"  # "loaded"
            monkeypatch.setattr(
                llama_mod.urllib.request, "urlopen",
                lambda *a, **k: (_ for _ in ()).throw(AssertionError("forwarded a cancelled request")))
        gate = provider._gen_gate
        gate.acquire()  # another request is generating
        releaser = threading.Timer(1.0, gate.release)
        releaser.start()
        try:
            abort = AbortEvent()
            abort.set()  # the client is already gone
            request = ChatRequest.model_validate(
                {"messages": [{"role": "user", "content": "hi"}]})
            assert list(provider.create_chat_completion(request, abort)) == []
            assert gate.snapshot()["waiting"] == 0
        finally:
            releaser.cancel()
            releaser.join()
            reset_process_gate()


# ---------------------------------------------------------------------------
# Provider config registry (single source of truth for known providers)
# ---------------------------------------------------------------------------

class TestProviderConfigRegistry:
    # That the registry keys match the ModelConfig.provider Literal is
    # test_config_effects_adversarial.py::
    # test_provider_registry_and_the_provider_literal_stay_in_sync; that the
    # validator builds the registry's class is test_config.py
    # test_pydantic_construction_round_trip (row validator_uses_registry).

    def test_unknown_provider_rejected(self):
        from heylook_llm.config import ModelConfig

        with pytest.raises(Exception):
            ModelConfig.model_validate(
                {"id": "m", "provider": "onnx", "config": {"model_path": "/x"}}
            )
