# tests/unit/test_generation_chunk.py
#
# Contract tests for the owned GenerationChunk provider-output type and the
# BaseProvider capability surface (plan Phase 7a seam hardening).
#
# Claims (what breaks if a test is deleted):
# - slots/defaults tests: the chunk type regresses to a non-slotted attr-bag
#   and silent runtime attr-patching (the old GenerationResponse mechanism)
#   can return.
# - from_engine test: the duck-conversion from engine chunk shapes (mlx-lm
#   GenerationResponse, mlx-vlm diffusion chunks) drifts from the fields the
#   API layer scrapes.
# - telemetry latch tests: ChunkTelemetry.absorb regresses to last-write-wins,
#   zeroing first-chunk-only telemetry (the cache report /
#   queue_wait_ms) now that every field exists on every chunk.
# - capability-surface tests: neutral code goes back to reading private
#   MLXProvider attrs (_template_info) or class-name sniffing.
# - abort_event signature tests: concrete providers drift from the abstract
#   contract again (the pre-7a state).

import inspect
from types import SimpleNamespace

import pytest

from heylook_llm.providers.base import BaseProvider, CacheReport, GenerationChunk, SpecReport
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
            (GenerationChunk,
             dict(text="", token=None, thinking=None, finish_reason=None,
                  prompt_tokens=0, generation_tokens=0, prompt_tps=0.0,
                  generation_tps=0.0, peak_memory=0.0, cache=None, spec=None,
                  queue_wait_ms=0.0)),
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
        ids=["defaults", "slotted-no-attr-patching", "from-engine-full", "from-engine-sparse"],
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
# BaseProvider capability surface
# ---------------------------------------------------------------------------

class TestProviderSurface:
    def test_base_defaults(self):
        assert BaseProvider.provider_name == ""
        assert BaseProvider.is_vlm is False

        class P(BaseProvider):
            def load_model(self):
                pass

            def create_chat_completion(self, request, abort_event=None):
                yield GenerationChunk()

        p = P("m", {}, False)
        assert p.template_info() is None

    def test_abort_event_in_abstract_signature(self):
        sig = inspect.signature(BaseProvider.create_chat_completion)
        assert "abort_event" in sig.parameters

    def test_concrete_providers_accept_abort_event(self):
        mlx = pytest.importorskip("mlx")  # noqa: F841 -- import gate only
        from heylook_llm.providers.mlx_provider import MLXProvider

        sig = inspect.signature(MLXProvider.create_chat_completion)
        assert "abort_event" in sig.parameters

    def test_provider_name_set_on_concrete_classes(self):
        mlx = pytest.importorskip("mlx")  # noqa: F841
        from heylook_llm.providers.mlx_provider import MLXProvider

        assert MLXProvider.provider_name == "mlx"


# ---------------------------------------------------------------------------
# Provider config registry (single source of truth for known providers)
# ---------------------------------------------------------------------------

class TestProviderConfigRegistry:
    # That the registry keys match the ModelConfig.provider Literal is
    # test_config_effects_adversarial.py::
    # test_provider_registry_and_the_provider_literal_stay_in_sync.

    def test_validator_uses_registry(self):
        from heylook_llm.config import ModelConfig, MLXModelConfig

        mc = ModelConfig.model_validate(
            {"id": "m", "provider": "mlx", "config": {"model_path": "/tmp/fake"}}
        )
        assert isinstance(mc.config, MLXModelConfig)

    def test_unknown_provider_rejected(self):
        from heylook_llm.config import ModelConfig

        with pytest.raises(Exception):
            ModelConfig.model_validate(
                {"id": "m", "provider": "onnx", "config": {"model_path": "/x"}}
            )
