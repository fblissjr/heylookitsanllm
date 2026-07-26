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
#   zeroing first-chunk-only telemetry (cached_tokens / kv_cache_bytes /
#   queue_wait_ms) now that every field exists on every chunk.
# - capability-surface tests: neutral code goes back to reading private
#   MLXProvider attrs (_template_info) or class-name sniffing.
# - abort_event signature tests: concrete providers drift from the abstract
#   contract again (the pre-7a state).

import inspect
from types import SimpleNamespace

import pytest

from heylook_llm.providers.base import BaseProvider, GenerationChunk
from heylook_llm.perf_collector import ChunkTelemetry


# ---------------------------------------------------------------------------
# GenerationChunk shape
# ---------------------------------------------------------------------------

class TestGenerationChunkShape:
    def test_defaults(self):
        c = GenerationChunk()
        assert c.text == ""
        assert c.token is None
        assert c.logprobs is None
        assert c.thinking is None
        assert c.finish_reason is None
        assert c.prompt_tokens == 0
        assert c.generation_tokens == 0
        assert c.prompt_tps == 0.0
        assert c.generation_tps == 0.0
        assert c.peak_memory == 0.0
        assert c.cached_tokens == 0
        assert c.kv_cache_bytes == 0
        assert c.queue_wait_ms == 0.0
        assert c.from_draft is False

    def test_slotted_no_attr_patching(self):
        c = GenerationChunk(text="hi")
        with pytest.raises(AttributeError):
            c.surprise_field = 1  # type: ignore[attr-defined]

    def test_from_engine_full(self):
        engine = SimpleNamespace(
            text="tok",
            token=42,
            logprobs="fake-array",
            finish_reason="stop",
            prompt_tokens=10,
            generation_tokens=5,
            prompt_tps=100.0,
            generation_tps=50.0,
            peak_memory=1.5,
            from_draft=True,
        )
        c = GenerationChunk.from_engine(engine)
        assert c.text == "tok"
        assert c.token == 42
        assert c.logprobs == "fake-array"
        assert c.finish_reason == "stop"
        assert c.prompt_tokens == 10
        assert c.generation_tokens == 5
        assert c.prompt_tps == 100.0
        assert c.generation_tps == 50.0
        assert c.peak_memory == 1.5
        assert c.from_draft is True

    def test_from_engine_sparse(self):
        # Diffusion / first-vision-token chunks carry only a subset.
        c = GenerationChunk.from_engine(SimpleNamespace(text="x"))
        assert c.text == "x"
        assert c.token is None
        assert c.prompt_tokens == 0
        assert c.finish_reason is None


# ---------------------------------------------------------------------------
# ChunkTelemetry latch semantics (fields now ALWAYS present on chunks)
# ---------------------------------------------------------------------------

class TestTelemetryLatch:
    def test_first_chunk_snapshot_fields_survive_later_zeros(self):
        t = ChunkTelemetry()
        t.absorb(GenerationChunk(text="a", cached_tokens=7, kv_cache_bytes=1024,
                                 queue_wait_ms=5.5, prompt_tokens=10,
                                 generation_tokens=1, prompt_tps=100.0,
                                 generation_tps=50.0))
        # Later chunks have the fields but at their defaults (0) -- the old
        # getattr-absence trick no longer protects them.
        t.absorb(GenerationChunk(text="b", prompt_tokens=10,
                                 generation_tokens=2, prompt_tps=100.0,
                                 generation_tps=51.0))
        assert t.cached_tokens == 7
        assert t.kv_cache_bytes == 1024
        assert t.queue_wait_ms == 5.5
        assert t.completion_tokens == 2

    def test_zero_tps_does_not_regress(self):
        # The vision first-token chunk has no rates; it must not wipe the
        # engine's numbers absorbed from surrounding chunks.
        t = ChunkTelemetry()
        t.absorb(GenerationChunk(text="a", prompt_tps=120.0, generation_tps=80.0))
        t.absorb(GenerationChunk(text="b"))
        assert t.prompt_tps == 120.0
        assert t.generation_tps == 80.0

    def test_finish_reason_latches(self):
        t = ChunkTelemetry()
        t.absorb(GenerationChunk(text="a", finish_reason="length"))
        t.absorb(GenerationChunk(text=""))
        assert t.finish_reason == "length"

    def test_peak_memory_monotonic(self):
        t = ChunkTelemetry()
        t.absorb(GenerationChunk(peak_memory=2.0))
        t.absorb(GenerationChunk(peak_memory=1.0))
        assert t.peak_memory_gb == 2.0

    def test_draft_acceptance_latches(self):
        # Spec-decode counters are cumulative running totals; the final
        # chunk carries the request's totals and zeros must not reset them.
        t = ChunkTelemetry()
        t.absorb(GenerationChunk(text="a", draft_tokens=10, draft_accepted=4))
        t.absorb(GenerationChunk(text="b", draft_tokens=20, draft_accepted=9))
        t.absorb(GenerationChunk(text=""))
        assert t.draft_tokens == 20
        assert t.draft_accepted == 9


# ---------------------------------------------------------------------------
# BaseProvider capability surface
# ---------------------------------------------------------------------------

class TestProviderSurface:
    def test_base_defaults(self):
        assert BaseProvider.provider_name == ""
        assert BaseProvider.is_vlm is False
        assert BaseProvider.effective_loader is None

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
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        for cls in (MLXProvider, MLXEmbeddingProvider):
            sig = inspect.signature(cls.create_chat_completion)
            assert "abort_event" in sig.parameters, cls.__name__

    def test_provider_name_set_on_concrete_classes(self):
        mlx = pytest.importorskip("mlx")  # noqa: F841
        from heylook_llm.providers.mlx_provider import MLXProvider
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        assert MLXProvider.provider_name == "mlx"
        assert MLXEmbeddingProvider.provider_name == "mlx_embedding"


# ---------------------------------------------------------------------------
# Provider config registry (single source of truth for known providers)
# ---------------------------------------------------------------------------

class TestProviderConfigRegistry:
    def test_registry_keys_match_literal(self):
        import typing

        from heylook_llm.config import PROVIDER_CONFIG_CLASSES, ModelConfig

        # The registry and the ModelConfig.provider Literal must never drift.
        literal = set(typing.get_args(ModelConfig.model_fields["provider"].annotation))
        assert set(PROVIDER_CONFIG_CLASSES) == literal

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
