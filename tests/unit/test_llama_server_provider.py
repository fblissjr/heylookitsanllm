# tests/unit/test_llama_server_provider.py
#
# Unit tests for the llama-server subprocess provider (plan Phase 7b) --
# everything testable WITHOUT a running llama-server: config validation,
# the sampler cascade -> request payload, and the SSE -> GenerationChunk
# adapter fed canned llama-server stream frames.
#
# Claims (what breaks if a test is deleted):
# - config tests: "gguf" regresses out of the provider registry / Literal and
#   models.toml entries stop validating.
# - payload tests: the sampler cascade (floor -> default_sampler -> named
#   sampler -> explicit request fields) or the llama.cpp param mapping
#   (repetition_penalty -> repeat_penalty, always-send max_tokens, thinking
#   -> chat_template_kwargs) silently drifts.
# - SSE adapter tests: llama-server frames (reasoning_content deltas, usage
#   chunk with timings, keepalive comments, [DONE]) stop mapping onto
#   GenerationChunk fields -- telemetry goes dark or thinking is dropped.

import io

import pytest

from heylook_llm.config import ChatRequest, ModelConfig, GGUFModelConfig, PROVIDER_CONFIG_CLASSES
from heylook_llm.providers.base import GenerationChunk
from heylook_llm.providers.llama_server_provider import LlamaServerProvider


def req(**kw):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return ChatRequest.model_validate(body)


def make_provider(**config):
    config.setdefault("model_path", "/fake/model.gguf")
    return LlamaServerProvider("test-gguf", config, False)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

class TestGGUFConfig:
    def test_registry_has_gguf(self):
        assert PROVIDER_CONFIG_CLASSES["gguf"] is GGUFModelConfig

    def test_model_config_builds_gguf(self):
        mc = ModelConfig.model_validate({
            "id": "m", "provider": "gguf",
            "config": {"model_path": "/x/model.gguf", "mmproj_path": "/x/mmproj.gguf"},
        })
        assert isinstance(mc.config, GGUFModelConfig)
        assert mc.config.mmproj_path == "/x/mmproj.gguf"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            GGUFModelConfig.model_validate({"model_path": "/x.gguf", "surprise": True})

    def test_capability_inference(self):
        from heylook_llm.api import _infer_model_capabilities

        plain = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf", "config": {"model_path": "/x.gguf"}})
        assert _infer_model_capabilities(plain) == ["chat"]

        vision = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf",
             "config": {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf"}})
        assert "vision" in _infer_model_capabilities(vision)

        thinking = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf",
             "config": {"model_path": "/x.gguf", "supports_thinking": True}})
        caps = _infer_model_capabilities(thinking)
        assert "thinking" in caps
        assert "hidden_states" not in caps  # MLX-only feature stays MLX-only


# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------

class TestProviderSurface:
    def test_provider_name_and_template_info(self):
        p = make_provider()
        assert LlamaServerProvider.provider_name == "gguf"
        assert p.template_info() is None  # llama-server owns templating/split

    def test_unload_without_load_is_safe(self):
        make_provider().unload()  # must not raise


# ---------------------------------------------------------------------------
# Sampler cascade -> payload
# ---------------------------------------------------------------------------

class TestPayload:
    def test_floor_applied_and_max_tokens_always_sent(self):
        p = make_provider()
        payload = p._build_payload(req())
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 4096  # llama default is UNLIMITED; must always send
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_request_overrides_and_param_mapping(self):
        p = make_provider()
        payload = p._build_payload(req(temperature=0.1, repetition_penalty=1.3, max_tokens=64, seed=7, presence_penalty=1.5, top_k=20))
        assert payload["temperature"] == 0.1
        assert payload["repeat_penalty"] == 1.3  # llama.cpp's name
        assert "repetition_penalty" not in payload
        assert payload["max_tokens"] == 64
        assert payload["seed"] == 7
        assert payload["presence_penalty"] == 1.5
        assert payload["top_k"] == 20

    def test_model_config_max_tokens_beats_floor(self):
        # Deleting this resurrects the dead-overlay bug (code-review
        # 2026-07-26): the floor pre-seeds max_tokens, so a guarded
        # "if not in merged" write could never fire and a model-level
        # max_tokens silently fell back to 4096.
        p = make_provider(max_tokens=8000)
        payload = p._build_payload(req())
        assert payload["max_tokens"] == 8000

    def test_request_max_tokens_beats_model_config(self):
        p = make_provider(max_tokens=8000)
        payload = p._build_payload(req(max_tokens=64))
        assert payload["max_tokens"] == 64

    def test_default_sampler_overlays_floor(self):
        from heylook_llm.samplers import get_sampler_registry

        det_temp = get_sampler_registry()._presets["deterministic"]["temperature"]
        assert det_temp != 0.7  # must differ from the floor for this to prove anything
        p = make_provider(default_sampler="deterministic")
        payload = p._build_payload(req())
        assert payload["temperature"] == det_temp

    def test_named_request_sampler_beats_default_sampler(self):
        from heylook_llm.samplers import get_sampler_registry

        reg = get_sampler_registry()._presets
        assert reg["balanced"]["temperature"] != reg["deterministic"]["temperature"]
        p = make_provider(default_sampler="deterministic")
        payload = p._build_payload(req(sampler="balanced"))
        assert payload["temperature"] == reg["balanced"]["temperature"]

    def test_enable_thinking_maps_to_chat_template_kwargs(self):
        p = make_provider()
        on = p._build_payload(req(enable_thinking=True))
        off = p._build_payload(req(enable_thinking=False))
        unset = p._build_payload(req())
        assert on["chat_template_kwargs"] == {"enable_thinking": True}
        assert off["chat_template_kwargs"] == {"enable_thinking": False}
        assert "chat_template_kwargs" not in unset

    def test_multimodal_content_parts_pass_through(self):
        p = make_provider()
        content = [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        payload = p._build_payload(req(messages=[{"role": "user", "content": content}]))
        assert payload["messages"][0]["content"] == content


# ---------------------------------------------------------------------------
# SSE stream -> GenerationChunk
# ---------------------------------------------------------------------------

def _stream_bytes(*frames: str) -> io.BytesIO:
    return io.BytesIO(("\n".join(frames) + "\n").encode())


CANNED = [
    ": keepalive ping",
    'data: {"choices":[{"delta":{"role":"assistant","content":null},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"reasoning_content":"let me "},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"reasoning_content":"think"},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"content":" world"},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}',
    'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,'
    '"prompt_tokens_details":{"cached_tokens":2}},'
    '"timings":{"prompt_per_second":100.5,"predicted_per_second":42.0,'
    '"cache_n":2,"prompt_n":5,"predicted_n":3,"draft_n":12,"draft_n_accepted":5}}',
    "data: [DONE]",
    'data: {"choices":[{"delta":{"content":"MUST NOT APPEAR"},"index":0}]}',
]


class TestSSEAdapter:
    def collect(self, frames):
        p = make_provider()
        return list(p._stream_chunks(_stream_bytes(*frames), abort_event=None))

    def test_full_stream_mapping(self):
        chunks = self.collect(CANNED)
        assert all(isinstance(c, GenerationChunk) for c in chunks)

        thinking = "".join(c.thinking for c in chunks if c.thinking)
        text = "".join(c.text for c in chunks if c.text)
        assert thinking == "let me think"
        assert text == "Hello world"
        # nothing after [DONE]
        assert "MUST NOT APPEAR" not in text

        finish = [c.finish_reason for c in chunks if c.finish_reason]
        assert finish == ["stop"]

        final = chunks[-1]
        assert final.prompt_tokens == 7
        assert final.generation_tokens == 3
        assert final.cached_tokens == 2
        assert final.prompt_tps == 100.5
        assert final.generation_tps == 42.0
        # spec-decode counters (present only when MTP/draft was active)
        assert final.draft_tokens == 12
        assert final.draft_accepted == 5

    def test_abort_stops_stream(self):
        class Abort:
            def __init__(self):
                self.calls = 0

            def is_set(self):
                self.calls += 1
                return self.calls > 2

        p = make_provider()
        chunks = list(p._stream_chunks(_stream_bytes(*CANNED), abort_event=Abort()))
        assert len(chunks) < 6  # cut short, never reached the end of the stream

    def test_malformed_frame_raises_generation_failed(self):
        from heylook_llm.providers.base import GenerationFailed

        p = make_provider()
        with pytest.raises(GenerationFailed):
            list(p._stream_chunks(_stream_bytes("data: {not json"), abort_event=None))
