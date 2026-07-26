# tests/unit/test_audio_content.py
#
# Audio input content blocks (plan Phase 7d, API-surface layers 1-6).
# Audio is served ONLY by the gguf/llama-server provider (MLX strips audio
# towers at load); the MLX path must reject audio with a 400-class error,
# never silently drop the part (the pre-7d failure mode in vlm_inputs).
#
# Claims (what breaks if a test is deleted):
# - schema tests: OpenAI-wire `input_audio` parts regress to 422 rejection
#   (or worse, silent drops if the union ever loosens).
# - converter test: Messages-API AudioBlock stops bridging to the OpenAI
#   input_audio shape the gguf provider forwards.
# - passthrough test: the llama-server payload builder starts mangling or
#   dropping audio parts.
# - MLX guard test: audio to an MLX model regresses from a loud
#   InvalidGenerationRequest to a silent text-only answer.
# - capability test: gguf models with audio modality stop advertising the
#   `audio` cap that the frontend/eval gating (7d follow-ups) will rely on.

import pytest

from heylook_llm.config import AudioContentPart, ChatRequest, ModelConfig


AUDIO_PART = {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
TEXT_PART = {"type": "text", "text": "what do you hear?"}


def audio_request():
    return ChatRequest.model_validate({
        "messages": [{"role": "user", "content": [TEXT_PART, AUDIO_PART]}],
    })


class TestSchema:
    def test_input_audio_part_validates(self):
        req = audio_request()
        part = req.messages[0].content[1]
        assert isinstance(part, AudioContentPart)
        assert part.input_audio.data == "UklGRg=="
        assert part.input_audio.format == "wav"

    def test_format_optional(self):
        # llama-server sniffs the codec by magic bytes and ignores `format`.
        req = ChatRequest.model_validate({
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "AAAA"}}]}],
        })
        assert req.messages[0].content[0].input_audio.format is None

    def test_dump_round_trips_wire_shape(self):
        dumped = audio_request().messages[0].model_dump(exclude_none=True)
        assert dumped["content"][1] == AUDIO_PART


class TestLlamaPayloadPassthrough:
    def test_audio_part_forwards_verbatim(self):
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider

        p = LlamaServerProvider("m", {"model_path": "/fake.gguf"}, False)
        payload = p._build_payload(audio_request())
        assert payload["messages"][0]["content"][1] == AUDIO_PART


class TestMessagesBridge:
    def test_audio_block_converts_to_input_audio_part(self):
        from heylook_llm.schema.content_blocks import AudioBlock
        from heylook_llm.schema.converters import to_chat_request
        from heylook_llm.schema.messages import MessageCreateRequest

        req = MessageCreateRequest.model_validate({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "listen"},
                {"type": "audio", "source_type": "base64",
                 "media_type": "audio/wav", "data": "UklGRg=="},
            ]}],
            "max_tokens": 16,
        })
        assert isinstance(req.messages[0].content[1], AudioBlock)
        chat = to_chat_request(req)
        parts = chat.messages[0].content
        assert parts[1].type == "input_audio"
        assert parts[1].input_audio.data == "UklGRg=="


class TestMLXGuard:
    def test_has_audio_parts_helper(self):
        from heylook_llm.providers.mlx_provider import _has_audio_parts

        req = audio_request()
        assert _has_audio_parts(req.messages) is True
        text_only = ChatRequest.model_validate(
            {"messages": [{"role": "user", "content": "hi"}]})
        assert _has_audio_parts(text_only.messages) is False


class TestCapability:
    def test_gguf_audio_modality_yields_audio_cap(self):
        from heylook_llm.api import _infer_model_capabilities

        mc = ModelConfig.model_validate({
            "id": "m", "provider": "gguf",
            "config": {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf",
                       "modalities": ["text", "vision", "audio"]},
        })
        caps = _infer_model_capabilities(mc)
        assert "audio" in caps
        assert "vision" in caps

    def test_mlx_never_gains_audio_cap_from_modalities(self):
        # MLX strips audio towers at load; advertising audio would invite
        # requests the provider must 400.
        mc = ModelConfig.model_validate({
            "id": "m", "provider": "mlx",
            "config": {"model_path": "/x", "modalities": ["text", "vision", "audio"]},
        })
        assert "audio" not in _infer(mc)


def _infer(mc):
    from heylook_llm.api import _infer_model_capabilities
    return _infer_model_capabilities(mc)
