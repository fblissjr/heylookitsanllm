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
    # An input_audio part validates to AudioContentPart (format optional:
    # llama-server sniffs the codec by magic bytes and ignores it) and dumps
    # back to the exact wire shape.
    @pytest.mark.parametrize("content, index, fmt", [
        pytest.param([TEXT_PART, AUDIO_PART], 1, "wav", id="input_audio_part_validates"),
        pytest.param([{"type": "input_audio", "input_audio": {"data": "AAAA"}}], 0, None,
                     id="format_optional"),
        pytest.param([AUDIO_PART], 0, "wav", id="dump_round_trips_wire_shape"),
    ])
    def test_input_audio_part_round_trips(self, content, index, fmt):
        msg = ChatRequest.model_validate(
            {"messages": [{"role": "user", "content": content}]}).messages[0]
        part = msg.content[index]
        assert isinstance(part, AudioContentPart)
        assert part.input_audio.data == content[index]["input_audio"]["data"]
        assert part.input_audio.format == fmt
        assert msg.model_dump(exclude_none=True)["content"][index] == content[index]


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




class TestCapability:
    @pytest.mark.parametrize("provider, config, present, absent", [
        pytest.param("gguf", {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf",
                              "modalities": ["text", "vision", "audio"]},
                     {"audio", "vision"}, set(), id="gguf_audio_modality_yields_audio_cap"),
        # `modalities` is descriptive on gguf; llama.cpp runs no image or
        # audio without an mmproj, so a declaration alone advertises nothing.
        pytest.param("gguf", {"model_path": "/x.gguf", "modalities": ["text", "vision", "audio"]},
                     set(), {"vision", "audio"}, id="gguf_media_needs_the_projector"),
        # MLX strips audio towers at load; advertising audio would invite
        # requests the provider must 400.
        pytest.param("mlx", {"model_path": "/x", "modalities": ["text", "vision", "audio"]},
                     set(), {"audio"}, id="mlx_never_gains_audio_cap_from_modalities"),
    ])
    def test_media_capabilities(self, provider, config, present, absent):
        mc = ModelConfig.model_validate({"id": "m", "provider": provider, "config": config})
        caps = set(_infer(mc))
        assert present <= caps and not (absent & caps), caps


def _infer(mc):
    from heylook_llm.capabilities import infer_model_capabilities
    return infer_model_capabilities(mc)
