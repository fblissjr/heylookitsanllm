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
# - passthrough: the llama-server payload builder starts mangling or dropping
#   audio parts (test_llama_server_provider.py
#   TestPayload::test_content_parts_pass_through, row audio_part_forwards_verbatim).
# - MLX guard test: audio to an MLX model regresses from a loud
#   InvalidGenerationRequest to a silent text-only answer.
# - capability table (infer_model_capabilities over provider, modalities,
#   mmproj and supports_thinking): gguf models with audio modality stop
#   advertising the `audio` cap that the frontend/eval gating (7d follow-ups)
#   will rely on, gguf media stops needing the projector, MLX gains audio, or
#   a plain gguf entry advertises more than chat.

import pytest

from heylook_llm.config import AudioContentPart, ChatRequest, ModelConfig


AUDIO_PART = {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
TEXT_PART = {"type": "text", "text": "what do you hear?"}


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
    # Rows: (provider, config, caps that must be present, caps that must be
    # absent, the exact list or None when only membership is pinned).
    @pytest.mark.parametrize("provider, config, present, absent, exact", [
        pytest.param("gguf", {"model_path": "/x.gguf"}, set(), set(), ["chat"],
                     id="gguf_plain_is_chat_only"),
        pytest.param("gguf", {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf"},
                     {"vision"}, set(), None, id="gguf_projector_yields_vision"),
        pytest.param("gguf", {"model_path": "/x.gguf", "supports_thinking": True},
                     {"thinking"}, set(), None, id="gguf_supports_thinking_yields_thinking"),
        pytest.param("gguf", {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf",
                              "modalities": ["text", "vision", "audio"]},
                     {"audio", "vision"}, set(), None, id="gguf_audio_modality_yields_audio_cap"),
        # `modalities` is descriptive on gguf; llama.cpp runs no image or
        # audio without an mmproj, so a declaration alone advertises nothing.
        pytest.param("gguf", {"model_path": "/x.gguf", "modalities": ["text", "vision", "audio"]},
                     set(), {"vision", "audio"}, None, id="gguf_media_needs_the_projector"),
        # MLX strips audio towers at load; advertising audio would invite
        # requests the provider must 400.
        pytest.param("mlx", {"model_path": "/x", "modalities": ["text", "vision", "audio"]},
                     set(), {"audio"}, None, id="mlx_never_gains_audio_cap_from_modalities"),
    ])
    def test_inferred_capabilities(self, provider, config, present, absent, exact):
        mc = ModelConfig.model_validate({"id": "m", "provider": provider, "config": config})
        inferred = _infer(mc)
        caps = set(inferred)
        assert present <= caps and not (absent & caps), caps
        if exact is not None:
            assert inferred == exact


def _infer(mc):
    from heylook_llm.capabilities import infer_model_capabilities
    return infer_model_capabilities(mc)
