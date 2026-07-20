"""Tests for client: payload building, model picking, response parsing."""

import pytest

from batch_labeler.client import (
    GenerationOptions,
    LabelResponse,
    build_payload,
    parse_chat_response,
    pick_vision_model,
    vision_models,
)


DATA_URL = "data:image/jpeg;base64,AAAA"


class TestBuildPayload:
    def test_minimal(self):
        payload = build_payload(
            model_id="m",
            system_prompt="sys",
            user_prompt="usr",
            image_data_url=DATA_URL,
            options=GenerationOptions(),
        )
        assert payload["model"] == "m"
        assert payload["stream"] is False
        assert payload["include_performance"] is True
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        user = payload["messages"][1]
        assert user["role"] == "user"
        types = [part["type"] for part in user["content"]]
        assert types == ["image_url", "text"]
        assert user["content"][1]["text"] == "usr"
        # No None-valued optional fields leak into the payload
        for key in ("temperature", "sampler", "enable_thinking", "vision_tokens",
                    "resize_max", "image_quality", "seed", "top_p"):
            assert key not in payload

    def test_all_options_forwarded(self):
        opts = GenerationOptions(
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            seed=42,
            sampler="vlm-extract",
            enable_thinking=True,
            vision_tokens=1024,
            resize_max=1024,
            image_quality=90,
        )
        payload = build_payload(
            model_id="m", system_prompt="s", user_prompt="u",
            image_data_url=DATA_URL, options=opts,
        )
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.9
        assert payload["seed"] == 42
        assert payload["sampler"] == "vlm-extract"
        assert payload["enable_thinking"] is True
        assert payload["vision_tokens"] == 1024
        assert payload["resize_max"] == 1024
        assert payload["image_quality"] == 90

    def test_thinking_off_is_forwarded(self):
        # False must be sent (explicit off), only None is omitted
        opts = GenerationOptions(enable_thinking=False)
        payload = build_payload(
            model_id="m", system_prompt="s", user_prompt="u",
            image_data_url=DATA_URL, options=opts,
        )
        assert payload["enable_thinking"] is False


class TestParseChatResponse:
    def test_content_and_thinking(self):
        data = {
            "model": "m",
            "choices": [{"message": {
                "role": "assistant",
                "content": '{"a": 1}',
                "thinking": "hmm",
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "performance": {"generation_tps": 42.0},
        }
        resp = parse_chat_response(data)
        assert isinstance(resp, LabelResponse)
        assert resp.content == '{"a": 1}'
        assert resp.thinking == "hmm"
        assert resp.usage["completion_tokens"] == 5
        assert resp.performance["generation_tps"] == 42.0

    def test_no_thinking_no_performance(self):
        data = {
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {},
        }
        resp = parse_chat_response(data)
        assert resp.thinking is None
        assert resp.performance is None

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError):
            parse_chat_response({"choices": [], "usage": {}})


MODELS = [
    {"id": "text-only", "capabilities": ["chat"]},
    {"id": "vlm-a", "capabilities": ["chat", "vision"]},
    {"id": "vlm-b", "modalities": ["text", "vision"]},
    {"id": "embed", "provider": "mlx_embedding"},
]


class TestModelPicking:
    def test_vision_models_by_capability_or_modality(self):
        ids = [m["id"] for m in vision_models(MODELS)]
        assert ids == ["vlm-a", "vlm-b"]

    def test_pick_returns_sole_vision_model(self):
        models = [MODELS[0], MODELS[1]]
        assert pick_vision_model(models) == "vlm-a"

    def test_pick_ambiguous_returns_none(self):
        assert pick_vision_model(MODELS) is None

    def test_pick_no_vision_returns_none(self):
        assert pick_vision_model([MODELS[0]]) is None
