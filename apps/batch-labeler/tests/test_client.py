"""Tests for client: payload building, image blocks, model picking, response parsing."""

import base64
import io

import pytest
from PIL import Image

from batch_labeler.client import (
    GenerationOptions,
    LabelResponse,
    build_payload,
    image_block,
    parse_message_response,
    pick_vision_model,
    vision_models,
)


IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}


class TestBuildPayload:
    def test_messages_shape_and_none_omitted(self):
        payload = build_payload(
            model_id="m", system_prompt="sys", user_prompt="usr",
            image=IMAGE, options=GenerationOptions(),
        )
        assert payload["model"] == "m"
        assert payload["system"] == "sys"
        assert payload["stream"] is False
        (user,) = payload["messages"]
        assert user["role"] == "user"
        assert [part["type"] for part in user["content"]] == ["image", "text"]
        assert user["content"][1]["text"] == "usr"
        for key in ("max_tokens", "temperature", "top_p", "seed", "thinking"):
            assert key not in payload

    def test_options_forwarded_and_false_kept(self):
        opts = GenerationOptions(max_tokens=512, temperature=0.3, top_p=0.9, seed=42,
                                 thinking=False)
        payload = build_payload(model_id="m", system_prompt="s", user_prompt="u",
                                image=IMAGE, options=opts)
        assert (payload["max_tokens"], payload["temperature"], payload["top_p"],
                payload["seed"], payload["thinking"]) == (512, 0.3, 0.9, 42, False)


def _png(tmp_path, w, h, name="i.png"):
    path = tmp_path / name
    Image.new("RGB", (w, h), (10, 20, 30)).save(path)
    return path


class TestImageBlock:
    def test_small_image_is_sent_unchanged(self, tmp_path):
        path = _png(tmp_path, 64, 32)
        block = image_block(path, max_edge=2048)
        assert block["source"]["media_type"] == "image/png"
        assert base64.b64decode(block["source"]["data"]) == path.read_bytes()

    def test_large_image_is_capped_on_its_longest_edge(self, tmp_path):
        block = image_block(_png(tmp_path, 3000, 1500), max_edge=1000)
        im = Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))
        assert im.size == (1000, 500)

    def test_zero_sends_as_is(self, tmp_path):
        path = _png(tmp_path, 3000, 10)
        assert base64.b64decode(image_block(path, max_edge=0)["source"]["data"]) == path.read_bytes()


class TestParseMessageResponse:
    def test_content_and_thinking(self):
        data = {
            "model": "m",
            "content": [{"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": '{"a": 1}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "performance": {"generation_tps": 42.0},
        }
        resp = parse_message_response(data)
        assert isinstance(resp, LabelResponse)
        assert (resp.content, resp.thinking) == ('{"a": 1}', "hmm")
        assert resp.usage["output_tokens"] == 5
        assert resp.performance["generation_tps"] == 42.0

    def test_no_thinking_no_performance(self):
        resp = parse_message_response({"model": "m", "content": [{"type": "text", "text": "hi"}]})
        assert (resp.thinking, resp.performance) == (None, None)

    def test_no_content_raises(self):
        with pytest.raises(ValueError):
            parse_message_response({"content": [], "usage": {}})


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
