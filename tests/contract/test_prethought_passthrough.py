# tests/contract/test_prethought_passthrough.py
#
# A provider that PRE-SPLITS reasoning (llama-server sends reasoning_content
# separately; the adapter maps it to GenerationChunk.thinking) must have its
# thinking reach the client's thinking channel WITHOUT the text parser
# involved -- and its text must flow as plain content.
#
# Claim: deleting these tests orphans GenerationChunk.thinking -- the consume
# loops could drop pre-split reasoning on the floor (the pre-7a state) and
# only a live llama-server run would notice. Pinned on /v1/messages, both
# modes; the OpenAI route they were first written against went in v1.79.66.

import json

import pytest

from heylook_llm.providers.base import BaseProvider, GenerationChunk


class PreSplitProvider(BaseProvider):
    """Yields chunks the way a llama-server adapter will: reasoning first
    (thinking field, empty text), then content text, then a final counts
    chunk.

    Subclasses BaseProvider so a new provider obligation lands here with its
    default rather than as a 500 from a route (see conftest.FakeProvider).
    """

    provider_name = "gguf"

    def __init__(self, model_id: str):
        super().__init__(model_id, {"model_path": "/fake/model.gguf"}, False)
        self.processor = None

    def load_model(self):
        """Nothing to load."""

    def template_info(self):
        return None  # gguf providers pre-split; parser stack goes pass-through

    def create_chat_completion(self, request, abort_event=None):
        yield GenerationChunk(thinking="let me think", token=1)
        yield GenerationChunk(thinking=" about this", token=2)
        yield GenerationChunk(text="Hello", token=3)
        yield GenerationChunk(text=" world", token=4,
                              prompt_tokens=5, generation_tokens=4,
                              finish_reason="stop")


@pytest.fixture
def presplit_model(mock_router):
    mock_router.providers["test-mlx-model"] = PreSplitProvider("test-mlx-model")
    yield "test-mlx-model"
    mock_router.providers.pop("test-mlx-model", None)


def test_non_streaming_carries_presplit_thinking(client, presplit_model):
    # The 2026-07-26 review found this surface silently diverging from its
    # then-sibling route (draft telemetry omitted); the thinking channel must
    # not repeat that.
    r = client.post("/v1/messages", json={
        "model": presplit_model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    })
    assert r.status_code == 200, r.text
    blocks = r.json()["content"]
    thinking = "".join(b["text"] for b in blocks if b["type"] == "thinking")
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    assert thinking == "let me think about this"
    assert text == "Hello world"


def test_streaming_emits_thinking_deltas(client, presplit_model):
    with client.stream("POST", "/v1/messages", json={
        "model": presplit_model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "stream": True,
    }) as r:
        assert r.status_code == 200
        thinking_parts, content_parts = [], []
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[len("data: "):])
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload["delta"]
            if delta.get("type") == "thinking_delta":
                thinking_parts.append(delta["thinking"])
            elif delta.get("type") == "text_delta":
                content_parts.append(delta["text"])
    assert "".join(thinking_parts) == "let me think about this"
    assert "".join(content_parts) == "Hello world"
