# tests/contract/test_prethought_passthrough.py
#
# A provider that PRE-SPLITS reasoning (llama-server sends reasoning_content
# separately; the adapter maps it to GenerationChunk.thinking) must have its
# thinking reach the client's thinking channel WITHOUT the text parser
# involved -- and its text must flow as plain content.
#
# Claim: deleting these tests orphans GenerationChunk.thinking -- the four
# consume loops could drop pre-split reasoning on the floor (the pre-7a
# state) and only a live llama-server run would notice.

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


def test_non_streaming_carries_thinking(client, presplit_model):
    r = client.post("/v1/chat/completions", json={
        "model": presplit_model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "Hello world"
    assert msg["thinking"] == "let me think about this"


def test_messages_api_carries_presplit_thinking(client, presplit_model):
    # Deleting this leaves the /v1/messages consume path unpinned -- the
    # 2026-07-26 review found exactly this surface silently diverging from
    # /v1/chat/completions (draft telemetry omitted); the thinking channel
    # must not repeat that.
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
    with client.stream("POST", "/v1/chat/completions", json={
        "model": presplit_model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        thinking_parts, content_parts = [], []
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[len("data: "):])["choices"][0].get("delta", {})
            if "thinking" in delta:
                thinking_parts.append(delta["thinking"])
            if "content" in delta:
                content_parts.append(delta["content"])
    assert "".join(thinking_parts) == "let me think about this"
    assert "".join(content_parts) == "Hello world"
