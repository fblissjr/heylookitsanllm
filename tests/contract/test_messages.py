# tests/contract/test_messages.py
#
# Contract tests for POST /v1/messages (Anthropic Messages-inspired API).

import json

import pytest


class TestMessagesNonStreaming:
    """Tests for POST /v1/messages (non-streaming)."""

    def test_valid_request_returns_200(self, client):
        """A valid messages request returns 200 with content blocks."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        assert resp.status_code == 200

        data = resp.json()
        assert data["role"] == "assistant"
        assert isinstance(data["content"], list)
        assert len(data["content"]) >= 1

    def test_response_has_text_block(self, client):
        """Response content includes at least one text block."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        data = resp.json()
        text_blocks = [b for b in data["content"] if b["type"] == "text"]
        assert len(text_blocks) >= 1
        assert len(text_blocks[0]["text"]) > 0

    def test_response_has_usage(self, client):
        """Response includes usage with input/output tokens."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        data = resp.json()
        assert "usage" in data
        assert "output_tokens" in data["usage"]

    def test_response_has_model_and_id(self, client):
        """Response includes model and id fields."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        data = resp.json()
        assert "id" in data
        assert data["model"] == "test-mlx-model"

    def test_content_blocks_with_typed_input(self, client):
        """Content blocks as input (not just string) are accepted."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello typed"}],
            }],
            "max_tokens": 128,
        })
        assert resp.status_code == 200

    def test_system_prompt_as_top_level_param(self, client):
        """System prompt is a top-level parameter, not in messages."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a helpful assistant.",
            "max_tokens": 128,
        })
        assert resp.status_code == 200

    def test_missing_messages_returns_422(self, client):
        """Request without messages returns 422."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "max_tokens": 128,
        })
        assert resp.status_code == 422


class TestMessagesStreaming:
    """Tests for POST /v1/messages with stream=true."""

    def test_streaming_returns_sse_events(self, client):
        """stream=true returns SSE with structured event types."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        # Extract event type lines
        event_lines = [l for l in body.split("\n") if l.startswith("event: ")]
        event_types = [l.split("event: ", 1)[1] for l in event_lines]

        # Must start with message_start and end with message_stop
        assert event_types[0] == "message_start"
        assert event_types[-1] == "message_stop"

    def test_streaming_has_content_block_events(self, client):
        """Streaming includes content_block_start, content_block_delta, content_block_stop."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "stream": True,
        })
        body = resp.text
        event_lines = [l for l in body.split("\n") if l.startswith("event: ")]
        event_types = [l.split("event: ", 1)[1] for l in event_lines]

        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types

    def test_streaming_data_lines_are_valid_json(self, client):
        """Each data: line in SSE is valid JSON."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "stream": True,
        })
        body = resp.text
        for line in body.split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                parsed = json.loads(payload)
                assert "type" in parsed
