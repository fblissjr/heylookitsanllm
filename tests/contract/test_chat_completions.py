# tests/contract/test_chat_completions.py
#
# Contract tests for POST /v1/chat/completions (OpenAI-compatible).

import pytest

from heylook_llm.providers.common.generation_gate import ModelBusyError


class TestChatCompletionsBackpressure:
    """The 503 backpressure path: a full generation queue is rejected, not queued
    forever. Previously this 503 code existed but was never triggered."""

    def test_full_queue_returns_503_with_headers(self, client, mock_router):
        # Force provider creation, then make it report a full queue.
        provider = mock_router.get_provider("test-mlx-model")
        orig_check = provider.check_capacity
        orig_stats = getattr(provider, "generation_queue_stats", None)
        provider.check_capacity = lambda: (_ for _ in ()).throw(
            ModelBusyError("MODEL_BUSY: 8 request(s) already queued (max_waiting=8)")
        )
        provider.generation_queue_stats = lambda: {
            "active": 1, "waiting": 8, "max_waiting": 8, "capacity": 9
        }
        try:
            resp = client.post("/v1/chat/completions", json={
                "model": "test-mlx-model",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        finally:
            provider.check_capacity = orig_check
            if orig_stats is not None:
                provider.generation_queue_stats = orig_stats
            else:
                del provider.generation_queue_stats

        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "model_overloaded"
        assert resp.headers.get("Retry-After") == "1"
        # Capacity header reflects the queue (1 active + 8 waiting), not "1".
        assert resp.headers.get("X-RateLimit-Limit") == "9"


class TestChatCompletionsNonStreaming:
    """Tests for POST /v1/chat/completions (non-streaming)."""

    def test_valid_request_returns_200(self, client):
        """A valid chat request returns 200 with generated content."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200

        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_response_has_usage_fields(self, client):
        """Response includes usage with token counts."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert "usage" in data
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_response_has_id_and_model(self, client):
        """Response includes id and model fields."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")
        assert data["model"] == "test-mlx-model"

    def test_missing_messages_returns_422(self, client):
        """Request without messages field returns 422."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
        })
        assert resp.status_code == 422

    def test_empty_messages_returns_422(self, client):
        """Request with empty messages array returns 422."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [],
        })
        assert resp.status_code == 422

    def test_unknown_model_returns_500(self, client):
        """Request for a model not in config returns 500."""
        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        # The router raises ValueError which becomes 500
        assert resp.status_code == 500

    def test_generated_content_matches_fake_chunks(self, client):
        """Non-streaming response content matches FakeProvider output."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "Hello, world!"


class TestChatCompletionsStreaming:
    """Tests for POST /v1/chat/completions with stream=true."""

    def test_streaming_returns_sse(self, client):
        """stream=true returns text/event-stream with data: lines ending in [DONE]."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.startswith("data: ")]
        assert len(lines) >= 2  # At least one chunk + [DONE]
        assert lines[-1] == "data: [DONE]"

    def test_streaming_chunks_are_valid_json(self, client):
        """Each data: line (except [DONE]) is valid JSON."""
        import json

        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        body = resp.text
        for line in body.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                assert chunk["object"] == "chat.completion.chunk"
                assert "choices" in chunk


class TestIncludePerformance:
    """include_performance=true populates the response's `performance` block
    from native mlx-lm telemetry; omitted/false leaves it null."""

    def test_performance_null_by_default(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        assert resp.json().get("performance") is None

    def test_performance_populated_when_requested(self, client, mock_router):
        from helpers.mlx_mock import FakeChunk

        provider = mock_router.get_provider("test-mlx-model")
        orig = provider.create_chat_completion

        def gen(request, abort_event=None):
            chunk = FakeChunk("Hello", token_id=1)
            chunk.prompt_tps = 100.5
            chunk.generation_tps = 42.5
            chunk.peak_memory = 1.5
            yield chunk

        provider.create_chat_completion = gen
        try:
            resp = client.post("/v1/chat/completions", json={
                "model": "test-mlx-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "include_performance": True,
            })
        finally:
            provider.create_chat_completion = orig

        assert resp.status_code == 200
        perf = resp.json()["performance"]
        assert perf["prompt_tps"] == 100.5
        assert perf["generation_tps"] == 42.5
        assert perf["peak_memory_gb"] == 1.5
