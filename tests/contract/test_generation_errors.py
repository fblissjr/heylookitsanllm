# tests/contract/test_generation_errors.py
#
# Generation failures must surface as errors, not as assistant content.
# History: the provider first yielded error TEXT as a normal chunk (clients
# rendered "Error: MLX generation failed..." as a model reply); then an
# `is_error` sentinel chunk checked at 4 API sites -- which batch_processor
# and rlm.py missed, concatenating error text into results (RLM fed it back
# into its REPL loop). The provider now RAISES typed exceptions, so every
# consumer -- present and future -- fails loudly by default:
#
# - GenerationFailed          -> HTTP 500 (non-streaming) / SSE error payload
# - InvalidGenerationRequest  -> HTTP 400 (client error, e.g. images sent to
#                                a text-only model) / SSE error payload when
#                                streaming (headers already sent)
# - streaming: `data: {"error": {...}}` then [DONE]; never a content delta

import json

import pytest

from heylook_llm.providers.base import GenerationFailed, InvalidGenerationRequest
from helpers.mlx_mock import FakeChunk


class _FailingProvider:
    """Yields one real chunk, then raises -- the mid-stream failure shape."""

    def __init__(self, exc):
        self.model_id = "test-mlx-model"
        self.processor = None
        self._exc = exc

    def check_capacity(self):
        pass

    def create_chat_completion(self, request, abort_event=None):
        yield FakeChunk("partial", token_id=1)
        raise self._exc


class _PreflightFailingProvider(_FailingProvider):
    """Raises before ANY chunk -- the pre-generation validation shape."""

    def create_chat_completion(self, request, abort_event=None):
        raise self._exc
        yield  # pragma: no cover -- makes this a generator function


@pytest.fixture
def swap_provider(mock_router):
    original = mock_router.providers.get("test-mlx-model")

    def _swap(provider):
        mock_router.providers["test-mlx-model"] = provider

    yield _swap
    if original is not None:
        mock_router.providers["test-mlx-model"] = original
    else:
        mock_router.providers.pop("test-mlx-model", None)


def _stream(client):
    return client.post("/v1/chat/completions", json={
        "model": "test-mlx-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    })


def test_streaming_failure_is_error_payload_not_content(client, swap_provider):
    swap_provider(_FailingProvider(GenerationFailed(
        "MLX generation failed: There is no Stream(gpu, 19) in current thread.")))
    resp = _stream(client)
    assert resp.status_code == 200  # headers already sent mid-stream

    data_lines = [l[len("data: "):] for l in resp.text.split("\n") if l.startswith("data: ")]
    payloads = [json.loads(l) for l in data_lines if l and l != "[DONE]"]

    error_payloads = [p for p in payloads if "error" in p]
    assert error_payloads, f"expected an error payload, got: {payloads}"
    err = error_payloads[0]["error"]
    assert "MLX generation failed" in err["message"]
    assert err["code"] == "generation_failed"

    content_deltas = [
        p["choices"][0]["delta"].get("content", "")
        for p in payloads
        if p.get("choices") and p["choices"][0].get("delta")
    ]
    assert not any("MLX generation failed" in c for c in content_deltas)
    assert data_lines[-1] == "[DONE]"


def test_streaming_preflight_failure_also_error_payload(client, swap_provider):
    swap_provider(_PreflightFailingProvider(InvalidGenerationRequest(
        "Model 'test-mlx-model' is text-only and cannot process images.")))
    resp = _stream(client)
    payloads = [json.loads(l[6:]) for l in resp.text.split("\n")
                if l.startswith("data: ") and l != "data: [DONE]"]
    assert any("error" in p for p in payloads)


def test_non_streaming_server_failure_returns_500(client, swap_provider):
    swap_provider(_FailingProvider(GenerationFailed("MLX generation failed: boom")))
    resp = client.post("/v1/chat/completions", json={
        "model": "test-mlx-model",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 500
    assert "MLX generation failed" in resp.json()["detail"]


def test_non_streaming_client_error_returns_400(client, swap_provider):
    # e.g. images sent to a text-only model: the CLIENT's mistake, not a 500.
    swap_provider(_PreflightFailingProvider(InvalidGenerationRequest(
        "Model 'test-mlx-model' is text-only and cannot process images.")))
    resp = client.post("/v1/chat/completions", json={
        "model": "test-mlx-model",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 400
    assert "text-only" in resp.json()["detail"]


def test_exception_hierarchy():
    # Consumers may catch GenerationFailed alone and still see client errors.
    assert issubclass(InvalidGenerationRequest, GenerationFailed)
