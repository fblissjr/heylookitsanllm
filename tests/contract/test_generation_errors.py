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
# - GenerationFailed          -> HTTP 500 (non-streaming) / `event: error`
#                                typed api_error when streaming
# - InvalidGenerationRequest  -> HTTP 400 (client error, e.g. images sent to
#                                a text-only model) / `event: error` typed
#                                invalid_request_error when streaming (headers
#                                already sent)
# - streaming: the error event ENDS the stream; never a content delta
#
# Pinned on /v1/messages, the inference wire. The OpenAI route these were first
# written against was removed in v1.79.66.

import json

import pytest

from heylook_llm.providers.base import (
    BaseProvider, GenerationFailed, InvalidGenerationRequest,
)
from helpers.mlx_mock import FakeChunk


class _FailingProvider(BaseProvider):
    """Yields one real chunk, then raises -- the mid-stream failure shape.

    Subclasses BaseProvider so a new provider obligation lands here with its
    default rather than as a 500 from a route (see conftest.FakeProvider).
    """

    provider_name = "mlx"

    def __init__(self, exc):
        super().__init__("test-mlx-model", {"model_path": "/fake/mlx-model", "vision": False}, False)
        self.processor = None
        self._exc = exc

    def load_model(self):
        """Nothing to load."""

    def template_info(self):
        return None

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


_BODY = {
    "model": "test-mlx-model",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hello"}],
}


def _stream(client):
    return client.post("/v1/messages", json={**_BODY, "stream": True})


def _events(resp):
    """(event, payload) pairs off a Messages SSE body."""
    out, event = [], None
    for line in resp.text.split("\n"):
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            out.append((event, json.loads(line[len("data: "):])))
    return out


def _text_deltas(events):
    return [p["delta"]["text"] for e, p in events
            if e == "content_block_delta" and p["delta"].get("type") == "text_delta"]


def test_streaming_failure_is_error_event_not_content(client, swap_provider):
    swap_provider(_FailingProvider(GenerationFailed(
        "MLX generation failed: There is no Stream(gpu, 19) in current thread.")))
    resp = _stream(client)
    assert resp.status_code == 200  # headers already sent mid-stream

    events = _events(resp)
    errors = [p["error"] for e, p in events if e == "error"]
    assert errors, f"expected an error event, got: {events}"
    assert "MLX generation failed" in errors[0]["message"]
    assert errors[0]["type"] == "api_error"

    assert not any("MLX generation failed" in t for t in _text_deltas(events))
    # The error ENDS the stream: nothing claims the message finished after it.
    assert events[-1][0] == "error"
    assert not any(e == "message_stop" for e, _ in events)


def test_streaming_preflight_failure_also_error_event(client, swap_provider):
    swap_provider(_PreflightFailingProvider(InvalidGenerationRequest(
        "Model 'test-mlx-model' is text-only and cannot process images.")))
    events = _events(_stream(client))
    assert any(e == "error" for e, _ in events), events


def test_streaming_client_error_is_typed_invalid_request(client, swap_provider):
    # /code-review 53b266c finding 2: provider request-validation guards
    # (audio-on-MLX, the continuation guards) fire at first next(), after
    # headers flushed -- a real 400 is impossible, but the in-band event must
    # be TYPED as the client error it is (Anthropic's invalid_request_error),
    # not fall through to api_error.
    swap_provider(_PreflightFailingProvider(InvalidGenerationRequest(
        "user-role continuation is not supported on gguf models")))
    resp = _stream(client)
    assert resp.status_code == 200  # headers already sent
    errors = [p["error"] for e, p in _events(resp) if e == "error"]
    assert errors, f"expected an error event, got: {resp.text}"
    assert errors[0]["type"] == "invalid_request_error"
    assert "continuation" in errors[0]["message"]


def test_non_streaming_server_failure_returns_500(client, swap_provider):
    swap_provider(_FailingProvider(GenerationFailed("MLX generation failed: boom")))
    resp = client.post("/v1/messages", json=_BODY)
    assert resp.status_code == 500
    assert "MLX generation failed" in resp.json()["detail"]


def test_non_streaming_client_error_returns_400(client, swap_provider):
    # e.g. images sent to a text-only model: the CLIENT's mistake, not a 500.
    swap_provider(_PreflightFailingProvider(InvalidGenerationRequest(
        "Model 'test-mlx-model' is text-only and cannot process images.")))
    resp = client.post("/v1/messages", json=_BODY)
    assert resp.status_code == 400
    assert "text-only" in resp.json()["detail"]


def test_model_load_failure_is_500_not_400(client, mock_router, monkeypatch):
    """A failed model LOAD is a server fault, even though it surfaces as a
    ValueError from the same call that reports a bad model id.

    Claim: `get_provider` re-raises whatever `load_model` threw, and
    mlx-lm/transformers raise bare ValueError for corrupt weights, an
    unsupported `model_type`, or a malformed config.json. Only the router's own
    id-resolution failures (ModelNotFound) are the client's fault. Delete this
    and a corrupt model on disk reports as a non-retryable 400 with no
    traceback for the operator.
    """
    def _boom(model_id):
        raise ValueError("Model type qwen9 not supported.")

    monkeypatch.setattr(mock_router, "get_provider", _boom)
    resp = client.post("/v1/messages", json=_BODY)
    assert resp.status_code == 500


def test_unresolvable_model_returns_400(client):
    """Model routing failure is a 400, not a server fault.

    Claim: the route resolves the model through `router.get_provider`, which
    raises ValueError for an unknown id and for "no model named, no default
    configured". Delete this and the client's typo silently reports as a
    server fault again (it was a 500 through v1.44.4).
    """
    resp = client.post("/v1/messages", json={
        "model": "no-such-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 400
    assert "no-such-model" in resp.json()["detail"]


def test_exception_hierarchy():
    # Consumers may catch GenerationFailed alone and still see client errors.
    assert issubclass(InvalidGenerationRequest, GenerationFailed)
