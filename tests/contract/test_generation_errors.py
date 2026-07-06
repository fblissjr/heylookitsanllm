# tests/contract/test_generation_errors.py
#
# Generation failures must surface as errors, not as assistant content.
# Historically the provider yielded the error text as a normal chunk, so SSE
# clients rendered "Error: MLX generation failed: ..." as a model response
# (and chat frontends persisted it as an assistant message).
#
# Contract:
# - streaming: a `data: {"error": {...}}` payload (then [DONE]), and NO
#   content delta carrying the error text
# - non-streaming: HTTP 500 with the error in `detail`

import json

import pytest

from helpers.mlx_mock import FakeChunk


class _ErrorChunk(FakeChunk):
    """FakeChunk plus the provider's MLXErrorChunk `is_error` marker."""

    is_error = True


class _ErroringProvider:
    def __init__(self, model_id="test-mlx-model"):
        self.model_id = model_id
        self.processor = None

    def check_capacity(self):
        pass

    def create_chat_completion(self, request, abort_event=None):
        yield _ErrorChunk("Error: MLX generation failed: There is no Stream(gpu, 19) in current thread.")


@pytest.fixture
def erroring_model(mock_router):
    """Swap the model's provider for one that yields an error chunk."""
    original = mock_router.providers.get("test-mlx-model")
    mock_router.providers["test-mlx-model"] = _ErroringProvider()
    yield
    if original is not None:
        mock_router.providers["test-mlx-model"] = original
    else:
        mock_router.providers.pop("test-mlx-model", None)


def test_streaming_error_is_error_payload_not_content(client, erroring_model):
    resp = client.post("/v1/chat/completions", json={
        "model": "test-mlx-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    })
    assert resp.status_code == 200  # headers already sent when mid-stream errors occur

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
    assert not any("MLX generation failed" in c for c in content_deltas), (
        "error text must not be delivered as assistant content"
    )
    assert data_lines[-1] == "[DONE]"


def test_non_streaming_error_returns_500(client, erroring_model):
    resp = client.post("/v1/chat/completions", json={
        "model": "test-mlx-model",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 500
    assert "MLX generation failed" in resp.json()["detail"]
