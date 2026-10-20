"""Exercise the public embedding boundary, using real schemas and media handling."""
import json
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from heylook_llm.embeddings_api import embeddings_router
from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider
from heylook_llm.providers.common.generation_gate import GenerationGate
from heylook_llm.providers.common.embedding import finish, resolved_items


class EmbeddingProvider(BaseProvider):
    provider_name = "mlx"

    def load_model(self):
        pass

    def create_chat_completion(self, request, abort_event=None):
        raise AssertionError("embedding must never generate")


@pytest.fixture
def embedding_client(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "embedding_gemma2", "vision_config": {}, "audio_config": {},
    }))
    config = AppConfig(models=[{"id": "embed", "provider": "mlx", "config": {"model_path": str(tmp_path)}}])
    provider = EmbeddingProvider("embed", {"model_path": str(tmp_path)}, False)
    provider._gen_gate = GenerationGate()
    router = SimpleNamespace(app_config=config, cache_lock=threading.RLock(),
                             providers=OrderedDict(embed=provider), get_provider=lambda model: provider)
    app = FastAPI()
    app.include_router(embeddings_router)
    app.state.router_instance = router
    with TestClient(app) as client:
        yield client, provider


def test_endpoint_preserves_joint_items_and_returns_normalized_vectors(embedding_client):
    client, provider = embedding_client
    def run(p, items, abort):
        assert p.active_generations > 0
        assert p._gen_gate.busy
        assert len(items) == 2
        assert [part["type"] for part in items[0][1]] == ["text", "image", "audio", "video"]
        return [[1.0]*768, [2.0, *([0.0]*767)]], 12
    body = {"model": "embed", "dimensions": 128, "input": [{"content": [
        {"type": "text", "text": "scene"},
        *[{"type": kind, "source": {"type": "base64", "media_type": mime, "data": "eA=="}}
          for kind, mime in (("image", "image/png"), ("audio", "audio/wav"), ("video", "video/mp4"))],
    ]}, "query"]}
    with patch("heylook_llm.providers.mlx_embeddings.embed", side_effect=run):
        response = client.post("/v1/embeddings", json=body, headers={"X-Request-ID": "embed-test"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["request_id"] == response.headers["X-Request-ID"] == "embed-test"
    assert result["input_tokens"] == 12
    assert len(result["data"]) == 2 and len(result["data"][0]["embedding"]) == 128
    assert result["data"][1]["embedding"][0] == 1.0
    assert not provider._gen_gate.busy and provider.active_generations == 0


@pytest.mark.parametrize("change,status", [
    ({"model": "unknown"}, 404), ({"dimensions": 64}, 400),
    ({"input": []}, 422), ({"temperature": 0.1}, 422),
    ({"input": [{"prompt_name": "unknown", "content": [{"type": "text", "text": "x"}]}]}, 400),
])
def test_bad_requests_never_execute(embedding_client, change, status):
    client, _ = embedding_client
    with patch("heylook_llm.providers.mlx_embeddings.embed", side_effect=AssertionError("must not execute")):
        response = client.post("/v1/embeddings", json={"model": "embed", "input": "x", **change})
    assert response.status_code == status, response.text
