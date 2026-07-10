"""Contract tests for the /v1/jspace API (routing + guards).

Uses a bare FastAPI app with the jspace_router and fake app.state, stubbing the
heavy analyze pipeline so no model/MLX is needed.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import heylook_llm.jspace_api as jspace_api
from heylook_llm.jspace_api import jspace_router


class _FakeProvider:
    model = object()
    processor = object()
    is_vlm = False
    model_id = "m1"


class _FakeRouterInstance:
    def get_provider(self, model_id):
        return _FakeProvider()


class _FakeRegistry:
    base_dir = None

    def available(self):
        return ["m1"]

    def has(self, model_id):
        return model_id == "m1"

    def get(self, model_id):
        return object()

    def normalizer(self, model_id):
        return None

    def router(self, model_id):
        return None

    def provenance(self, model_id):
        return {"provisional": True, "n_prompts": 1}


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(jspace_router)
    app.state.jspace_registry = _FakeRegistry()
    app.state.router_instance = _FakeRouterInstance()
    monkeypatch.setattr(jspace_api, "run_analyze",
                        lambda *a, **k: {"ok": True, "onset_strip": []})
    return TestClient(app)


def test_models_lists_available(client):
    r = client.get("/v1/jspace/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == ["m1"]
    assert body["meta"]["m1"]["provisional"] is True   # per-lens provenance rides along


def test_analyze_404_when_no_lens(client):
    r = client.post("/v1/jspace/analyze", json={"model": "nope", "prompt": "hi"})
    assert r.status_code == 404


def test_analyze_422_when_no_prompt_or_messages(client):
    r = client.post("/v1/jspace/analyze", json={"model": "m1"})
    assert r.status_code == 422


def test_analyze_success_stubbed(client):
    r = client.post("/v1/jspace/analyze", json={"model": "m1", "prompt": "The capital of France is"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "onset_strip": []}


def test_analyze_accepts_messages(client):
    r = client.post("/v1/jspace/analyze",
                    json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_analyze_forwards_heatmap_top_k(client, monkeypatch):
    seen = {}

    def _capture(*a, **k):
        seen.update(k)
        return {"ok": True}

    monkeypatch.setattr(jspace_api, "run_analyze", _capture)
    r = client.post("/v1/jspace/analyze",
                    json={"model": "m1", "prompt": "hi", "heatmap": True, "heatmap_top_k": 5})
    assert r.status_code == 200
    assert seen["heatmap"] is True and seen["heatmap_top_k"] == 5
