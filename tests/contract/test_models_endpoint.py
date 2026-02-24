# tests/contract/test_models_endpoint.py
#
# Contract tests for GET /v1/models (OpenAI-compatible model listing).

import pytest


class TestListModels:
    """Tests for GET /v1/models."""

    def test_returns_model_list(self, client):
        """GET /v1/models returns a list with expected structure."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200

        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 2  # test-mlx-model + test-gguf-model

    def test_model_entry_has_required_fields(self, client):
        """Each model entry has id, object, and owned_by fields."""
        resp = client.get("/v1/models")
        data = resp.json()

        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "owned_by" in model

    def test_model_ids_match_config(self, client):
        """Model IDs in response match the test config."""
        resp = client.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        assert ids == {"test-mlx-model", "test-gguf-model"}

    def test_model_has_provider_field(self, client):
        """Models include provider information."""
        resp = client.get("/v1/models")
        providers = {m["id"]: m.get("provider") for m in resp.json()["data"]}
        assert providers["test-mlx-model"] == "mlx"
        assert providers["test-gguf-model"] == "gguf"
