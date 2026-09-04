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
        assert len(data["data"]) == 2  # test-mlx-model + test-gguf-model (conftest roster)

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


class TestThinkingDefaultOnModelRows:
    """v1.79.63: every /v1/models entry carries `thinking_default`, the
    cascade's answer for an empty request -- the same value the admin row
    reports, so a page reading either list labels 'model default' the same."""

    def test_present_and_bool_on_every_entry(self, client):
        data = client.get("/v1/models").json()["data"]
        assert data
        for entry in data:
            assert isinstance(entry["thinking_default"], bool)
            if "thinking" not in entry.get("capabilities", []):
                assert entry["thinking_default"] is False

    def test_agrees_with_the_admin_row(self, client):
        listed = {m["id"]: m["thinking_default"] for m in client.get("/v1/models").json()["data"]}
        admin = {m["id"]: m["thinking_default"] for m in client.get("/v1/admin/models").json()["models"]}
        for mid, value in listed.items():
            assert admin.get(mid) == value, mid
