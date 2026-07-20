# tests/contract/test_admin.py
#
# Contract tests for /v1/admin/models/ endpoints.


class TestAdminListModels:
    """Tests for GET /v1/admin/models (list all configs)."""

    def test_returns_model_list(self, client):
        """GET /v1/admin/models returns all model configs."""
        resp = client.get("/v1/admin/models")
        assert resp.status_code == 200

        data = resp.json()
        assert "models" in data
        assert "total" in data
        assert data["total"] == 1

    def test_model_entries_have_full_config(self, client):
        """Admin model entries include config dict and enabled status."""
        resp = client.get("/v1/admin/models")
        for model in resp.json()["models"]:
            assert "id" in model
            assert "provider" in model
            assert "enabled" in model
            assert "config" in model
            assert isinstance(model["config"], dict)


class TestAdminSamplerPresets:
    """Tests for GET /v1/admin/models/sampler-presets (list sampler presets).

    Terminology note (2026-07-20): the import/admin paths used to call these
    'profiles'; they are the same registry as ChatRequest.preset and are now
    named sampler presets everywhere. Distinct from /v1/presets (saved user
    prompt+sampler bundles in the DuckDB store).
    """

    def test_returns_preset_list(self, client):
        """GET /v1/admin/models/sampler-presets returns exactly the bundled presets.

        Derive the expectation from the preset registry (the source of truth)
        rather than a hardcoded count, so adding/removing a preset doesn't
        silently rot this contract test.
        """
        from heylook_llm.presets import get_preset_registry

        expected = set(get_preset_registry().list_names())
        assert expected, "preset registry is empty -- bundled presets missing?"

        resp = client.get("/v1/admin/models/sampler-presets")
        assert resp.status_code == 200

        data = resp.json()
        assert "presets" in data
        returned = {p["name"] for p in data["presets"]}
        assert returned == expected

    def test_presets_have_name_and_description(self, client):
        """Each sampler-preset entry has name and description."""
        resp = client.get("/v1/admin/models/sampler-presets")
        for preset in resp.json()["presets"]:
            assert "name" in preset
            assert "description" in preset
            assert len(preset["name"]) > 0
            assert len(preset["description"]) > 0


class TestAdminScan:
    """Tests for POST /v1/admin/models/scan."""

    def test_scan_returns_results(self, client):
        """POST /v1/admin/models/scan returns scan results structure."""
        resp = client.post("/v1/admin/models/scan", json={
            "paths": [],
            "scan_hf_cache": False,
        })
        assert resp.status_code == 200

        data = resp.json()
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)


class TestAdminModelStatus:
    """Tests for GET /v1/admin/models/{model_id}/status."""

    def test_status_not_shadowed_by_catchall(self, client):
        """The /status sub-resource route is not swallowed by the catch-all GET.

        This validates the route registration order fix: sub-resource routes
        (/{model_id:path}/status) must register before the greedy catch-all
        (/{model_id:path}).
        """
        # If shadowed, this would return 404 from the catch-all trying to
        # look up model_id="test-mlx-model/status"
        resp = client.get("/v1/admin/models/test-mlx-model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert isinstance(data["loaded"], bool)
