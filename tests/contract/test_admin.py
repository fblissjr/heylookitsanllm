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


class TestAdminProfiles:
    """Tests for GET /v1/admin/models/profiles (list profiles)."""

    def test_returns_profile_list(self, client):
        """GET /v1/admin/models/profiles returns all available profiles."""
        resp = client.get("/v1/admin/models/profiles")
        assert resp.status_code == 200

        data = resp.json()
        assert "profiles" in data
        assert len(data["profiles"]) == 9

    def test_profiles_have_name_and_description(self, client):
        """Each profile entry has name and description."""
        resp = client.get("/v1/admin/models/profiles")
        for profile in resp.json()["profiles"]:
            assert "name" in profile
            assert "description" in profile
            assert len(profile["name"]) > 0
            assert len(profile["description"]) > 0


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

    def test_status_for_known_model(self, client):
        """Status endpoint returns load state for a known model."""
        resp = client.get("/v1/admin/models/test-mlx-model/status")
        assert resp.status_code == 200

        data = resp.json()
        assert "loaded" in data
        assert isinstance(data["loaded"], bool)

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
        assert "loaded" in resp.json()
