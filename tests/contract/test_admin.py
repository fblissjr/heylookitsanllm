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


class TestAdminEffectiveLoader:
    """`effective_loader` on the admin row: which MLX library decodes.

    Provider `mlx` is TWO upstream repos (mlx-lm text, mlx-vlm vision) with
    separate release trains, so "provider" does not name an engine. The live
    smoke harness picks one model per ENGINE and has to do it from this
    endpoint.
    """

    def test_answered_for_an_unloaded_model(self, client):
        """The whole point: it is derived from the config, not read off a
        loaded provider.

        `MLXProvider.effective_loader` exists only on a resident process, so a
        field sourced from there would be null for every model the harness has
        not already loaded -- which is exactly the set it needs to choose from.
        Nothing in this fixture is loaded, and the answer is still there.
        """
        resp = client.get("/v1/admin/models")
        assert resp.status_code == 200
        rows = {m["id"]: m for m in resp.json()["models"]}
        row = rows["test-mlx-model"]
        assert row["loaded"] is False
        # Non-vision mlx -> the text loader, with no probe of mlx-vlm's
        # registry and no read of a model dir that does not exist.
        assert row["effective_loader"] == "mlx-lm"

    def test_single_model_route_agrees_with_the_list(self, client):
        """Both admin reads build the same response object; pin that they
        keep answering the same way rather than growing a second derivation."""
        listed = client.get("/v1/admin/models").json()["models"][0]
        single = client.get(f"/v1/admin/models/{listed['id']}").json()
        assert single["effective_loader"] == listed["effective_loader"]


class TestAdminSamplers:
    """Tests for GET /v1/admin/models/samplers (list named samplers).

    Terminology note (2026-07-20): the import/admin paths used to call these
    'profiles'; they are the same registry as ChatRequest.preset and are now
    named samplers everywhere. Distinct from /v1/presets (saved user
    prompt+sampler bundles in the DuckDB store).
    """

    def test_returns_preset_list(self, client):
        """GET /v1/admin/models/samplers returns exactly the bundled presets.

        Derive the expectation from the preset registry (the source of truth)
        rather than a hardcoded count, so adding/removing a preset doesn't
        silently rot this contract test.
        """
        from heylook_llm.samplers import get_sampler_registry

        expected = set(get_sampler_registry().list_names())
        assert expected, "preset registry is empty -- bundled presets missing?"

        resp = client.get("/v1/admin/models/samplers")
        assert resp.status_code == 200

        data = resp.json()
        assert "samplers" in data
        returned = {p["name"] for p in data["samplers"]}
        assert returned == expected

    def test_samplers_have_name_and_description(self, client):
        """Each named-sampler entry has name and description."""
        resp = client.get("/v1/admin/models/samplers")
        for entry in resp.json()["samplers"]:
            assert "name" in entry
            assert "description" in entry
            assert len(entry["name"]) > 0
            assert len(entry["description"]) > 0


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


class TestAdminReload:
    """POST /v1/admin/models/{id}/reload[?warm=true] -- ONE server-owned
    unload+load(+warm). The browser-driven unload-then-load pair could
    strand a model unloaded if the client died between the calls; this
    route shares load's exact body, so the warm contract cannot fork."""

    def test_reload_returns_load_shape(self, client):
        resp = client.post("/v1/admin/models/test-mlx-model/reload?warm=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["model_id"] == "test-mlx-model"
        assert data["warmed"] is True
        assert isinstance(data["warm_ms"], int)

    def test_reload_of_unloaded_model_is_just_a_load(self, client, mock_router):
        mock_router.unload_model("test-mlx-model")
        resp = client.post("/v1/admin/models/test-mlx-model/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        assert "warmed" not in resp.json()

    def test_reload_unknown_model_400(self, client):
        resp = client.post("/v1/admin/models/nope/reload?warm=true")
        assert resp.status_code == 400

    def test_reload_pinned_model_409s_with_the_reason(self, client, mock_router, monkeypatch):
        # A pinned model (RLM job / j-space analysis) is a CONFLICT the caller
        # can act on -- previously the RuntimeError escaped as an opaque 500.
        def _pinned(model_id, force=False):
            raise RuntimeError(f"Model '{model_id}' is pinned (batch job in progress).")
        monkeypatch.setattr(mock_router, "unload_model", _pinned)
        resp = client.post("/v1/admin/models/test-mlx-model/reload")
        assert resp.status_code == 409
        assert "pinned" in resp.json()["detail"]

    def test_reload_refuses_while_a_load_is_in_flight(self, client, mock_router, monkeypatch):
        # unload_model cannot see an in-flight load (the provider isn't
        # published yet), so without this refusal the route would silently
        # JOIN the old-config load and report a reload that never happened.
        monkeypatch.setattr(mock_router, "is_loading",
                            lambda mid: mid == "test-mlx-model")
        resp = client.post("/v1/admin/models/test-mlx-model/reload")
        assert resp.status_code == 409
        assert "in flight" in resp.json()["detail"]

    def test_unload_pinned_model_409s_too(self, client, mock_router, monkeypatch):
        # Ride-along: /unload shared the raw-500 mechanism.
        def _pinned(model_id, force=False):
            raise RuntimeError(f"Model '{model_id}' is pinned (batch job in progress).")
        monkeypatch.setattr(mock_router, "unload_model", _pinned)
        resp = client.post("/v1/admin/models/test-mlx-model/unload")
        assert resp.status_code == 409
        assert "pinned" in resp.json()["detail"]
