# tests/contract/test_models_endpoint.py
#
# Contract tests for GET /v1/models (OpenAI-compatible model listing).

import pytest
from .conftest import TEST_MODEL_IDS


class TestListModels:
    """Tests for GET /v1/models."""

    def test_returns_model_list(self, client):
        """GET /v1/models returns the OpenAI list shape: one entry per
        configured model (ids == the test roster), each with id, object and
        owned_by, carrying its provider."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200

        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == len(TEST_MODEL_IDS)

        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "owned_by" in model

        assert {m["id"] for m in data["data"]} == set(TEST_MODEL_IDS)

        providers = {m["id"]: m.get("provider") for m in data["data"]}
        assert providers["test-mlx-model"] == "mlx"


class TestThinkingDefaultOnModelRows:
    """v1.79.62/63: every model row carries `thinking_default`, the cascade's
    answer for an empty request -- derived, so answered for unloaded models --
    and the value a UI's 'model default' choice actually means. Both lists
    report it, so a page reading either labels 'model default' the same.

    Rows: (route, list key, whether every row must carry `capabilities`; the
    /v1/models row omits the key when the list is empty)."""

    @pytest.mark.parametrize("route, key, caps_required", [
        pytest.param("/v1/models", "data", False, id="present_and_bool_on_every_entry"),
        pytest.param("/v1/admin/models", "models", True,
                     id="admin_thinking_default_is_on_every_row_and_a_bool"),
    ])
    def test_present_and_bool_on_every_entry(self, client, route, key, caps_required):
        rows = client.get(route).json()[key]
        assert rows
        for row in rows:
            assert isinstance(row["thinking_default"], bool)
            caps = row["capabilities"] if caps_required else row.get("capabilities", [])
            # A model without the thinking capability cannot default to on.
            if "thinking" not in caps:
                assert row["thinking_default"] is False


class TestRowsAgreeWithTheAdminRow:
    """/v1/models and /v1/admin/models report the same value per id. Two row
    builders derived these separately once and drifted; one derivation
    (capabilities.derived_model_facts) feeds both, so equality is the
    assertion that keeps it. `engine` is compared whole, not just the context
    ceiling."""

    @pytest.mark.parametrize("field", [
        pytest.param("thinking_default", id="thinking_default_agrees_with_the_admin_row"),
        pytest.param("sampler_defaults", id="sampler_defaults_agrees_with_the_admin_row"),
        pytest.param("engine", id="engine_agrees_with_the_admin_row"),
    ])
    def test_agrees_with_the_admin_row(self, client, field):
        listed = {m["id"]: m[field] for m in client.get("/v1/models").json()["data"]}
        admin = {m["id"]: m[field] for m in client.get("/v1/admin/models").json()["models"]}
        assert listed
        for mid, value in listed.items():
            assert admin.get(mid) == value, mid


class TestSamplerDefaultsOnModelRows:
    """v2.0.21: every row carries `sampler_defaults`, what each sampler key
    resolves to for a request that says nothing.

    Through the ROUTE, deliberately. `sampler_defaults()` has unit tests, but
    a unit test of the reporter passes whether or not any route binds it --
    the exact shape this repo has shipped green twice (a guard on a model no
    wire produces, a rename guard dead since v1.79.66). The settings panel
    reads this off the response, so the response is what has to be pinned.
    """

    def test_present_and_flat_on_every_entry(self, client):
        data = client.get("/v1/models").json()["data"]
        assert data
        for entry in data:
            bag = entry["sampler_defaults"]
            assert bag, entry["id"]
            # ONE bag since v2.0.33. It was {"off","on"} while the anti-loop
            # overlay moved a sampler value off the thinking switch; with that
            # gone the halves were identical but for `enable_thinking`, so the
            # nesting reported a distinction the cascade no longer makes.
            assert not (set(bag) & {"off", "on"}), \
                f"{entry['id']}: sampler_defaults is nested again"
            # The panel prints these as placeholders; a non-scalar would
            # render as "[object Object]" rather than a number.
            for key, value in bag.items():
                assert isinstance(value, (int, float, bool, str)), (entry["id"], key)

    def test_every_reported_key_is_one_a_request_accepts(self, client):
        # A key here that no request takes would drive a control that cannot
        # do anything. Derived from the shared tuple, never a second list.
        from heylook_llm.samplers import REQUEST_SAMPLER_FIELDS

        for entry in client.get("/v1/models").json()["data"]:
            assert set(entry["sampler_defaults"]) <= set(REQUEST_SAMPLER_FIELDS), entry["id"]

    def test_the_reported_thinking_value_matches_the_rows_own_field(self, client):
        # `thinking_default` is the field that OWNS this answer; the bag
        # carries it too because it is a request sampler field. One cascade
        # call feeds both now, so a disagreement means something re-derived it.
        for entry in client.get("/v1/models").json()["data"]:
            if "thinking_default" not in entry:
                continue
            assert entry["sampler_defaults"]["enable_thinking"] == entry["thinking_default"], \
                entry["id"]


class TestContextLengthOnModelRows:
    """Every /v1/models entry carries the context ceiling
    (`engine.context.length`) from the ONE resolver the admin row and the
    provider's over-length guard read -- an int, or null when the model files
    do not say (every fake path here)."""

    def test_present_on_every_entry(self, client):
        data = client.get("/v1/models").json()["data"]
        assert data
        for entry in data:
            value = entry["engine"]["context"]["length"]["value"]
            assert value is None or isinstance(value, int), entry["id"]
