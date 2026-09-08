# tests/contract/test_models_endpoint.py
#
# Contract tests for GET /v1/models (OpenAI-compatible model listing).

import pytest
from .conftest import TEST_MODEL_IDS


class TestListModels:
    """Tests for GET /v1/models."""

    def test_returns_model_list(self, client):
        """GET /v1/models returns a list with expected structure."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200

        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == len(TEST_MODEL_IDS)

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
        assert ids == set(TEST_MODEL_IDS)

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


class TestSamplerDefaultsOnModelRows:
    """v2.0.21: every row carries `sampler_defaults`, what each sampler key
    resolves to for a request that says nothing.

    Through the ROUTE, deliberately. `sampler_defaults()` has unit tests, but
    a unit test of the reporter passes whether or not any route binds it --
    the exact shape this repo has shipped green twice (a guard on a model no
    wire produces, a rename guard dead since v1.79.66). The settings panel
    reads this off the response, so the response is what has to be pinned.
    """

    def test_present_and_keyed_by_thinking_on_every_entry(self, client):
        data = client.get("/v1/models").json()["data"]
        assert data
        for entry in data:
            bags = entry["sampler_defaults"]
            assert set(bags) == {"off", "on"}, entry["id"]
            # The panel prints these as placeholders; a non-scalar would
            # render as "[object Object]" rather than a number.
            for bag in bags.values():
                assert bag, entry["id"]
                for key, value in bag.items():
                    assert isinstance(value, (int, float, bool, str)), (entry["id"], key)

    def test_every_reported_key_is_one_a_request_accepts(self, client):
        # A key here that no request takes would drive a control that cannot
        # do anything. Derived from the shared tuple, never a second list.
        from heylook_llm.samplers import REQUEST_SAMPLER_FIELDS

        for entry in client.get("/v1/models").json()["data"]:
            for bag in entry["sampler_defaults"].values():
                assert set(bag) <= set(REQUEST_SAMPLER_FIELDS), entry["id"]

    def test_the_thinking_switch_is_the_key_it_claims_to_be(self, client):
        # The whole reason the field is keyed: each half must report its own
        # side of the switch, or the panel labels a control with the other
        # state's numbers.
        for entry in client.get("/v1/models").json()["data"]:
            bags = entry["sampler_defaults"]
            assert bags["off"]["enable_thinking"] is False, entry["id"]
            assert bags["on"]["enable_thinking"] is True, entry["id"]

    def test_agrees_with_the_admin_row(self, client):
        # Two row builders derived these separately once and drifted; one
        # derivation feeds both, so equality is the assertion that keeps it.
        listed = {m["id"]: m["sampler_defaults"]
                  for m in client.get("/v1/models").json()["data"]}
        admin = {m["id"]: m["sampler_defaults"]
                 for m in client.get("/v1/admin/models").json()["models"]}
        for mid, value in listed.items():
            assert admin.get(mid) == value, mid


class TestContextLengthOnModelRows:
    """v1.79.65: every /v1/models entry carries `context_length` from the ONE
    resolver the admin row and the provider's over-length guard read -- an
    int, or null when the model files do not say (every fake path here)."""

    def test_present_on_every_entry(self, client):
        data = client.get("/v1/models").json()["data"]
        assert data
        for entry in data:
            assert "context_length" in entry, entry["id"]
            assert entry["context_length"] is None or isinstance(entry["context_length"], int)

    def test_agrees_with_the_admin_row(self, client):
        listed = {m["id"]: m["context_length"] for m in client.get("/v1/models").json()["data"]}
        admin = {m["id"]: m["context_length"] for m in client.get("/v1/admin/models").json()["models"]}
        for mid, value in listed.items():
            assert admin.get(mid) == value, mid
