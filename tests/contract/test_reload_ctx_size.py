"""Contract: ``POST /v1/admin/models/{id}/reload?ctx_size=N`` and the admin
row's context fields -- the backend half of choosing a context size at load.

Claims (what breaks if a test is deleted):
- gguf-only: the parameter on an MLX model is a 400 naming the reason, not a
  silent no-op that then reloads the model for nothing.
- ONE writer: the value lands through ``ModelService.update_config`` (the
  models.toml write a PATCH makes), never through a second store the models
  page cannot see.
- 0 = Auto = unset: the stored key is dropped, not stored as 0 -- absent is
  how a default is spelled in models.toml, and ``GGUFModelConfig.ctx_size``
  refuses values under 512 anyway.
- no gratuitous restart: the same value on a resident, non-stale model is a
  plain load (the provider object survives), so a Load press with the same
  choice does not throw away a warm process.
- the row carries ``context_length`` / ``context_running`` for every model
  (null where unknown) so a client can key on the field, not on its absence.
"""

import pytest


@pytest.fixture(autouse=True)
def _unload_after(client, mock_service):
    yield
    client.post("/v1/admin/models/test-gguf-model/unload")
    client.post("/v1/admin/models/test-mlx-model/unload")
    # Leave the roster as this file found it: later files read the gguf
    # row's stored config.
    mock_service.update_config("test-gguf-model", {"config": {"ctx_size": None}})
    mock_service.update_calls.clear()


class TestReloadCtxSize:
    def test_mlx_model_is_400(self, client, mock_service):
        resp = client.post("/v1/admin/models/test-mlx-model/reload?ctx_size=32768")
        assert resp.status_code == 400
        assert "gguf" in resp.json()["detail"]
        assert mock_service.update_calls == []

    def test_persists_through_the_config_writer_then_loads(self, client, mock_service, mock_router):
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=32768&warm=true")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "loaded"
        assert resp.json()["warmed"] is True
        assert mock_service.update_calls == [
            ("test-gguf-model", {"config": {"ctx_size": 32768}}),
        ]
        stored = mock_service.get_config("test-gguf-model").config.model_dump(exclude_unset=True)
        assert stored["ctx_size"] == 32768
        assert "test-gguf-model" in mock_router.providers

    def test_same_value_on_resident_model_does_not_restart(self, client, mock_service, mock_router):
        client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=32768")
        before = mock_router.providers["test-gguf-model"]
        mock_service.update_calls.clear()
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=32768")
        assert resp.status_code == 200
        assert mock_service.update_calls == []
        assert mock_router.providers["test-gguf-model"] is before

    def test_changed_value_on_resident_model_restarts(self, client, mock_router):
        client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=32768")
        before = mock_router.providers["test-gguf-model"]
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=65536")
        assert resp.status_code == 200
        assert mock_router.providers["test-gguf-model"] is not before

    def test_zero_means_auto_and_unsets(self, client, mock_service):
        client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=32768")
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=0")
        assert resp.status_code == 200
        assert mock_service.update_calls[-1] == (
            "test-gguf-model", {"config": {"ctx_size": None}})
        stored = mock_service.get_config("test-gguf-model").config.model_dump(exclude_unset=True)
        assert "ctx_size" not in stored

    def test_zero_when_already_auto_writes_nothing(self, client, mock_service):
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=0")
        assert resp.status_code == 200
        assert mock_service.update_calls == []

    def test_below_the_field_minimum_is_400(self, client, mock_service):
        """The Query accepts any non-negative int; the config class (ge=512)
        is the authority, and its refusal must surface as a bad request."""
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=100")
        assert resp.status_code == 400
        assert "ctx_size" in resp.json()["detail"]

    def test_negative_is_422(self, client):
        resp = client.post("/v1/admin/models/test-gguf-model/reload?ctx_size=-1")
        assert resp.status_code == 422

    def test_reload_without_the_param_is_unchanged(self, client, mock_service, mock_router):
        resp = client.post("/v1/admin/models/test-gguf-model/reload")
        assert resp.status_code == 200
        assert mock_service.update_calls == []


class TestAdminRowContextFields:
    def test_fields_present_on_every_row(self, client):
        rows = {m["id"]: m for m in client.get("/v1/admin/models").json()["models"]}
        for row in rows.values():
            context = row["engine"]["context"]
            assert set(context) == {"length", "running"}
        # The fake gguf path does not exist: header unreadable -> null, not an
        # error, and not a listing failure.
        assert rows["test-gguf-model"]["engine"]["context"]["length"]["value"] is None
        assert rows["test-mlx-model"]["engine"]["context"]["length"]["value"] is None
        # MLX has no fixed allocation: not applicable, which is not unknown.
        assert rows["test-mlx-model"]["engine"]["context"]["running"]["provenance"] == "not_applicable"

    def test_running_is_null_until_a_provider_reports_one(self, client, mock_router):
        """The running context reaches the row through the REAL gguf observed
        half (LlamaServerProvider.describe_observed), not a stand-in."""
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider

        provider = LlamaServerProvider.__new__(LlamaServerProvider)
        provider.model_id = "test-gguf-model"
        provider.config = {"model_path": "/fake/model.gguf"}
        provider.running_ctx = None
        provider.loaded_chat_template = None
        mock_router.providers["test-gguf-model"] = provider

        def running():
            row = client.get("/v1/admin/models/test-gguf-model").json()
            return row["engine"]["context"]["running"]

        assert running()["value"] is None
        provider.running_ctx = 32768
        assert (running()["value"], running()["provenance"]) == (32768, "observed")
