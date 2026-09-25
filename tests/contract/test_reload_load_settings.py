"""Contract: ``POST /v1/admin/models/{id}/reload`` with a body of load
settings (plan W1), and the admin row's context fields -- the backend half of
the load panel.

Claims (what breaks if a test is deleted):
- the body takes exactly the provider's ``load_setting`` fields: anything
  else (an MLX model has none) is a 400 naming what the model does take, not
  a silent no-op that then reloads the model for nothing.
- ONE writer: values land through ``ModelService.update_config`` (the write a
  PATCH makes), never through a second store the models page cannot see.
- null = Auto = unset: the stored key is dropped, never written as a value --
  absent is how a default is spelled in the config file.
- no gratuitous restart: the same values on a resident, non-stale model are a
  plain load (the provider object survives), so a Load press with the same
  choices does not throw away a warm process.
- the row carries ``context_length`` / ``context_running`` for every model
  (null where unknown) so a client can key on the field, not on its absence.
"""

import pytest

GGUF = "/v1/admin/models/test-gguf-model/reload"


@pytest.fixture(autouse=True)
def _unload_after(client, mock_service):
    yield
    client.post("/v1/admin/models/test-gguf-model/unload")
    client.post("/v1/admin/models/test-mlx-model/unload")
    # Leave the roster as this file found it: later files read the gguf
    # row's stored config.
    mock_service.update_config("test-gguf-model",
                               {"config": {"ctx_size": None, "flash_attn": None}})
    mock_service.update_calls.clear()


def _stored(mock_service):
    return mock_service.get_config("test-gguf-model").config.model_dump(exclude_unset=True)


class TestReloadLoadSettings:
    def test_a_field_the_provider_does_not_offer_is_400(self, client, mock_service):
        resp = client.post("/v1/admin/models/test-mlx-model/reload", json={"ctx_size": 32768})
        assert resp.status_code == 400
        assert "not a load setting" in resp.json()["detail"]
        resp = client.post(GGUF, json={"n_gpu_layers": 10})
        assert resp.status_code == 400
        assert "ctx_size" in resp.json()["detail"]  # names what it does take
        assert mock_service.update_calls == []

    def test_persists_through_the_config_writer_then_loads(self, client, mock_service, mock_router):
        resp = client.post(GGUF + "?warm=true", json={"ctx_size": 32768, "flash_attn": "off"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "loaded"
        assert resp.json()["warmed"] is True
        assert mock_service.update_calls == [
            ("test-gguf-model", {"config": {"ctx_size": 32768, "flash_attn": "off"}}),
        ]
        stored = _stored(mock_service)
        assert (stored["ctx_size"], stored["flash_attn"]) == (32768, "off")
        assert "test-gguf-model" in mock_router.providers

    def test_same_values_on_resident_model_do_not_restart(self, client, mock_service, mock_router):
        client.post(GGUF, json={"ctx_size": 32768})
        before = mock_router.providers["test-gguf-model"]
        mock_service.update_calls.clear()
        resp = client.post(GGUF, json={"ctx_size": 32768, "flash_attn": None})
        assert resp.status_code == 200
        assert mock_service.update_calls == []
        assert mock_router.providers["test-gguf-model"] is before

    def test_changed_value_on_resident_model_restarts(self, client, mock_router):
        client.post(GGUF, json={"ctx_size": 32768})
        before = mock_router.providers["test-gguf-model"]
        resp = client.post(GGUF, json={"ctx_size": 65536})
        assert resp.status_code == 200
        assert mock_router.providers["test-gguf-model"] is not before

    def test_null_means_auto_and_unsets(self, client, mock_service):
        client.post(GGUF, json={"ctx_size": 32768, "flash_attn": "on"})
        resp = client.post(GGUF, json={"ctx_size": None, "flash_attn": None})
        assert resp.status_code == 200
        assert mock_service.update_calls[-1] == (
            "test-gguf-model", {"config": {"ctx_size": None, "flash_attn": None}})
        assert not {"ctx_size", "flash_attn"} & set(_stored(mock_service))

    def test_null_when_already_auto_writes_nothing(self, client, mock_service):
        resp = client.post(GGUF, json={"ctx_size": None})
        assert resp.status_code == 200
        assert mock_service.update_calls == []

    @pytest.mark.parametrize("body", [{"ctx_size": 100}, {"flash_attn": "auto"}])
    def test_the_config_class_is_the_authority_on_values(self, client, body):
        """The route checks names only; the field's own type refuses a bad
        value (ctx_size ge=512; "auto" is spelled null, never stored)."""
        resp = client.post(GGUF, json=body)
        assert resp.status_code == 400
        assert next(iter(body)) in resp.json()["detail"]

    def test_reload_without_a_body_is_unchanged(self, client, mock_service, mock_router):
        resp = client.post(GGUF)
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
