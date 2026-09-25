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


@pytest.fixture
def heylook_toml(app, tmp_path, monkeypatch):
    """A REAL ModelService over a throwaway heylook.toml, standing in for the
    session's mock service for one test: what the route wrote is then read
    back from the file and through GET, not from a record of calls."""
    from heylook_llm.model_service import ModelService

    path = tmp_path / "heylook.toml"
    path.write_text(
        "# the owner's file\n"
        "[[models]]\n"
        'id = "test-gguf-model"\n'
        'provider = "gguf"\n'
        "enabled = true\n\n"
        "[models.config]\n"
        'model_path = "/fake/model.gguf"\n'
    )
    monkeypatch.setattr(app.state, "model_service", ModelService(str(path)))
    return path


def _row_config(client):
    return client.get("/v1/admin/models/test-gguf-model").json()["config"]


class TestReloadLoadSettings:
    def test_a_field_the_provider_does_not_offer_is_400(self, client, mock_service):
        resp = client.post("/v1/admin/models/test-mlx-model/reload", json={"ctx_size": 32768})
        assert resp.status_code == 400
        assert "not a load setting" in resp.json()["detail"]
        resp = client.post(GGUF, json={"n_gpu_layers": 10})
        assert resp.status_code == 400
        assert "ctx_size" in resp.json()["detail"]  # names what it does take
        assert mock_service.update_calls == []

    def test_persists_through_the_config_writer_then_loads(self, client, heylook_toml, mock_router):
        from heylook_llm.model_service import ModelService

        resp = client.post(GGUF + "?warm=true", json={"ctx_size": 32768, "flash_attn": "off"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "loaded"
        assert resp.json()["warmed"] is True
        row = _row_config(client)
        assert (row["ctx_size"], row["flash_attn"]) == (32768, "off")
        # In the file the one writer owns: a fresh service reads it back.
        reread = ModelService(str(heylook_toml)).get_config("test-gguf-model")
        assert reread is not None
        assert (reread.config.ctx_size, reread.config.flash_attn) == (32768, "off")
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

    def test_null_means_auto_and_unsets(self, client, heylook_toml):
        client.post(GGUF, json={"ctx_size": 32768, "flash_attn": "on"})
        assert "ctx_size" in heylook_toml.read_text()
        resp = client.post(GGUF, json={"ctx_size": None, "flash_attn": None})
        assert resp.status_code == 200
        assert not {"ctx_size", "flash_attn"} & set(_row_config(client))
        text = heylook_toml.read_text()
        assert "ctx_size" not in text and "flash_attn" not in text

    @pytest.mark.parametrize("body", [
        pytest.param({"ctx_size": None}, id="null_when_already_auto_writes_nothing"),
        pytest.param(None, id="reload_without_a_body_writes_nothing"),
    ])
    def test_nothing_to_change_leaves_the_file_alone(self, client, heylook_toml, body):
        before = (heylook_toml.read_bytes(), heylook_toml.stat().st_mtime_ns)
        resp = client.post(GGUF, json=body) if body is not None else client.post(GGUF)
        assert resp.status_code == 200
        assert (heylook_toml.read_bytes(), heylook_toml.stat().st_mtime_ns) == before
        assert not heylook_toml.with_suffix(".toml.bak").exists()

    @pytest.mark.parametrize("body", [{"ctx_size": 100}, {"flash_attn": "auto"}])
    def test_the_config_class_is_the_authority_on_values(self, client, body):
        """The route checks names only; the field's own type refuses a bad
        value (ctx_size ge=512; "auto" is spelled null, never stored)."""
        resp = client.post(GGUF, json=body)
        assert resp.status_code == 400
        assert next(iter(body)) in resp.json()["detail"]


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
