"""Unit tests for model service: the import wizard's chat-template detection
and model-size handling. (The load-time smart defaults it used to emit --
MLX KV cache settings -- were retired with the mlx-vlm engine, plan W10
stage 3.)"""

import json

import pytest

from heylook_llm.model_importer import ModelImporter


class TestImportWizardChatTemplateDetection:
    """Derive-at-load (Wave 1 / 6a, 2026-07-28): the importer NO LONGER
    materializes the auto-detected chat_template_source -- load-time auto
    resolution (template_info.py: jinja > embedded > chat_template.json)
    already applies the same policy deterministically, so recording it was
    a copy that rots. Only an explicit CLI ``--chat-template`` override is
    written (operator intent)."""

    def _make_mlx_dir(self, tmp_path, *, with_jinja=False):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)
        if with_jinja:
            (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
        return tmp_path

    # An imported MLX entry carries only model_path + operator intent: no
    # chat_template_source (whether or not a jinja sits in the folder), no
    # modalities/vision (the config validator detects at load), no auto
    # description/tags.
    @pytest.mark.parametrize("dirname, config, with_jinja", [
        pytest.param("some-model", {"model_type": "llama"}, True,
                     id="jinja_in_folder_is_not_materialized"),
        pytest.param("some-vision-model", {"model_type": "gemma4", "vision_config": {}}, False,
                     id="mlx_entry_is_thin"),
        pytest.param("some-model", {"model_type": "llama"}, False,
                     id="no_jinja_in_folder_leaves_source_unset"),
    ])
    def test_imported_mlx_entry_is_thin(self, tmp_path, dirname, config, with_jinja):
        model_dir = tmp_path / dirname
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 64)
        if with_jinja:
            (model_dir / "chat_template.jinja").write_text("{{ messages }}")

        models = ModelImporter().scan_directory(str(tmp_path))

        assert len(models) == 1
        entry = models[0]
        for key in ("chat_template_source", "modalities", "vision"):
            assert key not in entry["config"], key
        for key in ("description", "tags"):
            assert key not in entry, key
        assert entry["config"]["model_path"] == str(model_dir)

    def test_embedding_checkpoint_is_skipped_not_served_as_chat(self, tmp_path):
        # v2.0.41 removed the embedding provider; without a skip these fell
        # through to the mlx detector and imported as an ENABLED chat entry.
        for name, marker in (("bidir", "config"), ("dense", "dir")):
            d = tmp_path / name
            d.mkdir()
            cfg = {"model_type": "gemma2", "hidden_size": 768}
            if marker == "config":
                cfg["use_bidirectional_attention"] = True
            (d / "config.json").write_text(json.dumps(cfg))
            (d / "model.safetensors").write_bytes(b"\x00" * 64)
            if marker == "dir":
                (d / "2_Dense").mkdir()
        (tmp_path / "chat").mkdir()  # a real chat model beside them
        self._make_mlx_dir(tmp_path / "chat")

        models = ModelImporter().scan_directory(str(tmp_path))

        assert [m["id"] for m in models] == ["chat"]
