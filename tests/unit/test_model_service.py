"""Unit tests for model service after C4 preset migration.

Historical context: the v1.19.0 profile system baked sampler fields into
``models.toml`` at import time. C4 deleted that behavior -- sampler fields
now live in the runtime preset registry and are applied at request time
via the cascade. ``ModelProfile.apply()`` now just records
``default_preset`` on the model's config; ``get_smart_defaults()`` emits
only load-time fields (cache type, KV quantization, draft tokens).

Preset-registry semantics are covered by ``test_preset_registry.py``.
This file focuses on:
  - The thin ``ModelProfile`` adapter's new set-``default_preset`` behavior
  - ``get_smart_defaults`` returning only load-time fields
  - ``ModelImporter`` model-size regex and embedding detection (unchanged)
"""

import json
from pathlib import Path

import pytest

from heylook_llm.model_service import (
    ModelProfile,
    get_smart_defaults,
    load_profiles,
)
from heylook_llm.model_importer import ModelImporter


class TestModelProfileAdapter:
    """``ModelProfile.apply`` now sets ``default_preset`` only."""

    def _make_profile(self, name="moderate", defaults=None):
        return ModelProfile(
            name=name,
            description="test",
            defaults=defaults or {},
        )

    def test_apply_sets_default_preset_on_mlx(self):
        profile = self._make_profile(name="creative")
        config = {"model_path": "/p", "vision": False, "cache_type": "standard"}

        result = profile.apply(config, {"provider": "mlx"})

        assert result["default_preset"] == "creative"
        # Load-time fields preserved.
        assert result["model_path"] == "/p"
        assert result["cache_type"] == "standard"
        # No sampler-field baking.
        assert "temperature" not in result
        assert "top_k" not in result

    def test_apply_skips_non_mlx_provider(self):
        """Embedding provider doesn't use sampler presets; no default_preset."""
        profile = self._make_profile(name="embedding")
        config = {"model_path": "/p", "max_length": 2048}

        result = profile.apply(config, {"provider": "mlx_embedding"})

        assert "default_preset" not in result
        assert result["max_length"] == 2048

    def test_apply_does_not_mutate_input(self):
        profile = self._make_profile(name="creative")
        config = {"model_path": "/p"}

        profile.apply(config, {"provider": "mlx"})

        assert "default_preset" not in config


class TestLoadProfilesAdapter:
    """``load_profiles`` is now a thin adapter over ``PresetRegistry``."""

    def test_returns_adapter_for_each_bundled_preset(self):
        profiles = load_profiles()
        assert "moderate" in profiles
        assert "creative" in profiles
        for name, profile in profiles.items():
            assert profile.name == name
            assert isinstance(profile.defaults, dict)


class TestSmartDefaultsLoadTimeOnly:
    """After C4, sampler fields (temperature, top_k, etc.) are NEVER in
    ``get_smart_defaults``. Only cache/KV/draft-token load-time config."""

    def test_no_sampler_fields(self):
        defaults = get_smart_defaults({
            "provider": "mlx", "name": "test", "size_gb": 7,
            "is_vision": False,
        })
        for forbidden in (
            "temperature", "top_p", "top_k", "min_p",
            "max_tokens", "repetition_penalty", "repetition_context_size",
        ):
            assert forbidden not in defaults, (
                f"sampler field {forbidden!r} leaked into get_smart_defaults"
            )

    def test_model_large_relative_to_ram_gets_quantized_cache(self, monkeypatch):
        # 40GB weights on a 64GB machine: real memory pressure -> quantize.
        monkeypatch.setattr("heylook_llm.model_service._system_ram_gb", lambda: 64.0)
        defaults = get_smart_defaults({
            "provider": "mlx", "name": "big", "size_gb": 40,
        })
        assert defaults["cache_type"] == "quantized"
        assert defaults["kv_bits"] == 8

    def test_same_model_on_big_ram_machine_gets_standard_cache(self, monkeypatch):
        # The SAME 40GB model on a 192GB machine: no pressure -> fp16 KV.
        # KV quantization is a memory trade-off, not a free default.
        monkeypatch.setattr("heylook_llm.model_service._system_ram_gb", lambda: 192.0)
        defaults = get_smart_defaults({
            "provider": "mlx", "name": "big", "size_gb": 40,
        })
        assert defaults["cache_type"] == "standard"

    def test_small_model_gets_standard_cache(self, monkeypatch):
        monkeypatch.setattr("heylook_llm.model_service._system_ram_gb", lambda: 64.0)
        defaults = get_smart_defaults({
            "provider": "mlx", "name": "small", "size_gb": 3,
        })
        assert defaults["cache_type"] == "standard"

    def test_max_kv_size_is_never_a_default(self, monkeypatch):
        # max_kv_size is a RotatingKVCache cap that SILENTLY DROPS context
        # beyond it -- truncation must be an explicit user choice, never an
        # import-time default (it shipped 2048 on every >30GB model once).
        for ram, size in ((64.0, 40), (192.0, 155), (32.0, 20)):
            monkeypatch.setattr("heylook_llm.model_service._system_ram_gb", lambda r=ram: r)
            defaults = get_smart_defaults({
                "provider": "mlx", "name": "m", "size_gb": size,
            })
            assert "max_kv_size" not in defaults

    def test_mlx_embedding_returns_max_length_only(self):
        defaults = get_smart_defaults({"provider": "mlx_embedding", "name": "e"})
        assert defaults == {"max_length": 2048}


class TestModelSizeRegex:
    """_get_model_size returns (param-count LABEL from the name, real weight
    GB from safetensors bytes). The old code returned "7B" name matches as
    size_gb=7.0 -- billions of params masquerading as gigabytes -- and fed
    that to get_smart_defaults' RAM-relative threshold (audit 2026-07-06)."""

    def _get_size(self, path_str: str):
        importer = ModelImporter()
        return importer._get_model_size(Path(path_str))

    def test_label_parsed_but_gb_none_without_files(self):
        # Name gives the label; GB must come from real bytes, which a
        # non-existent path doesn't have.
        label, gb = self._get_size("/models/Llama-3.1-8B")
        assert label == "8B"
        assert gb is None

    def test_decimal_label(self):
        label, gb = self._get_size("/models/Qwen3-0.6B")
        assert label == "0.6B"
        assert gb is None

    def test_million_label(self):
        label, gb = self._get_size("/models/SmolLM-135M")
        assert label == "135M"
        assert gb is None

    def test_no_size_marker(self):
        label, gb = self._get_size("/models/Phi-3-mini-128k")
        assert label is None
        assert gb is None

    def test_gb_comes_from_safetensors_bytes_not_name(self, tmp_path):
        # A "7B" name over 2 GiB of actual weights: label says 7B,
        # size_gb says 2.0 -- never 7.0.
        model_dir = tmp_path / "Fake-7B-instruct"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\0" * (2 * 1024 ** 2))
        importer = ModelImporter()
        label, gb = importer._get_model_size(model_dir)
        assert label == "7B"
        assert gb == pytest.approx(2 * 1024 ** 2 / 1024 ** 3)


class TestImportWizardChatTemplateDetection:
    """Import wizard records ``chat_template_source = "jinja"`` when the
    model folder ships a ``chat_template.jinja``. CLI ``--chat-template``
    overrides this detection. Both are wired so models.toml reflects the
    policy explicitly instead of relying on HF's version-dependent auto-
    detection."""

    def _make_mlx_dir(self, tmp_path, *, with_jinja=False):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)
        if with_jinja:
            (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
        return tmp_path

    def test_auto_detects_jinja_from_folder(self, tmp_path):
        model_dir = tmp_path / "some-model"
        model_dir.mkdir()
        self._make_mlx_dir(model_dir, with_jinja=True)
        importer = ModelImporter()

        models = importer.scan_directory(str(tmp_path))

        assert len(models) == 1
        assert models[0]["config"].get("chat_template_source") == "jinja"

    def test_no_jinja_in_folder_leaves_source_unset(self, tmp_path):
        model_dir = tmp_path / "some-model"
        model_dir.mkdir()
        self._make_mlx_dir(model_dir, with_jinja=False)
        importer = ModelImporter()

        models = importer.scan_directory(str(tmp_path))

        assert len(models) == 1
        assert "chat_template_source" not in models[0]["config"]

    def test_cli_override_wins_over_folder_detection(self, tmp_path):
        """CLI ``--chat-template /custom/path.jinja`` overrides the auto-
        detected 'jinja' even when the folder has its own chat_template.jinja."""
        model_dir = tmp_path / "some-model"
        model_dir.mkdir()
        self._make_mlx_dir(model_dir, with_jinja=True)
        importer = ModelImporter(chat_template_override="tokenizer_config")

        models = importer.scan_directory(str(tmp_path))

        assert models[0]["config"]["chat_template_source"] == "tokenizer_config"


class TestEmbeddingModelDetection:
    """Embedding model detection is unchanged by C4."""

    def _make_embedding_dir(self, tmp_path, *, bidirectional=True, dense_dirs=False):
        config = {"model_type": "gemma2", "hidden_size": 768}
        if bidirectional:
            config["use_bidirectional_attention"] = True
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)
        if dense_dirs:
            (tmp_path / "2_Dense").mkdir()
            (tmp_path / "3_Dense").mkdir()
        return tmp_path

    def _make_generative_dir(self, tmp_path):
        config = {"model_type": "llama", "hidden_size": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)
        return tmp_path

    def test_detects_embedding_from_bidirectional_config(self, tmp_path):
        self._make_embedding_dir(tmp_path, bidirectional=True)
        importer = ModelImporter()
        assert importer._is_embedding_model(tmp_path) is True

    def test_detects_embedding_from_dense_dirs(self, tmp_path):
        self._make_embedding_dir(tmp_path, bidirectional=False, dense_dirs=True)
        importer = ModelImporter()
        assert importer._is_embedding_model(tmp_path) is True

    def test_rejects_generative_model(self, tmp_path):
        self._make_generative_dir(tmp_path)
        importer = ModelImporter()
        assert importer._is_embedding_model(tmp_path) is False

    def test_create_embedding_entry_sets_provider(self, tmp_path):
        self._make_embedding_dir(tmp_path, bidirectional=True)
        importer = ModelImporter()
        entry = importer._create_embedding_entry(tmp_path)

        assert entry is not None
        assert entry["provider"] == "mlx_embedding"
        assert entry["config"]["model_path"] == str(tmp_path)
        assert entry["config"]["max_length"] == 2048
        assert "temperature" not in entry["config"]
        assert "embedding" in entry["tags"]

    def test_scan_finds_embedding_model(self, tmp_path):
        model_dir = tmp_path / "embeddinggemma-300m"
        model_dir.mkdir()
        self._make_embedding_dir(model_dir, bidirectional=True)

        importer = ModelImporter()
        models = importer.scan_directory(str(tmp_path))

        assert len(models) == 1
        assert models[0]["provider"] == "mlx_embedding"
        assert "embedding" in models[0]["tags"]
