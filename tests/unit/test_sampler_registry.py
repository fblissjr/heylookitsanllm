"""Tests for the runtime preset registry (C1 of S1.2b-onwards).

Presets live at ``src/heylook_llm/data/presets/*.toml``. Each preset file has
a ``[meta]`` section (name, description) and a ``[defaults]`` section with the
sampler knobs. Presets are referenced at request time via
``ChatRequest.preset`` — unlike the prior profile system which baked values
into ``models.toml`` at import time, the registry applies preset fields
dynamically at request resolution.

Cascade:

1. Global hardcoded floor (`_apply_model_defaults`)
2. Model sampler fields (from `models.toml`, if set)
3. Request's preset fields (if `request.preset` is set; looked up here)
4. Request-level explicit field values

Each layer overrides the previous for fields it sets; unset fields pass
through.
"""

from __future__ import annotations

import pytest

from heylook_llm.presets import (
    PresetNotFound,
    PresetRegistry,
    get_preset_registry,
)


class TestPresetRegistry:
    def test_load_bundled_presets(self):
        """The bundled roster: mechanism presets + the import-default 'balanced'.

        Exact-set assertion on purpose -- flavor presets with no consumer
        (moderate/code/creative) were removed 2026-07-20; a preset added here
        should come with a consumer.
        """
        registry = PresetRegistry.from_bundled()
        assert set(registry.list_names()) == {
            "balanced", "deterministic", "thinking", "vlm-describe", "vlm-extract",
        }
        assert registry.get("balanced")["temperature"] == pytest.approx(0.7)

    def test_list_names_is_sorted(self):
        registry = PresetRegistry.from_bundled()
        names = registry.list_names()
        assert names == sorted(names)
        assert len(names) >= 1

    def test_unknown_preset_raises(self):
        registry = PresetRegistry.from_bundled()
        with pytest.raises(PresetNotFound):
            registry.get("this-preset-does-not-exist")

    def test_empty_registry(self, tmp_path):
        """A registry pointed at an empty dir should produce no presets."""
        registry = PresetRegistry.from_directory(tmp_path)
        assert registry.list_names() == []
        assert "balanced" not in registry

    def test_custom_preset_loaded(self, tmp_path):
        """A TOML with [meta] + [defaults] produces a loadable preset."""
        (tmp_path / "spicy.toml").write_text(
            '[meta]\nname = "spicy"\ndescription = "high temp"\n'
            "[defaults]\ntemperature = 1.1\ntop_p = 0.95\n"
        )
        registry = PresetRegistry.from_directory(tmp_path)
        assert "spicy" in registry
        assert registry.get("spicy") == {"temperature": 1.1, "top_p": 0.95}

    def test_meta_name_overrides_filename(self, tmp_path):
        """If [meta].name is set, it wins over the filename."""
        (tmp_path / "aaa.toml").write_text(
            '[meta]\nname = "zzz"\n[defaults]\ntemperature = 0.2\n'
        )
        registry = PresetRegistry.from_directory(tmp_path)
        assert "zzz" in registry
        assert "aaa" not in registry

    def test_missing_defaults_section_is_empty_preset(self, tmp_path):
        """[meta]-only file resolves to an empty default dict (no-op preset)."""
        (tmp_path / "stub.toml").write_text('[meta]\nname = "stub"\n')
        registry = PresetRegistry.from_directory(tmp_path)
        assert registry.get("stub") == {}

    def test_malformed_toml_skipped_not_fatal(self, tmp_path, caplog):
        """A broken TOML file should be logged and skipped, not crash startup."""
        (tmp_path / "broken.toml").write_text("this is = not [valid toml")
        (tmp_path / "good.toml").write_text(
            '[meta]\nname = "good"\n[defaults]\ntemperature = 0.5\n'
        )
        registry = PresetRegistry.from_directory(tmp_path)
        assert "good" in registry
        assert "broken" not in registry

    def test_global_singleton_is_memoized(self):
        a = get_preset_registry()
        b = get_preset_registry()
        assert a is b


class TestApplyPreset:
    """``apply_preset`` is the cascade helper: it mutates a merged_config dict
    in place, overlaying preset fields for keys the preset sets, leaving the
    rest untouched. It's called between the model-level and request-level
    layers in ``_apply_model_defaults``.
    """

    def test_apply_overlays_keys_the_preset_sets(self, tmp_path):
        (tmp_path / "spicy.toml").write_text(
            '[meta]\nname = "spicy"\n'
            "[defaults]\ntemperature = 1.1\ntop_p = 0.95\n"
        )
        registry = PresetRegistry.from_directory(tmp_path)

        merged = {"temperature": 0.7, "top_k": 40, "max_tokens": 512}
        registry.apply_preset(merged, "spicy")

        assert merged == {
            "temperature": 1.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 512,
        }

    def test_apply_unknown_preset_raises(self, tmp_path):
        registry = PresetRegistry.from_directory(tmp_path)
        merged = {"temperature": 0.7}
        with pytest.raises(PresetNotFound):
            registry.apply_preset(merged, "nope")

    def test_apply_none_is_noop(self, tmp_path):
        """``apply_preset(..., None)`` should be a no-op -- the cascade calls
        this unconditionally; the None-guard lives in the helper so callers
        don't need an if/else around every call site."""
        registry = PresetRegistry.from_directory(tmp_path)
        merged = {"temperature": 0.7}
        registry.apply_preset(merged, None)
        assert merged == {"temperature": 0.7}
