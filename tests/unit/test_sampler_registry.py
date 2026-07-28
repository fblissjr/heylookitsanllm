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

from heylook_llm.samplers import (
    SamplerNotFound,
    SamplerRegistry,
    get_sampler_registry,
)


class TestSamplerRegistry:
    def test_load_bundled_presets(self):
        """The bundled roster: mechanism presets + the import-default 'balanced'.

        Exact-set assertion on purpose -- flavor presets with no consumer
        (moderate/code/creative) were removed 2026-07-20; a preset added here
        should come with a consumer.
        """
        registry = SamplerRegistry.from_bundled()
        assert set(registry.list_names()) == {
            "balanced", "deterministic", "thinking", "vlm-describe", "vlm-extract",
        }
        assert registry.get("balanced")["temperature"] == pytest.approx(0.7)

    def test_list_names_is_sorted(self):
        registry = SamplerRegistry.from_bundled()
        names = registry.list_names()
        assert names == sorted(names)
        assert len(names) >= 1

    def test_unknown_preset_raises(self):
        registry = SamplerRegistry.from_bundled()
        with pytest.raises(SamplerNotFound):
            registry.get("this-preset-does-not-exist")

    def test_empty_registry(self, tmp_path):
        """A registry pointed at an empty dir should produce no presets."""
        registry = SamplerRegistry.from_directory(tmp_path)
        assert registry.list_names() == []
        assert "balanced" not in registry

    def test_custom_preset_loaded(self, tmp_path):
        """A TOML with [meta] + [defaults] produces a loadable preset."""
        (tmp_path / "spicy.toml").write_text(
            '[meta]\nname = "spicy"\ndescription = "high temp"\n'
            "[defaults]\ntemperature = 1.1\ntop_p = 0.95\n"
        )
        registry = SamplerRegistry.from_directory(tmp_path)
        assert "spicy" in registry
        assert registry.get("spicy") == {"temperature": 1.1, "top_p": 0.95}

    def test_meta_name_overrides_filename(self, tmp_path):
        """If [meta].name is set, it wins over the filename."""
        (tmp_path / "aaa.toml").write_text(
            '[meta]\nname = "zzz"\n[defaults]\ntemperature = 0.2\n'
        )
        registry = SamplerRegistry.from_directory(tmp_path)
        assert "zzz" in registry
        assert "aaa" not in registry

    def test_missing_defaults_section_is_empty_preset(self, tmp_path):
        """[meta]-only file resolves to an empty default dict (no-op preset)."""
        (tmp_path / "stub.toml").write_text('[meta]\nname = "stub"\n')
        registry = SamplerRegistry.from_directory(tmp_path)
        assert registry.get("stub") == {}

    def test_malformed_toml_skipped_not_fatal(self, tmp_path, caplog):
        """A broken TOML file should be logged and skipped, not crash startup."""
        (tmp_path / "broken.toml").write_text("this is = not [valid toml")
        (tmp_path / "good.toml").write_text(
            '[meta]\nname = "good"\n[defaults]\ntemperature = 0.5\n'
        )
        registry = SamplerRegistry.from_directory(tmp_path)
        assert "good" in registry
        assert "broken" not in registry

    def test_global_singleton_is_memoized(self):
        a = get_sampler_registry()
        b = get_sampler_registry()
        assert a is b


class TestApplyPreset:
    """``apply_sampler`` is the cascade helper: it mutates a merged_config dict
    in place, overlaying preset fields for keys the preset sets, leaving the
    rest untouched. It's called between the model-level and request-level
    layers in ``_apply_model_defaults``.
    """

    def test_apply_overlays_keys_the_preset_sets(self, tmp_path):
        (tmp_path / "spicy.toml").write_text(
            '[meta]\nname = "spicy"\n'
            "[defaults]\ntemperature = 1.1\ntop_p = 0.95\n"
        )
        registry = SamplerRegistry.from_directory(tmp_path)

        merged = {"temperature": 0.7, "top_k": 40, "max_tokens": 512}
        registry.apply_sampler(merged, "spicy")

        assert merged == {
            "temperature": 1.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 512,
        }

    def test_apply_unknown_preset_raises(self, tmp_path):
        registry = SamplerRegistry.from_directory(tmp_path)
        merged = {"temperature": 0.7}
        with pytest.raises(SamplerNotFound):
            registry.apply_sampler(merged, "nope")

    def test_apply_none_is_noop(self, tmp_path):
        """``apply_sampler(..., None)`` should be a no-op -- the cascade calls
        this unconditionally; the None-guard lives in the helper so callers
        don't need an if/else around every call site."""
        registry = SamplerRegistry.from_directory(tmp_path)
        merged = {"temperature": 0.7}
        registry.apply_sampler(merged, None)
        assert merged == {"temperature": 0.7}


class TestVendorSampling:
    """load_vendor_sampling: source of the per-model vendor cascade layer.

    Claim: a model dir's generation_config.json yields exactly the sampling
    keys (temperature/top_p/top_k), and a missing or broken file yields {}
    rather than blocking a load.
    """

    def test_reads_sampling_keys_only(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            '{"temperature": 1.0, "top_k": 64, "top_p": 0.95,'
            ' "do_sample": true, "eos_token_id": [1, 106, 50]}'
        )
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {
            "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        }

    def test_missing_file_is_empty(self, tmp_path):
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {}

    def test_malformed_json_is_empty(self, tmp_path):
        (tmp_path / "generation_config.json").write_text("{nope")
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {}

    def test_non_numeric_values_dropped(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            '{"temperature": "high", "top_k": true, "top_p": 0.9}'
        )
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {"top_p": 0.9}


class TestResolveEffectiveSampling:
    """resolve_effective_sampling: the ONE cascade shared by both providers.

    Claim: MLX and gguf resolve request sampling through the same function
    with the same semantics; the two hand-mirrored implementations (and
    their duplicated thinking fallbacks) are gone. Deleting any test here
    lets the providers' cascades drift apart again.
    """

    @staticmethod
    def _req(**kw):
        from heylook_llm.config import ChatRequest

        body = {"messages": [{"role": "user", "content": "hi"}]}
        body.update(kw)
        return ChatRequest.model_validate(body)

    def test_floor_only(self):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(self._req(), {})
        assert merged["temperature"] == 0.7
        assert merged["max_tokens"] == 4096

    def test_vendor_overlay_beats_floor(self):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(
            self._req(), {}, vendor={"temperature": 1.0, "top_k": 64})
        assert merged["temperature"] == 1.0
        assert merged["top_k"] == 64

    def test_request_thinking_engages_overlay(self):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(self._req(enable_thinking=True), {})
        assert merged["presence_penalty"] == 1.5
        assert merged["enable_thinking"] is True

    def test_model_config_thinking_engages_overlay(self):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(self._req(), {"enable_thinking": True})
        assert merged["presence_penalty"] == 1.5

    def test_request_false_beats_model_thinking(self):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(
            self._req(enable_thinking=False), {"enable_thinking": True})
        assert merged["presence_penalty"] == 0.0
        assert merged["enable_thinking"] is False

    def test_request_sampler_suppresses_default_sampler_layer(self):
        """MLX semantics, now shared: a request naming a sampler replaces the
        model's default_sampler layer entirely -- fields the default set but
        the request sampler doesn't revert to floor/vendor."""
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(
            self._req(sampler="deterministic"), {"default_sampler": "thinking"})
        assert merged["presence_penalty"] == 0.0  # thinking layer skipped

    def test_unknown_default_sampler_logs_and_skips(self):
        """Models are validated at startup; a registry miss here means the
        registry changed post-startup -- inference must not die for it."""
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(self._req(), {"default_sampler": "gone"})
        assert merged["temperature"] == 0.7  # cascade survived

    def test_unknown_request_sampler_raises(self):
        from heylook_llm.samplers import SamplerNotFound, resolve_effective_sampling

        with pytest.raises(SamplerNotFound):
            resolve_effective_sampling(self._req(sampler="gone"), {})

    def test_explicit_request_fields_win(self):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(
            self._req(enable_thinking=True, presence_penalty=0.2, temperature=0.9),
            {"temperature": 0.3},
            vendor={"temperature": 1.0},
        )
        assert merged["temperature"] == 0.9
        assert merged["presence_penalty"] == 0.2
