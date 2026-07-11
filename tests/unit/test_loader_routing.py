# tests/unit/test_loader_routing.py
"""Effective-loader resolution (Phase 6 refinement 2026-07-11, slice 3).

Turns the registry's DESCRIPTION (modalities) + ROUTING hint (loader) into the
engine that actually loads: mlx-vlm or mlx-lm. This is the library-aware half
(vision models that mlx-vlm can't load degrade to mlx-lm instead of crashing),
kept as a pure function with the mlx-vlm registry check injected so it tests
without importing mlx.
"""
import pytest

from heylook_llm.providers.common.loader_routing import resolve_effective_loader


def _getter(value, calls):
    def g():
        calls.append(1)
        return value
    return g


@pytest.mark.unit
class TestResolveEffectiveLoader:
    def test_explicit_mlx_vlm(self):
        calls = []
        assert resolve_effective_loader(
            {"loader": "mlx-vlm", "modalities": ["text"]}, _getter("x", calls),
            vlm_supports=lambda mt: True) == "mlx-vlm"
        assert calls == []                         # explicit -> no registry probe

    def test_explicit_mlx_lm_overrides_vision(self):
        # The Qwen-as-text escape hatch: force mlx-lm even for a vision model.
        assert resolve_effective_loader(
            {"loader": "mlx-lm", "modalities": ["text", "vision"]},
            _getter("qwen3_5", []), vlm_supports=lambda mt: True) == "mlx-lm"

    def test_auto_no_vision_is_mlx_lm(self):
        calls = []
        assert resolve_effective_loader(
            {"loader": "auto", "modalities": ["text", "audio"]},
            _getter("x", calls), vlm_supports=lambda mt: True) == "mlx-lm"
        assert calls == []                         # no vision -> model_type unread

    def test_auto_vision_supported_is_mlx_vlm(self):
        assert resolve_effective_loader(
            {"loader": "auto", "modalities": ["text", "vision"]},
            _getter("qwen3_5", []), vlm_supports=lambda mt: True) == "mlx-vlm"

    def test_auto_vision_unsupported_degrades_to_mlx_lm(self):
        # The robustness fix: vision model mlx-vlm can't load -> text loader,
        # not a crash.
        assert resolve_effective_loader(
            {"loader": "auto", "modalities": ["text", "vision"]},
            _getter("some_new_vlm", []), vlm_supports=lambda mt: False) == "mlx-lm"

    def test_auto_vision_unknown_model_type_trusts_vision(self):
        # config.json unreadable -> model_type None: keep the historical
        # vision->mlx-vlm default rather than degrade a possibly-fine VLM.
        calls = []
        assert resolve_effective_loader(
            {"loader": "auto", "modalities": ["text", "vision"]},
            _getter(None, calls), vlm_supports=lambda mt: False) == "mlx-vlm"
        assert calls == [1]                        # probed, got None, trusted vision

    def test_legacy_vision_bool_without_modalities(self):
        # The provider accepts raw dicts (no modalities key) -> derive from the
        # legacy vision bool, matching MLXModelConfig._resolve_modalities.
        assert resolve_effective_loader(
            {"vision": True}, _getter("gemma4", []),
            vlm_supports=lambda mt: True) == "mlx-vlm"
        assert resolve_effective_loader(
            {"vision": False}, _getter("x", []),
            vlm_supports=lambda mt: True) == "mlx-lm"

    def test_loader_defaults_to_auto_when_absent(self):
        assert resolve_effective_loader(
            {"modalities": ["text", "vision"]}, _getter("gemma4", []),
            vlm_supports=lambda mt: True) == "mlx-vlm"
