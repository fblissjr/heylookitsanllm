# tests/unit/test_loader_routing.py
"""Served-vision resolution: is an MLX model served with vision?

The registry's DESCRIPTION (modalities) plus whether mlx-vlm registers the
model_type. A vision model mlx-vlm can't run as a VLM is served as text rather
than crashing at load. Pure, with the mlx-vlm registry check injected, so it
tests without importing mlx.
"""
import pytest

from heylook_llm.providers.common.loader_routing import (
    resolve_serves_vision,
    serves_vision_for_config,
)


def _getter(value, calls):
    def g():
        calls.append(1)
        return value
    return g


@pytest.mark.unit
class TestResolveServesVision:

    def test_no_vision_declared_is_text_and_reads_nothing(self):
        calls = []
        assert resolve_serves_vision(
            {"modalities": ["text", "audio"]},
            _getter("x", calls), vlm_supports=lambda mt: True) is False
        assert calls == []                         # no vision -> model_type unread

    def test_vision_follows_the_mlx_vlm_registry(self):
        vision = {"modalities": ["text", "vision"]}
        assert resolve_serves_vision(
            vision, _getter("qwen3_5", []), vlm_supports=lambda mt: True) is True
        # mlx-vlm can't run it as a VLM -> text, not a crash at load.
        assert resolve_serves_vision(
            vision, _getter("some_new_vlm", []), vlm_supports=lambda mt: False) is False

    def test_unknown_model_type_trusts_the_declaration(self):
        # config.json unreadable -> model_type None: keep vision rather than
        # degrade a possibly-fine VLM.
        calls = []
        assert resolve_serves_vision(
            {"modalities": ["text", "vision"]},
            _getter(None, calls), vlm_supports=lambda mt: False) is True
        assert calls == [1]

    def test_legacy_vision_bool_without_modalities(self):
        # The provider accepts raw dicts (no modalities key) -> derive from the
        # legacy vision bool, matching MLXModelConfig._resolve_modalities.
        assert resolve_serves_vision(
            {"vision": True}, _getter("gemma4", []), vlm_supports=lambda mt: True) is True
        assert resolve_serves_vision(
            {"vision": False}, _getter("x", []), vlm_supports=lambda mt: True) is False


@pytest.mark.unit
class TestServesVisionForConfig:
    """The same answer without a process: the provider gate, the model_type
    read, and the refusal of an unvalidated config."""

    def test_none_for_gguf_even_before_the_guard(self):
        # gguf's vision is its projector; this resolver has no answer there,
        # and must not refuse a config on its way to None.
        assert serves_vision_for_config("gguf", {"modalities": ["text", "vision"]}) is None
        assert serves_vision_for_config("gguf", {}) is None

    def test_missing_model_path_does_not_raise(self):
        # Discovered entries, MTP heads, half-written configs: a read that
        # cannot happen degrades to the declaration, not to a 500.
        assert serves_vision_for_config("mlx", {"modalities": ["text", "vision"]}) is True
        assert serves_vision_for_config(
            "mlx", {"model_path": None, "modalities": ["text"]}) is False

    def test_a_config_declaring_nothing_is_refused(self):
        """`merge_discovered` returns RAW dicts; the declaration is derived at
        validation. Answering from its absence would report every model as
        text-only with no exception and no log line (two sessions were caught
        by exactly that on 2026-09-08). The legacy `vision` key still counts
        as a declaration."""
        with pytest.raises(ValueError, match="modalities"):
            serves_vision_for_config("mlx", {"model_path": "/synthetic/x"})
        assert serves_vision_for_config("mlx", {"vision": True, "model_path": ""}) is True
        assert serves_vision_for_config("mlx", {"vision": False}) is False
