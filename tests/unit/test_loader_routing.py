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


class TestEffectiveLoaderForConfig:
    """`effective_loader_for_config` -- the same answer, without a process.

    The routing rule is `resolve_effective_loader`; this is the wrapper the
    admin listing calls, and the only thing it adds is the provider gate and
    the model_type read. Both are the parts that can be wrong on the wire.
    """

    def test_none_for_every_non_mlx_provider(self):
        # The question is WHICH MLX LIBRARY. gguf is one engine and is already
        # named by `provider`; an embedding model has no answer at all. Naming
        # a loader for either would be a claim the field cannot support.
        from heylook_llm.providers.common.loader_routing import effective_loader_for_config

        cfg = {"loader": "auto", "modalities": ["text", "vision"]}
        assert effective_loader_for_config("gguf", cfg) is None
        assert effective_loader_for_config("mlx_embedding", cfg) is None
        assert effective_loader_for_config("mlx", cfg) in ("mlx-lm", "mlx-vlm")

    def test_explicit_loader_wins_over_the_vision_capability(self):
        # The conflation this whole field exists to prevent: an explicit
        # `loader = "mlx-lm"` on a dual-capable VLM still declares vision, so
        # anything splitting on the capability would call it mlx-vlm and go
        # green having run mlx-lm twice.
        from heylook_llm.providers.common.loader_routing import effective_loader_for_config

        assert effective_loader_for_config(
            "mlx", {"loader": "mlx-lm", "modalities": ["text", "vision"]}) == "mlx-lm"

    def test_missing_model_path_does_not_raise(self):
        # Discovered entries, draft/MTP heads, half-written configs: a read
        # that cannot happen must degrade to the vision declaration, not to a
        # 500 on the models page.
        from heylook_llm.providers.common.loader_routing import effective_loader_for_config

        assert effective_loader_for_config(
            "mlx", {"loader": "auto", "modalities": ["text", "vision"]}) == "mlx-vlm"
        assert effective_loader_for_config(
            "mlx", {"loader": "auto", "model_path": None,
                    "modalities": ["text"]}) == "mlx-lm"


@pytest.mark.unit
class TestUnresolvedDescriptionIsRefused:
    """A config with no capability declaration gets a refusal, not an answer.

    `auto` routing reads the declaration. `merge_discovered` returns RAW dicts
    and the declaration is derived at validation, so a config taken straight
    from the merge declares nothing -- and answering from that absence returns
    the text loader for every model, vision ones included, with no exception
    and no log line. Two sessions were caught by exactly that on 2026-09-08.

    It matters past a wrong count: anything comparing a served set before and
    after a config edit calls this per model, so an unvalidated snapshot on
    either side reports engine changes that never happened.
    """

    def _f(self):
        from heylook_llm.providers.common.loader_routing import effective_loader_for_config
        return effective_loader_for_config

    def test_a_config_declaring_nothing_is_refused(self):
        with pytest.raises(ValueError, match="modalities"):
            self._f()("mlx", {"loader": "auto", "model_path": "/synthetic/x"})

    def test_an_explicit_loader_needs_no_declaration(self):
        # The explicit loader wins outright, so the declaration is never read
        # and its absence cannot mislead. Refusing here would be a false alarm.
        assert self._f()("mlx", {"loader": "mlx-lm"}) == "mlx-lm"

    def test_the_legacy_vision_bool_counts_as_a_declaration(self):
        # `_modalities_of` still honours the old `vision` key, so a config
        # carrying it HAS declared -- and must be answered, not refused.
        assert self._f()("mlx", {"loader": "auto", "vision": True,
                                 "model_path": ""}) == "mlx-vlm"
        assert self._f()("mlx", {"loader": "auto", "vision": False}) == "mlx-lm"

    def test_a_non_mlx_provider_short_circuits_before_the_guard(self):
        # The question is WHICH MLX LIBRARY; gguf has no answer and must not
        # be refused on its way to None.
        assert self._f()("gguf", {"loader": "auto"}) is None
        assert self._f()("mlx_embedding", {"loader": "auto"}) is None
