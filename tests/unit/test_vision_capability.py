# tests/unit/test_vision_capability.py
"""The reported ``vision`` capability and the provider's image guard.

Found live 2026-08-29: ``Qwen3.5-0.8B-MLX-8bit-textonly`` advertised
``vision`` on ``/v1/models`` and ``MLXProvider`` then refused the image with a
400 saying the model is text-only. Both were reading a real signal -- the
capability read the checkpoint's DECLARATION (its dir carries
``vision_config``, and that entry points at a dual-capable dir on purpose),
the guard read ``is_vlm`` -- and a client doing exactly what the API docs say,
gating on ``capabilities``, got refused anyway.

(That entry's `loader = "mlx-lm"` field was retired with the mlx-vlm
engine, plan W10 stage 3; the declaration is now the only input.)

The claim these tests pin is not "text-only models report no vision". It is
that ONE resolver answers for both surfaces, so they cannot drift apart
again: ``capabilities.py`` and ``MLXProvider.__init__`` both call
``resolve_effective_loader`` on the same config dict.
"""

import json

import pytest


def _model_dir(tmp_path, *, vision: bool, model_type: str = "qwen3_5"):
    """A checkpoint dir whose config.json declares (or does not declare) vision.

    Written as a real config.json rather than passing ``modalities`` in, because
    deriving modalities from the dir IS the path every thin registry entry takes
    -- an entry that spells its modalities out is the rare one.
    """
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": model_type, **({
            "vision_config": {"hidden_size": 8}, "image_token_id": 151655,
        } if vision else {})}))
    return tmp_path


def _entry(model_dir, **config_overrides):
    """A registry entry for ``model_dir``. Validated through ``MLXModelConfig``
    so the modality derivation these tests depend on actually runs."""
    from heylook_llm.config import MLXModelConfig, ModelConfig

    config = MLXModelConfig.model_validate(
        {"model_path": str(model_dir), **config_overrides})
    return ModelConfig(id="m", provider="mlx", config=config), config


def _caps_and_guard(model_dir, **config_overrides):
    """Both surfaces' answers for one entry: the reported capability, and the
    attribute ``MLXProvider``'s image refusal actually branches on.

    The provider is CONSTRUCTED, not loaded -- ``__init__`` resolves the engine
    and touches no weights, so this reads the production attribute rather than a
    re-derivation of it.
    """
    from heylook_llm.capabilities import effective_capabilities
    from heylook_llm.providers.mlx_provider import MLXProvider

    mc, config = _entry(model_dir, **config_overrides)
    provider = MLXProvider("m", config.model_dump(), verbose=False)
    return effective_capabilities(mc), provider.is_vlm


@pytest.mark.unit
class TestVisionCapabilityMatchesTheProviderGuard:

    def test_a_text_checkpoint_reports_no_vision(self, tmp_path):
        caps, is_vlm = _caps_and_guard(_model_dir(tmp_path, vision=False))
        assert "vision" not in caps
        assert is_vlm is False

    @pytest.mark.parametrize("vision", [True, False])
    def test_the_two_surfaces_agree_on_every_combination(self, tmp_path, vision):
        """The invariant, stated as the equality rather than as six expected
        answers: whatever the router decides, the advertised capability and the
        guard say the same thing. A future change to the routing rule moves both
        or fails here -- which is the property, since either surface alone can be
        "correct" while the pair is a lie to the client.
        """
        caps, is_vlm = _caps_and_guard(_model_dir(tmp_path, vision=vision))
        assert ("vision" in caps) is is_vlm

    def test_an_explicit_capabilities_override_still_wins(self, tmp_path):
        """The override is documented as short-circuiting inference entirely,
        and this fix must not quietly turn it into a suggestion -- an operator
        who hand-writes `capabilities` is asserting something the derivation
        cannot see."""
        from heylook_llm.capabilities import effective_capabilities

        mc, _ = _entry(_model_dir(tmp_path, vision=False))
        mc.capabilities = ["chat", "vision"]
        assert effective_capabilities(mc) == ["chat", "vision"]


@pytest.mark.unit
class TestVisionCapabilityFailsOpen:
    def test_a_thin_entry_with_an_unreadable_dir_reports_no_vision(self, tmp_path):
        """The COMMON shape, and the one the docstring used to describe
        wrongly. A thin entry writes no `modalities`, so validation derives
        them from the dir; when that cannot be read it falls back to
        ["text"] and the loader router never reaches its fail-open branch.
        Correct -- an unreadable checkpoint is not evidence of a vision tower
        -- but it is NOT the "keeps the capability" the docstring claimed, and
        only the explicit-modalities case below ever exercised that path."""
        from heylook_llm.capabilities import effective_capabilities

        mc, config = _entry(tmp_path / "not-downloaded-yet")
        assert config.modalities == ["text"]
        assert "vision" not in effective_capabilities(mc)

    def test_an_unreadable_model_dir_keeps_an_EXPLICIT_declared_capability(self, tmp_path):
        """Inherited from the router on purpose: only POSITIVE non-support
        drops the engine to mlx-lm, so a path that cannot be read (an HF repo
        id, a not-yet-downloaded dir) must not silently strip vision off a
        working VLM. Degrading on uncertainty is the failure mode the loader
        router was written to avoid; the capability surface inherits it rather
        than inventing a second policy.
        """
        from heylook_llm.capabilities import effective_capabilities

        mc, _ = _entry(tmp_path / "not-downloaded-yet",
                       modalities=["text", "vision"])
        assert "vision" in effective_capabilities(mc)
