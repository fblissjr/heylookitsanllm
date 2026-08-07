# src/heylook_llm/capabilities.py
"""Derived model capabilities -- what the server will actually SERVE.

Extracted from api.py (2026-08-07) because there are TWO readers and only
one of them was inferring: ``/v1/models`` (what the v3 chat UI gates its
attach button, thinking toggle and vision controls on) and
``/v1/admin/models`` (what the Models page lists). The admin surface
reported the STORED ``ModelConfig.capabilities`` override, which is empty on
every entry that never hand-wrote one -- so the Models page showed no
capabilities for anything, and increasingly so once derive-at-load made
entries thin.

Capabilities are deliberately narrower than ``modalities``: modalities are
the author's DESCRIPTION of the checkpoint, capabilities are gated to what
this server will serve it as. An MLX gemma declares an audio modality and
never gets the audio capability, because MLX strips audio towers at load.
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def template_supports_thinking(model_path: str) -> bool:
    """Whether the model's own chat template references ``enable_thinking``.

    The template kwarg is the cross-model thinking mechanism (Qwen3 renders
    <think> blocks, gemma-4 renders thought channels; transformers forwards
    extra apply_chat_template kwargs as template variables), so the template
    referencing the variable IS the capability signal -- no manual
    models.toml flag needed. Cached per path: file reads on every
    /v1/models call would add up, and templates only change with a restart
    in practice.
    """
    try:
        from heylook_llm.providers.common.template_info import read_template_info
        return read_template_info(Path(model_path), None).supports_enable_thinking
    except Exception:
        return False


def infer_model_capabilities(model_config) -> list[str]:
    """Infer model capabilities from config when not explicitly set."""
    capabilities = []
    provider = model_config.provider
    config = model_config.config

    # Chat models (MLX)
    if provider == "mlx":
        capabilities.append("chat")

        # Check for vision capability
        if hasattr(config, "vision") and config.vision:
            capabilities.append("vision")

        # Thinking capability is DERIVED: the enable_thinking default-on
        # flag, else the model's own chat template (enable_thinking
        # reference). No manual MLX flag -- supports_thinking is GGUF-only
        # (nothing cheap to probe inside GGUF metadata).
        if hasattr(config, "enable_thinking") and config.enable_thinking:
            capabilities.append("thinking")
        elif getattr(config, "model_path", None) and template_supports_thinking(
            str(config.model_path)
        ):
            capabilities.append("thinking")

        # MLX models support hidden states extraction
        capabilities.append("hidden_states")

    # GGUF via llama-server subprocess. Capabilities come from the entry's
    # own description (mmproj sidecar / modalities / explicit thinking flag)
    # -- no template probing (the template lives inside GGUF metadata), and
    # NEVER hidden_states/logprobs (MLX-only surfaces). The explicit
    # ModelConfig.capabilities override short-circuits this entirely.
    elif provider == "gguf":
        capabilities.append("chat")
        modalities = getattr(config, "modalities", None) or []
        if getattr(config, "mmproj_path", None) or "vision" in modalities:
            capabilities.append("vision")
        if "audio" in modalities:
            # gguf only: MLX strips audio towers at load, so the mlx branch
            # above must never emit this cap even when the model declares
            # the modality.
            capabilities.append("audio")
        if getattr(config, "supports_thinking", None):
            capabilities.append("thinking")

    return capabilities


def effective_capabilities(model_config) -> list[str]:
    """The capabilities to REPORT for a model.

    An explicit ``ModelConfig.capabilities`` list is an override and
    short-circuits inference entirely; otherwise infer. Every surface that
    reports capabilities must go through here so they cannot disagree.
    """
    if model_config.capabilities:
        return model_config.capabilities
    return infer_model_capabilities(model_config)
