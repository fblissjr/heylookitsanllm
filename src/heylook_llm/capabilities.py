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
The same gap is why the MLX vision capability is resolved through the loader
router rather than read off the declaration -- see :func:`_mlx_serves_vision`.
"""

from functools import lru_cache
from pathlib import Path

from heylook_llm.providers.common.loader_routing import effective_loader_for_config


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


@lru_cache(maxsize=64)
def template_supports_reasoning_effort(model_path: str) -> bool:
    """Whether the model's template reads ``reasoning_effort``.

    Separate from the thinking capability on purpose: harmony models read
    reasoning_effort and never mention enable_thinking, so a UI gating depth
    on `thinking` hides it exactly where it is the only control that works.

    Cached like its sibling above -- and not as an optimization nicety: this
    probe shipped UNCACHED (v1.71.0) and read+parsed every MLX model's
    template files on every /v1/models call. Measured live 2026-08-18 on a
    29-model registry: ~1.65s PER CALL, every call, which delayed every
    page's first paint-to-usable window (and every generation start paid the
    single-model slice via effective_capabilities). Same invalidation
    tradeoff as the sibling: templates change only with a restart in
    practice.
    """
    try:
        from heylook_llm.providers.common.template_info import read_template_info
        return read_template_info(Path(model_path), None).supports_reasoning_effort
    except Exception:
        return False


def _mlx_serves_vision(model_config) -> bool:
    """Whether the MLX provider will actually ACCEPT an image for this model.

    NOT the same question as "the checkpoint declares vision", and reporting
    the declaration is what let two surfaces contradict each other: a
    hand-made text-only variant whose directory still carries the vision
    blocks (``Qwen3.5-0.8B-MLX-8bit-textonly``, found 2026-08-29) advertised
    the capability on ``/v1/models``, and ``MLXProvider`` then refused the
    image with a 400 naming the model text-only. A client that does exactly
    what the API docs tell it to -- gate on ``capabilities`` -- got the
    refusal anyway.

    ``MLXProvider``'s guard reads ``is_vlm``, which IS
    ``effective_loader == "mlx-vlm"``, so deriving the capability from the
    same resolver makes the two agree BY CONSTRUCTION rather than by two
    rules kept in step by hand. It also picks up the case nobody had
    reported: an explicit ``loader = "mlx-lm"`` on a genuinely dual-capable
    VLM refuses images too, and used to advertise them.

    Inherits the router's fail-open direction deliberately -- an unreadable
    ``config.json`` yields ``mlx-vlm`` and keeps the capability, so
    uncertainty leaves a working VLM alone and only POSITIVE non-support
    (mlx-vlm registers no loader for this ``model_type``) drops it.
    """
    config = model_config.config
    resolved = (config.model_dump() if hasattr(config, "model_dump")
                else dict(config) if isinstance(config, dict) else {})
    return effective_loader_for_config(model_config.provider, resolved) == "mlx-vlm"


def infer_model_capabilities(model_config) -> list[str]:
    """Infer model capabilities from config when not explicitly set."""
    capabilities = []
    provider = model_config.provider
    config = model_config.config

    # Chat models (MLX)
    if provider == "mlx":
        capabilities.append("chat")

        # Vision is what the LOADER ROUTER says, not what the checkpoint
        # declares -- the provider's own image guard reads the same answer.
        if _mlx_serves_vision(model_config):
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

        # Depth is probed PRECISELY here: the template file is readable, so
        # emit the cap only when it actually reads reasoning_effort. Note this
        # is NOT implied by thinking -- Qwen3.5 reads enable_thinking and not
        # reasoning_effort, gpt-oss the reverse.
        if getattr(config, "model_path", None) and template_supports_reasoning_effort(
            str(config.model_path)
        ):
            capabilities.append("reasoning_effort")

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
        # Depth on gguf rides supports_thinking, which is BEST-EFFORT rather
        # than probed: the template lives inside GGUF metadata, so there is no
        # cheap file to scan the way the MLX branch does. A thinking-capable
        # GGUF whose template ignores reasoning_effort therefore shows the
        # control and the kwarg goes unread -- the alternative was hiding it on
        # Qwen3.8, the model the knob exists for.
        if getattr(config, "supports_thinking", False):
            capabilities.append("reasoning_effort")
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
