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

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from heylook_llm.providers.common.loader_routing import effective_loader_for_config
from heylook_llm.samplers import thinking_default


def config_dict(config, *, exclude_unset: bool = False) -> dict:
    """A provider config as a plain dict.

    ONE spelling for a shape that varies by caller: production hands a
    pydantic config model (``ModelConfig.config``), the contract fixtures and
    the provider unit tests hand a raw dict. Five hand-written copies of the
    ``hasattr(x, "model_dump")`` branch drifted in wording before this
    existed, which is the hand-copied second copy this repo derives away
    everywhere else. ``exclude_unset`` is the admin row's STORED-keys view
    (absent IS how a default is spelled in models.toml); the default is the
    RESOLVED view every derivation reads.
    """
    if hasattr(config, "model_dump"):
        return config.model_dump(exclude_unset=exclude_unset)
    if isinstance(config, dict):
        return dict(config)
    return {}


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


def _mlx_serves_vision(model_config, effective_loader: str | None = None) -> bool:
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

    The router's fail-open rule applies at ITS layer only, and the earlier
    claim here that "an unreadable config.json keeps the capability" was false
    for the common case. ``MLXModelConfig._resolve_modalities`` derives
    modalities AT VALIDATION and falls back to ``["text"]`` when the directory
    cannot be read, so a THIN entry -- which is most of them -- has already
    lost ``vision`` before ``resolve_effective_loader`` is reached, and its
    ``"vision" not in modalities -> mlx-lm`` branch settles it. Fail-open is
    real only for an entry that spells its ``modalities`` out explicitly, the
    shape CLAUDE.md calls the rare one.

    That is the right behaviour -- an unreadable checkpoint is not evidence of
    a vision tower, and advertising one we cannot confirm is the over-report
    this function exists to stop -- but it is not what "fails open" describes,
    so it is written down as what it is.
    """
    if effective_loader is not None:
        return effective_loader == "mlx-vlm"
    resolved = config_dict(model_config.config)
    return effective_loader_for_config(model_config.provider, resolved) == "mlx-vlm"


def infer_model_capabilities(model_config, effective_loader: str | None = None) -> list[str]:
    """Infer model capabilities from config when not explicitly set."""
    capabilities = []
    provider = model_config.provider
    config = model_config.config

    # Chat models (MLX)
    if provider == "mlx":
        capabilities.append("chat")

        # Vision is what the LOADER ROUTER says, not what the checkpoint
        # declares -- the provider's own image guard reads the same answer.
        if _mlx_serves_vision(model_config, effective_loader):
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


def effective_capabilities(model_config, effective_loader: str | None = None) -> list[str]:
    """The capabilities to REPORT for a model.

    An explicit ``ModelConfig.capabilities`` list is an override and
    short-circuits inference entirely; otherwise infer. Every surface that
    reports capabilities must go through here so they cannot disagree.
    """
    if model_config.capabilities:
        return model_config.capabilities
    return infer_model_capabilities(model_config, effective_loader)


# Context-length keys in transformers' priority order: max_position_embeddings
# is the canonical decoder-only field; the rest are the spellings Llama /
# Mistral / Qwen forks and the GPT-2 lineage use.
_CONTEXT_LENGTH_KEYS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_seq_length",
    "seq_length",
    "n_positions",
)


@lru_cache(maxsize=64)
def _mlx_context_length(model_path: str) -> int | None:
    """The context window an MLX checkpoint declares in its config.json:
    a top-level key first, then the nested ``text_config`` /
    ``language_config`` block VLM wrappers and Qwen-style MoE configs put
    the language head in. Cached per path like the template probes above,
    for the same reason (one file read per model per /v1/models call adds
    up; checkpoints change only with a restart in practice)."""
    import json
    try:
        with open(Path(model_path) / "config.json", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    blocks = [config] + [config.get(k) for k in ("text_config", "language_config")]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for key in _CONTEXT_LENGTH_KEYS:
            value = block.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def model_context_length(provider: str, model_path: str | None,
                         override: int | None = None) -> int | None:
    """The model's context window in tokens, or None when unknown.

    ONE resolver for every surface that names the number -- the admin row,
    /v1/models and the provider's own over-length guard -- so a client is
    shown the ceiling the server enforces. Per
    provider it is read where that provider keeps it: the GGUF header's
    ``<arch>.context_length`` (the TRAINING context llama-server sizes from
    when ``ctx_size`` is unset) for gguf, config.json for MLX. Derived from
    the files, so it is answered for unloaded models too. Embedding models
    have no chat context and answer None.

    ``override`` is the entry's own ``context_length`` (the MLX config field)
    and wins over the files whatever they say: a YaRN-scaled checkpoint
    often ships the ORIGINAL ``max_position_embeddings`` with the factor in
    ``rope_scaling``, and the file value alone would refuse a prompt the
    model takes. Non-positive values are not a window and are ignored.
    """
    if isinstance(override, int) and not isinstance(override, bool) and override > 0:
        return override
    if not model_path:
        return None
    path = Path(str(model_path)).expanduser()
    if provider == "gguf":
        from heylook_llm import gguf_metadata
        return gguf_metadata.context_length(path)
    if provider == "mlx":
        return _mlx_context_length(str(path))
    return None


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """Every DERIVED fact a model row reports, resolved once (see
    :func:`derived_model_facts`)."""
    resolved: dict
    capabilities: list[str]
    effective_loader: str | None
    thinking_default: bool
    context_length: int | None
    context_running: int | None


def derived_model_facts(model_config, router=None) -> ModelFacts:
    """The derived facts ``/v1/models`` and ``/v1/admin/models`` both report,
    from ONE derivation.

    Two row builders used to derive these separately, each with its own
    config-dict spelling, and they had already drifted once: the admin row
    built the resolved dump for mlx entries only, so a gguf entry's
    models.toml ``enable_thinking`` or ``default_sampler`` never reached its
    ``thinking_default`` while the same value on ``/v1/models`` did. The dump
    is built for EVERY row here because three consumers read it (loader
    routing, the thinking cascade, the context resolver); the per-row cost
    that moved the admin read routes off the event loop is unchanged -- the
    router work behind ``effective_loader_for_config`` still returns on its
    first line for anything but mlx.

    ``router`` is optional: only ``context_running`` (what a RESIDENT
    llama-server process was sized to) needs it, and ``/v1/models`` does not
    carry that field.
    """
    resolved = config_dict(model_config.config)
    # ONE resolution, three consumers. `effective_capabilities` derives the
    # vision capability from this same value (v1.79.43), so letting it
    # resolve its own would rebuild the dump and re-run the router per row.
    effective_loader = effective_loader_for_config(model_config.provider, resolved)
    capabilities = effective_capabilities(model_config, effective_loader)
    # What thinking resolves to with nothing said: the SAME cascade the
    # providers run (an empty request through resolve_effective_sampling),
    # so the row reports the value generation will use and not a re-derived
    # guess. `thinking_capable` is the served capability, which is also the
    # cascade's own last fallback.
    default_thinking = thinking_default(
        resolved, thinking_capable="thinking" in capabilities)
    context_length = model_context_length(
        model_config.provider, resolved.get("model_path"),
        override=resolved.get("context_length"))
    # gguf only: MLX has no fixed allocation to report. None until ready and
    # None when /props does not say (LlamaServerProvider.running_ctx).
    context_running = None
    if model_config.provider == "gguf" and router is not None:
        provider = router.get_loaded_models().get(model_config.id)
        context_running = getattr(provider, "running_ctx", None)
    return ModelFacts(
        resolved=resolved,
        capabilities=capabilities,
        effective_loader=effective_loader,
        thinking_default=default_thinking,
        context_length=context_length,
        context_running=context_running,
    )
