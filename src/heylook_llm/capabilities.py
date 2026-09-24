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

from heylook_llm.providers.common.loader_routing import serves_vision_for_config
from heylook_llm.providers.contract import EngineDescription, describe
from heylook_llm.samplers import sampler_defaults, thinking_default


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


# The chat-template files a probe below depends on, in the model's own dir.
# `chat_template.heylook.jinja` is the operator override and is WRITTEN AT
# RUNTIME by PUT/DELETE /v1/admin/models/{id}/chat-template, which is what
# makes a plain per-path cache wrong here rather than merely stale-ish.
# DERIVED, never re-listed: template_info owns what it reads. This was a
# hand-written tuple of the four template sources and silently omitted
# tokenizer.json + generation_config.json, which the stop-token check reads and
# which can change WHICH template wins -- so the cache key was already
# incomplete, not merely rot-prone.
from heylook_llm.providers.common.template_info import TEMPLATE_INPUT_FILES

_TEMPLATE_SOURCE_FILES = TEMPLATE_INPUT_FILES


def _template_stamp(model_path: str) -> tuple:
    """File identity for every chat-template source in a model directory.

    The cache key for the two template probes, and NOT a per-path key. Those
    probes were keyed on the path alone, each justified by a comment saying
    templates only change with a restart. The operator template override
    (v2.0.22) made that false: the admin routes write
    `chat_template.heylook.jinja` into the model's own directory at runtime,
    and it is the TOP rung of the auto ladder. Nothing invalidated the caches
    -- a model reload did not either -- so an override that enables thinking
    was honoured by generation while /v1/models kept reporting the model as
    unable to think and v3 kept hiding the toggle, until the process restarted.

    Same shape as `gguf_metadata`'s context-length cache: the FILE's identity
    is the key, so a write is seen and an unchanged file is never re-read. A
    missing file stamps as absent, so creating or deleting the override both
    count as a change.
    """
    stamp = []
    for name in _TEMPLATE_SOURCE_FILES:
        try:
            st = (Path(model_path) / name).stat()
            stamp.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append((name, None, None))
    return tuple(stamp)


@lru_cache(maxsize=64)
def _template_supports_thinking(model_path: str, _stamp: tuple) -> bool:
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
        from heylook_llm.providers.common.template_info import cached_template_info
        return cached_template_info(Path(model_path)).supports_enable_thinking
    except Exception:
        return False


@lru_cache(maxsize=64)
def _template_supports_reasoning_effort(model_path: str, _stamp: tuple) -> bool:
    """Whether the model's template reads ``reasoning_effort``.

    Separate from the thinking capability on purpose: harmony models read
    reasoning_effort and never mention enable_thinking, so a UI gating depth
    on `thinking` hides it exactly where it is the only control that works.

    Cached like its sibling above -- and not as an optimization nicety: this
    probe shipped UNCACHED (v1.71.0) and read+parsed every MLX model's
    template files on every /v1/models call, which delayed every page's first
    paint-to-usable window (and every generation start paid the single-model
    slice via effective_capabilities). Keyed on the template files' identity,
    not on the path -- see `_template_stamp`.
    """
    try:
        from heylook_llm.providers.common.template_info import cached_template_info
        return cached_template_info(Path(model_path)).supports_reasoning_effort
    except Exception:
        return False


def template_supports_thinking(model_path: str) -> bool:
    """Public probe: cached, but re-read when a template file changes.

    Stamping is one `stat` per candidate filename, which is what buys the
    correctness the plain per-path cache did not have. Cheap next to the
    read-and-parse it guards, and unavoidable: the alternative is a
    `cache_clear` call at every site that can write a template, which is the
    hand-maintained second copy this repo derives away everywhere else.
    """
    return _template_supports_thinking(model_path, _template_stamp(model_path))


def template_supports_thinking_budget(model_path: str) -> bool:
    """Whether the MLX engine can enforce a thinking budget on this model's
    format (``template_info.thinking_budget_markers``)."""
    try:
        from heylook_llm.providers.common.template_info import (
            cached_template_info, thinking_budget_markers)
        return thinking_budget_markers(cached_template_info(Path(model_path))) is not None
    except Exception:
        return False


def template_supports_reasoning_effort(model_path: str) -> bool:
    """Public probe: cached, but re-read when a template file changes.

    Same shape and same reason as its sibling above.
    """
    return _template_supports_reasoning_effort(model_path, _template_stamp(model_path))


def _mlx_serves_vision(model_config, serves_vision: bool | None = None) -> bool:
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
    ``resolve_serves_vision``'s answer, so deriving the capability from the
    same resolver makes the two agree BY CONSTRUCTION rather than by two
    rules kept in step by hand.

    The router's fail-open rule applies at ITS layer only, and the earlier
    claim here that "an unreadable config.json keeps the capability" was false
    for the common case. ``MLXModelConfig._resolve_modalities`` derives
    modalities AT VALIDATION and falls back to ``["text"]`` when the directory
    cannot be read, so a THIN entry -- which is most of them -- has already
    lost ``vision`` before ``resolve_serves_vision`` is reached, and its
    ``"vision" not in modalities -> text`` branch settles it. Fail-open is
    real only for an entry that spells its ``modalities`` out explicitly, the
    shape CLAUDE.md calls the rare one.

    That is the right behaviour -- an unreadable checkpoint is not evidence of
    a vision tower, and advertising one we cannot confirm is the over-report
    this function exists to stop -- but it is not what "fails open" describes,
    so it is written down as what it is.
    """
    if serves_vision is not None:
        return serves_vision
    resolved = config_dict(model_config.config)
    return bool(serves_vision_for_config(model_config.provider, resolved))


def infer_model_capabilities(model_config, serves_vision: bool | None = None) -> list[str]:
    """Infer model capabilities from config when not explicitly set."""
    capabilities = []
    provider = model_config.provider
    config = model_config.config

    # Chat models (MLX)
    if provider == "mlx":
        capabilities.append("chat")

        # Vision is what the served-vision resolver says, not what the checkpoint
        # declares -- the provider's own image guard reads the same answer.
        if _mlx_serves_vision(model_config, serves_vision):
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

        # A hard thinking cap (plan W7), where the engine can close the
        # model's thinking format with one forced token.
        if "thinking" in capabilities and template_supports_thinking_budget(
            str(getattr(config, "model_path", "") or "")
        ):
            capabilities.append("thinking_budget")

        # Depth is probed PRECISELY here: the template file is readable, so
        # emit the cap only when it actually reads reasoning_effort. Note this
        # is NOT implied by thinking -- Qwen3.5 reads enable_thinking and not
        # reasoning_effort, gpt-oss the reverse.
        if getattr(config, "model_path", None) and template_supports_reasoning_effort(
            str(config.model_path)
        ):
            capabilities.append("reasoning_effort")

    # GGUF via llama-server subprocess. Capabilities come from the entry's
    # own description (mmproj sidecar / modalities / explicit thinking flag)
    # -- no template probing (the template lives inside GGUF metadata). The
    # explicit ModelConfig.capabilities override short-circuits this entirely.
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
            # llama-server's reasoning budget, applied where it found the
            # template's thinking end tags (its own analysis; not visible
            # from here, so a template it cannot read runs uncapped).
            capabilities.append("thinking_budget")

    return capabilities


def effective_capabilities(model_config, serves_vision: bool | None = None) -> list[str]:
    """The capabilities to REPORT for a model.

    An explicit ``ModelConfig.capabilities`` list is an override and
    short-circuits inference entirely; otherwise infer. Every surface that
    reports capabilities must go through here so they cannot disagree.
    """
    if model_config.capabilities:
        return model_config.capabilities
    return infer_model_capabilities(model_config, serves_vision)


def model_context_length(provider: str, model_path: str | None,
                         override: int | None = None) -> int | None:
    """The model's context window in tokens, or None when unknown.

    ONE resolver for every surface that names the number -- the admin row,
    /v1/models and the provider's own over-length guard -- so a client is
    shown the ceiling the server enforces. Per
    provider it is read where that provider keeps it: the GGUF header's
    ``<arch>.context_length`` (the TRAINING context llama-server sizes from
    when ``ctx_size`` is unset) for gguf, config.json for MLX. Derived from
    the files, so it is answered for unloaded models too.

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
    # Where the number lives is each engine's own answer (its describer's
    # file_context_length); this function only owns the override rule.
    from heylook_llm.providers.contract import describer_for
    describer = describer_for(provider)
    if describer is None:
        return None
    return describer.file_context_length(Path(str(model_path)).expanduser())


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """Every DERIVED fact a model row reports, resolved once (see
    :func:`derived_model_facts`). ``engine`` is the engine contract
    (providers/contract.py): which library runs the model, its context,
    template and every setting with its provenance."""
    resolved: dict
    capabilities: list[str]
    thinking_default: bool
    sampler_defaults: dict
    engine: EngineDescription


@lru_cache(maxsize=64)
def _vendor_sampling_pairs(provider: str, model_path: str) -> tuple:
    """The model's own recommended decode settings, from wherever ITS engine
    keeps them -- the same read the provider does at generation time.

    ONE concept, two spellings on disk: MLX reads the model dir's
    generation_config.json, gguf reads `general.sampling.*` out of the GGUF
    header (which converters write FROM that same file). Both must be here.
    An engine that gains a vendor layer in its provider and not here reports
    the global FLOOR for its models while generation uses the vendor values
    -- a plausible number that is not the one in force, which is the exact
    failure `sampler_defaults` exists to prevent. That is not hypothetical:
    it happened within one commit of this function being written, when gguf
    gained its header vendor layer (v2.0.22) and this gate still said mlx.

    Behind the per-row cache the sibling derived reads use, and returning
    sorted PAIRS rather than a dict: a cached mutable would let one caller's
    edit poison every later row (the trap the Metal device-info cache
    carries its own comment about).
    """
    # WHERE each engine keeps its vendor layer is that engine's describer's
    # answer (vendor_sampling); this stays the one cached entry point every
    # caller goes through, so the pairing still has exactly one home.
    from heylook_llm.providers.contract import describer_for
    describer = describer_for(provider)
    vendor = describer.vendor_sampling(Path(model_path)) if describer else {}
    return tuple(sorted(vendor.items()))


def derived_model_facts(model_config, router=None) -> ModelFacts:
    """The derived facts ``/v1/models`` and ``/v1/admin/models`` both report,
    from ONE derivation.

    Two row builders used to derive these separately, each with its own
    config-dict spelling, and they had already drifted once: the admin row
    built the resolved dump for mlx entries only, so a gguf entry's
    models.toml ``enable_thinking`` never reached its
    ``thinking_default`` while the same value on ``/v1/models`` did. The dump
    is built for EVERY row here because three consumers read it (loader
    routing, the thinking cascade, the context resolver); the per-row cost
    that moved the admin read routes off the event loop is unchanged -- the
    router work behind ``serves_vision_for_config`` still returns on its
    first line for anything but mlx.

    ``router`` supplies what the engine contract needs beyond the config:
    which models have a models.toml entry and what discovery derives for
    them (so only real overrides read as configured), and the loaded
    providers' observed halves. Both routes pass it.
    """
    resolved = config_dict(model_config.config)
    # ONE resolution, three consumers. `effective_capabilities` derives the
    # vision capability from this same value (v1.79.43), so letting it
    # resolve its own would rebuild the dump and re-run the router per row.
    serves_vision = serves_vision_for_config(model_config.provider, resolved)
    capabilities = effective_capabilities(model_config, serves_vision)
    # What thinking resolves to with nothing said: the SAME cascade the
    # providers run (an empty request through resolve_effective_sampling),
    # so the row reports the value generation will use and not a re-derived
    # guess. `thinking_capable` is the served capability, which is also the
    # cascade's own last fallback.
    default_thinking = thinking_default(
        resolved, thinking_capable="thinking" in capabilities)
    # Every sampler key's unset answer, from the SAME cascade, so the panel
    # labels a blank field with the value generation will really use. The
    # vendor layer is passed exactly where the provider passes it -- MLX
    # reads generation_config.json, gguf never does.
    # Every engine's vendor layer is resolved in ONE place, keyed by provider
    # -- see _vendor_sampling_pairs for why a missing engine there is a silent
    # wrong number rather than a missing one. `test_vendor_layer_reaches_the
    # _report_on_every_engine` pins the pairing.
    vendor = None
    if resolved.get("model_path"):
        vendor = dict(_vendor_sampling_pairs(
            model_config.provider, str(resolved["model_path"]))) or None
    defaults = sampler_defaults(
        resolved, thinking_capable="thinking" in capabilities, vendor=vendor)
    return ModelFacts(
        resolved=resolved,
        capabilities=capabilities,
        thinking_default=default_thinking,
        sampler_defaults=defaults,
        engine=_engine_with_sampler_answers(
            describe(model_config, router), defaults, resolved,
            thinking_capable="thinking" in capabilities, vendor=vendor),
    )


def _engine_with_sampler_answers(engine, defaults: dict, resolved: dict, *,
                                 thinking_capable: bool, vendor):
    """The engine's settings, with every sampler key carrying the CASCADE's
    answer rather than a second derivation.

    The keys are the overlap of the settings and ``sampler_defaults``
    (derived, never listed). ``value`` is the cascade's answer, which
    ``sampler_defaults`` also reports; ``auto`` is the same cascade with the
    model's stored sampler values removed, i.e. what applies if nothing were
    stored. Works on a copy: ``describe`` may hand back a cached object.
    """
    overlap = sorted(set(engine.settings) & set(defaults))
    if not overlap:
        return engine
    engine = engine.model_copy(deep=True)
    stored = {k for k in overlap if engine.settings[k].provenance == "configured"}
    unstored = sampler_defaults(
        {k: v for k, v in resolved.items() if k not in stored},
        thinking_capable=thinking_capable, vendor=vendor) if stored else defaults
    cascade = ("the sampling cascade: this model's stored value, else its own "
               "vendor defaults, else heylook's floor")
    for key in overlap:
        entry = engine.settings[key]
        engine.settings[key] = entry.model_copy(update={
            "value": defaults[key],
            "auto": unstored.get(key),
            "reason": entry.reason if key in stored else cascade,
        })
    return engine
