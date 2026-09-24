# src/heylook_llm/providers/gguf_describe.py
"""gguf's static half of the engine contract (providers/contract.py).

Plain functions over the config and the model's files; never spawns or talks
to llama-server. Every decision it reports is made by the SAME class-level
helper the spawn uses (resolve_chat_template, binary_report,
image_token_cap_decision, keep_alive_choice), so the unloaded answer and the
spawned one cannot drift. What only a spawn decides (the auto micro-batch
from live headroom, /props) is LlamaServerProvider.describe_observed().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from heylook_llm.providers.contract import (
    ContextFacts,
    EngineDescription,
    Fact,
    prefix_stable_fact,
    Setting,
    StaticInputs,
    TemplateFacts,
    config_digest,
    config_settings,
    public_value,
    sha256_text,
    store_name,
)


def _provider():
    # Pure stdlib module (no MLX), imported lazily to keep this module cheap.
    from heylook_llm.providers.llama_server_provider import LlamaServerProvider
    return LlamaServerProvider


def vendor_sampling(model_path: Path) -> dict:
    """The model's own recommended decode settings: gguf keeps them in the
    header's general.sampling.* (written from the same generation_config)."""
    from heylook_llm import gguf_metadata
    return gguf_metadata.vendor_sampling(model_path)


def file_context_length(model_path: Path):
    """The training context in the GGUF header (<arch>.context_length), the
    number llama-server sizes from when ctx_size is unset; None if absent."""
    from heylook_llm import gguf_metadata
    return gguf_metadata.context_length(model_path)


def static_inputs(model_id: str, cfg: dict, written: bool,
                  derived: dict) -> StaticInputs:
    from heylook_llm.providers.common.template_info import HEYLOOK_TEMPLATE_FILENAME

    P = _provider()
    weights = Path(str(cfg.get("model_path") or "")).expanduser()
    files = [str(weights),
             str(weights.parent / "chat_template.jinja"),
             str(weights.parent / HEYLOOK_TEMPLATE_FILENAME),
             str(P.DEFAULT_BUILD.parents[1] / "heylook-build.json")]
    for key in ("chat_template_path", "mmproj_path", "draft_model_path", "server_binary"):
        if cfg.get(key):
            files.append(str(Path(str(cfg[key])).expanduser()))
    # Environment that changes an answer here: the binary override and an
    # inherited keep-alive.
    from heylook_llm.providers.llama_server_provider import METAL_KEEP_ALIVE_ENV
    env = f"{os.environ.get('HEYLOOK_LLAMA_SERVER', '')}|{os.environ.get(METAL_KEEP_ALIVE_ENV, '')}"
    return StaticInputs(files=tuple(files), engine_build=env,
                        config_digest=config_digest(cfg, written, derived))


def _template(model_id: str, cfg: dict) -> TemplateFacts:
    from heylook_llm import chat_template_files

    path, origin = _provider().resolve_chat_template(cfg, model_id, log=False)
    view = chat_template_files.view(model_id, "gguf", cfg)
    ladder = ("the gguf template ladder (chat_template_path > override > "
              "chat_template.jinja beside the .gguf > embedded)")
    return TemplateFacts(
        origin=Fact(value=public_value(origin), provenance="derived", source=ladder),
        path=Fact(value=public_value(path) if path else None,
                  provenance="derived",
                  source=ladder if path else "embedded in the GGUF: no file"),
        sha256=Fact(value=sha256_text(view.template),
                    provenance="derived" if view.template else "unknown",
                    source=(public_value(path) if path else "the GGUF header")
                    if view.template else "no template body resolved"),
        running_sha256=Fact(provenance="unknown", source="not loaded"),
        prefix_stable=prefix_stable_fact(view.template),
    )


def describe_static(model_id: str, cfg: dict, config_obj: Any, *,
                    written: bool, derived: dict) -> EngineDescription:
    from heylook_llm.capabilities import model_context_length
    from heylook_llm.config import GGUFModelConfig

    P = _provider()
    length = model_context_length("gguf", cfg.get("model_path"))
    settings = config_settings(GGUFModelConfig, config_obj, written=written,
                               derived=derived, engine_default="llama-server")

    # The micro-batch, when not stored, is decided at spawn from live
    # working-set headroom: unknown until then.
    if settings["n_ubatch"].provenance == "configured":
        effective, clamp = P.effective_ubatch(cfg, None)
        if clamp:
            settings["n_ubatch"] = settings["n_ubatch"].model_copy(update={
                "value": effective,
                "reason": f"set in this model's {store_name(written)}; in force: {clamp}"})
    else:
        settings["n_ubatch"] = settings["n_ubatch"].model_copy(update={
            "value": None, "auto": None, "provenance": "unknown",
            "reason": (f"decided at load: {P.AUTO_UBATCH} when working-set "
                       f"headroom clears the thin threshold, else llama-server's "
                       f"{P.LLAMA_DEFAULT_N_UBATCH}")})

    if not cfg.get("mmproj_path"):
        image_cap = Setting(provenance="not_applicable",
                            reason="no projector: the model takes no images")
    else:
        cap, reason, level = P.image_token_cap_decision(cfg, None)
        if level is None or cfg.get("n_ubatch") is not None:
            # Causal projector, or the micro-batch is stored: the decision
            # the spawn will make is fully known now.
            image_cap = Setting(value=cap, auto=cap, provenance="derived",
                                reason=reason or "no projector type in the mmproj header")
        else:
            image_cap = Setting(provenance="unknown", reason=(
                "decided at load: depends on the micro-batch the spawn picks"))

    binary, binary_reason = P.binary_report(cfg)
    keep_alive, keep_alive_reason, _ = P.keep_alive_choice(os.environ)
    settings.update({
        "image_max_tokens": image_cap,
        "metal_keep_alive": Setting(value=keep_alive,
                                    auto=str(P.METAL_RESIDENCY_KEEP_ALIVE_S),
                                    reason=keep_alive_reason, provenance="derived"),
        "binary": Setting(value=binary, auto=binary, reason=binary_reason,
                          provenance="derived"),
    })

    return EngineDescription(
        cache=_cache_profile(cfg, settings),
        runtime=Fact(value="llama.cpp", provenance="derived",
                     source="provider gguf runs one llama-server per model"),
        context=ContextFacts(
            length=Fact(value=length, provenance="derived" if length else "unknown",
                        source="the GGUF header" if length
                        else "the GGUF header declares no context length"),
            running=Fact(provenance="unknown", source="not loaded"),
        ),
        template=_template(model_id, cfg),
        settings=settings,
        thinking=_thinking(model_id, cfg),
        speculative=_speculative(cfg, derived, written),
    )


def _thinking(model_id: str, cfg: dict):
    """The in-force template's thinking controls (plan W2): rendered from
    the template, never listed. Null when there is no template to judge."""
    from heylook_llm import chat_template_files
    from heylook_llm.thinking_controls import detect

    return detect(chat_template_files.view(model_id, "gguf", cfg).template)


_KIND_LABEL = {
    "hybrid_recurrent": ("checkpointed", "a recurrent state that cannot be truncated"),
    "sliding_window": ("checkpointed", "sliding-window attention"),
    "full_attention": ("truncates anywhere", "full attention"),
}


def _where(model_path: Path, drafter: Path) -> str:
    """Where a drafter sits relative to its model, in folder names only."""
    home = model_path if model_path.is_dir() else model_path.parent
    if drafter.parent == home:
        return "beside the weights"
    if drafter.parent.parent == home:
        return f"in subfolder {drafter.parent.name}/"
    if drafter.parent == home.parent:
        return "at the repo root, one folder up"
    if drafter.parent.parent == home.parent:
        return f"in the neighbouring folder {drafter.parent.name}/"
    return "at a path outside the model's folders"


def _speculative(cfg: dict, derived: dict, written) -> dict:
    """What spec decode this model is set up to run, before load (the
    running process's answer is describe_observed's ``in_force``). Read from
    the config the spawn uses, against what discovery derived, so a drafter
    the model's own file unset still shows as found."""
    from heylook_llm import gguf_metadata
    from heylook_llm.providers.contract import store_name

    drafter, spec = cfg.get("draft_model_path"), cfg.get("spec_type")
    found = derived.get("draft_model_path") or derived.get("spec_type")
    set_here = bool(written) and (drafter != derived.get("draft_model_path")
                                  or spec != derived.get("spec_type"))
    prov = "configured" if set_here else "derived"
    how = f"set in this model's {store_name(written)}" if set_here else "found by discovery"
    if drafter:
        path = Path(str(drafter)).expanduser()
        d = Fact(value=path.name, provenance=prov,
                 source=f"{how}, {_where(Path(str(cfg.get('model_path') or '')), path)}")
        inferred = gguf_metadata.spec_type_from_gguf(gguf_metadata.splits(path))
        t = Fact(value=spec or inferred, provenance=prov if spec else "derived",
                 source="pinned by spec_type" if spec else
                 "the drafter's own header, as llama.cpp infers it" if inferred
                 else "not inferable from the drafter's header")
    elif spec:
        d = Fact(value="built-in MTP head", provenance=prov,
                 source=f"{how}: the target's own weights carry the head")
        t = Fact(value=spec, provenance=prov, source="spec_type")
    else:
        d = Fact(value=None, provenance=prov, source=(
            f"turned off in this model's {store_name(written)}; discovery found one"
            if found and set_here else
            "none found beside the weights, in a subfolder, in the weights, or in a "
            "neighbouring folder whose header names this model"))
        t = Fact(value=None, provenance=prov, source="no drafter")
    return {"drafter": d, "type": t,
            "in_force": Fact(provenance="unknown", source="not loaded")}


def _cache_profile(cfg: dict, settings: dict) -> dict:
    """How llama-server reuses this model's prompt across requests (plan
    W5): the reuse class from the header, KV shift, the host-RAM prompt
    cache and the checkpoint settings, each as it will be spawned."""
    from heylook_llm import gguf_metadata

    P = _provider()
    out = {}
    kind = gguf_metadata.memory_kind(Path(str(cfg.get("model_path") or "")).expanduser())
    if kind:
        cls, why = _KIND_LABEL[kind]
        out["reuse_class"] = Fact(
            value=cls, provenance="derived",
            source=(f"the GGUF header says {why}: a new request reuses up to "
                    + ("the latest checkpoint at or before where it diverges"
                       if cls == "checkpointed" else "the point where it diverges")
                    + ". llama.cpp decides at load from the memory it builds, so "
                      "this is the header's answer"))
    else:
        out["reuse_class"] = Fact(provenance="unknown", source="the GGUF header is unreadable")

    if cfg.get("mmproj_path"):
        out["kv_shift"] = Fact(value=False, provenance="derived", source=(
            "llama-server forces KV shifting (cache_reuse) and context shift off "
            "whenever a projector loads"))
    else:
        shift = P.extra_arg_value(cfg.get("extra_args"), {"--cache-reuse"})
        out["kv_shift"] = Fact(
            value=bool(shift and shift != "0"), provenance="derived",
            source="--cache-reuse in extra_args" if shift
            else "llama-server's default: cache_reuse 0, off")

    # extra_args follows -cram in the argv, so it is what llama-server keeps.
    ram = cfg.get("cache_ram_mb")
    ram_extra = P.extra_arg_value(cfg.get("extra_args"), {"-cram", "--cache-ram"})
    if ram_extra is not None:
        out["ram_budget_mib"] = Fact(value=int(ram_extra), provenance="configured",
                                     source="-cram in extra_args (-1 unlimited, 0 off)")
    elif ram is not None:
        out["ram_budget_mib"] = Fact(value=ram, provenance=settings["cache_ram_mb"].provenance,
                                     source="cache_ram_mb for this model (-1 unlimited, 0 off)")
    else:
        out["ram_budget_mib"] = Fact(
            value=P.LLAMA_DEFAULT_CACHE_RAM_MIB, provenance="derived",
            source=("llama-server's default host-RAM prompt cache; an entry "
                    "larger than it is skipped, not stored"))

    for key, names, default, what in (
        ("checkpoints", {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"},
         P.LLAMA_DEFAULT_CTX_CHECKPOINTS, "context checkpoints kept per slot"),
        ("checkpoint_min_spacing", {"-cms", "--checkpoint-min-step"},
         P.LLAMA_DEFAULT_CHECKPOINT_MIN_STEP, "minimum tokens between checkpoints"),
    ):
        given = P.extra_arg_value(cfg.get("extra_args"), names)
        out[key] = Fact(value=int(given) if given is not None else default,
                        provenance="configured" if given is not None else "derived",
                        source=f"{what} (" + ("extra_args" if given is not None
                                              else "llama-server's default") + ")")
    return out

