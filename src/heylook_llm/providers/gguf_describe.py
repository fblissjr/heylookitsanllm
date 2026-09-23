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
    Setting,
    StaticInputs,
    TemplateFacts,
    config_digest,
    config_settings,
    public_value,
    sha256_text,
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
    if settings["n_ubatch"].provenance != "configured":
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
    )
