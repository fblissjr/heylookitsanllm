# src/heylook_llm/providers/mlx_describe.py
"""MLX's static half of the engine contract (providers/contract.py).

Plain functions over the config and the model directory: never imports MLX,
never needs the model loaded. What only the loaded model can answer (the
prompt-cache verdict, the template body actually installed) is
MLXProvider.describe_observed().
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

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


@lru_cache(maxsize=1)
def _engine_build() -> str:
    """The installed mlx-lm and mlx-vlm, by commit where installed from git.

    Read from each distribution's own install record (direct_url.json), not
    from uv.lock: the lock can say one thing while the venv holds another,
    and loader routing depends on what mlx-vlm actually registers. Constant
    for the life of the process, since the libraries are imported once.
    """
    from importlib import metadata

    parts = []
    for dist in ("mlx-lm", "mlx-vlm"):
        try:
            d = metadata.distribution(dist)
        except metadata.PackageNotFoundError:
            parts.append(f"{dist}@absent")
            continue
        label = d.version
        try:
            direct = json.loads(d.read_text("direct_url.json") or "{}")
            commit = (direct.get("vcs_info") or {}).get("commit_id")
            if commit:
                label = commit[:10]
        except (ValueError, TypeError):
            pass
        parts.append(f"{dist}@{label}")
    return " ".join(parts)


def _model_dir(cfg: dict) -> Path:
    return Path(str(cfg.get("model_path") or "")).expanduser()


def static_inputs(model_id: str, cfg: dict, written: bool,
                  derived: dict) -> StaticInputs:
    from heylook_llm.providers.common.template_info import TEMPLATE_INPUT_FILES

    d = _model_dir(cfg)
    files = [str(d / name) for name in TEMPLATE_INPUT_FILES + ("config.json",)]
    source = cfg.get("chat_template_source")
    if isinstance(source, str) and ("/" in source or source.startswith("~")):
        files.append(str(Path(source).expanduser()))
    return StaticInputs(files=tuple(files), engine_build=_engine_build(),
                        config_digest=config_digest(cfg, written, derived))


def _template(model_id: str, cfg: dict) -> TemplateFacts:
    from heylook_llm import chat_template_files
    from heylook_llm.providers.common.template_info import SOURCE_FILES

    view = chat_template_files.view(model_id, "mlx", cfg)
    origin = view.origin
    path: Optional[str] = SOURCE_FILES.get(origin)
    if path is None and isinstance(origin, str) and ("/" in origin or origin.startswith("~")):
        path = public_value(origin)
    ladder = "the MLX template ladder (override > chat_template.jinja > tokenizer_config > chat_template.json)"
    return TemplateFacts(
        origin=Fact(value=public_value(origin), provenance="derived", source=ladder),
        path=Fact(value=path, provenance="derived" if path else "unknown",
                  source=ladder if path else "no template file resolved"),
        sha256=Fact(value=sha256_text(view.template),
                    provenance="derived" if view.template else "unknown",
                    source=path or "no template body resolved"),
        running_sha256=Fact(provenance="unknown", source="not loaded"),
    )


def describe_static(model_id: str, cfg: dict, config_obj: Any, *,
                    written: bool, derived: dict) -> EngineDescription:
    from heylook_llm.capabilities import model_context_length
    from heylook_llm.config import MLXModelConfig
    from heylook_llm.providers.common.loader_routing import effective_loader_for_config

    loader = effective_loader_for_config("mlx", cfg)
    override = cfg.get("context_length")
    length = model_context_length("mlx", cfg.get("model_path"), override=override)
    if isinstance(override, int) and not isinstance(override, bool) and override > 0:
        length_fact = Fact(value=length, provenance="configured",
                           source="context_length set for this model")
    else:
        length_fact = Fact(value=length, provenance="derived" if length else "unknown",
                           source="the model's config.json" if length
                           else "config.json declares no context length")

    settings = config_settings(MLXModelConfig, config_obj, written=written,
                               derived=derived, engine_default="mlx-lm/mlx-vlm")
    settings["prompt_cache"] = _prompt_cache_static(cfg)

    return EngineDescription(
        runtime=Fact(value=loader, provenance="derived" if loader else "unknown",
                     source="loader routing: modalities, the loader setting, and "
                            "whether mlx-vlm registers this model_type"),
        context=ContextFacts(
            length=length_fact,
            running=Fact(provenance="not_applicable",
                         source="MLX has no fixed context allocation"),
        ),
        template=_template(model_id, cfg),
        settings=settings,
    )


def _prompt_cache_static(cfg: dict) -> Setting:
    """What can be said about cache reuse before the model is loaded: the
    config and drafter gates (cache_defaults.static_reuse_gate, the same
    function the cache path calls). The mRoPE gate needs the loaded model,
    so a model that passes these reads unknown until then."""
    from heylook_llm.cache_defaults import resolve_cache_config, static_reuse_gate

    # The cache config the provider will run with: cache_type None (auto) is
    # resolved at load exactly like this (MLXProvider.load_model).
    resolved = {**cfg, **resolve_cache_config(cfg, log=False)}
    gate, why = static_reuse_gate(resolved, allow_reuse=not cfg.get("draft_model_path"))
    if gate is not None:
        return Setting(value=False, auto=False, provenance="derived", reason=why)
    return Setting(provenance="unknown", reason=(
        "decided at load: needs the loaded model's position state"))
