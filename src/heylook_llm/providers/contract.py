# src/heylook_llm/providers/contract.py
"""One engine contract (plan W13).

Every engine answers the same questions about a model, and ``/v1/models``,
the admin row and the frontend read ONE shape, ``EngineDescription``, with no
engine switches. Each engine answers in two halves:

- a STATIC half (``mlx_describe`` / ``gguf_describe``): plain functions over
  the config and the model's files, so it answers for models that are not
  loaded. Cached by a stamp over every input that can change the answer
  (``StaticInputs``): stat-only, never a header read;
- an OBSERVED half (``provider.describe_observed()``): what the running
  process decided at load, read back from fields ``load_model`` recorded. No
  call to the process, no lock.

Every value carries its provenance, so a consumer never infers it from which
half produced it. A slot a later workstream fills is an explicit null until
that workstream reports it (see each field's description).

Nothing here carries an absolute path: ``/v1/models`` is read by LAN clients.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

Provenance = Literal[
    "derived",          # computed from config and files
    "configured",       # stored for this model (a models.toml entry today)
    "observed",         # reported by the running process
    "observed_cached",  # observed at an earlier load, cached on disk
    "unknown",          # not known yet (e.g. decided at load, model not loaded)
    "not_applicable",   # the question does not apply to this engine
]

T = TypeVar("T")


class Fact(BaseModel, Generic[T]):
    """One value, where it came from, and a sentence saying so."""
    value: Optional[T] = Field(default=None)
    provenance: Provenance = Field(default="unknown")
    source: Optional[str] = Field(
        default=None,
        description="Where the value came from, written for a person to read. "
                    "Never an absolute path.")


class ContextFacts(BaseModel):
    length: Fact[int] = Field(description="The context ceiling the model's files declare.")
    running: Fact[int] = Field(
        description="What the running process was sized to. Not applicable on "
                    "MLX, which has no fixed allocation.")


class TemplateFacts(BaseModel):
    origin: Fact[str] = Field(description="Which rung of the template ladder won.")
    path: Fact[str] = Field(description="The file in force (a basename); null when embedded.")
    sha256: Fact[str] = Field(description="The body that will be used at the next load.")
    running_sha256: Fact[str] = Field(
        description="The body the running process loaded. Differs from "
                    "sha256 when the template changed since load.")
    prefix_stable: Fact[bool] = Field(
        default_factory=lambda: Fact(provenance="unknown", source="not checked"),
        description="Whether each turn's prompt extends the previous one's, "
                    "which a prompt cache needs (plan W3's lint, "
                    "chat_template_files.prefix_stability). false = every "
                    "turn re-processes history; the source says where.")


def prefix_stable_fact(body: Optional[str]) -> "Fact[bool]":
    """The lint's answer as a Fact, the same way on every engine."""
    from ..chat_template_files import prefix_stability

    stable, why = prefix_stability(body)
    return Fact(value=stable, provenance="derived" if stable is not None else "unknown",
                source=f"rendered in an engine-mirroring jinja environment: {why}")


class Setting(BaseModel):
    """One setting: the value in force, what is stored, what auto picks, why."""
    value: Any = Field(default=None, description="In force (or, unloaded, what the next load uses).")
    configured: Any = Field(default=None, description="Stored for this model, or null.")
    auto: Any = Field(default=None, description="What heylook or the engine picks when nothing is stored.")
    reason: str = Field(description="Why the value is what it is, for a person to read.")
    provenance: Provenance
    effect: Optional[str] = Field(
        default=None,
        description="When a change takes effect (config.EFFECT_CLASSES); "
                    "null for a load decision with no config field.")


class EngineDescription(BaseModel):
    """What one model runs on, with what, and why. Same keys on every engine."""
    runtime: Fact[str] = Field(description="The engine library: mlx-vlm or llama.cpp.")
    context: ContextFacts
    template: TemplateFacts
    settings: Dict[str, Setting] = Field(
        description="Every config field plus the load decisions with no field, "
                    "keyed by name.")
    cache: Optional[Dict[str, Fact]] = Field(
        default=None,
        description="Cache profile (plan W5): how this model reuses a prompt "
                    "across requests, each fact with its provenance.")
    thinking: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Thinking controls the in-force template offers (plan W2, "
                    "thinking_controls.detect): {switch, depth: {variable, "
                    "values, aliases, default, unknown, changes_prefix}}. "
                    "Values are the template's own spellings. Null when there "
                    "is no template to judge.")
    image: Optional[Dict[str, Any]] = Field(
        default=None, description="Image geometry (plan W4). Null until reported.")
    speculative: Optional[Dict[str, Fact]] = Field(
        default=None,
        description="Speculative decoding: `drafter` (the file, the built-in "
                    "MTP head, or none, and where discovery found it), `type` "
                    "(the draft type llama.cpp runs), and `in_force` (whether "
                    "the running process drafts, and why not when it does not: "
                    "the drafter did not fit, did not load, or is unset).")
    steering: Optional[Dict[str, Any]] = Field(
        default=None, description="Activation steering (plan W14). Null until reported.")


@dataclass
class Observed:
    """What a loaded provider reports about itself (its describe_observed())."""
    context_running: Optional[Fact] = None
    loaded_template: Optional[str] = None
    settings: Dict[str, Setting] = field(default_factory=dict)
    cache: Dict[str, Fact] = field(default_factory=dict)
    speculative: Dict[str, Fact] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers the engine describers share
# ---------------------------------------------------------------------------

def public_value(value: Any) -> Any:
    """``value`` with any filesystem path reduced to its basename.

    Path-valued settings (mmproj_path, chat_template_path, server_binary,
    paths inside extra_args) would otherwise put home paths on /v1/models.
    Decided by the VALUE, not a list of field names, so a new path field is
    covered without anyone remembering to add it.
    """
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str) and (os.path.isabs(value) or value.startswith("~")):
        return Path(value).name
    if isinstance(value, (list, tuple)):
        return [public_value(v) for v in value]
    if isinstance(value, dict):
        return {k: public_value(v) for k, v in value.items()}
    return value


def sha256_text(text: Optional[str]) -> Optional[str]:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def config_digest(config: dict, written: bool, derived: Optional[dict] = None) -> str:
    blob = (json.dumps(config, sort_keys=True, default=str) + f"|{written}|"
            + json.dumps(derived or {}, sort_keys=True, default=str))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StaticInputs:
    """Every input the static half reads. A missing file still counts, so
    creating one (an operator template override, say) invalidates."""
    files: tuple
    engine_build: Optional[str]
    config_digest: str


def stamp(inputs: StaticInputs) -> tuple:
    parts = []
    for f in inputs.files:
        try:
            st = os.stat(f)
            parts.append((f, st.st_mtime_ns, st.st_size))
        except OSError:
            parts.append((f, None, None))
    return (tuple(parts), inputs.engine_build, inputs.config_digest)


def _looks_like_path(v: Any) -> bool:
    return isinstance(v, str) and (os.path.isabs(v) or v.startswith("~"))


def _same(a: Any, b: Any) -> bool:
    """Equal values, with two spellings of one file counted as equal."""
    if a == b:
        return True
    if _looks_like_path(a) and _looks_like_path(b):
        try:
            return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
        except (OSError, ValueError):
            return False
    return False


DEFAULT_STORE = "models.toml entry"


def store_name(written: Any) -> str:
    """Where a model's stored values live, for a reason string."""
    return written if isinstance(written, str) else DEFAULT_STORE


def config_settings(config_cls: type, config_obj: Any, *, written: Any,
                    derived: Optional[dict], engine_default: str) -> Dict[str, Setting]:
    """One Setting per configurable field of ``config_cls``.

    The field list is DERIVED (config.configurable_fields), never listed.
    A value counts as CONFIGURED only when a models.toml entry stores it AND
    it differs from what derivation gives that file (``derived``, the config
    discovery builds for the same path, else the schema default). Admin edits
    materialize an entry with every derived value in it, so "the entry has
    this key" would light up values nobody chose; a stored value equal to the
    derived one reads as derived, marked as a stored copy (what the
    registry-sidecars prune removes). A discovered model stores nothing.
    """
    from heylook_llm.config import configurable_fields, field_effect

    explicit = set(getattr(config_obj, "model_fields_set", set())) if written else set()
    derived = derived or {}
    out: Dict[str, Setting] = {}
    for name in sorted(configurable_fields(config_cls)):
        info = config_cls.model_fields[name]
        default = info.get_default(call_default_factory=True)
        current = getattr(config_obj, name, default)
        effect = field_effect(info)
        baseline = derived[name] if name in derived else default
        if name in explicit and not _same(current, baseline):
            out[name] = Setting(
                value=public_value(current), configured=public_value(current),
                auto=public_value(baseline),
                reason=f"set in this model's {store_name(written)}",
                provenance="configured", effect=effect)
        elif name in explicit:
            out[name] = Setting(
                value=public_value(current), auto=public_value(baseline),
                reason=f"stored in this model's {store_name(written)}, same as derived",
                provenance="derived", effect=effect)
        elif not _same(current, default):
            out[name] = Setting(
                value=public_value(current), auto=public_value(current),
                reason="derived from the model's own files", provenance="derived",
                effect=effect)
        else:
            reason = (f"not set; {engine_default} decides" if default is None
                      else "not set; heylook's default")
            out[name] = Setting(
                value=public_value(default), auto=public_value(default),
                reason=reason, provenance="derived", effect=effect)
    return out


def with_observed(static: EngineDescription, observed: Observed) -> EngineDescription:
    desc = static.model_copy(deep=True)
    if observed.context_running is not None:
        desc.context.running = observed.context_running
    desc.template.running_sha256 = Fact(
        value=sha256_text(observed.loaded_template),
        provenance="observed" if observed.loaded_template else "unknown",
        source="the template body read when this process loaded"
        if observed.loaded_template else "the running process recorded no template body")
    desc.settings.update(observed.settings)
    if observed.cache:
        desc.cache = {**(desc.cache or {}), **observed.cache}
    if observed.speculative:
        desc.speculative = {**(desc.speculative or {}), **observed.speculative}
    return desc


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def _describers() -> dict:
    # Imported lazily: neither describer imports MLX, but providers/__init__
    # must not pull either in at import time.
    from heylook_llm.providers import gguf_describe, mlx_describe
    return {"mlx": mlx_describe, "gguf": gguf_describe}


def describer_for(provider: str):
    """The engine describer for a provider key, or None for an unknown one."""
    return _describers().get(provider)


_STATIC_CACHE: Dict[tuple, tuple] = {}
_STATIC_LOCK = threading.Lock()


def _provenance_inputs(model_id: str, router: Any) -> tuple[Any, dict]:
    """(written, derived) for one model, from what the router recorded at
    its last config load. ``written`` is where the model's stored values live
    (a models.toml entry, or its model.heylook.toml), or False."""
    written_ids = getattr(router, "written_ids", None) if router is not None else None
    derived_all = getattr(router, "derived_configs", None) if router is not None else None
    derived = derived_all.get(model_id, {}) if isinstance(derived_all, dict) else {}
    stored_in = getattr(router, "stored_in", None) if router is not None else None
    if isinstance(stored_in, dict) and model_id in stored_in:
        return stored_in[model_id], derived
    if isinstance(written_ids, (set, frozenset)):
        return (DEFAULT_STORE if model_id in written_ids else False), derived
    # Unknown (no router, or a stand-in without the attribute): read the
    # entry's keys against the schema defaults, as an entry-backed row.
    return DEFAULT_STORE, derived


_THINKING_CACHE: Dict[tuple, tuple] = {}


def thinking_controls(model_config: Any) -> Optional[dict]:
    """The model's thinking controls, for callers that need only those (the
    capability report, the routes' pre-stream check). The same detection the
    describers put on ``engine.thinking``, cached behind the describer's
    stat-only stamp so a request never re-reads a template that has not
    changed."""
    from heylook_llm.capabilities import config_dict
    from heylook_llm import chat_template_files
    from heylook_llm.thinking_controls import detect

    describer = _describers().get(model_config.provider)
    if describer is None:
        return None
    cfg = config_dict(model_config.config)
    key = (model_config.provider, model_config.id)
    current = stamp(describer.static_inputs(model_config.id, cfg, True, {}))
    with _STATIC_LOCK:
        hit = _THINKING_CACHE.get(key)
    if hit is not None and hit[0] == current:
        return hit[1]
    value = detect(chat_template_files.view(model_config.id, model_config.provider, cfg).template)
    with _STATIC_LOCK:
        _THINKING_CACHE[key] = (current, value)
    return value


def describe(model_config: Any, router: Any = None) -> EngineDescription:
    """The one description ``/v1/models`` and the admin row both carry."""
    from heylook_llm.capabilities import config_dict

    describer = _describers()[model_config.provider]
    cfg = config_dict(model_config.config)
    written, derived = _provenance_inputs(model_config.id, router)
    key = (model_config.provider, model_config.id)
    current = stamp(describer.static_inputs(model_config.id, cfg, written, derived))
    with _STATIC_LOCK:
        hit = _STATIC_CACHE.get(key)
    if hit is not None and hit[0] == current:
        static = hit[1]
    else:
        static = describer.describe_static(model_config.id, cfg, model_config.config,
                                           written=written, derived=derived)
        with _STATIC_LOCK:
            _STATIC_CACHE[key] = (current, static)

    live = None
    if router is not None:
        try:
            live = router.get_loaded_models().get(model_config.id)
        except Exception:  # noqa: BLE001 -- a description never fails a listing
            live = None
    observe = getattr(live, "describe_observed", None)
    if callable(observe):
        observed = observe()
        if isinstance(observed, Observed):
            return with_observed(static, observed)
    return static
