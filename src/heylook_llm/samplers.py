"""Named sampler registry ("samplers": bundled sampler-setting configs).

Terminology (2026-07-20): these were called "presets" until the name collided
with the /v1/presets user-preset system (v3's saved prompt+sampler bundles,
DuckDB-backed, client-expanded). A "sampler" here is a named, versioned
sampler-settings bundle shipped with the server.

Presets are bundles of sampler knobs (``temperature``, ``top_p``, ``top_k``,
``min_p``, ``max_tokens``, ``repetition_penalty``, ``repetition_context_size``,
``presence_penalty``, ``seed``, ``enable_thinking``) that get resolved at
request time, not baked into ``models.toml`` at import time.

Each preset lives as its own TOML file under
``src/heylook_llm/data/samplers/`` with the shape::

    [meta]
    name = "balanced"
    description = "Middle ground on temperature and output length"

    [defaults]
    temperature = 0.7
    top_k = 40
    min_p = 0.05
    max_tokens = 512
    repetition_penalty = 1.05

The registry loads every ``.toml`` under the presets directory on startup
(malformed files are logged and skipped, never fatal). Callers look up a
preset by name and overlay its fields onto a cascade dict via
``apply_sampler`` — unset keys pass through from previous layers.

Cascade order in ``MLXProvider._apply_model_defaults``::

    1. Global hardcoded floor
    2. Model sampler fields (``models.toml`` per-model overrides)
    3. Request's sampler (if ``ChatRequest.sampler`` is set)  <- this module
    4. Request-level explicit field values

Keeping per-model sampler fields in the cascade (layer 2) is intentional --
some models genuinely want non-standard defaults, and a request can still
override them. The preset layer is the new surface for "users choose how
verbose / creative / deterministic this turn is" without editing
``models.toml``.
"""

from __future__ import annotations

import json
import logging
import threading
import tomllib
from pathlib import Path
from typing import Any, Iterable


_BUNDLED_DIR = Path(__file__).resolve().parent / "data" / "samplers"


class SamplerNotFound(KeyError):
    """Raised when a preset name is not registered."""


class SamplerRegistry:
    """In-memory map of preset name -> defaults dict + descriptions.

    Instances are cheap; the module-level ``get_sampler_registry()`` returns
    a memoized singleton that loads the bundled presets directory once.
    """

    def __init__(
        self,
        presets: dict[str, dict[str, Any]],
        descriptions: dict[str, str] | None = None,
    ):
        self._presets = dict(presets)
        self._descriptions = dict(descriptions or {})

    # ---- constructors ----

    @classmethod
    def from_directory(cls, directory: Path | str) -> "SamplerRegistry":
        """Load every ``*.toml`` under ``directory``. Malformed files are
        logged and skipped."""
        path = Path(directory)
        presets: dict[str, dict[str, Any]] = {}
        descriptions: dict[str, str] = {}
        if not path.is_dir():
            return cls(presets, descriptions)

        for toml_path in sorted(path.glob("*.toml")):
            parsed = cls._parse_one(toml_path)
            if parsed is None:
                continue
            name, defaults, description = parsed
            if name in presets:
                logging.warning(
                    "preset name collision: %r from %s already registered; "
                    "skipping duplicate",
                    name,
                    toml_path,
                )
                continue
            presets[name] = defaults
            if description:
                descriptions[name] = description
        return cls(presets, descriptions)

    @classmethod
    def from_bundled(cls) -> "SamplerRegistry":
        """Load the presets shipped with the package."""
        return cls.from_directory(_BUNDLED_DIR)

    # ---- query ----

    def __contains__(self, name: str) -> bool:
        return name in self._presets

    def list_names(self) -> list[str]:
        return sorted(self._presets.keys())

    def describe(self, name: str) -> str:
        """Return the preset's [meta].description, or '' if unset/unknown."""
        return self._descriptions.get(name, "")

    def list_info(self) -> list[dict[str, str]]:
        """Return ``[{name, description}, ...]`` for API surfaces."""
        return [
            {"name": name, "description": self._descriptions.get(name, "")}
            for name in self.list_names()
        ]

    def get(self, name: str) -> dict[str, Any]:
        if name not in self._presets:
            raise SamplerNotFound(
                f"preset {name!r} not found; known: {self.list_names()}"
            )
        return dict(self._presets[name])

    # ---- cascade helper ----

    def apply_sampler(
        self, merged_config: dict[str, Any], name: str | None
    ) -> None:
        """Overlay preset fields onto ``merged_config`` in place.

        ``name=None`` is a no-op so the cascade can call this unconditionally
        without an if/else at every call site. An unknown preset name raises
        ``SamplerNotFound`` -- silent fallback would mask typos.
        """
        if name is None:
            return
        if name not in self._presets:
            raise SamplerNotFound(
                f"preset {name!r} not found; known: {self.list_names()}"
            )
        merged_config.update(self._presets[name])

    # ---- internals ----

    @staticmethod
    def _parse_one(toml_path: Path) -> tuple[str, dict[str, Any], str] | None:
        try:
            with toml_path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            logging.warning("skipping malformed preset %s: %s", toml_path, exc)
            return None
        except OSError as exc:
            logging.warning("skipping unreadable preset %s: %s", toml_path, exc)
            return None

        meta = data.get("meta") or {}
        name = meta.get("name") or toml_path.stem
        description = meta.get("description") or ""
        defaults = data.get("defaults") or {}
        if not isinstance(defaults, dict):
            logging.warning(
                "preset %s: [defaults] is not a table; treating as empty",
                toml_path,
            )
            defaults = {}
        cleaned = {k: v for k, v in defaults.items() if v is not None}
        return name, cleaned, description


_LOCK = threading.Lock()
_SINGLETON: SamplerRegistry | None = None


def get_sampler_registry() -> SamplerRegistry:
    """Memoized accessor for the process-wide preset registry.

    First call loads the bundled presets directory. Subsequent calls return
    the same instance -- presets are read-only after startup, so caching is
    safe and avoids re-parsing TOML on every request.
    """
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            _SINGLETON = SamplerRegistry.from_bundled()
    return _SINGLETON


def reset_sampler_registry_for_test(replacement: SamplerRegistry | None = None) -> None:
    """Test hook: replace or clear the memoized singleton.

    Production code should never call this; it exists so tests can swap in
    a registry built from a ``tmp_path`` directory.
    """
    global _SINGLETON
    with _LOCK:
        _SINGLETON = replacement


def known_preset_names() -> Iterable[str]:
    """Convenience for diagnostics / API surfaces that want the list."""
    return get_sampler_registry().list_names()


# Layer-1 sampler floor: what a request gets when neither the request, a
# named sampler, nor the model config says anything. Chat-sane values (the
# old 0.1/512 floor made freshly imported models near-greedy and truncated
# long answers mid-sentence). Shared by ALL providers -- MLX overlays it in
# _apply_model_defaults, the llama-server provider in _build_payload.
GLOBAL_SAMPLER_FLOOR = {
    'temperature': 0.7,
    'top_p': 1.0,
    'top_k': 0,
    'min_p': 0.0,
    'max_tokens': 4096,
    'repetition_penalty': 1.0,
    'presence_penalty': 0.0,
}

# Vendor layer: a model dir's generation_config.json carries the vendor's
# recommended decode settings (gemma-4: 1.0/64/0.95; Qwen3 thinking models:
# 0.6/20/0.95). Providers overlay it directly above the floor, so models.toml
# fields, samplers, and request fields all still override it.
VENDOR_SAMPLING_KEYS = ('temperature', 'top_p', 'top_k')


# Model-config / request keys the cascade resolves. Providers whose config
# class lacks a key (GGUFModelConfig has only max_tokens/default_sampler)
# simply never contribute it; unknown keys in the result are ignored by the
# consumer (gguf's payload map picks only what llama-server understands).
EFFECTIVE_SAMPLER_KEYS = (
    'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
    'repetition_penalty', 'repetition_context_size', 'presence_penalty',
    'enable_thinking', 'vision_tokens',
)
REQUEST_SAMPLER_FIELDS = EFFECTIVE_SAMPLER_KEYS + ('seed',)


def resolve_effective_sampling(request: Any, model_config: dict,
                               vendor: dict | None = None) -> dict[str, Any]:
    """THE effective-request cascade, shared by every provider.

    Layers, later overriding earlier (each only for fields it sets):
      1.  Global floor (``GLOBAL_SAMPLER_FLOOR``).
      1b. Vendor layer -- the model's own generation_config.json values,
          passed by the caller (``load_vendor_sampling``); gguf passes None.
      2.  Thinking anti-loop overlay (the slimmed 'thinking' sampler),
          keyed on the EFFECTIVE switch: request.enable_thinking when
          present, else the model config flag. Hardcoded fallback mirrors
          thinking.toml so inference survives the file's removal.
      3.  Model sampler fields from models.toml.
      3b. Model default_sampler -- only when the request names no sampler;
          unknown name logs-and-skips (models validate at startup, so a
          miss here is post-startup registry drift, not a request error).
      4.  Request sampler -- unknown name raises SamplerNotFound (route
          handlers translate to HTTP 400).
      5.  Request explicit fields -- always win.
    """
    merged = dict(GLOBAL_SAMPLER_FLOOR)
    if vendor:
        merged.update(vendor)

    registry = get_sampler_registry()

    request_thinking = getattr(request, 'enable_thinking', None)
    thinking_active = bool(request_thinking if request_thinking is not None
                           else model_config.get('enable_thinking', False))
    # Materialize the effective switch ALWAYS, not only when the overlay
    # below fires. Downstream, an absent key is not "no opinion": gguf's
    # payload builder only sends chat_template_kwargs for a non-None value,
    # so an omitted key handed llama-server's --jinja run to the GGUF's own
    # template default -- thinking ON for gemma-4/Qwen3.6/DeepSeek-V4 --
    # while MLX resolved the very same unset request to False. One v3
    # checkbox, opposite meanings per engine, and no way to turn thinking
    # off on a gguf model at all. Unset means OFF; both engines now say so.
    merged['enable_thinking'] = thinking_active
    if thinking_active:
        if 'thinking' in registry:
            registry.apply_sampler(merged, 'thinking')
        else:
            merged.update({'presence_penalty': 1.5, 'enable_thinking': True})

    merged.update({k: v for k, v in model_config.items()
                   if k in EFFECTIVE_SAMPLER_KEYS and v is not None})

    request_sampler = getattr(request, 'sampler', None)
    if not request_sampler:
        default_sampler = model_config.get('default_sampler')
        if default_sampler:
            if default_sampler in registry:
                registry.apply_sampler(merged, default_sampler)
            else:
                logging.warning(
                    "model default_sampler %r not in registry; skipping layer",
                    default_sampler,
                )
    registry.apply_sampler(merged, request_sampler)

    for field in REQUEST_SAMPLER_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            merged[field] = value
    return merged


def load_vendor_sampling(model_path: str) -> dict[str, Any]:
    """Sampling defaults from ``<model_path>/generation_config.json``.

    Best-effort by design: a missing file, malformed JSON, or non-numeric
    values yield {}/are dropped -- a broken vendor file must never block a
    model load.
    """
    try:
        with open(Path(model_path) / 'generation_config.json', 'rb') as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in VENDOR_SAMPLING_KEYS:
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    return out
