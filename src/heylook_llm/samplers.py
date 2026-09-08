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
# named sampler, nor the model config says anything. Shared by ALL providers
# -- MLX overlays it in _apply_model_defaults, the llama-server provider in
# _build_payload. Owner ruling 2026-09-01 (v1.79.60): temperature 1.0 and
# top_p 0.95, because a narrowed distribution flattens generative prose and
# low temperature was judged worse on real output; the 0.7/1.0 before it was
# a chat-sane guess, and the 0.1/512 before that made freshly imported models
# near-greedy and truncated long answers mid-sentence. The vendor layer below
# still overlays a model's own generation_config.json where one ships.
GLOBAL_SAMPLER_FLOOR = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 0,
    'min_p': 0.0,
    'max_tokens': 4096,
    'repetition_penalty': 1.0,
    'presence_penalty': 0.0,
}

# Vendor layer: the model's OWN recommended decode settings, overlaid directly
# above the floor so models.toml fields, samplers and request fields all still
# override it. Each engine reads the same values from where its models keep
# them: MLX from the model dir's generation_config.json (`load_vendor_sampling`
# below), gguf from the `general.sampling.*` block in the GGUF header
# (`gguf_metadata.vendor_sampling`), which converters write FROM that same
# generation_config.json. One concept, two spellings on disk -- gguf went
# without it until v2.0.22 on the reasoning that a gguf dir ships no
# generation_config.json, which is true and was the wrong conclusion: the
# values had moved into the header, and heylook was sending top_k 0 at models
# whose own files asked for 20 (Qwen3.6) and 64 (gemma-4).
VENDOR_SAMPLING_KEYS = ('temperature', 'top_p', 'top_k')


# Model-config / request keys the cascade resolves. Providers whose config
# class lacks a key (GGUFModelConfig has only max_tokens/default_sampler)
# simply never contribute it; unknown keys in the result are ignored by the
# consumer (gguf's payload map picks only what llama-server understands).
EFFECTIVE_SAMPLER_KEYS = (
    'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
    'repetition_penalty', 'repetition_context_size', 'presence_penalty',
    'enable_thinking', 'reasoning_effort', 'vision_tokens',
)
REQUEST_SAMPLER_FIELDS = EFFECTIVE_SAMPLER_KEYS + ('seed',)


def resolve_effective_sampling(request: Any, model_config: dict,
                               vendor: dict | None = None, *,
                               thinking_capable: bool = False) -> dict[str, Any]:
    """THE effective-request cascade, shared by every provider.

    ``thinking_capable`` is whether the MODEL can think at all (the same
    answer as the served ``thinking`` capability: gguf's ``supports_thinking``,
    MLX's template probe). It is the LAST fallback for the thinking switch --
    see the resolution order at ``thinking_active`` below -- and the caller
    passes it because the cascade has no file or template access of its own.

    Layers, later overriding earlier (each only for fields it sets):
      1.  Global floor (``GLOBAL_SAMPLER_FLOOR``).
      1b. Vendor layer -- the model's own generation_config.json values,
          passed by the caller: MLX from generation_config.json
          (``load_vendor_sampling``), gguf from the GGUF header's
          ``general.sampling.*`` (``gguf_metadata.vendor_sampling``, v2.0.22).
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

    # The thinking switch, resolved in ONE order: the request's explicit
    # value, else the model's models.toml `enable_thinking`, else whether the
    # model CAN think. That last fallback is v1.79.62: from v1.50.0 unset
    # meant OFF on both engines, chosen because the two engines disagreed on
    # unset (gguf omitted the kwarg and got the template's own default,
    # thinking-ON for gemma-4/Qwen3.6; MLX resolved it to False) and because
    # the UI could send only true-or-absent, so "unset = off" was the only
    # way to have an off switch at all. The UI now sends an explicit false,
    # so that reason is gone -- and a thinking model that silently does not
    # think unless someone finds the models.toml flag was the standing
    # complaint (2026-09-04). A model that cannot think still resolves to
    # OFF, so the thinking sampler overlay below never fires on one.
    #
    # Materialized ALWAYS, not only when the overlay fires: gguf's payload
    # builder only sends chat_template_kwargs for a non-None value, so an
    # absent key is not "no opinion" downstream -- both engines must read the
    # same resolved bool, and `thinking_default()` reports this same answer.
    request_thinking = getattr(request, 'enable_thinking', None)
    config_thinking = model_config.get('enable_thinking')
    thinking_active = bool(
        request_thinking if request_thinking is not None
        else config_thinking if config_thinking is not None
        else thinking_capable
    )
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



class _NoRequest:
    """The empty request: every cascade field absent (getattr -> None)."""

    def __getattr__(self, name):  # pragma: no cover - trivial
        return None


def thinking_default(model_config: dict, *, thinking_capable: bool) -> bool:
    """What thinking resolves to when a request says NOTHING about it.

    The cascade's own answer for an empty request -- a `default_sampler`
    naming the thinking sampler, the models.toml `enable_thinking` flag and
    the capability fallback all count, exactly as they do at generation
    time. Reported on the admin row (`thinking_default`) so a UI can label
    its "model default" choice with the value it actually means instead of
    leaving the user to find out by generating.
    """
    return bool(resolve_effective_sampling(
        _NoRequest(), model_config, thinking_capable=thinking_capable,
    ).get('enable_thinking'))


class _ThinkingRequest(_NoRequest):
    """An empty request that states the thinking switch and nothing else."""

    def __init__(self, enable_thinking: bool):
        self.enable_thinking = enable_thinking


def sampler_defaults(model_config: dict, *, thinking_capable: bool,
                     vendor: dict | None = None) -> dict[str, dict[str, Any]]:
    """What EVERY sampler key resolves to when a request says nothing.

    Sibling of :func:`thinking_default` and bound by the same rule: this is
    the cascade's own answer, run for real, never a re-derivation. Reported
    on the admin row and ``/v1/models`` so the settings panel can show a
    blank field's actual value instead of the word "auto" -- a user who has
    to generate to find out what temperature they are running is the
    complaint this closes.

    Keyed by the THINKING SWITCH, ``{"off": {...}, "on": {...}}``, because
    the anti-loop overlay fires off that switch (thinking.toml sets
    presence_penalty) while the panel's thinking control is live and
    independent. Reporting one state's numbers while the user has selected
    the other would put a WRONG number on screen, which is worse than the
    "auto" it replaces. Two dict merges and no I/O, so the honest shape is
    also the cheap one.

    ``vendor`` must be passed exactly as the model's own PROVIDER passes it
    at generation time -- MLX from the model dir's generation_config.json
    (``load_vendor_sampling``), gguf from the GGUF header's
    ``general.sampling.*`` (``gguf_metadata.vendor_sampling``, v2.0.22).
    temperature/top_p/top_k are precisely the vendor keys, so omitting it for
    an engine that has one reports the global floor for every model that
    overrides it -- the models where the number matters most.
    ``capabilities._vendor_sampling_pairs`` is the one place that pairing
    lives; it drifted once already, within a commit.
    """
    out: dict[str, dict[str, Any]] = {}
    for state, active in (("off", False), ("on", True)):
        merged = resolve_effective_sampling(
            _ThinkingRequest(active), model_config, vendor,
            thinking_capable=thinking_capable,
        )
        out[state] = {k: merged[k] for k in REQUEST_SAMPLER_FIELDS if k in merged}
    return out


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
