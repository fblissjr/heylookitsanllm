"""Runtime preset registry (C1 of S1.2b).

Presets are bundles of sampler knobs (``temperature``, ``top_p``, ``top_k``,
``min_p``, ``max_tokens``, ``repetition_penalty``, ``repetition_context_size``,
``presence_penalty``, ``seed``, ``enable_thinking``) that get resolved at
request time, not baked into ``models.toml`` at import time.

Each preset lives as its own TOML file under
``src/heylook_llm/data/presets/`` with the shape::

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
``apply_preset`` — unset keys pass through from previous layers.

Cascade order in ``MLXProvider._apply_model_defaults``::

    1. Global hardcoded floor
    2. Model sampler fields (``models.toml`` per-model overrides)
    3. Request's preset (if ``ChatRequest.preset`` is set)  <- this module
    4. Request-level explicit field values

Keeping per-model sampler fields in the cascade (layer 2) is intentional --
some models genuinely want non-standard defaults, and a request can still
override them. The preset layer is the new surface for "users choose how
verbose / creative / deterministic this turn is" without editing
``models.toml``.
"""

from __future__ import annotations

import logging
import threading
import tomllib
from pathlib import Path
from typing import Any, Iterable


_BUNDLED_DIR = Path(__file__).resolve().parent / "data" / "presets"


class PresetNotFound(KeyError):
    """Raised when a preset name is not registered."""


class PresetRegistry:
    """In-memory map of preset name -> defaults dict + descriptions.

    Instances are cheap; the module-level ``get_preset_registry()`` returns
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
    def from_directory(cls, directory: Path | str) -> "PresetRegistry":
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
    def from_bundled(cls) -> "PresetRegistry":
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
            raise PresetNotFound(
                f"preset {name!r} not found; known: {self.list_names()}"
            )
        return dict(self._presets[name])

    # ---- cascade helper ----

    def apply_preset(
        self, merged_config: dict[str, Any], name: str | None
    ) -> None:
        """Overlay preset fields onto ``merged_config`` in place.

        ``name=None`` is a no-op so the cascade can call this unconditionally
        without an if/else at every call site. An unknown preset name raises
        ``PresetNotFound`` -- silent fallback would mask typos.
        """
        if name is None:
            return
        if name not in self._presets:
            raise PresetNotFound(
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
_SINGLETON: PresetRegistry | None = None


def get_preset_registry() -> PresetRegistry:
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
            _SINGLETON = PresetRegistry.from_bundled()
    return _SINGLETON


def reset_preset_registry_for_test(replacement: PresetRegistry | None = None) -> None:
    """Test hook: replace or clear the memoized singleton.

    Production code should never call this; it exists so tests can swap in
    a registry built from a ``tmp_path`` directory.
    """
    global _SINGLETON
    with _LOCK:
        _SINGLETON = replacement


def known_preset_names() -> Iterable[str]:
    """Convenience for diagnostics / API surfaces that want the list."""
    return get_preset_registry().list_names()
