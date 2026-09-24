# src/heylook_llm/model_service.py
"""
Service layer for model discovery, validation, and configuration management.

Provides CRUD operations on models.toml and on a model's own
model.heylook.toml, plus validation. Thread-safe for concurrent API access.
"""

import copy
import logging
import os
import re
import shutil
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tomli_w  # type: ignore[import-untyped]

from heylook_llm.config import (
    PROVIDER_CONFIG_CLASSES,
    AppConfig,
    ModelConfig,
    reload_required_fields,
)
from heylook_llm.toml_comments import merge_comments

logger = logging.getLogger(__name__)


# =============================================================================
# Smart defaults
# =============================================================================


# Fields that require a model reload vs runtime-changeable.
#
# DERIVED per provider from the `effect` metadata on each config field (see
# config.py) -- do not hand-edit. The old hand-written frozenset was
# MLX-shaped, so it named no gguf load-time field at all: changing `ctx_size`
# on a loaded gguf model reported "no reload required" and the server kept
# serving the old argv. It also still listed `supports_thinking`, which was
# removed from MLXModelConfig in v1.46.0.
#
# Kept as a module-level name because callers without a provider in hand still
# need "is this field reload-required for ANY provider"; prefer
# `reload_required_for(provider)` when the provider IS known.
RELOAD_REQUIRED_FIELDS: frozenset = frozenset().union(
    *(reload_required_fields(cls) for cls in PROVIDER_CONFIG_CLASSES.values())
)


def reload_required_for(provider: Optional[str]) -> frozenset:
    """Reload-required fields for one provider, or the union if unknown.

    The union is the conservative fallback: over-reporting costs a needless
    reload prompt, under-reporting silently serves a stale process.
    """
    cls = PROVIDER_CONFIG_CLASSES.get(provider or "")
    return reload_required_fields(cls) if cls is not None else RELOAD_REQUIRED_FIELDS

RUNTIME_CHANGEABLE_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "repetition_penalty",
        "presence_penalty",
        "enable_thinking",
        "repetition_context_size",
    }
)

# Valid model ID pattern: alphanumeric, hyphens, underscores, dots, slashes
MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-/]*$")


@dataclass
class ValidationResult:
    """Result of config validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PathValidation:
    """Result of path validation."""

    valid: bool
    resolved_path: str = ""
    error: str = ""


class ModelService:
    """Service layer for model discovery, validation, and config management.

    Thread-safe: all config mutations go through _lock. Reads are lock-free.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if self.config_path.suffix != ".toml":
            self.config_path = self.config_path.with_suffix(".toml")
        self._lock = threading.Lock()

    # --- TOML I/O ---

    def _read_toml(self) -> dict:
        """Read and parse the models.toml file."""
        if not self.config_path.exists():
            return {"models": [], "default_model": "none", "max_loaded_models": 1}
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def _write_toml(self, data: dict) -> None:
        """Atomic write: write to .tmp, validate, rename. Creates backup.

        tomli_w's render is authoritative for values; comments from the
        existing file are carried onto it best-effort (see toml_comments) --
        a failed carry degrades to a comment-less write, never a refusal.
        """
        from heylook_llm.model_registry import refuse_if_readonly

        refuse_if_readonly(self.config_path.name)
        tmp_path = self.config_path.with_suffix(".toml.tmp")
        backup_path = self.config_path.with_suffix(".toml.bak")

        toml_text = tomli_w.dumps(data)
        if self.config_path.exists():
            try:
                old_text = self.config_path.read_text(encoding="utf-8")
            except OSError:
                old_text = None
            if old_text:
                toml_text = merge_comments(old_text, toml_text)

        # Write to temp file
        toml_bytes = toml_text.encode("utf-8")
        tmp_path.write_bytes(toml_bytes)

        # Validate the written file can be parsed back
        try:
            with open(tmp_path, "rb") as f:
                parsed = tomllib.load(f)
            # Validate it produces a valid AppConfig. `models` is defaulted in
            # rather than required: since v1.69.0 a models.toml carrying only
            # [scan] is a legitimate config (everything is discovered), and
            # AppConfig.models is a REQUIRED field, so validating the raw
            # parse rejected the entry-less file the registry design promotes.
            # Saving watch folders on a fresh install failed on exactly this.
            AppConfig(**{"models": [], **parsed})
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Generated TOML failed validation: {e}") from e

        # Backup existing config
        if self.config_path.exists():
            shutil.copy2(self.config_path, backup_path)

        # Atomic rename
        tmp_path.rename(self.config_path)
        logger.info(f"Config written to {self.config_path}")

    # --- [scan] watch folders -------------------------------------------

    def get_scan_config(self) -> dict:
        """The ``[scan]`` table as stored, with defaults filled in."""
        raw = self._read_toml().get("scan") or {}
        return {
            "folders": [str(f) for f in (raw.get("folders") or [])],
            "scan_interval_seconds": int(raw.get("scan_interval_seconds", 900)),
        }

    def set_scan_config(self, folders=None, scan_interval_seconds=None) -> dict:
        """Update ``[scan]``; only the arguments given are changed.

        These folders decide what the server SERVES (model_registry), so this
        is a real config write, not a UI preference -- it goes through
        _write_toml like any other, which means the comment carry and the
        atomic replace apply. It deliberately does NOT reload the router: the
        caller decides, because turning a folder on can add many models at
        once and the cost of that belongs where it can be reported.
        """
        with self._lock:
            data = self._read_toml()
            scan = dict(data.get("scan") or {})
            if folders is not None:
                cleaned = [str(f).strip() for f in folders if str(f).strip()]
                # Preserve order, drop repeats: two spellings of one folder
                # would just make discovery walk it twice.
                seen, unique = set(), []
                for f in cleaned:
                    if f not in seen:
                        seen.add(f)
                        unique.append(f)
                scan["folders"] = unique
            if scan_interval_seconds is not None:
                interval = int(scan_interval_seconds)
                if interval < 0:
                    raise ValueError("scan_interval_seconds must be >= 0 "
                                     "(0 disables scanning entirely)")
                scan["scan_interval_seconds"] = interval
            data["scan"] = scan
            self._write_toml(data)
        return self.get_scan_config()

    # --- Config CRUD ---

    def list_configs(self) -> list[ModelConfig]:
        """List all model configs WRITTEN DOWN (including disabled).

        Deliberately models.toml only -- do NOT fold discovery in here
        (found by review after an earlier version did): rescanning per call
        makes the admin surface disagree with the router in the WORSE
        direction, listing a model downloaded after startup that
        ``router.get_provider`` cannot resolve, so the v3 Load button 400s.

        The admin list route composes ``router.app_config.models`` instead --
        one snapshot, so everything listed is loadable. See admin_api.
        """
        configs = []
        data = self._read_toml()
        for model_data in data.get("models", []):
            try:
                configs.append(ModelConfig(**model_data))
            except Exception as e:
                logger.warning(
                    f"Skipping invalid model config '{model_data.get('id', '?')}': {e}"
                )
        return configs

    def get_config(self, model_id: str) -> ModelConfig | None:
        """Get a WRITTEN-DOWN model's config by ID (constructs only the one
        entry -- constructing all N triggers each model's derive-at-load
        detection).

        models.toml only, for the same two reasons as list_configs: a
        per-request filesystem walk on the event loop, and answering for
        models the router's snapshot cannot load. Admin routes fall back to
        ``router.app_config`` for the discovered case.
        """
        for model_data in self._read_toml().get("models", []):
            if model_data.get("id") == model_id:
                try:
                    return ModelConfig(**model_data)
                except Exception as e:
                    logger.warning(f"Invalid model config '{model_id}': {e}")
                    return None
        return None

    def add_config(self, model_data: dict) -> ModelConfig:
        """Add a new model config. Validates and writes atomically."""
        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])

            # Check for duplicate ID
            model_id = model_data.get("id", "")
            if any(m.get("id") == model_id for m in models):
                raise ValueError(f"Model '{model_id}' already exists")

            # Validate ID format
            if not MODEL_ID_PATTERN.match(model_id):
                raise ValueError(
                    f"Invalid model ID '{model_id}'. "
                    "Must start with alphanumeric and contain only alphanumeric, -, _, ., /"
                )

            # Validate path
            config = model_data.get("config", {})
            model_path = config.get("model_path", "")
            if model_path:
                path_result = self.validate_path(model_path)
                if not path_result.valid:
                    raise ValueError(f"Invalid model path: {path_result.error}")

            # Validate the complete model config
            try:
                validated = ModelConfig(**model_data)
            except Exception as e:
                raise ValueError(f"Invalid model config: {e}") from e

            # Add to config
            models.append(model_data)
            data["models"] = models
            self._write_toml(data)

            return validated

    def _update_model_file(self, data: dict, model_id: str,
                           updates: dict) -> tuple[ModelConfig, list[str]]:
        """Apply an edit to a discovered model by writing its own
        ``model.heylook.toml`` (plan_registry_sidecars Phase 3).

        This replaced materialization, which copied the model's whole derived
        config into a models.toml entry: an entry replaces the derived config
        wholesale, so one edit froze every derived value for good and opted
        the model out of every later improvement to derivation. The file
        holds only what was set; everything else keeps being derived.

        ``{"config": {key: value}}`` sets a field; ``value = None`` drops the
        setting so the derived value comes back (and takes the field off
        ``unset``). An edit that leaves the file empty deletes it: reverting
        is deleting a file. Validated as the whole effective config before
        anything is written, the property the retired import writer held.
        """
        from heylook_llm.model_importer import SIDECAR_FILENAME
        from heylook_llm.model_registry import merge_discovered, refuse_if_readonly, scan

        extra = set(updates) - {"config"}
        if extra:
            raise ValueError(
                f"{', '.join(sorted(extra))}: not stored per model; a model's own file "
                f"holds config fields only")
        existing = len(data.get("models") or [])
        merged = merge_discovered(data, scan(data).entries)
        entry = next((e for e in (merged.get("models") or [])[existing:]
                      if str(e.get("id")) == str(model_id)), None)
        if entry is None:
            raise ValueError(f"Model '{model_id}' not found")

        model_path = Path(str(entry["config"]["model_path"]))
        folder = model_path if model_path.is_dir() else model_path.parent
        file = folder / SIDECAR_FILENAME
        try:
            old_text = file.read_text(encoding="utf-8") if file.is_file() else ""
            stored = tomllib.loads(old_text) if old_text else {}
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ValueError(f"{file} does not read ({e}); fix it by hand first") from e
        unset = [k for k in stored.pop("unset", []) if isinstance(k, str)]

        changes = updates.get("config") or {}
        for key, value in changes.items():
            if key in ("model_path", "id"):
                raise ValueError(f"`{key}` is not settable: the folder is the model")
            stored.pop(key, None)
            if key in unset:
                unset.remove(key)
            if value is not None:
                stored[key] = value

        derived = dict(entry.get("derived") or entry["config"])
        effective = {k: v for k, v in derived.items() if k not in unset}
        for key, value in stored.items():
            if key.endswith("_path") and isinstance(value, str) and not Path(value).expanduser().is_absolute():
                value = str(folder / value)
            effective[key] = value
        try:
            validated = ModelConfig(**{"id": model_id, "provider": entry["provider"],
                                       "config": dict(effective)})
        except Exception as e:
            raise ValueError(f"Updated config is invalid: {e}") from e

        reload_fields = reload_required_for(entry.get("provider"))
        before = entry["config"]
        changed = [k for k in changes if k in reload_fields and before.get(k) != effective.get(k)]

        refuse_if_readonly(file.name)
        if unset:
            stored["unset"] = unset
        if not stored:
            file.unlink(missing_ok=True)
            logger.info("[registry] %s: removed %s, back to derived", model_id, file)
        else:
            text = tomli_w.dumps(stored)
            if old_text:
                text = merge_comments(old_text, text)
            tmp = file.with_name(file.name + f".tmp.{os.getpid()}")
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, file)
            finally:
                tmp.unlink(missing_ok=True)
            logger.info("[registry] %s: wrote %s", model_id, file)
        return validated, changed

    def update_config(
        self, model_id: str, updates: dict
    ) -> tuple[ModelConfig, list[str]]:
        """Update model config fields. Returns (updated_config, reload_required_fields).

        The reload_required_fields list tells the caller which changed fields
        need a model reload to take effect.
        """
        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])

            idx = None
            for i, m in enumerate(models):
                if m.get("id") == model_id:
                    idx = i
                    break

            if idx is None:
                return self._update_model_file(data, model_id, updates)

            # Work on a deep copy so the original is untouched if validation fails
            model = copy.deepcopy(models[idx])
            changed_reload_fields = []

            # Validate path BEFORE applying updates
            if "model_path" in updates.get("config", {}):
                path_result = self.validate_path(updates["config"]["model_path"])
                if not path_result.valid:
                    raise ValueError(f"Invalid model path: {path_result.error}")

            # Apply top-level updates
            for key in ("description", "tags", "capabilities"):
                if key in updates:
                    model[key] = updates[key]

            # Apply provider config updates
            if "config" in updates and isinstance(updates["config"], dict):
                if "config" not in model:
                    model["config"] = {}
                # Provider-aware: a gguf entry's reload-required set is not the
                # MLX one (this is what silently missed ctx_size before).
                reload_fields = reload_required_for(model.get("provider"))
                for key, value in updates["config"].items():
                    had_key = key in model["config"]
                    old_value = model["config"].get(key)
                    if value is None:
                        # An explicit null means "unset this -- go back to the
                        # default". TOML cannot express null (tomli_w raises
                        # TypeError on a None VALUE), and for these fields
                        # "absent" IS how the default is spelled, so remove the
                        # key rather than storing None. Without this, any
                        # reset-to-default control 500s on the TOML write after
                        # passing validation, because Optional[...] = None is
                        # perfectly valid to pydantic and only fails at the
                        # serializer.
                        model["config"].pop(key, None)
                        changed = had_key
                    else:
                        model["config"][key] = value
                        changed = old_value != value
                    if changed and key in reload_fields:
                        changed_reload_fields.append(key)

            # Validate the updated model -- only commit if valid
            try:
                validated = ModelConfig(**model)
            except Exception as e:
                raise ValueError(f"Updated config is invalid: {e}") from e

            models[idx] = model
            data["models"] = models
            self._write_toml(data)

            return validated, changed_reload_fields

    def remove_config(self, model_id: str) -> bool:
        """Remove a model's entry from config. Files stay on disk.

        A discovered model has no entry to remove: the next scan serves it
        straight back. To stop serving it, take it out of the scan folder
        (owner decision 2026-09-23: presence in a scan folder IS enabled).

        """
        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])
            original_len = len(models)
            models = [m for m in models if m.get("id") != model_id]

            if len(models) == original_len:
                return False

            # Update default_model if we removed it
            if data.get("default_model") == model_id:
                data["default_model"] = (
                    "none" if not models else models[0]["id"]
                )

            data["models"] = models
            self._write_toml(data)
            return True

    def validate_config(self, config_data: dict) -> ValidationResult:
        """Validate a model config without saving."""
        errors = []
        warnings = []

        # Check required fields
        if not config_data.get("id"):
            errors.append("Model ID is required")
        elif not MODEL_ID_PATTERN.match(config_data["id"]):
            errors.append("Invalid model ID format")

        if not config_data.get("provider"):
            errors.append("Provider is required")
        elif config_data["provider"] not in ("mlx", "gguf"):
            errors.append(f"Unknown provider: {config_data['provider']}")

        config = config_data.get("config", {})
        if not config.get("model_path"):
            errors.append("model_path is required in config")
        else:
            path_result = self.validate_path(config["model_path"])
            if not path_result.valid:
                errors.append(f"Invalid model_path: {path_result.error}")

        # Check for duplicate ID
        existing = self.get_config(config_data.get("id", ""))
        if existing:
            warnings.append(f"Model ID '{config_data['id']}' already exists")

        # Try to construct the ModelConfig
        if not errors:
            try:
                ModelConfig(**config_data)
            except Exception as e:
                errors.append(f"Config validation failed: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_path(self, path: str) -> PathValidation:
        """Validate that a model path resolves and exists.

        No allowed-roots check: models live wherever the operator keeps them
        (watch folders, the HF cache, anywhere an explicit entry points)."""
        try:
            p = Path(path).expanduser().resolve()
        except Exception as e:
            return PathValidation(valid=False, error=f"Invalid path: {e}")

        if not p.exists():
            return PathValidation(
                valid=False, resolved_path=str(p), error="Path does not exist"
            )

        return PathValidation(valid=True, resolved_path=str(p))

    # --- Helpers ---

    def get_field_reload_info(self, provider: Optional[str] = None) -> dict[str, str]:
        """Return field -> reload requirement mapping.

        Pass ``provider`` to get that provider's answer; without it the union
        is used, which over-reports rather than under-reports.
        """
        info = {}
        # Hand-written list FIRST so the derived set overwrites it, never the
        # other way round. RUNTIME_CHANGEABLE_FIELDS is still maintained by
        # hand; if it ever names a field that config.py declares
        # requires_reload, the two disagree and the derived declaration is the
        # one to trust -- reporting a spawn-time flag as a live knob is how the
        # UI ends up telling someone a change took effect when the process
        # kept the old value. (The sets are disjoint today, and a test keeps
        # them that way; this ordering makes the collision harmless meanwhile.)
        for f in RUNTIME_CHANGEABLE_FIELDS:
            info[f] = "runtime"
        for f in reload_required_for(provider):
            info[f] = "reload_required"
        return info
