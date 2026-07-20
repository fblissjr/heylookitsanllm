# src/heylook_llm/model_service.py
"""
Service layer for model discovery, validation, and configuration management.

Provides CRUD operations on models.toml, filesystem scanning for importable models,
smart defaults, and sampler-preset stamping (default_preset). Thread-safe for
concurrent API access.

This module is the single source of truth for:
- Sampler presets (load_sampler_presets, SAMPLER_PRESETS, SamplerPreset) -- views over the PresetRegistry
- Smart defaults (get_smart_defaults)
- HuggingFace cache paths (get_hf_cache_paths)
- Config CRUD, scanning, import, validation
"""

import copy
import logging
import os
import platform
import re
import shutil
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tomli_w  # type: ignore[import-untyped]

from heylook_llm.config import AppConfig, ModelConfig
from heylook_llm.providers.common.template_info import detect_chat_template_source

logger = logging.getLogger(__name__)


# =============================================================================
# HuggingFace cache paths (platform-specific)
# =============================================================================


def get_hf_cache_paths() -> list[str]:
    """Get platform-specific HuggingFace cache paths."""
    if platform.system() == "Windows":
        return [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/AppData/Local/huggingface/hub"),
        ]
    elif platform.system() == "Darwin":
        return [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/Library/Caches/huggingface/hub"),
        ]
    else:
        return [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.huggingface/hub"),
        ]


# =============================================================================
# Sampler presets (terminology note: the import/admin paths called these
# "profiles" until 2026-07-20; same registry as ChatRequest.preset)
# =============================================================================


@dataclass
class SamplerPreset:
    """View over a named preset from the runtime ``PresetRegistry``.

    Historically this dataclass carried a sampler-field ``defaults`` dict that
    ``apply()`` baked into ``models.toml`` at import time. After C4, sampler
    fields live in ``src/heylook_llm/data/presets/*.toml`` and are resolved
    at REQUEST time via the registry cascade. ``apply()`` now just records
    the preset name on the model's config as ``default_preset`` -- callers
    keep the same method signature during the transition; internal
    semantics differ.
    """

    name: str
    description: str
    defaults: dict[str, Any] = field(default_factory=dict)

    def apply(
        self, config: dict[str, Any], model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Record the preset name on the model's config.

        The request-time cascade (see ``MLXProvider._apply_model_defaults``)
        picks up ``default_preset`` and applies the preset fields when no
        per-request preset is specified. No baking.
        """
        provider = model_info.get("provider", "mlx")
        result = dict(config)
        if provider == "mlx":
            result["default_preset"] = self.name
        return result


def load_sampler_presets(presets_dir: Path | None = None) -> dict[str, SamplerPreset]:
    """Return a preset-dataclass view over the runtime preset registry.

    The ``presets_dir`` override exists for tests that want to load from a
    custom directory; production paths use the bundled presets via the
    process-wide ``PresetRegistry``.
    """
    from heylook_llm.presets import PresetRegistry, get_preset_registry

    if presets_dir is not None:
        registry = PresetRegistry.from_directory(presets_dir)
    else:
        registry = get_preset_registry()

    return {
        name: SamplerPreset(
            name=name,
            description=registry.describe(name),
            defaults=registry.get(name),
        )
        for name in registry.list_names()
    }


def available_sampler_presets() -> list[str]:
    """Return sorted list of available preset names (for argparse choices)."""
    from heylook_llm.presets import get_preset_registry

    return get_preset_registry().list_names()


def _presets_view() -> dict[str, SamplerPreset]:
    """Accessor used by functions that previously read the module dict.
    Returns a fresh view each call -- cheap (registry is memoized)."""
    return load_sampler_presets()


# Module-level snapshot; ``_presets_view()`` gives a fresh snapshot backed by
# the registry when needed.
SAMPLER_PRESETS = _presets_view()


# =============================================================================
# Smart defaults
# =============================================================================


def _system_ram_gb() -> float:
    """Total unified memory in GB. Conservative fallback if psutil is absent."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 64.0


def get_smart_defaults(model_info: dict[str, Any]) -> dict[str, Any]:
    """Generate LOAD-TIME smart defaults based on model characteristics.

    Post-C4, sampler fields (temperature, top_k, top_p, min_p, max_tokens,
    repetition_penalty) are NOT returned here -- those are the request-time
    concern of the preset cascade. This function emits only the fields that
    determine how the model loads and what its cache looks like: cache_type,
    KV quantization knobs, draft-token count.

    Users pick a sampler preset via ``--preset NAME`` on import (records
    ``default_preset``) or per-request via ``ChatRequest.preset``.
    """
    provider = model_info.get("provider", "mlx")
    if provider == "mlx_embedding":
        return {"max_length": 2048}

    defaults: dict[str, Any] = {}
    size_gb = model_info.get("size_gb", 0)

    if provider == "mlx":
        # KV quantization is a memory/quality trade-off, so it must be
        # RAM-relative, not an absolute weight threshold: a 40GB model is
        # "large" on a 64GB MacBook and trivial on a 192GB Studio. Quantize
        # only when the weights alone claim over ~35% of unified memory
        # (leaving the rest for KV, vision towers, and the OS).
        #
        # max_kv_size is deliberately NEVER defaulted: it creates a
        # RotatingKVCache that silently drops context beyond the cap --
        # truncation is an explicit user choice, not an import default.
        if size_gb > _system_ram_gb() * 0.35:
            defaults["cache_type"] = "quantized"
            defaults["kv_bits"] = 8
            defaults["kv_group_size"] = 64
        else:
            defaults["cache_type"] = "standard"

        # num_draft_tokens is deliberately NOT emitted: it only matters when
        # a draft_model_path is configured (speculative decoding), and import
        # never configures one -- stamping it on every model is dead config.

    return defaults


# Fields that require a model reload vs runtime-changeable
RELOAD_REQUIRED_FIELDS = frozenset(
    {
        "model_path",
        "vision",
        "cache_type",
        "kv_bits",
        "kv_group_size",
        "max_kv_size",
        "draft_model_path",
        "num_draft_tokens",
        "default_hidden_layer",
        "default_max_length",
        "supports_thinking",
    }
)

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
class ScannedModel:
    """A model discovered during filesystem scan."""

    id: str
    path: str
    provider: str  # "mlx", "mlx_embedding"
    size_gb: float
    vision: bool
    quantization: Optional[str] = None
    already_configured: bool = False
    tags: list[str] = field(default_factory=list)
    description: str = ""


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
        self._allowed_roots = self._compute_allowed_roots()

    def _compute_allowed_roots(self) -> list[Path]:
        """Compute allowed root paths for model files."""
        roots = []
        # HF cache paths
        for p in get_hf_cache_paths():
            expanded = Path(p).expanduser()
            if expanded.exists():
                roots.append(expanded.resolve())
        # Project modelzoo
        project_root = self.config_path.parent
        modelzoo = project_root / "modelzoo"
        if modelzoo.exists():
            roots.append(modelzoo.resolve())
        # Paths already in config
        try:
            data = self._read_toml()
            for model in data.get("models", []):
                config = model.get("config", {})
                model_path = config.get("model_path", "")
                if model_path:
                    p = Path(model_path)
                    if p.exists():
                        # Add parent directory as allowed root
                        parent = p.parent.resolve()
                        if parent not in roots:
                            roots.append(parent)
        except Exception:
            pass
        return roots

    # --- TOML I/O ---

    def _read_toml(self) -> dict:
        """Read and parse the models.toml file."""
        if not self.config_path.exists():
            return {"models": [], "default_model": "none", "max_loaded_models": 1}
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def _write_toml(self, data: dict) -> None:
        """Atomic write: write to .tmp, validate, rename. Creates backup."""
        tmp_path = self.config_path.with_suffix(".toml.tmp")
        backup_path = self.config_path.with_suffix(".toml.bak")

        # Write to temp file
        toml_bytes = tomli_w.dumps(data).encode("utf-8")
        tmp_path.write_bytes(toml_bytes)

        # Validate the written file can be parsed back
        try:
            with open(tmp_path, "rb") as f:
                parsed = tomllib.load(f)
            # Validate it produces a valid AppConfig
            AppConfig(**parsed)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Generated TOML failed validation: {e}") from e

        # Backup existing config
        if self.config_path.exists():
            shutil.copy2(self.config_path, backup_path)

        # Atomic rename
        tmp_path.rename(self.config_path)
        logger.info(f"Config written to {self.config_path}")

    # --- Discovery ---

    def scan_directory(
        self, path: str, identity: tuple[set[str], set[str]] | None = None
    ) -> list[ScannedModel]:
        """Scan a directory for importable models.

        ``identity`` lets a caller that's scanning multiple sources (see
        ``scan_paths``) pass in a precomputed ``_configured_identity()``
        result so models.toml isn't re-read and re-validated per source.
        """
        from heylook_llm.model_importer import ModelImporter

        importer = ModelImporter()
        raw_models = importer.scan_directory(path)
        configured_ids, configured_paths = identity or self._configured_identity()
        return [self._raw_to_scanned(m, configured_ids, configured_paths) for m in raw_models]

    def scan_hf_cache(
        self, identity: tuple[set[str], set[str]] | None = None
    ) -> list[ScannedModel]:
        """Scan HuggingFace cache directories for models.

        ``identity`` lets a caller that's scanning multiple sources (see
        ``scan_paths``) pass in a precomputed ``_configured_identity()``
        result so models.toml isn't re-read and re-validated per source.
        """
        from heylook_llm.model_importer import ModelImporter

        importer = ModelImporter()
        raw_models = importer.scan_hf_cache()
        configured_ids, configured_paths = identity or self._configured_identity()
        return [self._raw_to_scanned(m, configured_ids, configured_paths) for m in raw_models]

    def _configured_identity(self) -> tuple[set[str], set[str]]:
        """Configured ids AND resolved weight paths.

        ``already_configured`` must match on either: a rescan can derive a
        different id for weights that are already configured (id matching
        alone invited duplicate entries pointing at the same weights).
        Paths are resolved so symlinked spellings compare equal.
        """
        ids: set[str] = set()
        paths: set[str] = set()
        for m in self.list_configs():
            ids.add(m.id)
            model_path = getattr(m.config, "model_path", "")
            if model_path:
                paths.add(str(Path(model_path).expanduser().resolve()))
        return ids, paths

    def scan_paths(
        self, paths: list[str] | None = None, scan_hf: bool = True
    ) -> list[ScannedModel]:
        """Scan multiple paths and optionally HF cache."""
        results: list[ScannedModel] = []
        seen_ids: set[str] = set()

        # Computed once and threaded through every scan_directory/scan_hf_cache
        # call below -- each call independently recomputes this from a full
        # models.toml read + per-model Pydantic validation, so without sharing
        # it a K-source scan re-reads and re-resolves the config K times.
        identity = self._configured_identity()

        if paths:
            for p in paths:
                for model in self.scan_directory(p, identity=identity):
                    if model.id not in seen_ids:
                        results.append(model)
                        seen_ids.add(model.id)

        if scan_hf:
            for model in self.scan_hf_cache(identity=identity):
                if model.id not in seen_ids:
                    results.append(model)
                    seen_ids.add(model.id)

        return results

    def _raw_to_scanned(
        self,
        raw: dict,
        configured_ids: set[str],
        configured_paths: set[str] | None = None,
    ) -> ScannedModel:
        """Convert raw importer dict to ScannedModel."""
        config = raw.get("config", {})
        model_path = config.get("model_path", "")

        # Detect quantization from name
        name_lower = raw.get("id", "").lower()
        quantization = None
        if "4bit" in name_lower or "q4" in name_lower:
            quantization = "4bit"
        elif "8bit" in name_lower or "q8" in name_lower:
            quantization = "8bit"
        elif "mxfp4" in name_lower:
            quantization = "mxfp4"

        # Estimate size
        size_gb = 0.0
        p = Path(model_path)
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*.safetensors"))
            size_gb = total / (1024**3)
        elif p.is_file():
            size_gb = p.stat().st_size / (1024**3)

        # Configured if the id matches OR the resolved weights path is
        # already configured under any id (see _configured_identity).
        already = raw.get("id", "") in configured_ids
        if not already and configured_paths and model_path:
            already = str(Path(model_path).expanduser().resolve()) in configured_paths

        return ScannedModel(
            id=raw.get("id", ""),
            path=model_path,
            provider=raw.get("provider", "mlx"),
            size_gb=round(size_gb, 2),
            vision=config.get("vision", False),
            quantization=quantization,
            already_configured=already,
            tags=raw.get("tags", []),
            description=raw.get("description", ""),
        )

    # --- Smart Defaults ---

    def get_smart_defaults(self, model_info: dict) -> dict:
        """Generate smart defaults based on model characteristics."""
        return get_smart_defaults(model_info)

    def get_sampler_presets(self) -> dict[str, dict]:
        """Get available sampler presets with descriptions."""
        return {
            name: {
                "name": preset.name,
                "description": preset.description,
            }
            for name, preset in SAMPLER_PRESETS.items()
        }

    def apply_sampler_preset(self, config: dict, preset_name: str, model_info: dict) -> dict:
        """Record a named sampler preset as the config's default_preset."""
        preset = SAMPLER_PRESETS.get(preset_name)
        if not preset:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(SAMPLER_PRESETS.keys())}"
            )
        return preset.apply(config, model_info)

    # --- Config CRUD ---

    def list_configs(self) -> list[ModelConfig]:
        """List all model configs (including disabled)."""
        data = self._read_toml()
        configs = []
        for model_data in data.get("models", []):
            try:
                configs.append(ModelConfig(**model_data))
            except Exception as e:
                logger.warning(
                    f"Skipping invalid model config '{model_data.get('id', '?')}': {e}"
                )
        return configs

    def get_config(self, model_id: str) -> ModelConfig | None:
        """Get a single model's config by ID."""
        for config in self.list_configs():
            if config.id == model_id:
                return config
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
                raise ValueError(f"Model '{model_id}' not found")

            # Work on a deep copy so the original is untouched if validation fails
            model = copy.deepcopy(models[idx])
            changed_reload_fields = []

            # Validate path BEFORE applying updates
            if "model_path" in updates.get("config", {}):
                path_result = self.validate_path(updates["config"]["model_path"])
                if not path_result.valid:
                    raise ValueError(f"Invalid model path: {path_result.error}")

            # Apply top-level updates
            for key in ("description", "tags", "enabled", "capabilities"):
                if key in updates:
                    model[key] = updates[key]

            # Apply provider config updates
            if "config" in updates and isinstance(updates["config"], dict):
                if "config" not in model:
                    model["config"] = {}
                for key, value in updates["config"].items():
                    old_value = model["config"].get(key)
                    model["config"][key] = value
                    if old_value != value and key in RELOAD_REQUIRED_FIELDS:
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
        """Remove a model from config. Files stay on disk."""
        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])
            original_len = len(models)
            models = [m for m in models if m.get("id") != model_id]

            if len(models) == original_len:
                return False

            # Update default_model if we removed it -- prefer enabled models
            if data.get("default_model") == model_id:
                enabled_models = [m for m in models if m.get("enabled", True)]
                data["default_model"] = (
                    enabled_models[0]["id"]
                    if enabled_models
                    else ("none" if not models else models[0]["id"])
                )

            data["models"] = models
            self._write_toml(data)
            return True

    def toggle_enabled(self, model_id: str) -> ModelConfig:
        """Toggle a model's enabled state."""
        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])

            for model in models:
                if model.get("id") == model_id:
                    model["enabled"] = not model.get("enabled", True)
                    data["models"] = models
                    self._write_toml(data)
                    return ModelConfig(**model)

            raise ValueError(f"Model '{model_id}' not found")

    def bulk_set_default_preset(
        self, model_ids: list[str], preset_name: str
    ) -> list[ModelConfig]:
        """Set ``default_preset`` on multiple models at once.

        Records the name on each target model's config so the request-time
        cascade picks it up. No sampler-field baking.
        """
        from heylook_llm.presets import get_preset_registry
        registry = get_preset_registry()
        if preset_name not in registry:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {registry.list_names()}"
            )

        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])
            updated = []

            for model in models:
                if model.get("id") in model_ids:
                    config = model.get("config", {}) or {}
                    if model.get("provider") == "mlx":
                        config["default_preset"] = preset_name
                        model["config"] = config
                    updated.append(ModelConfig(**model))

            data["models"] = models
            self._write_toml(data)
            return updated

    # --- Import ---

    def import_models(
        self,
        models_to_import: list[dict],
        default_preset: str | None = "balanced",
    ) -> list[ModelConfig]:
        """Import scanned models into config with an optional default sampler preset."""
        with self._lock:
            data = self._read_toml()
            existing_models = data.get("models", [])
            existing_ids = {m.get("id") for m in existing_models}
            imported = []

            for model_data in models_to_import:
                model_id = model_data.get("id", "")

                # Build model entry
                provider = model_data.get("provider", "mlx")
                config = model_data.get("config", {})
                model_path = config.get("model_path", model_data.get("path", ""))
                vision = config.get("vision", model_data.get("vision", False))

                # Build model info for smart defaults
                model_info = {
                    "name": model_id,
                    "provider": provider,
                    "is_vision": vision,
                    "size_gb": model_data.get("size_gb", 0),
                    "is_quantized": model_data.get("quantization") is not None,
                }

                if provider == "mlx_embedding":
                    # Embedding models: no vision, no generation params, no sampler presets
                    entry_config = {
                        "model_path": model_path,
                        "max_length": 2048,
                    }
                else:
                    # Start with base config
                    entry_config = {
                        "model_path": model_path,
                        "vision": vision,
                    }

                    # Apply smart defaults
                    smart = get_smart_defaults(model_info)
                    entry_config.update(smart)

                    # Stamp default_preset if specified
                    if default_preset:
                        preset = SAMPLER_PRESETS.get(default_preset)
                        if preset:
                            entry_config = preset.apply(entry_config, model_info)

                    # Same detection as the CLI import wizard (shared helper --
                    # the two paths drifted once): record the explicit template
                    # policy so models.toml reflects it instead of relying on
                    # HF's version-dependent auto-detection.
                    if model_path:
                        detected = detect_chat_template_source(model_path)
                        if detected:
                            entry_config["chat_template_source"] = detected

                # Apply any overrides from the import request
                overrides = model_data.get("overrides", {})
                entry_config.update(overrides)

                entry = {
                    "id": model_id,
                    "provider": provider,
                    "description": model_data.get(
                        "description", f"Imported {provider.upper()} model"
                    ),
                    "tags": model_data.get("tags", []),
                    "enabled": model_data.get("enabled", True),
                    "config": entry_config,
                }

                try:
                    validated = ModelConfig(**entry)
                except Exception as e:
                    logger.error(f"Failed to import model '{model_id}': {e}")
                    continue

                if model_id in existing_ids:
                    # Re-import = PUT semantics: replace the existing entry
                    # with the freshly built one (was skip-not-update, which
                    # made refreshing an entry from a rescan impossible
                    # without hand-editing the TOML).
                    idx = next(
                        i for i, m in enumerate(existing_models)
                        if m.get("id") == model_id
                    )
                    existing_models[idx] = entry
                    logger.info(f"Re-import: updated existing model '{model_id}'")
                else:
                    existing_models.append(entry)
                    existing_ids.add(model_id)
                imported.append(validated)

            if imported:
                data["models"] = existing_models
                # Set default model if none exists
                if data.get("default_model") in (None, "none", ""):
                    data["default_model"] = imported[0].id
                self._write_toml(data)

            return imported

    # --- Validation ---

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
        elif config_data["provider"] not in ("mlx", "mlx_embedding"):
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
        """Validate a model path against allowed roots."""
        try:
            p = Path(path).expanduser().resolve()
        except Exception as e:
            return PathValidation(valid=False, error=f"Invalid path: {e}")

        if not p.exists():
            return PathValidation(
                valid=False, resolved_path=str(p), error="Path does not exist"
            )

        # Check it's under an allowed root
        for root in self._allowed_roots:
            try:
                p.relative_to(root)
                return PathValidation(valid=True, resolved_path=str(p))
            except ValueError:
                continue

        # If no allowed roots matched, still allow but warn
        # (the user might have models in a custom location)
        return PathValidation(valid=True, resolved_path=str(p))

    # --- Helpers ---

    def get_field_reload_info(self) -> dict[str, str]:
        """Return field -> reload requirement mapping."""
        info = {}
        for f in RELOAD_REQUIRED_FIELDS:
            info[f] = "reload_required"
        for f in RUNTIME_CHANGEABLE_FIELDS:
            info[f] = "runtime"
        return info
