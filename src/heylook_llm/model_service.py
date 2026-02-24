# src/heylook_llm/model_service.py
"""
Service layer for model discovery, validation, and configuration management.

Provides CRUD operations on models.toml, filesystem scanning for importable models,
smart defaults, profile definitions, and profile application. Thread-safe for
concurrent API access.

This module is the single source of truth for:
- Model profiles (load_profiles, PROFILES, ModelProfile) -- loaded from profiles/*.toml
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
# Model profiles
# =============================================================================

@dataclass
class ModelProfile:
    """Profile with smart defaults for different model types.

    Automatically filters provider-specific parameters.
    """
    name: str
    description: str
    defaults: dict[str, Any] = field(default_factory=dict)

    # Provider-specific parameter sets
    MLX_ONLY_PARAMS = {
        'cache_type', 'kv_bits', 'kv_group_size', 'quantized_kv_start',
        'max_kv_size', 'draft_model_path', 'num_draft_tokens'
    }

    def apply(self, config: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
        """Apply profile defaults to config, overriding existing values.

        Profile values take precedence over smart defaults. MLX-only parameters
        are filtered out for non-MLX providers.
        """
        result = config.copy()
        provider = model_info.get('provider', 'mlx')

        for key, value in self.defaults.items():
            if provider != 'mlx' and key in self.MLX_ONLY_PARAMS:
                continue

            if callable(value):
                result[key] = value(model_info)
            else:
                result[key] = value

        return result


def _load_profiles_from_toml(profiles_dir: Path | None = None) -> dict[str, ModelProfile]:
    """Load profile definitions from TOML files in the profiles/ directory.

    Each .toml file must have [meta] (name, description) and [defaults] tables.
    Loaded once and cached module-level.
    """
    if profiles_dir is None:
        # Default: profiles/ relative to project root (two levels up from this file)
        profiles_dir = Path(__file__).parent.parent.parent / "profiles"

    profiles: dict[str, ModelProfile] = {}

    if not profiles_dir.is_dir():
        logger.warning(f"Profiles directory not found: {profiles_dir}")
        return profiles

    for toml_file in sorted(profiles_dir.glob("*.toml")):
        try:
            with open(toml_file, 'rb') as f:
                data = tomllib.load(f)
            meta = data.get("meta", {})
            defaults = data.get("defaults", {})

            name = meta.get("name", toml_file.stem)
            description = meta.get("description", "")

            # Coerce numeric types: TOML reads all numbers as int or float,
            # but booleans come through correctly
            profiles[name] = ModelProfile(
                name=name,
                description=description,
                defaults=defaults,
            )
        except Exception as e:
            logger.error(f"Failed to load profile {toml_file.name}: {e}")

    return profiles


# Module-level cache: loaded once on first access
_profiles_cache: dict[str, ModelProfile] | None = None


def load_profiles(profiles_dir: Path | None = None) -> dict[str, ModelProfile]:
    """Get all available profiles. Loads from TOML on first call, then cached."""
    global _profiles_cache
    if _profiles_cache is None or profiles_dir is not None:
        loaded = _load_profiles_from_toml(profiles_dir)
        if profiles_dir is None:
            _profiles_cache = loaded
        return loaded
    return _profiles_cache


def get_available_profiles() -> list[str]:
    """Return sorted list of available profile names (for argparse choices)."""
    return sorted(load_profiles().keys())


PROFILES = load_profiles()


# =============================================================================
# Smart defaults
# =============================================================================

def get_smart_defaults(model_info: dict[str, Any]) -> dict[str, Any]:
    """Generate smart defaults based on model characteristics.

    NOTE: These defaults are optimized for text GENERATION use cases.
    For text ENCODING (hidden states extraction for Z-Image, etc.):
    - max_tokens does NOT limit hidden states (request-level max_length does)
    - Temperature, top_k, etc. are ignored (no generation happens)
    - Only quantization level affects hidden state precision
    - Use --profile embedding for encoder-focused defaults
    """
    defaults: dict[str, Any] = {}

    size_gb = model_info.get('size_gb', 0)
    is_vision = model_info.get('is_vision', False)
    provider = model_info.get('provider', 'mlx')

    # Temperature based on model type
    if 'instruct' in model_info.get('name', '').lower():
        defaults['temperature'] = 0.7
    elif 'chat' in model_info.get('name', '').lower():
        defaults['temperature'] = 0.8
    else:
        defaults['temperature'] = 0.9

    # Cache strategy based on size
    if provider == 'mlx':
        if size_gb > 30:
            defaults['cache_type'] = 'quantized'
            defaults['kv_bits'] = 8
            defaults['kv_group_size'] = 32
            defaults['quantized_kv_start'] = 512
            defaults['max_kv_size'] = 2048
        elif size_gb > 13:
            defaults['cache_type'] = 'quantized'
            defaults['kv_bits'] = 8
            defaults['kv_group_size'] = 64
            defaults['quantized_kv_start'] = 1024
        else:
            defaults['cache_type'] = 'standard'

    # Sampling strategy
    if size_gb > 30:
        defaults['top_k'] = 20
        defaults['min_p'] = 0.05
        defaults['max_tokens'] = 256
    elif size_gb < 3:
        defaults['top_k'] = 50
        defaults['top_p'] = 0.95
        defaults['max_tokens'] = 1024
    else:
        defaults['top_k'] = 40
        defaults['min_p'] = 0.05
        defaults['max_tokens'] = 512

    # Vision model specifics
    if is_vision:
        defaults['max_tokens'] = min(defaults.get('max_tokens', 512), 512)

    # Repetition penalty
    defaults['repetition_penalty'] = 1.05
    defaults['repetition_context_size'] = 20

    # Speculative decoding default (used when draft_model_path is set)
    if provider == 'mlx':
        defaults['num_draft_tokens'] = 3

    return defaults

# Fields that require a model reload vs runtime-changeable
RELOAD_REQUIRED_FIELDS = frozenset({
    "model_path", "vision", "cache_type", "kv_bits", "kv_group_size",
    "quantized_kv_start", "max_kv_size", "draft_model_path", "num_draft_tokens",
    "default_hidden_layer", "default_max_length", "supports_thinking",
    "fp32", "use_local_attention", "local_attention_context",
    "chunk_duration", "overlap_duration",
})

RUNTIME_CHANGEABLE_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "max_tokens",
    "repetition_penalty", "presence_penalty", "enable_thinking",
    "repetition_context_size",
})

# Valid model ID pattern: alphanumeric, hyphens, underscores, dots, slashes
MODEL_ID_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._\-/]*$')


@dataclass
class ScannedModel:
    """A model discovered during filesystem scan."""
    id: str
    path: str
    provider: str  # "mlx"
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
        if self.config_path.suffix != '.toml':
            self.config_path = self.config_path.with_suffix('.toml')
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
        with open(self.config_path, 'rb') as f:
            return tomllib.load(f)

    def _write_toml(self, data: dict) -> None:
        """Atomic write: write to .tmp, validate, rename. Creates backup."""
        tmp_path = self.config_path.with_suffix('.toml.tmp')
        backup_path = self.config_path.with_suffix('.toml.bak')

        # Write to temp file
        toml_bytes = tomli_w.dumps(data).encode('utf-8')
        tmp_path.write_bytes(toml_bytes)

        # Validate the written file can be parsed back
        try:
            with open(tmp_path, 'rb') as f:
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

    def scan_directory(self, path: str) -> list[ScannedModel]:
        """Scan a directory for importable models."""
        from heylook_llm.model_importer import ModelImporter
        importer = ModelImporter()
        raw_models = importer.scan_directory(path)
        configured_ids = {m.id for m in self.list_configs()}
        return [self._raw_to_scanned(m, configured_ids) for m in raw_models]

    def scan_hf_cache(self) -> list[ScannedModel]:
        """Scan HuggingFace cache directories for models."""
        from heylook_llm.model_importer import ModelImporter
        importer = ModelImporter()
        raw_models = importer.scan_hf_cache()
        configured_ids = {m.id for m in self.list_configs()}
        return [self._raw_to_scanned(m, configured_ids) for m in raw_models]

    def scan_paths(self, paths: list[str] | None = None, scan_hf: bool = True) -> list[ScannedModel]:
        """Scan multiple paths and optionally HF cache."""
        results: list[ScannedModel] = []
        seen_ids: set[str] = set()

        if paths:
            for p in paths:
                for model in self.scan_directory(p):
                    if model.id not in seen_ids:
                        results.append(model)
                        seen_ids.add(model.id)

        if scan_hf:
            for model in self.scan_hf_cache():
                if model.id not in seen_ids:
                    results.append(model)
                    seen_ids.add(model.id)

        return results

    def _raw_to_scanned(self, raw: dict, configured_ids: set[str]) -> ScannedModel:
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

        return ScannedModel(
            id=raw.get("id", ""),
            path=model_path,
            provider=raw.get("provider", "mlx"),
            size_gb=round(size_gb, 2),
            vision=config.get("vision", False),
            quantization=quantization,
            already_configured=raw.get("id", "") in configured_ids,
            tags=raw.get("tags", []),
            description=raw.get("description", ""),
        )

    # --- Smart Defaults ---

    def get_smart_defaults(self, model_info: dict) -> dict:
        """Generate smart defaults based on model characteristics."""
        return get_smart_defaults(model_info)

    def get_profiles(self) -> dict[str, dict]:
        """Get available preset profiles with descriptions."""
        return {
            name: {
                "name": profile.name,
                "description": profile.description,
            }
            for name, profile in PROFILES.items()
        }

    def apply_profile(self, config: dict, profile_name: str, model_info: dict) -> dict:
        """Apply a named profile to config."""
        profile = PROFILES.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")
        return profile.apply(config, model_info)

    # --- Config CRUD ---

    def list_configs(self) -> list[ModelConfig]:
        """List all model configs (including disabled)."""
        data = self._read_toml()
        configs = []
        for model_data in data.get("models", []):
            try:
                configs.append(ModelConfig(**model_data))
            except Exception as e:
                logger.warning(f"Skipping invalid model config '{model_data.get('id', '?')}': {e}")
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

    def update_config(self, model_id: str, updates: dict) -> tuple[ModelConfig, list[str]]:
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
                data["default_model"] = enabled_models[0]["id"] if enabled_models else ("none" if not models else models[0]["id"])

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

    def bulk_apply_profile(self, model_ids: list[str], profile_name: str) -> list[ModelConfig]:
        """Apply a profile to multiple models at once."""
        profile = PROFILES.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        with self._lock:
            data = self._read_toml()
            models = data.get("models", [])
            updated = []

            for model in models:
                if model.get("id") in model_ids:
                    config = model.get("config", {})
                    model_info = {
                        "name": model.get("id", ""),
                        "provider": model.get("provider", "mlx"),
                        "size_gb": 0,
                    }
                    model["config"] = profile.apply(config, model_info)
                    updated.append(ModelConfig(**model))

            data["models"] = models
            self._write_toml(data)
            return updated

    # --- Import ---

    def import_models(
        self,
        models_to_import: list[dict],
        profile_name: str | None = "moderate",
    ) -> list[ModelConfig]:
        """Import scanned models into config with optional profile."""
        with self._lock:
            data = self._read_toml()
            existing_models = data.get("models", [])
            existing_ids = {m.get("id") for m in existing_models}
            imported = []

            for model_data in models_to_import:
                model_id = model_data.get("id", "")
                if model_id in existing_ids:
                    logger.warning(f"Skipping duplicate model: {model_id}")
                    continue

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

                # Start with base config
                entry_config = {
                    "model_path": model_path,
                    "vision": vision,
                }

                # Apply smart defaults
                smart = get_smart_defaults(model_info)
                entry_config.update(smart)

                # Apply profile if specified
                if profile_name:
                    profile = PROFILES.get(profile_name)
                    if profile:
                        entry_config = profile.apply(entry_config, model_info)

                # Apply any overrides from the import request
                overrides = model_data.get("overrides", {})
                entry_config.update(overrides)

                entry = {
                    "id": model_id,
                    "provider": provider,
                    "description": model_data.get("description", f"Imported {provider.upper()} model"),
                    "tags": model_data.get("tags", []),
                    "enabled": model_data.get("enabled", True),
                    "config": entry_config,
                }

                try:
                    validated = ModelConfig(**entry)
                    existing_models.append(entry)
                    existing_ids.add(model_id)
                    imported.append(validated)
                except Exception as e:
                    logger.error(f"Failed to import model '{model_id}': {e}")

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
        elif config_data["provider"] not in ("mlx", "mlx_stt"):
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
            return PathValidation(valid=False, resolved_path=str(p), error="Path does not exist")

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
