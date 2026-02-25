# src/heylook_llm/model_importer.py
"""
CLI wrapper for model scanning and import.

Provides the ModelImporter class for filesystem scanning and TOML generation,
and the import_models CLI handler. Profiles, smart defaults, and HF cache paths
are defined in model_service.py (single source of truth).
"""

import json
import logging
import os
import re
import tomli_w
from pathlib import Path
from typing import Any, Optional

from heylook_llm.model_service import (
    PROFILES,
    ModelProfile,
    get_available_profiles,
    get_hf_cache_paths,
    get_smart_defaults,
    load_profiles,
)

# Re-export for backwards compatibility (server.py imports import_models from here)
__all__ = [
    "ModelImporter",
    "ModelProfile",
    "PROFILES",
    "get_available_profiles",
    "get_hf_cache_paths",
    "get_smart_defaults",
    "import_models",
    "load_profiles",
]

HF_CACHE_PATHS = get_hf_cache_paths()


class ModelImporter:
    """Scan directories and generate models.toml entries."""

    def __init__(self, profile: Optional[str] = None, overrides: Optional[dict[str, Any]] = None):
        self.models: list[dict] = []
        self.existing_ids: set[str] = set()
        if profile:
            self.profile = PROFILES.get(profile)
            if not self.profile:
                raise ValueError(f"Unknown profile: {profile}. Available: {sorted(PROFILES.keys())}")
        else:
            self.profile = None
        self.overrides = overrides or {}

    def scan_directory(self, path: str) -> list[dict]:
        """Scan a directory recursively for models."""
        path_obj = Path(path).expanduser().resolve()
        logging.info(f"Scanning directory: {path_obj}")

        if not path_obj.exists():
            logging.error(f"Path does not exist: {path_obj}")
            return []

        models = []
        dirs_scanned = 0

        for root, dirs, files in os.walk(path_obj, followlinks=True):
            root_path = Path(root)
            dirs_scanned += 1

            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            if dirs_scanned % 10 == 0:
                logging.debug(f"Scanned {dirs_scanned} directories, found {len(models)} models so far")

            rel_path = root_path.relative_to(path_obj)
            if str(rel_path) != ".":
                logging.debug(f"Scanning: {rel_path}")

            if self._is_mlx_model(root_path):
                logging.info(f"Found MLX model in: {rel_path}")
                model = self._create_mlx_entry(root_path)
                if model:
                    models.append(model)
                    logging.info(f"Added MLX model: {model['id']}")

        logging.info(f"Scan complete: {dirs_scanned} directories scanned, {len(models)} models imported")
        return models

    def scan_hf_cache(self) -> list[dict]:
        """Scan HuggingFace cache directories for models."""
        models = []

        for cache_path in HF_CACHE_PATHS:
            path = Path(cache_path).expanduser()
            if path.exists():
                logging.info(f"Scanning HF cache: {path}")
                for model_dir in path.glob("models--*"):
                    if model_dir.is_dir():
                        snapshots = model_dir / "snapshots"
                        if snapshots.exists():
                            for snapshot in snapshots.iterdir():
                                if snapshot.is_dir():
                                    found_models = self._scan_hf_snapshot(snapshot)
                                    models.extend(found_models)
        return models

    def _scan_hf_snapshot(self, snapshot_path: Path) -> list[dict]:
        """Scan a HF cache snapshot directory."""
        models = []

        if self._is_mlx_model(snapshot_path):
            model = self._create_mlx_entry(snapshot_path)
            if model:
                parts = snapshot_path.parent.parent.name.split("--")
                if len(parts) >= 2:
                    model['id'] = f"{parts[1]}/{parts[2]}" if len(parts) > 2 else parts[1]
                    model['config']['model_path'] = str(snapshot_path)
                models.append(model)

        return models

    def _is_mlx_model(self, path: Path) -> bool:
        """Check if a directory contains an MLX model."""
        mlx_indicators = [
            "mlx_config.json", "model.safetensors.index.json",
            "weights.00.safetensors", "model.00.safetensors", "config.json"
        ]
        for indicator in mlx_indicators:
            if (path / indicator).exists():
                if indicator == "config.json":
                    if any(path.glob("*.safetensors")):
                        return True
                else:
                    return True
        return False

    def _is_vision_model(self, path: Path) -> bool:
        """Check if a model supports vision."""
        vision_files = ["mmproj", "vision_tower", "image_encoder", "visual_encoder"]
        for file in path.iterdir():
            if any(v in file.name.lower() for v in vision_files):
                return True

        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if any(key in config for key in ["vision_config", "is_vision", "image_size"]):
                        return True
            except Exception:
                pass
        return False

    def _get_model_size(self, path: Path) -> tuple[Optional[str], Optional[float]]:
        """Estimate model size from name or files."""
        path_str = str(path).lower()

        size_patterns = [
            (r'(\d+\.\d+)b', lambda x: (f"{x}B", float(x))),
            (r'(\d+)b', lambda x: (f"{x}B", float(x))),
            (r'(\d+)m', lambda x: (f"{int(x)/1000:.1f}B" if int(x) >= 1000 else f"{x}M", int(x)/1000)),
        ]

        for pattern, formatter in size_patterns:
            match = re.search(pattern, path_str)
            if match:
                return formatter(match.group(1))

        if path.is_dir():
            total_size = 0
            for file in path.rglob("*.safetensors"):
                total_size += file.stat().st_size

            if total_size > 0:
                size_gb = total_size / (1024**3)
                if size_gb >= 1:
                    return f"{size_gb:.1f}B", size_gb
                else:
                    return f"{int(size_gb * 1000)}M", size_gb

        return None, None

    def _create_mlx_entry(self, path: Path) -> Optional[dict]:
        """Create a models.toml entry for an MLX model."""
        model_id = path.name
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        is_quantized = any(q in path.name.lower() for q in ['4bit', '8bit', 'q4', 'q8'])
        is_vision = self._is_vision_model(path)
        size_str, size_gb = self._get_model_size(path)

        model_info = {
            'name': model_id, 'provider': 'mlx',
            'is_quantized': is_quantized, 'is_vision': is_vision,
            'size_gb': size_gb or 0,
        }

        tags = self._detect_tags(model_id, is_vision, is_quantized, size_gb)

        config: dict[str, Any] = {"model_path": str(path), "vision": is_vision}
        config.update(get_smart_defaults(model_info))

        if self.profile:
            config = self.profile.apply(config, model_info)
        config.update(self.overrides)

        if size_gb and size_gb < 1 and not is_vision:
            tags.append("draft")
            config["max_tokens"] = 128

        return {
            "id": model_id, "provider": "mlx",
            "description": f"Auto-imported MLX model{' with vision' if is_vision else ''}{f' ({size_str})' if size_str else ''}",
            "tags": tags, "enabled": True, "config": config,
        }

    def _detect_tags(self, model_id: str, is_vision: bool, is_quantized: bool, size_gb: Optional[float]) -> list[str]:
        """Detect tags from model characteristics."""
        tags = []
        if is_vision:
            tags.append("vision")
        if is_quantized:
            tags.append("quantized")
        if size_gb:
            if size_gb >= 30:
                tags.append("large")
            elif size_gb <= 3:
                tags.append("small")

        model_lower = model_id.lower()
        for family in ["llama", "qwen", "gemma", "mistral"]:
            if family in model_lower:
                tags.append(family)
                break

        if 'instruct' in model_lower or 'chat' in model_lower:
            tags.append("instruct")
        return tags

    def generate_toml(self, models: list[dict], output_file: Optional[str] = None) -> str:
        """Generate models.toml content from discovered models."""
        config = {
            "default_model": models[0]['id'] if models else "none",
            "max_loaded_models": 1,
            "models": list(models),
        }

        toml_lines = [
            "# Auto-generated models configuration",
            "# Edit with: heylookllm models config",
            "",
            f'default_model = "{config["default_model"]}"',
            f'max_loaded_models = {config["max_loaded_models"]}',
            "",
            "# --- MLX Models ---",
            "",
        ]

        for model in models:
            toml_lines.extend(self._model_to_toml_lines(model))
            toml_lines.append("")

        toml_content = "\n".join(toml_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(toml_content)
            logging.info(f"Wrote configuration to {output_file}")

        return toml_content

    def _model_to_toml_lines(self, model: dict) -> list[str]:
        """Convert a model dict to TOML table lines."""
        lines = ["[[models]]"]
        lines.append(f'id = "{model["id"]}"')
        lines.append(f'provider = "{model["provider"]}"')

        if 'description' in model:
            lines.append(f'description = "{model["description"]}"')
        if 'tags' in model:
            tags_str = ", ".join(f'"{tag}"' for tag in model['tags'])
            lines.append(f'tags = [{tags_str}]')

        lines.append(f'enabled = {str(model.get("enabled", True)).lower()}')
        lines.append("")

        if 'config' in model:
            lines.append("  [models.config]")
            config_toml = tomli_w.dumps({"config": model['config']})
            config_lines = config_toml.split('\n')[1:]
            for line in config_lines:
                if line.strip():
                    lines.append(f"  {line}")

        return lines


def import_models(args: Any) -> None:
    """CLI handler for model import."""
    overrides = {}
    if hasattr(args, 'override') and args.override:
        for override in args.override:
            key, value = override.split('=', 1)
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
            overrides[key] = value

    importer = ModelImporter(
        profile=getattr(args, 'profile', None),
        overrides=overrides,
    )

    models = []

    if args.folder:
        folder_models = importer.scan_directory(args.folder)
        models.extend(folder_models)
        logging.info(f"Found {len(folder_models)} models in {args.folder}")

    if args.hf_cache:
        cache_models = importer.scan_hf_cache()
        models.extend(cache_models)
        logging.info(f"Found {len(cache_models)} models in HF cache")

    if not models:
        logging.warning("No models found!")
        return

    # Interactive mode: let user customize sampler/KV cache settings per model
    if getattr(args, 'interactive', False):
        try:
            import questionary
            from heylook_llm.config_tui import ConfigEditor
        except ImportError:
            logging.error("Interactive mode requires 'questionary'. Install with: uv add questionary")
            return

        editor = ConfigEditor()
        print(f"\nDiscovered {len(models)} model(s):")
        for i, m in enumerate(models):
            print(f"  [{i}] {m['id']} ({m['provider']})")

        # Let user pick which models to customize
        model_choices = [
            questionary.Choice(title=f"{m['id']} ({m['provider']})", value=i)
            for i, m in enumerate(models)
        ]
        selected_indices = questionary.checkbox(
            "Which models would you like to customize?",
            choices=model_choices,
            style=editor.style,
        ).ask()

        if selected_indices is None:
            # User cancelled (Ctrl+C)
            print("Cancelled.")
            return

        for idx in selected_indices:
            model = models[idx]
            config = model.get('config', {})
            print(f"\n--- Customizing: {model['id']} ---")

            # Sampler params
            sampler_keys = ('temperature', 'top_p', 'top_k', 'min_p',
                            'max_tokens', 'repetition_penalty', 'repetition_context_size')
            before_sampler = {k: config[k] for k in sampler_keys if k in config}
            updated_sampler = editor.edit_sampler_params(before_sampler or None)
            if updated_sampler != before_sampler:
                if editor.confirm_changes(before_sampler, updated_sampler):
                    config.update(updated_sampler)

            # KV cache params (MLX only)
            if model.get('provider') == 'mlx':
                model_info = {
                    'size_gb': config.get('size_gb', 0),
                    'name': model['id'],
                }
                kv_params = editor.edit_kv_cache_params(model_info=model_info)
                if kv_params:
                    before_kv = {k: config.get(k) for k in kv_params}
                    if kv_params != before_kv and editor.confirm_changes(before_kv, kv_params):
                        config.update(kv_params)

            model['config'] = config

    # Print profile details before writing
    if hasattr(args, 'profile') and args.profile:
        profile = PROFILES.get(args.profile)
        if profile:
            print(f"\nProfile: {profile.name}")
            for key, value in profile.defaults.items():
                print(f"  {key:<25} = {value}")

    output_file = args.output or "models.toml"
    importer.generate_toml(models, output_file)

    print(f"\nFound {len(models)} models:")
    for model in models:
        print(f"  - {model['id']} ({model['provider']})")

    if hasattr(args, 'profile') and args.profile:
        print(f"\nApplied profile: {args.profile}")
    if overrides:
        print(f"\nApplied overrides: {overrides}")

    print(f"\nConfiguration written to: {output_file}")

    if args.merge:
        print("\nTo merge with existing models.toml, review the file and copy desired entries.")
    else:
        print("\nTo use this configuration, rename to models.toml or copy desired entries.")

    if not hasattr(args, 'profile') or not args.profile:
        print("\nAvailable profiles:")
        for name, profile in sorted(PROFILES.items()):
            summary_keys = [k for k in list(profile.defaults.keys())[:4]]
            summary = ", ".join(f"{k}={profile.defaults[k]}" for k in summary_keys)
            print(f"  --profile {name:<20} {profile.description}")
            print(f"           {'':20} [{summary}]")
