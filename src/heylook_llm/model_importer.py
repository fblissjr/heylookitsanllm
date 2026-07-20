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
    available_samplers,
    get_hf_cache_paths,
    get_smart_defaults,
)
from heylook_llm.providers.common.template_info import detect_chat_template_source

__all__ = [
    "ModelImporter",
    "available_samplers",
    "get_hf_cache_paths",
    "get_smart_defaults",
    "import_models",
]

HF_CACHE_PATHS = get_hf_cache_paths()


class ModelImporter:
    """Scan directories and generate models.toml entries."""

    def __init__(
        self,
        sampler: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
        chat_template_override: Optional[str] = None,
    ):
        self.models: list[dict] = []
        self.existing_ids: set[str] = set()
        self.sampler_name: Optional[str] = None
        if sampler:
            from heylook_llm.samplers import get_sampler_registry
            registry = get_sampler_registry()
            if sampler not in registry:
                raise ValueError(
                    f"Unknown sampler: {sampler}. Available: {registry.list_names()}"
                )
            self.sampler_name = sampler
        self.overrides = overrides or {}
        # CLI `--chat-template` override. When set, recorded on every
        # imported model regardless of what's in its folder. Users point at
        # a custom .jinja path or force "tokenizer_config" to bypass jinja.
        self.chat_template_override = chat_template_override

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

            config_data = self._read_model_config(root_path)

            if self._is_embedding_model(root_path, config_data):
                logging.info(f"Found embedding model in: {rel_path}")
                model = self._create_embedding_entry(root_path)
                if model:
                    models.append(model)
                    logging.info(f"Added embedding model: {model['id']}")
            elif self._is_mlx_model(root_path):
                logging.info(f"Found MLX model in: {rel_path}")
                model = self._create_mlx_entry(root_path, config_data)
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
        config_data = self._read_model_config(snapshot_path)

        if self._is_embedding_model(snapshot_path, config_data):
            model = self._create_embedding_entry(snapshot_path)
        elif self._is_mlx_model(snapshot_path):
            model = self._create_mlx_entry(snapshot_path, config_data)
        else:
            model = None

        if model:
            parts = snapshot_path.parent.parent.name.split("--")
            if len(parts) >= 2:
                model['id'] = f"{parts[1]}/{parts[2]}" if len(parts) > 2 else parts[1]
                model['config']['model_path'] = str(snapshot_path)
            models.append(model)

        return models

    def _read_model_config(self, path: Path) -> Optional[dict]:
        """Read and parse config.json from a model directory. Returns None on failure."""
        config_path = path / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _is_embedding_model(self, path: Path, config_data: Optional[dict] = None) -> bool:
        """Check if a directory contains an embedding model.

        Detects two signals (either is sufficient):
        - config.json has "use_bidirectional_attention": true
        - Presence of *_Dense subdirectories (sentence-transformer projection layers)
        """
        if config_data is None:
            config_data = self._read_model_config(path)

        if config_data and config_data.get("use_bidirectional_attention") is True:
            return True

        # Check for sentence-transformer Dense projection dirs
        if any(d.is_dir() and d.name.endswith("_Dense") for d in path.iterdir()):
            return True

        return False

    def _create_embedding_entry(self, path: Path) -> Optional[dict]:
        """Create a models.toml entry for an embedding model."""
        model_id = path.name
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        is_quantized = any(q in path.name.lower() for q in ['4bit', '8bit', 'q4', 'q8'])

        tags = ["embedding"]
        model_lower = model_id.lower()
        for family in ["llama", "qwen", "gemma", "mistral"]:
            if family in model_lower:
                tags.append(family)
                break
        if is_quantized:
            tags.append("quantized")

        config: dict[str, Any] = {
            "model_path": str(path),
            "max_length": 2048,
        }

        return {
            "id": model_id,
            "provider": "mlx_embedding",
            "description": "Auto-imported embedding model",
            "tags": tags,
            "enabled": True,
            "config": config,
        }

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

    def _has_vision_files(self, path: Path) -> bool:
        """Vision-tower / mmproj sidecar files -- the fallback signal for sparse
        checkpoints (GGUF/split) whose config.json lacks a vision block."""
        vision_files = ["mmproj", "vision_tower", "image_encoder", "visual_encoder"]
        try:
            return any(
                any(v in f.name.lower() for v in vision_files) for f in path.iterdir()
            )
        except OSError:
            return False

    def detect_modalities(self, path: Path, config_data: Optional[dict] = None) -> list[str]:
        """The model's author-declared modality set, ``text`` always first.

        Primary signal is the config's OWN structure -- ``vision_config`` /
        ``audio_config`` sub-blocks and ``*_token_id`` keys are how the model
        declares which modalities it routes (ground truth). ``mmproj``-style
        weight files are a vision fallback for sparse checkpoints. Pure
        description: whether mlx-vlm can *load* it (the loader=auto gate) is a
        separate, library-aware decision made in the provider.

        Robust by construction -- a draft/MTP head or a dir with no/odd
        config.json yields ``["text"]`` rather than raising.
        """
        if config_data is None:
            config_data = self._read_model_config(path)
        cfg = config_data or {}

        mods = ["text"]
        if (
            "vision_config" in cfg
            or "image_token_id" in cfg
            or "image_token_index" in cfg   # LLaVA/Mistral/Pixtral spelling of the above
            or "vision_start_token_id" in cfg
            or "image_size" in cfg          # legacy signal, kept as a weak fallback
            or self._has_vision_files(path)
        ):
            mods.append("vision")
        if "audio_config" in cfg or "audio_token_id" in cfg:
            mods.append("audio")
        if "video_config" in cfg or "video_token_id" in cfg:
            mods.append("video")
        return mods

    def _is_vision_model(self, path: Path, config_data: Optional[dict] = None) -> bool:
        """Back-compat shim: vision is one modality of :meth:`detect_modalities`."""
        return "vision" in self.detect_modalities(path, config_data)

    def _get_model_size(self, path: Path) -> tuple[Optional[str], Optional[float]]:
        """Return (param-count label from the name, ACTUAL weight bytes in GB).

        These are different units and must not be conflated: the old code
        returned "7" from a `-7B` name as size_gb=7.0 (billions of params,
        not gigabytes) and fed it to get_smart_defaults, whose KV-quant
        threshold is real GB relative to RAM. size_gb now always comes from
        the safetensors byte-sum (matching the admin scan path); the name
        regex only supplies the human-facing label.
        """
        # Only the model DIRECTORY name -- matching the full path lets size-
        # looking fragments in parent dirs (e.g. a tmp dir "…680b…") win.
        path_str = path.name.lower()

        label = None
        for pattern, fmt in [
            (r'(\d+\.\d+)b', lambda x: f"{x}B"),
            (r'(\d+)b', lambda x: f"{x}B"),
            (r'(\d+)m', lambda x: f"{int(x)/1000:.1f}B" if int(x) >= 1000 else f"{x}M"),
        ]:
            match = re.search(pattern, path_str)
            if match:
                label = fmt(match.group(1))
                break

        size_gb = None
        if path.is_dir():
            total_size = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
            if total_size > 0:
                size_gb = total_size / (1024 ** 3)

        return label, size_gb

    def _create_mlx_entry(self, path: Path, config_data: Optional[dict] = None) -> Optional[dict]:
        """Create a models.toml entry for an MLX model."""
        model_id = path.name
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        is_quantized = any(q in path.name.lower() for q in ['4bit', '8bit', 'q4', 'q8'])
        modalities = self.detect_modalities(path, config_data)
        is_vision = "vision" in modalities
        size_str, size_gb = self._get_model_size(path)

        model_info = {
            'name': model_id, 'provider': 'mlx',
            'is_quantized': is_quantized, 'is_vision': is_vision,
            'size_gb': size_gb or 0,
        }

        tags = self._detect_tags(model_id, is_vision, is_quantized, size_gb)

        # ``modalities`` is the description of record; ``vision`` is retained as
        # a derived mirror for back-compat readers (config schema keeps them in
        # sync). Non-text modalities (audio/video) can only be expressed here.
        config: dict[str, Any] = {
            "model_path": str(path), "vision": is_vision, "modalities": modalities,
        }
        config.update(get_smart_defaults(model_info))

        # Chat-template source policy (C4.5):
        # 1) CLI --chat-template override wins.
        # 2) Else record the shared detection (same helper as the /v1/admin
        #    import route) so HF's auto-detection (which varies by
        #    transformers version) becomes explicit + user-editable.
        if self.chat_template_override:
            config["chat_template_source"] = self.chat_template_override
        else:
            detected = detect_chat_template_source(path)
            if detected:
                config["chat_template_source"] = detected

        if self.sampler_name:
            config["default_sampler"] = self.sampler_name
        config.update(self.overrides)

        if size_gb and size_gb < 1 and not is_vision:
            tags.append("draft")

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
        sampler=getattr(args, 'sampler', None),
        overrides=overrides,
        chat_template_override=getattr(args, 'chat_template', None),
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

    # Print sampler details before writing
    if getattr(args, 'sampler', None):
        from heylook_llm.samplers import get_sampler_registry
        registry = get_sampler_registry()
        if args.sampler in registry:
            print(f"\nSampler: {args.sampler}")
            for key, value in registry.get(args.sampler).items():
                print(f"  {key:<25} = {value}")

    output_file = args.output or "models.toml"
    importer.generate_toml(models, output_file)

    print(f"\nFound {len(models)} models:")
    for model in models:
        print(f"  - {model['id']} ({model['provider']})")

    if getattr(args, 'sampler', None):
        print(f"\nApplied sampler (recorded as default_sampler): {args.sampler}")
    if overrides:
        print(f"\nApplied overrides: {overrides}")

    print(f"\nConfiguration written to: {output_file}")

    if args.merge:
        print("\nTo merge with existing models.toml, review the file and copy desired entries.")
    else:
        print("\nTo use this configuration, rename to models.toml or copy desired entries.")

    if not getattr(args, 'sampler', None):
        from heylook_llm.samplers import get_sampler_registry
        print("\nAvailable samplers:")
        for info in get_sampler_registry().list_info():
            print(f"  --sampler {info['name']:<20} {info['description']}")
