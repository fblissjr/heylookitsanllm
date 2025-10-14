# src/heylook_llm/model_importer.py
"""
Model importer for scanning directories and HuggingFace cache to auto-generate models.yaml entries.

Supports:
- MLX models (detects by mlx_config.json or model.safetensors.index.json)
- GGUF models (detects by .gguf extension)
- Vision models (detects by mmproj files or vision config)
- Smart defaults based on model size, type, and quantization
- Override profiles for common use cases
"""

import os
import json
import yaml
import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass, field


def get_hf_cache_paths():
    """Get platform-specific HuggingFace cache paths."""
    if platform.system() == "Windows":
        # Windows uses %USERPROFILE%\.cache\huggingface\hub or AppData
        return [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/AppData/Local/huggingface/hub"),
        ]
    elif platform.system() == "Darwin":
        # macOS specific paths
        return [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/Library/Caches/huggingface/hub"),
        ]
    else:
        # Linux and other Unix-like systems
        return [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.huggingface/hub"),
        ]


# Common HF cache paths (platform-specific)
HF_CACHE_PATHS = get_hf_cache_paths()

@dataclass
class ModelProfile:
    """Profile with smart defaults for different model types."""
    name: str
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)

    def apply(self, config: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply profile defaults to config."""
        result = config.copy()

        # Apply defaults
        for key, value in self.defaults.items():
            if key not in result:
                # Handle dynamic values
                if callable(value):
                    result[key] = value(model_info)
                else:
                    result[key] = value

        return result


# Predefined profiles
PROFILES = {
    "fast": ModelProfile(
        name="fast",
        description="Optimized for speed - aggressive settings",
        defaults={
            "temperature": 0.3,
            "top_k": 10,
            "min_p": 0.1,
            "max_tokens": 256,
            "repetition_penalty": 1.0,
            "cache_type": lambda m: "quantized" if m.get('size_gb', 0) > 13 else "standard"
        }
    ),
    "balanced": ModelProfile(
        name="balanced",
        description="Balance between speed and quality",
        defaults={
            "temperature": 0.7,
            "top_k": 40,
            "min_p": 0.05,
            "max_tokens": 512,
            "repetition_penalty": 1.05,
            "cache_type": lambda m: "quantized" if m.get('size_gb', 0) > 30 else "standard"
        }
    ),
    "quality": ModelProfile(
        name="quality",
        description="Optimized for quality - conservative settings",
        defaults={
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 1024,
            "repetition_penalty": 1.1,
            "cache_type": "standard"
        }
    ),
    "memory": ModelProfile(
        name="memory",
        description="Optimized for low memory usage",
        defaults={
            "cache_type": "quantized",
            "kv_bits": 4,
            "kv_group_size": 32,
            "quantized_kv_start": 256,
            "max_kv_size": 1024,
            "max_tokens": 256
        }
    ),
    "interactive": ModelProfile(
        name="interactive",
        description="Optimized for chat/interactive use",
        defaults={
            "temperature": 0.7,
            "top_k": 30,
            "min_p": 0.05,
            "max_tokens": 512,
            "repetition_penalty": 1.05,
            "repetition_context_size": 20
        }
    )
}

# Smart defaults based on model characteristics
def get_smart_defaults(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate smart defaults based on model characteristics."""
    defaults = {}

    size_gb = model_info.get('size_gb', 0)
    is_quantized = model_info.get('is_quantized', False)
    is_vision = model_info.get('is_vision', False)
    provider = model_info.get('provider', 'mlx')

    # Temperature based on model type
    if 'instruct' in model_info.get('name', '').lower():
        defaults['temperature'] = 0.7  # More deterministic for instruct models
    elif 'chat' in model_info.get('name', '').lower():
        defaults['temperature'] = 0.8  # Slightly more creative for chat
    else:
        defaults['temperature'] = 0.9  # Base models can be more creative

    # Cache strategy based on size
    if provider == 'mlx':
        if size_gb > 30:
            defaults['cache_type'] = 'quantized'
            defaults['kv_bits'] = 4
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
        # Large models: faster sampling
        defaults['top_k'] = 20
        defaults['min_p'] = 0.05
        defaults['max_tokens'] = 256
    elif size_gb < 3:
        # Small models: can afford broader sampling
        defaults['top_k'] = 50
        defaults['top_p'] = 0.95
        defaults['max_tokens'] = 1024
    else:
        # Medium models: balanced
        defaults['top_k'] = 40
        defaults['min_p'] = 0.05
        defaults['max_tokens'] = 512

    # Vision model specifics
    if is_vision:
        defaults['max_tokens'] = min(defaults.get('max_tokens', 512), 512)  # Limit for vision models

    # Repetition penalty
    defaults['repetition_penalty'] = 1.05
    defaults['repetition_context_size'] = 20

    # Provider-specific
    if provider in ['llama_cpp', 'gguf']:  # Support both names
        defaults['n_gpu_layers'] = -1
        defaults['n_ctx'] = 8192 if size_gb > 30 else 4096
        defaults['n_batch'] = 512
        defaults['use_mmap'] = True
        defaults['use_mlock'] = False

    return defaults

class ModelImporter:
    """Scan directories and generate models.yaml entries."""

    def __init__(self, profile: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
        self.models = []
        self.existing_ids = set()
        self.profile = PROFILES.get(profile, PROFILES['balanced']) if profile else None
        self.overrides = overrides or {}

    def scan_directory(self, path: str) -> List[Dict]:
        """Scan a directory recursively for models."""
        path = Path(path).expanduser().resolve()
        logging.info(f"Scanning directory: {path}")

        if not path.exists():
            logging.error(f"Path does not exist: {path}")
            return []

        models = []
        dirs_scanned = 0
        files_found = 0

        # Look for model directories - follow symlinks
        for root, dirs, files in os.walk(path, followlinks=True):
            root_path = Path(root)
            dirs_scanned += 1

            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            # Log progress every 10 directories
            if dirs_scanned % 10 == 0:
                logging.debug(f"Scanned {dirs_scanned} directories, found {len(models)} models so far")

            # Log current directory being scanned
            rel_path = root_path.relative_to(path)
            if str(rel_path) != ".":
                logging.debug(f"Scanning: {rel_path}")

            # Check for MLX models
            if self._is_mlx_model(root_path):
                logging.info(f"Found MLX model in: {rel_path}")
                model = self._create_mlx_entry(root_path)
                if model:
                    models.append(model)
                    logging.info(f"Added MLX model: {model['id']}")

            # Check for GGUF models
            gguf_files = [f for f in files if f.endswith('.gguf')]
            if gguf_files:
                logging.info(f"Found {len(gguf_files)} GGUF files in: {rel_path}")
                files_found += len(gguf_files)

            for gguf_file in gguf_files:
                logging.info(f"Processing GGUF: {gguf_file}")
                model = self._create_gguf_entry(root_path / gguf_file)
                if model:
                    models.append(model)
                    logging.info(f"Added GGUF model: {model['id']}")

        logging.info(f"Scan complete: {dirs_scanned} directories scanned, {files_found} GGUF files found, {len(models)} models imported")

        return models

    def scan_hf_cache(self) -> List[Dict]:
        """Scan HuggingFace cache directories for models."""
        models = []

        for cache_path in HF_CACHE_PATHS:
            path = Path(cache_path).expanduser()
            if path.exists():
                logging.info(f"Scanning HF cache: {path}")

                # HF cache structure: models--org--name/snapshots/hash/
                models_dir = path / "models--*"
                for model_dir in path.glob("models--*"):
                    if model_dir.is_dir():
                        # Look in snapshots
                        snapshots = model_dir / "snapshots"
                        if snapshots.exists():
                            for snapshot in snapshots.iterdir():
                                if snapshot.is_dir():
                                    found_models = self._scan_hf_snapshot(snapshot)
                                    models.extend(found_models)

        return models

    def _scan_hf_snapshot(self, snapshot_path: Path) -> List[Dict]:
        """Scan a HF cache snapshot directory."""
        models = []

        # Check if it's an MLX model
        if self._is_mlx_model(snapshot_path):
            model = self._create_mlx_entry(snapshot_path)
            if model:
                # Extract org/name from path
                parts = snapshot_path.parent.parent.name.split("--")
                if len(parts) >= 2:
                    model['id'] = f"{parts[1]}/{parts[2]}" if len(parts) > 2 else parts[1]
                    model['config']['model_path'] = str(snapshot_path)
                models.append(model)

        # Check for GGUF files
        for gguf_file in snapshot_path.glob("*.gguf"):
            model = self._create_gguf_entry(gguf_file)
            if model:
                # Extract org/name from path
                parts = snapshot_path.parent.parent.name.split("--")
                if len(parts) >= 2:
                    base_id = f"{parts[1]}/{parts[2]}" if len(parts) > 2 else parts[1]
                    model['id'] = f"{base_id}-{gguf_file.stem}"
                    model['config']['model_path'] = str(gguf_file)
                models.append(model)

        return models

    def _is_mlx_model(self, path: Path) -> bool:
        """Check if a directory contains an MLX model."""
        # MLX model indicators
        mlx_indicators = [
            "mlx_config.json",
            "model.safetensors.index.json",
            "weights.00.safetensors",
            "model.00.safetensors",
            "config.json"  # Check config.json with additional validation
        ]

        for indicator in mlx_indicators:
            if (path / indicator).exists():
                # For config.json, do additional checks
                if indicator == "config.json":
                    # Must have safetensors files too
                    has_safetensors = any(path.glob("*.safetensors"))
                    if has_safetensors:
                        return True
                else:
                    return True
        return False

    def _is_vision_model(self, path: Path) -> bool:
        """Check if a model supports vision."""
        # Check for vision indicators
        vision_files = [
            "mmproj",
            "vision_tower",
            "image_encoder",
            "visual_encoder"
        ]

        for file in path.iterdir():
            if any(v in file.name.lower() for v in vision_files):
                return True

        # Check config for vision flags
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if any(key in config for key in ["vision_config", "is_vision", "image_size"]):
                        return True
            except:
                pass

        return False

    def _get_model_size(self, path: Path) -> Tuple[Optional[str], Optional[float]]:
        """Estimate model size from name or files.
        Returns: (size_string, size_in_gb)
        """
        path_str = str(path).lower()

        # Common size patterns
        size_patterns = [
            (r'(\d+)b', lambda x: (f"{x}B", float(x))),
            (r'(\d+\.\d+)b', lambda x: (f"{x}B", float(x))),
            (r'(\d+)m', lambda x: (f"{int(x)/1000:.1f}B" if int(x) >= 1000 else f"{x}M", int(x)/1000)),
        ]

        for pattern, formatter in size_patterns:
            match = re.search(pattern, path_str)
            if match:
                return formatter(match.group(1))

        # Try to estimate from file sizes
        if path.is_dir():
            total_size = 0
            for file in path.rglob("*.safetensors"):
                total_size += file.stat().st_size
            for file in path.rglob("*.gguf"):
                total_size += file.stat().st_size

            if total_size > 0:
                size_gb = total_size / (1024**3)
                if size_gb >= 1:
                    return f"{size_gb:.1f}B", size_gb
                else:
                    return f"{int(size_gb * 1000)}M", size_gb

        return None, None

    def _create_mlx_entry(self, path: Path) -> Optional[Dict]:
        """Create a models.yaml entry for an MLX model."""
        model_id = path.name

        # Skip if already processed
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        # Check if it's quantized
        is_quantized = any(q in path.name.lower() for q in ['4bit', '8bit', 'q4', 'q8'])

        # Detect vision capability
        is_vision = self._is_vision_model(path)

        # Get model size
        size_str, size_gb = self._get_model_size(path)

        # Build model info for smart defaults
        model_info = {
            'name': model_id,
            'provider': 'mlx',
            'is_quantized': is_quantized,
            'is_vision': is_vision,
            'size_gb': size_gb or 0
        }

        # Build tags
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

        # Detect model family
        model_lower = model_id.lower()
        if 'llama' in model_lower:
            tags.append("llama")
        elif 'qwen' in model_lower:
            tags.append("qwen")
        elif 'gemma' in model_lower:
            tags.append("gemma")
        elif 'mistral' in model_lower:
            tags.append("mistral")

        if 'instruct' in model_lower or 'chat' in model_lower:
            tags.append("instruct")

        # Start with base config
        config = {
            "model_path": str(path),
            "vision": is_vision
        }

        # Apply smart defaults
        smart_config = get_smart_defaults(model_info)
        config.update(smart_config)

        # Apply profile if set
        if self.profile:
            config = self.profile.apply(config, model_info)

        # Apply user overrides
        config.update(self.overrides)

        # Check for draft model
        if size_gb and size_gb < 1 and not is_vision:
            # Small text-only model might be good for drafting
            tags.append("draft")
            config["max_tokens"] = 128  # Limit for draft models

        entry = {
            "id": model_id,
            "provider": "mlx",
            "description": f"Auto-imported MLX model{' with vision' if is_vision else ''}{f' ({size_str})' if size_str else ''}",
            "tags": tags,
            "enabled": False,  # Default to disabled
            "config": config
        }

        return entry

    def _create_gguf_entry(self, path: Path) -> Optional[Dict]:
        """Create a models.yaml entry for a GGUF model."""
        model_id = path.stem  # Remove .gguf extension

        # Skip if already processed
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        # Check for mmproj file in same directory
        mmproj_files = list(path.parent.glob("*mmproj*.gguf"))
        is_vision = len(mmproj_files) > 0

        # Get quantization from name
        quant_match = re.search(r'(q\d+_[kKmM]|Q\d+_[kKmM])', path.name)
        quant = quant_match.group(1) if quant_match else "unknown"
        is_quantized = quant != "unknown"

        # Get model size
        size_str, size_gb = self._get_model_size(path)
        if not size_gb:
            # Estimate from file size for GGUF
            size_gb = path.stat().st_size / (1024**3)

        # Build model info for smart defaults
        model_info = {
            'name': model_id,
            'provider': 'gguf',  # Use the simpler name
            'is_quantized': is_quantized,
            'is_vision': is_vision,
            'size_gb': size_gb or 0
        }

        # Build tags
        tags = ["gguf"]
        if is_vision:
            tags.append("vision")
        if size_gb:
            if size_gb >= 30:
                tags.append("large")
            elif size_gb <= 3:
                tags.append("small")

        # Detect model family
        model_lower = model_id.lower()
        if 'llama' in model_lower:
            tags.append("llama")
        elif 'qwen' in model_lower:
            tags.append("qwen")
        elif 'mistral' in model_lower:
            tags.append("mistral")

        if 'instruct' in model_lower or 'chat' in model_lower:
            tags.append("instruct")

        # Start with base config
        config = {
            "model_path": str(path),
            "vision": is_vision
        }

        # Apply smart defaults (includes llama_cpp specific settings)
        smart_config = get_smart_defaults(model_info)
        config.update(smart_config)

        # Apply profile if set
        if self.profile:
            config = self.profile.apply(config, model_info)

        # Apply user overrides
        config.update(self.overrides)

        # Add vision config
        if is_vision and mmproj_files:
            config["mmproj_path"] = str(mmproj_files[0])

        # Add chat format if detectable and not overridden
        if 'chat_format' not in config:
            if 'llama' in model_lower:
                config["chat_format"] = "llama-3"
            elif 'qwen' in model_lower:
                config["chat_format"] = "qwen"
            elif 'mistral' in model_lower:
                config["chat_format"] = "mistral"

        entry = {
            "id": model_id,
            "provider": "gguf",  # Use the simpler name
            "description": f"Auto-imported GGUF model ({quant}){' with vision' if is_vision else ''}{f' ({size_gb:.1f}GB)' if size_gb else ''}",
            "tags": tags,
            "enabled": False,
            "config": config
        }

        return entry

    def generate_yaml(self, models: List[Dict], output_file: Optional[str] = None) -> str:
        """Generate models.yaml content from discovered models."""
        # Group by provider
        mlx_models = [m for m in models if m['provider'] == 'mlx']
        gguf_models = [m for m in models if m['provider'] in ['llama_cpp', 'gguf']]

        # Build the structure
        config = {
            "# Auto-generated models configuration": None,
            "default_model": models[0]['id'] if models else "none",
            "max_loaded_models": 1,
            "models": []
        }

        # Add MLX models
        if mlx_models:
            config["models"].append({"# --- MLX Models ---": None})
            config["models"].extend(mlx_models)

        # Add GGUF models
        if gguf_models:
            config["models"].append({"# --- GGUF Models ---": None})
            config["models"].extend(gguf_models)

        # Generate YAML
        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)

        # Clean up the output
        yaml_content = yaml_content.replace("'# ", "# ")
        yaml_content = yaml_content.replace(": null", "")
        yaml_content = yaml_content.replace("- # ", "\n  # ")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(yaml_content)
            logging.info(f"Wrote configuration to {output_file}")

        return yaml_content


def import_models(args):
    """CLI handler for model import."""
    # Parse overrides if provided
    overrides = {}
    if hasattr(args, 'override') and args.override:
        for override in args.override:
            key, value = override.split('=', 1)
            # Try to parse as number/bool
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

    # Create importer with profile and overrides
    importer = ModelImporter(
        profile=getattr(args, 'profile', None),
        overrides=overrides
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

    # Generate YAML
    output_file = args.output or "models_imported.yaml"
    yaml_content = importer.generate_yaml(models, output_file)

    print(f"\nFound {len(models)} models:")
    for model in models:
        print(f"  - {model['id']} ({model['provider']})")

    if hasattr(args, 'profile') and args.profile:
        print(f"\nApplied profile: {args.profile}")

    if overrides:
        print(f"\nApplied overrides: {overrides}")

    print(f"\nConfiguration written to: {output_file}")

    if args.merge:
        print("\nTo merge with existing models.yaml, review the file and copy desired entries.")
    else:
        print("\nTo use this configuration, replace your models.yaml or copy desired entries.")

    # Show profile options if no profile was used
    if not hasattr(args, 'profile') or not args.profile:
        print("\nAvailable profiles for different use cases:")
        for name, profile in PROFILES.items():
            print(f"  --profile {name:<12} {profile.description}")
