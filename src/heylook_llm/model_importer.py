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
from heylook_llm.modality_detect import (
    detect_modalities,
    has_vision_weight_files,
    read_model_config_json,
)

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

            if self._is_drafter_checkpoint(config_data):
                # HF-format ASSISTANT/drafter SOURCE checkpoint (config.json +
                # safetensors -- the same on-disk shape as a real MLX model).
                # These pair with a GGUF's MTP head; they are not servable on
                # their own and must be refused BEFORE the mlx branch below
                # would otherwise happily import them.
                logging.info(f"Skipping drafter/assistant checkpoint (not servable): {rel_path}")
            elif self._is_embedding_model(root_path, config_data):
                logging.info(f"Found embedding model in: {rel_path}")
                model = self._create_embedding_entry(root_path)
                if model:
                    models.append(model)
                    logging.info(f"Added embedding model: {model['id']}")
            elif self._is_gguf_model(root_path):
                logging.info(f"Found GGUF model in: {rel_path}")
                model = self._create_gguf_entry(root_path)
                if model:
                    models.append(model)
                    logging.info(f"Added GGUF model: {model['id']}")
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

        if self._is_drafter_checkpoint(config_data):
            model = None
        elif self._is_embedding_model(snapshot_path, config_data):
            model = self._create_embedding_entry(snapshot_path)
        elif self._is_gguf_model(snapshot_path):
            model = self._create_gguf_entry(snapshot_path)
        elif self._is_mlx_model(snapshot_path):
            model = self._create_mlx_entry(snapshot_path, config_data)
        else:
            model = None

        if model:
            parts = snapshot_path.parent.parent.name.split("--")
            if len(parts) >= 2:
                model['id'] = f"{parts[1]}/{parts[2]}" if len(parts) > 2 else parts[1]
                # A gguf model_path is the primary .gguf FILE (sidecars live
                # alongside it) -- overwriting it with the snapshot DIRECTORY
                # would point GGUFModelConfig.model_path at the wrong thing.
                if model.get('provider') != 'gguf':
                    model['config']['model_path'] = str(snapshot_path)
            models.append(model)

        return models

    def _read_model_config(self, path: Path) -> Optional[dict]:
        """Delegates to the shared reader (modality_detect.py, 6a)."""
        return read_model_config_json(path)

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

    def _is_drafter_checkpoint(self, config_data: Optional[dict]) -> bool:
        """HF-format ASSISTANT/drafter SOURCE checkpoint.

        Signal: config.json's "architectures" contains a string with
        "Assistant" in it (e.g. "Gemma4AssistantForCausalLM",
        "Gemma4UnifiedAssistantForCausalLM"). On disk these look exactly
        like a real MLX model (config.json + model.safetensors), but they
        are drafter/MTP SOURCE checkpoints -- inputs to GGUF conversion,
        never servable on their own -- and must be refused before the mlx
        detector would otherwise claim them.
        """
        if not config_data:
            return False
        architectures = config_data.get("architectures") or []
        return any("assistant" in str(a).lower() for a in architectures)

    def _is_gguf_model(self, path: Path) -> bool:
        """Dir containing >=1 PRIMARY .gguf file (root level only).

        "Primary" excludes mmproj-* sidecars and mtp-* drafter sidecars --
        those are paired onto a primary entry by ``_create_gguf_entry``,
        never their own entries. ``imatrix_*.gguf_file`` calibration data
        has a DIFFERENT extension (``.gguf_file``, not ``.gguf``) and is
        excluded by the suffix check itself.
        """
        return self._pick_primary_gguf(path) is not None

    def _iter_root_gguf_files(self, path: Path):
        """Root-level (non-recursive) ``*.gguf`` files -- never ``.gguf_file``
        (imatrix calibration data) and never anything in a nested subdir
        (e.g. an MTP/ precision-variants folder)."""
        try:
            for f in path.iterdir():
                if f.is_file() and f.name.endswith(".gguf"):
                    yield f
        except OSError:
            return

    def _pick_primary_gguf(self, path: Path) -> Optional[Path]:
        """The primary servable .gguf weight file, largest wins if several."""
        candidates = [
            f for f in self._iter_root_gguf_files(path)
            # "mmproj" appears as a prefix (unsloth: mmproj-F16.gguf) OR a
            # suffix (google: gemma-4-E4B-it-mmproj.gguf) -- match anywhere.
            if "mmproj" not in f.name.lower()
            and not f.name.lower().startswith("mtp-")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.stat().st_size)

    # Precision preference for the multimodal projector sidecar: F16 is the
    # sweet spot for vision-tower activations, BF16 next, F32 (largest/
    # slowest) is the last resort rather than the default.
    _MMPROJ_PRECISION_PREFERENCE = ("mmproj-f16.gguf", "mmproj-bf16.gguf", "mmproj-f32.gguf")

    def _pick_mmproj(self, path: Path) -> Optional[Path]:
        """Best mmproj sidecar by precision preference, else any mmproj* file."""
        candidates = {
            f.name.lower(): f
            for f in self._iter_root_gguf_files(path)
            # anywhere, not prefix-only: google names projectors <model>-mmproj.gguf
            if "mmproj" in f.name.lower()
        }
        if not candidates:
            return None
        for preferred in self._MMPROJ_PRECISION_PREFERENCE:
            if preferred in candidates:
                return candidates[preferred]
        # Arbitrary mmproj-named file that doesn't match a known precision
        # suffix -- pick deterministically (sorted by name) rather than
        # dict/iteration order.
        return sorted(candidates.values(), key=lambda f: f.name)[0]

    def _pick_draft(self, path: Path) -> Optional[Path]:
        """Root-level mtp-*.gguf drafter sidecar, if any.

        An MTP/ subdirectory may hold additional precision variants of the
        same drafter -- those are alternates, never used here. A servable
        pairing needs exactly one drafter path, and only the root-level file
        is that path.
        """
        candidates = [
            f for f in self._iter_root_gguf_files(path)
            if f.name.lower().startswith("mtp-")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.stat().st_size)

    def _create_gguf_entry(self, path: Path) -> Optional[dict]:
        """Create a models.toml entry for a GGUF model (served by llama-server)."""
        model_id = path.name
        if model_id in self.existing_ids:
            return None

        primary = self._pick_primary_gguf(path)
        if primary is None:
            return None
        self.existing_ids.add(model_id)

        mmproj = self._pick_mmproj(path)
        draft = self._pick_draft(path)
        is_vision = mmproj is not None
        is_quantized = any(q in path.name.lower() for q in ['4bit', '8bit', 'q4', 'q8'])
        _, size_gb = self._get_model_size(path)

        # Modality DESCRIPTION mirrors detect_modalities' intent, but GGUF
        # dirs carry no config.json to read -- the only cheap signal is the
        # mmproj sidecar. Audio is deliberately NOT auto-detected: it would
        # need reading the GGUF's own metadata (out of scope here).
        modalities = ["text"]
        if is_vision:
            modalities.append("vision")

        tags = self._detect_tags(model_id, is_vision, is_quantized, size_gb)
        tags.append("gguf")

        config: dict[str, Any] = {
            "model_path": str(primary),
            "modalities": modalities,
        }
        if mmproj is not None:
            config["mmproj_path"] = str(mmproj)
        if draft is not None:
            config["draft_model_path"] = str(draft)
        # spec_type is DELIBERATELY left unset even when a draft sidecar is
        # paired: whether speculative decoding actually helps is measured
        # per-model (draft-accept rate varies a lot), so import only pairs
        # the drafter PATH automatically -- turning spec decode ON via
        # spec_type stays an explicit owner choice, never inferred here.

        if self.sampler_name:
            config["default_sampler"] = self.sampler_name
        config.update(self.overrides)

        return {
            "id": model_id,
            "provider": "gguf",
            "description": "Auto-imported GGUF model (llama-server)",
            "tags": tags,
            "enabled": True,
            "config": config,
        }

    def _has_vision_files(self, path: Path) -> bool:
        """Delegates to the shared detector (modality_detect.py, 6a)."""
        return has_vision_weight_files(path)

    def detect_modalities(self, path: Path, config_data: Optional[dict] = None) -> list[str]:
        """Delegates to the shared detector (modality_detect.py) -- ONE
        implementation serves both scan-time display and the load-time
        derivation in MLXModelConfig._resolve_modalities."""
        if config_data is None:
            config_data = self._read_model_config(path)
        return detect_modalities(path, config_data)

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
            # GGUF dirs have no *.safetensors -- sum *.gguf too (this also
            # covers mmproj/mtp sidecars, since rglob("*.gguf") doesn't
            # match imatrix's ".gguf_file" extension).
            total_size = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
            total_size += sum(f.stat().st_size for f in path.rglob("*.gguf"))
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
        _, size_gb = self._get_model_size(path)

        model_info = {
            'name': model_id, 'provider': 'mlx',
            'is_quantized': is_quantized, 'is_vision': is_vision,
            'size_gb': size_gb or 0,
        }

        # Derive-at-load (6a, 2026-07-28): entries are THIN -- path + operator
        # intent only. modalities/vision are detected at config-load time
        # (MLXModelConfig._resolve_modalities, same shared detector), the
        # chat-template source is auto-resolved at model load
        # (template_info.py), and description/tags auto-text is not worth
        # storing. Materializing any of these is a copy that rots when the
        # model dir changes in place. Only an explicit CLI --chat-template
        # override (operator intent) is recorded.
        config: dict[str, Any] = {"model_path": str(path)}
        config.update(get_smart_defaults(model_info))

        if self.chat_template_override:
            config["chat_template_source"] = self.chat_template_override

        if self.sampler_name:
            config["default_sampler"] = self.sampler_name
        config.update(self.overrides)

        return {
            "id": model_id, "provider": "mlx", "enabled": True, "config": config,
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

    # Stable section order for a mixed scan; anything outside this map
    # (a future provider) still gets emitted, under "Other Models".
    _SECTION_HEADERS: list[tuple[str, str]] = [
        ("mlx", "# --- MLX Models ---"),
        ("mlx_embedding", "# --- Embedding Models ---"),
        ("gguf", "# --- GGUF Models ---"),
    ]

    def generate_toml(self, models: list[dict], output_file: Optional[str] = None) -> str:
        """Generate models.toml content from discovered models.

        Entries are grouped into one ``# --- <Provider> Models ---`` section
        per provider (stable order: MLX, embedding, GGUF, then anything
        else) so a mixed scan reads as organized sections instead of one
        undifferentiated list under a single "MLX Models" header.
        """
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
        ]

        by_provider: dict[str, list[dict]] = {}
        for model in models:
            by_provider.setdefault(model.get("provider", "mlx"), []).append(model)

        known_providers = {key for key, _ in self._SECTION_HEADERS}
        for provider_key, header in self._SECTION_HEADERS:
            group = by_provider.get(provider_key, [])
            if not group:
                continue
            toml_lines.append(header)
            toml_lines.append("")
            for model in group:
                toml_lines.extend(self._model_to_toml_lines(model))
                toml_lines.append("")

        others = [
            model
            for provider_key, group in by_provider.items()
            if provider_key not in known_providers
            for model in group
        ]
        if others:
            toml_lines.append("# --- Other Models ---")
            toml_lines.append("")
            for model in others:
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

    # (Interactive per-model customization retired 2026-07-28 with config_tui:
    # dead under derive-at-load thin entries. Operator intent at import =
    # --sampler / --override flags; richer editing is the Wave 4 admin CRUD.)

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
