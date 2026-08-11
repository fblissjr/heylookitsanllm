# src/heylook_llm/model_importer.py
"""
CLI wrapper for model scanning and import.

Provides the ModelImporter class for filesystem scanning and TOML generation,
and the import_models CLI handler. Profiles, smart defaults, and HF cache paths
are defined in model_service.py (single source of truth).
"""

import glob
import logging
import os
import re
import tomli_w
from pathlib import Path
from typing import Any, Optional

from heylook_llm import gguf_metadata
from heylook_llm.model_service import get_hf_cache_paths
from heylook_llm.modality_detect import (
    detect_modalities,
    has_vision_weight_files,
    read_model_config_json,
)

__all__ = ["ModelImporter", "get_hf_cache_paths", "import_models"]

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
        return self._validate(models)

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
        return self._validate(models)

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

        return self._validate(models)

    def _validate(self, models: list[dict]) -> list[dict]:
        """Reject entries that would not load, BEFORE they reach models.toml.

        The importer writes the config file directly and validated nothing, so
        a single mistyped ``--override`` (``ctx_sze=8192``) produced a
        successful-looking import and a server that then refused to start --
        with the failure pointing at config load, far from the command that
        caused it, and no indication which of N imported entries was at fault.

        ``--override`` is free-form by design (it has to reach provider config
        fields this code does not enumerate), so the config CLASS is the only
        thing that knows what is valid. Same reasoning as the derived reload
        set and import allowlist: ask the schema, do not maintain a second
        list of field names here.
        """
        from heylook_llm.config import (
            PROVIDER_CONFIG_CLASSES,
            ModelConfig,
            configurable_fields,
        )

        for model in models:
            try:
                ModelConfig(**model)
            except Exception as e:
                provider = model.get("provider", "?")
                cls = PROVIDER_CONFIG_CLASSES.get(provider)
                valid = (
                    ", ".join(sorted(configurable_fields(cls))) if cls else "unknown provider"
                )
                raise ValueError(
                    f"import would write an invalid entry for "
                    f"'{model.get('id', '?')}' (provider={provider}): {e}\n"
                    f"Settable config keys for {provider}: {valid}"
                ) from e
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

        config: dict[str, Any] = {
            "model_path": str(path),
            "max_length": 2048,
        }

        return {
            "id": model_id,
            "provider": "mlx_embedding",
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

        "Primary" excludes mmproj-* sidecars and drafter sidecars (see
        ``_DRAFTER_PREFIXES``) -- those are paired onto a primary entry by
        ``_create_gguf_entry``, never their own entries.
        ``imatrix_*.gguf_file`` calibration data has a DIFFERENT extension
        (``.gguf_file``, not ``.gguf``) and is excluded by the suffix check
        itself.
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

    # Drafter-sidecar prefixes, mirroring llama.cpp's own sibling resolution
    # (common/download.cpp find_best_sibling): one prefix per speculative
    # family. A file carrying any of these is a drafter, never the primary --
    # and `mtp-` alone was leaving DSpark/DFlash/EAGLE3 drafters unpaired
    # (DeepSeek-V4-Flash ships `dspark-*.gguf`).
    _DRAFTER_PREFIXES = ("mtp-", "dspark-", "dflash-", "eagle3-")

    # llama.cpp shard naming: `<prefix>-00001-of-00005.gguf`. Only the FIRST
    # shard is loadable -- llama_model_loader hard-errors on any other
    # ("model must be loaded with the first split"), because it derives its
    # siblings from the given file's own split index.
    _SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

    @classmethod
    def _shard_index(cls, f: Path) -> Optional[int]:
        """1-based shard index if ``f`` is part of a split set, else None."""
        m = cls._SHARD_RE.search(f.name)
        return int(m.group(1)) if m else None

    @classmethod
    def _is_loadable_shard(cls, f: Path) -> bool:
        """True unless ``f`` is a non-first shard of a split set."""
        idx = cls._shard_index(f)
        return idx is None or idx == 1

    @classmethod
    def _is_drafter(cls, f: Path) -> bool:
        return f.name.lower().startswith(cls._DRAFTER_PREFIXES)

    @classmethod
    def _servable_size(cls, f: Path) -> int:
        """Bytes this entry would actually serve.

        For a first shard that is the WHOLE split set, not the 5 MB index
        shard -- otherwise a sharded 155 GB model loses `max()` to any
        standalone .gguf sitting beside it.
        """
        m = cls._SHARD_RE.search(f.name)
        if m is None:
            return f.stat().st_size
        prefix = f.name[: m.start()]
        return sum(
            s.stat().st_size
            for s in f.parent.glob(f"{glob.escape(prefix)}-*-of-*.gguf")
        )

    def _pick_primary_gguf(self, path: Path) -> Optional[Path]:
        """The primary servable .gguf weight file, largest wins if several."""
        candidates = [
            f for f in self._iter_root_gguf_files(path)
            # "mmproj" appears as a prefix (unsloth: mmproj-F16.gguf) OR a
            # suffix (google: gemma-4-E4B-it-mmproj.gguf) -- match anywhere.
            if "mmproj" not in f.name.lower()
            and not self._is_drafter(f)
            and self._is_loadable_shard(f)
        ]
        if not candidates:
            return None
        return max(candidates, key=self._servable_size)

    # Precision preference for the multimodal projector sidecar: F16 is the
    # sweet spot for vision-tower activations, BF16 next, F32 (largest/
    # slowest) is the last resort rather than the default.
    _MMPROJ_PRECISION_PREFERENCE = ("mmproj-f16.gguf", "mmproj-bf16.gguf", "mmproj-f32.gguf")

    def _mmprojs_in(self, path: Path) -> dict:
        return {
            f.name.lower(): f
            for f in self._iter_root_gguf_files(path)
            # anywhere, not prefix-only: google names projectors <model>-mmproj.gguf
            if "mmproj" in f.name.lower()
        }

    def _pick_mmproj(self, path: Path, primary: Optional[Path] = None) -> Optional[Path]:
        """Best mmproj sidecar by precision preference, else any mmproj* file.

        Searches the repo root one level up for a per-quant VARIANT folder,
        exactly as :meth:`_pick_draft` does -- a multimodal model shipped as
        quant subdirectories keeps its projector beside them. Without this the
        projector is silently dropped, and a silently-dropped projector is
        worse than a loud failure: the model imports as text-only and its
        vision (and audio) simply never work.
        """
        candidates = self._mmprojs_in(path)
        if not candidates and primary is not None and self._is_variant_dir(path, primary):
            candidates = self._mmprojs_in(path.parent)
        if not candidates:
            return None
        for preferred in self._MMPROJ_PRECISION_PREFERENCE:
            if preferred in candidates:
                return candidates[preferred]
        # Arbitrary mmproj-named file that doesn't match a known precision
        # suffix -- pick deterministically (sorted by name) rather than
        # dict/iteration order.
        return sorted(candidates.values(), key=lambda f: f.name)[0]

    def _pick_draft(self, path: Path, primary: Optional[Path] = None) -> Optional[Path]:
        """Drafter sidecar (``_DRAFTER_PREFIXES``), if any.

        An MTP/ subdirectory may hold additional precision variants of the
        same drafter -- those are alternates, never used here. A servable
        pairing needs exactly one drafter path, and only a root-level file is
        that path. A sharded drafter is picked at its first shard, same rule
        as the primary.

        When the weights sit in a per-quant VARIANT folder, the repo root one
        level up is also searched: HF ships the drafter beside the quant
        folders, not inside them, so DeepSeek-V4-Flash's ``dspark-*.gguf``
        lives next to ``UD-IQ4_XS/`` rather than in it. Searching only the
        model's own directory silently drops it.
        """
        candidates = self._drafters_in(path)
        if not candidates and primary is not None and self._is_variant_dir(path, primary):
            candidates = self._drafters_in(path.parent)
        if not candidates:
            return None
        return max(candidates, key=self._servable_size)

    def _drafters_in(self, path: Path) -> list:
        return [
            f for f in self._iter_root_gguf_files(path)
            if self._is_drafter(f) and self._is_loadable_shard(f)
        ]

    @classmethod
    def _model_name_from_file(cls, primary: Path) -> str:
        """The weight file's model name: basename minus shard suffix and ``.gguf``."""
        return cls._SHARD_RE.sub("", primary.name).removesuffix(".gguf")

    @classmethod
    def _is_variant_dir(cls, path: Path, primary: Path) -> bool:
        """Whether ``path`` is a per-quant VARIANT folder, not the model folder.

        HF repos that ship many quants of one big model put each in its own
        subdirectory (unsloth's large models; also the layout llama-server's
        own ``--models-dir`` documents), so a download preserving repo
        structure looks like::

            <repo>/dspark-<model>-Q8_0.gguf          <- sidecars at repo root
            <repo>/UD-IQ4_XS/<model>-UD-IQ4_XS-00001-of-00004.gguf

        The tell is that a variant folder's name is already spelled out in the
        weight file's own name -- ``UD-IQ4_XS`` inside
        ``...-UD-IQ4_XS-00001-of-00004``. A directory-named repo
        (``unsloth_gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf``)
        fails that test, which is what keeps existing behaviour intact.

        Two things follow from it: the id must come from the file, and sidecars
        must be looked for one level UP as well.

        The comparison is against the SHARD-STRIPPED name and requires a
        PROPER substring. Both guards are load-bearing: ``foo/foo.gguf`` and
        ``foo/foo-00001-of-00002.gguf`` are ordinary model directories, and a
        plain "is the dir name inside the file name" test calls both of them
        variant folders -- which would then let them adopt a drafter belonging
        to some unrelated sibling model upstairs.
        """
        base = cls._model_name_from_file(primary).lower()
        name = path.name.lower()
        return name != base and name in base

    @classmethod
    def _gguf_model_id(cls, path: Path, primary: Path) -> str:
        """Model id: the directory name, or the weight file's own name when the
        directory only labels a quant (see :meth:`_is_variant_dir`).

        Taking the directory name in a variant layout yields ``UD-IQ4_XS`` --
        uninformative, and colliding across every model quantised the same way.
        No id already in a models.toml moves: directory-named repos are not
        variant dirs.
        """
        if cls._is_variant_dir(path, primary):
            return cls._model_name_from_file(primary)
        return path.name

    def _create_gguf_entry(self, path: Path) -> Optional[dict]:
        """Create a models.toml entry for a GGUF model (served by llama-server)."""
        primary = self._pick_primary_gguf(path)
        if primary is None:
            return None

        model_id = self._gguf_model_id(path, primary)
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        mmproj = self._pick_mmproj(path, primary)
        draft = self._pick_draft(path, primary)

        # Modality DESCRIPTION read from the projector's own header
        # (clip.has_vision_encoder / clip.has_audio_encoder) rather than
        # inferred from "an mmproj exists". The two disagree on every omni
        # projector: gemma-4's mmproj sets BOTH flags, so presence-only
        # detection silently dropped its audio tower.
        modalities = gguf_metadata.detect_modalities(primary, mmproj)

        config: dict[str, Any] = {
            "model_path": str(primary),
            "modalities": modalities,
        }
        # Thinking capability from the GGUF's OWN embedded chat template, by
        # the same enable_thinking rule the MLX path uses. This was a manual
        # flag on the grounds that GGUF metadata had nothing cheap to probe;
        # reading the header directly is that cheap probe. Left unset when
        # there is no template (an MTP head legitimately has none) rather
        # than asserting a false.
        thinking = gguf_metadata.supports_thinking(primary)
        if thinking is not None:
            config["supports_thinking"] = thinking
        if mmproj is not None:
            config["mmproj_path"] = str(mmproj)
        if draft is not None:
            config["draft_model_path"] = str(draft)
        # spec_type is DELIBERATELY left unset even when a draft sidecar is
        # paired: whether speculative decoding actually helps is measured
        # per-model (draft-accept rate varies a lot), so import only pairs
        # the drafter PATH automatically -- turning spec decode ON via
        # spec_type stays an explicit owner choice, never inferred here.
        #
        # But WHICH spec type a drafter requires is a fact about the file, not
        # a choice, and guessing it wrong is a load failure. Report it so the
        # decision is "do I want this on", not "what is this drafter called".
        if draft is not None:
            spec_type = gguf_metadata.infer_spec_type(draft)
            if spec_type:
                logging.info(
                    f"[import] {model_id}: paired drafter {draft.name} -- "
                    f"set spec_type = \"{spec_type}\" to enable speculative decoding"
                )
            else:
                logging.warning(
                    f"[import] {model_id}: drafter {draft.name} has no recognised "
                    f"spec-type prefix (mtp-/dspark-/dflash-/eagle3-); spec_type "
                    f"must be set by hand"
                )

        if self.sampler_name:
            config["default_sampler"] = self.sampler_name
        config.update(self.overrides)

        return {
            "id": model_id, "provider": "gguf", "enabled": True, "config": config,
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

    def _create_mlx_entry(self, path: Path, config_data: Optional[dict] = None) -> Optional[dict]:
        """Create a models.toml entry for an MLX model."""
        model_id = path.name
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        # Derive-at-load (6a, 2026-07-28): entries are THIN -- path + operator
        # intent only. modalities/vision are detected at config-load time
        # (MLXModelConfig._resolve_modalities, same shared detector), the
        # chat-template source is auto-resolved at model load
        # (template_info.py), and description/tags auto-text is not worth
        # storing. Materializing any of these is a copy that rots when the
        # model dir changes in place. Only an explicit CLI --chat-template
        # override (operator intent) is recorded.
        config: dict[str, Any] = {"model_path": str(path)}

        if self.chat_template_override:
            config["chat_template_source"] = self.chat_template_override

        if self.sampler_name:
            config["default_sampler"] = self.sampler_name
        config.update(self.overrides)

        return {
            "id": model_id, "provider": "mlx", "enabled": True, "config": config,
        }

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
