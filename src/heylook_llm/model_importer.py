# src/heylook_llm/model_importer.py
"""
Filesystem scanner behind model discovery.

ModelImporter turns what is on disk into entry-shaped dicts for discovery
(model_registry.scan), layering each model's own model.heylook.toml over what
it derives. It writes nothing: the `heylookllm import` CLI and its TOML writer
were retired in v2.0.72, since every model lives in a [scan].folders watch
folder.
"""

import glob
import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Optional

from heylook_llm import gguf_metadata
from heylook_llm.modality_detect import (
    detect_modalities,
    has_vision_weight_files,
    read_model_config_json,
)

__all__ = ["ModelImporter"]

# A model's own settings, in its own folder (plan_registry_sidecars Phase 2).
# The `<what>.heylook.<ext>` shape of chat_template.heylook.jinja: no vendor
# ships the name, so a re-download never overwrites it.
SIDECAR_FILENAME = "model.heylook.toml"


class ModelImporter:
    """Scan directories into heylook.toml-shaped entries."""

    def __init__(self):
        self.models: list[dict] = []
        # Ids already produced by this instance, so one scan never yields the
        # same id twice. Discovery uses a fresh instance per call.
        self.existing_ids: set[str] = set()
        # (path, reason) for each model a scan found and dropped because its
        # entry would not validate. Dropped, not raised: one bad directory
        # must not take its whole folder's models with it.
        self.rejected: list[tuple[str, str]] = []

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
            elif self._is_embedding_checkpoint(root_path, config_data):
                logging.info(f"Skipping embedding checkpoint (no provider serves it): {rel_path}")
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

    def _apply_sidecar(self, entry: dict, folder: Path) -> Optional[dict]:
        """Layer ``folder/model.heylook.toml`` over the derived ``entry``.

        The file holds provider-config fields by name, plus ``unset``, a list
        of derived fields to drop (TOML has no null: ``unset =
        ["draft_model_path", "spec_type"]`` turns off a spec decode discovery
        found). A relative ``*_path`` value is relative to the model folder.
        ``model_path`` and an id are not settable: the file's location is the
        model's identity, and the id is the folder name (owner decision
        2026-09-24).

        Returns the entry with ``derived`` (the config before the file) and
        ``sidecar`` (the file) recorded, so the engine contract can say which
        values the file set. A file that does not parse, or sets what it may
        not, rejects the model into ``self.rejected`` (None): serving it on
        derived defaults would silently drop the owner's settings.
        """
        f = folder / SIDECAR_FILENAME
        if not f.is_file():
            return entry
        try:
            with open(f, "rb") as fp:
                data = tomllib.load(fp)
            unset = data.pop("unset", [])
            if not isinstance(unset, list) or not all(isinstance(k, str) for k in unset):
                raise ValueError("`unset` must be a list of field names")
            for key in ("model_path", "id"):
                if key in data or key in unset:
                    raise ValueError(f"`{key}` is not settable here: the folder is the model")
            from heylook_llm.config import PROVIDER_CONFIG_CLASSES

            fields = PROVIDER_CONFIG_CLASSES[entry["provider"]].model_fields
            if unknown := sorted(set(unset) - set(fields)):
                raise ValueError(f"`unset` names no field of this engine: {', '.join(unknown)}")
        except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
            logging.warning("[scan] not served: %s: %s", f, e)
            self.rejected.append((str(f), str(e)))
            return None
        derived = dict(entry["config"])
        config = {k: v for k, v in derived.items() if k not in unset}
        for key, value in data.items():
            if key.endswith("_path") and isinstance(value, str) and not Path(value).expanduser().is_absolute():
                value = str(folder / value)
            config[key] = value
        return {**entry, "config": config, "derived": derived, "sidecar": str(f)}

    def _validate(self, models: list[dict]) -> list[dict]:
        """Drop entries that would not load, at scan time, and record each in
        ``self.rejected``.

        Every entry goes through ModelConfig, so the config CLASS decides what
        is valid (the same reasoning as the derived reload set: ask the
        schema, never a second list of field names). This used to RAISE,
        which failed the whole folder: with one scan folder, one malformed
        model directory unserved every discovered model.
        """
        from heylook_llm.config import (
            PROVIDER_CONFIG_CLASSES,
            ModelConfig,
            configurable_fields,
        )

        kept = []
        for model in models:
            try:
                ModelConfig(**model)
            except Exception as e:
                provider = model.get("provider", "?")
                cls = PROVIDER_CONFIG_CLASSES.get(provider)
                valid = (
                    ", ".join(sorted(configurable_fields(cls))) if cls else "unknown provider"
                )
                path = str((model.get("config") or {}).get("model_path") or model.get("id", "?"))
                reason = (f"invalid entry for '{model.get('id', '?')}' (provider={provider}): "
                          f"{e}\nSettable config keys for {provider}: {valid}")
                logging.warning("[scan] not served: %s", reason)
                self.rejected.append((path, reason))
                continue
            kept.append(model)
        return kept

    def _read_model_config(self, path: Path) -> Optional[dict]:
        """Delegates to the shared reader (modality_detect.py, 6a)."""
        return read_model_config_json(path)

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

    def _is_embedding_checkpoint(self, path: Path, config_data: Optional[dict]) -> bool:
        """A sentence-embedding checkpoint: not servable by any provider.

        The `mlx_embedding` provider that used to claim these went in
        v2.0.41. Without this guard they fall through to the mlx detector,
        which sees config.json + safetensors and imports an ENABLED chat
        entry for a model that has no causal head -- a load-time 500
        dressed as a servable model. Same shape as the drafter skip above:
        refuse it here, before the mlx branch. Two signals, either suffices:
        config.json's ``use_bidirectional_attention: true``, or a
        sentence-transformers ``*_Dense`` projection directory.
        """
        if config_data and config_data.get("use_bidirectional_attention") is True:
            return True
        try:
            return any(d.is_dir() and d.name.endswith("_Dense") for d in path.iterdir())
        except OSError:
            return False

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
    _SHARD_RE = gguf_metadata.SPLIT_RE

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
        exactly as :meth:`_pick_spec` does -- a multimodal model shipped as
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

    def _pick_spec(self, path: Path, primary: Path) -> tuple[Optional[Path], Optional[str], Optional[str]]:
        """``(drafter, spec_type to pin, why)`` for this model, or all None.

        Spec decode is on by default wherever a model ships a drafter (owner
        decision 2026-09-24), so discovery looks for one in this order and
        takes the first rung that answers:

        1. A drafter file beside the weights (``_DRAFTER_PREFIXES``), or in
           the repo folder one level up when the weights sit in a per-quant
           variant folder; the largest wins.
        2. A drafter file in an immediate subfolder (an ``MTP/`` folder) of
           either place; the first by path wins when there are several.
        3. An MTP head built into the weights: llama.cpp's own rule on the
           target (``gguf_metadata.spec_type_from_gguf``), read across every
           split. No file; ``spec_type = "draft-mtp"`` makes llama.cpp load
           the head from the target.
        4. A drafter file in a NEIGHBOURING folder (same parent) whose own
           header names the same model as the target's
           (``gguf_metadata.model_names``). Two quantizers of one model ship
           as sibling folders, and the drafter often comes with only one.

        ``spec_type`` is pinned only where llama.cpp cannot infer it from the
        drafter's first split: a built-in head, a sharded drafter, and eagle3
        (which llama.cpp does not infer at all). Everything found and not used
        is logged with the reason.
        """
        roots = [path]
        if self._is_variant_dir(path, primary):
            roots.append(path.parent)

        # The model's own folder first; the repo root only when it has none.
        own = next((c for c in map(self._drafters_in, roots) if c), [])
        if own:
            draft = max(own, key=self._servable_size)
            return draft, self._pin_for(draft), f"drafter beside the weights: {draft.name}"

        nested = sorted(
            f for r in roots for sub in self._subdirs(r) if sub not in roots
            for f in self._drafters_in(sub))
        # A variant layout's sibling quant folders are subfolders of the repo
        # root too, and hold full models, never drafters by name -- the
        # prefix filter is what keeps them out.
        if nested:
            draft = nested[0]
            extra = f" ({len(nested)} found; first by path)" if len(nested) > 1 else ""
            return draft, self._pin_for(draft), f"drafter in subfolder {draft.parent.name}/: {draft.name}{extra}"

        if gguf_metadata.spec_type_from_gguf(gguf_metadata.splits(primary)) == "draft-mtp":
            return None, "draft-mtp", "MTP head built into the weights"

        names = gguf_metadata.model_names(primary)
        parent = roots[-1].parent
        matched, unmatched = [], []
        for sib in self._subdirs(parent):
            if sib in roots:
                continue
            for f in self._drafters_in(sib):
                (matched if names and names & gguf_metadata.model_names(f) else unmatched).append(f)
        for f in unmatched:
            logging.debug(f"[import] {path.name}: neighbouring drafter {f.parent.name}/{f.name} "
                          f"names a different model; not paired")
        if matched:
            draft = sorted(matched)[0]
            extra = f" ({len(matched)} matched; first by path)" if len(matched) > 1 else ""
            return draft, self._pin_for(draft), (
                f"drafter from neighbouring folder {draft.parent.name}/: {draft.name}, "
                f"matched on the model name in both headers{extra}")
        return None, None, None

    @staticmethod
    def _subdirs(path: Path) -> list:
        try:
            return sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith("."))
        except OSError:
            return []

    @classmethod
    def _pin_for(cls, draft: Path) -> Optional[str]:
        """The ``--spec-type`` to pin for a drafter file, or None when llama.cpp
        infers it itself from the file (its first split)."""
        if draft.name.lower().startswith("eagle3-"):
            return "draft-eagle3"  # llama.cpp infers no eagle3 from a header
        if cls._shard_index(draft) is None:
            return None
        return gguf_metadata.spec_type_from_gguf(gguf_metadata.splits(draft))

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
        No id already in a heylook.toml moves: directory-named repos are not
        variant dirs.
        """
        if cls._is_variant_dir(path, primary):
            return cls._model_name_from_file(primary)
        return path.name

    def _create_gguf_entry(self, path: Path) -> Optional[dict]:
        """Create a heylook.toml entry for a GGUF model (served by llama-server)."""
        primary = self._pick_primary_gguf(path)
        if primary is None:
            return None

        model_id = self._gguf_model_id(path, primary)
        if model_id in self.existing_ids:
            return None
        self.existing_ids.add(model_id)

        mmproj = self._pick_mmproj(path, primary)
        draft, spec_type, spec_why = self._pick_spec(path, primary)

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
        # Spec decode is on wherever a drafter is found (owner decision
        # 2026-09-24): pairing the PATH turns it on (the provider emits
        # `-md` on draft_model_path alone and llama.cpp infers the type from
        # the drafter's header); spec_type is pinned only where llama.cpp
        # cannot infer it, and alone it reaches a built-in head. The off
        # switch is per model: a stored entry without these fields.
        if draft is not None:
            config["draft_model_path"] = str(draft)
        if spec_type is not None:
            config["spec_type"] = spec_type
        if spec_why:
            logging.info(f"[import] {model_id}: speculative decoding on -- {spec_why}")

        return self._apply_sidecar(
            {"id": model_id, "provider": "gguf", "config": config}, path)

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

    def _create_mlx_entry(self, path: Path, config_data: Optional[dict] = None) -> Optional[dict]:
        """Create a heylook.toml entry for an MLX model."""
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
        # model dir changes in place.
        config: dict[str, Any] = {"model_path": str(path)}

        return self._apply_sidecar(
            {"id": model_id, "provider": "mlx", "config": config}, path)
