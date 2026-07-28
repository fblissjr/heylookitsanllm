# src/heylook_llm/modality_detect.py
#
# Modality detection from a model dir's own files -- the vendor ground truth
# (Wave 1 / 6a derive-at-load, 2026-07-28). Extracted from ModelImporter so
# BOTH consumers share one implementation: the importer (scan-time display)
# and MLXModelConfig._resolve_modalities (load-time derivation for thin
# entries that materialize no `modalities`). Stdlib-only on purpose --
# config.py imports this, so it must never pull providers/ or heavy deps.

import json
from pathlib import Path
from typing import Optional


def read_model_config_json(path: Path) -> Optional[dict]:
    """Parse ``<path>/config.json``; None on missing/unreadable/invalid."""
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return None


def has_vision_weight_files(path: Path) -> bool:
    """Vision-tower / mmproj sidecar files -- the fallback signal for sparse
    checkpoints (GGUF/split) whose config.json lacks a vision block."""
    vision_files = ["mmproj", "vision_tower", "image_encoder", "visual_encoder"]
    try:
        return any(
            any(v in f.name.lower() for v in vision_files) for f in Path(path).iterdir()
        )
    except OSError:
        return False


def detect_modalities(path: Path, config_data: Optional[dict] = None) -> list[str]:
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
        config_data = read_model_config_json(path)
    cfg = config_data or {}

    mods = ["text"]
    if (
        "vision_config" in cfg
        or "image_token_id" in cfg
        or "image_token_index" in cfg   # LLaVA/Mistral/Pixtral spelling of the above
        or "vision_start_token_id" in cfg
        or "image_size" in cfg          # legacy signal, kept as a weak fallback
        or has_vision_weight_files(path)
    ):
        mods.append("vision")
    if "audio_config" in cfg or "audio_token_id" in cfg:
        mods.append("audio")
    if "video_config" in cfg or "video_token_id" in cfg:
        mods.append("video")
    return mods
