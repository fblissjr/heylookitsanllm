"""Embedding recognition and semantic metadata. No MLX imports.

Architecture support is deliberately positive knowledge. Extend this table only
with an upstream-supported architecture and its reference fixtures.
"""
import json
from functools import lru_cache
from pathlib import Path

EMBEDDING_GEMMA2 = {"mlx": "embedding_gemma2", "gguf": "gemma-embedding2"}
# Published Matryoshka training contract, not inferred from the hidden width.
GEMMA2_DIMENSIONS = (128, 256, 512, 768)
GEMMA2_CONTEXT = 8192


def read_json(path):
    path = Path(path)
    try:
        stat = path.stat()
        return _read_json(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}


@lru_cache(maxsize=256)
def _read_json(path, mtime, size):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def embedding_info(provider, config):
    if provider not in EMBEDDING_GEMMA2:
        return None
    path = Path(config.get("model_path") or "")
    if provider == "gguf":
        from . import gguf_metadata
        if gguf_metadata.architecture(path) != EMBEDDING_GEMMA2[provider]:
            return None
        root = path.parent
        modalities = gguf_metadata.detect_modalities(path, Path(config["mmproj_path"]) if config.get("mmproj_path") else None)
        modalities = ["image" if m == "vision" else m for m in modalities]
        if "image" in modalities:
            modalities.append("video")
    else:
        root = path
        cfg = read_json(root / "config.json")
        if cfg.get("model_type") != EMBEDDING_GEMMA2.get(provider):
            return None
        modalities = ["text"]
        if cfg.get("vision_config") is not None:
            modalities.extend(("image", "video"))
        if cfg.get("audio_config") is not None:
            modalities.append("audio")
    prompts = read_json(root / "config_sentence_transformers.json").get("prompts", {})
    return {
        "modalities": modalities, "dimensions": list(GEMMA2_DIMENSIONS),
        "native_dimensions": GEMMA2_DIMENSIONS[-1], "max_input_tokens": GEMMA2_CONTEXT,
        "prompts": {k: v for k, v in prompts.items() if isinstance(v, str)},
        "normalization": "l2", "video_audio": "separate_audio_block",
        "provenance": "model_files_and_architecture_contract",
    }


def gguf_embedding_profile(config):
    """One non-causal batch must accommodate the whole expanded input."""
    context = config.get("ctx_size") or GEMMA2_CONTEXT
    if context > GEMMA2_CONTEXT:
        raise ValueError(f"EmbeddingGemma 2 context cannot exceed {GEMMA2_CONTEXT}")
    batch = config.get("n_batch") or context
    ubatch = config.get("n_ubatch") or context
    if not context <= ubatch <= batch:
        raise ValueError("Embeddings require ctx_size <= n_ubatch <= n_batch")
    if config.get("draft_model_path") or config.get("spec_type") or config.get("extra_args"):
        raise ValueError("Embedding models do not accept drafting or extra_args")
    return {"ctx_size": context, "n_batch": batch, "n_ubatch": ubatch}
