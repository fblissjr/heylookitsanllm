# src/heylook_llm/cache_defaults.py
#
# RAM-relative KV-cache defaults -- ONE implementation (Wave 1 / 6a,
# 2026-07-28) shared by import-time smart defaults (model_service) and the
# load-time auto resolution (MLXProvider.load_model for entries with
# cache_type = None). Computing this at load is the point: an import-time
# copy froze the decision against whatever machine/weights existed at
# import and rotted when either changed.

from pathlib import Path
from typing import Any


def _system_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 64.0


# load_model recurs under idle-unload/LRU cycles; skip the rglob+stat pass
# when the dir is unchanged. Keyed on (path, dir mtime) -- coarse (a swapped
# shard set changes the dir listing, hence its mtime) but cheap.
_SIZE_CACHE: dict[str, tuple[float, float]] = {}


def weights_size_gb(model_path: str) -> float:
    """Actual weight bytes under a model dir (safetensors + gguf), in GB.

    0.0 for a missing/empty/non-dir path -- callers treat that as "small",
    never as an error.
    """
    path = Path(model_path)
    if not path.is_dir():
        return 0.0
    try:
        mtime = path.stat().st_mtime
        cached = _SIZE_CACHE.get(str(path))
        if cached is not None and cached[0] == mtime:
            return cached[1]
        total = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
        total += sum(f.stat().st_size for f in path.rglob("*.gguf"))
    except OSError:
        return 0.0
    size = total / (1024 ** 3)
    _SIZE_CACHE[str(path)] = (mtime, size)
    return size


def smart_cache_defaults(size_gb: float) -> dict[str, Any]:
    """Cache fields for a model of ``size_gb`` weight bytes on THIS machine.

    KV quantization is a memory/quality trade-off, so it must be
    RAM-relative, not an absolute weight threshold: a 40GB model is "large"
    on a 64GB MacBook and trivial on a 192GB Studio. Quantize only when the
    weights alone claim over ~35% of unified memory (leaving the rest for
    KV, vision towers, and the OS).

    max_kv_size is deliberately NEVER defaulted: it creates a
    RotatingKVCache that silently drops context beyond the cap --
    truncation is an explicit user choice, not a default.
    """
    if size_gb > _system_ram_gb() * 0.35:
        return {"cache_type": "quantized", "kv_bits": 8, "kv_group_size": 64}
    return {"cache_type": "standard"}
