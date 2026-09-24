# src/heylook_llm/cache_defaults.py
#
# Weight bytes under a model dir. It used to also hold the RAM-relative MLX
# KV-cache defaults; those fields were retired with the mlx-vlm engine (plan
# W10 stage 3), and the size probe is what remains in use (model_service).

from pathlib import Path


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
