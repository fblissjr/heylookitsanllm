# src/heylook_llm/cache_defaults.py
#
# RAM-relative KV-cache defaults -- ONE implementation (Wave 1 / 6a,
# 2026-07-28) shared by import-time smart defaults (model_service) and the
# load-time auto resolution (MLXProvider.load_model for entries with
# cache_type = None). Computing this at load is the point: an import-time
# copy froze the decision against whatever machine/weights existed at
# import and rotted when either changed.

import logging
from pathlib import Path
from typing import Any


def _system_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 64.0


def weights_size_gb(model_path: str) -> float:
    """Actual weight bytes under a model dir (safetensors + gguf), in GB.

    0.0 for a missing/empty/non-dir path -- callers treat that as "small",
    never as an error.
    """
    path = Path(model_path)
    if not path.is_dir():
        return 0.0
    try:
        total = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
        total += sum(f.stat().st_size for f in path.rglob("*.gguf"))
    except OSError:
        return 0.0
    return total / (1024 ** 3)


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


def resolve_cache_config(config: dict) -> dict[str, Any]:
    """Updates for a provider config whose ``cache_type`` is None (= auto).

    Returns only the fields to fill in: an explicit ``cache_type`` yields
    {}, and knobs the operator pinned (kv_bits/kv_group_size) are never
    overridden even when auto picks the quantized cache type.
    """
    if config.get("cache_type") is not None:
        return {}
    defaults = smart_cache_defaults(weights_size_gb(config.get("model_path", "")))
    updates = {
        k: v for k, v in defaults.items() if config.get(k) is None
    }
    if updates.get("cache_type") != "standard":
        logging.info(f"Auto cache defaults resolved at load: {updates}")
    return updates
