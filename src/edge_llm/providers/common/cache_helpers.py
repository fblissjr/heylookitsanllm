# src/edge_llm/providers/common/cache_helpers.py
"""
Why this file exists:
This module abstracts away all KV cache logic, replicating the advanced features
of mlx-lm. It handles the creation of standard, rotating, or quantized caches
and prepares the quantization hook function that is called during generation,
ensuring memory efficiency.
"""
from __future__ import annotations

from typing import Tuple, Any, Callable
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import make_prompt_cache, load_prompt_cache

__all__ = ["build_or_load_cache"]

def build_or_load_cache(model, cfg: dict) -> Tuple[list[Any], Callable]:
    """
    Return (prompt_cache, quantize_hook) ready for generate_step.

    Args:
        model: The MLX model that owns this cache.
        cfg: Provider-level kwargs. Recognised keys: `prompt_cache_file`,
             `max_kv_size`, `kv_bits`, `kv_group_size`, `quantized_kv_start`.
    """
    # 1. (Optional) warm-start from a .safetensors cache file
    if pc_file := cfg.get("prompt_cache_file"):
        prompt_cache, _ = load_prompt_cache(pc_file, return_metadata=True)
    else:
        prompt_cache = make_prompt_cache(model, max_kv_size=cfg.get("max_kv_size"))

    # 2. Build a closure that applies quantization after each step
    def quantize_hook():
        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=cfg.get("quantized_kv_start", 5000),
            kv_group_size=cfg.get("kv_group_size", 64),
            kv_bits=cfg.get("kv_bits"),
        )

    return prompt_cache, quantize_hook
