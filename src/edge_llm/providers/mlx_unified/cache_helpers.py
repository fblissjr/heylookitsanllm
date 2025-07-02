# src/edge_llm/providers/mlx_unified/cache_helpers.py
"""Prompt‑cache helpers that replicate **exactly** what `mlx_lm.generate` does.

• builds the right cache type (`KVCache`, `RotatingKVCache`, `QuantizedKVCache`)
  via `make_prompt_cache()`
• supports on‑disk prompt‑cache files
• exposes a *quantize hook* that back‑ends can call after each forward
  (mirrors `maybe_quantize_kv_cache` in `mlx_lm.generate`)
"""
from __future__ import annotations

from typing import Tuple, Any
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    make_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
)

__all__ = [
    "build_or_load_cache",
    "save_prompt_cache",
]


def build_or_load_cache(model, cfg: dict) -> Tuple[list[Any], callable]:
    """Return *(prompt_cache, quantize_hook)* ready for `generate_step`.

    Parameters
    ----------
    model : nn.Module
        The MLX model that owns this cache.
    cfg : dict
        Provider‑level kwargs. Recognised keys (all optional):
        ``prompt_cache_file``, ``max_kv_size``, ``kv_bits``,
        ``kv_group_size``, ``quantized_kv_start``.
    """
    # 1)  (Optional) warm‑start from a .safetensors cache file
    pc_file = cfg.get("prompt_cache_file")
    if pc_file:
        prompt_cache, _ = load_prompt_cache(pc_file, return_metadata=True)
    else:
        prompt_cache = make_prompt_cache(model, max_kv_size=cfg.get("max_kv_size"))

    # 2)  Build a closure that applies quantisation after each step
    def quantize_hook():
        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start = cfg.get("quantized_kv_start", 0),
            kv_group_size      = cfg.get("kv_group_size", 64),
            kv_bits            = cfg.get("kv_bits"),
        )

    return prompt_cache, quantize_hook
