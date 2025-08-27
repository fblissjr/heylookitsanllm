# src/heylook_llm/providers/common/cache_helpers.py
from typing import Any, List
import mlx.nn as nn
import logging

# This relies on mlx_lm being installed.
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

def make_cache(model: nn.Module, config: dict) -> List[Any]:
    """
    Construct the model's cache based on the provider's configuration.
    
    Supported cache types:
    - standard: Regular KVCache (default)
    - quantized: 8-bit quantized cache for memory efficiency
    - rotating: Fixed-size cache that rotates old entries
    
    Configuration options:
    - cache_type: "standard", "quantized", or "rotating"
    - max_kv_size: Maximum size for rotating cache (required for rotating)
    - kv_group_size: Group size for quantized cache (default 64)
    - kv_bits: Number of bits for quantized cache (default 8)
    """
    cache_type = config.get("cache_type", "standard")

    # If the model has its own custom cache logic (like RecurrentGemma), let it handle it.
    if hasattr(model, "make_cache"):
        logging.debug(f"Using model's custom cache creation")
        return model.make_cache()

    num_layers = len(model.layers)

    if cache_type == "rotating":
        max_size = config.get("max_kv_size")
        if not max_size:
            raise ValueError("'max_kv_size' must be set for 'rotating' cache type.")
        logging.info(f"Creating rotating KV cache with max_size={max_size}")
        return [RotatingKVCache(max_size=max_size) for _ in range(num_layers)]

    elif cache_type == "quantized":
        # Create quantized cache with configurable parameters
        group_size = config.get("kv_group_size", 64)
        bits = config.get("kv_bits", 8)
        logging.info(f"Creating quantized KV cache with bits={bits}, group_size={group_size}")
        return [QuantizedKVCache(group_size=group_size, bits=bits) for _ in range(num_layers)]
        
    elif cache_type == "standard":
        logging.debug("Creating standard KV cache")
        return [KVCache() for _ in range(num_layers)]

    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")
