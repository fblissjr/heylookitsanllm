# src/heylook_llm/providers/common/cache_helpers.py
from typing import Any, List
import mlx.nn as nn

# This relies on mlx_lm being installed.
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

def make_cache(model: nn.Module, config: dict) -> List[Any]:
    """
    Construct the model's cache based on the provider's configuration.
    """
    cache_type = config.get("cache_type", "standard")

    # If the model has its own custom cache logic (like RecurrentGemma), let it handle it.
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)

    if cache_type == "rotating":
        max_size = config.get("max_kv_size")
        if not max_size:
            raise ValueError("'max_kv_size' must be set for 'rotating' cache type.")
        return [RotatingKVCache(max_size=max_size) for _ in range(num_layers)]

    # For quantized, we start with a standard cache that will be converted later.
    elif cache_type == "quantized" or cache_type == "standard":
        return [KVCache() for _ in range(num_layers)]

    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")
