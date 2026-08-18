# tests/unit/test_prompt_cache_gating.py
#
# Prefix reuse is only correct for plain full-precision KVCache:
# restore_kv_from_snapshot prefix-trims keys[..., :N, :], which is wrong for
# QuantizedKVCache (packed tuple state) and impossible for RotatingKVCache.
# Non-standard cache configs must bypass the cache path entirely -- both
# lookup and store (audit 2026-07-06; the silent-wrong-output risk the
# eligibility gate exists for). The claims predate the Q7 single-slot
# rewrite and survive it unchanged.

from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from heylook_llm.providers.common.prompt_cache import (
    get_global_cache_manager,
    process_prompt_with_cache,
    store_generation_cache,
)


class _TinyModel:
    """Just enough for make_cache: layers with head dims."""
    class _Layer:
        pass
    layers = [_Layer()]


def _seeded_cache(model_id, tokens):
    """Run a standard-cache generation so the model's slot has a snapshot."""
    manager = get_global_cache_manager()
    pc = manager.get_or_create_cache(model_id, _TinyModel(), {"cache_type": "standard"})
    with patch("heylook_llm.providers.common.prompt_cache.make_cache",
               return_value=[KVCache()]):
        process_prompt_with_cache(pc, tokens, _TinyModel(), {"cache_type": "standard"})
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, len(tokens), 4)), mx.zeros((1, 2, len(tokens), 4)))
    store_generation_cache(pc, tokens, [kv])
    return manager


def test_standard_cache_is_radix_eligible():
    tokens = list(range(64))
    manager = _seeded_cache("gate-std", tokens)
    assert manager.get_cache_info()["gate-std"]["slot_bytes"] > 0  # store happened


def test_quantized_cache_bypasses_radix_store():
    manager = get_global_cache_manager()
    cfg = {"cache_type": "quantized", "kv_bits": 8}
    pc = manager.get_or_create_cache("gate-quant", _TinyModel(), cfg)
    with patch("heylook_llm.providers.common.prompt_cache.make_cache",
               return_value=[KVCache()]):
        process_prompt_with_cache(pc, list(range(64)), _TinyModel(), cfg)
    assert pc._radix_eligible is False
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 64, 4)), mx.zeros((1, 2, 64, 4)))
    store_generation_cache(pc, list(range(64)), [kv])
    assert manager.get_cache_info()["gate-quant"]["slot_bytes"] == 0  # nothing published


def test_max_kv_size_bypasses_radix_lookup():
    tokens = list(range(64))
    _seeded_cache("gate-rot", tokens)  # a slot EXISTS; the config must refuse it
    manager = get_global_cache_manager()
    cfg = {"cache_type": "standard", "max_kv_size": 2048}
    pc = manager.get_or_create_cache("gate-rot", _TinyModel(), cfg)
    with patch("heylook_llm.providers.common.prompt_cache.make_cache",
               return_value=[KVCache()]) as mk:
        to_process, _ = process_prompt_with_cache(pc, tokens, _TinyModel(), cfg)
    # Would have been a cache hit (same tokens seeded) -- must miss instead.
    assert to_process == tokens
    assert mk.called
