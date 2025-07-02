# src/edge_llm/providers/mlx_unified/text_backend.py
"""Text‑only back‑end that wraps **mlx‑lm** and exposes the same streaming API
expected by the provider layer.

It adds:
* prompt‑cache + KV‑quant hooks (via `cache_helpers`)
* sampler / logits‑processor construction (via `samplers`)
* default hyper‑params identical to the mlx‑lm CLI constants.
"""
from __future__ import annotations

from typing import Generator, Any

from mlx_lm.utils import load as lm_load
from mlx_lm.models import cache
from mlx_lm.models.cache import (
    QuantizedKVCache,
    load_prompt_cache,
)

class LmBackend:
    name = "mlx-text"

    def load(self, path, adapter=None, **cfg):
            self.model, self.tokenizer = lm_load(path, adapter_path=adapter)
            self.cache, self.q_fn      = cache(self.model, cfg)
            return self

    def stream(self, prompt, **kw):
            sampler, procs = build_sampling_objects(self.tokenizer, kw)
            return lm_stream(
                self.model,
                self.tokenizer,
                prompt,
                sampler            = sampler,
                logits_processors  = procs,
                prompt_cache       = self.cache,
                max_tokens         = kw.get("max_tokens", 256),
                max_kv_size        = kw.get("max_kv_size"),
                kv_bits            = kw.get("kv_bits"),
                kv_group_size      = kw.get("kv_group_size", 64),
                quantized_kv_start = kw.get("quantized_kv_start", 0),
            )
