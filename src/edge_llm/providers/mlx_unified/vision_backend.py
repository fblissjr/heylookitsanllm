# src/edge_llm/providers/mlx_unified/vision_backend.py
"""Vision‑language back‑end that wraps **mlx‑vlm** and plugs in all the
performance niceties from `mlx_lm` (samplers, prompt‑cache, KV quant).
"""
from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, Mapping

from mlx_vlm.utils import load as vlm_load, stream_generate as vlm_stream

from .samplers import build as build_sampler
from .cache_helpers import build_or_load_cache

__all__ = ["VlmBackend"]

DEFAULTS: Mapping[str, Any] = {
    "max_tokens": 256,
    "temp": 1.0,
    "top_p": 0.95,
    "min_p": 0.0,
    "top_k": 0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "min_tokens_to_keep": 1,
    "seed": 42,
    "quantized_kv_start": 5000,
}


class VlmBackend:
    """Backend selected when a model (or the request) has `vision: true`."""

    name = "mlx-vlm"

    # --------------------------------------------------- public API ----------
    def load(self, model_path: str, *, adapter_path: str | None = None, **cfg):
        """Load VLM weights + processor and prime the prompt cache."""
        self.model, self.tokenizer = vlm_load(model_path, adapter_path=adapter_path)
        self.cache, self.quantize_fn = build_or_load_cache(self.model, cfg)
        return self

    # The provider calls this for every ChatCompletion / stream request
    def stream(self, prompt: str | list[int], images: list[Any], **kw: Any) -> Generator:
        # Merge defaults with runtime kwargs
        gen_cfg: Dict[str, Any] = {**DEFAULTS, **kw}

        sampler, processors = build_sampler(self.tokenizer, gen_cfg)

        gen = vlm_stream(
            self.model,
            self.tokenizer,
            prompt,
            images,
            sampler            = sampler,
            logits_processors  = processors,
            prompt_cache       = self.cache,
            max_tokens         = gen_cfg["max_tokens"],
            max_kv_size        = gen_cfg.get("max_kv_size"),
            kv_bits            = gen_cfg.get("kv_bits"),
            kv_group_size      = gen_cfg.get("kv_group_size", 64),
            quantized_kv_start = gen_cfg.get("quantized_kv_start", 0),
        )

        # Insert our in‑place quantisation call exactly like mlx_lm.generate._step
        for step in gen:
            self.quantize_fn()
            yield step
