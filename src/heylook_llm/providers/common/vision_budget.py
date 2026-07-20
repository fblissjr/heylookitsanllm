# src/heylook_llm/providers/common/vision_budget.py
"""Model-agnostic per-image visual token budget.

Maps the wire-level knob (``ChatRequest.vision_tokens``: target visual
tokens per image) onto whatever budget parameter the loaded model's image
processor actually exposes -- duck-typed on processor attributes, never on
model names:

- gemma-4 family (has ``max_soft_tokens``): discrete buckets; the budget
  snaps to the nearest supported value (ties prefer the smaller/cheaper).
- qwen2/3-VL family (has ``patch_size`` + ``merge_size``): continuous
  pixel budget; tokens x (patch x merge)^2 -> ``max_pixels``.
- anything else: {} -- the request degrades gracefully to the processor's
  own defaults.

The returned dict is passed as call-time kwargs through
``mlx_vlm.utils.prepare_inputs`` -> the transformers processor, which
validates values itself (e.g. gemma rejects non-bucket max_soft_tokens).
NB: any site caching vision features must key on this mapping too --
feature shapes differ per budget.
"""

from __future__ import annotations

# Mirror of transformers' gemma-4 _SUPPORTED_SOFT_TOKENS, used only if the
# private import moves; the live value is preferred so a future bucket set
# is picked up automatically.
_GEMMA_FALLBACK_BUCKETS = (70, 140, 280, 560, 1120)


def _gemma_buckets() -> tuple[int, ...]:
    try:
        from transformers.models.gemma4.image_processing_pil_gemma4 import (
            _SUPPORTED_SOFT_TOKENS,
        )
        return tuple(_SUPPORTED_SOFT_TOKENS)
    except Exception:
        return _GEMMA_FALLBACK_BUCKETS


def vision_budget_kwargs(processor, vision_tokens: int | None) -> dict:
    """Per-call processor kwargs realizing ``vision_tokens``, or {}."""
    if not vision_tokens:
        return {}
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return {}

    if hasattr(image_processor, "max_soft_tokens"):
        buckets = _gemma_buckets()
        snapped = min(buckets, key=lambda b: (abs(b - vision_tokens), b))
        return {"max_soft_tokens": snapped}

    patch = getattr(image_processor, "patch_size", None)
    merge = getattr(image_processor, "merge_size", None)
    if patch and merge:
        return {"max_pixels": int(vision_tokens) * (patch * merge) ** 2}

    return {}
