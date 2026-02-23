# src/heylook_llm/providers/common/vlm_generation.py
"""
VLM vision generation helper.

Thin wrapper around mlx_vlm.generate.stream_generate that handles
Qwen-specific image_grid_thw and forwards logits_processors.
"""

import mlx.core as mx
from typing import Generator, List, Any
from mlx_vlm.generate import stream_generate as vlm_stream_generate


def stream_generate_vlm_vision(
    model,
    processor,
    prompt: str,
    image: List,
    sampler: callable = None,
    processors: List[callable] = None,
    max_tokens: int = 512,
    **kwargs,
) -> Generator[Any, None, None]:
    """Stream generation for VLM vision requests.

    Args:
        model: VLM model instance
        processor: VLM processor
        prompt: Formatted text prompt
        image: List of PIL images
        sampler: Unused (mlx-vlm has its own sampling); kept for call-site compat
        processors: Logits processors forwarded to vlm_stream_generate
        max_tokens: Maximum tokens to generate
        **kwargs: Forwarded to vlm_stream_generate (temperature, top_p, etc.)
    """
    if processors:
        kwargs['logits_processors'] = processors

    # Qwen models need grid_thw for image processing
    model_type = str(type(model)).lower()
    if 'qwen' in model_type and image is not None:
        num_images = len(image) if isinstance(image, list) else 1
        image_grid_thw = mx.array([[1, 24, 24]] * num_images, dtype=mx.int32)
        kwargs['image_grid_thw'] = image_grid_thw

    yield from vlm_stream_generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        audio=None,
        max_tokens=max_tokens,
        **kwargs,
    )
