# src/heylook_llm/providers/vlm_vision_strategy_optimized.py
"""
Optimized VLM Vision Strategy with parallel image processing.

This replaces the VLMVisionStrategy class with proper batch processing support.
"""

from typing import List, Tuple, Generator, Dict, Any
from PIL import Image
import mlx.core as mx

from ..config import ChatRequest
from .common.batch_vision import BatchVisionProcessor
from .common.performance_monitor import time_mlx_operation
from .common.vlm_generation import create_vlm_generator_with_sampling
from .mlx_batch_vision import BatchVisionEncoder
from ..utils import load_image
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template


class VLMVisionStrategyOptimized:
    """Optimized strategy for VLM requests with parallel image processing."""

    def __init__(self):
        self._cached_generator = None
        self._batch_encoder = None
        self._batch_vision_processor = BatchVisionProcessor(max_workers=4)

    @time_mlx_operation("generation", "vlm_vision_optimized")
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        # Create generator with sampling (cached)
        if self._cached_generator is None:
            self._cached_generator = create_vlm_generator_with_sampling(model, processor)

        # Initialize batch encoder if needed
        if self._batch_encoder is None:
            self._batch_encoder = BatchVisionEncoder(model, processor)

        # Prepare VLM inputs with parallel image loading
        images, formatted_prompt, _ = self._prepare_vlm_inputs_parallel(
            request.messages, processor, model.config, model
        )
        
        # Use batch encoding if multiple images
        if len(images) > 1:
            # For now, we'll use the standard generation path
            # In the future, we can modify the generator to accept pre-encoded features
            # vision_features = self._batch_encoder.encode_batch(images)
            pass

        # Use generation with mlx-lm sampling
        yield from self._cached_generator.stream_generate_with_sampling(
            prompt=formatted_prompt,
            image=images,
            sampler=sampler,
            processors=processors,
            max_tokens=effective_request['max_tokens'],
            temperature=effective_request.get('temperature', 0.1),
            top_p=effective_request.get('top_p', 1.0)
        )

    def _prepare_vlm_inputs_parallel(self, messages: List, processor, config, model=None) -> Tuple[List[Image.Image], str, bool]:
        """Prepare VLM inputs with parallel image loading."""
        image_urls = []
        text_messages = []
        has_images = False
        image_counter = 0

        # First pass: collect image URLs and build text structure
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                
                for part in content:
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        image_urls.append(part.image_url.url)
                        has_images = True
                        image_counter += 1
                        
                        # Add image placeholder to maintain position
                        model_type = str(type(model)).lower() if model else ""
                        if 'gemma' not in model_type:
                            if processor.tokenizer and hasattr(processor.tokenizer, 'image_token'):
                                text_parts.append(processor.tokenizer.image_token)
                            else:
                                text_parts.append(f"[Image {image_counter}]")
                
                # Combine text parts
                combined_content = " ".join(text_parts) if text_parts else ""
                text_messages.append({"role": msg.role, "content": combined_content})
            elif isinstance(content, str):
                text_messages.append({"role": msg.role, "content": content})

        # Load all images in parallel
        if image_urls:
            images = self._batch_vision_processor.load_images_parallel(image_urls)
        else:
            images = []

        # Format prompt
        formatted_prompt = vlm_apply_chat_template(
            processor, config, text_messages, num_images=len(images)
        )
        
        return images, formatted_prompt, has_images


# Export the optimized class
VLMVisionStrategy = VLMVisionStrategyOptimized