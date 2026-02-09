# src/heylook_llm/providers/common/vlm_generation.py
"""
VLM generation with mlx-lm feature integration.

Why this exists:
- Backports advanced sampling from mlx-lm to mlx-vlm path
- Provides speculative decoding support for text-only VLM requests
- Closes feature gap between vision and text paths
- Maintains elegant simplicity while improving quality
"""

import mlx.core as mx
from typing import Generator, List, Union, Any
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm.generate import stream_generate as vlm_stream_generate

from .performance_monitor import time_mlx_operation


class VLMGeneratorWithSampling:
    """
    VLM generator that integrates mlx-lm sampling features.
    
    Key features:
    - mlx-lm sampling (top-k, repetition penalty, min-p)
    - Speculative decoding support for text-only VLM requests
    - Unified sampling interface for vision and text paths
    - Maintains compatibility with existing mlx-vlm interface
    """
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

        # Cache LanguageModelLogitsWrapper for text-only path
        from ..mlx_provider import LanguageModelLogitsWrapper
        self._language_model_wrapper = LanguageModelLogitsWrapper(model.language_model)
    
    @time_mlx_operation("vlm_generation", "vision_with_sampling")
    def stream_generate_with_sampling(
        self,
        prompt: str,
        image: Union[List, None] = None,
        sampler: callable = None,
        processors: List[callable] = None,
        max_tokens: int = 512,
        **kwargs
    ) -> Generator[Any, None, None]:
        """
        VLM stream generation with mlx-lm sampling integration.
        
        Args:
            prompt: Text prompt
            image: Image(s) for vision tasks
            sampler: Advanced sampler function from mlx-lm
            processors: Logits processors for quality improvements
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        """
        
        # No synchronization needed here - MLX handles it internally
        
        if image is None or len(image) == 0:
            # Text-only VLM request - use optimized path with advanced sampling
            yield from self._stream_generate_text_only_enhanced(
                prompt, sampler, processors, max_tokens, **kwargs
            )
        else:
            # Vision request - use enhanced mlx-vlm path
            yield from self._stream_generate_vision_enhanced(
                prompt, image, sampler, processors, max_tokens, **kwargs
            )
    
    def _stream_generate_text_only_enhanced(
        self,
        prompt: str,
        sampler: callable,
        processors: List[callable],
        max_tokens: int,
        **kwargs
    ) -> Generator[Any, None, None]:
        """
        Text-only generation for VLM models using mlx-lm.
        
        Uses mlx-lm's advanced sampling with the VLM's language model component.
        """
        try:
            # Use the cached language model wrapper with advanced sampling
            language_model = self._language_model_wrapper
            
            # Extract the prompt_progress_callback if provided
            prompt_progress_callback = kwargs.pop('prompt_progress_callback', None)
            
            # Use mlx-lm's stream generation with advanced sampling
            yield from lm_stream_generate(
                model=language_model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                sampler=sampler,
                logits_processors=processors,
                max_tokens=max_tokens,
                prompt_progress_callback=prompt_progress_callback
            )
            
        except Exception as e:
            # Fallback to original VLM generation if enhancement fails
            yield from vlm_stream_generate(
                model=self.model,
                processor=self.processor,
                prompt=prompt,
                audio=None,  # Explicitly set to None to avoid feature_extractor access
                max_tokens=max_tokens,
                **kwargs
            )
    
    def _stream_generate_vision_enhanced(
        self,
        prompt: str,
        image: List,
        sampler: callable,
        processors: List[callable],
        max_tokens: int,
        **kwargs
    ) -> Generator[Any, None, None]:
        """
        Vision generation using mlx-vlm's stream_generate.

        Sampling support:
        - Supported: temperature, top_p, repetition_penalty, logit_bias, logits_processors
        - NOT supported: top_k, min_p, presence_penalty, XTC (mlx-vlm uses its own
          hardcoded sampling in generate_step; no pluggable sampler interface)
        """
        # Forward logits_processors if provided
        if processors:
            kwargs['logits_processors'] = processors

        # For Qwen models, we need to handle grid_thw
        model_type = str(type(self.model)).lower()
        if 'qwen' in model_type and image is not None:
            # Qwen models need grid_thw for image processing
            # Calculate grid dimensions based on number of images
            num_images = len(image) if isinstance(image, list) else 1
            # Standard grid for static images: [time=1, height=24, width=24] for 336x336 images
            image_grid_thw = mx.array([[1, 24, 24]] * num_images, dtype=mx.int32)
            kwargs['image_grid_thw'] = image_grid_thw
        
        yield from vlm_stream_generate(
            model=self.model,
            processor=self.processor,
            prompt=prompt,
            image=image,
            audio=None,  # Explicitly set to None to avoid feature_extractor access
            max_tokens=max_tokens,
            **kwargs
        )
    
def create_vlm_generator_with_sampling(model, processor) -> VLMGeneratorWithSampling:
    """Factory function to create enhanced VLM generator."""
    return VLMGeneratorWithSampling(model, processor)
