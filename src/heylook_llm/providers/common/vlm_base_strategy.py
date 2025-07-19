# src/heylook_llm/providers/common/vlm_base_strategy.py
"""
Base VLM strategy class to reduce code duplication.

This provides common functionality for all VLM strategies.
"""

from typing import List, Tuple, Dict, Any, Generator
from abc import ABC, abstractmethod
from PIL import Image
import mlx.core as mx

from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
from ...config import ChatRequest
from ...utils import load_image


class VLMBaseStrategy(ABC):
    """Base class for VLM strategies with common functionality."""
    
    def __init__(self):
        self._cached_generator = None
        self._cached_wrapper = None
    
    @abstractmethod
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        """Generate response - must be implemented by subclasses."""
        pass
    
    def _prepare_vlm_inputs(self, messages: List, processor, config, model=None) -> Tuple[List[Image.Image], str, bool]:
        """Prepare VLM inputs - common implementation."""
        images, text_messages, has_images = [], [], False
        image_counter = 0

        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                message_has_images = False
                
                for part in content:
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        images.append(load_image(part.image_url.url))
                        has_images = True
                        message_has_images = True
                        image_counter += 1
                        # Add image placeholder to maintain position
                        # Note: Some models (like Gemma) handle image tokens automatically
                        # in their chat template, so we shouldn't add them manually
                        model_type = str(type(model)).lower() if model else ""
                        if 'gemma' not in model_type:
                            if processor.tokenizer and hasattr(processor.tokenizer, 'image_token'):
                                text_parts.append(processor.tokenizer.image_token)
                            else:
                                # Fallback for models without explicit image tokens
                                text_parts.append(f"[Image {image_counter}]")
                
                # Combine text parts, preserving image positions
                combined_content = " ".join(text_parts) if text_parts else ""
                text_messages.append({"role": msg.role, "content": combined_content})
            elif isinstance(content, str):
                text_messages.append({"role": msg.role, "content": content})

        formatted_prompt = vlm_apply_chat_template(
            processor, config, text_messages, num_images=len(images)
        )
        return images, formatted_prompt, has_images
    
    def _get_tokenizer(self, processor):
        """Extract tokenizer from processor."""
        return processor.tokenizer if hasattr(processor, "tokenizer") else processor
    
    def _create_language_wrapper_if_needed(self, model):
        """Create and cache language model wrapper."""
        if self._cached_wrapper is None:
            from ..mlx_provider import LanguageModelLogitsWrapper
            self._cached_wrapper = LanguageModelLogitsWrapper(model.language_model)
        return self._cached_wrapper