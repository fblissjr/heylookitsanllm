# src/heylook_llm/providers/mlx_provider_optimized.py
import gc
import logging
import traceback
import json
from typing import Generator, Dict, List, Tuple, Protocol
from abc import ABC, abstractmethod

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from mlx_lm.utils import load as lm_load
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.generate import stream_generate as vlm_stream_generate
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template

from ..config import ChatRequest
from .base import BaseProvider
from .common.samplers import build as build_sampler
from .common.performance_monitor import time_mlx_operation, performance_monitor
from ..utils import load_image


class OptimizedLanguageModelWrapper(nn.Module):
    """
    Optimized wrapper for VLM language models to use with mlx-lm.
    
    Why this exists:
    - Caches frequently accessed attributes for better performance
    - Provides direct logits extraction without object creation overhead
    - Maintains mlx-lm compatibility for text-only VLM requests
    """
    
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        
        # Cache frequently accessed attributes to avoid repeated lookups
        self._cached_layers = None
        self._cached_config = None
        self._cached_head_dim = None
        self._cache_populated = False
        
    def _populate_cache(self):
        """Populate attribute cache on first access."""
        if not self._cache_populated:
            # Cache layers - check both common attribute locations
            if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'layers'):
                self._cached_layers = self.language_model.model.layers
            elif hasattr(self.language_model, 'layers'):
                self._cached_layers = self.language_model.layers
            
            # Cache config
            if hasattr(self.language_model, 'config'):
                self._cached_config = self.language_model.config
            elif hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'config'):
                self._cached_config = self.language_model.model.config
            
            # Cache head dimension if available
            if self._cached_config and hasattr(self._cached_config, 'head_dim'):
                self._cached_head_dim = self._cached_config.head_dim
            elif self._cached_config and hasattr(self._cached_config, 'hidden_size'):
                self._cached_head_dim = self._cached_config.hidden_size
                
            self._cache_populated = True
    
    def __call__(self, *args, **kwargs):
        """Direct logits extraction - the core optimization."""
        # Direct logits extraction avoids creating intermediate objects
        result = self.language_model(*args, **kwargs)
        return result.logits if hasattr(result, 'logits') else result
    
    @property
    def layers(self):
        """Cached layers property."""
        if self._cached_layers is None:
            self._populate_cache()
        return self._cached_layers
    
    @property
    def config(self):
        """Cached config property."""
        if self._cached_config is None:
            self._populate_cache()
        return self._cached_config
    
    @property
    def head_dim(self):
        """Cached head dimension property."""
        if self._cached_head_dim is None:
            self._populate_cache()
        return self._cached_head_dim
    
    def __getattr__(self, name):
        """Fast forwarding for any other attributes."""
        # Forward all other attribute access to the wrapped model
        return getattr(self.language_model, name)


class GenerationStrategy(Protocol):
    """Protocol for generation strategies."""
    
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        """Generate response using this strategy."""
        ...


class TextOnlyStrategy:
    """Strategy for text-only LLM requests."""
    
    def __init__(self, draft_model=None):
        self.draft_model = draft_model
    
    @time_mlx_operation("generation", "text_only")
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        
        # Apply chat template once
        prompt = tokenizer.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in request.messages], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        yield from lm_stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            sampler=sampler,
            logits_processors=processors,
            max_tokens=effective_request['max_tokens'],
            draft_model=self.draft_model
        )


class VLMTextOnlyStrategy:
    """Strategy for VLM text-only requests (using mlx-lm path)."""
    
    def __init__(self):
        self._cached_wrapper = None
    
    @time_mlx_operation("generation", "vlm_text")
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        
        # Cache the wrapper model to avoid recreation
        if self._cached_wrapper is None:
            self._cached_wrapper = OptimizedLanguageModelWrapper(model.language_model)
        
        # Prepare VLM inputs but extract only the text prompt
        images, formatted_prompt, _ = self._prepare_vlm_inputs(request.messages, processor, model.config)
        
        yield from lm_stream_generate(
            model=self._cached_wrapper,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            sampler=sampler,
            logits_processors=processors,
            max_tokens=effective_request['max_tokens']
        )
    
    def _prepare_vlm_inputs(self, messages: List, processor, config) -> Tuple[List[Image.Image], str, bool]:
        """Prepare VLM inputs - extracted for reuse."""
        images, text_messages, has_images = [], [], False
        
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        images.append(load_image(part.image_url.url))
                        has_images = True
                text_messages.append({"role": msg.role, "content": "".join(text_parts)})
            elif isinstance(content, str):
                text_messages.append({"role": msg.role, "content": content})

        formatted_prompt = vlm_apply_chat_template(
            processor, config, text_messages, num_images=len(images)
        )
        return images, formatted_prompt, has_images


class VLMVisionStrategy:
    """Strategy for VLM requests with images."""
    
    @time_mlx_operation("generation", "vlm_vision")
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        # Prepare VLM inputs
        images, formatted_prompt, _ = self._prepare_vlm_inputs(request.messages, processor, model.config)
        
        yield from vlm_stream_generate(
            model=model,
            processor=processor,
            prompt=formatted_prompt,
            image=images,
            temperature=effective_request['temperature'],
            max_tokens=effective_request['max_tokens'],
            top_p=effective_request['top_p']
        )
    
    def _prepare_vlm_inputs(self, messages: List, processor, config) -> Tuple[List[Image.Image], str, bool]:
        """Prepare VLM inputs - extracted for reuse."""
        images, text_messages, has_images = [], [], False
        
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        images.append(load_image(part.image_url.url))
                        has_images = True
                text_messages.append({"role": msg.role, "content": "".join(text_parts)})
            elif isinstance(content, str):
                text_messages.append({"role": msg.role, "content": content})

        formatted_prompt = vlm_apply_chat_template(
            processor, config, text_messages, num_images=len(images)
        )
        return images, formatted_prompt, has_images


class MLXProvider(BaseProvider):
    """
    Optimized MLX Provider with dual-path architecture.
    
    Key optimizations:
    1. Pre-compiled path decision logic using strategy pattern
    2. Cached generation strategies to avoid object creation
    3. Optimized LanguageModelWrapper with attribute caching
    4. Single-pass content scanning for path decisions
    """
    
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None
        self.processor = None
        self.draft_model = None
        self.is_vlm = self.config.get("vision", False)
        
        # Pre-compile generation strategies (avoids runtime branching)
        self._strategies = {}
        self._content_cache = {}  # Cache for image detection results
        
    @time_mlx_operation("model_loading")
    def load_model(self):
        model_path = self.config['model_path']
        load_fn = vlm_load if self.is_vlm else lm_load
        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path} using {'mlx_vlm' if self.is_vlm else 'mlx_lm'} loader.")
        
        try:
            self.model, self.processor = load_fn(model_path)
        except Exception as e:
            raise e
        
        # Load draft model if specified
        if draft_path := self.config.get('draft_model_path'):
            if self.is_vlm:
                logging.warning("Speculative decoding is not currently supported for VLM models.")
            else:
                self.draft_model, _ = lm_load(draft_path)
        
        # Pre-compile generation strategies after model loading
        self._compile_strategies()
    
    def _compile_strategies(self):
        """Pre-compile generation strategies to avoid runtime branching."""
        if self.is_vlm:
            self._strategies['vlm_text'] = VLMTextOnlyStrategy()
            self._strategies['vlm_vision'] = VLMVisionStrategy()
        else:
            self._strategies['text_only'] = TextOnlyStrategy(self.draft_model)
    
    @time_mlx_operation("path_decision")
    def _detect_images_optimized(self, messages: List) -> bool:
        """
        Optimized image detection with single-pass scanning.
        
        Why this optimization matters:
        - Avoids multiple passes over message content
        - Caches results for repeated requests
        - Early termination on first image found
        """
        # Create a simple cache key based on message structure
        cache_key = id(messages)  # Use object id as a simple cache key
        
        if cache_key in self._content_cache:
            return self._content_cache[cache_key]
        
        # Single-pass scan with early termination
        has_images = False
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == 'image_url':
                        has_images = True
                        break
                if has_images:
                    break
        
        # Cache the result
        self._content_cache[cache_key] = has_images
        return has_images
    
    def _apply_model_defaults(self, request: ChatRequest) -> dict:
        """Apply model defaults with minimal object creation."""
        global_defaults = {
            'temperature': 0.1, 
            'top_p': 1.0, 
            'top_k': 0, 
            'min_p': 0.0, 
            'max_tokens': 512, 
            'repetition_penalty': 1.0
        }
        
        # Efficient config merging
        merged_config = global_defaults.copy()
        merged_config.update({k: v for k, v in self.config.items() if v is not None})
        merged_config.update({k: v for k, v in request.model_dump().items() if v is not None})
        
        return merged_config
    
    @time_mlx_operation("chat_completion")
    def create_chat_completion(self, request: ChatRequest) -> Generator:
        """
        Optimized chat completion with strategy pattern.
        
        Path decision logic is pre-compiled and cached to minimize runtime overhead.
        """
        effective_request = self._apply_model_defaults(request)
        
        if self.verbose:
            logging.debug(f"MLX effective request params: {json.dumps(effective_request, indent=2)}")

        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        sampler, processors = build_sampler(tokenizer, effective_request)

        try:
            if not self.is_vlm:
                # Text-only model path
                if self._detect_images_optimized(request.messages):
                    class MLXErrorChunk:
                        def __init__(self, text):
                            self.text = text
                    
                    yield MLXErrorChunk(text=f"Error: Model '{self.model_id}' is text-only and cannot process images. Please use a vision model like 'gemma3n-e4b-it' for image inputs.")
                    return
                
                # Use pre-compiled text-only strategy
                strategy = self._strategies['text_only']
                yield from strategy.generate(request, effective_request, self.model, self.processor, sampler, processors)
                return
            
            # VLM model path - decide between text-only and vision
            has_images = self._detect_images_optimized(request.messages)
            
            if has_images:
                # Use pre-compiled vision strategy
                strategy = self._strategies['vlm_vision']
            else:
                # Use pre-compiled text-only VLM strategy (faster mlx-lm path)
                strategy = self._strategies['vlm_text']
            
            yield from strategy.generate(request, effective_request, self.model, self.processor, sampler, processors)
            
        except Exception as e:
            logging.error(f"MLX model call failed: {e}", exc_info=True)
            
            class MLXErrorChunk:
                def __init__(self, text):
                    self.text = text
            
            yield MLXErrorChunk(text=f"Error: MLX generation failed: {str(e)}")
    
    def log_performance_summary(self):
        """Log current performance metrics."""
        try:
            summary = performance_monitor.get_performance_summary()
            logging.info(f"MLX Provider Performance Summary for {self.model_id}:\n{summary}")
            
            # Log path comparisons if available
            path_comparison = performance_monitor.compare_paths("generation")
            if path_comparison:
                logging.info(f"Generation Path Performance Comparison: {path_comparison}")
        except Exception as e:
            logging.debug(f"Failed to log performance summary: {e}")
    
    def unload(self):
        """Enhanced cleanup with cache clearing and performance logging."""
        logging.info(f"Unloading MLX model: {self.model_id}")
        
        # Log performance summary before cleanup
        self.log_performance_summary()
        
        # Clear caches
        self._content_cache.clear()
        self._strategies.clear()
        
        # Clean up models
        if hasattr(self, 'model'): 
            del self.model
        if hasattr(self, 'processor'): 
            del self.processor
        if hasattr(self, 'draft_model'): 
            del self.draft_model
        
        gc.collect()
        mx.clear_cache()
