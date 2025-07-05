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
from .common.enhanced_vlm_generation import enhanced_vlm_stream_generate, create_enhanced_vlm_generator
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
        # Use object.__setattr__ to avoid triggering __getattr__ during initialization
        object.__setattr__(self, 'language_model', language_model)
        object.__setattr__(self, '_cached_layers', None)
        object.__setattr__(self, '_cached_config', None)
        object.__setattr__(self, '_cached_head_dim', None)
        object.__setattr__(self, '_cache_populated', False)
        
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
        # Only forward if language_model exists to avoid recursion
        if 'language_model' in self.__dict__:
            return getattr(self.language_model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


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
    """Strategy for VLM text-only requests (using mlx-lm path with speculative decoding)."""
    
    def __init__(self, draft_model=None):
        self._cached_wrapper = None
        self._cached_generator = None
        self.draft_model = draft_model
    
    @time_mlx_operation("generation", "vlm_text")
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        
        # Check if we can use speculative decoding
        if self.draft_model is not None:
            # Use speculative decoding with enhanced sampling
            yield from self._generate_with_speculative_decoding(
                request, effective_request, model, processor, sampler, processors
            )
        else:
            # Use standard enhanced text-only generation
            yield from self._generate_standard_enhanced(
                request, effective_request, model, processor, sampler, processors
            )
    
    def _generate_with_speculative_decoding(self, request, effective_request, model, processor, sampler, processors):
        """Generate with speculative decoding support."""
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        
        # Cache the wrapper model to avoid recreation
        if self._cached_wrapper is None:
            self._cached_wrapper = OptimizedLanguageModelWrapper(model.language_model)
        
        # Prepare VLM inputs but extract only the text prompt
        images, formatted_prompt, _ = self._prepare_vlm_inputs(request.messages, processor, model.config)
        
        # Use mlx-lm with speculative decoding
        yield from lm_stream_generate(
            model=self._cached_wrapper,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            sampler=sampler,
            logits_processors=processors,
            max_tokens=effective_request['max_tokens'],
            draft_model=self.draft_model
        )
    
    def _generate_standard_enhanced(self, request, effective_request, model, processor, sampler, processors):
        """Generate with standard enhanced sampling (no speculative decoding)."""
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        
        # Cache the wrapper model to avoid recreation
        if self._cached_wrapper is None:
            self._cached_wrapper = OptimizedLanguageModelWrapper(model.language_model)
        
        # Prepare VLM inputs but extract only the text prompt
        images, formatted_prompt, _ = self._prepare_vlm_inputs(request.messages, processor, model.config)
        
        # Use mlx-lm with advanced sampling
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
    """Strategy for VLM requests with images - Enhanced with mlx-lm sampling."""
    
    def __init__(self):
        self._cached_generator = None
    
    @time_mlx_operation("generation", "vlm_vision")
    def generate(self, request: ChatRequest, effective_request: dict, model, processor, sampler, processors) -> Generator:
        # Create enhanced generator (cached)
        if self._cached_generator is None:
            self._cached_generator = create_enhanced_vlm_generator(model, processor)
        
        # Prepare VLM inputs
        images, formatted_prompt, _ = self._prepare_vlm_inputs(request.messages, processor, model.config)
        
        # Use enhanced generation with mlx-lm quality sampling
        yield from self._cached_generator.stream_generate_enhanced(
            prompt=formatted_prompt,
            image=images,
            sampler=sampler,
            processors=processors,
            max_tokens=effective_request['max_tokens'],
            temperature=effective_request.get('temperature', 0.1),
            top_p=effective_request.get('top_p', 1.0)
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
        
        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path}")
        
        try:
            if self.is_vlm:
                # Load VLM model with fallback strategies for common issues
                logging.info("Loading VLM model using resilient MLX VLM loading")
                self.model, self.processor = self._load_vlm_with_fallback(model_path)
            else:
                # Load text-only model with MLX LM
                logging.info("Loading text-only model using MLX LM")
                self.model, self.processor = lm_load(model_path)
            
            logging.info(f"✅ Successfully loaded {'VLM' if self.is_vlm else 'LLM'} model")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise e
        
        # Load draft model if specified
        if draft_path := self.config.get('draft_model_path'):
            logging.info(f"Loading draft model for speculative decoding: {draft_path}")
            try:
                # Draft models are always text-only
                self.draft_model, _ = lm_load(draft_path)
                logging.info("✅ Draft model loaded successfully for speculative decoding")
            except Exception as e:
                logging.warning(f"Failed to load draft model: {e}")
                self.draft_model = None
        
        # Pre-compile generation strategies after model loading
        self._compile_strategies()
    
    def _load_vlm_with_fallback(self, model_path):
        """Load VLM model with fallback strategies for common issues."""
        
        # Strategy 1: Try with skip_audio=True (most common fix)
        try:
            logging.debug("Attempting VLM load with skip_audio=True")
            return vlm_load(model_path, skip_audio=True)
        except Exception as e:
            logging.debug(f"VLM load with skip_audio failed: {e}")
        
        # Strategy 2: Try standard loading 
        try:
            logging.debug("Attempting standard VLM load")
            return vlm_load(model_path)
        except Exception as e:
            logging.debug(f"Standard VLM load failed: {e}")
            
            # Strategy 3: Handle specific weight mismatch errors
            if "language_model.lm_head.weight" in str(e):
                logging.info("Detected language_model.lm_head.weight error, applying model-specific fix")
                return self._load_vlm_with_weight_fix(model_path)
            else:
                raise e
    
    def _load_vlm_with_weight_fix(self, model_path):
        """Handle specific weight mismatch issues."""
        import mlx_vlm.utils
        import importlib
        
        try:
            # Strategy 3a: Try loading with strict=False directly in vlm_load
            logging.debug("Attempting VLM load with strict=False")
            
            # Try to call vlm_load with strict=False if it supports it
            try:
                return vlm_load(model_path, strict=False)
            except TypeError:
                # vlm_load doesn't support strict parameter
                pass
            
            # Strategy 3b: Try patching the load_model function
            logging.debug("Attempting VLM load with patched load_model")
            
            original_load_model = mlx_vlm.utils.load_model
            
            def patched_load_model(model_path, **kwargs):
                try:
                    return original_load_model(model_path, **kwargs)
                except Exception as e:
                    if "language_model.lm_head.weight" in str(e):
                        logging.debug("Applying weight mismatch fix by setting strict=False")
                        # Try loading without strict weight matching
                        kwargs['strict'] = False
                        return original_load_model(model_path, **kwargs)
                    else:
                        raise e
            
            # Apply the patch
            mlx_vlm.utils.load_model = patched_load_model
            
            try:
                result = vlm_load(model_path)
                return result
            finally:
                # Restore original function
                mlx_vlm.utils.load_model = original_load_model
                
        except Exception as e:
            logging.debug(f"Weight fix strategy failed: {e}")
            
            # Strategy 3c: Try loading model components separately
            try:
                logging.debug("Attempting alternative model loading approach")
                return self._alternative_vlm_load(model_path)
            except Exception as e2:
                logging.error(f"All VLM loading strategies failed. Last error: {e2}")
                raise e2
    
    def _alternative_vlm_load(self, model_path):
        """Alternative loading approach for problematic models."""
        from pathlib import Path
        import json
        
        try:
            # Check if this is a model conversion issue
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                
                model_type = config.get('model_type', 'unknown')
                logging.debug(f"Detected model type: {model_type}")
                
                # For gemma models, this might be a version compatibility issue
                if 'gemma' in model_type.lower():
                    logging.info(f"Gemma model detected. This error suggests the model weights ")
                    logging.info(f"don't match the expected MLX VLM structure for {model_type}.")
                    logging.info(f"This could be due to:")
                    logging.info(f"1. Model was converted with an incompatible MLX VLM version")
                    logging.info(f"2. Model weights are from a different architecture variant")
                    logging.info(f"3. Model files are corrupted or incomplete")
                    
                    # Try one last approach with minimal loading
                    logging.debug("Attempting minimal VLM loading")
                    
                    # Import here to avoid circular imports
                    from mlx_vlm import load as vlm_load_function
                    
                    # Try loading with all optional parameters disabled
                    try:
                        return vlm_load_function(model_path, trust_remote_code=False)
                    except Exception as e:
                        logging.debug(f"Minimal loading failed: {e}")
                        
                        # Final attempt: try loading as a different model type
                        logging.debug("Final attempt: trying to load with model type override")
                        
                        # This is a last-ditch effort - we can't actually fix the weight mismatch
                        # but we can provide a clear error message
                        raise Exception(
                            f"Model '{model_path}' has incompatible weights. "
                            f"The model appears to be missing 'language_model.lm_head.weight' or has "
                            f"a weight shape mismatch. This typically indicates: \n"
                            f"1. The model was converted with an incompatible MLX VLM version\n"
                            f"2. The model files are corrupted or incomplete\n"
                            f"3. This specific model variant is not supported by the current MLX VLM version\n\n"
                            f"Suggested fixes:\n"
                            f"- Try downloading a different variant of the model\n"
                            f"- Re-convert the model with the current MLX VLM version\n"
                            f"- Use a different model that's known to work (e.g., qwen2.5-vl-72b-inst-gguf)\n"
                        )
            
            # If we can't determine the model type, give a generic error
            raise Exception(
                f"Failed to load VLM model at '{model_path}'. "
                f"All loading strategies failed with weight mismatch errors."
            )
            
        except Exception as e:
            logging.error(f"Alternative VLM loading failed: {e}")
            raise e
    
    def _compile_strategies(self):
        """Pre-compile generation strategies to avoid runtime branching."""
        if self.is_vlm:
            # Pass draft model to VLM text strategy for speculative decoding
            self._strategies['vlm_text'] = VLMTextOnlyStrategy(draft_model=self.draft_model)
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
