# src/heylook_llm/providers/mlx_provider.py
import gc
import logging
import json
import threading
import time
from typing import Generator, Dict, List, Tuple

import mlx.core as mx
from PIL import Image

from mlx_lm.utils import load as lm_load
from mlx_lm.generate import wired_limit
from mlx_lm.models.cache import make_prompt_cache
from mlx_vlm.utils import load as vlm_load, prepare_inputs as vlm_prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template as mlx_vlm_apply_chat_template

from ..config import ChatRequest, ModelMetrics
from .abort import AbortEvent
from .base import BaseProvider
from .common.samplers import build as build_sampler
from .common.vlm_inputs import _reconstruct_thinking
from .common.model_wrappers import LanguageModelLogitsWrapper
from .common.generation_core import generate_text, run_generation
from .common.batch_vision import BatchVisionProcessor
from .common.prompt_cache import get_global_cache_manager

# Create dedicated generation stream for better Metal utilization
# This allows async evaluation and improves pipeline performance
generation_stream = mx.new_stream(mx.default_device())


def vlm_apply_chat_template(processor, config, messages, num_images=None):
    """
    Apply chat template using mlx-vlm's prompt_utils.

    This uses the library's apply_chat_template which properly formats messages
    with image tokens based on num_images and model_type.

    Args:
        processor: The model processor (contains tokenizer)
        config: Model config (contains model_type for proper formatting)
        messages: List of message dicts with 'role' and 'content'
        num_images: Number of images to add tokens for

    Returns:
        str: Formatted prompt string with proper image tokens
    """
    return mlx_vlm_apply_chat_template(
        processor,
        config,
        messages,
        num_images=num_images or 0
    )


class UnifiedTextStrategy:
    """Unified strategy for all text-based generation (text-only and VLM text).

    Dispatches on is_vlm for:
    - Chat template application (tokenizer.apply_chat_template vs vlm_apply_chat_template)
    - Model selection (raw model vs LanguageModelLogitsWrapper)

    Everything else -- cache config, prompt cache lookup, generation loop,
    acceptance tracking, KV snapshot storage -- is handled by generation_core.
    """

    def __init__(self, draft_model=None, model_id=None, model_config=None, is_vlm=False):
        self.draft_model = draft_model
        self.model_id = model_id
        self.model_config = model_config or {}
        self.is_vlm = is_vlm
        self._cached_wrapper = None
        self.cache_manager = get_global_cache_manager()

    def generate(self, request: ChatRequest, effective_request: dict, model, processor, abort_event: AbortEvent | None = None) -> Generator:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        messages_for_template = self._prepare_messages(request.messages)
        prompt_tokens = self._apply_template(messages_for_template, tokenizer, processor, model, effective_request)
        gen_model = self._get_generation_model(model)

        yield from generate_text(
            model=gen_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            effective_request=effective_request,
            model_id=self.model_id,
            draft_model=self.draft_model,
            cache_manager=self.cache_manager,
            abort_event=abort_event,
        )

    def _prepare_messages(self, messages) -> list[dict]:
        """Prepare messages for template application. Shared for both paths."""
        messages_for_template = []
        for msg in messages:
            msg_dict = msg.model_dump(exclude_none=True)
            if isinstance(msg_dict.get('content'), list):
                text_parts = [part['text'] for part in msg_dict['content'] if part.get('type') == 'text']
                msg_dict['content'] = ' '.join(text_parts)
            msg_dict = _reconstruct_thinking(msg_dict)
            messages_for_template.append(msg_dict)
        return messages_for_template

    def _apply_template(self, messages, tokenizer, processor, model, effective_request) -> list[int]:
        """Dispatch template application based on is_vlm.

        Text-only: tokenizer.apply_chat_template with enable_thinking support.
        VLM text: vlm_apply_chat_template (uses processor + model config).
        """
        if self.is_vlm:
            prompt = vlm_apply_chat_template(
                processor, model.config, messages, num_images=0
            )
        else:
            last_is_assistant = messages[-1].get('role') == 'assistant' if messages else False
            add_gen_prompt = not last_is_assistant

            enable_thinking = effective_request.get("enable_thinking")
            if enable_thinking is None:
                enable_thinking = self.model_config.get('enable_thinking', True)

            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=add_gen_prompt,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=add_gen_prompt,
                )

        if isinstance(prompt, str):
            return tokenizer.encode(prompt)
        return prompt

    def _get_generation_model(self, model):
        """Return raw model for text-only, LanguageModelLogitsWrapper for VLM.

        The wrapper is cached per strategy instance so it's created once and reused.
        For VLM, wired_limit and cache operations use the wrapper (which wraps
        the language model component), not the full VLM model.
        """
        if not self.is_vlm:
            return model

        if self._cached_wrapper is None:
            self._cached_wrapper = LanguageModelLogitsWrapper(model.language_model)
        return self._cached_wrapper


class _VisionTokenResponse:
    """Lightweight response for the first token from VLM vision encoding.

    Compatible with the GenerationResponse interface that api.py expects
    (needs .text, optionally .token and .logprobs).
    """
    __slots__ = ('text', 'token', 'logprobs')

    def __init__(self, text: str, token: int, logprobs=None):
        self.text = text
        self.token = token
        self.logprobs = logprobs


class VLMVisionStrategy:
    """Strategy for VLM requests with images.

    Uses the pre-filled cache pattern (inspired by vllm-mlx):
    1. Prepare inputs via mlx_vlm.utils.prepare_inputs
    2. Create KV cache for the language model
    3. Run full VLM forward pass (vision + language), filling the cache
    4. Sample first token from logits
    5. Continue generation via generation_core.run_generation()

    This gives vision requests the full sampler suite, abort support, and
    speculative decoding from generation_core -- a single code path for all
    MLX generation.
    """

    def __init__(self):
        self._batch_vision_processor = None
        self._cached_wrapper = None

    def generate(self, request: ChatRequest, effective_request: dict, model, processor, abort_event: AbortEvent | None = None) -> Generator:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        sampler, processors = build_sampler(tokenizer, effective_request)

        # Initialize batch vision processor for parallel image loading
        if self._batch_vision_processor is None:
            self._batch_vision_processor = BatchVisionProcessor(max_workers=4)

        # Prepare VLM inputs: extract images, format prompt with chat template
        images, formatted_prompt, _ = self._prepare_vlm_inputs_parallel(
            request.messages, processor, model.config, model
        )

        num_images = len(images) if images else 0
        model_type = getattr(model.config, 'model_type', 'unknown')
        logging.info(f"[VLM VISION] Processing {num_images} image(s) | Model: {model_type}")

        # Tokenize and prepare pixel values via mlx_vlm.utils.prepare_inputs.
        # This handles image_grid_thw for Qwen models automatically.
        image_token_index = getattr(model.config, 'image_token_index', None)
        inputs = vlm_prepare_inputs(
            processor,
            images=images if images else None,
            prompts=formatted_prompt,
            image_token_index=image_token_index,
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        mask = inputs.get("attention_mask")
        # Collect model-specific extras (e.g. image_grid_thw for Qwen)
        extra_kwargs = {
            k: v for k, v in inputs.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")
        }

        # Build kwargs for VLM forward pass
        vlm_kwargs = dict(extra_kwargs)
        if pixel_values is not None:
            vlm_kwargs["pixel_values"] = pixel_values
        if mask is not None:
            vlm_kwargs["attention_mask"] = mask

        # Ensure wrapper is cached for language model generation
        if self._cached_wrapper is None:
            self._cached_wrapper = LanguageModelLogitsWrapper(model.language_model)

        # Create KV cache sized for the language model
        request_cache = make_prompt_cache(self._cached_wrapper)

        # Phase 1: Vision encoding -- run full VLM forward pass.
        # The VLM passes cache= through to model.language_model(), so the
        # language model writes KV state directly into request_cache.
        with wired_limit(model, [generation_stream]):
            if input_ids.ndim == 1:
                input_ids = input_ids[None, :]

            output = model(input_ids, cache=request_cache, **vlm_kwargs)

            # Extract logits (may be LanguageModelOutput or raw tensor)
            logits = output.logits if hasattr(output, 'logits') else output

            # Sample first token
            last_logits = logits[:, -1, :]
            first_logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
            first_token_id = sampler(first_logprobs).item()
            mx.eval(first_logprobs)

        # Check abort before yielding
        if abort_event and abort_event.is_set():
            logging.info("Generation aborted during vision encoding")
            return

        # Yield the first token
        first_text = tokenizer.decode([first_token_id])
        yield _VisionTokenResponse(
            text=first_text,
            token=first_token_id,
            logprobs=first_logprobs.squeeze(0),
        )

        # Phase 2: Continue generation using the language model wrapper
        # with the pre-filled cache from the VLM forward pass.
        yield from run_generation(
            model=self._cached_wrapper,
            tokenizer=tokenizer,
            prompt_tokens=[first_token_id],
            effective_request=effective_request,
            sampler=sampler,
            processors=processors,
            abort_event=abort_event,
            pre_filled_cache=request_cache,
        )

    def _prepare_vlm_inputs_parallel(self, messages: List, processor, config, model=None) -> Tuple[List[Image.Image], str, bool]:
        """Prepare VLM inputs with parallel image loading. Delegates to standalone function."""
        from .common.vlm_inputs import prepare_vlm_inputs_parallel
        return prepare_vlm_inputs_parallel(
            messages, processor, config, self._batch_vision_processor,
            vlm_apply_chat_template, model=model,
        )


class MLXProvider(BaseProvider):
    """
    MLX Provider with dual-path architecture for VLM and text-only generation.

    Key optimizations:
    1. Pre-compiled path decision logic using strategy pattern
    2. Cached generation strategies to avoid object creation
    3. LanguageModelLogitsWrapper for mlx-lm compatibility
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

        # Batch text processor (lazy-initialized)
        self._batch_processor = None

        # Add generation lock to prevent Metal command buffer conflicts
        self._generation_lock = threading.Lock()

        # Cooperative abort signal for cancelling in-flight generation
        self._abort_event = AbortEvent()

        # Reference counting for safe unload -- prevents eviction during active generation
        self._active_generations = 0
        self._active_lock = threading.Lock()

    def load_model(self):
        model_path = self.config['model_path']

        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path}")

        try:
            if self.is_vlm:
                # Load VLM model with fallback strategies for common issues
                logging.info("Loading VLM model with fallback strategies")
                self.model, self.processor = self._load_vlm_with_fallback(model_path)
            else:
                # Load text-only model with MLX LM
                logging.info("Loading text-only model using MLX LM")
                self.model, self.processor = lm_load(model_path)

            logging.info(f"Successfully loaded {'VLM' if self.is_vlm else 'LLM'} model")

            # Debug model structure for KV cache optimization
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("=== Model Structure Debug ===")
                if hasattr(self.model, '__dict__'):
                    logging.debug(f"Model attributes: {list(self.model.__dict__.keys())[:10]}...")
                if hasattr(self.model, 'config'):
                    logging.debug(f"Config type: {type(self.model.config)}")
                    if hasattr(self.model.config, '__dict__'):
                        logging.debug(f"Config attributes: {list(self.model.config.__dict__.keys())[:10]}...")
                    if hasattr(self.model.config, 'text_config'):
                        logging.debug(f"Text config type: {type(self.model.config.text_config)}")
                        if hasattr(self.model.config.text_config, '__dict__'):
                            text_attrs = list(self.model.config.text_config.__dict__.keys())
                            logging.debug(f"Text config attributes: {text_attrs[:15]}...")
                            # Log specific KV cache related attributes
                            tc = self.model.config.text_config
                            logging.debug(f"Text config details: num_hidden_layers={getattr(tc, 'num_hidden_layers', 'N/A')}, "
                                        f"num_attention_heads={getattr(tc, 'num_attention_heads', 'N/A')}, "
                                        f"hidden_size={getattr(tc, 'hidden_size', 'N/A')}")
                if hasattr(self.model, 'args'):
                    logging.debug(f"Args type: {type(self.model.args)}")
                    if hasattr(self.model.args, '__dict__'):
                        logging.debug(f"Args attributes: {list(self.model.args.__dict__.keys())[:10]}...")
                if hasattr(self.model, 'model_args'):
                    logging.debug(f"Model args: {self.model.model_args}")
                if hasattr(self.model, 'layers'):
                    logging.debug(f"Number of layers: {len(self.model.layers)}")
                    if len(self.model.layers) > 0:
                        logging.debug(f"First layer type: {type(self.model.layers[0])}")
                logging.debug("===========================")

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise e

        # Load draft model if specified
        if draft_path := self.config.get('draft_model_path'):
            logging.info(f"Loading draft model for speculative decoding: {draft_path}")
            try:
                # Draft models are always text-only
                self.draft_model, _ = lm_load(draft_path)
                logging.info("Draft model loaded successfully for speculative decoding")
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
        # Unified text strategy handles both text-only and VLM text paths
        self._strategies['text'] = UnifiedTextStrategy(
            draft_model=self.draft_model,
            model_id=self.model_id,
            model_config=self.config,
            is_vlm=self.is_vlm,
        )
        if self.is_vlm:
            self._strategies['vision'] = VLMVisionStrategy()

    def _detect_images_optimized(self, messages: List) -> bool:
        """Single-pass scan for images with early termination."""
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if hasattr(part, 'type'):
                        if part.type == 'image_url':
                            return True
                    elif isinstance(part, dict) and part.get('type') == 'image_url':
                        return True
        return False

    def _apply_model_defaults(self, request: ChatRequest) -> dict:
        """Apply model defaults with minimal object creation.

        Supports thinking mode defaults for Qwen3 models.
        """
        global_defaults = {
            'temperature': 0.1,
            'top_p': 1.0,
            'top_k': 0,
            'min_p': 0.0,
            'max_tokens': 512,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0
        }

        # Qwen3 thinking mode optimal settings (from Qwen3 best practices)
        thinking_defaults = {
            'temperature': 0.6,  # Don't use 0 - causes repetition
            'top_p': 0.95,
            'top_k': 20,
            'min_p': 0.0,
            'presence_penalty': 1.5  # Reduces repetition in thinking
        }

        # Start with global defaults
        merged_config = global_defaults.copy()

        # If thinking mode is enabled for this model, apply thinking defaults
        if self.config.get('enable_thinking', False):
            merged_config.update(thinking_defaults)

        # Apply model config overrides
        config_keys = ['temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
                       'repetition_penalty', 'presence_penalty', 'enable_thinking']
        merged_config.update({k: v for k, v in self.config.items() if k in config_keys and v is not None})

        # Include cache and speculative decoding config from model config
        cache_keys = ['cache_type', 'kv_bits', 'kv_group_size', 'max_kv_size',
                      'quantized_kv_start', 'num_draft_tokens']
        for key in cache_keys:
            if key not in merged_config and key in self.config:
                merged_config[key] = self.config[key]

        # Get only the scalar parameter fields from the request (highest priority)
        # Uses getattr instead of model_dump() to avoid serializing the entire message list
        request_fields = ['temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
                          'repetition_penalty', 'presence_penalty', 'enable_thinking', 'seed']
        for field in request_fields:
            val = getattr(request, field, None)
            if val is not None:
                merged_config[field] = val

        return merged_config

    def _get_batch_processor(self):
        """Lazy-initialize batch text processor."""
        if self._batch_processor is None:
            from .mlx_batch_text import TextBatchProcessor

            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

            # Get batch configuration from model config
            batch_config = self.config.get('batch', {})
            completion_batch_size = batch_config.get('completion_batch_size', 32)
            prefill_batch_size = batch_config.get('prefill_batch_size', 8)
            prefill_step_size = batch_config.get('prefill_step_size', 2048)

            self._batch_processor = TextBatchProcessor(
                model=self.model,
                tokenizer=tokenizer,
                max_tokens=self.config.get('max_tokens', 512),
                completion_batch_size=completion_batch_size,
                prefill_batch_size=prefill_batch_size,
                prefill_step_size=prefill_step_size
            )

            logging.info(
                f"[BATCH] Initialized batch processor for {self.model_id}: "
                f"completion_batch_size={completion_batch_size}, "
                f"prefill_batch_size={prefill_batch_size}"
            )

        return self._batch_processor

    def create_batch_chat_completion(self, requests: List[ChatRequest]) -> List[Dict]:
        """
        Process batch of chat completion requests.

        This uses mlx-lm's BatchGenerator for efficient parallel processing.
        Only works for text-only models without streaming.

        Args:
            requests: List of ChatRequest objects

        Returns:
            List of completion dictionaries
        """
        if self.is_vlm:
            raise ValueError("Batch processing is currently only supported for text-only models")

        # Check all requests are compatible with batching
        if any(req.stream for req in requests):
            raise ValueError("Batch processing does not support streaming requests")

        # Prepare all prompts
        prompts = []
        max_tokens_list = []

        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        for req in requests:
            # Apply chat template
            messages_for_template = []
            for msg in req.messages:
                msg_dict = msg.model_dump(exclude_none=True)
                # If content is a list, extract text
                if isinstance(msg_dict.get('content'), list):
                    text_parts = [part['text'] for part in msg_dict['content'] if part.get('type') == 'text']
                    msg_dict['content'] = ' '.join(text_parts)
                # Reconstruct thinking tags for assistant messages with edited thinking
                msg_dict = _reconstruct_thinking(msg_dict)
                messages_for_template.append(msg_dict)

            # Prefill: if last message is assistant, don't add generation prompt
            last_is_assistant = messages_for_template[-1].get('role') == 'assistant' if messages_for_template else False
            add_gen_prompt = not last_is_assistant

            prompt = tokenizer.apply_chat_template(
                messages_for_template,
                tokenize=False,
                add_generation_prompt=add_gen_prompt
            )

            tokens = tokenizer.encode(prompt)
            prompts.append(tokens)
            max_tokens_list.append(req.max_tokens or self.config.get('max_tokens', 512))

        # Process batch
        processor = self._get_batch_processor()

        with self._active_lock:
            self._active_generations += 1
        try:
            with self._generation_lock:
                results = processor.process_batch(prompts, max_tokens_list)
        finally:
            with self._active_lock:
                self._active_generations -= 1

        # Convert to API response format
        completions = []
        for result in results:
            completion = {
                'text': result.text,
                'finish_reason': result.finish_reason,
                'prompt_tokens': result.prompt_tokens,
                'completion_tokens': result.generation_tokens,
                'total_tokens': result.prompt_tokens + result.generation_tokens
            }
            completions.append(completion)

        return completions

    def create_chat_completion(self, request: ChatRequest) -> Generator:
            """
            Create chat completion using appropriate generation strategy.

            Path decision logic is pre-compiled and cached to minimize runtime overhead.
            """
            class MLXErrorChunk:
                def __init__(self, text):
                    self.text = text

            # Increment active generation counter (prevents safe unload during generation)
            with self._active_lock:
                self._active_generations += 1

            # Preemption: if the lock is held by another generation, abort it
            # so we can start this request sooner instead of blocking.
            lock_acquired = self._generation_lock.acquire(blocking=False)
            if not lock_acquired:
                logging.info(f"[MLX GENERATION] Preempting current generation on {self.model_id}")
                self._abort_event.set()
                self._generation_lock.acquire(blocking=True)
            # Clean slate for this generation
            self._abort_event.clear()

            try:
                effective_request = self._apply_model_defaults(request)

                if self.verbose:
                    logging.debug(f"MLX effective request params: {json.dumps(effective_request, indent=2)}")

                # Add null check for processor before accessing tokenizer
                if self.processor is None:
                    yield MLXErrorChunk(text=f"Error: Model processor not loaded for '{self.model_id}'")
                    return

                try:
                    has_images = self._detect_images_optimized(request.messages)

                    if not self.is_vlm and has_images:
                        yield MLXErrorChunk(text=f"Error: Model '{self.model_id}' is text-only and cannot process images. Please use a vision model for image inputs.")
                        return

                    if self.is_vlm and has_images:
                        strategy = self._strategies['vision']
                        logging.info(f"[MLX STRATEGY] Vision path | Model: {self.model_id}")
                    else:
                        strategy = self._strategies['text']
                        logging.info(f"[MLX STRATEGY] Text path (vlm={self.is_vlm}) | Model: {self.model_id}")

                    yield from strategy.generate(request, effective_request, self.model, self.processor, abort_event=self._abort_event)

                except Exception as e:
                    logging.error(f"MLX model call failed: {e}", exc_info=True)

                    # Reset MLX state on error to prevent stream context issues
                    try:
                        gc.collect()
                        mx.eval()  # Synchronize any pending MLX operations
                    except Exception as cleanup_error:
                        logging.debug(f"Cleanup error (non-critical): {cleanup_error}")

                    yield MLXErrorChunk(text=f"Error: MLX generation failed: {str(e)}")
            finally:
                # Always release the generation lock
                self._generation_lock.release()
                # Decrement active generation counter
                with self._active_lock:
                    self._active_generations -= 1
                # Release MLX internal memory cache between requests
                mx.clear_cache()

    def _get_context_capacity(self) -> int:
        """Get max context window size from model config."""
        if not hasattr(self.model, 'config'):
            return 32768  # Default fallback

        config = self.model.config
        if hasattr(config, 'max_position_embeddings'):
            return config.max_position_embeddings
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'max_position_embeddings'):
            return config.text_config.max_position_embeddings
        if hasattr(config, 'max_seq_len'):
            return config.max_seq_len
        return 32768

    def _get_context_used(self) -> int:
        """Get current context usage from prompt cache (thread-safe)."""
        try:
            cache_manager = get_global_cache_manager()
            return cache_manager.get_context_usage(self.model_id)
        except Exception as e:
            logging.debug(f"Could not get context usage from cache: {e}")
            return 0

    def get_metrics(self) -> ModelMetrics:
        """Get current metrics for this model (context usage, memory, etc.)."""
        try:
            metal_memory_mb = mx.metal.get_active_memory() / (1024 * 1024)
            context_used = self._get_context_used()
            context_capacity = self._get_context_capacity()
            context_percent = (context_used / context_capacity * 100) if context_capacity > 0 else 0.0

            return ModelMetrics(
                context_used=context_used,
                context_capacity=context_capacity,
                context_percent=round(context_percent, 1),
                memory_mb=round(metal_memory_mb, 1),
                requests_active=self._active_generations
            )
        except Exception as e:
            logging.warning(f"Failed to get MLX metrics: {e}")
            return ModelMetrics(
                context_used=0,
                context_capacity=0,
                context_percent=0.0,
                memory_mb=0.0,
                requests_active=0
            )

    def clear_cache(self) -> bool:
        """Clear the prompt cache for this model."""
        try:
            cache_manager = get_global_cache_manager()
            cache_manager.invalidate_cache(self.model_id)
            logging.info(f"Cleared prompt cache for {self.model_id}")
            return True
        except Exception as e:
            logging.warning(f"Failed to clear cache for {self.model_id}: {e}")
            return False

    def unload(self):
        """Cleanup with cache clearing and performance logging.

        Waits for active generations to complete before releasing model resources
        to prevent Metal command buffer crashes during LRU cache eviction.
        """
        logging.info(f"Unloading MLX model: {self.model_id}")

        # Wait for active generations to finish before destroying model resources
        max_wait = 30  # seconds
        start = time.time()
        while True:
            with self._active_lock:
                if self._active_generations == 0:
                    break
                active = self._active_generations
            elapsed = time.time() - start
            if elapsed > max_wait:
                logging.warning(
                    f"Force unloading {self.model_id} after {max_wait}s "
                    f"with {active} active generation(s)"
                )
                break
            # Log every 2 seconds to avoid spam
            if int(elapsed * 10) % 20 == 0:
                logging.info(
                    f"Waiting for {active} active generation(s) on {self.model_id} "
                    f"before unload ({elapsed:.1f}s elapsed)"
                )
            time.sleep(0.1)

        # Clear caches
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
