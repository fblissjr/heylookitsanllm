# src/heylook_llm/providers/mlx_provider.py
import gc
import logging
import json
import threading
import time
from pathlib import Path
from typing import Generator, Dict, List, Tuple

import mlx.core as mx
from PIL import Image

from mlx_lm.utils import load as lm_load
from mlx_lm.generate import wired_limit
from mlx_lm.models.cache import make_prompt_cache
from mlx_vlm.utils import load as vlm_load, prepare_inputs as vlm_prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template as mlx_vlm_apply_chat_template

from ..config import ChatRequest, ModelMetrics, MLX_RUNTIME_DEFAULT_FIELDS
from ..presets import get_preset_registry
from .abort import AbortEvent
from .base import BaseProvider, GenerationFailed, InvalidGenerationRequest
from .common.samplers import build as build_sampler
from .common.vlm_inputs import _reconstruct_thinking
from .common.model_wrappers import wrap_language_model
from .common.generation_core import generate_text, run_generation
from .common.batch_vision import BatchVisionProcessor
from .common.prompt_cache import get_global_cache_manager
from .common.vision_feature_cache import VisionFeatureCache
from .common.loader_routing import resolve_effective_loader, read_model_type
from .common.generation_gate import GenerationGate, GenerationCancelled
from .common.template_info import (
    install_chat_template,
    is_explicit_source,
    missing_template_error,
    read_template_info,
)

# -- transformers 5.x compatibility patches (no torchvision) --
# On MLX-only setups, torchvision is absent and transformers assumes it is
# present when loading VLM processors. Two patches remain (verified still
# needed against transformers 5.5.4):
#   1. AutoVideoProcessor.from_pretrained hard-fails with ImportError
#   2. ProcessorMixin.__init__ rejects None for video_processor sub-processor
# Both make video-processor loading gracefully degrade to None.
# (Two earlier patches were removed 2026-07-06 as verified dead: the
# VIDEO_PROCESSOR_MAPPING_NAMES access is now backend-gated and raises before
# the patch can apply, and `from transformers.utils import auto_docstring`
# now binds the decorator function, not the submodule -- both no-oped.)
_log = logging.getLogger(__name__)


def _apply_transformers_patches():
    try:
        from transformers.models.auto.video_processing_auto import AutoVideoProcessor
        _orig_vp_from_pretrained = AutoVideoProcessor.from_pretrained.__func__

        @classmethod
        def _soft_vp_from_pretrained(cls, *args, **kwargs):
            try:
                return _orig_vp_from_pretrained(cls, *args, **kwargs)
            except (ImportError, TypeError, ValueError):
                return None

        AutoVideoProcessor.from_pretrained = _soft_vp_from_pretrained
    except Exception as e:
        _log.debug("transformers patch 1 (AutoVideoProcessor) skipped: %s", e)

    try:
        from transformers import processing_utils as pu
        _orig_check = pu.ProcessorMixin.check_argument_for_proper_class

        def _lenient_check(self, attribute_name, arg):
            if arg is None and "video" in attribute_name:
                return None
            return _orig_check(self, attribute_name, arg)

        pu.ProcessorMixin.check_argument_for_proper_class = _lenient_check
    except Exception as e:
        _log.debug("transformers patch 2 (ProcessorMixin) skipped: %s", e)


_apply_transformers_patches()
del _apply_transformers_patches

# Create dedicated generation stream for better Metal utilization
# This allows async evaluation and improves pipeline performance.
#
# Must be thread-local: generation runs on FastAPI's thread pool
# (asyncio.to_thread / run_in_executor), not this import thread. MLX streams
# are bound to the thread that creates them -- a plain mx.new_stream() here
# would raise "There is no Stream(gpu, 0) in current thread." when wired_limit
# synchronizes it from a pool worker. new_thread_local_stream materializes the
# stream per-thread (matching mlx_lm.generate's own generation_stream).
generation_stream = mx.new_thread_local_stream(mx.default_device())


# Process-global generation gate. There is ONE GPU, so generation must serialize
# across ALL loaded MLX models, not just within a single provider -- otherwise
# with max_loaded_models>1 two providers would run concurrent generations on the
# shared Metal command queue. Shared across every MLXProvider instance; the first
# provider created sets max_queue_depth (process-wide, documented in config).
_GENERATION_GATE = None
_GENERATION_GATE_LOCK = threading.Lock()


def _get_generation_gate(max_waiting: int) -> "GenerationGate":
    global _GENERATION_GATE
    with _GENERATION_GATE_LOCK:
        if _GENERATION_GATE is None:
            _GENERATION_GATE = GenerationGate(max_waiting=max_waiting)
        return _GENERATION_GATE


def _resolve_enable_thinking(effective_request: dict, model_config: dict):
    """Request value wins; else the model-config default. Always returns an
    explicit bool on the router path (model_dump gives enable_thinking=False
    by default) -- an ABSENT kwarg is uncontrollable: mlx-lm's
    TokenizerWrapper silently injects enable_thinking=True and raw
    transformers falls back to template defaults."""
    enable_thinking = effective_request.get("enable_thinking")
    if enable_thinking is None:
        enable_thinking = model_config.get('enable_thinking', True)
    return enable_thinking


def vlm_apply_chat_template(processor, config, messages, num_images=None, enable_thinking=None):
    """
    Apply chat template using mlx-vlm's prompt_utils.

    Uses mlx-vlm to build messages with image tokens, then flattens any
    list-typed content to strings before passing to the tokenizer's
    apply_chat_template.  Some models (mistral3, pixtral) produce list
    content that their own Jinja templates cannot render.

    Args:
        processor: The model processor (contains tokenizer)
        config: Model config (contains model_type for proper formatting)
        messages: List of message dicts with 'role' and 'content'
        num_images: Number of images to add tokens for
        enable_thinking: Template thinking toggle. None = don't pass the kwarg
            (template default applies); a bool is forwarded to the template.
            Transformers passes extra kwargs through as template variables, so
            models without the variable ignore it.

    Returns:
        str: Formatted prompt string with proper image tokens
    """
    num_images = num_images or 0

    # Step 1: let mlx-vlm insert image tokens into the messages structure
    formatted_messages = mlx_vlm_apply_chat_template(
        processor, config, messages, num_images=num_images, return_messages=True
    )

    # Step 2: flatten any list content to strings so all tokenizer
    # Jinja templates can handle them
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token",
                          getattr(tokenizer, "image_token", "<image>"))

    for msg in formatted_messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type in ("text", "input_text"):
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            parts.append(text)
                    elif item_type in ("image", "image_url", "input_image"):
                        parts.append(image_token)
                elif isinstance(item, str):
                    parts.append(item)
            msg["content"] = " ".join(parts).strip() if parts else ""

    # Step 3: apply the tokenizer's own chat template
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    try:
        return tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True,
            **template_kwargs,
        )
    except TypeError:
        # Tokenizer template still can't handle the messages -- manual fallback
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in formatted_messages
        )


def resolve_add_generation_prompt(messages) -> bool:
    """Prefill convention: a trailing assistant message means the client
    wants the model to CONTINUE that message (no new generation prompt);
    otherwise open a fresh assistant turn."""
    last_is_assistant = messages[-1].get('role') == 'assistant' if messages else False
    return not last_is_assistant


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
        tokenizer = getattr(processor, "tokenizer", processor)

        messages_for_template = self._prepare_messages(request.messages)
        prompt_tokens = self._apply_template(messages_for_template, tokenizer, processor, model, effective_request)
        gen_model = self._get_generation_model(model)

        # Compute system prompt token boundary for segment-aware cache eviction.
        # Only re-tokenizes the system prefix (not the full prompt again).
        system_prefix_len = self._find_system_boundary(
            messages_for_template, prompt_tokens, tokenizer, processor, model, effective_request
        )

        # VLM mRoPE position state is reset in run_generation via _reset_vlm_positions().

        yield from generate_text(
            model=gen_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            effective_request=effective_request,
            model_id=self.model_id,
            draft_model=self.draft_model,
            cache_manager=self.cache_manager,
            abort_event=abort_event,
            system_prefix_len=system_prefix_len,
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
        enable_thinking = _resolve_enable_thinking(effective_request, self.model_config)

        if self.is_vlm:
            prompt = vlm_apply_chat_template(
                processor, model.config, messages, num_images=0,
                enable_thinking=enable_thinking,
            )
        else:
            add_gen_prompt = resolve_add_generation_prompt(messages)

            try:
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
            except ValueError as e:
                # transformers raises a raw ValueError when the tokenizer has
                # no chat template at all; that message surfaces verbatim as
                # the HTTP error detail, so make it name the model and the fix.
                # Decided from tokenizer state, not upstream error prose.
                err = missing_template_error(tokenizer, self.model_id)
                if err is not None:
                    raise err from e
                raise

        if isinstance(prompt, str):
            return tokenizer.encode(prompt)
        return prompt

    def _find_system_boundary(self, messages, prompt_tokens, tokenizer, processor, model, effective_request) -> int:
        """Find the token boundary where system prompt ends.

        Tokenizes just the system messages + empty user message, then compares
        against the already-computed prompt_tokens to find the divergence point.
        Only one extra tokenization (the system prefix), not two.

        Returns 0 if no system messages or computation fails.
        """
        # Count leading system messages
        num_system = 0
        for msg in messages:
            if msg.get('role') == 'system':
                num_system += 1
            else:
                break

        if num_system == 0:
            return 0

        try:
            # Tokenize system messages + empty user to find boundary
            sys_messages = messages[:num_system] + [{"role": "user", "content": ""}]
            sys_tokens = self._apply_template(sys_messages, tokenizer, processor, model, effective_request)

            # Find divergence point against already-computed full prompt
            sys_end = 0
            for i, (a, b) in enumerate(zip(sys_tokens, prompt_tokens)):
                if a != b:
                    sys_end = i
                    break
            else:
                sys_end = min(len(sys_tokens), len(prompt_tokens))

            if sys_end > 0:
                logging.debug(f"System prefix: {sys_end} tokens ({num_system} system messages)")
            return sys_end
        except Exception:
            return 0

    def _get_generation_model(self, model):
        """Return raw model for text-only, LanguageModelLogitsWrapper for VLM.

        The wrapper is cached per strategy instance so it's created once and reused.
        For VLM, wired_limit and cache operations use the wrapper (which wraps
        the language model component), not the full VLM model.
        """
        if not self.is_vlm:
            return model

        if self._cached_wrapper is None:
            self._cached_wrapper = wrap_language_model(model)
        return self._cached_wrapper


class _VisionTokenResponse:
    """Lightweight response for the first token from VLM vision encoding.

    Compatible with the GenerationResponse interface that api.py expects
    (needs .text, optionally .token and .logprobs). ``queue_wait_ms`` is a slot
    so create_chat_completion can tag this first vision token with the FIFO
    queue-wait time (a plain GenerationResponse is non-slotted; this isn't).
    """
    __slots__ = ('text', 'token', 'logprobs', 'queue_wait_ms')

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

    Vision feature caching (inspired by mlx-vlm VisionFeatureCache):
    When a model supports encode_image(), vision encoder outputs are cached
    by image URL so multi-turn conversations skip the expensive vision tower
    forward pass on repeated images.

    This gives vision requests the full sampler suite, abort support, and
    speculative decoding from generation_core -- a single code path for all
    MLX generation.
    """

    def __init__(self, model_config=None):
        self.model_config = model_config or {}
        self._batch_vision_processor = None
        self._cached_wrapper = None
        self._vision_cache = VisionFeatureCache(max_entries=20)

    def generate(self, request: ChatRequest, effective_request: dict, model, processor, abort_event: AbortEvent | None = None) -> Generator:
        tokenizer = getattr(processor, "tokenizer", processor)
        sampler, processors = build_sampler(tokenizer, effective_request)

        # Initialize batch vision processor for parallel image loading
        if self._batch_vision_processor is None:
            self._batch_vision_processor = BatchVisionProcessor(max_workers=4)

        # Prepare VLM inputs: extract images, format prompt with chat template
        images, formatted_prompt, _, image_urls = self._prepare_vlm_inputs_parallel(
            request.messages, processor, model.config, model,
            enable_thinking=_resolve_enable_thinking(effective_request, self.model_config),
        )

        num_images = len(images) if images else 0
        model_type = getattr(model.config, 'model_type', 'unknown')
        logging.info(f"[VLM VISION] Processing {num_images} image(s) | Model: {model_type}")

        # Model-agnostic visual token budget: mapped onto the processor's own
        # knob (gemma buckets / qwen pixel budget) as call-time kwargs; {}
        # when unset or the family exposes no budget parameter.
        from .common.vision_budget import vision_budget_kwargs
        budget_kwargs = vision_budget_kwargs(
            processor, effective_request.get('vision_tokens')
        )
        if budget_kwargs:
            logging.info(f"[VLM VISION] vision_tokens={effective_request.get('vision_tokens')} -> {budget_kwargs}")

        # Tokenize and prepare pixel values via mlx_vlm.utils.prepare_inputs.
        # This handles image_grid_thw for Qwen models automatically.
        image_token_index = getattr(model.config, 'image_token_index', None)
        inputs = vlm_prepare_inputs(
            processor,
            images=images if images else None,
            prompts=formatted_prompt,
            image_token_index=image_token_index,
            **budget_kwargs,
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
        # VLM models expect `mask` (not `attention_mask`). Default to all-ones
        # so models with a required mask positional arg (mistral3, pixtral, llava_next) don't fail.
        if mask is None:
            mask = mx.ones(input_ids.shape, dtype=mx.int32)
        vlm_kwargs["mask"] = mask

        # Vision feature caching: reuse cached vision encoder outputs across turns.
        # Follows mlx-vlm's generate pattern (now mlx_vlm/generate/ package):
        # - If model has encode_image(), we can compute and cache vision features
        # - cached_image_features kwarg bypasses the vision tower in the model's
        #   get_input_embeddings() method
        has_encode_image = hasattr(model, 'encode_image')
        cache_key = image_urls if image_urls else None
        if cache_key and budget_kwargs:
            # Feature shapes differ per budget: a URL-keyed entry cached at
            # one budget must not serve another. (Pixel-hash keys are safe
            # by construction -- the pixels themselves change.)
            budget_tag = ",".join(f"{k}={v}" for k, v in sorted(budget_kwargs.items()))
            cache_key = [f"vision_budget[{budget_tag}]", *cache_key]
        if has_encode_image and pixel_values is not None:
            # Try URL-based key first; fall back to pixel content hash for
            # base64/PIL images that don't have a stable URL.
            cached_features = self._vision_cache.get(cache_key, pixel_values=pixel_values)
            if cached_features is not None:
                vlm_kwargs["cached_image_features"] = cached_features
                logging.info("[VLM VISION] Using cached vision features (skipping vision encoder)")
            else:
                # Compute and cache vision features separately
                with wired_limit(model, [generation_stream]):
                    features = model.encode_image(pixel_values)
                    mx.async_eval(features)
                self._vision_cache.put(cache_key, features, pixel_values=pixel_values)
                vlm_kwargs["cached_image_features"] = features
                logging.info("[VLM VISION] Computed and cached vision features")

        # Ensure wrapper is cached for language model generation
        if self._cached_wrapper is None:
            self._cached_wrapper = wrap_language_model(model)

        # Create KV cache sized for the language model
        request_cache = make_prompt_cache(self._cached_wrapper)

        # Phase 1: Vision encoding -- run full VLM forward pass.
        # The VLM passes cache= through to model.language_model(), so the
        # language model writes KV state directly into request_cache.
        # When cached_image_features is set, the vision tower is skipped
        # inside the model's get_input_embeddings().
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

        # Yield the first token RAW (skip_special_tokens=False): the first
        # sampled token after a vision prefill can be a structural marker
        # (gemma-4 opens thinking with <|channel>) that the reasoning parser
        # must see; parsers strip non-structural specials from routed text.
        first_text = tokenizer.decode([first_token_id], skip_special_tokens=False)
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

    def _prepare_vlm_inputs_parallel(self, messages: List, processor, config, model=None, enable_thinking=None) -> Tuple[List[Image.Image], str, bool, List[str]]:
        """Prepare VLM inputs with parallel image loading. Delegates to standalone function."""
        from .common.vlm_inputs import prepare_vlm_inputs_parallel
        return prepare_vlm_inputs_parallel(
            messages, processor, config, self._batch_vision_processor,
            vlm_apply_chat_template, model=model, enable_thinking=enable_thinking,
        )


# Layer-1 sampler floor: what a request gets when neither the request, a
# preset, nor the model config says anything. Chat-sane values -- this is the
# de-facto default for freshly imported models with no default_preset (the
# old 0.1/512 floor made every such model near-greedy and truncated long
# answers mid-sentence).
GLOBAL_SAMPLER_FLOOR = {
    'temperature': 0.7,
    'top_p': 1.0,
    'top_k': 0,
    'min_p': 0.0,
    'max_tokens': 4096,
    'repetition_penalty': 1.0,
    'presence_penalty': 0.0,
}


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
        # Engine routing: modalities (description) + loader (hint) -> the mlx
        # engine that loads. is_vlm derives from it. "auto" degrades a vision
        # model mlx-vlm can't load to mlx-lm rather than crashing at load; an
        # explicit loader forces the engine. See common/loader_routing.py.
        self.effective_loader = resolve_effective_loader(
            self.config, lambda: read_model_type(self.config.get("model_path", "")))
        self.is_vlm = self.effective_loader == "mlx-vlm"

        # Pre-compile generation strategies (avoids runtime branching)
        self._strategies = {}

        # Batch text processor (lazy-initialized)
        self._batch_processor = None

        # Serialize generation FIFO across concurrent requests. Process-global
        # (shared across all providers -- one GPU); queues in arrival order (no
        # preemption); check_capacity() rejects with ModelBusyError (-> 503)
        # once max_queue_depth are waiting. The abort signal is NOT stored here:
        # it is per-request (created/passed by create_chat_completion) so one
        # client's disconnect can't abort another client's generation.
        self._gen_gate = _get_generation_gate(
            int(self.config.get("max_queue_depth", 8))
        )

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

            # Stop-token completeness: raw HF tokenizers on the mlx-vlm path
            # don't absorb generation_config's eos list (gemma-4: <turn|>
            # missing -> generation runs past end-of-turn). mlx-lm loads
            # already carry the full set; the union is a no-op there.
            # NB the old decode()-level special-token hygiene patch is GONE
            # on purpose: it stripped the structural channel markers before
            # the reasoning parser could see them -- the parsers strip
            # declared specials from ROUTED text themselves (strip_tokens).
            from .common.stop_tokens import extend_eos_from_generation_config
            extend_eos_from_generation_config(self.get_tokenizer(), model_path)

            # Read the model's chat template + tokenizer config (C4.5). The
            # resulting ModelTemplateInfo is the single source of truth for
            # output-parsing decisions: reasoning parser selection, strip-
            # tokens set, observability label. No hardcoded format lookup.
            self._template_info = read_template_info(
                Path(model_path),
                self.config.get("chat_template_source"),
            )
            logging.info(
                "template: source=%s harmony=%s thinking=%s specials=%d (model=%s)",
                self._template_info.template_source,
                self._template_info.has_harmony_structure,
                self._template_info.has_thinking_markers,
                len(self._template_info.special_tokens),
                self.model_id,
            )
            # NOTE: the reasoning parser is instantiated PER REQUEST from
            # _template_info (api.py) -- parser buffers must never be shared
            # across requests. The strip-regex compile cost that once
            # justified a load-time instance is covered by the pattern cache
            # in reasoning_parser._compile_strip_pattern.
            # Explicit chat_template_source: registry entry is authoritative,
            # overwrite whatever the tokenizer loaded. Auto (unset or "auto"):
            # only fill a MISSING tokenizer template (e.g. chat_template.json-
            # only models that AutoTokenizer loads nothing for).
            tok = self.get_tokenizer()
            installed = install_chat_template(
                tok, self._template_info,
                force=is_explicit_source(self.config.get("chat_template_source")),
            )
            # Warn only when NOTHING can render: no install happened, the
            # tokenizer has no HF template, and there's no wrapper-level
            # python template (mlx-lm chat_template_type sets
            # has_chat_template on the wrapper while the inner tokenizer's
            # chat_template attr stays None).
            has_template = (
                installed
                or bool(getattr(tok, "chat_template", None))
                or getattr(tok, "has_chat_template", False)
                or getattr(self.processor, "has_chat_template", False)
            )
            if not has_template:
                logging.warning(
                    "Model %s has NO chat template (no chat_template.jinja, no "
                    "tokenizer_config chat_template, no chat_template.json). "
                    "Chat requests will fail -- add a template file to the model "
                    "folder or set chat_template_source in models.toml.",
                    self.model_id,
                )

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
        """Handle specific weight mismatch issues.

        mlx-vlm's load() has accepted ``strict`` directly since well before
        the pinned version (verified 0.6.3: ``load(..., strict=True, **kw)``),
        so the old TypeError fallback + load_model monkeypatch are gone.
        """
        try:
            logging.debug("Attempting VLM load with strict=False")
            return vlm_load(model_path, strict=False)
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
            self._strategies['vision'] = VLMVisionStrategy(model_config=self.config)

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
        """Six-layer cascade producing the effective request config.

        Each layer overrides the previous for fields it sets; unset fields
        pass through. Request explicit > request preset > model default_preset
        > model sampler fields > thinking-mode flag > global floor.

        Layers:
            1. Global hardcoded floor.
            2. Thinking-mode defaults (when MODEL config sets
               ``enable_thinking=true``). Sourced from the 'thinking' preset.
            3. Model sampler fields from ``models.toml`` (per-model defaults).
            3b. Model ``default_preset`` (C4): applied only when the request
                has NO explicit preset. Unknown name logs-and-skips -- models
                are validated at startup, so a miss here means the registry
                changed post-startup and inference shouldn't die for it.
            4. Request preset (``ChatRequest.preset``). Overrides the model's
               default_preset. Unknown name propagates PresetNotFound ->
               translated to HTTP 400 by the route handler.
            5. Request-level explicit field values.
        """
        global_defaults = dict(GLOBAL_SAMPLER_FLOOR)

        merged_config = global_defaults.copy()

        registry = get_preset_registry()

        # Layer 2: when model declares itself thinking-capable, apply the
        # 'thinking' preset automatically. Registry is canonical source;
        # hardcoded fallback mirrors thinking.toml so inference keeps working
        # if the file is removed.
        if self.config.get('enable_thinking', False):
            if 'thinking' in registry:
                registry.apply_preset(merged_config, 'thinking')
            else:
                merged_config.update({
                    'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
                    'min_p': 0.0, 'presence_penalty': 1.5,
                })

        config_keys = ['temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
                       'repetition_penalty', 'repetition_context_size', 'presence_penalty', 'enable_thinking',
                       'vision_tokens']
        merged_config.update({k: v for k, v in self.config.items() if k in config_keys and v is not None})

        # Cache + speculative-decoding fields tagged with
        # json_schema_extra={"is_runtime_default": True} on MLXModelConfig.
        # Adding a new tagged field auto-propagates here.
        for key in MLX_RUNTIME_DEFAULT_FIELDS:
            if key not in merged_config and key in self.config:
                merged_config[key] = self.config[key]

        request_preset = getattr(request, 'preset', None)

        # Layer 3b: model default_preset applies only when the request didn't
        # pick one. Log-and-skip on unknown name (registry drift, not fatal).
        if not request_preset:
            model_default_preset = self.config.get('default_preset')
            if model_default_preset:
                if model_default_preset in registry:
                    registry.apply_preset(merged_config, model_default_preset)
                else:
                    logging.warning(
                        "model default_preset %r not in registry; skipping layer",
                        model_default_preset,
                    )

        # Layer 4: request preset. PresetNotFound propagates; route handlers
        # translate to HTTP 400. Keeping the provider transport-agnostic.
        registry.apply_preset(merged_config, request_preset)

        # Layer 5: request explicit fields.
        request_fields = ['temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
                          'repetition_penalty', 'repetition_context_size', 'presence_penalty', 'enable_thinking', 'seed',
                          'vision_tokens']
        for field in request_fields:
            val = getattr(request, field, None)
            if val is not None:
                merged_config[field] = val

        return merged_config

    def _get_batch_processor(self):
        """Lazy-initialize batch text processor."""
        if self._batch_processor is None:
            from .mlx_batch_text import TextBatchProcessor

            tokenizer = getattr(self.processor, "tokenizer", self.processor)

            # Get batch configuration from model config
            batch_config = self.config.get('batch', {})
            completion_batch_size = batch_config.get('completion_batch_size', 32)
            prefill_batch_size = batch_config.get('prefill_batch_size', 8)
            prefill_step_size = batch_config.get('prefill_step_size', 2048)

            self._batch_processor = TextBatchProcessor(
                model=self.model,
                tokenizer=tokenizer,
                max_tokens=self.config.get('max_tokens', GLOBAL_SAMPLER_FLOOR['max_tokens']),
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

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

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
            add_gen_prompt = resolve_add_generation_prompt(messages_for_template)

            try:
                prompt = tokenizer.apply_chat_template(
                    messages_for_template,
                    tokenize=False,
                    add_generation_prompt=add_gen_prompt
                )
            except ValueError as e:
                err = missing_template_error(tokenizer, self.model_id)
                if err is not None:
                    raise err from e
                raise

            tokens = tokenizer.encode(prompt)
            prompts.append(tokens)
            max_tokens_list.append(req.max_tokens or self.config.get('max_tokens', GLOBAL_SAMPLER_FLOOR['max_tokens']))

        # Process batch
        processor = self._get_batch_processor()

        with self._active_lock:
            self._active_generations += 1
        # Share the generation gate with chat completions so batch and chat
        # never run on the GPU concurrently. Batch is internal -- it queues
        # (no capacity check / 503).
        self._gen_gate.acquire()
        try:
            results = processor.process_batch(prompts, max_tokens_list)
        finally:
            self._gen_gate.release()
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

    def check_capacity(self) -> None:
        """Reject (ModelBusyError -> 503) when the FIFO queue is already full.

        Lets HTTP entry points apply backpressure before committing to a
        response. Generation itself still queues via the gate; this only bounds
        how deep the queue is allowed to grow for externally-submitted requests.
        """
        self._gen_gate.check_capacity()

    def generation_queue_stats(self) -> dict:
        """Snapshot of the FIFO generation queue (active/waiting/capacity)."""
        return self._gen_gate.snapshot()

    def create_chat_completion(self, request: ChatRequest, abort_event: "AbortEvent | None" = None) -> Generator:
            """
            Create chat completion using appropriate generation strategy.

            ``abort_event`` is the per-request cooperative cancel signal. The HTTP
            routes create one per request and share it with the streaming layer
            (which sets it on client disconnect); internal callers (batch, RLM)
            omit it and a fresh one is created. It is NOT a provider-level shared
            object -- that would let one client's disconnect abort another's
            in-flight generation.

            Path decision logic is pre-compiled and cached to minimize runtime overhead.
            """
            if abort_event is None:
                abort_event = AbortEvent()

            # FIFO queue: wait our turn instead of preempting the in-flight
            # generation. Concurrent requests complete in arrival order rather
            # than cannibalizing each other. If the client already disconnected
            # (abort_event set by the streaming layer while we were queued), bail
            # out of the queue instead of waiting our turn to do pointless work.
            _queue_wait_start = time.perf_counter()
            try:
                self._gen_gate.acquire(cancel_check=abort_event.is_set)
            except GenerationCancelled:
                return
            queue_wait_ms = (time.perf_counter() - _queue_wait_start) * 1000.0

            # Count as active only AFTER acquiring -- a queued request is
            # 'waiting' (tracked by the gate), not 'active'. (Prevents safe
            # unload during generation; avoids double-counting with requests_queued.)
            with self._active_lock:
                self._active_generations += 1

            try:
                effective_request = self._apply_model_defaults(request)

                if self.verbose:
                    logging.debug(f"MLX effective request params: {json.dumps(effective_request, indent=2)}")

                # Add null check for processor before accessing tokenizer
                if self.processor is None:
                    raise GenerationFailed(f"Model processor not loaded for '{self.model_id}'")

                try:
                    has_images = self._detect_images_optimized(request.messages)

                    if not self.is_vlm and has_images:
                        raise InvalidGenerationRequest(
                            f"Model '{self.model_id}' is text-only and cannot process images. "
                            f"Please use a vision model for image inputs."
                        )

                    if self.is_vlm and has_images:
                        strategy = self._strategies['vision']
                        logging.info(f"[MLX STRATEGY] Vision path | Model: {self.model_id}")
                    else:
                        strategy = self._strategies['text']
                        logging.info(f"[MLX STRATEGY] Text path (vlm={self.is_vlm}) | Model: {self.model_id}")

                    # Tag the FIRST taggable chunk with the FIFO queue-wait time
                    # (constant per request -- the route carries it forward, so
                    # one tag suffices and avoids a per-token write in the hot
                    # loop). Use an explicit loop (not `yield from`) but close the
                    # inner generator in a finally so GeneratorExit still
                    # propagates -- the gate must release promptly on disconnect.
                    inner = strategy.generate(request, effective_request, self.model, self.processor, abort_event=abort_event)
                    tagged = False
                    try:
                        for chunk in inner:
                            if not tagged:
                                try:
                                    chunk.queue_wait_ms = queue_wait_ms  # type: ignore[attr-defined]
                                    tagged = True
                                except (AttributeError, TypeError):
                                    pass
                            yield chunk
                    finally:
                        inner.close()

                except GenerationFailed:
                    raise
                except Exception as e:
                    logging.error(f"MLX model call failed: {e}", exc_info=True)

                    # Reset MLX state on error to prevent stream context issues
                    try:
                        gc.collect()
                        mx.eval()  # Synchronize any pending MLX operations
                    except Exception as cleanup_error:
                        logging.debug(f"Cleanup error (non-critical): {cleanup_error}")

                    raise GenerationFailed(f"MLX generation failed: {e}") from e
            finally:
                # Decrement active counter and clear the MLX cache BEFORE
                # releasing the slot, so GPU cleanup completes before the next
                # waiter is admitted (preserves one-generation-at-a-time).
                with self._active_lock:
                    self._active_generations -= 1
                mx.clear_cache()
                self._gen_gate.release()

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

    def warmup(self) -> None:
        """Text-only JIT prime. See BaseProvider.warmup() for the contract.

        VLM vision-tower compilation is intentionally NOT exercised -- add a
        synthetic-image warmup once the request-events log shows VLM
        cold-start pain.
        """
        tok = self.get_tokenizer()
        if tok is None:
            logging.debug(f"warmup: {self.model_id} has no tokenizer; skipping")
            return

        dummy_text = "The quick brown fox jumps over the lazy dog. " * 3
        try:
            prompt_tokens = list(tok.encode(dummy_text))
        except Exception:
            logging.debug(f"warmup: {self.model_id} tokenizer.encode failed", exc_info=True)
            return
        if not prompt_tokens:
            return

        from .common.generation_core import generate_text
        # Resolve the generation model through the SAME method real requests use
        # (UnifiedTextStrategy._get_generation_model) rather than re-deriving it
        # here. This keeps warmup structurally on the request path -- warmup
        # drifting from that path is exactly how the VLM LanguageModelOutput bug
        # (raw model -> non-subscriptable logits) went unnoticed. Strategies are
        # compiled in load() before the router calls warmup(); fall back to the
        # raw model defensively if that ever changes.
        text_strategy = self._strategies.get('text')
        gen_model = text_strategy._get_generation_model(self.model) if text_strategy else self.model

        t0 = time.time()
        try:
            for _ in generate_text(
                gen_model,
                tok,
                prompt_tokens,
                {"max_tokens": 4, "num_draft_tokens": 0},
                # model_id=None so warmup tokens don't land in the prompt cache;
                # draft_model=None to skip the speculative-decoding path.
                model_id=None,
                draft_model=None,
                abort_event=None,
            ):
                pass
        except Exception:
            # Warmup is best-effort (see BaseProvider.warmup contract), but a
            # failure means this model is never JIT-primed and the first real
            # request pays the full compilation cost. Log at WARNING so a
            # consistently-failing warmup is visible rather than buried -- this
            # is how the VLM LanguageModelOutput bug stayed hidden.
            logging.warning(
                f"warmup: {self.model_id} failed to prime; first request will pay "
                f"JIT compilation cost. Continuing without warmup.",
                exc_info=True,
            )
            return
        logging.info(f"warmup: {self.model_id} primed in {(time.time() - t0) * 1000:.0f}ms")

    def get_metrics(self) -> ModelMetrics:
        """Get current metrics for this model (context usage, memory, etc.)."""
        try:
            metal_memory_mb = mx.get_active_memory() / (1024 * 1024)
            context_used = self._get_context_used()
            context_capacity = self._get_context_capacity()
            context_percent = (context_used / context_capacity * 100) if context_capacity > 0 else 0.0

            return ModelMetrics(
                context_used=context_used,
                context_capacity=context_capacity,
                context_percent=round(context_percent, 1),
                memory_mb=round(metal_memory_mb, 1),
                requests_active=self._active_generations,
                requests_queued=self._gen_gate.snapshot()["waiting"],
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

        Waits for generation traffic to drain before releasing model
        resources -- ACTIVE generations (Metal command buffer crashes on
        teardown mid-decode) AND gate WAITERS: the active counter is
        decremented BEFORE gate.release() admits the next waiter, so an
        active-only wait can free weights exactly as a woken waiter starts
        generating. Living here (not in router call sites) means every
        teardown path -- LRU eviction, clear_cache, explicit unload, idle
        unload -- inherits the guarantee.

        The gate is process-global, so with multiple loaded models this
        also waits out OTHER models' traffic: accepted conservatism,
        bounded by the same 30s force-unload cap as before.
        """
        logging.info(f"Unloading MLX model: {self.model_id}")

        # Wait for active generations AND queued waiters to drain.
        max_wait = 30  # seconds
        start = time.time()
        while True:
            with self._active_lock:
                active = self._active_generations
            stats = self.generation_queue_stats() or {}
            waiting = stats.get("waiting", 0)
            if active == 0 and waiting == 0:
                break
            elapsed = time.time() - start
            if elapsed > max_wait:
                logging.warning(
                    f"Force unloading {self.model_id} after {max_wait}s "
                    f"with {active} active / {waiting} waiting generation(s)"
                )
                break
            # Log every 2 seconds to avoid spam
            if int(elapsed * 10) % 20 == 0:
                logging.info(
                    f"Waiting for {active} active / {waiting} waiting generation(s) "
                    f"on {self.model_id} before unload ({elapsed:.1f}s elapsed)"
                )
            time.sleep(0.1)

        # Clear vision feature cache before dropping strategy references
        vision_strategy = self._strategies.get('vision')
        if vision_strategy is not None and hasattr(vision_strategy, '_vision_cache'):
            cache = vision_strategy._vision_cache
            if cache is not None:
                stats = cache.stats()
                if stats["hits"] + stats["misses"] > 0:
                    logging.info(
                        f"Vision feature cache stats: {stats['hits']} hits, "
                        f"{stats['misses']} misses, {stats['hit_rate']:.0%} hit rate"
                    )
                cache.clear()

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
