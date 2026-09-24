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

from mlx_vlm.utils import load as vlm_load, prepare_inputs as vlm_prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template as mlx_vlm_apply_chat_template
from mlx_vlm.generate.common import generation_stream, wired_limit

from ..config import ChatRequest, ModelMetrics, MLX_RUNTIME_DEFAULT_FIELDS
from .abort import AbortEvent
from .base import BaseProvider, CacheReport, GenerationChunk, GenerationFailed, InvalidGenerationRequest
# Layer-1 sampler floor -- provider-shared, defined in heylook_llm.samplers
# (the llama-server provider applies the same floor).
from ..capabilities import model_context_length
from ..samplers import GLOBAL_SAMPLER_FLOOR, load_vendor_sampling, resolve_effective_sampling
from ..thinking_controls import detect as detect_thinking
from .common.samplers import build as build_sampler
from .common.vlm_inputs import (
    carry_message_extras, continue_from_generation_prompt, thinking_for_template)
from .common.generation_core import continuation_detokenizer, detokenizer_source
from .common import vlm_engine
from .common.batch_vision import BatchVisionProcessor
from .common.vision_feature_cache import VisionFeatureCache
from .common.loader_routing import resolve_serves_vision, read_model_type
from .common.generation_gate import GenerationGate, GenerationCancelled, get_process_gate
from .common.template_info import (
    install_chat_template,
    missing_template_error,
    read_template_info,
    should_force_install,
    thinking_budget_markers,
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



# Process-global generation gate. There is ONE GPU, so generation must serialize
# across ALL loaded MLX models, not just within a single provider -- otherwise
# with max_loaded_models>1 two providers would run concurrent generations on the
# shared Metal command queue. Shared across every MLXProvider instance; the first
# provider created sets max_queue_depth (process-wide, documented in config).
def _get_generation_gate(max_waiting: int) -> "GenerationGate":
    # Moved to generation_gate.get_process_gate so the llama-server provider
    # can share it (v1.79.60). Kept as a name because tests and comments here
    # refer to it.
    return get_process_gate(max_waiting)


def _resolve_enable_thinking(effective_request: dict) -> bool:
    """The thinking flag this prompt is templated with.

    Read straight off the effective request: the shared cascade
    (``samplers.resolve_effective_sampling``) materializes ``enable_thinking``
    unconditionally, so the key is always present and already carries every
    layer -- request field, then model config. There is nothing left to
    resolve here.

    It must stay an explicit bool: an ABSENT kwarg hands the decision to the
    template's own default. Same value ``BaseProvider.effective_thinking``
    reports to the parser -- by construction now, not by two call sites
    agreeing to read the same thing.
    """
    return bool(effective_request.get("enable_thinking"))


def _depth_kwargs(effective_request: dict, template_info) -> dict | None:
    """``{variable: value}`` for the requested depth, named by the in-force
    template's own depth variable (plan W2: Muse's reasoning_strength,
    MiniMax's thinking_mode, ...). None when no depth is set."""
    from ..thinking_controls import depth_variable, detect

    value = effective_request.get("reasoning_effort")
    if not value:
        return None
    return {depth_variable(detect(getattr(template_info, "chat_template", None))): value}


def _thinking_budget_criteria(request: ChatRequest, effective_request: dict,
                              template_info, tokenizer):
    """mlx-vlm's ThinkingBudgetCriteria for this request, or None (plan W7).

    None when no budget was asked for, the model has no thinking format, or
    thinking is off: there is nothing to cap. A budget on a format the engine
    cannot force shut (harmony) is refused rather than ignored, whatever the
    switch says: harmony has no switch and always reasons, so dropping the
    budget would let it think uncapped while the client believes it capped.
    The capability report never offers it there. The count starts where the
    parser starts reading thinking (``starts_inside_thinking``); a resumed
    thought's earlier text is not counted against the new budget.
    """
    budget = effective_request.get("thinking_budget_tokens")
    if not budget:
        return None
    markers = thinking_budget_markers(template_info)
    if markers is None and getattr(template_info, "has_harmony_structure", False):
        raise InvalidGenerationRequest(
            "thinking budget_tokens is not supported for this model: its thinking "
            "format cannot be closed by the engine (harmony models leave the "
            "analysis channel with a multi-token sequence)")
    thinking = _resolve_enable_thinking(effective_request) or request.resumes_thinking()
    if markers is None or not thinking:
        return None
    from mlx_vlm.utils import ThinkingBudgetCriteria
    from ..reasoning_parser import starts_inside_thinking

    opener, closer = markers
    return ThinkingBudgetCriteria(
        tokenizer, int(budget), thinking_end_token=closer, thinking_start_token=opener,
        enable_thinking=True,
        prompt_preopens_thinking=starts_inside_thinking(
            template_info, thinking_enabled=True,
            continuing=request.is_continuation(),
            resumes_thinking=request.resumes_thinking()))


def _thinking_resume(request: ChatRequest) -> str | None:
    """The partial thinking to RESUME, or None for every other request.

    The decision is the request's own (``ChatRequest.resumes_thinking``, the
    one predicate the route's parser selection and the preview also read);
    this only hands back the trace when it says yes.
    """
    return request.messages[-1].thinking if request.resumes_thinking() else None


def _thinking_resume_opener(template_info) -> str | None:
    """The marker a mid-thought resume must re-open, by template family, or
    None when the family has no resumable block. Read off ModelTemplateInfo
    -- the same probe the reasoning parser is selected from, so the opener
    appended here is the one the parser will treat as already open.
    All three served families since v1.79.63: the channel parsers take an
    initial-thinking state now, so the opener appended here is one the
    selected parser starts inside of.
    """
    if template_info is None:
        return None
    if getattr(template_info, "has_harmony_structure", False):
        return "<|channel|>analysis<|message|>"
    if getattr(template_info, "has_gemma_channel_structure", False):
        return "<|channel>thought\n"
    if getattr(template_info, "has_thinking_markers", False):
        return "<think>\n"
    return None


def _append_thinking_resume(prompt: str, thinking: str, template_info) -> str:
    """``prompt`` (a fresh generation prompt, thinking ON) with the partial
    ``thinking`` appended INSIDE the block. If the template's generation
    prompt already opened the block (Qwen3.5+ pre-fills ``<think>``), the
    trace follows it directly; otherwise the opener is added first (Qwen3,
    whose model emits ``<think>`` itself)."""
    opener = _thinking_resume_opener(template_info)
    if opener is None:
        raise InvalidGenerationRequest(
            "Resuming inside the thinking block is not supported for this "
            "model's template family yet -- edit the response box (even one "
            "word) to continue the answer instead, or regenerate.")
    head = prompt.rstrip()
    if head.endswith(opener.rstrip()):
        return head + "\n" + thinking.lstrip()
    return prompt + opener + thinking.lstrip()


def _apply_chat_template(tokenizer, messages, *, enable_thinking, depth,
                         continuing: bool, model_id=None):
    """Render ``messages`` through the tokenizer's chat template -- the ONE
    place that builds the kwargs and maps the failures, for the text path and
    the vision/VLM path alike.

    There were three hand-copies of this (text, VLM, batch) and they had
    drifted: only the text copy retried a narrow wrapper, only the text and
    batch copies named a missing template, the batch copy passed no
    ``enable_thinking`` at all (so mlx-lm silently injected True), and the VLM
    copy answered a TypeError by returning a ``role: content`` join -- a prompt
    with no template in it, handed to the model as if nothing had happened.

    ``enable_thinking`` None = omit the key (the template's own default);
    a bool is always sent.

    ``depth`` is ``{variable: value}`` for the template's own depth variable
    (``_depth_kwargs``; plan W2), or None to send nothing.

    ``continuing`` leaves the final message's turn OPEN so generation finishes
    it. transformers refuses ``continue_final_message`` together with
    ``add_generation_prompt``, hence the flip. A template stack that cannot
    continue is refused LOUDLY (400): rendering a closed turn instead would
    silently restart the message the caller asked to continue.
    """
    base_kwargs: dict = {"tokenize": False, "add_generation_prompt": not continuing}
    if continuing:
        base_kwargs["continue_final_message"] = True

    # Deliberately NOT in base_kwargs: the TypeError retry below re-passes
    # base_kwargs verbatim and drops only what is spelled out here, so a
    # wrapper with a narrow signature must be able to lose the depth variable
    # the same way it loses enable_thinking. In base_kwargs it would survive
    # the retry and fail it again. It is sent WHENEVER SET, never gated on
    # thinking: gpt-oss/harmony reads it unconditionally and has no
    # enable_thinking at all.
    template_kwargs: dict = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    if depth:
        template_kwargs.update(depth)

    try:
        try:
            return tokenizer.apply_chat_template(messages, **template_kwargs, **base_kwargs)
        except TypeError:
            # Can't tell WHICH kwarg the wrapper rejected; retry without the
            # template variables only. If continue_final_message itself is the
            # problem the retry fails the same way, and that is a refusal.
            try:
                return tokenizer.apply_chat_template(messages, **base_kwargs)
            except TypeError as te:
                if continuing:
                    raise InvalidGenerationRequest(
                        "This model's template stack cannot continue the "
                        "final message (continue_final_message unsupported)."
                    ) from te
                raise
    except ValueError as e:
        # transformers raises a raw ValueError when the tokenizer has no chat
        # template at all; that message surfaces verbatim as the HTTP error
        # detail, so make it name the model and the fix. Decided from
        # tokenizer state, not upstream error prose.
        err = missing_template_error(tokenizer, model_id)
        if err is not None:
            raise err from e
        if continuing:
            # transformers also ValueErrors when the template rewrites the
            # final message so its content no longer ends the rendered text
            # -- user-actionable, not a server fault.
            raise InvalidGenerationRequest(
                f"Cannot continue the final message with this model's "
                f"chat template: {e}"
            ) from e
        raise


def vlm_apply_chat_template(processor, config, messages, num_images=None, enable_thinking=None,
                            depth=None,
                            continue_final_message=False, model_id=None):
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
    # mlx-vlm rebuilds every message as role + content only, so any other key
    # the template reads -- `reasoning_content` above all -- was dropped on
    # this path, silently: gemma-4's continued turn lost its thought and the
    # model was asked to extend a reply with no thought channel, which
    # degenerates. Carry the rest of each message across (one output message
    # per input, same order).
    formatted_messages = carry_message_extras(messages, formatted_messages)

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

    # Step 3: the tokenizer's own chat template, through the shared renderer.
    # A template that still cannot take these messages RAISES. (It used to
    # return a "role: content" join here -- a prompt with no template in it.
    # Checked before removing, 2026-09-21: no installed mlx-vlm-routed model
    # reaches that branch, across image, multi-turn and thinking-history
    # shapes.)
    return _apply_chat_template(
        tokenizer, formatted_messages, enable_thinking=enable_thinking,
        depth=depth, continuing=continue_final_message,
        model_id=model_id)


class UnifiedTextStrategy:
    """Every text request, on every MLX model (text-only and vision models
    alike): render the prompt (template path by ``is_vlm``, thinking,
    continuation shapes), tokenize with mlx-vlm's ``prepare_inputs``, and
    run it through ``vlm_engine`` (plan W10)."""

    def __init__(self, model_id=None, model_config=None, is_vlm=False,
                 template_info=None, context_length=None, owner=None):
        # The provider: its APC store and stop set are read at CALL time,
        # so clear_cache replacing the store is seen by the next request.
        self.owner = owner
        self.context_length = context_length  # the provider's, for the over-length guard
        self.model_id = model_id
        self.model_config = model_config or {}
        self.is_vlm = is_vlm
        # ModelTemplateInfo (or None): names the thinking opener a mid-thought
        # resume has to re-open -- see _thinking_resume_opener.
        self.template_info = template_info

    def build_prompt(self, request: ChatRequest, effective_request: dict, model, processor):
        """The templated prompt for ``request``: a string, or (defensively)
        an already-tokenized list from a template stack that returns one.

        ONE builder for generation and for the prompt PREVIEW (v1.79.62), so
        what a user is shown is what the model is fed by construction.

        A continuation whose final assistant message carries thinking and NO
        content is a RESUME INSIDE THE THINKING BLOCK (a reply stopped
        mid-thought, then Save & Continue): the turn is rendered as a fresh
        generation prompt with thinking ON and the partial trace appended
        after the template's own opener, so the model's next token continues
        the reasoning. The closed-block reconstruction (`_reconstruct_thinking`)
        would instead present the fragment as FINISHED reasoning and start
        the answer -- which is what happened before, with the route then
        gluing the new trace onto the old one.
        """
        tokenizer = getattr(processor, "tokenizer", processor)
        resume = _thinking_resume(request)
        source = request.messages[:-1] if resume is not None else request.messages
        messages_for_template = self._prepare_messages(source)
        prompt = self._render_template(
            messages_for_template, tokenizer, processor, model, effective_request,
            continuing=request.is_continuation() and resume is None,
            resume_thinking=resume,
        )
        return prompt

    def render_prompt(self, request: ChatRequest, effective_request: dict, model, processor) -> str:
        """``build_prompt`` as the exact STRING the model sees (special
        tokens included). A tokenized result is decoded without skipping
        specials so the string still shows every marker."""
        prompt = self.build_prompt(request, effective_request, model, processor)
        if isinstance(prompt, str):
            return prompt
        tokenizer = getattr(processor, "tokenizer", processor)
        return tokenizer.decode(prompt, skip_special_tokens=False)

    def generate(self, request: ChatRequest, effective_request: dict, model, processor, abort_event: AbortEvent | None = None) -> Generator:
        """Render, tokenize the way mlx-vlm's own loop does (prepare_inputs),
        and run the request through vlm_engine (plan W10, A2)."""
        tokenizer = getattr(processor, "tokenizer", processor)

        prompt = self.build_prompt(request, effective_request, model, processor)
        if isinstance(prompt, str):
            raw = vlm_prepare_inputs(processor, prompts=prompt)
        else:
            raw = {"input_ids": mx.array([list(prompt)])}
        input_ids = raw["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            raw = {**raw, "input_ids": input_ids}
        sampler, processors = build_sampler(tokenizer, effective_request)
        yield from vlm_engine.generate(
            model=model, processor=processor,
            apc_manager=getattr(self.owner, "_apc", None),
            input_ids=input_ids, raw_inputs=raw,
            sampler=sampler, processors=processors,
            stop_tokens=getattr(self.owner, "_stop_tokens", ()),
            max_tokens=effective_request['max_tokens'],
            greedy=float(effective_request.get('temperature') or 0.0) == 0.0,
            abort_event=abort_event,
            # Both continuation shapes: a content prefill and a mid-thought
            # resume. The first token completes prefilled text either way.
            continuing=request.is_continuation(),
            context_length=self.context_length,
            prefill_step_size=effective_request.get('prefill_step_size'),
            model_id=self.model_id,
            detokenizer=(self.owner.streaming_detokenizer(request.is_continuation())
                         if self.owner is not None else None),
            thinking_budget=_thinking_budget_criteria(
                request, effective_request, self.template_info, tokenizer),
        )

    def _prepare_messages(self, messages) -> list[dict]:
        """Prepare messages for template application. Shared for both paths."""
        messages_for_template = []
        for msg in messages:
            msg_dict = msg.model_dump(exclude_none=True)
            if isinstance(msg_dict.get('content'), list):
                text_parts = [part['text'] for part in msg_dict['content'] if part.get('type') == 'text']
                msg_dict['content'] = ' '.join(text_parts)
            # Prior thinking, the way THIS template takes it: reasoning_content
            # where the template reads it (gemma-4, Qwen3+), reconstructed
            # <think> tags for a marker template that does not, nothing for a
            # family that has neither (v1.79.63 -- before, every family got
            # <think> text baked into content, gemma-4 included).
            msg_dict = thinking_for_template(msg_dict, self.template_info)
            messages_for_template.append(msg_dict)
        return messages_for_template

    def _render_template(self, messages, tokenizer, processor, model, effective_request,
                         continuing: bool = False, resume_thinking: str | None = None):
        """Dispatch template application based on is_vlm.

        Text-only: tokenizer.apply_chat_template with enable_thinking support.
        VLM text: vlm_apply_chat_template (uses processor + model config).

        ``continuing``: leave the final message's turn OPEN so generation
        finishes it (continue_final_message). Before this kwarg the trailing-
        assistant convention only suppressed the generation prompt, which
        still rendered the turn CLOSED -- the model saw a finished turn and
        nothing to continue.

        ``resume_thinking``: the partial trace to resume (see build_prompt).
        ``messages`` then EXCLUDES the message being resumed; the turn is
        rendered as a generation prompt with thinking forced ON and the
        trace appended after the opener. Mutually exclusive with
        ``continuing``.
        """
        enable_thinking = _resolve_enable_thinking(effective_request)
        if resume_thinking is not None:
            enable_thinking = True
        depth = _depth_kwargs(effective_request, self.template_info)

        def render(msgs, cont):
            if self.is_vlm:
                return vlm_apply_chat_template(
                    processor, model.config, msgs, num_images=0,
                    enable_thinking=enable_thinking,
                    depth=depth,
                    continue_final_message=cont, model_id=self.model_id,
                )
            return _apply_chat_template(
                tokenizer, msgs, enable_thinking=enable_thinking,
                depth=depth, continuing=cont,
                model_id=self.model_id)

        prompt = render(messages, continuing)
        if continuing:
            prompt = continue_from_generation_prompt(
                prompt, messages, lambda msgs: render(msgs, False))

        if resume_thinking is not None:
            if not isinstance(prompt, str):
                raise InvalidGenerationRequest(
                    "Cannot resume inside the thinking block: this model's "
                    "template stack returns tokens, not text.")
            prompt = _append_thinking_resume(prompt, resume_thinking, self.template_info)
        return prompt

def _has_audio_parts(messages) -> bool:
    """True if any message carries an input_audio content part.

    MLX loads gemma-4 audio models with skip_audio=True (the audio tower is
    stripped), so audio requests must fail LOUDLY here -- vlm_inputs would
    otherwise silently drop the part and answer text-only. Audio is served
    by provider="gguf" (llama-server).
    """
    for message in messages:
        content = message.content
        if isinstance(content, list):
            for part in content:
                if getattr(part, 'type', None) == 'input_audio':
                    return True
    return False


def _non_user_image_roles(messages) -> list[str]:
    """Roles other than ``user`` that carry an image part, in order.

    mlx-vlm can only render an image marker on a USER turn: the role is gated
    three times over in its ``prompt_utils`` -- media on a non-user message is
    not counted (``_content_media_count`` is skipped for system/assistant/tool),
    the surplus is reallocated to the last user turn, and the per-message
    formatter itself tests ``role == "user"`` before emitting a marker. So an
    image attached to an assistant turn does not fail there; it MOVES, and the
    model is told the picture arrived in the user's latest message.

    That silent relocation is the reason this refuses instead of proceeding.
    llama-server has no such gate (it rewrites an image part into a positional
    media marker wherever it sits, whatever the role), so the same conversation
    is servable by a gguf model -- which is what the error says.
    """
    roles = []
    for message in messages:
        role = getattr(message, 'role', None)
        if role == 'user':
            continue
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, 'type', None) == 'image_url':
                roles.append(str(role))
                break
    return roles



class VLMVisionStrategy:
    """Strategy for VLM requests with images.

    Renders the conversation with its images (``prepare_vlm_inputs``),
    builds the pixel tensors and the expanded prompt with mlx-vlm's
    ``prepare_inputs``, and hands the request to ``vlm_engine`` -- the same
    engine every MLX request runs on, prefix cache included (plan W10).

    Vision feature caching: when a model supports encode_image(), vision
    encoder outputs are cached by the request's image list and reach the
    embedding step only (``embed_extras``), so a turn with an unchanged image
    list skips the vision tower.
    """

    def __init__(self, model_config=None, template_info=None, model_id=None,
                 context_length=None, owner=None):
        self.owner = owner  # the provider (APC store, stop set), read per call
        self.model_config = model_config or {}
        self.template_info = template_info  # how history thinking is rendered
        self.model_id = model_id
        self.context_length = context_length  # the provider's, for the over-length guard
        self._batch_vision_processor = None
        self._vision_cache = VisionFeatureCache(max_entries=20)

    def generate(self, request: ChatRequest, effective_request: dict, model, processor, abort_event: AbortEvent | None = None) -> Generator:
        tokenizer = getattr(processor, "tokenizer", processor)
        sampler, processors = build_sampler(tokenizer, effective_request)

        # Per-request peak-memory scoping, and it has to start HERE: image
        # encoding is part of this request, so the engine is told not to
        # reset again (reset_peak=False).
        mx.reset_peak_memory()

        # Initialize batch vision processor for parallel image loading
        if self._batch_vision_processor is None:
            self._batch_vision_processor = BatchVisionProcessor(max_workers=4)

        # Prepare VLM inputs: extract images, format prompt with chat template
        # reasoning_effort rides the VISION path too. Without it the setting
        # worked on a text turn and silently reverted to the template default
        # the moment an image was attached -- same model, same conversation.
        #
        # Continuation takes the SAME two shapes as the text path
        # (UnifiedTextStrategy.build_prompt): a content prefill leaves the
        # final turn open through the template, and a mid-thought resume
        # renders everything BEFORE the message being resumed as a fresh
        # generation prompt with thinking on, then appends the partial trace
        # after the family's opener. Dropping that final message costs no
        # image: it is an assistant turn, and _non_user_image_roles has
        # already refused media anywhere but a user turn.
        resume = _thinking_resume(request)
        images, formatted_prompt, _, image_urls = self._prepare_vlm_inputs_parallel(
            request.messages[:-1] if resume is not None else request.messages,
            processor, model.config,
            enable_thinking=True if resume is not None else _resolve_enable_thinking(effective_request),
            depth=_depth_kwargs(effective_request, self.template_info),
            continue_final_message=request.is_continuation() and resume is None,
        )
        if resume is not None:
            formatted_prompt = _append_thinking_resume(formatted_prompt, resume, self.template_info)

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

        # What the EMBEDDING step takes beside ids/pixels/mask.
        extras = dict(extra_kwargs)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        # Vision feature caching: reuse cached vision encoder outputs across turns.
        # Follows mlx-vlm's generate pattern (now mlx_vlm/generate/ package):
        # - If model has encode_image(), we can compute and cache vision features
        # - cached_image_features kwarg bypasses the vision tower in the model's
        #   get_input_embeddings() method
        has_encode_image = hasattr(model, 'encode_image')
        cache_key = image_urls if image_urls else None
        if has_encode_image and pixel_values is not None:
            # Try URL-based key first; fall back to pixel content hash for
            # base64/PIL images that don't have a stable URL.
            cached_features = self._vision_cache.get(cache_key, pixel_values=pixel_values)
            if cached_features is not None:
                extras["cached_image_features"] = cached_features
                logging.info("[VLM VISION] Using cached vision features (skipping vision encoder)")
            else:
                # Compute and cache vision features separately
                with wired_limit(model, [generation_stream]):
                    features = model.encode_image(pixel_values)
                    mx.async_eval(features)
                self._vision_cache.put(cache_key, features, pixel_values=pixel_values)
                extras["cached_image_features"] = features
                logging.info("[VLM VISION] Computed and cached vision features")

        prompt_token_count = int(input_ids.shape[1])
        if self.context_length and prompt_token_count > self.context_length:
            raise InvalidGenerationRequest(
                f"Prompt is {prompt_token_count} tokens; {self.model_id or 'this model'} "
                f"has a context of {self.context_length} tokens. Shorten the "
                f"conversation, remove an image, or send a smaller one.")

        # The whole request runs on mlx-vlm's engine (plan W10, A2): the
        # prompt, images included, is prefilled there in chunks and restored
        # from APC across turns. The cached image features reach the
        # embedding step only (embed_extras), never the generator.
        embed_extras = {}
        if "cached_image_features" in extras:
            embed_extras["cached_image_features"] = extras["cached_image_features"]
        raw = {**inputs, "input_ids": input_ids}
        yield from vlm_engine.generate(
            model=model, processor=processor,
            apc_manager=getattr(self.owner, "_apc", None),
            input_ids=input_ids, raw_inputs=raw, embed_extras=embed_extras,
            sampler=sampler, processors=processors,
            stop_tokens=getattr(self.owner, "_stop_tokens", ()),
            max_tokens=effective_request['max_tokens'],
            greedy=float(effective_request.get('temperature') or 0.0) == 0.0,
            abort_event=abort_event,
            continuing=request.is_continuation(),
            prefill_step_size=effective_request.get('prefill_step_size'),
            model_id=self.model_id,
            # Reset at the top of this method: image encoding is part of
            # this request's peak.
            reset_peak=False,
            detokenizer=(self.owner.streaming_detokenizer(request.is_continuation())
                         if self.owner is not None else None),
            thinking_budget=_thinking_budget_criteria(
                request, effective_request, self.template_info,
                getattr(processor, "tokenizer", processor)),
        )

    def _prepare_vlm_inputs_parallel(self, messages: List, processor, config,
                                     enable_thinking=None, depth=None,
                                     continue_final_message: bool = False) -> Tuple[List[Image.Image], str, bool, List[str]]:
        """Prepare VLM inputs with parallel image loading. Delegates to standalone function."""
        from .common.vlm_inputs import prepare_vlm_inputs_parallel
        return prepare_vlm_inputs_parallel(
            messages, processor, config, self._batch_vision_processor,
            vlm_apply_chat_template, enable_thinking=enable_thinking,
            depth=depth, template_info=self.template_info,
            continue_final_message=continue_final_message,
        )


class DiffusionStrategy:
    """Strategy for masked-diffusion LMs (diffusion_gemma and friends).

    Diffusion models are NOT autoregressive. They denoise a fixed-length
    canvas (``config.canvas_length``) over N steps instead of extending a
    sequence one token at a time. The autoregressive engine every other MLX
    request takes (``vlm_engine``) forwards the prompt, samples the last
    position, and repeats. Handed a diffusion checkpoint that samples one meaningless
    token which lands on an EOS almost immediately, so the request completes
    with zero emitted tokens and the client renders an empty reply. That is
    the bug this strategy exists to fix.

    Three things differ from the AR strategies:

    1. The engine is mlx-vlm's ``stream_diffusion_generate``, called DIRECTLY
       rather than via ``stream_diffusion_generate_from_kwargs``. The
       from_kwargs wrapper routes through ``_stream_model_diffusion_generate``,
       which collects the ENTIRE generation into a list (``on_result=emit``)
       before yielding any of it -- correct for a CLI, but on a streaming
       server it turns every request into full-duration dead air followed by
       a burst. The underlying generator streams token-by-token.
    2. The prompt cache is bypassed entirely -- this strategy never
       touches the global cache manager. The denoising loop owns its own KV
       cache (``diffusion_prefill_cache`` / ``model.make_cache``), and
       heylook's prompt cache assumes AR trim-to-a-prefix semantics that a
       canvas rewrite violates.
    3. Generation parameters come from the checkpoint's own
       ``config.generation_config`` -- canvas length, denoising steps, the
       entropy-bound sampler, confidence/stability thresholds and the linear
       temperature schedule are all resolved inside the engine. heylook's AR
       sampler cascade (GLOBAL_SAMPLER_FLOOR, samplers, presets) deliberately
       does NOT apply: top_p/top_k/repetition_penalty have no meaning for a
       canvas update. Only the two knobs a caller set EXPLICITLY on the
       request -- temperature and max_tokens -- override the checkpoint.

    Text is yielded RAW (``skip_special_token_ids=[]``) for the same reason
    VLMVisionStrategy yields its first token raw: gemma-4 opens reasoning with
    the structural marker ``<|channel>thought``, and the reasoning parser must
    see it. Parsers strip non-structural specials from routed text themselves.
    """

    def __init__(self, model_config=None, template_info=None):
        self.model_config = model_config or {}
        self.template_info = template_info
        self._batch_vision_processor = None

    def generate(self, request: ChatRequest, effective_request: dict, model, processor, abort_event: AbortEvent | None = None) -> Generator:
        # Imported lazily and from the defining module: `is_diffusion_model`
        # and `stream_diffusion_generate` are not in mlx_vlm.generate.__all__
        # (only the buffering from_kwargs wrapper is reachable there).
        from mlx_vlm.generate.diffusion import stream_diffusion_generate
        from mlx_vlm.generate import (
            generation_stream as vlm_generation_stream,
            wired_limit as vlm_wired_limit,
        )

        tokenizer = getattr(processor, "tokenizer", processor)

        if self._batch_vision_processor is None:
            self._batch_vision_processor = BatchVisionProcessor(max_workers=4)

        # Same prompt construction as the VLM paths -- diffusion_gemma is a
        # VLM, so images ride the ordinary mlx-vlm input pipeline and the
        # denoising loop takes pixel_values/mm_token_type_ids directly.
        from .common.vlm_inputs import prepare_vlm_inputs_parallel
        images, formatted_prompt, has_images, _ = prepare_vlm_inputs_parallel(
            request.messages, processor, model.config, self._batch_vision_processor,
            vlm_apply_chat_template,
            enable_thinking=_resolve_enable_thinking(effective_request),
            depth=_depth_kwargs(effective_request, self.template_info),
            template_info=self.template_info,
        )

        inputs = vlm_prepare_inputs(
            processor,
            images=images if images else None,
            prompts=formatted_prompt,
            image_token_index=getattr(model.config, 'image_token_index', None),
        )
        input_ids = inputs["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        # Checkpoint defaults unless the CALLER set the knob. effective_request
        # carries the AR sampler floor (its temperature and max_tokens),
        # which would silently override the checkpoint's own schedule and
        # 256-token canvas -- so read the raw request instead. max_tokens=0
        # makes the engine fall back to generation_config.max_new_tokens.
        temperature = request.temperature if request.temperature is not None else 0.0
        max_tokens = request.max_tokens if request.max_tokens is not None else 0

        logging.info(
            "[MLX DIFFUSION] canvas=%s images=%d temp=%s max_tokens=%s | Model: %s",
            getattr(model.config, 'canvas_length', '?'),
            len(images) if images else 0,
            temperature,
            max_tokens or "checkpoint",
            getattr(model.config, 'model_type', 'unknown'),
        )

        # Per-request peak-memory scoping (same contract as vlm_engine);
        # the engine reports mx.get_peak_memory() on every chunk.
        mx.reset_peak_memory()

        stream = stream_diffusion_generate(
            model,
            processor,
            tokenizer,
            input_ids,
            inputs.get("pixel_values"),
            inputs.get("attention_mask"),
            max_tokens=max_tokens,
            # Raw text: the reasoning parser needs the structural markers.
            skip_special_token_ids=[],
            temperature=temperature,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        # stream_diffusion_generate manages mx.stream() itself but NOT the
        # wired limit -- from_kwargs owned that. Use mlx-vlm's stream here, not
        # heylook's: wired_limit synchronizes the stream it is handed on exit,
        # and the denoising loop ran on mlx-vlm's.
        try:
            with vlm_wired_limit(model, [vlm_generation_stream]):
                for chunk in stream:
                    if abort_event and abort_event.is_set():
                        logging.info("Generation aborted during diffusion denoising")
                        return
                    # Draft chunks are the in-progress canvas for terminal
                    # redraw (text="", canvas in draft_text). Never content.
                    # Not requested (diffusion_show_unmasking defaults off), so
                    # this is belt-and-braces.
                    if chunk.is_draft:
                        continue
                    yield GenerationChunk.from_engine(chunk)
        finally:
            stream.close()




class MLXProvider(BaseProvider):
    """
    MLX Provider: every model loads with mlx-vlm and generates through
    ``vlm_engine``. Strategies, built once at load, pick the request's shape:
    text (template path by ``is_vlm``), vision (images through mlx-vlm's
    ``prepare_inputs``), or a masked-diffusion model's denoising loop.
    """

    provider_name = "mlx"

    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None
        self.processor = None
        # mlx-vlm's prefix cache for this model (vlm_engine), its mode, the
        # stop set resolved ONCE at load, and the last request's context.
        self._apc = None
        self._apc_mode = None
        self._stop_tokens = frozenset()
        self._context_used = 0
        # Served with vision? Declared modalities + whether mlx-vlm registers
        # the model_type; the capability report reads the same resolver. A
        # vision model mlx-vlm can't run as a VLM is served as text rather
        # than crashing at load. See common/loader_routing.py.
        self.is_vlm = resolve_serves_vision(
            self.config, lambda: read_model_type(self.config.get("model_path", "")))
        # Masked-diffusion checkpoint (canvas denoising, not AR). Decided at
        # load time from the loaded model -- see _detect_diffusion.
        self.is_diffusion = False

        # Pre-compile generation strategies (avoids runtime branching)
        self._strategies = {}

        # Vendor sampling layer (generation_config.json), lazy-read + cached
        # in _apply_model_defaults. None = not read yet; {} = none found.
        self._vendor_sampling = None

        # Serialize generation FIFO across concurrent requests. Process-global
        # (shared across all providers -- one GPU); queues in arrival order (no
        # preemption); check_capacity() rejects with ModelBusyError (-> 503)
        # once max_queue_depth are waiting. The abort signal is NOT stored here:
        # it is per-request (created/passed by create_chat_completion) so one
        # client's disconnect can't abort another client's generation.
        self._gen_gate = _get_generation_gate(
            int(self.config.get("max_queue_depth", 8))
        )

        # (_active_generations / _active_lock now live on BaseProvider, so the
        # router can ask EVERY provider the same question before a teardown --
        # they were MLX-only, which made a guard built on them cover half the
        # app. Re-initialising them here would be harmless but misleading.)

    def streaming_detokenizer(self, continuing: bool):
        """A reset streaming detokenizer for one request: mlx-lm's, vendored
        (primed at load by detokenizer_source), which streams per token where
        mlx-vlm's holds space-free text until the end, with the continuation
        seam seeded (continuation_detokenizer)."""
        tok = self.get_tokenizer()
        if tok is None:
            return None
        source = detokenizer_source(tok)
        with continuation_detokenizer(source, continuing):
            return source.detokenizer

    def describe_observed(self):
        """The engine contract's observed half: the template body installed
        at load, and the prefix cache as built at load (vlm_engine)."""
        from .contract import Fact, Observed

        cache = {}
        if self._apc is not None:
            checkpoints = self._apc_mode == "exact"
            cache["reuse_mode"] = Fact(
                value="checkpoints" if checkpoints else (self._apc_mode or "blocks"),
                provenance="observed",
                source=("from this model's cache layers: hybrid and sliding-window "
                        "models restore whole-cache checkpoints taken during prefill; "
                        "plain KV models reuse hashed blocks"))
            cache["image_reuse"] = Fact(
                value=checkpoints, provenance="observed",
                source=("checkpoint models restore image turns; block models "
                        "discard in-memory blocks that contain an image, and the "
                        "disk tier that would serve them is off"))
            cache["memory_budget_bytes"] = Fact(
                value=int(self._apc.memory_max_bytes), provenance="observed",
                source="mlx-vlm's automatic prefix-cache budget, sized from the Metal working set")
        return Observed(loaded_template=self.loaded_chat_template, cache=cache)

    def load_model(self):
        model_path = self.config['model_path']

        logging.info(f"Loading model with mlx-vlm from: {model_path}")

        try:
            # Every MLX model, text-only included, loads and runs on mlx-vlm's
            # engine (plan W10, outcome A2).
            self.model, self.processor = self._load_vlm_with_fallback(model_path)

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

            # Prime the detokenizer source HERE, where model_path is known:
            # streaming_detokenizer hands vlm_engine this source's streaming
            # detokenizer, and one built without the path takes the naive
            # detokenizer (quadratic per line, read-only `text` -- see
            # detokenizer_source).
            gen_tokenizer = self.get_tokenizer()
            if gen_tokenizer is not None:
                detokenizer_source(gen_tokenizer, model_path)
            # The stop set, resolved ONCE here and checked in vlm_engine's own
            # loop -- never added per request to the shared tokenizer.
            from .common.stop_tokens import resolve_stop_tokens
            self._stop_tokens = frozenset(resolve_stop_tokens(self.get_tokenizer()))

            # The context window, from the ONE resolver the admin row and
            # /v1/models read -- so the ceiling a client is shown is the one
            # vlm_engine refuses an over-length prompt against.
            self.context_length = model_context_length(
                "mlx", model_path, override=self.config.get("context_length"))

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
                force=should_force_install(
                    self._template_info,
                    self.config.get("chat_template_source"),
                ),
                # The vision render reads the PROCESSOR's template first --
                # see install_chat_template. Passing it is what makes an
                # override apply on the mlx-vlm path, not just mlx-lm's.
                processor=self.processor,
            )
            # What this process is actually rendering with, READ BACK off the
            # object rather than assumed from the ladder. Under auto,
            # install_chat_template returns early when the tokenizer already
            # has a template, so the ladder's answer can be a body this
            # process never installed -- asserting it would make `stale` say
            # "up to date" about a template the model is not using, which is
            # the same read-it/report-it/never-use-it failure the processor
            # targeting fixed for the override case. Holder chosen the way
            # mlx-vlm's get_chat_template chooses it: the processor when it
            # carries one, else the tokenizer.
            holder = (self.processor
                      if getattr(self.processor, "chat_template", None)
                      else tok)
            self.loaded_chat_template = getattr(holder, "chat_template", None) or None
            # Warn only when NOTHING can render: no install happened and
            # neither holder carries a template.
            has_template = (
                installed
                or bool(getattr(tok, "chat_template", None))
                or bool(getattr(self.processor, "chat_template", None))
            )
            if not has_template:
                logging.warning(
                    "Model %s has NO chat template (no chat_template.jinja, no "
                    "tokenizer_config chat_template, no chat_template.json). "
                    "Chat requests will fail -- add a template file to the model "
                    "folder or set chat_template_source in models.toml.",
                    self.model_id,
                )

            # Diffusion detection must happen after load: the trait lives on
            # the loaded model's config, and the predicate is mlx-vlm's own so
            # heylook's routing can't drift from the engine's.
            self.is_diffusion = self._detect_diffusion()
            if self.is_diffusion:
                logging.info(
                    "diffusion: %s is a masked-diffusion checkpoint "
                    "(canvas_length=%s) -- routing to the denoising engine, "
                    "not autoregressive generation",
                    self.model_id,
                    getattr(self.model.config, "canvas_length", "?"),
                )

            if not self.is_diffusion:
                from mlx_vlm import apc as _apc
                self._apc = vlm_engine.make_apc_manager()
                self._apc_mode = _apc.APCCoordinator(
                    self._apc, self.model.language_model).legacy_mode

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

        # Pre-compile generation strategies after model loading
        self._compile_strategies()

    def _detect_diffusion(self) -> bool:
        """Whether the loaded model needs the diffusion denoising engine.

        Delegates to mlx-vlm's own predicate rather than matching model_type,
        so a new diffusion architecture is picked up without a heylook change.
        It keys on the engine-driven trait (``config.canvas_length``, or a
        ``mask_token_id``) plus a callable ``language_model.generate``.

        Best-effort: a predicate failure must degrade to the AR path (today's
        behaviour) rather than fail the load.
        """
        try:
            from mlx_vlm.generate.diffusion import is_diffusion_model
            return bool(is_diffusion_model(self.model))
        except Exception:
            logging.debug(
                f"diffusion detection failed for {self.model_id}; assuming autoregressive",
                exc_info=True,
            )
            return False

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
            model_id=self.model_id,
            model_config=self.config,
            is_vlm=self.is_vlm,
            template_info=getattr(self, "_template_info", None),
            context_length=self.context_length,
            owner=self,
        )
        if self.is_vlm:
            self._strategies['vision'] = VLMVisionStrategy(
                model_config=self.config, template_info=getattr(self, "_template_info", None),
                model_id=self.model_id, context_length=self.context_length, owner=self)
        # Diffusion handles BOTH its text and vision requests -- the denoising
        # loop takes pixel_values directly, so there is no separate vision
        # split.
        if self.is_diffusion:
            self._strategies['diffusion'] = DiffusionStrategy(
                model_config=self.config, template_info=getattr(self, "_template_info", None))

    def _detect_images_optimized(self, messages: List) -> bool:
        """Single-pass scan for images with early termination."""
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == 'image_url':
                        return True
        return False

    @property
    def thinking_capable(self) -> bool:
        """The served thinking capability, as capabilities.py derives it for
        MLX: the config's explicit enable_thinking, else the template probe
        (``supports_enable_thinking``). Same answer before and after load --
        the probe reads the template FILE, which is what the capability
        inference reads too."""
        if self.config.get("enable_thinking"):
            return True
        info = getattr(self, "_template_info", None)
        if info is None:
            from .common.template_info import read_template_info
            try:
                info = read_template_info(Path(self.config.get("model_path", "")),
                                          self.config.get("chat_template_source"))
            except Exception:
                return False
        return bool(getattr(info, "supports_enable_thinking", False))

    def render_prompt(self, request: ChatRequest) -> str:
        """The exact prompt string for ``request`` -- the text strategy's own
        builder, so a preview cannot drift from what generation feeds the
        model. Images in history are NOT represented (the vision strategy
        renders through mlx-vlm's prepare_inputs, which has no text-only
        rendering); the text template with the text parts is what shows.
        Templating only: no forward pass, no gate, no thread pinning."""
        if self.processor is None:
            raise GenerationFailed(f"Model '{self.model_id}' is not loaded")
        effective_request = self._apply_model_defaults(request)
        strategy = self._strategies.get('text')
        if strategy is None:
            raise GenerationFailed(f"Model '{self.model_id}' has no text strategy")
        return strategy.render_prompt(request, effective_request, self.model, self.processor)

    def _apply_model_defaults(self, request: ChatRequest) -> dict:
        """Effective request config: the shared cascade + MLX runtime fields.

        Layer semantics live in ``samplers.resolve_effective_sampling`` (one
        implementation for every provider -- do not re-inline a cascade
        here). This wrapper adds the two MLX-only pieces: the cached vendor
        layer read and the runtime-default (cache/spec-decode) fields.
        """
        # Vendor layer source: the model's own generation_config.json, read
        # once and cached on the provider.
        if self._vendor_sampling is None:
            self._vendor_sampling = load_vendor_sampling(self.config.get('model_path', ''))

        merged_config = resolve_effective_sampling(
            request, self.config, vendor=self._vendor_sampling,
            thinking_capable=self.thinking_capable,
            thinking=detect_thinking(getattr(getattr(self, "_template_info", None),
                                             "chat_template", None)))

        # Cache + speculative-decoding fields tagged with
        # json_schema_extra={"is_runtime_default": True} on MLXModelConfig.
        # Adding a new tagged field auto-propagates here. Disjoint from the
        # sampler keys, so ordering vs the cascade is immaterial.
        for key in MLX_RUNTIME_DEFAULT_FIELDS:
            if key not in merged_config and key in self.config:
                merged_config[key] = self.config[key]

        return merged_config

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
            (which sets it on client disconnect); internal callers (RLM)
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

                    if _has_audio_parts(request.messages):
                        raise InvalidGenerationRequest(
                            f"Model '{self.model_id}' is served by the MLX provider, "
                            f"which does not support audio input (audio towers are "
                            f"skipped at load). Use a gguf model for audio."
                        )

                    non_user_image_roles = _non_user_image_roles(request.messages)
                    if non_user_image_roles:
                        raise InvalidGenerationRequest(
                            f"Model '{self.model_id}' is served by the MLX provider, "
                            f"which can only place an image on a user turn -- an image "
                            f"on a "
                            f"{', '.join(sorted(set(non_user_image_roles)))} message "
                            f"would be silently moved to the latest user message and "
                            f"described to the model as if it had arrived there. Use a "
                            f"gguf model to put an image on that turn."
                        )

                    # A denoising engine has no turn to leave open at all.
                    # Refuse rather than silently restart the message. (Image
                    # history continues fine: VLMVisionStrategy leaves the
                    # final turn open through the same template kwarg the
                    # text path uses.)
                    if request.is_continuation() and self.is_diffusion:
                        raise InvalidGenerationRequest(
                            "continue_final_message is not supported on "
                            "diffusion models -- a denoising engine has no "
                            "open turn to continue."
                        )

                    # Diffusion first: the AR text/vision split does not apply
                    # to a denoising engine, which takes images inline.
                    if self.is_diffusion:
                        strategy = self._strategies['diffusion']
                        logging.info(f"[MLX STRATEGY] Diffusion path (images={has_images}) | Model: {self.model_id}")
                    elif self.is_vlm and has_images:
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
                                chunk.queue_wait_ms = queue_wait_ms
                                tagged = True
                            if getattr(chunk, "prompt_tokens", 0):
                                self._context_used = chunk.prompt_tokens + (chunk.generation_tokens or 0)
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
        """Max context window: the shared resolver's answer (config.json,
        read once at load) first, the loaded model's config object as the
        fallback for a checkpoint whose files did not say."""
        if self.context_length:
            return self.context_length
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
        """The last request's context (prompt plus generated tokens)."""
        return self._context_used

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

        # Diffusion checkpoints must prime the denoising loop, not the AR
        # decode -- priming the path real requests DON'T take is how the VLM
        # LanguageModelOutput bug stayed hidden (see this docstring's warning).
        if self.is_diffusion:
            self._warmup_diffusion(prompt_tokens)
            return

        # Through the SAME engine real requests use (vlm_engine), without the
        # prefix cache, so warmup tokens never land in it.
        t0 = time.time()
        try:
            ids = mx.array([prompt_tokens])
            sampler, processors = build_sampler(tok, {"temperature": 0.0})
            for _ in vlm_engine.generate(
                model=self.model, processor=self.processor, apc_manager=None,
                input_ids=ids, raw_inputs={"input_ids": ids},
                sampler=sampler, processors=processors,
                stop_tokens=self._stop_tokens, max_tokens=4, greedy=True,
                model_id=self.model_id,
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

    def _warmup_diffusion(self, prompt_tokens: list) -> None:
        """JIT-prime the denoising engine. Best-effort, same contract as warmup().

        Deliberately mirrors DiffusionStrategy's engine call rather than going
        through the strategy: no ChatRequest to synthesize, and warmup tokens
        must not touch conversation state. One short canvas is enough to
        compile the decoder forward, the softcap, and the canvas sampler.
        """
        from mlx_vlm.generate.diffusion import stream_diffusion_generate
        from mlx_vlm.generate import (
            generation_stream as vlm_generation_stream,
            wired_limit as vlm_wired_limit,
        )

        t0 = time.time()
        try:
            input_ids = mx.array([prompt_tokens])
            stream = stream_diffusion_generate(
                self.model,
                self.processor,
                self.get_tokenizer(),
                input_ids,
                None,
                None,
                max_tokens=4,
                skip_special_token_ids=[],
                temperature=0.0,
            )
            try:
                with vlm_wired_limit(self.model, [vlm_generation_stream]):
                    for _ in stream:
                        pass
            finally:
                stream.close()
        except Exception:
            logging.warning(
                f"warmup: {self.model_id} failed to prime the diffusion engine; "
                f"first request will pay JIT compilation cost. Continuing without warmup.",
                exc_info=True,
            )
            return
        logging.info(f"warmup: {self.model_id} primed (diffusion) in {(time.time() - t0) * 1000:.0f}ms")

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
        """Clear this model's prefix cache (a fresh, empty APC store)."""
        try:
            if self._apc is not None:
                self._apc = vlm_engine.make_apc_manager()
            logging.info(f"Cleared prompt cache for {self.model_id}")
            return True
        except Exception as e:
            logging.warning(f"Failed to clear cache for {self.model_id}: {e}")
            return False

    def unload(self, *, drain: bool = True):
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
        if not drain:
            # Called from __del__, which must not block: the loop below polls
            # for up to 30s on whatever thread the GC happened to fire on.
            #
            # Skipping it does NOT keep the model loaded. A destructor that
            # returns early retains nothing -- the weights are released as it
            # returns, whatever it decided (see BaseProvider.__del__). What
            # the skip actually avoids is the poll, and the engine teardown at
            # the tail of this method (`gc.collect()` + `mx.clear_cache()`),
            # neither of which belongs on an arbitrary GC thread while another
            # may be mid-decode.
            #
            # Only THIS provider's counter is read. `generation_queue_stats()`
            # reports the PROCESS-GLOBAL gate, so its waiters can belong
            # entirely to another model -- and a warning naming this model for
            # someone else's traffic sends the reader after a bug that is not
            # there.
            with self._active_lock:
                active = self._active_generations
            if active:
                logging.warning(
                    f"{self.model_id} was garbage-collected with {active} active "
                    "generation(s). Its weights are released either way -- a "
                    "destructor cannot hold them -- so this is a dropped reference "
                    "or a leaked counter to go and find, not a teardown to tune."
                )
                return

        logging.info(f"Unloading MLX model: {self.model_id}")

        # Wait for active generations AND queued waiters to drain.
        max_wait = 30  # seconds
        start = time.time()
        while drain:
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
        self._apc = None

        # Clean up models
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if not drain:
            # THE DESTRUCTOR STOPS HERE. Dropping references above is free --
            # deallocation happens anyway -- but `gc.collect()` and
            # `mx.clear_cache()` are ENGINE calls, and this runs on whatever
            # thread the GC fired on.
            #
            # It matters most in the case this branch reaches with `active ==
            # 0`: the active counter is decremented BEFORE `gate.release()`
            # admits the next waiter, so a woken waiter can be starting a
            # decode exactly here (tests/unit/test_unload_waiter_safety.py,
            # and CLAUDE.md's "never gate teardown on actives alone"). The
            # deliberate path answers that by WAITING; a destructor cannot
            # wait, so it declines to make the calls instead.
            #
            # The cost is the Metal buffer cache going unswept on a path that
            # should not be taken at all -- a provider collected without a
            # deliberate unload. Every real teardown (LRU evict, clear_cache,
            # explicit unload, idle unload) passes drain=True and sweeps.
            return

        gc.collect()
        mx.clear_cache()
