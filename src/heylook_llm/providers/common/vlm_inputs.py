# src/heylook_llm/providers/common/vlm_inputs.py
"""
Standalone VLM input preparation for vision requests.

Extracts image URLs from messages, loads images in parallel, and formats
the prompt using the VLM's chat template. Handles both ContentPart objects
and dict formats, with error recovery for template failures.

Previously embedded as VLMVisionStrategy._prepare_vlm_inputs_parallel().
"""

from typing import List, Tuple

from PIL import Image


def thinking_for_template(msg_dict: dict, template_info=None) -> dict:
    """Hand an assistant message's ``thinking`` to the chat template the way
    THIS template can take it (v1.79.63):

    - the template reads ``reasoning_content`` -> pass it under that key and
      let the template render (or drop) it: Qwen3.8 keeps every turn's
      reasoning, gemma-4 only tool-call turns' and strips channel markers
      from content;
    - a ``<think>``-marker template that does not -> reconstruct the tags
      into content (the pre-v1.79.63 behaviour, and still right for Qwen3);
    - anything else -> drop it. Baking ``<think>`` text into a gemma or
      harmony prompt is noise the model never emitted.

    ``template_info=None`` (no probe available) keeps the old reconstruction,
    so a caller without a probe gets the pre-v1.79.63 behaviour. The ``thinking``
    key never reaches the template either way.
    """
    thinking = msg_dict.get('thinking')
    if not thinking or msg_dict.get('role') != 'assistant':
        msg_dict.pop('thinking', None)
        return msg_dict
    if template_info is None or getattr(template_info, 'has_thinking_markers', False) \
            and not getattr(template_info, 'reads_reasoning_content', False):
        return _reconstruct_thinking(msg_dict)
    msg_dict.pop('thinking', None)
    if getattr(template_info, 'reads_reasoning_content', False):
        msg_dict['reasoning_content'] = thinking
    return msg_dict


def _reconstruct_thinking(msg_dict: dict) -> dict:
    """Reconstruct model-specific thinking tags in assistant message content.

    If an assistant message carries a 'thinking' field, prepend <think>...</think>
    tags so the tokenizer sees the full thinking block as part of the content.
    The 'thinking' key is removed from the dict so it does not leak into the
    chat template.
    """
    thinking = msg_dict.pop('thinking', None)
    if thinking and msg_dict.get('role') == 'assistant':
        content = msg_dict.get('content', '')
        msg_dict['content'] = f"<think>\n{thinking}\n</think>\n{content}"
    return msg_dict


def content_for_template(text: str, num_images: int):
    """Content for ONE message, in the shape mlx-vlm allocates media from.

    A message carrying images travels as BLOCK-form content with one bare
    ``{"type": "image"}`` marker per image, because mlx-vlm's
    ``apply_chat_template`` attributes media PER MESSAGE by counting explicit
    markers (``_content_media_count``) and dumps only what it CANNOT attribute
    onto the last user turn. Handing it a flattened string plus a bare
    ``num_images=`` total -- what this function did until v2.0.15 -- attributed
    nothing, so EVERY image in a conversation rendered its marker on the final
    user message: an image attached in turn 1 was presented to the model as if
    it had arrived in the latest turn, and two images from different turns
    arrived adjacent, in load order. The bytes were always in context; what was
    wrong was which turn they were announced in.

    The marker is BARE on purpose. mlx-vlm re-derives each message's content
    from the extracted text plus the COUNT, emitting markers in that model's
    own order (llava appends, qwen3_5 prepends), so this function never has to
    know the per-model shape, and a multi-MB data URI never enters the template
    call.

    Text-only messages keep travelling as a plain string, so a conversation
    with no images renders byte-identically to before this change.
    """
    if num_images <= 0:
        return text
    return [{"type": "image"}] * num_images + [{"type": "text", "text": text}]


def system_prefix_tokens(system_text: str, render, tokenize) -> list[int] | None:
    """The tokens every conversation that opens with this system message
    starts with, as the template in force renders them.

    The template is rendered twice, the system message followed by two
    different one-word user turns, and cut where the two token lists part.
    That is the system turn plus whatever the template puts before a user's
    words, found without knowing any family's role markers. The prefix cache
    snapshots there, so a new conversation with the same system prompt
    restores it (``vlm_engine.install_capture_policy``). None when the two
    renders share nothing."""
    renders = [render([{"role": "system", "content": system_text},
                       {"role": "user", "content": word}]) for word in ("a", "b")]
    a, b = (tokenize(r) for r in renders)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return list(a[:n]) or None


def carry_message_extras(source: list, rebuilt: list) -> list:
    """``rebuilt`` (mlx-vlm's per-model message shapes) with every key of
    the matching ``source`` message it lacks put back.

    ``mlx_vlm.prompt_utils.apply_chat_template(return_messages=True)`` keeps
    only role and content, so ``reasoning_content`` (and anything else a
    template reads off a message) never reached the template on the VLM
    path. Same count and order in, same out; if mlx-vlm ever changes the
    count, the messages are returned untouched rather than mismatched.
    """
    if len(source) != len(rebuilt):
        return rebuilt
    for src, dst in zip(source, rebuilt):
        if not isinstance(src, dict) or not isinstance(dst, dict):
            continue
        for key, value in src.items():
            if key not in dst:
                dst[key] = value
    return rebuilt


def continue_from_generation_prompt(prompt, messages, render_generation_prompt):
    """A content continuation, rebuilt on the prompt the reply was generated
    under when the template's history render lost part of it.

    ``continue_final_message`` re-renders the partial reply as a HISTORY
    turn. Where a template's generation prompt adds something after the role
    header that its history render does not reproduce -- gemma-4 with
    thinking off opens and closes an empty thought channel there -- the model
    is asked to extend a turn shaped unlike the one it wrote, and the
    continuation degrades into repetition (fixed-seed A/B, 2026-09-23, record
    in internal/claude/w2/: with the channel restored it stays clean). So: if the
    generation prompt for the history before the reply EXTENDS everything
    the continuation render put before the reply's text, continue from the
    generation prompt plus that text -- exactly what the model saw. Anything
    else (the reply carries thinking, the render rewrote the text, a
    tokenized template) keeps the continuation render unchanged."""
    content = messages[-1].get("content") if messages else None
    if not isinstance(prompt, str) or not isinstance(content, str) or not content:
        return prompt
    if not prompt.endswith(content):
        return prompt
    head = prompt[: len(prompt) - len(content)]
    generation = render_generation_prompt(messages[:-1])
    if isinstance(generation, str) and generation != head and generation.startswith(head):
        return generation + content
    return prompt


def prepare_vlm_inputs_parallel(
    messages: List,
    processor,
    config,
    batch_vision_processor,
    vlm_apply_chat_template_fn,
    enable_thinking=None,
    reasoning_effort=None,
    template_info=None,
    continue_final_message: bool = False,
) -> Tuple[List[Image.Image], str, bool, List[str]]:
    """Prepare VLM inputs with parallel image loading.

    Args:
        messages: List of ChatMessage objects (with .content, .role, etc.)
        processor: VLM processor (for tokenizer and chat template)
        config: Model config (model_type, etc.)
        batch_vision_processor: BatchVisionProcessor for parallel image loading
        vlm_apply_chat_template_fn: Function to apply VLM chat template
        enable_thinking: Template thinking toggle forwarded to the template
            function (None = leave the template to its default)
        continue_final_message: leave the final message's turn OPEN so
            generation finishes it.

    A template that cannot render these messages RAISES (whatever
    ``vlm_apply_chat_template_fn`` raises; it maps a continuation failure to a
    400 itself). Until v2.0.58 this function caught every exception and walked
    a ladder instead -- the tokenizer's template with a fresh generation
    prompt, then a bare ``role: content`` join -- so a broken template produced
    a confident answer to a prompt the model was never trained on, with one
    ERROR line in the log to show for it. Checked before removing: no installed
    mlx-vlm-routed model reached either rung.

    Returns:
        Tuple of (images, formatted_prompt, has_images, image_urls)
    """
    image_urls = []
    text_messages = []
    has_images = False

    # First pass: collect image URLs and build text structure
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            text_parts = []
            # Per-message count, not just the running total: this is what lets
            # each image render its marker on the turn it was attached to.
            msg_images = 0

            # Parts are ContentPart OBJECTS: ChatMessage validates dicts into
            # them at construction, on every path that builds a request.
            for part in content:
                if part.type == 'text':
                    text_parts.append(part.text)
                elif part.type == 'image_url':
                    image_urls.append(part.image_url.url)
                    msg_images += 1
                    has_images = True

            # Combine text parts. `image_urls` is appended in message order and
            # the markers are placed in that same order, so the flat image list
            # the caller loads still lines up with the markers the template
            # renders.
            combined_content = " ".join(text_parts) if text_parts else ""
            msg_dict = {"role": msg.role,
                        "content": content_for_template(combined_content, msg_images)}
            # Prior thinking, the way this template takes it (see helper)
            if hasattr(msg, 'thinking') and msg.thinking:
                msg_dict = thinking_for_template({**msg_dict, 'thinking': msg.thinking}, template_info)
            text_messages.append(msg_dict)
        elif isinstance(content, str):
            msg_dict = {"role": msg.role, "content": content}
            if hasattr(msg, 'thinking') and msg.thinking:
                msg_dict = thinking_for_template({**msg_dict, 'thinking': msg.thinking}, template_info)
            text_messages.append(msg_dict)

    # Load all images in parallel
    if image_urls:
        try:
            images = batch_vision_processor.load_images_parallel(image_urls)
        except Exception as e:  # noqa: BLE001 - any unreadable image is the client's to fix
            from ..base import InvalidGenerationRequest
            raise InvalidGenerationRequest(f"An image in this request could not be read: {e}") from e
    else:
        images = []

    # Format prompt -- coerce a stray non-string SCALAR (some templates have
    # bugs with non-string content) without touching block-form content: a
    # blanket str() here would hand the template a Python repr of the block
    # list, image markers and all, which is how the media placement above
    # would silently stop working. Rebuilding each dict from role+content
    # (what this did before) also DROPPED any key thinking_for_template had
    # set; they survive this function now, but note that mlx-vlm rebuilds the
    # message itself for every model_type in its MODEL_CONFIG, so
    # reasoning_content still does not reach those templates. Fixing that is
    # an upstream change, not one available here.
    safe_messages = []
    for msg in text_messages:
        safe = dict(msg)
        if not isinstance(safe.get("role"), str):
            safe["role"] = str(safe.get("role"))
        if not isinstance(safe.get("content"), (str, list)):
            safe["content"] = str(safe.get("content"))
        safe_messages.append(safe)

    # vlm_apply_chat_template performs the enable_thinking None-guard itself
    formatted_prompt = vlm_apply_chat_template_fn(
        processor, config, safe_messages, num_images=len(images),
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        continue_final_message=continue_final_message,
    )
    if continue_final_message:
        # The final message is an assistant turn (no media), so the image
        # count is the same without it.
        formatted_prompt = continue_from_generation_prompt(
            formatted_prompt, safe_messages,
            lambda msgs: vlm_apply_chat_template_fn(
                processor, config, msgs, num_images=len(images),
                enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
                continue_final_message=False))

    return images, formatted_prompt, has_images, image_urls
