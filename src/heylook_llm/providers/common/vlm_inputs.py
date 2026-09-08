# src/heylook_llm/providers/common/vlm_inputs.py
"""
Standalone VLM input preparation for vision requests.

Extracts image URLs from messages, loads images in parallel, and formats
the prompt using the VLM's chat template. Handles both ContentPart objects
and dict formats, with error recovery for template failures.

Previously embedded as VLMVisionStrategy._prepare_vlm_inputs_parallel().
"""

import logging
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


def _flatten_for_log(content) -> str:
    """Block-form or string content as plain text, for the no-template fallback
    and for error logs. Never used to build a real prompt."""
    if isinstance(content, list):
        return " ".join(b.get("text") or "" for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return content if isinstance(content, str) else str(content)


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


def prepare_vlm_inputs_parallel(
    messages: List,
    processor,
    config,
    batch_vision_processor,
    vlm_apply_chat_template_fn,
    model=None,
    enable_thinking=None,
    reasoning_effort=None,
    template_info=None,
) -> Tuple[List[Image.Image], str, bool, List[str]]:
    """Prepare VLM inputs with parallel image loading.

    Args:
        messages: List of ChatMessage objects (with .content, .role, etc.)
        processor: VLM processor (for tokenizer and chat template)
        config: Model config (model_type, etc.)
        batch_vision_processor: BatchVisionProcessor for parallel image loading
        vlm_apply_chat_template_fn: Function to apply VLM chat template
        model: Optional model instance (unused currently, reserved for future)
        enable_thinking: Template thinking toggle forwarded to the template
            function (None = leave the template to its default)

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

            for part in content:
                # Handle both object and dict formats
                if hasattr(part, 'type'):
                    # Object format (ContentPart)
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        image_urls.append(part.image_url.url)
                        msg_images += 1
                        has_images = True
                elif isinstance(part, dict):
                    # Dict format
                    if part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    elif part.get('type') == 'image_url':
                        image_url = part.get('image_url', {})
                        if isinstance(image_url, dict):
                            url = image_url.get('url', '')
                        else:
                            url = image_url.url if hasattr(image_url, 'url') else ''
                        if url:
                            image_urls.append(url)
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
        images = batch_vision_processor.load_images_parallel(image_urls)
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

    try:
        # vlm_apply_chat_template performs the enable_thinking None-guard itself
        formatted_prompt = vlm_apply_chat_template_fn(
            processor, config, safe_messages, num_images=len(images),
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        logging.error(f"Chat template error: {e}")
        logging.error(f"Text messages: {text_messages}")
        # Fallback: apply the tokenizer's own chat template with string content
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        try:
            template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            formatted_prompt = tokenizer.apply_chat_template(
                safe_messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception as fallback_error:
            logging.error(f"Fallback template error: {fallback_error}")
            # Last ditch, no template at all. Block-form content has to be
            # flattened by hand here -- interpolating the list would put its
            # repr in the prompt.
            formatted_prompt = "\n".join(
                f"{msg['role']}: {_flatten_for_log(msg['content'])}" for msg in text_messages
            )

    return images, formatted_prompt, has_images, image_urls
