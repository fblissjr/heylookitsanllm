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


def prepare_vlm_inputs_parallel(
    messages: List,
    processor,
    config,
    batch_vision_processor,
    vlm_apply_chat_template_fn,
    model=None,
) -> Tuple[List[Image.Image], str, bool]:
    """Prepare VLM inputs with parallel image loading.

    Args:
        messages: List of ChatMessage objects (with .content, .role, etc.)
        processor: VLM processor (for tokenizer and chat template)
        config: Model config (model_type, etc.)
        batch_vision_processor: BatchVisionProcessor for parallel image loading
        vlm_apply_chat_template_fn: Function to apply VLM chat template
        model: Optional model instance (unused currently, reserved for future)

    Returns:
        Tuple of (images, formatted_prompt, has_images)
    """
    image_urls = []
    text_messages = []
    has_images = False

    # First pass: collect image URLs and build text structure
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            text_parts = []

            for part in content:
                # Handle both object and dict formats
                if hasattr(part, 'type'):
                    # Object format (ContentPart)
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        image_urls.append(part.image_url.url)
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
                            has_images = True

            # Combine text parts
            combined_content = " ".join(text_parts) if text_parts else ""
            msg_dict = {"role": msg.role, "content": combined_content}
            # Reconstruct thinking for assistant messages
            if hasattr(msg, 'thinking') and msg.thinking:
                msg_dict = _reconstruct_thinking({**msg_dict, 'thinking': msg.thinking})
            text_messages.append(msg_dict)
        elif isinstance(content, str):
            msg_dict = {"role": msg.role, "content": content}
            if hasattr(msg, 'thinking') and msg.thinking:
                msg_dict = _reconstruct_thinking({**msg_dict, 'thinking': msg.thinking})
            text_messages.append(msg_dict)

    # Load all images in parallel
    if image_urls:
        images = batch_vision_processor.load_images_parallel(image_urls)
    else:
        images = []

    # Format prompt
    try:
        # Ensure all content is strings (some templates have bugs with non-string content)
        safe_messages = []
        for msg in text_messages:
            safe_msg = {
                "role": str(msg["role"]) if not isinstance(msg["role"], str) else msg["role"],
                "content": str(msg["content"]) if not isinstance(msg["content"], str) else msg["content"]
            }
            safe_messages.append(safe_msg)

        formatted_prompt = vlm_apply_chat_template_fn(
            processor, config, safe_messages, num_images=len(images)
        )
    except Exception as e:
        logging.error(f"Chat template error: {e}")
        logging.error(f"Text messages: {text_messages}")
        # Fallback: Try without num_images parameter
        try:
            formatted_prompt = vlm_apply_chat_template_fn(
                processor, config, text_messages
            )
        except Exception as fallback_error:
            logging.error(f"Fallback template error: {fallback_error}")
            # Final fallback: manually format messages
            formatted_prompt = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in text_messages
            ])

    return images, formatted_prompt, has_images
