# src/edge_llm/utils.py
import base64, io, requests
import logging
from PIL import Image, ImageOps
from typing import List, Tuple, Union

def load_image(source_str: str) -> Image.Image:
    """Load an image from various sources: file path, URL, or base64 data."""
    try:
        if source_str.startswith("data:image"):
            header, encoded = source_str.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        elif source_str.startswith("http"):
            response = requests.get(source_str, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        else:
            return ImageOps.exif_transpose(Image.open(source_str)).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to load image from {source_str[:100]}...: {e}")
        raise ValueError(f"Unable to load image: {e}")

def process_vlm_messages(processor, model_config, messages: List) -> Tuple[List[Image.Image], str]:
    """
    Process multimodal messages and return images and formatted text prompt.

    Args:
        processor: The VLM processor (contains tokenizer and image processor)
        model_config: Model configuration for chat template formatting
        messages: List of message dictionaries with potential multimodal content

    Returns:
        Tuple of (images_list, formatted_prompt_string)
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    images = []
    text_messages = []

    for msg in messages:
        if isinstance(msg.get('content'), list):
            # Multimodal message with text and image parts
            text_parts = []
            for part in msg['content']:
                if part.get('type') == 'text':
                    text_parts.append(part['text'])
                elif part.get('type') == 'image_url':
                    try:
                        image = load_image(part['image_url']['url'])
                        images.append(image)
                        logging.debug(f"Loaded image {len(images)} successfully")
                    except Exception as e:
                        logging.warning(f"Failed to load image: {e}")
                        # Continue without the image rather than failing completely

            # Create a text-only message with combined text parts
            if text_parts:
                text_messages.append({
                    "role": msg['role'],
                    "content": "".join(text_parts)
                })
        else:
            # Pure text message
            text_messages.append(msg)

    # Apply the chat template to get formatted prompt
    try:
        formatted_prompt = apply_chat_template(
            processor,
            model_config,
            text_messages,
            num_images=len(images)
        )
    except Exception as e:
        logging.error(f"Failed to apply chat template: {e}")
        # Fallback: just concatenate message contents
        formatted_prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in text_messages
            if isinstance(msg.get('content'), str)
        ])

    logging.debug(f"Processed {len(images)} images and formatted prompt of length {len(formatted_prompt)}")
    return images, formatted_prompt

def validate_model_config(config: dict) -> dict:
    """Validate and normalize model configuration."""
    required_fields = ['model_path']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Set sensible defaults
    defaults = {
        'vision': False,
        'temperature': 1.0,
        'top_p': 0.95,
        'max_tokens': 512,
        'repetition_penalty': 1.1,
        'num_draft_tokens': 5,
    }

    # Apply defaults for missing values
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value

    return config

def safe_model_call(func, *args, **kwargs):
    """Safely call a model function with error handling and logging."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Model call failed: {e}")
        logging.debug(f"Function: {func.__name__}, Args: {args}, Kwargs: {kwargs}")
        raise
