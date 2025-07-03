# src/edge_llm/utils.py
import base64, io, requests
import logging
from PIL import Image, ImageOps
from typing import List, Tuple, Union

def load_image(source_str: str) -> Image.Image:
    """Load an image from various sources: file path, URL, or base64 data."""
    try:
        if source_str.startswith("data:image"):
            # Handle base64 encoded images
            try:
                header, encoded = source_str.split(",", 1)
                # Validate base64 data
                if len(encoded) < 10:
                    raise ValueError("Base64 data too short")

                # Decode and verify it's valid image data
                image_data = base64.b64decode(encoded)
                if len(image_data) < 10:
                    raise ValueError("Decoded image data too short")

                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                logging.debug(f"Successfully loaded base64 image: {image.size}")
                return image
            except Exception as e:
                logging.error(f"Failed to decode base64 image: {e}")
                # Create a simple 1x1 red pixel as fallback
                fallback = Image.new('RGB', (1, 1), color='red')
                logging.warning("Using 1x1 red pixel fallback for corrupted image")
                return fallback

        elif source_str.startswith("http"):
            response = requests.get(source_str, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        else:
            return ImageOps.exif_transpose(Image.open(source_str)).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to load image from {source_str[:100]}...: {e}")
        # Create a simple fallback image instead of raising
        fallback = Image.new('RGB', (64, 64), color='gray')
        logging.warning("Using 64x64 gray fallback for failed image load")
        return fallback

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
    images = []
    text_messages = []

    # Safely process messages
    if not messages or not isinstance(messages, list):
        logging.warning("Invalid or empty messages list provided")
        return [], "Hello"  # Safe fallback

    for i, msg in enumerate(messages):
        try:
            if not isinstance(msg, dict):
                logging.warning(f"Message {i} is not a dictionary, skipping")
                continue

            if isinstance(msg.get('content'), list):
                # Multimodal message with text and image parts
                text_parts = []
                content_list = msg.get('content', [])

                for j, part in enumerate(content_list):
                    try:
                        if not isinstance(part, dict):
                            continue

                        if part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                        elif part.get('type') == 'image_url':
                            image_url = part.get('image_url', {})
                            if isinstance(image_url, dict):
                                url = image_url.get('url', '')
                                if url:
                                    try:
                                        image = load_image(url)
                                        images.append(image)
                                        logging.debug(f"Loaded image {len(images)} successfully: {image.size}")
                                    except Exception as e:
                                        logging.warning(f"Failed to load image {j} in message {i}: {e}")
                                        # Continue without the image rather than failing
                    except Exception as e:
                        logging.warning(f"Error processing content part {j} in message {i}: {e}")
                        continue

                # Create a text-only message with combined text parts
                if text_parts:
                    text_messages.append({
                        "role": msg.get('role', 'user'),
                        "content": "".join(text_parts)
                    })
            else:
                # Pure text message
                text_messages.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
        except Exception as e:
            logging.warning(f"Error processing message {i}: {e}")
            continue

    # Apply the chat template to get formatted prompt
    formatted_prompt = ""
    try:
        if text_messages:
            from mlx_vlm.prompt_utils import apply_chat_template
            formatted_prompt = apply_chat_template(
                processor,
                model_config,
                text_messages,
                num_images=len(images)
            )
        else:
            formatted_prompt = "Hello"  # Fallback if no text messages
    except Exception as e:
        logging.error(f"Failed to apply chat template: {e}")
        # Fallback: just concatenate message contents safely
        try:
            text_parts = []
            for msg in text_messages:
                if isinstance(msg.get('content'), str):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    text_parts.append(f"{role}: {content}")
            formatted_prompt = "\n".join(text_parts) if text_parts else "Hello"
        except Exception as e2:
            logging.error(f"Fallback prompt creation failed: {e2}")
            formatted_prompt = "Hello"  # Final fallback

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
