# src/heylook_llm/utils.py
import base64
import io
import json
import logging
import re
import requests
from PIL import Image, ImageOps
from typing import Dict, Any

def load_image(source_str: str) -> Image.Image:
    """Load an image from various sources: file path, URL, or base64 data."""
    try:
        if source_str.startswith("data:image"):
            try:
                header, encoded = source_str.split(",", 1)
                if len(encoded) < 10: raise ValueError("Base64 data too short")
                image_data = base64.b64decode(encoded)
                if len(image_data) < 10: raise ValueError("Decoded image data too short")
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                logging.debug(f"Successfully loaded base64 image: {image.size}")
                return image
            except Exception as e:
                logging.error(f"Failed to decode base64 image: {e}", exc_info=True)
                return Image.new('RGB', (1, 1), color='red')
        elif source_str.startswith("http"):
            headers = {
                'User-Agent': 'heylookitsanllm/1.0 (https://github.com/fblissjr/heylookitsanllm) Image processing bot'
            }
            response = requests.get(source_str, headers=headers, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        else:
            return ImageOps.exif_transpose(Image.open(source_str)).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to load image from {source_str[:100]}...: {e}", exc_info=True)
        # Return a small red image to indicate failure
        return Image.new('RGB', (64, 64), color='red')

def sanitize_request_for_debug(chat_request) -> str:
    """
    Create a debug-friendly JSON representation of a ChatRequest that truncates 
    base64 image data while showing useful image metadata.
    
    Why: Full base64 image data can be massive (100KB+) and clutters debug logs.
    This shows that images are present and provides metadata without the noise.
    """
    # Convert to dict for manipulation
    request_dict = chat_request.model_dump()
    
    # Process messages to handle image content
    if 'messages' in request_dict:
        for message in request_dict['messages']:
            if isinstance(message.get('content'), list):
                # Handle structured content with potential images
                for content_part in message['content']:
                    if (content_part.get('type') == 'image_url' and 
                        'image_url' in content_part and 
                        'url' in content_part['image_url']):
                        
                        url = content_part['image_url']['url']
                        content_part['image_url'] = _get_image_summary(url)
    
    return json.dumps(request_dict, indent=2)

def sanitize_dict_for_debug(data: Dict[str, Any]) -> str:
    """
    Create a debug-friendly JSON representation of a raw request dictionary
    that may contain image data in various formats (Ollama, OpenAI, etc.).
    
    Why: Raw request dictionaries can contain base64 image data in different
    structures depending on the API format.
    """
    # Deep copy to avoid modifying original
    import copy
    sanitized = copy.deepcopy(data)
    
    # Handle different API formats that might contain images
    _sanitize_dict_recursive(sanitized)
    
    return json.dumps(sanitized, indent=2)

def _sanitize_dict_recursive(obj: Any) -> None:
    """
    Recursively walk through a dictionary/list structure and sanitize image data.
    Modifies the object in place.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'url' and isinstance(value, str) and value.startswith('data:image'):
                # Direct URL field with base64 image
                obj[key] = _get_image_summary(value)['url']
            elif key == 'image' and isinstance(value, str) and value.startswith('data:image'):
                # Ollama format image field
                obj[key] = _get_image_summary(value)['url']
            elif key == 'images' and isinstance(value, list):
                # Ollama format images array
                for i, img in enumerate(value):
                    if isinstance(img, str) and img.startswith('data:image'):
                        value[i] = _get_image_summary(img)['url']
            else:
                # Recurse into nested structures
                _sanitize_dict_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _sanitize_dict_recursive(item)

def _get_image_summary(url: str) -> Dict[str, Any]:
    """
    Create a concise summary of an image URL for debug logging.
    
    Returns metadata about the image without the full data payload.
    """
    if url.startswith("data:image"):
        # Extract format and size info from base64 data URL
        try:
            header, encoded = url.split(",", 1)
            # Parse the header: data:image/jpeg;base64
            format_match = re.search(r'data:image/([^;]+)', header)
            image_format = format_match.group(1) if format_match else "unknown"
            
            # Estimate size
            encoded_size = len(encoded)
            estimated_bytes = (encoded_size * 3) // 4  # base64 to binary ratio
            
            return {
                "url": f"[BASE64_IMAGE:{image_format.upper()}]",
                "format": image_format,
                "base64_chars": encoded_size,
                "estimated_bytes": estimated_bytes,
                "size_human": _format_bytes(estimated_bytes)
            }
        except Exception:
            # Fallback for malformed data URLs
            return {
                "url": "[BASE64_IMAGE:PARSE_ERROR]",
                "note": "Could not parse image data"
            }
    elif url.startswith("http"):
        # HTTP/HTTPS URL - show the URL but keep it manageable
        return {
            "url": url if len(url) <= 100 else f"{url[:97]}...",
            "type": "HTTP_URL"
        }
    else:
        # File path or other
        return {
            "url": url if len(url) <= 100 else f"{url[:97]}...",
            "type": "FILE_PATH"
        }

def _format_bytes(bytes_count: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}GB"
