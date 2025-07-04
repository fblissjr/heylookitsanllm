# src/heylook_llm/utils.py
import base64
import io
import logging
import requests
from PIL import Image, ImageOps

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
