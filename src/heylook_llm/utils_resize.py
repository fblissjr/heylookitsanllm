# src/heylook_llm/utils_resize.py
"""
Image resizing utilities for the standard API endpoint.
"""

import logging
import time
from PIL import Image
import io
import base64
from typing import Optional, Tuple

def resize_image_if_needed(
    image: Image.Image,
    resize_max: Optional[int] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    image_quality: int = 85,
    preserve_alpha: bool = False
) -> Tuple[Image.Image, bool]:
    """
    Resize image based on provided parameters.
    
    Returns:
        Tuple of (image, was_resized)
    """
    if not any([resize_max, resize_width, resize_height]):
        return image, False
    
    width, height = image.size
    new_width, new_height = width, height
    needs_resize = False
    
    if resize_width and resize_height:
        # Specific dimensions requested
        new_width = resize_width
        new_height = resize_height
        needs_resize = True
    elif resize_width:
        # Only width specified, maintain aspect ratio
        scale = resize_width / width
        new_width = resize_width
        new_height = int(height * scale)
        needs_resize = True
    elif resize_height:
        # Only height specified, maintain aspect ratio
        scale = resize_height / height
        new_width = int(width * scale)
        new_height = resize_height
        needs_resize = True
    elif resize_max and resize_max > 0:
        # Max dimension specified
        max_dim = max(width, height)
        if max_dim > resize_max:
            scale = resize_max / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            needs_resize = True
    
    if needs_resize:
        start_time = time.time()
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resize_time = time.time() - start_time
        
        # Calculate size reduction
        original_pixels = width * height
        new_pixels = new_width * new_height
        reduction_percent = ((original_pixels - new_pixels) / original_pixels) * 100
        
        logging.info(f"[IMAGE RESIZE] Resized from {width}x{height} to {new_width}x{new_height} | "
                   f"Reduction: {reduction_percent:.1f}% | Time: {resize_time*1000:.1f}ms")
        return resized, True
    
    return image, False


def process_image_url_with_resize(
    image_url: str,
    resize_max: Optional[int] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    image_quality: int = 85,
    preserve_alpha: bool = False
) -> str:
    """
    Process an image URL (including base64) and apply resizing if needed.
    Returns the potentially resized image as a base64 URL.
    """
    if not image_url.startswith("data:image"):
        # For non-base64 URLs, return as-is (resize would need to happen during loading)
        return image_url
    
    try:
        # Decode base64 image
        _, encoded = image_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        
        # Log original size
        logging.info(f"[IMAGE RESIZE] Processing base64 image: original size {image.size}")
        
        # Resize if needed
        image, was_resized = resize_image_if_needed(
            image,
            resize_max=resize_max,
            resize_width=resize_width,
            resize_height=resize_height,
            image_quality=image_quality,
            preserve_alpha=preserve_alpha
        )
        
        if was_resized:
            # Convert back to base64
            buffer = io.BytesIO()
            
            # Handle format conversion
            if preserve_alpha and image.mode in ('RGBA', 'LA'):
                image.save(buffer, format='PNG', optimize=True)
                mime_type = "image/png"
            else:
                # Convert to RGB for JPEG
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(buffer, format='JPEG', quality=image_quality, optimize=True)
                mime_type = "image/jpeg"
            
            # Create new base64 URL
            buffer.seek(0)
            new_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            new_url = f"data:{mime_type};base64,{new_base64}"
            
            # Log size comparison
            original_size = len(encoded)
            new_size = len(new_base64)
            size_reduction = ((original_size - new_size) / original_size) * 100
            logging.info(f"[IMAGE RESIZE] Base64 size reduced by {size_reduction:.1f}% "
                       f"({_format_bytes((original_size * 3) // 4)} → {_format_bytes((new_size * 3) // 4)})")
            
            return new_url
        
    except Exception as e:
        logging.error(f"[IMAGE RESIZE] Failed to process image: {e}")
    
    # Return original if no resize or error
    return image_url


def _format_bytes(bytes_count: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}GB"