# src/heylook_llm/api_capabilities.py
"""
Capabilities endpoint for clients to detect server features.
"""

from fastapi import Request
from typing import Dict, Any

async def get_capabilities(request: Request) -> Dict[str, Any]:
    """
    Return server capabilities for client feature detection.
    
    This helps clients like ComfyUI nodes understand what features
    are available without trial and error.
    """
    return {
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "multipart": "/v1/chat/completions/multipart"
        },
        "features": {
            "image_support": True,
            "image_resize": True,
            "batch_processing": True,
            "streaming": True,
            "multipart_upload": True,
            "vision_models": True,
            "speculative_decoding": True
        },
        "image_processing": {
            "resize_parameters": {
                "resize_max": "Resize to max dimension (512, 768, 1024, etc.)",
                "resize_width": "Resize to specific width",
                "resize_height": "Resize to specific height",
                "image_quality": "JPEG quality (1-100, default 85)",
                "preserve_alpha": "Preserve transparency (outputs PNG)"
            },
            "supported_formats": ["JPEG", "PNG", "WEBP", "BMP", "GIF"],
            "max_image_size": 50 * 1024 * 1024,  # 50MB
            "max_images_per_request": 10
        },
        "batch_processing_modes": [
            "conversation",
            "sequential", 
            "sequential_with_context",
            "parallel",
            "parallel_with_context"
        ],
        "performance_optimizations": {
            "orjson": True,
            "uvloop": True,
            "turbojpeg": True,
            "xxhash": True,
            "image_caching": True
        }
    }