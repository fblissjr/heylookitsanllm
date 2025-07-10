# src/heylook_llm/utils.py
import base64
import io
import json
import logging
import re
import requests
import time
import os
from PIL import Image, ImageOps
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

def load_image(source_str: str) -> Image.Image:
    """Load an image from various sources: file path, URL, or base64 data."""
    try:
        if source_str.startswith("data:image"):
            try:
                _, encoded = source_str.split(",", 1)
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
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15'
            }
            response = requests.get(source_str, headers=headers, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            return ImageOps.exif_transpose(Image.open(source_str)).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to load image from {source_str[:100]}...: {e}", exc_info=True)
        # Return a small red image to indicate failure
        return Image.new('RGB', (64, 64), color='red')

def sanitize_request_for_debug(chat_request) -> str:
    """
    Create a debug-friendly JSON representation of a ChatRequest that truncates
    base64 image data to 1-2 lines max while preserving the actual data structure.

    Why: Full base64 image data can be massive (100KB+) and clutters debug logs.
    This shows the actual base64 data but truncated for readability.
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
                        content_part['image_url']['url'] = _truncate_image_url(url)

    return json.dumps(request_dict, indent=2)

def sanitize_dict_for_debug(data: Dict[str, Any]) -> str:
    """
    Create a debug-friendly JSON representation of a raw request dictionary
    that truncates base64 image data to 1-2 lines max.

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
    Recursively walk through a dictionary/list structure and truncate image data.
    Modifies the object in place.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'url' and isinstance(value, str) and value.startswith('data:image'):
                # Direct URL field with base64 image
                obj[key] = _truncate_image_url(value)
            elif key == 'image' and isinstance(value, str) and value.startswith('data:image'):
                # Ollama format image field
                obj[key] = _truncate_image_url(value)
            elif key == 'images' and isinstance(value, list):
                # Ollama format images array
                for i, img in enumerate(value):
                    if isinstance(img, str) and img.startswith('data:image'):
                        value[i] = _truncate_image_url(img)
            else:
                # Recurse into nested structures
                _sanitize_dict_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _sanitize_dict_recursive(item)

def _truncate_image_url(url: str, max_chars: int = 100) -> str:
    """
    Truncate a base64 image URL to show beginning and end, keeping it to 1-2 lines.

    Why: Preserves the actual data structure while making logs readable.
    Shows enough to verify the image is there without massive output.

    Set HEYLOOK_FULL_IMAGE_DEBUG=1 to see complete base64 data.
    """
    import os

    # Check for full image debug mode
    if os.getenv('HEYLOOK_FULL_IMAGE_DEBUG') == '1':
        return url  # Return full base64 data

    if url.startswith("data:image"):
        try:
            header, encoded = url.split(",", 1)
            encoded_size = len(encoded)

            # If it's already short enough, keep it as-is
            if encoded_size <= max_chars:
                return url

            # Truncate: show first 50 chars + size info + last 20 chars
            # This keeps it to about 1-2 lines in most terminals
            first_part = encoded[:50]
            last_part = encoded[-20:]
            size_info = f"...[{_format_bytes((encoded_size * 3) // 4)}]..."

            truncated_encoded = f"{first_part}{size_info}{last_part}"
            return f"{header},{truncated_encoded}"

        except Exception:
            # Fallback for malformed data URLs
            return url[:max_chars + 20] + "..." if len(url) > max_chars + 20 else url

    # For non-base64 URLs, just truncate if too long
    return url[:max_chars + 20] + "..." if len(url) > max_chars + 20 else url

def _format_bytes(bytes_count: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}GB"

@dataclass
class RequestMetrics:
    """Container for tracking individual request performance."""
    request_id: str
    start_time: float
    model_id: str
    stage: str = "starting"
    prompt_tokens: int = 0
    generated_tokens: int = 0
    memory_mb: float = 0.0

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def tokens_per_second(self) -> float:
        elapsed = self.elapsed_time()
        if elapsed == 0:
            return 0.0
        return self.generated_tokens / elapsed

class RealTimeLogger:
    """
    Real-time performance logger for request processing.

    Provides immediate visibility into what's happening during request processing.
    Why: User wants to see what's happening while it's happening, not just summaries.
    """

    def __init__(self):
        self.active_requests: Dict[str, RequestMetrics] = {}
        self._process = None

    def start_request(self, request_id: str, model_id: str) -> RequestMetrics:
        """Start tracking a new request."""
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.time(),
            model_id=model_id
        )
        self.active_requests[request_id] = metrics

        # Log request start with system info
        memory_info = self._get_memory_info()
        logging.info(
            f"👀 Request {request_id[:8]} started | Model: {model_id} | "
            f"Memory: {memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB "
            f"({memory_info['percent']:.1f}%)"
        )
        return metrics

    def update_stage(self, request_id: str, stage: str, **kwargs):
        """Update request processing stage with optional metrics."""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests[request_id]
        metrics.stage = stage
        elapsed = metrics.elapsed_time()

        # Update any provided metrics
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        # Stage-specific logging
        if stage == "model_loading":
            logging.info(f"⚡ Request {request_id[:8]} | Loading model... | {elapsed:.2f}s")
        elif stage == "prompt_processing":
            prompt_tokens = kwargs.get('prompt_tokens', metrics.prompt_tokens)
            logging.info(
                f"🙏 Request {request_id[:8]} | Processing prompt | {prompt_tokens} tokens | {elapsed:.2f}s"
            )
        elif stage == "generating":
            logging.info(f"✨ Request {request_id[:8]} | Generating response... | {elapsed:.2f}s")
        elif stage == "streaming":
            tokens = metrics.generated_tokens
            tps = metrics.tokens_per_second()
            logging.info(
                f"🤔 Request {request_id[:8]} | Streaming | {tokens} tokens @ {tps:.1f} tok/s | {elapsed:.2f}s"
            )

    def update_tokens(self, request_id: str, generated_tokens: int):
        """Update token count for ongoing generation."""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests[request_id]
        metrics.generated_tokens = generated_tokens

        # Log periodic updates for long generations
        if generated_tokens > 0 and generated_tokens % 50 == 0:
            tps = metrics.tokens_per_second()
            elapsed = metrics.elapsed_time()
            logging.debug(
                f"😙 Request {request_id[:8]} | {generated_tokens} tokens @ {tps:.1f} tok/s | {elapsed:.2f}s"
            )

    def complete_request(self, request_id: str, success: bool = True, error_msg: Optional[str] = None):
        """Mark request as complete and log final metrics."""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests[request_id]
        elapsed = metrics.elapsed_time()
        tps = metrics.tokens_per_second()

        if success:
            status_icon = "😅"
            status_msg = "completed"
        else:
            status_icon = "❌"
            status_msg = f"failed: {error_msg or 'unknown error'}"

        logging.info(
            f"{status_icon} Request {request_id[:8]} {status_msg} | "
            f"{metrics.generated_tokens} tokens @ {tps:.1f} tok/s | {elapsed:.2f}s total"
        )

        # Clean up
        del self.active_requests[request_id]

    def log_active_requests(self):
        """Log summary of currently active requests."""
        if not self.active_requests:
            return

        logging.info(f"😀 Active requests: {len(self.active_requests)}")
        for request_id, metrics in self.active_requests.items():
            elapsed = metrics.elapsed_time()
            tps = metrics.tokens_per_second()
            logging.info(
                f"👀 {request_id[:8]} | {metrics.stage} | {metrics.model_id} | "
                f"{metrics.generated_tokens} tokens @ {tps:.1f} tok/s | {elapsed:.2f}s"
            )

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        if not PSUTIL_AVAILABLE:
            return {'used_gb': 0.0, 'total_gb': 0.0, 'percent': 0.0}

        try:
            if self._process is None:
                self._process = psutil.Process(os.getpid())

            memory_info = self._process.memory_info()
            virtual_memory = psutil.virtual_memory()

            return {
                'used_gb': memory_info.rss / (1024**3),
                'total_gb': virtual_memory.total / (1024**3),
                'percent': virtual_memory.percent
            }
        except Exception:
            # Fallback if psutil fails
            return {'used_gb': 0.0, 'total_gb': 0.0, 'percent': 0.0}

# Global real-time logger instance
rt_logger = RealTimeLogger()

def log_request_start(request_id: str, model_id: str) -> RequestMetrics:
    """Start logging a new request."""
    return rt_logger.start_request(request_id, model_id)

def log_request_stage(request_id: str, stage: str, **kwargs):
    """Log request stage update."""
    rt_logger.update_stage(request_id, stage, **kwargs)

def log_token_update(request_id: str, generated_tokens: int):
    """Log token generation update."""
    rt_logger.update_tokens(request_id, generated_tokens)

def log_request_complete(request_id: str, success: bool = True, error_msg: Optional[str] = None):
    """Log request completion."""
    rt_logger.complete_request(request_id, success, error_msg)
