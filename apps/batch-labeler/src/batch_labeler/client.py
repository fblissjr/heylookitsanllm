"""HTTP client for VLM image labeling via OpenAI-compatible chat completions."""

import base64
import mimetypes
from pathlib import Path

import httpx


MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
}


def _detect_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return MIME_TYPES.get(ext, mimetypes.guess_type(str(path))[0] or 'application/octet-stream')


def label_image(
    client: httpx.Client,
    model_id: str,
    system_prompt: str,
    image_path: Path,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Send a single image to the VLM and return the response text.

    Args:
        client: Pre-configured httpx.Client with base_url set.
        model_id: Model ID to use for labeling.
        system_prompt: System prompt with labeling instructions.
        image_path: Path to the image file.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
    """
    image_data = image_path.read_bytes()
    b64 = base64.b64encode(image_data).decode('ascii')
    mime = _detect_mime(image_path)
    data_url = f"data:{mime};base64,{b64}"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Analyze this image."},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
