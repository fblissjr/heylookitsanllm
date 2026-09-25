"""HTTP client for VLM image labeling via heylookitsanllm's Messages API
(``POST /v1/messages``, Anthropic-shaped).

Images are resized CLIENT-side before they are sent: the server's resize
parameters went with its OpenAI route (v1.79.66), and a vision tower's cost
grows much faster than the pixel count, so an uncapped camera photo is the
expensive case. ``max_edge`` caps the longest edge, like the chat page's cap.
"""

import base64
import io
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .scanner import MIME_TYPES

# The chat page's own cap (frontend/js/image-prep.js MAX_EDGE_PX).
DEFAULT_MAX_EDGE = 2048


@dataclass(frozen=True)
class GenerationOptions:
    """Request knobs beyond the prompts. None means 'omit from the payload'
    so the server's model-default cascade decides."""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    thinking: bool | None = None


@dataclass
class LabelResponse:
    content: str
    thinking: str | None
    usage: dict
    performance: dict | None
    model: str
    request_ms: int = 0


class ServerError(RuntimeError):
    """Raised when the server is unreachable or returns an unusable response."""


def _detect_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return MIME_TYPES.get(
        ext, mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    )


def image_block(image_path: Path, max_edge: int | None = DEFAULT_MAX_EDGE) -> dict:
    """A Messages image block (base64 source) for ``image_path``.

    The file is sent as-is when it already fits ``max_edge`` (or when
    ``max_edge`` is None or 0); otherwise it is downscaled to fit and
    re-encoded (JPEG for photos, PNG where the original was PNG).
    """
    raw = image_path.read_bytes()
    media_type = _detect_mime(image_path)
    if max_edge:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(raw)) as im:
            if max(im.size) > max_edge:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                if media_type == "image/png":
                    im.save(buf, format="PNG")
                else:
                    im.convert("RGB").save(buf, format="JPEG", quality=90)
                    media_type = "image/jpeg"
                raw = buf.getvalue()
    return {"type": "image", "source": {
        "type": "base64", "media_type": media_type,
        "data": base64.b64encode(raw).decode("ascii")}}


def build_payload(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    image: dict,
    options: GenerationOptions,
) -> dict:
    """Build a /v1/messages payload. Optional fields set to None are omitted
    entirely so server-side defaults (the model's config) apply."""
    payload: dict = {
        "model": model_id,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": [image, {"type": "text", "text": user_prompt}]},
        ],
        "stream": False,
    }
    for key in ("max_tokens", "temperature", "top_p", "seed", "thinking"):
        value = getattr(options, key)
        if value is not None:
            payload[key] = value
    return payload


def parse_message_response(data: dict) -> LabelResponse:
    """The reply's text and thinking from a Messages response's content
    blocks. A thinking block carries its text as ``thinking`` (heylook's
    older ``text`` spelling is read too)."""
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ValueError(f"response has no content blocks: {str(data)[:200]}")
    text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
    thinking = "".join(b.get("thinking") or b.get("text") or ""
                       for b in blocks if b.get("type") == "thinking")
    if not text and not thinking:
        raise ValueError(f"response carries no text: {str(data)[:200]}")
    return LabelResponse(
        content=text,
        thinking=thinking or None,
        usage=data.get("usage") or {},
        performance=data.get("performance"),
        model=data.get("model", ""),
    )


def label_image(
    client: httpx.Client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    image_path: Path,
    options: GenerationOptions,
    retries: int = 2,
    max_edge: int | None = DEFAULT_MAX_EDGE,
) -> LabelResponse:
    """Send one image for labeling. Retries transient failures (timeouts,
    connection errors, 5xx) with linear backoff; 4xx errors raise immediately.
    """
    payload = build_payload(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=image_block(image_path, max_edge),
        options=options,
    )

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(2.0 * attempt)
        start = time.time()
        try:
            response = client.post("/v1/messages", json=payload)
            response.raise_for_status()
            result = parse_message_response(response.json())
            result.request_ms = int((time.time() - start) * 1000)
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
            last_exc = e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
    assert last_exc is not None
    raise last_exc


def fetch_models(client: httpx.Client) -> list[dict]:
    """GET /v1/models; raises ServerError with a friendly message if down."""
    try:
        response = client.get("/v1/models")
        response.raise_for_status()
    except (httpx.TransportError, httpx.TimeoutException) as e:
        raise ServerError(
            f"cannot reach server at {client.base_url} ({e.__class__.__name__}). "
            "Is heylookitsanllm running?"
        ) from e
    except httpx.HTTPStatusError as e:
        raise ServerError(
            f"server at {client.base_url} returned HTTP {e.response.status_code} for /v1/models"
        ) from e
    return response.json().get("data", [])


def is_vision_model(model: dict) -> bool:
    return (
        "vision" in (model.get("capabilities") or [])
        or "vision" in (model.get("modalities") or [])
    )


def vision_models(models: list[dict]) -> list[dict]:
    return [m for m in models if is_vision_model(m)]


def pick_vision_model(models: list[dict]) -> str | None:
    """The sole vision-capable model's id, or None if zero or ambiguous."""
    vlms = vision_models(models)
    if len(vlms) == 1:
        return vlms[0]["id"]
    return None
