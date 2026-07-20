"""HTTP client for VLM image labeling via the heylookitsanllm OpenAI-compatible API.

Also works against any OpenAI-compatible server; the heylook-specific fields
(sampler, enable_thinking, vision_tokens, resize_max, image_quality,
include_performance) are simply ignored elsewhere.
"""

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .scanner import MIME_TYPES


@dataclass(frozen=True)
class GenerationOptions:
    """Request knobs beyond the prompts. None means 'omit from the payload'
    so the server's preset/model-default cascade decides."""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    sampler: str | None = None
    enable_thinking: bool | None = None
    vision_tokens: int | None = None
    resize_max: int | None = None
    image_quality: int | None = None


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


def image_data_url(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{_detect_mime(image_path)};base64,{b64}"


def build_payload(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    options: GenerationOptions,
) -> dict:
    """Build a /v1/chat/completions payload. Optional fields set to None are
    omitted entirely so server-side defaults (model config + preset) apply."""
    payload: dict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "stream": False,
        "include_performance": True,
    }
    for key in (
        "max_tokens", "temperature", "top_p", "seed", "sampler",
        "enable_thinking", "vision_tokens", "resize_max", "image_quality",
    ):
        value = getattr(options, key)
        if value is not None:
            payload[key] = value
    return payload


def parse_chat_response(data: dict) -> LabelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {str(data)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError(f"response message has no content: {str(data)[:200]}")
    return LabelResponse(
        content=content,
        thinking=message.get("thinking"),
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
) -> LabelResponse:
    """Send one image for labeling. Retries transient failures (timeouts,
    connection errors, 5xx) with linear backoff; 4xx errors raise immediately.
    """
    payload = build_payload(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_data_url=image_data_url(image_path),
        options=options,
    )

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(2.0 * attempt)
        start = time.time()
        try:
            response = client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = parse_chat_response(response.json())
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
