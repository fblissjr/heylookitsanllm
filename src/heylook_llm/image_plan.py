# src/heylook_llm/image_plan.py
"""What an image of a given size costs a RESIDENT model (plan W4).

Derived from the engine itself, never from a copy of its resize rules:

- MLX: the loaded model's own processor, through the same
  `prepare_inputs` heylook's vision path calls, on a synthetic image of each
  size. `tokens` is what the image adds to the prompt (placeholders plus any
  start/end markers); `target` is the size the processor resized to, where
  it says (a Qwen grid, or a 4-D pixel tensor).
- gguf: the running llama-server's own counter
  (`/v1/chat/completions/input_tokens`, which runs no vision encode), an
  image request minus the same request without it. llama.cpp does not say
  what size it resized to, so `target` is null.

Resident only, like the prompt preview: planning must never load a model.
The plan's replica-and-rule-name design (a hand copy of each family's resize
arithmetic) was dropped for this: the engine's own answer cannot drift from
the engine.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Any

from .providers.base import InvalidGenerationRequest

# A bound on one call, reasoned: planning is interactive (staging an image),
# and each size runs the processor once.
MAX_SIZES = 16
MAX_SIDE_PX = 16384


def _synthetic(w: int, h: int):
    from PIL import Image

    return Image.new("RGB", (w, h), (128, 128, 128))


def _check(sizes: list) -> list[tuple[int, int]]:
    if not sizes or len(sizes) > MAX_SIZES:
        raise InvalidGenerationRequest(f"give 1 to {MAX_SIZES} sizes")
    out = []
    for s in sizes:
        w, h = int(s[0]), int(s[1])
        if not (1 <= w <= MAX_SIDE_PX and 1 <= h <= MAX_SIDE_PX):
            raise InvalidGenerationRequest(f"size {w}x{h} is outside 1..{MAX_SIDE_PX} px")
        out.append((w, h))
    return out


def plan(provider: Any, sizes: list) -> dict:
    """{engine, images: [{size, tokens, target}], source}."""
    sizes = _check(sizes)
    if getattr(provider, "provider_name", "") == "gguf":
        return _plan_gguf(provider, sizes)
    return _plan_mlx(provider, sizes)


def _plan_mlx(provider, sizes) -> dict:
    if not getattr(provider, "is_vlm", False):
        raise InvalidGenerationRequest(f"{provider.model_id} is served as text; it takes no images")
    import mlx.core as mx
    from mlx_vlm.generate.common import generation_stream

    from .providers.mlx_provider import vlm_apply_chat_template, vlm_prepare_inputs
    from .streaming_utils import _executor_pool

    model, processor = provider.model, provider.processor
    config = model.config
    token_id = getattr(config, "image_token_id", None) or getattr(config, "image_token_index", None)
    messages = [{"role": "user", "content": "x"}]
    patch = getattr(getattr(processor, "image_processor", None), "patch_size", None)

    def work():
        # On a pinned MLX thread in the generation stream: prepare_inputs
        # builds MLX arrays, and MLX work belongs where the server runs it.
        with mx.stream(generation_stream):
            text_prompt = vlm_apply_chat_template(processor, config, messages, num_images=0,
                                                  enable_thinking=False)
            base = vlm_prepare_inputs(processor, prompts=text_prompt)["input_ids"].shape[-1]
            prompt = vlm_apply_chat_template(processor, config, messages, num_images=1,
                                             enable_thinking=False)
            rows = []
            for w, h in sizes:
                out = vlm_prepare_inputs(processor, images=[_synthetic(w, h)], prompts=prompt,
                                         image_token_index=token_id)
                n = int(out["input_ids"].shape[-1])
                target = None
                grid = out.get("image_grid_thw")
                pixels = out.get("pixel_values")
                if grid is not None and patch:
                    _, gh, gw = [int(x) for x in grid.tolist()[0]]
                    target = [gw * int(patch), gh * int(patch)]
                elif pixels is not None and pixels.ndim == 4:
                    target = [int(pixels.shape[-1]), int(pixels.shape[-2])]
                rows.append({"size": [w, h], "tokens": n - base, "target": target})
            return rows

    executor = _executor_pool.acquire()
    try:
        rows = executor.submit(work).result()
    finally:
        _executor_pool.release(executor)
    return {"engine": "mlx-vlm", "images": rows,
            "source": "the loaded model's own processor (prepare_inputs), per size"}


def _plan_gguf(provider, sizes) -> dict:
    if not (provider.config.get("mmproj_path") or getattr(provider, "is_vlm", False)):
        raise InvalidGenerationRequest(f"{provider.model_id} is served with no projector; it takes no images")
    base_url = getattr(provider, "_base_url", None)
    if not base_url:
        raise InvalidGenerationRequest(f"{provider.model_id} has no running llama-server")

    def count(content) -> int:
        body = json.dumps({"messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(base_url + "/v1/chat/completions/input_tokens", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(json.loads(resp.read())["input_tokens"])

    base = count([{"type": "text", "text": "x"}])
    rows = []
    for w, h in sizes:
        buf = io.BytesIO()
        _synthetic(w, h).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        n = count([{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "x"}])
        rows.append({"size": [w, h], "tokens": n - base, "target": None})
    return {"engine": "llama.cpp", "images": rows,
            "source": "the running llama-server's own token count (no vision encode)"}
