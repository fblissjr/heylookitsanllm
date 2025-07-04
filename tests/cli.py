# tests/cli.py
"""cli.py – Simple CLI wrapper around a local /v1/chat/completions endpoint
with rich template support and full set of generation flags

Highlights
~~~~~~~~~~~
* **Template system** (10 system + 10 user templates) with sensible defaults.
* **Raw prompt mode**: `--system-prompt` and `--user-text` let you bypass templates
  without crafting JSON.
* **Multi‑image input** via `--image` (URLs or local files, unlimited).
* **Full generation flag coverage** (temperature, top‑p, KV cache, etc.).
* **Pydantic v2 compatible** (`model_dump_json`, `field_validator`).

Priority order for message construction
--------------------------------------
1. `--messages` (inline JSON or `@file`) → **everything else ignored**.
2. `--system-prompt` / `--user-text` + optional `--image`.
3. Template keys (`--system-template` + `--user-template`) + optional `--image`.

Quick demo
~~~~~~~~~~
```bash
# Default: vision scene analyst on cat.jpg
python cli.py --image cat.jpg

# Raw string prompts (no templates)
python cli.py \
  --system-prompt "You are a helpful assistant." \
  --user-text "Describe this image in detail." \
  --image https://picsum.photos/400

# Object detector on two URLs using template
python cli.py \
  --system-template object_detector \
  --user-template detect_objects \
  --image https://e.com/1.jpg --image https://e.com/2.jpg
```
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, field_validator

###############################################################################
# Template dictionaries (full versions)
###############################################################################

DEFAULT_SYSTEM_TEMPLATES: Dict[str, str] = {
    # Vision‑focused (7)
    "video_scene_analyst": (
        "You are an expert Video Scene Analyst AI. Your purpose is to look at a static image "
        "and extrapolate it into a dynamic, 5‑second video scene description for a downstream "
        "generation model.\n\n"
        "Rules:\n"
        "1. Markdown only with sections: `## Visual Description`, `## Camera Movement`, `## Sound Design`.\n"
        "2. Present‑tense visual description (micro‑movements, ambient motion).\n"
        "3. One simple camera movement.\n"
        "4. 2–3 diegetic sounds as bullets. No extra text."
    ),
    "image_captioner": (
        "You are an advanced Image Captioner AI. Given a single image, output one concise sentence "
        "describing key objects, actions, and context. No extra commentary."
    ),
    "object_detector": (
        "You are an Object Detection Analyst AI. Output JSON array of objects with `label`, `confidence`, "
        "and `bbox` (x,y,w,h). No other text."
    ),
    "scene_graph_generator": (
        "You are a Scene Graph Generator AI. Output JSON list of (subject, predicate, object) triples."
    ),
    "visual_qa": (
        "You are a Visual QA AI. Answer questions about an image; use bullets if multiple points."
    ),
    "video_summarizer": (
        "You are a Video Summarization AI. Provide a one‑paragraph synopsis of a 10‑second clip in 3rd person present tense."
    ),
    "style_transfer_explainer": (
        "You are an Image Style Transfer Explainability AI. Given source + target style images, describe changes in bullets: Color, Texture, Stroke. No extra text."
    ),
    # General (3)
    "code_debugger": (
        "You are a senior software engineer / debugging expert.\n"
        "1. Summarize bug in one sentence.\n2. Show relevant code excerpt.\n3. Root cause & fix bullets.\n4. One‑line corrected code."
    ),
    "travel_planner": (
        "You are a travel planner. Return day‑by‑day itinerary in Markdown (## Day N then – **TIME**: Activity — Rationale)."
    ),
    "creative_writer": (
        "You are a novelist. Turn prompt into 200‑word micro‑fiction with setting, dialogue, twist ending. Present tense only."
    ),
}

DEFAULT_USER_TEMPLATES: Dict[str, Any] = {
    "analyze_scene": [
        {"type": "text", "text": "Analyze this scene."},
        {"type": "image_url", "image_url": {"url": "@image_path_or_url"}},
    ],
    "caption_image": [
        {"type": "text", "text": "Provide a one‑sentence caption for this image."},
        {"type": "image_url", "image_url": {"url": "@image_path_or_url"}},
    ],
    "detect_objects": [
        {"type": "text", "text": "Detect and list objects in this image."},
        {"type": "image_url", "image_url": {"url": "@image_path_or_url"}},
    ],
    "scene_graph": [
        {"type": "text", "text": "Generate a scene graph for this image."},
        {"type": "image_url", "image_url": {"url": "@image_path_or_url"}},
    ],
    "visual_qa": [
        {"type": "text", "text": "What objects and actions are happening here?"},
        {"type": "image_url", "image_url": {"url": "@image_path_or_url"}},
    ],
    "summarize_video": [
        {"type": "text", "text": "Summarize the following 10‑second video clip."},
        {"type": "video_url", "video_url": {"url": "@video_path_or_url"}},
    ],
    "explain_style_transfer": [
        {"type": "text", "text": "Explain how style transfer will alter this image."},
        {"type": "image_url", "image_url": {"url": "@source_image_url"}},
        {"type": "image_url", "image_url": {"url": "@target_style_url"}},
    ],
    "debug_code": [
        {"type": "text", "text": "Here’s my Python function—please fix the bug:"},
        {"type": "code", "language": "python", "text": "def add(a, b):\n    return a - b\n"},
    ],
    "plan_paris_trip": [
        {"type": "text", "text": "Plan a 3‑day itinerary for Paris in June, €1500 budget."},
    ],
    "write_poem": [
        {"type": "text", "text": "Write a 200‑word micro‑fiction set in a medieval forest."},
    ],
}

###############################################################################
# Pydantic models
###############################################################################

class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    # core
    model: Optional[str] = None
    messages: List[ChatMessage]

    # generation
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(0.95, ge=0, le=1)
    top_k: int = Field(0, ge=0)
    min_p: float = Field(0.0, ge=0, le=1)
    repetition_penalty: float = Field(1.1, ge=0.1, le=2)
    repetition_context_size: int = Field(20, ge=1)
    max_tokens: int = Field(512, gt=0, le=8192)
    stream: bool = False
    include_performance: bool = False

    # advanced
    draft_model: Optional[str] = None
    num_draft_tokens: int = Field(5, ge=1, le=20)
    seed: Optional[int] = None

    # sampling extras
    xtc_probability: float = Field(0.0, ge=0, le=1)
    xtc_threshold: float = Field(0.0, ge=0, le=1)
    logit_bias: Optional[Dict[str, float]] = None

    # cache
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = Field(None, ge=1, le=8)
    quantized_kv_start: int = Field(5000, ge=0)

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be > 0")
        return v

###############################################################################
# Helper utilities
###############################################################################

def parse_logit_bias(val: str | None) -> Optional[Dict[str, float]]:
    if val is None:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--logit_bias must be valid JSON") from exc


def load_messages(src: str) -> Any:
    if src.startswith("@"):
        with open(src[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(src)


def encode_local_image(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"Error: image file '{path}' not found")
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as fh:
        data64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{data64}"

def make_image_items(image_args: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for img in image_args:
        url = img if img.startswith("http://") or img.startswith("https://") else encode_local_image(img)
        items.append({"type": "image_url", "image_url": {"url": url}})
    return items


def merge_images_into_template(template_content: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace placeholders starting with @image_path_or_url and append remaining images."""
    output: List[Dict[str, Any]] = []
    img_iter = iter(images)
    for block in template_content:
        if block.get("type") == "image_url" and block["image_url"].get("url", "").startswith("@"):
            try:
                output.append(next(img_iter))
            except StopIteration:
                # Skip placeholder if no more images
                continue
        else:
            output.append(block)
    # Append any leftover images
    output.extend(img_iter)
    return output

###############################################################################
# CLI setup
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local chat/completions CLI with templates, raw prompts, and multi‑image support")

    # Endpoint & modes
    p.add_argument("--url", default="http://localhost:8080/v1/chat/completions", help="Endpoint URL")
    p.add_argument("--messages", help="Inline JSON or @file (highest priority, bypasses templates/prompts/images)")

    # Raw prompt mode
    p.add_argument("--system-prompt", help="Raw string for system message")
    p.add_argument("--user-text", help="Raw string for user text message")

    # Template mode (defaults)
    p.add_argument("--system-template", choices=DEFAULT_SYSTEM_TEMPLATES.keys(), default="video_scene_analyst")
    p.add_argument("--user-template", choices=DEFAULT_USER_TEMPLATES.keys(), default="analyze_scene")

    # Images
    p.add_argument("--image", action="append", default=[], help="Image URL or local file path (repeatable)")

    # Generation params group
    g = p.add_argument_group("generation")
    g.add_argument("--model")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--top_k", type=int, default=0)
    g.add_argument("--min_p", type=float, default=0.0)
    g.add_argument("--repetition_penalty", type=float, default=1.1)
    g.add_argument("--repetition_context_size", type=int, default=20)
    g.add_argument("--max_tokens", type=int, default=512)
    g.add_argument("--stream", action="store_true")
    g.add_argument("--include_performance", action="store_true")
    g.add_argument("--draft_model")
    g.add_argument("--num_draft_tokens", type=int, default=5)
    g.add_argument("--seed", type=int)
    g.add_argument("--xtc_probability", type=float, default=0.0)
    g.add_argument("--xtc_threshold", type=float, default=0.0)
    g.add_argument("--logit_bias", type=parse_logit_bias)
    g.add_argument("--max_kv_size", type=int)
    g.add_argument("--kv_bits", type=int)
    g.add_argument("--quantized_kv_start", type=int, default=5000)

    return p

###############################################################################
# Main entrypoint
###############################################################################

def main() -> None:
    args = build_parser().parse_args()

    # PREP IMAGES (encoded or passthrough)
    image_items = make_image_items(args.image)

    # PRIORITY 1: --messages provided
    if args.messages:
        if args.system_prompt or args.user_text:
            print("Warning: --system-prompt/--user-text ignored because --messages provided", file=sys.stderr)
        if args.image:
            print("Warning: --image ignored because --messages provided", file=sys.stderr)
        raw_msgs = load_messages(args.messages)

    # PRIORITY 2: raw string prompts
    elif args.system_prompt or args.user_text:
        if not (args.system_prompt and args.user_text):
            sys.exit("Error: both --system-prompt and --user-text are required together (or use templates)")

        user_blocks: List[Dict[str, Any]] = [{"type": "text", "text": args.user_text}]
        user_blocks.extend(image_items)

        raw_msgs = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_blocks},
        ]

    # PRIORITY 3: template mode
    else:
        # Build user content from template & images
        tmpl_blocks = DEFAULT_USER_TEMPLATES[args.user_template]
        merged_blocks = merge_images_into_template(tmpl_blocks, image_items)

        raw_msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATES[args.system_template]},
            {"role": "user", "content": merged_blocks},
        ]

    # Validate messages via Pydantic model
    msg_models = [ChatMessage(**m) for m in raw_msgs]

    # Build request payload
    req = ChatRequest(
        model=args.model,
        messages=msg_models,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
        max_tokens=args.max_tokens,
        stream=args.stream,
        include_performance=args.include_performance,
        draft_model=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        seed=args.seed,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        logit_bias=args.logit_bias,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        quantized_kv_start=args.quantized_kv_start,
    )

    headers = {"Content-Type": "application/json"}
    payload = req.model_dump_json(exclude_none=True)

    try:
        resp = requests.post(args.url, headers=headers, data=payload, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"HTTP request failed: {e}")

    # Print formatted JSON or raw text
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)


if __name__ == "__main__":
    main()
