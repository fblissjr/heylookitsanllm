#!/usr/bin/env python3
"""Vision benchmark for mlx-vlm inference path.

Loads a VLM model directly via mlx-vlm and measures generation performance
with both text-only and vision prompts. Produces grep-friendly stdout and
detailed JSON results.

Usage:
    uv run scripts/bench_vlm.py
    uv run scripts/bench_vlm.py --model-path <path>
    uv run scripts/bench_vlm.py --reset-baseline
    uv run scripts/bench_vlm.py --text-only
    uv run scripts/bench_vlm.py --vision-only
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from PIL import Image, ImageDraw
from mlx_lm.generate import stream_generate as lm_stream_generate, wired_limit
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_vlm.utils import load as vlm_load, prepare_inputs as vlm_prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template

from bench_common import (
    VLM_DIR,
    baseline_metrics_from_result,
    build_result_data,
    check_hard_constraints,
    compute_composite_score,
    ensure_dirs,
    get_hardware_info,
    load_baseline,
    print_results,
    save_baseline,
    save_run,
    sync_barrier,
)


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen3.5-27B-mxfp8-mlx"


def resolve_model_path(model_path: str | None) -> str:
    """Resolve model path -- use provided path or default HF model ID."""
    if model_path:
        return model_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(f"mlx-community/{DEFAULT_MODEL}")
    except Exception:
        return f"mlx-community/{DEFAULT_MODEL}"


# ---------------------------------------------------------------------------
# Test image generation
# ---------------------------------------------------------------------------

def make_simple_image() -> Image.Image:
    """Create a 224x224 gradient image (minimal vision load)."""
    img = Image.new("RGB", (224, 224))
    for y in range(224):
        for x in range(224):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    return img


def make_complex_image() -> Image.Image:
    """Create a 448x448 image with geometric shapes."""
    img = Image.new("RGB", (448, 448), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Red rectangle
    draw.rectangle([50, 50, 200, 150], fill=(220, 50, 50), outline=(0, 0, 0))
    # Blue circle
    draw.ellipse([250, 80, 400, 230], fill=(50, 50, 220), outline=(0, 0, 0))
    # Green rectangle
    draw.rectangle([100, 280, 300, 400], fill=(50, 180, 50), outline=(0, 0, 0))
    # Yellow triangle
    draw.polygon([(350, 350), (420, 430), (280, 430)], fill=(220, 220, 50), outline=(0, 0, 0))
    # Small circles
    for i in range(5):
        x = 60 + i * 80
        draw.ellipse([x, 220, x + 30, 250], fill=(150, 50, 150))
    return img


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_PROMPTS = [
    {
        "name": "text_short",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "images": None,
    },
    {
        "name": "text_medium",
        "messages": [
            {"role": "user", "content": "Explain how hash tables work in 3-4 sentences."},
        ],
        "images": None,
    },
]

VISION_PROMPTS = [
    {
        "name": "vision_simple",
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ]},
        ],
        "image_factory": make_simple_image,
    },
    {
        "name": "vision_complex",
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "What objects are in this image and where are they positioned?"},
            ]},
        ],
        "image_factory": make_complex_image,
    },
]

MAX_TOKENS = 256
SEED = 42


# ---------------------------------------------------------------------------
# LanguageModelLogitsWrapper (inline -- avoids importing from src)
# ---------------------------------------------------------------------------

class _LogitsWrapper(nn.Module):
    """Minimal wrapper to extract logits from VLM language model output."""

    def __init__(self, language_model):
        super().__init__()
        object.__setattr__(self, "_lm", language_model)

    def __call__(self, *args, **kwargs):
        result = self._lm(*args, **kwargs)
        if hasattr(result, "logits"):
            return result.logits
        return result

    def __getattr__(self, name):
        if "_lm" in self.__dict__:
            return getattr(self._lm, name)
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Text-only VLM benchmark
# ---------------------------------------------------------------------------

def bench_text_prompt(
    model,
    processor,
    prompt: dict,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run a text-only prompt through the VLM text path."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    messages = prompt["messages"]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(formatted)
    num_prompt_tokens = len(prompt_tokens)

    # Build sampler
    mx.random.seed(SEED)
    sampler = make_sampler(temp=0.0)

    # Use the language model wrapper for text generation through VLM
    wrapper = _LogitsWrapper(model.language_model)

    completion_tokens = 0
    ttft_ms = 0.0
    gen_start = 0.0

    sync_barrier()
    start = time.perf_counter()

    for _ in lm_stream_generate(
        model=wrapper,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        sampler=sampler,
        max_tokens=max_tokens,
    ):
        if completion_tokens == 0:
            sync_barrier()
            now = time.perf_counter()
            ttft_ms = (now - start) * 1000
            gen_start = now
        completion_tokens += 1

    sync_barrier()
    end = time.perf_counter()

    if completion_tokens > 1 and gen_start > 0:
        gen_time_s = end - gen_start
        gen_tps = (completion_tokens - 1) / gen_time_s if gen_time_s > 0 else 0.0
    else:
        gen_tps = 0.0

    prefill_time_s = ttft_ms / 1000 if ttft_ms > 0 else 0.001
    prefill_tps = num_prompt_tokens / prefill_time_s

    return {
        "name": prompt["name"],
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": num_prompt_tokens,
        "had_images": False,
    }


# ---------------------------------------------------------------------------
# Vision VLM benchmark
# ---------------------------------------------------------------------------

def bench_vision_prompt(
    model,
    processor,
    prompt: dict,
    test_image: Image.Image,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run a vision prompt through the VLM vision path (pre-filled cache pattern)."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Format prompt with chat template
    messages = prompt["messages"]
    num_images = 1

    sync_barrier()
    img_start = time.perf_counter()

    formatted = vlm_apply_chat_template(
        processor, model.config, messages, num_images=num_images,
    )

    # Prepare inputs (tokenize + image processing)
    image_token_index = getattr(model.config, "image_token_index", None)
    inputs = vlm_prepare_inputs(
        processor,
        images=[test_image],
        prompts=formatted,
        image_token_index=image_token_index,
    )

    sync_barrier()
    img_time_ms = (time.perf_counter() - img_start) * 1000

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    extra_kwargs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }

    # Build VLM forward kwargs
    vlm_kwargs = dict(extra_kwargs)
    if pixel_values is not None:
        vlm_kwargs["pixel_values"] = pixel_values
    if mask is None:
        mask = mx.ones(input_ids.shape, dtype=mx.int32)
    vlm_kwargs["mask"] = mask

    num_prompt_tokens = input_ids.size

    # Create KV cache and wrapper
    wrapper = _LogitsWrapper(model.language_model)
    request_cache = make_prompt_cache(wrapper)

    # Build sampler
    mx.random.seed(SEED)
    sampler = make_sampler(temp=0.0)

    # Phase 1: Vision encoding (fills KV cache)
    sync_barrier()
    vision_start = time.perf_counter()

    with wired_limit(model, [mx.default_stream(mx.default_device())]):
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        output = model(input_ids, cache=request_cache, **vlm_kwargs)
        logits = output.logits if hasattr(output, "logits") else output

        # Sample first token
        last_logits = logits[:, -1, :]
        first_logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
        first_token_id = int(sampler(first_logprobs).item())
        mx.eval(first_logprobs)

    sync_barrier()
    vision_time_ms = (time.perf_counter() - vision_start) * 1000
    ttft_ms = img_time_ms + vision_time_ms

    # Phase 2: Text generation with pre-filled cache
    completion_tokens = 1  # first token already sampled
    gen_start = time.perf_counter()

    for _ in lm_stream_generate(
        model=wrapper,
        tokenizer=tokenizer,
        prompt=[first_token_id],
        sampler=sampler,
        max_tokens=max_tokens - 1,
        prompt_cache=request_cache,
    ):
        completion_tokens += 1

    sync_barrier()
    end = time.perf_counter()

    if completion_tokens > 1:
        gen_time_s = end - gen_start
        gen_tps = (completion_tokens - 1) / gen_time_s if gen_time_s > 0 else 0.0
    else:
        gen_tps = 0.0

    prefill_time_s = ttft_ms / 1000 if ttft_ms > 0 else 0.001
    prefill_tps = num_prompt_tokens / prefill_time_s

    return {
        "name": prompt["name"],
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": num_prompt_tokens,
        "had_images": True,
        "vision_ms": vision_time_ms,
        "img_processing_ms": img_time_ms,
    }


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    model_path: str,
    runs: int = 3,
    warmup: int = 1,
    reset_baseline: bool = False,
    text_only: bool = False,
    vision_only: bool = False,
) -> dict:
    """Run the full VLM benchmark suite."""
    ensure_dirs()

    # Load model
    print(f"Loading model: {model_path}", file=sys.stderr)
    model, processor = vlm_load(model_path)
    model_name = Path(model_path).name if "/" not in model_path or Path(model_path).exists() else model_path.split("/")[-1]
    print(f"Model loaded: {model_name}", file=sys.stderr)

    # Reset peak memory tracking
    mx.reset_peak_memory()

    # Generate test images
    simple_img = make_simple_image()
    complex_img = make_complex_image()
    test_images = {"vision_simple": simple_img, "vision_complex": complex_img}

    hardware = get_hardware_info()
    all_prompt_results = []

    # Select prompts
    prompts_to_run = []
    if not vision_only:
        for p in TEXT_PROMPTS:
            prompts_to_run.append(("text", p, None))
    if not text_only:
        for p in VISION_PROMPTS:
            img = test_images[p["name"]]
            prompts_to_run.append(("vision", p, img))

    for prompt_type, prompt, img in prompts_to_run:
        # Warmup
        for w in range(warmup):
            print(f"  warmup {w + 1}/{warmup}: {prompt['name']}...", end="\r", file=sys.stderr)
            if prompt_type == "text":
                bench_text_prompt(model, processor, prompt)
            else:
                bench_vision_prompt(model, processor, prompt, img)
            mx.clear_cache()

        # Measured runs
        prompt_runs = []
        for r in range(runs):
            print(f"  run {r + 1}/{runs}: {prompt['name']}...   ", end="\r", file=sys.stderr)
            if prompt_type == "text":
                result = bench_text_prompt(model, processor, prompt)
            else:
                result = bench_vision_prompt(model, processor, prompt, img)
            prompt_runs.append(result)
            mx.clear_cache()

        # Average across runs
        avg_result = {
            "name": prompt["name"],
            "gen_tps": sum(r["gen_tps"] for r in prompt_runs) / len(prompt_runs),
            "ttft_ms": sum(r["ttft_ms"] for r in prompt_runs) / len(prompt_runs),
            "prefill_tps": sum(r["prefill_tps"] for r in prompt_runs) / len(prompt_runs),
            "completion_tokens": sum(r["completion_tokens"] for r in prompt_runs) / len(prompt_runs),
            "prompt_tokens": prompt_runs[0]["prompt_tokens"],
            "had_images": prompt_runs[0]["had_images"],
        }
        # Vision-specific averages
        if prompt_runs[0].get("vision_ms") is not None:
            avg_result["vision_ms"] = sum(r["vision_ms"] for r in prompt_runs) / len(prompt_runs)
            avg_result["img_processing_ms"] = sum(r["img_processing_ms"] for r in prompt_runs) / len(prompt_runs)

        all_prompt_results.append(avg_result)
        extra = f", vision={avg_result.get('vision_ms', 0):.1f}ms" if avg_result.get("vision_ms") else ""
        print(f"  {prompt['name']}: gen={avg_result['gen_tps']:.1f} tps, "
              f"ttft={avg_result['ttft_ms']:.1f}ms, "
              f"prefill={avg_result['prefill_tps']:.1f} tps{extra}", file=sys.stderr)

    # Aggregate metrics
    avg_gen_tps = sum(r["gen_tps"] for r in all_prompt_results) / len(all_prompt_results)
    avg_ttft_ms = sum(r["ttft_ms"] for r in all_prompt_results) / len(all_prompt_results)
    avg_prefill_tps = sum(r["prefill_tps"] for r in all_prompt_results) / len(all_prompt_results)
    peak_memory_gb = mx.get_peak_memory() / (1024 ** 3)
    total_runs = len(prompts_to_run) * runs

    # Vision-specific aggregate
    vision_results = [r for r in all_prompt_results if r.get("vision_ms") is not None]
    avg_vision_ms = sum(r["vision_ms"] for r in vision_results) / len(vision_results) if vision_results else 0.0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    metrics = {
        "avg_gen_tps": round(avg_gen_tps, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "avg_prefill_tps": round(avg_prefill_tps, 1),
        "peak_memory_gb": round(peak_memory_gb, 1),
    }
    if avg_vision_ms > 0:
        metrics["avg_vision_ms"] = round(avg_vision_ms, 1)

    # Baseline
    baseline_data = load_baseline(VLM_DIR)
    if baseline_data is None or reset_baseline:
        result_data = build_result_data(
            bench="vlm", model=model_name, timestamp=timestamp,
            composite_score=1.0, metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        save_baseline(VLM_DIR, result_data)
        save_run(VLM_DIR, result_data, timestamp.replace(":", ""))
        composite_score = 1.0
        print("\nBaseline established.", file=sys.stderr)
    else:
        baseline_metrics = baseline_metrics_from_result(baseline_data)
        composite_score = compute_composite_score(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb, baseline_metrics,
        )
        avg_completion = sum(r["completion_tokens"] for r in all_prompt_results) / len(all_prompt_results)
        violations = check_hard_constraints(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb,
            avg_completion, baseline_metrics,
        )
        if violations:
            print(f"\nHard constraint violations:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)

        result_data = build_result_data(
            bench="vlm", model=model_name, timestamp=timestamp,
            composite_score=round(composite_score, 4), metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        save_run(VLM_DIR, result_data, timestamp.replace(":", ""))

    # Print grep-friendly output
    extra_lines = {}
    if avg_vision_ms > 0:
        extra_lines["avg_vision_ms"] = f"{avg_vision_ms:.1f}"
    print("", file=sys.stderr)
    print_results(
        composite_score=composite_score,
        avg_gen_tps=avg_gen_tps,
        avg_ttft_ms=avg_ttft_ms,
        avg_prefill_tps=avg_prefill_tps,
        peak_memory_gb=peak_memory_gb,
        runs=total_runs,
        model=model_name,
        bench="vlm",
        extra_lines=extra_lines if extra_lines else None,
    )

    return result_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vision benchmark (mlx-vlm path)")
    parser.add_argument("--model-path", default=None, help="Model path or HF repo ID")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per prompt")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per prompt")
    parser.add_argument("--reset-baseline", action="store_true", help="Re-establish baseline")
    parser.add_argument("--text-only", action="store_true", help="Skip vision prompts")
    parser.add_argument("--vision-only", action="store_true", help="Skip text prompts")
    args = parser.parse_args()

    if args.text_only and args.vision_only:
        print("Cannot specify both --text-only and --vision-only", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path(args.model_path)
    run_benchmark(
        model_path=model_path,
        runs=args.runs,
        warmup=args.warmup,
        reset_baseline=args.reset_baseline,
        text_only=args.text_only,
        vision_only=args.vision_only,
    )


if __name__ == "__main__":
    main()
