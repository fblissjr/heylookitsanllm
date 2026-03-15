#!/usr/bin/env python3
"""Vision benchmark for mlx-vlm inference path.

Loads a VLM model directly via mlx-vlm and measures generation performance
with both text-only and vision prompts. Produces grep-friendly stdout and
detailed JSON results. Includes output fingerprinting for correctness verification.

Usage:
    uv run apps/optloop/scripts/bench_vlm.py
    uv run apps/optloop/scripts/bench_vlm.py --model-path <path>
    uv run apps/optloop/scripts/bench_vlm.py --reset-baseline
    uv run apps/optloop/scripts/bench_vlm.py --text-only
    uv run apps/optloop/scripts/bench_vlm.py --vision-only
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
    check_fingerprints,
    check_hard_constraints,
    check_per_prompt_constraints,
    check_suspicion,
    check_variance,
    compute_composite_score,
    ensure_dirs,
    fingerprint_output,
    get_bench_params,
    get_constraints,
    get_hardware_info,
    get_scoring_weights,
    load_baseline,
    load_config,
    print_results,
    save_baseline,
    save_run,
    sync_barrier,
)


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
    draw.rectangle([50, 50, 200, 150], fill=(220, 50, 50), outline=(0, 0, 0))
    draw.ellipse([250, 80, 400, 230], fill=(50, 50, 220), outline=(0, 0, 0))
    draw.rectangle([100, 280, 300, 400], fill=(50, 180, 50), outline=(0, 0, 0))
    draw.polygon([(350, 350), (420, 430), (280, 430)], fill=(220, 220, 50), outline=(0, 0, 0))
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


def resolve_model_path(model_path: str | None, config: dict) -> str:
    """Resolve model path -- use provided path, config, or default HF model ID."""
    if model_path:
        return model_path
    vlm_config = config.get("bench", {}).get("vlm", {})
    model_id = vlm_config.get("model", "mlx-community/Qwen3.5-27B-mxfp8-mlx")
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id)
    except Exception:
        return model_id


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
    max_tokens: int = 256,
    seed: int = 42,
) -> dict:
    """Run a text-only prompt through the VLM text path."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    messages = prompt["messages"]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(formatted)
    num_prompt_tokens = len(prompt_tokens)

    mx.random.seed(seed)
    sampler = make_sampler(temp=0.0)

    wrapper = _LogitsWrapper(model.language_model)

    completion_tokens = 0
    token_ids = []
    ttft_ms = 0.0
    gen_start = 0.0

    sync_barrier()
    start = time.perf_counter()

    for response in lm_stream_generate(
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

        if hasattr(response, "token"):
            token_ids.append(int(response.token))

        completion_tokens += 1

    sync_barrier()
    end = time.perf_counter()

    if completion_tokens == 0:
        raise RuntimeError(f"No tokens generated for prompt '{prompt['name']}'")

    if completion_tokens > 1 and gen_start > 0:
        gen_time_s = end - gen_start
        gen_tps = (completion_tokens - 1) / gen_time_s if gen_time_s > 0 else 0.0
    else:
        gen_tps = 0.0

    prefill_time_s = ttft_ms / 1000 if ttft_ms > 0 else 0.001
    prefill_tps = num_prompt_tokens / prefill_time_s if prefill_time_s > 0 else 0.0

    fp = fingerprint_output(token_ids) if token_ids else ""

    return {
        "name": prompt["name"],
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": num_prompt_tokens,
        "had_images": False,
        "token_ids": token_ids,
        "fingerprint": fp,
    }


# ---------------------------------------------------------------------------
# Vision VLM benchmark
# ---------------------------------------------------------------------------

def bench_vision_prompt(
    model,
    processor,
    prompt: dict,
    test_image: Image.Image,
    max_tokens: int = 256,
    seed: int = 42,
) -> dict:
    """Run a vision prompt through the VLM vision path (pre-filled cache pattern)."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    messages = prompt["messages"]
    num_images = 1

    sync_barrier()
    img_start = time.perf_counter()

    formatted = vlm_apply_chat_template(
        processor, model.config, messages, num_images=num_images,
    )

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

    vlm_kwargs = dict(extra_kwargs)
    if pixel_values is not None:
        vlm_kwargs["pixel_values"] = pixel_values
    if mask is None:
        mask = mx.ones(input_ids.shape, dtype=mx.int32)
    vlm_kwargs["mask"] = mask

    num_prompt_tokens = input_ids.size

    wrapper = _LogitsWrapper(model.language_model)
    request_cache = make_prompt_cache(wrapper)

    mx.random.seed(seed)
    sampler = make_sampler(temp=0.0)

    # Phase 1: Vision encoding (fills KV cache)
    sync_barrier()
    vision_start = time.perf_counter()

    first_token_id = None
    with wired_limit(model, [mx.default_stream(mx.default_device())]):
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        output = model(input_ids, cache=request_cache, **vlm_kwargs)
        logits = output.logits if hasattr(output, "logits") else output

        last_logits = logits[:, -1, :]
        first_logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
        first_token_id = int(sampler(first_logprobs).item())
        mx.eval(first_logprobs)

    sync_barrier()
    vision_time_ms = (time.perf_counter() - vision_start) * 1000
    ttft_ms = img_time_ms + vision_time_ms

    # Phase 2: Text generation with pre-filled cache
    completion_tokens = 1
    token_ids = [first_token_id]
    gen_start = time.perf_counter()

    for response in lm_stream_generate(
        model=wrapper,
        tokenizer=tokenizer,
        prompt=[first_token_id],
        sampler=sampler,
        max_tokens=max_tokens - 1,
        prompt_cache=request_cache,
    ):
        if hasattr(response, "token"):
            token_ids.append(int(response.token))

        completion_tokens += 1

    sync_barrier()
    end = time.perf_counter()

    if completion_tokens <= 1:
        raise RuntimeError(f"No tokens generated in decode phase for prompt '{prompt['name']}'")

    gen_time_s = end - gen_start
    gen_tps = (completion_tokens - 1) / gen_time_s if gen_time_s > 0 else 0.0

    prefill_time_s = ttft_ms / 1000 if ttft_ms > 0 else 0.001
    prefill_tps = num_prompt_tokens / prefill_time_s if prefill_time_s > 0 else 0.0

    fp = fingerprint_output(token_ids) if token_ids else ""

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
        "token_ids": token_ids,
        "fingerprint": fp,
    }


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    model_path: str,
    runs: int = 3,
    warmup: int = 1,
    max_tokens: int = 256,
    seed: int = 42,
    reset_baseline: bool = False,
    text_only: bool = False,
    vision_only: bool = False,
    scoring_weights: dict | None = None,
    constraints: dict | None = None,
) -> dict:
    """Run the full VLM benchmark suite."""
    ensure_dirs()

    print(f"Loading model: {model_path}", file=sys.stderr)
    model, processor = vlm_load(model_path)
    model_name = Path(model_path).name if "/" not in model_path or Path(model_path).exists() else model_path.split("/")[-1]
    print(f"Model loaded: {model_name}", file=sys.stderr)

    mx.reset_peak_memory()

    simple_img = make_simple_image()
    complex_img = make_complex_image()
    test_images = {"vision_simple": simple_img, "vision_complex": complex_img}

    hardware = get_hardware_info()
    all_prompt_results = []
    all_run_results = []

    prompts_to_run = []
    if not vision_only:
        for p in TEXT_PROMPTS:
            prompts_to_run.append(("text", p, None))
    if not text_only:
        for p in VISION_PROMPTS:
            img = test_images[p["name"]]
            prompts_to_run.append(("vision", p, img))

    for prompt_type, prompt, img in prompts_to_run:
        for w in range(warmup):
            print(f"  warmup {w + 1}/{warmup}: {prompt['name']}...", end="\r", file=sys.stderr)
            if prompt_type == "text":
                bench_text_prompt(model, processor, prompt, max_tokens=max_tokens, seed=seed)
            else:
                bench_vision_prompt(model, processor, prompt, img, max_tokens=max_tokens, seed=seed)
            mx.clear_cache()

        prompt_runs = []
        for r in range(runs):
            print(f"  run {r + 1}/{runs}: {prompt['name']}...   ", end="\r", file=sys.stderr)
            if prompt_type == "text":
                result = bench_text_prompt(model, processor, prompt, max_tokens=max_tokens, seed=seed)
            else:
                result = bench_vision_prompt(model, processor, prompt, img, max_tokens=max_tokens, seed=seed)
            prompt_runs.append(result)
            mx.clear_cache()

        all_run_results.append(prompt_runs)

        avg_result = {
            "name": prompt["name"],
            "gen_tps": sum(r["gen_tps"] for r in prompt_runs) / len(prompt_runs),
            "ttft_ms": sum(r["ttft_ms"] for r in prompt_runs) / len(prompt_runs),
            "prefill_tps": sum(r["prefill_tps"] for r in prompt_runs) / len(prompt_runs),
            "completion_tokens": sum(r["completion_tokens"] for r in prompt_runs) / len(prompt_runs),
            "prompt_tokens": prompt_runs[0]["prompt_tokens"],
            "had_images": prompt_runs[0]["had_images"],
            "fingerprint": prompt_runs[-1]["fingerprint"],
        }
        if prompt_runs[0].get("vision_ms") is not None:
            avg_result["vision_ms"] = sum(r["vision_ms"] for r in prompt_runs) / len(prompt_runs)
            avg_result["img_processing_ms"] = sum(r["img_processing_ms"] for r in prompt_runs) / len(prompt_runs)

        all_prompt_results.append(avg_result)
        extra = f", vision={avg_result.get('vision_ms', 0):.1f}ms" if avg_result.get("vision_ms") else ""
        print(f"  {prompt['name']}: gen={avg_result['gen_tps']:.1f} tps, "
              f"ttft={avg_result['ttft_ms']:.1f}ms, "
              f"prefill={avg_result['prefill_tps']:.1f} tps{extra}, "
              f"fp={avg_result['fingerprint']}", file=sys.stderr)

    avg_gen_tps = sum(r["gen_tps"] for r in all_prompt_results) / len(all_prompt_results)
    avg_ttft_ms = sum(r["ttft_ms"] for r in all_prompt_results) / len(all_prompt_results)
    avg_prefill_tps = sum(r["prefill_tps"] for r in all_prompt_results) / len(all_prompt_results)
    peak_memory_gb = mx.get_peak_memory() / (1024 ** 3)
    total_runs = len(prompts_to_run) * runs

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

    baseline_data = load_baseline(VLM_DIR)
    fingerprint_match = True
    all_violations = []
    suspicion_warnings = []
    variance_warnings = []

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
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb,
            baseline_metrics, weights=scoring_weights,
        )
        avg_completion = sum(r["completion_tokens"] for r in all_prompt_results) / len(all_prompt_results)

        violations = check_hard_constraints(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb,
            avg_completion, baseline_metrics, constraints=constraints,
        )
        all_violations.extend(violations)

        per_prompt_violations = check_per_prompt_constraints(
            all_prompt_results, baseline_data, constraints=constraints,
        )
        all_violations.extend(per_prompt_violations)

        fp_violations = check_fingerprints(all_prompt_results, baseline_data)
        if fp_violations:
            fingerprint_match = False
            all_violations.extend(fp_violations)

        suspicion_warnings = check_suspicion(composite_score, constraints=constraints)

        transposed_runs = []
        for run_idx in range(runs):
            run_slice = [per_prompt[run_idx] for per_prompt in all_run_results]
            transposed_runs.append(run_slice)
        variance_warnings = check_variance(transposed_runs, constraints=constraints)

        if all_violations:
            print(f"\nHard constraint violations:", file=sys.stderr)
            for v in all_violations:
                print(f"  - {v}", file=sys.stderr)

        if suspicion_warnings:
            print(f"\nSuspicion warnings:", file=sys.stderr)
            for w in suspicion_warnings:
                print(f"  - {w}", file=sys.stderr)

        if variance_warnings:
            print(f"\nVariance warnings:", file=sys.stderr)
            for w in variance_warnings:
                print(f"  - {w}", file=sys.stderr)

        result_data = build_result_data(
            bench="vlm", model=model_name, timestamp=timestamp,
            composite_score=round(composite_score, 4), metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        result_data["fingerprint_match"] = fingerprint_match
        result_data["hard_constraint_violations"] = all_violations
        result_data["suspicion_warnings"] = suspicion_warnings
        result_data["variance_warnings"] = variance_warnings
        save_run(VLM_DIR, result_data, timestamp.replace(":", ""))

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
        fingerprint_match=fingerprint_match if not reset_baseline else None,
        extra_lines=extra_lines if extra_lines else None,
    )

    return result_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vision benchmark (mlx-vlm path)")
    parser.add_argument("--model-path", default=None, help="Model path or HF repo ID")
    parser.add_argument("--runs", type=int, default=None, help="Measured runs per prompt (overrides config)")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup runs per prompt (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generation tokens (overrides config)")
    parser.add_argument("--reset-baseline", action="store_true", help="Re-establish baseline")
    parser.add_argument("--text-only", action="store_true", help="Skip vision prompts")
    parser.add_argument("--vision-only", action="store_true", help="Skip text prompts")
    parser.add_argument("--config", default=None, help="Path to bench_config.toml")
    args = parser.parse_args()

    if args.text_only and args.vision_only:
        print("Cannot specify both --text-only and --vision-only", file=sys.stderr)
        sys.exit(1)

    config = load_config(Path(args.config) if args.config else None)
    bench_params = get_bench_params(config)
    scoring_weights = get_scoring_weights(config)
    constraints = get_constraints(config)

    runs = args.runs if args.runs is not None else bench_params["runs"]
    warmup = args.warmup if args.warmup is not None else bench_params["warmup"]
    max_tokens = args.max_tokens if args.max_tokens is not None else bench_params["max_tokens"]
    seed = bench_params["seed"]

    model_path = resolve_model_path(args.model_path, config)
    run_benchmark(
        model_path=model_path,
        runs=runs,
        warmup=warmup,
        max_tokens=max_tokens,
        seed=seed,
        reset_baseline=args.reset_baseline,
        text_only=args.text_only,
        vision_only=args.vision_only,
        scoring_weights=scoring_weights,
        constraints=constraints,
    )


if __name__ == "__main__":
    main()
