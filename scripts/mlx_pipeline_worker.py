#!/usr/bin/env python3
"""Distributed MLX pipeline worker (and optional 2-rank smoke runner).

When ``heylookllm`` serves an MLX model with ``pipeline_ratios = [0.8, 0.2]``
under ``mlx.launch``, Rank 0 runs the HTTP server and Rank >= 1 runs the
lockstep forward-step loop (:func:`run_pipeline_worker_loop`) waiting for shape
headers broadcast from Rank 0 (:func:`install_rank0_step_coordinator`).

Usage (2-rank loopback smoke test on one machine):
    mlx.launch -n 2 uv run python scripts/mlx_pipeline_worker.py \\
        --model-path /path/to/mlx-model \\
        --ratios 0.8,0.2 \\
        --smoke-prompt "Explain pipeline parallelism in one sentence."

Usage (worker-only on Rank >= 1):
    uv run python scripts/mlx_pipeline_worker.py \\
        --model-path /path/to/mlx-model \\
        --ratios 0.8,0.2
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlx.core as mx

from heylook_llm.providers.common.mlx_pipeline import (
    enable_fast_metal_synch,
    install_rank0_step_coordinator,
    run_pipeline_worker_loop,
    selective_pipeline_load,
    send_pipeline_shutdown,
)


def _parse_ratios(raw: str) -> list[float]:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) < 2 or any(p <= 0 for p in parts):
        raise ValueError(f"--ratios must be >= 2 positive floats, got {raw!r}")
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the MLX model directory (must be readable on all ranks).",
    )
    parser.add_argument(
        "--ratios",
        default="0.8,0.2",
        help="Comma-separated per-rank layer split ratios (default: 0.8,0.2).",
    )
    parser.add_argument(
        "--backend",
        default="any",
        choices=("any", "ring", "jaccl"),
        help="mlx.distributed backend (default: any).",
    )
    parser.add_argument(
        "--smoke-prompt",
        default=None,
        help="If set, Rank 0 runs a greedy generation for this prompt and then shuts down Rank 1.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate in --smoke-prompt mode (default: 32).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    enable_fast_metal_synch()
    group = mx.distributed.init(strict=False, backend=args.backend)
    ratios = _parse_ratios(args.ratios)

    logging.info(
        "rank=%d/%d loading %s with ratios=%s",
        group.rank(),
        group.size(),
        args.model_path,
        ratios,
    )
    model, processor = selective_pipeline_load(
        args.model_path,
        ratios=ratios,
        group=group,
        skip_audio=True,
    )

    if group.rank() > 0:
        logging.info("rank=%d entering pipeline worker loop", group.rank())
        steps = run_pipeline_worker_loop(model, group=group)
        logging.info("rank=%d exited cleanly after %d steps", group.rank(), steps)
        return 0

    if args.smoke_prompt is not None:
        install_rank0_step_coordinator(model, group=group)
        try:
            tokenizer = getattr(processor, "tokenizer", processor)
            ids = mx.array([list(tokenizer.encode(args.smoke_prompt))], dtype=mx.int32)
            lm = getattr(model, "language_model", model)
            cache = lm.make_cache()
            out_tokens: list[int] = []
            cur = ids
            for _ in range(args.max_tokens):
                out = lm(cur, cache=cache)
                logits = out.logits if hasattr(out, "logits") else out
                next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                out_tokens.append(next_tok)
                cur = mx.array([[next_tok]], dtype=mx.int32)
            text = tokenizer.decode(out_tokens)
            print(text)
        finally:
            send_pipeline_shutdown(group=group)
        return 0

    logging.info(
        "rank=0 loaded model with ratios=%s; pass --smoke-prompt to run a "
        "standalone check or configure pipeline_ratios in model.heylook.toml "
        "when serving via heylookllm.",
        ratios,
    )
    if group.size() > 1:
        send_pipeline_shutdown(group=group)
    return 0


if __name__ == "__main__":
    sys.exit(main())
