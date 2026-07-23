#!/usr/bin/env python3
# LLM behavior-eval harness. Generalizes two ad-hoc debugging scripts
# (full_matrix.py, repro_multiimage.py) into a reusable, filterable tool for
# catching behavior regressions -- chat template changes, stop-token changes,
# vision pipeline changes -- that unit/contract tests can't see because they
# need a real model actually decoding.
#
# This tool NEVER spawns a server. Point --server at an already-running
# `heylookllm` instance (models load on-demand via the server's own router --
# no separate preload step needed). Opt-in, not part of /test-suite: see
# README.md.
#
# Usage:
#   uv run python tests/eval/run.py --list-tasks
#   uv run python tests/eval/run.py --server http://localhost:8080 \
#       --models gemma-4-31b-it-8bit-mlx,Qwen3.5-27B-8bit-mlx
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tasks import TASKS

GREEN = "\x1b[32m"
RED = "\x1b[31m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"

DEFAULT_SERVER = "http://localhost:8080"  # this repo's conventional dev port (README.md)
DEFAULT_OUT = Path(__file__).parent / "results.jsonl"


def fetch_models(server: str) -> dict[str, set[str]]:
    """model_id -> capability set, from GET /v1/models. A model with no
    `capabilities` key (or an empty list) is treated as text/chat only."""
    req = urllib.request.Request(f"{server}/v1/models")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    return {entry["id"]: set(entry.get("capabilities") or []) for entry in body.get("data", [])}


def run_task(server: str, model: str, task) -> dict:
    """POST one task's request, judge the response, return a JSONL-ready
    result dict. Any exception (timeout, connection error, non-200, bad JSON
    shape) is caught here so one bad task never aborts the whole run."""
    start = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    body = task.build_request()
    body["model"] = model
    body.setdefault("stream", False)
    try:
        req = urllib.request.Request(
            f"{server}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=task.timeout) as resp:
            r = json.load(resp)
        message = r["choices"][0]["message"]
        usage = r.get("usage") or {}
        ctx = {
            "content": message.get("content") or "",
            "thinking": message.get("thinking"),
            "completion_tokens": usage.get("completion_tokens"),
            "max_tokens": body.get("max_tokens"),
            "finish_reason": r["choices"][0].get("finish_reason"),
        }
        verdict = task.judge(ctx)
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "model": model,
            "task": task.name,
            "category": task.category,
            "passed": verdict.passed,
            "evidence": verdict.evidence,
            "elapsed_ms": round(elapsed_ms, 1),
            "timestamp": timestamp,
        }
    except Exception as e:  # noqa: BLE001 -- a bad task must not abort the run
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "model": model,
            "task": task.name,
            "category": task.category,
            "passed": False,
            "evidence": "request failed",
            "error": str(e),
            "elapsed_ms": round(elapsed_ms, 1),
            "timestamp": timestamp,
        }


def print_result(result: dict) -> None:
    ok = result["passed"]
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    ms = f"{DIM}({result['elapsed_ms']:.0f}ms){RESET}"
    print(f"  {mark} {result['model']:32} {result['task']:36} {ms}")
    detail = result.get("error") or result["evidence"]
    color = "" if ok else RED
    print(f"      {color}{detail}{RESET}")


def print_summary(results: list[dict]) -> None:
    print()
    print("Summary")
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in by_model.items():
        passed = sum(1 for r in rs if r["passed"])
        color = GREEN if passed == len(rs) else RED
        print(f"  {color}{model}: {passed}/{len(rs)} passed{RESET}")
    total_passed = sum(1 for r in results if r["passed"])
    print(f"\n{total_passed}/{len(results)} checks passed")


def list_tasks() -> None:
    for t in TASKS:
        caps = ", ".join(t.required_capabilities) or "none"
        print(f"{t.name}  [{t.category}]  requires: {caps}")
        print(f"    {t.description}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in LLM behavior-eval harness (never spawns a server).")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"already-running heylookllm base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--models", help="comma-separated model ids (as returned by /v1/models)")
    parser.add_argument("--tasks", help="comma-separated category filter: vision,thinking,stop,text")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"JSONL results path (default: {DEFAULT_OUT})")
    parser.add_argument("--list-tasks", action="store_true", help="print the task bank and exit (no network calls)")
    args = parser.parse_args()

    if args.list_tasks:
        list_tasks()
        return 0

    if not args.models:
        parser.error("--models is required unless --list-tasks is given")

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    category_filter = {c.strip() for c in args.tasks.split(",")} if args.tasks else None
    tasks = [t for t in TASKS if category_filter is None or t.category in category_filter]
    if not tasks:
        print(f"No tasks match --tasks filter {args.tasks!r}", file=sys.stderr)
        return 1

    try:
        capability_map = fetch_models(args.server)
    except Exception as e:
        print(f"Failed to reach {args.server}/v1/models: {e}", file=sys.stderr)
        print("This harness never spawns a server -- point --server at an already-running instance.", file=sys.stderr)
        return 1

    results: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_f:
        for model in requested_models:
            if model not in capability_map:
                print(f"{RED}WARNING{RESET}: model {model!r} not found in {args.server}/v1/models -- skipping", file=sys.stderr)
                continue
            model_caps = capability_map[model]
            print(f"\n{model}")
            for task in tasks:
                if not set(task.required_capabilities) <= model_caps:
                    continue  # model lacks a required capability -- silently not applicable
                result = run_task(args.server, model, task)
                results.append(result)
                print_result(result)
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

    if not results:
        print("No applicable (model, task) pairs ran.", file=sys.stderr)
        return 1

    print_summary(results)
    print(f"\nResults written to {out_path}")
    return 1 if any(not r["passed"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
