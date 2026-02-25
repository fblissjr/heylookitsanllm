#!/usr/bin/env python3
"""HTTP-based benchmark for heylookitsanllm server.

Measures TTFT (streaming), generation TPS, and memory against a running server.
Supports both OpenAI and Messages API endpoints, streaming and non-streaming.

Usage:
    uv run scripts/benchmark.py                        # defaults
    uv run scripts/benchmark.py --url http://localhost:8080
    uv run scripts/benchmark.py --model <model-id>
    uv run scripts/benchmark.py --prompts short,medium
    uv run scripts/benchmark.py --mode streaming
    uv run scripts/benchmark.py --runs 3 --warmup 1
    uv run scripts/benchmark.py --endpoint messages
    uv run scripts/benchmark.py --json
"""

import argparse
import sys
import time
from dataclasses import dataclass, field

import requests

try:
    import orjson

    def json_loads(data):
        return orjson.loads(data)

    def json_dumps(data):
        return orjson.dumps(data).decode()
except ImportError:
    import json

    json_loads = json.loads
    json_dumps = json.dumps


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptSet:
    name: str
    system_prompt: str
    user_message: str


@dataclass
class BenchmarkResult:
    prompt_name: str
    mode: str
    ttft_ms: float  # streaming only, 0 for non-streaming
    total_time_ms: float
    generation_tps: float
    prompt_tokens: int
    completion_tokens: int
    server_reported_tps: float


@dataclass
class BenchmarkConfig:
    base_url: str = "http://localhost:8080"
    model: str = ""
    prompt_names: list[str] = field(default_factory=lambda: ["short", "medium", "long", "code"])
    modes: list[str] = field(default_factory=lambda: ["streaming", "non-streaming"])
    runs: int = 3
    warmup: int = 1
    max_tokens: int = 256
    endpoint: str = "openai"
    seed: int | None = None
    json_output: bool = False


# ---------------------------------------------------------------------------
# Built-in prompt sets
# ---------------------------------------------------------------------------

PROMPT_SETS = {
    "short": PromptSet(
        name="short",
        system_prompt="You are a helpful assistant. Be concise.",
        user_message="What is 7 * 13?",
    ),
    "medium": PromptSet(
        name="medium",
        system_prompt="You are a helpful assistant.",
        user_message="Explain how hash tables work in 3-4 sentences.",
    ),
    "long": PromptSet(
        name="long",
        system_prompt="You are a technical writer.",
        user_message="Write a short tutorial on Python decorators. Cover what they are, how to write one, and give two practical examples.",
    ),
    "code": PromptSet(
        name="code",
        system_prompt="You are an expert Python programmer. Write clean, working code.",
        user_message="Implement a thread-safe LRU cache class in Python with get() and put() methods.",
    ),
}


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------

def discover_model(base_url: str) -> str:
    """Auto-discover the first available model from GET /v1/models."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        data = json_loads(resp.content)
        models = data.get("data", [])
        if models:
            model_id = models[0]["id"]
            return model_id
    except Exception as e:
        print(f"Failed to discover model: {e}", file=sys.stderr)
    return ""


def get_memory_snapshot(base_url: str) -> dict | None:
    """Get system metrics snapshot."""
    try:
        resp = requests.get(f"{base_url}/v1/system/metrics?force_refresh=true", timeout=10)
        resp.raise_for_status()
        return json_loads(resp.content)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenAI endpoint runners
# ---------------------------------------------------------------------------

def run_openai_streaming(base_url: str, model: str, prompt: PromptSet, config: BenchmarkConfig) -> BenchmarkResult:
    """Run a streaming request against /v1/chat/completions."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_message},
        ],
        "stream": True,
        "max_tokens": config.max_tokens,
        "stream_options": {"include_usage": True},
    }
    if config.seed is not None:
        payload["seed"] = config.seed

    start = time.perf_counter()
    ttft = 0.0
    completion_tokens = 0
    prompt_tokens = 0
    server_tps = 0.0
    first_token_received = False

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json_loads(data_str)
        except Exception:
            continue

        # Check for content delta (first token)
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if (delta.get("content") or delta.get("thinking")) and not first_token_received:
                ttft = (time.perf_counter() - start) * 1000
                first_token_received = True

        # Capture usage from final chunk
        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        # Capture server TPS from performance field
        perf = chunk.get("performance")
        if perf:
            server_tps = perf.get("generation_tps", 0.0)

    total_time = (time.perf_counter() - start) * 1000
    gen_time_s = (total_time - ttft) / 1000 if ttft > 0 else total_time / 1000
    client_tps = completion_tokens / gen_time_s if gen_time_s > 0 and completion_tokens > 0 else 0.0

    return BenchmarkResult(
        prompt_name=prompt.name,
        mode="streaming",
        ttft_ms=round(ttft, 1),
        total_time_ms=round(total_time, 1),
        generation_tps=round(client_tps, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        server_reported_tps=round(server_tps, 1),
    )


def run_openai_non_streaming(base_url: str, model: str, prompt: PromptSet, config: BenchmarkConfig) -> BenchmarkResult:
    """Run a non-streaming request against /v1/chat/completions."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_message},
        ],
        "stream": False,
        "max_tokens": config.max_tokens,
        "include_performance": True,
    }
    if config.seed is not None:
        payload["seed"] = config.seed

    start = time.perf_counter()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    total_time = (time.perf_counter() - start) * 1000
    data = json_loads(resp.content)

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    server_tps = 0.0
    perf = data.get("performance")
    if perf:
        server_tps = perf.get("generation_tps", 0.0)

    gen_time_s = total_time / 1000
    client_tps = completion_tokens / gen_time_s if gen_time_s > 0 and completion_tokens > 0 else 0.0

    return BenchmarkResult(
        prompt_name=prompt.name,
        mode="non-streaming",
        ttft_ms=0.0,
        total_time_ms=round(total_time, 1),
        generation_tps=round(client_tps, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        server_reported_tps=round(server_tps, 1),
    )


# ---------------------------------------------------------------------------
# Messages endpoint runners
# ---------------------------------------------------------------------------

def run_messages_streaming(base_url: str, model: str, prompt: PromptSet, config: BenchmarkConfig) -> BenchmarkResult:
    """Run a streaming request against /v1/messages."""
    payload = {
        "model": model,
        "system": prompt.system_prompt,
        "messages": [{"role": "user", "content": prompt.user_message}],
        "stream": True,
        "max_tokens": config.max_tokens,
    }

    start = time.perf_counter()
    ttft = 0.0
    first_token_received = False
    prompt_tokens = 0
    completion_tokens = 0

    resp = requests.post(f"{base_url}/v1/messages", json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    event_type = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            event_type = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        try:
            chunk = json_loads(data_str)
        except Exception:
            continue

        # First content delta = TTFT
        if event_type == "content_block_delta" and not first_token_received:
            ttft = (time.perf_counter() - start) * 1000
            first_token_received = True

        # Capture usage from message_delta
        if event_type == "message_delta":
            usage = chunk.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

    total_time = (time.perf_counter() - start) * 1000
    gen_time_s = (total_time - ttft) / 1000 if ttft > 0 else total_time / 1000
    client_tps = completion_tokens / gen_time_s if gen_time_s > 0 and completion_tokens > 0 else 0.0

    return BenchmarkResult(
        prompt_name=prompt.name,
        mode="streaming",
        ttft_ms=round(ttft, 1),
        total_time_ms=round(total_time, 1),
        generation_tps=round(client_tps, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        server_reported_tps=0.0,
    )


def run_messages_non_streaming(base_url: str, model: str, prompt: PromptSet, config: BenchmarkConfig) -> BenchmarkResult:
    """Run a non-streaming request against /v1/messages."""
    payload = {
        "model": model,
        "system": prompt.system_prompt,
        "messages": [{"role": "user", "content": prompt.user_message}],
        "stream": False,
        "max_tokens": config.max_tokens,
    }

    start = time.perf_counter()
    resp = requests.post(f"{base_url}/v1/messages", json=payload, timeout=120)
    resp.raise_for_status()
    total_time = (time.perf_counter() - start) * 1000
    data = json_loads(resp.content)

    usage = data.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    server_tps = 0.0
    perf = data.get("performance")
    if perf:
        server_tps = perf.get("generation_tps", 0.0)

    gen_time_s = total_time / 1000
    client_tps = completion_tokens / gen_time_s if gen_time_s > 0 and completion_tokens > 0 else 0.0

    return BenchmarkResult(
        prompt_name=prompt.name,
        mode="non-streaming",
        ttft_ms=0.0,
        total_time_ms=round(total_time, 1),
        generation_tps=round(client_tps, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        server_reported_tps=round(server_tps, 1),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def select_runner(endpoint: str, mode: str):
    """Return the appropriate runner function."""
    runners = {
        ("openai", "streaming"): run_openai_streaming,
        ("openai", "non-streaming"): run_openai_non_streaming,
        ("messages", "streaming"): run_messages_streaming,
        ("messages", "non-streaming"): run_messages_non_streaming,
    }
    return runners[(endpoint, mode)]


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Execute the full benchmark suite."""
    prompts = [PROMPT_SETS[name] for name in config.prompt_names if name in PROMPT_SETS]
    if not prompts:
        print(f"No valid prompts found. Available: {', '.join(PROMPT_SETS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Auto-discover model if needed
    model = config.model or discover_model(config.base_url)
    if not model:
        print("No model available. Specify --model or ensure server has a model loaded.", file=sys.stderr)
        sys.exit(1)

    results: list[BenchmarkResult] = []

    for mode in config.modes:
        runner = select_runner(config.endpoint, mode)
        for prompt in prompts:
            # Warmup runs
            for i in range(config.warmup):
                if not config.json_output:
                    print(f"  warmup {i + 1}/{config.warmup}: {prompt.name} ({mode})...", end="\r")
                try:
                    runner(config.base_url, model, prompt, config)
                except Exception as e:
                    print(f"  warmup failed: {e}", file=sys.stderr)

            # Measured runs
            run_results: list[BenchmarkResult] = []
            for i in range(config.runs):
                if not config.json_output:
                    print(f"  run {i + 1}/{config.runs}: {prompt.name} ({mode})...   ", end="\r")
                try:
                    result = runner(config.base_url, model, prompt, config)
                    run_results.append(result)
                except Exception as e:
                    print(f"  run {i + 1} failed: {e}", file=sys.stderr)

            if run_results:
                # Average results
                avg = BenchmarkResult(
                    prompt_name=prompt.name,
                    mode=mode,
                    ttft_ms=round(sum(r.ttft_ms for r in run_results) / len(run_results), 1),
                    total_time_ms=round(sum(r.total_time_ms for r in run_results) / len(run_results), 1),
                    generation_tps=round(sum(r.generation_tps for r in run_results) / len(run_results), 1),
                    prompt_tokens=run_results[-1].prompt_tokens,
                    completion_tokens=round(sum(r.completion_tokens for r in run_results) / len(run_results)),
                    server_reported_tps=round(sum(r.server_reported_tps for r in run_results) / len(run_results), 1),
                )
                results.append(avg)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(results: list[BenchmarkResult], config: BenchmarkConfig, memory_before: dict | None, memory_after: dict | None):
    """Print results as a rich table, or plain text if rich is unavailable."""
    try:
        from rich.console import Console
        from rich.table import Table
        _print_rich_table(results, config, memory_before, memory_after)
    except ImportError:
        _print_plain_table(results, config, memory_before, memory_after)


def _print_rich_table(results: list[BenchmarkResult], config: BenchmarkConfig, memory_before: dict | None, memory_after: dict | None):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Benchmark Results ({config.endpoint} endpoint, {config.runs} runs)")
    table.add_column("Prompt", style="cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Total (ms)", justify="right")
    table.add_column("Client TPS", justify="right", style="green")
    table.add_column("Server TPS", justify="right", style="blue")
    table.add_column("Tokens (p/c)", justify="right")

    for r in results:
        ttft = f"{r.ttft_ms:.0f}" if r.ttft_ms > 0 else "-"
        table.add_row(
            r.prompt_name,
            r.mode,
            ttft,
            f"{r.total_time_ms:.0f}",
            f"{r.generation_tps:.1f}",
            f"{r.server_reported_tps:.1f}" if r.server_reported_tps > 0 else "-",
            f"{r.prompt_tokens}/{r.completion_tokens}",
        )

    console.print(table)

    # Memory info
    if memory_before and memory_after:
        sys_before = memory_before.get("system", {})
        sys_after = memory_after.get("system", {})
        console.print(f"\nMemory: {sys_before.get('ram_used_gb', 0):.1f}GB -> {sys_after.get('ram_used_gb', 0):.1f}GB used "
                      f"(of {sys_after.get('ram_total_gb', 0):.0f}GB total)")


def _print_plain_table(results: list[BenchmarkResult], config: BenchmarkConfig, memory_before: dict | None, memory_after: dict | None):
    header = f"{'Prompt':<10} {'Mode':<15} {'TTFT':>8} {'Total':>8} {'C-TPS':>8} {'S-TPS':>8} {'Tokens':>12}"
    print(f"\nBenchmark Results ({config.endpoint} endpoint, {config.runs} runs)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms > 0 else "-"
        print(f"{r.prompt_name:<10} {r.mode:<15} {ttft:>8} {r.total_time_ms:.0f}ms{'':<3} {r.generation_tps:>7.1f} "
              f"{r.server_reported_tps:>7.1f} {r.prompt_tokens:>5}/{r.completion_tokens:<5}")
    print("-" * len(header))

    if memory_before and memory_after:
        sys_before = memory_before.get("system", {})
        sys_after = memory_after.get("system", {})
        print(f"\nMemory: {sys_before.get('ram_used_gb', 0):.1f}GB -> {sys_after.get('ram_used_gb', 0):.1f}GB used "
              f"(of {sys_after.get('ram_total_gb', 0):.0f}GB total)")


def print_json_output(results: list[BenchmarkResult], config: BenchmarkConfig, memory_before: dict | None, memory_after: dict | None):
    """Print results as JSON."""
    output = {
        "config": {
            "base_url": config.base_url,
            "model": config.model,
            "endpoint": config.endpoint,
            "runs": config.runs,
            "warmup": config.warmup,
            "max_tokens": config.max_tokens,
        },
        "results": [
            {
                "prompt": r.prompt_name,
                "mode": r.mode,
                "ttft_ms": r.ttft_ms,
                "total_time_ms": r.total_time_ms,
                "generation_tps": r.generation_tps,
                "server_reported_tps": r.server_reported_tps,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            for r in results
        ],
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    print(json_dumps(output))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark heylookitsanllm server")
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--model", default="", help="Model ID (auto-discovers if omitted)")
    parser.add_argument("--prompts", default="short,medium,long,code", help="Comma-separated prompt names")
    parser.add_argument("--mode", default="both", choices=["streaming", "non-streaming", "both"], help="Request mode")
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs per prompt")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs per prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--endpoint", default="openai", choices=["openai", "messages"], help="API endpoint")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    args = parser.parse_args()

    modes = {"both": ["streaming", "non-streaming"], "streaming": ["streaming"], "non-streaming": ["non-streaming"]}

    return BenchmarkConfig(
        base_url=args.url.rstrip("/"),
        model=args.model,
        prompt_names=[p.strip() for p in args.prompts.split(",")],
        modes=modes[args.mode],
        runs=args.runs,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        endpoint=args.endpoint,
        seed=args.seed,
        json_output=args.json,
    )


def main():
    config = parse_args()

    # Check server is reachable
    try:
        resp = requests.get(f"{config.base_url}/v1/models", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"Cannot reach server at {config.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Auto-discover model
    if not config.model:
        config.model = discover_model(config.base_url)
        if not config.model:
            print("No model available on server.", file=sys.stderr)
            sys.exit(1)

    if not config.json_output:
        print(f"Benchmarking {config.base_url}")
        print(f"Model: {config.model}")
        print(f"Endpoint: {config.endpoint} | Modes: {', '.join(config.modes)}")
        print(f"Runs: {config.runs} | Warmup: {config.warmup} | Max tokens: {config.max_tokens}")
        print()

    memory_before = get_memory_snapshot(config.base_url)
    results = run_benchmark(config)
    memory_after = get_memory_snapshot(config.base_url)

    if not results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    if config.json_output:
        print_json_output(results, config, memory_before, memory_after)
    else:
        print()  # clear the progress line
        print_results_table(results, config, memory_before, memory_after)


if __name__ == "__main__":
    main()
