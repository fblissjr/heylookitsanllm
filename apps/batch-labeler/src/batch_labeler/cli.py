"""CLI for batch VLM image labeling.

Subcommands:
    run     label every image in a directory (resumable JSONL output)
    try     label one image and pretty-print the result (prompt iteration)
    models  list models on the server, marking vision/thinking capability
    tasks   list built-in tasks or show a task's full prompts
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import orjson
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich.text import Text
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import scanner, storage
from .client import (
    GenerationOptions,
    ServerError,
    fetch_models,
    is_vision_model,
    label_image,
    pick_vision_model,
    vision_models,
)
from .tasks import BUILTIN_TASKS, Task, get_task, load_task_file, missing_required_keys

DEFAULT_SERVER = os.environ.get("BATCH_LABELER_SERVER", "http://localhost:8080")


# ---------------------------------------------------------------- arg parsing

def _add_server_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Server base URL (default: {DEFAULT_SERVER}; env BATCH_LABELER_SERVER)",
    )


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model & task")
    g.add_argument(
        "--model", "-m",
        help="Model ID (default: the server's sole vision model, if unambiguous)",
    )
    g.add_argument(
        "--task", "-t",
        default="label",
        help=f"Built-in task: {', '.join(sorted(BUILTIN_TASKS))} (default: label)",
    )
    g.add_argument("--task-file", type=Path, help="Custom task TOML (overrides --task)")
    g.add_argument("--system-prompt", help="Override the task's system prompt")
    g.add_argument(
        "--system-prompt-file", type=Path, help="Override system prompt from a file"
    )
    g.add_argument("--user-prompt", help="Override the task's per-image user prompt")

    s = p.add_argument_group("generation")
    s.add_argument(
        "--sampler", "--preset", dest="sampler",
        help="Server named sampler (overrides the task's; --preset is an alias)",
    )
    s.add_argument("--max-tokens", type=int, help="Max tokens per response")
    s.add_argument("--temperature", type=float, help="Sampling temperature")
    s.add_argument("--top-p", type=float, help="Nucleus sampling threshold")
    s.add_argument("--seed", type=int, help="Sampling seed for reproducibility")
    think = s.add_mutually_exclusive_group()
    think.add_argument(
        "--think", dest="enable_thinking", action="store_true", default=None,
        help="Enable thinking mode (thinking-capable models)",
    )
    think.add_argument(
        "--no-think", dest="enable_thinking", action="store_false",
        help="Explicitly disable thinking mode",
    )

    v = p.add_argument_group("vision")
    v.add_argument(
        "--vision-tokens", type=int,
        help="Visual token budget per image (16-16384; snapped to the model's grid)",
    )
    v.add_argument(
        "--resize-max", type=int,
        help="Server-side resize to max dimension before encoding (e.g. 1024)",
    )
    v.add_argument(
        "--image-quality", type=int,
        help="JPEG quality for server-side resized images (1-100)",
    )

    p.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--retries", type=int, default=2,
        help="Retries per image on timeout/connection/5xx (default: 2)",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch-labeler",
        description="Label images with a VLM served by heylookitsanllm (or any OpenAI-compatible server).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Label every image in a directory (resumable)")
    run.add_argument("image_dir", help="Directory containing images")
    run.add_argument(
        "--output", "-o", default="results.jsonl",
        help="Output JSONL file (default: results.jsonl)",
    )
    run.add_argument(
        "--no-recursive", action="store_false", dest="recursive", default=True,
        help="Only scan the top-level directory",
    )
    run.add_argument("--limit", type=int, help="Process at most N images (sampling)")
    run.add_argument(
        "--dry-run", action="store_true",
        help="Scan, report counts, and show the resolved config without processing",
    )
    _add_server_arg(run)
    _add_generation_args(run)

    try_p = sub.add_parser("try", help="Label ONE image and pretty-print the result")
    try_p.add_argument("image", help="Path to a single image")
    _add_server_arg(try_p)
    _add_generation_args(try_p)

    models = sub.add_parser("models", help="List server models (vision marked)")
    _add_server_arg(models)

    tasks = sub.add_parser("tasks", help="List built-in tasks or show one")
    tasks.add_argument("name", nargs="?", help="Task name to show in full")

    return p


# ---------------------------------------------------------------- resolution

def _resolve_task(args, console: Console) -> Task | None:
    """Task file > --task name; then apply prompt overrides."""
    try:
        if args.task_file:
            task = load_task_file(args.task_file)
        else:
            task = get_task(args.task)
    except (KeyError, ValueError, OSError) as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        return None

    overrides = {}
    if args.system_prompt_file:
        if not args.system_prompt_file.exists():
            console.print(f"[red]System prompt file not found: {args.system_prompt_file}[/red]")
            return None
        overrides["system_prompt"] = args.system_prompt_file.read_text().strip()
    elif args.system_prompt:
        overrides["system_prompt"] = args.system_prompt
    if args.user_prompt:
        overrides["user_prompt"] = args.user_prompt

    if overrides:
        from dataclasses import replace
        task = replace(task, **overrides)
    return task


def _resolve_options(args, task: Task) -> GenerationOptions:
    return GenerationOptions(
        max_tokens=args.max_tokens if args.max_tokens is not None else task.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        sampler=args.sampler if args.sampler is not None else task.sampler,
        enable_thinking=args.enable_thinking,
        vision_tokens=args.vision_tokens,
        resize_max=args.resize_max,
        image_quality=args.image_quality,
    )


def _resolve_model(args, http_client: httpx.Client, console: Console) -> str | None:
    """Validate --model against the server, or auto-pick the sole vision model."""
    try:
        models = fetch_models(http_client)
    except ServerError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        return None

    if args.model:
        match = next((m for m in models if m.get("id") == args.model), None)
        if match is None:
            console.print(f"[red]Model {args.model!r} not on server.[/red]")
            _print_models_table(models, console)
            return None
        if not is_vision_model(match):
            console.print(
                f"[yellow]Warning: {args.model!r} does not advertise vision capability.[/yellow]"
            )
        return args.model

    picked = pick_vision_model(models)
    if picked:
        console.print(f"Auto-selected vision model: [bold]{picked}[/bold]")
        return picked

    vlms = vision_models(models)
    if not vlms:
        console.print("[red]No vision-capable models on the server.[/red]")
    else:
        console.print("[red]Multiple vision models available; pick one with --model:[/red]")
        _print_models_table(vlms, console)
    return None


def _settings_echo(model_id: str, task: Task, options: GenerationOptions) -> dict:
    """Compact non-None settings dict stored with each record for reproducibility."""
    settings = {"model": model_id, "task": task.name}
    for key in (
        "sampler", "max_tokens", "temperature", "top_p", "seed",
        "enable_thinking", "vision_tokens", "resize_max", "image_quality",
    ):
        value = getattr(options, key)
        if value is not None:
            settings[key] = value
    return settings


def _print_models_table(models: list[dict], console: Console) -> None:
    table = Table(box=None, pad_edge=False)
    table.add_column("model")
    table.add_column("provider")
    table.add_column("capabilities")
    for m in models:
        caps = set(m.get("capabilities") or [])
        caps.update(m.get("modalities") or [])
        caps.discard("text")
        marks = " ".join(sorted(caps)) or "-"
        style = "bold green" if is_vision_model(m) else ""
        table.add_row(m.get("id", "?"), m.get("provider", "?"), marks, style=style)
    console.print(table)


def _http_client(args) -> httpx.Client:
    return httpx.Client(
        base_url=args.server,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )


# ---------------------------------------------------------------- subcommands

def cmd_models(args, console: Console) -> int:
    with httpx.Client(base_url=args.server, timeout=httpx.Timeout(10.0)) as hc:
        try:
            models = fetch_models(hc)
        except ServerError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
            return 1
    if not models:
        console.print("No models on the server.")
        return 0
    _print_models_table(models, console)
    console.print(
        f"\n{len(vision_models(models))} vision-capable of {len(models)} total "
        "(vision models in green)"
    )
    return 0


def cmd_tasks(args, console: Console) -> int:
    if args.name:
        try:
            task = get_task(args.name)
        except KeyError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
            return 1
        console.print(Panel(Text(task.system_prompt), title=f"{task.name} -- system prompt"))
        console.print(Panel(Text(task.user_prompt), title="user prompt (per image)"))
        meta = f"expects_json={task.expects_json}"
        if task.required_keys:
            meta += f"  required_keys={list(task.required_keys)}"
        if task.sampler:
            meta += f"  sampler={task.sampler}"
        if task.max_tokens:
            meta += f"  max_tokens={task.max_tokens}"
        console.print(meta)
        return 0

    table = Table(box=None, pad_edge=False)
    table.add_column("task")
    table.add_column("output")
    table.add_column("sampler")
    table.add_column("description")
    for task in BUILTIN_TASKS.values():
        table.add_row(
            task.name,
            "json" if task.expects_json else "text",
            task.sampler or "-",
            task.description,
        )
    console.print(table)
    console.print("\nShow one in full: batch-labeler tasks <name>")
    console.print(r"Custom tasks: --task-file my_task.toml (same fields, \[task] table)")
    return 0


def cmd_try(args, console: Console) -> int:
    image = Path(args.image)
    if not image.is_file():
        console.print(f"[red]Image not found: {image}[/red]")
        return 1
    task = _resolve_task(args, console)
    if task is None:
        return 1

    with _http_client(args) as hc:
        model_id = _resolve_model(args, hc, console)
        if model_id is None:
            return 1
        options = _resolve_options(args, task)
        console.print(f"Task [bold]{task.name}[/bold] -> {model_id}")
        with console.status("Generating..."):
            try:
                result = label_image(
                    hc, model_id, task.system_prompt, task.user_prompt,
                    image, options, retries=args.retries,
                )
            except Exception as e:
                console.print(f"[red]Request failed: {escape(str(e))}[/red]")
                return 1

    if result.thinking:
        console.print(Panel(Text(result.thinking), title="thinking", style="dim"))
    if task.expects_json:
        label_json = storage.extract_json(result.content)
        if label_json is None:
            console.print("[yellow]Output is not valid JSON:[/yellow]")
            console.print(Panel(Text(result.content), title="raw output"))
        else:
            parsed = orjson.loads(label_json)
            console.print(Panel(
                Text(orjson.dumps(parsed, option=orjson.OPT_INDENT_2).decode()),
                title="label",
            ))
            missing = missing_required_keys(parsed, task)
            if missing:
                console.print(f"[yellow]Missing required keys: {missing}[/yellow]")
    else:
        console.print(Panel(Text(result.content), title="output"))

    stats = f"{result.request_ms} ms"
    usage = result.usage or {}
    if usage.get("completion_tokens"):
        stats += f"  ·  {usage.get('prompt_tokens', '?')} -> {usage['completion_tokens']} tokens"
    if result.performance and result.performance.get("generation_tps"):
        stats += f"  ·  {result.performance['generation_tps']:.1f} tok/s"
    console.print(stats)
    return 0


def cmd_run(args, console: Console) -> int:
    if not Path(args.image_dir).is_dir():
        console.print(f"[red]Image directory not found: {args.image_dir}[/red]")
        return 1
    task = _resolve_task(args, console)
    if task is None:
        return 1

    console.print(f"Scanning {args.image_dir}...")
    images = scanner.scan_images(args.image_dir, recursive=args.recursive)

    processed_hashes = storage.load_processed(args.output)
    remaining = []
    skipped = 0
    for img in images:
        h = scanner.file_hash(str(img))
        if h in processed_hashes:
            skipped += 1
        else:
            remaining.append((img, h))
    if args.limit is not None:
        remaining = remaining[: args.limit]

    console.print(
        f"Found {len(images)} images: {skipped} already done, {len(remaining)} to process"
    )
    if not remaining:
        console.print("Nothing to do.")
        return 0

    with _http_client(args) as hc:
        model_id = _resolve_model(args, hc, console)
        if model_id is None:
            return 1
        options = _resolve_options(args, task)
        settings = _settings_echo(model_id, task, options)

        if args.dry_run:
            console.print(Panel(
                Text(orjson.dumps(settings, option=orjson.OPT_INDENT_2).decode()),
                title="resolved settings (dry run)",
            ))
            return 0

        completed = failed = invalid_json = 0
        tps_samples: list[float] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("Labeling", total=len(remaining))
            try:
                for image_path, fhash in remaining:
                    desc = f"[cyan]{image_path.name}[/cyan]"
                    if tps_samples:
                        avg = sum(tps_samples) / len(tps_samples)
                        desc += f" [dim]{avg:.0f} tok/s[/dim]"
                    progress.update(bar, description=desc)

                    try:
                        result = label_image(
                            hc, model_id, task.system_prompt, task.user_prompt,
                            image_path, options, retries=args.retries,
                        )
                    except httpx.HTTPStatusError as e:
                        console.print(
                            f"[red]HTTP {e.response.status_code} for {image_path.name}: "
                            f"{escape(e.response.text[:200])}[/red]"
                        )
                        failed += 1
                        progress.advance(bar)
                        continue
                    except Exception as e:
                        console.print(f"[red]{image_path.name}: {escape(str(e))}[/red]")
                        failed += 1
                        progress.advance(bar)
                        continue

                    record = {
                        "file_path": str(image_path),
                        "file_hash": fhash,
                        "file_name": image_path.name,
                        "model_id": model_id,
                        "task": task.name,
                        "raw_output": result.content,
                        "generation_time_ms": result.request_ms,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "settings": settings,
                    }
                    if task.expects_json:
                        label_json = storage.extract_json(result.content)
                        if label_json is None:
                            record["label"] = None
                            record["parse_ok"] = False
                            invalid_json += 1
                        else:
                            parsed = orjson.loads(label_json)
                            record["label"] = parsed
                            record["parse_ok"] = True
                            missing = missing_required_keys(parsed, task)
                            if missing:
                                record["missing_keys"] = missing
                    else:
                        record["label"] = result.content
                        record["parse_ok"] = True
                    if result.thinking:
                        record["thinking"] = result.thinking
                    if result.usage:
                        record["usage"] = result.usage
                    if result.performance:
                        record["performance"] = result.performance
                        tps = result.performance.get("generation_tps")
                        if tps:
                            tps_samples.append(tps)

                    storage.append_result(args.output, record)
                    completed += 1
                    progress.advance(bar)
            except KeyboardInterrupt:
                console.print("\nInterrupted. Partial results saved; re-run to resume.")

    summary = Table(box=None, pad_edge=False, show_header=False)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("completed", str(completed))
    if invalid_json:
        summary.add_row("invalid JSON", f"{invalid_json} (stored with parse_ok=false)")
    if failed:
        summary.add_row("failed", f"{failed} (not stored; re-run to retry)")
    summary.add_row("skipped (resume)", str(skipped))
    if tps_samples:
        summary.add_row("avg speed", f"{sum(tps_samples) / len(tps_samples):.1f} tok/s")
    summary.add_row("results", args.output)
    console.print(summary)
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    console = Console()
    commands = {
        "run": cmd_run,
        "try": cmd_try,
        "models": cmd_models,
        "tasks": cmd_tasks,
    }
    return commands[args.command](args, console)


if __name__ == "__main__":
    sys.exit(main())
