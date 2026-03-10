"""CLI entry point for batch VLM image labeling."""

import argparse
import sys
import time
from pathlib import Path

import httpx
import orjson
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from . import scanner, client, storage


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='batch-labeler',
        description='Label images in a directory using a VLM via OpenAI-compatible API.',
    )
    p.add_argument('--image-dir', required=True, help='Path to directory containing images')
    p.add_argument('--model', required=True, help='VLM model ID to use')
    p.add_argument('--system-prompt', help='System prompt with labeling instructions')
    p.add_argument('--system-prompt-file', type=Path, help='Read system prompt from file')
    p.add_argument('--output', default='results.jsonl', help='Output JSONL file (default: results.jsonl)')
    p.add_argument('--server', default='http://localhost:8000', help='Server base URL (default: http://localhost:8000)')
    p.add_argument('--max-tokens', type=int, default=1024, help='Max tokens per response (default: 1024)')
    p.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (default: 0.1)')
    p.add_argument('--recursive', action='store_true', default=True, help='Scan subdirectories (default)')
    p.add_argument('--no-recursive', action='store_false', dest='recursive', help='Only scan top-level directory')
    p.add_argument('--dry-run', action='store_true', help='Scan and report counts without processing')
    p.add_argument('--timeout', type=float, default=120.0, help='Per-request timeout in seconds (default: 120)')
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    console = Console()

    # Resolve system prompt
    if args.system_prompt_file:
        if not args.system_prompt_file.exists():
            console.print(f"[red]System prompt file not found: {args.system_prompt_file}[/red]")
            return 1
        system_prompt = args.system_prompt_file.read_text().strip()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        console.print("[red]Provide --system-prompt or --system-prompt-file[/red]")
        return 1

    # Validate image dir
    if not Path(args.image_dir).is_dir():
        console.print(f"[red]Image directory not found: {args.image_dir}[/red]")
        return 1

    # Scan
    console.print(f"Scanning {args.image_dir}...")
    images = scanner.scan_images(args.image_dir, recursive=args.recursive)
    console.print(f"Found {len(images)} images")

    if not images:
        console.print("Nothing to process.")
        return 0

    # Resume support
    processed_hashes = storage.load_processed(args.output)
    remaining = []
    skipped = 0
    for img in images:
        h = scanner.file_hash(str(img))
        if h in processed_hashes:
            skipped += 1
        else:
            remaining.append((img, h))

    console.print(f"Total: {len(images)}, already done: {skipped}, remaining: {len(remaining)}")

    if args.dry_run:
        console.print("Dry run -- exiting.")
        return 0

    if not remaining:
        console.print("All images already processed.")
        return 0

    # Process
    completed = 0
    failed = 0

    http_client = httpx.Client(
        base_url=args.server,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Labeling", total=len(remaining))

            for image_path, fhash in remaining:
                progress.update(task, description=f"[cyan]{image_path.name}[/cyan]")

                gen_start = time.time()
                try:
                    raw_output = client.label_image(
                        http_client,
                        model_id=args.model,
                        system_prompt=system_prompt,
                        image_path=image_path,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    gen_ms = int((time.time() - gen_start) * 1000)

                    label_json = storage.extract_json(raw_output)
                    if label_json is None:
                        label_json = orjson.dumps({"raw_caption": raw_output}).decode()

                    storage.append_result(args.output, {
                        "file_path": str(image_path),
                        "file_hash": fhash,
                        "file_name": image_path.name,
                        "model_id": args.model,
                        "label": orjson.loads(label_json),
                        "raw_output": raw_output,
                        "generation_time_ms": gen_ms,
                        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                    })
                    completed += 1

                except KeyboardInterrupt:
                    console.print("\nInterrupted. Partial results saved.")
                    break
                except httpx.HTTPStatusError as e:
                    console.print(f"[red]HTTP {e.response.status_code} for {image_path.name}: {e.response.text[:200]}[/red]")
                    failed += 1
                except httpx.TimeoutException:
                    console.print(f"[red]Timeout for {image_path.name}[/red]")
                    failed += 1
                except Exception as e:
                    console.print(f"[red]Error on {image_path.name}: {e}[/red]")
                    failed += 1

                progress.advance(task)

    finally:
        http_client.close()

    console.print(f"\nDone: {completed} completed, {failed} failed, {skipped} skipped")
    console.print(f"Results: {args.output}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
