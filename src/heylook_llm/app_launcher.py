"""CLI launcher for companion apps under ``apps/`` (``heylookllm --<app>`` / ``heylookllm app <name>``).

Keeps ``mlx`` unimported until the target app explicitly loads it so lightweight
subcommands and ``tests/unit/test_mlx_overlay.py`` stay zero-MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Optional, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class AppEntry:
    name: str
    description: str
    rel_dir: str
    runner: Callable[[list[str]], int]
    aliases: tuple[str, ...] = ()


def _latest_subdir(root: Path, marker: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    matches: list[tuple[float, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if marker == "*" or (child / marker).exists() or any(child.glob(marker)):
            try:
                mtime = child.stat().st_mtime
            except OSError:
                mtime = 0.0
            matches.append((mtime, child))
    if not matches:
        return None
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return matches[0][1]


_STEERING_SUBCOMMANDS = frozenset(
    {"demo-bank", "capture", "build", "inspect-bank", "xray", "chain-lens", "serve"}
)


def _run_steering_lab(argv: list[str]) -> int:
    root = _repo_root()
    src_dir = root / "apps" / "steering-lab" / "src"
    if not src_dir.is_dir():
        print(f"Error: steering-lab source directory not found at {src_dir}", file=sys.stderr)
        return 1
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from steering_lab.cli import main as steering_main

    # Convenience: `heylookllm --steering-lab [--port 8111 ...]` defaults to `serve`
    # and auto-attaches the newest local SpanBank + matching vector pack if present.
    args = list(argv)
    if not args or (args[0].startswith("-") and args[0] not in ("-h", "--help")):
        args = ["serve", *args]

    if args and args[0] == "serve":
        data_root = root / "apps" / "steering-lab" / "data"
        has_bank = any(a == "--bank" or a.startswith("--bank=") for a in args[1:])
        has_vectors = any(a == "--vectors" or a.startswith("--vectors=") for a in args[1:])
        has_model = any(a == "--model" or a.startswith("--model=") for a in args[1:])
        if not has_bank:
            latest_bank = _latest_subdir(data_root / "banks", "bank.json")
            if latest_bank is not None:
                args.extend(["--bank", str(latest_bank)])
                if not has_vectors:
                    matched_vec = data_root / "vectors" / latest_bank.name
                    if matched_vec.is_dir():
                        args.extend(["--vectors", str(matched_vec)])
                        has_vectors = True
        if not has_vectors:
            latest_vec = _latest_subdir(data_root / "vectors", "*/vector.safetensors")
            if latest_vec is not None:
                args.extend(["--vectors", str(latest_vec)])
        if not has_model:
            args.extend(["--model", "Swift-1.5-Qwen3.8-27B-MLX-8bit"])

    return int(steering_main(args))


def _run_batch_labeler(argv: list[str]) -> int:
    app_dir = _repo_root() / "apps" / "batch-labeler"
    if not app_dir.is_dir():
        print(f"Error: batch-labeler directory not found at {app_dir}", file=sys.stderr)
        return 1
    uv_bin = shutil.which("uv")
    if uv_bin and (app_dir / "pyproject.toml").is_file():
        proc = subprocess.run(
            [uv_bin, "run", "--directory", str(app_dir), "batch-labeler", *argv],
            check=False,
        )
        return int(proc.returncode)
    src_dir = app_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from batch_labeler.cli import main as bl_main

    return int(bl_main(argv) or 0)


APP_REGISTRY: tuple[AppEntry, ...] = (
    AppEntry(
        name="steering-lab",
        description="MLX Activation Steering & Mechanistic Lab (default: serve on :8111)",
        rel_dir="apps/steering-lab",
        runner=_run_steering_lab,
        aliases=("steering",),
    ),
    AppEntry(
        name="batch-labeler",
        description="Batch VLM image labeling CLI targeting heylook /v1/messages",
        rel_dir="apps/batch-labeler",
        runner=_run_batch_labeler,
        aliases=("labeler",),
    ),
)


def _lookup_app(name: str) -> Optional[AppEntry]:
    norm = name.strip().lower().lstrip("-")
    for entry in APP_REGISTRY:
        if norm == entry.name or norm in entry.aliases:
            return entry
    return None


def print_apps_list() -> int:
    print("Available companion apps under apps/:")
    for entry in APP_REGISTRY:
        alias_txt = f" (alias: --{entry.aliases[0]})" if entry.aliases else ""
        print(f"  --{entry.name:<16s}  {entry.description}{alias_txt}")
    print("\nUsage:")
    print("  uv run heylookllm --steering-lab [--port 8111] [--host 127.0.0.1]")
    print("  uv run heylookllm --steering-lab capture --model ... --out ...")
    print("  uv run heylookllm app <name> [app-args...]")
    return 0


def maybe_dispatch_app_shortcut(argv: Sequence[str]) -> Optional[int]:
    """Intercept ``heylookllm --<app> ...``, ``--apps``, or ``app <name> ...`` before MLX imports.

    Returns an exit code when handled, or ``None`` to fall through to normal server CLI parsing.
    """
    args = list(argv)
    if not args:
        return None
    head = args[0]
    if head == "--apps":
        return print_apps_list()
    if head == "app":
        if len(args) < 2 or args[1] in ("list", "-h", "--help"):
            return print_apps_list()
        entry = _lookup_app(args[1])
        if entry is None:
            known = ", ".join(e.name for e in APP_REGISTRY)
            print(f"Unknown app {args[1]!r}. Available apps: {known}", file=sys.stderr)
            return 2
        return entry.runner(args[2:])
    if head.startswith("--"):
        entry = _lookup_app(head[2:])
        if entry is not None:
            return entry.runner(args[1:])
    return None
