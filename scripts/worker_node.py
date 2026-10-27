#!/usr/bin/env python3
"""Unified distributed worker node launcher for a secondary Mac.

Starts both distributed backends bound strictly to the worker's Thunderbolt
Bridge interface (never `0.0.0.0` unless explicitly requested) and tears both
down cleanly on `Ctrl+C` (`SIGINT`/`SIGTERM`):

1. **GGUF (`ggml-rpc-server`)** on `<host>:50052` with `-c` (local SSD tensor
   caching keyed by FNV-1a hash).
2. **MLX (`scripts/mlx_pipeline_worker.py --daemon`)** on `<host>:32423` with
   automatic on-demand Rank-1 shard streaming into `--models-root` (`~/.cache/heylook/models`).

Usage:
    # Auto-detect the local Thunderbolt Bridge (bridge0) IPv4 address:
    uv run python scripts/worker_node.py

    # Or specify the worker's Thunderbolt IP explicitly:
    uv run python scripts/worker_node.py --host <worker-tb-ip>
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from heylook_llm.providers.common.mlx_pipeline import (
    DEFAULT_CONTROL_PORT,
    resolve_default_models_root,
    sync_scan_folders_with_peer,
)


def detect_thunderbolt_bridge_ip() -> Optional[str]:
    """Return the IPv4 address configured on macOS `bridge0` (Thunderbolt Bridge),
    ignoring link-local addresses when a static IP is present."""
    try:
        out = subprocess.check_output(["ifconfig", "bridge0"], text=True, timeout=2.0)
    except Exception:
        return None
    ips = [m.group(1) for m in re.finditer(r"\binet\s+(\d+\.\d+\.\d+\.\d+)\b", out)]
    non_link_local = [ip for ip in ips if not ipaddress.ip_address(ip).is_link_local]
    if non_link_local:
        return non_link_local[0]
    return ips[0] if ips else None


def find_rpc_server_binary() -> Optional[Path]:
    """Locate `ggml-rpc-server` in `~/.heylook/llama.cpp/build/bin` or `coderef/llama.cpp/build/bin`."""
    candidates = [
        Path("~/.heylook/llama.cpp/build/bin/ggml-rpc-server").expanduser(),
        Path(__file__).resolve().parents[1] / "coderef/llama.cpp/build/bin/ggml-rpc-server",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_mode & 0o111:
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=None,
        help="Worker Thunderbolt IP to bind (default: auto-detected from bridge0).",
    )
    parser.add_argument(
        "--rpc-port",
        type=int,
        default=50052,
        help="TCP port for ggml-rpc-server (default: 50052; 0 to disable GGUF RPC).",
    )
    parser.add_argument(
        "--mlx-control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help=f"TCP control port for MLX pipeline daemon (default: {DEFAULT_CONTROL_PORT}; 0 to disable MLX).",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="Local directory for caching streamed MLX shards (default: [scan].folders[0] from heylook.toml, else ~/.cache/heylook/models).",
    )
    parser.add_argument(
        "--rpc-cache-dir",
        default="~/.cache/heylook/ggml-rpc",
        help="Local SSD cache directory for ggml-rpc-server -c (default: ~/.cache/heylook/ggml-rpc).",
    )
    parser.add_argument(
        "--sync-peer",
        default=None,
        help="Optional peer IP to bidirectionally sync [scan].folders model trees with over control port 32423.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Run --sync-peer synchronization and exit without starting background daemons.",
    )
    parser.add_argument(
        "--sync-metadata-only",
        action="store_true",
        help="When syncing with --sync-peer, sync only metadata/tokenizers/model.heylook.toml (stream weight shards on demand).",
    )
    parser.add_argument(
        "--sync-filter",
        default=None,
        help="Optional substring filter when syncing with --sync-peer (e.g. 'Qwen3.5').",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [worker_node] %(message)s",
    )

    resolved_models_root = resolve_default_models_root(args.models_root)

    if args.sync_peer:
        logging.info(
            "Synchronizing model folder tree (%s) with peer %s:%d...",
            resolved_models_root,
            args.sync_peer,
            args.mlx_control_port or DEFAULT_CONTROL_PORT,
        )
        summary = sync_scan_folders_with_peer(
            args.sync_peer,
            local_root=resolved_models_root,
            control_port=args.mlx_control_port or DEFAULT_CONTROL_PORT,
            include_weights=not args.sync_metadata_only,
            model_filter=args.sync_filter,
        )
        logging.info(
            "Sync complete: pulled %d file(s), pushed %d file(s) between %s and %s",
            len(summary["pulled"]),
            len(summary["pushed"]),
            summary["local_root"],
            summary["peer_root"],
        )
        if args.sync_only:
            return 0

    host = args.host or detect_thunderbolt_bridge_ip()
    if not host:
        logging.error(
            "Could not auto-detect an IPv4 address on Thunderbolt Bridge (bridge0). "
            "Pass `--host <worker-tb-ip>` explicitly."
        )
        return 1

    procs: list[tuple[str, subprocess.Popen[Any]]] = []

    if args.rpc_port > 0:
        rpc_bin = find_rpc_server_binary()
        if rpc_bin is not None:
            cache_dir = Path(args.rpc_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            rpc_cmd = [
                str(rpc_bin),
                "-H",
                host,
                "-p",
                str(args.rpc_port),
                "-c",
            ]
            logging.info("Starting GGUF RPC worker: %s", " ".join(rpc_cmd))
            procs.append(("ggml-rpc-server", subprocess.Popen(rpc_cmd)))
        else:
            logging.warning(
                "ggml-rpc-server binary not found (run `uv run python scripts/build_llama.py` "
                "to enable GGUF RPC); starting MLX worker daemon only."
            )

    if args.mlx_control_port > 0:
        mlx_script = Path(__file__).resolve().parent / "mlx_pipeline_worker.py"
        mlx_cmd = [
            sys.executable,
            str(mlx_script),
            "--daemon",
            "--host",
            host,
            "--control-port",
            str(args.mlx_control_port),
            "--models-root",
            str(resolved_models_root),
        ]
        logging.info("Starting MLX pipeline daemon: %s", " ".join(mlx_cmd))
        procs.append(("mlx-pipeline-daemon", subprocess.Popen(mlx_cmd)))

    if not procs:
        logging.error("No worker services started.")
        return 1

    def _shutdown(signum: int = 0, _frame: Any = None) -> None:
        if signum:
            logging.info("Stopping worker node services (signal %d)...", signum)
        for name, p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        deadline = time.monotonic() + 3.0
        for name, p in procs:
            while p.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    signal.signal(signal.SIGINT, lambda s, f: (_shutdown(s, f), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (_shutdown(s, f), sys.exit(0)))

    logging.info(
        "Worker node ready on %s (press Ctrl+C when done for the day to close all ports).",
        host,
    )
    try:
        while True:
            for name, p in procs:
                rc = p.poll()
                if rc is not None:
                    logging.error("Service %s exited with code %d; shutting down.", name, rc)
                    _shutdown()
                    return rc or 1
            time.sleep(0.5)
    finally:
        _shutdown()


if __name__ == "__main__":
    sys.exit(main())
