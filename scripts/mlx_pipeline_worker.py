#!/usr/bin/env python3
"""Distributed MLX pipeline worker daemon and 2-rank smoke runner.

Supports two operating modes on the worker Mac:

1. **Persistent Multi-Model Daemon Mode (default when `--model-path` is omitted)**:
   Start once on the worker Mac:
       uv run python scripts/mlx_pipeline_worker.py --daemon --host <worker-ip>
   Listens on `--control-port` (default: 32423). Whenever `heylookllm` on the
   primary Mac loads an MLX model with `pipeline_ratios` and `pipeline_peers`:
   - Automatically receives any missing `config.json`, `tokenizer*`, and Rank-1
     `.safetensors` shards (~20% of the model) into `--models-root` (`~/.cache/heylook/models`).
   - Loads Rank 1's layer slice and executes the lockstep forward loop.
   - When `heylookllm` unloads the model (`CMD_UNLOAD_MODEL`), frees Rank 1's
     GPU weights and stays connected on the ring for the next model load.
   - If `heylookllm` is restarted on the primary Mac (new `host_session_id`),
     automatically respawns a clean session worker child process.

2. **Single-Model / Smoke Mode (`--model-path` provided)**:
   Loopback 2-rank smoke on one Mac:
       mlx.launch -n 2 uv run python scripts/mlx_pipeline_worker.py \\
           --model-path /path/to/mlx-model --ratios 0.8,0.2 \\
           --smoke-prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

from heylook_llm.providers.common.mlx_pipeline import (
    DEFAULT_CONTROL_PORT,
    _recv_json_msg,
    _send_json_msg,
    enable_fast_metal_synch,
    handle_sync_control_request,
    init_pipeline_group,
    install_rank0_step_coordinator,
    receive_missing_files_on_worker,
    resolve_default_models_root,
    run_pipeline_worker_loop,
    selective_pipeline_load,
    send_pipeline_shutdown,
)


def _parse_ratios(raw: str) -> list[float]:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) < 2 or any(p <= 0 for p in parts):
        raise ValueError(f"--ratios must be >= 2 positive floats, got {raw!r}")
    return parts


def _run_internal_session_worker(backend: str = "any") -> int:
    """Persistent child process for one `host_session_id`.

    Reads JSON commands line-by-line from `sys.stdin`, reuses the process-wide
    `mx.distributed` ring across multiple `CMD_LOAD_MODEL` / `CMD_UNLOAD_MODEL`
    cycles, and frees GPU weights after each unload.
    """
    enable_fast_metal_synch()
    group = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        if cmd.get("op") == "shutdown":
            break
        model_path = cmd["model_path"]
        ratios = [float(r) for r in cmd["ratios"]]
        peers = cmd.get("peers")
        rank = int(cmd.get("rank", 1))

        if group is None:
            group = init_pipeline_group(peers=peers, rank=rank, backend=backend)

        logging.info(
            "Session worker rank=%d/%d loading %s with ratios=%s",
            group.rank(),
            group.size(),
            model_path,
            ratios,
        )
        model, processor = selective_pipeline_load(
            model_path,
            ratios=ratios,
            group=group,
            peers=peers,
            rank=rank,
            skip_audio=True,
            control_port=0,
        )
        steps = run_pipeline_worker_loop(model, group=group)
        logging.info(
            "Session worker rank=%d unloaded %s after %d steps (returning to idle)",
            group.rank(),
            Path(model_path).name,
            steps,
        )
        del model
        del processor
        gc.collect()
        mx.clear_cache()
    return 0


def _start_oneshot_control_responder(
    target_dir: Path,
    control_port: int,
    bind_host: str = "0.0.0.0",
) -> Optional[socket.socket]:
    """Start a background responder on `control_port` for one-shot `--model-path`
    worker runs so Rank 0's `prepare_worker_via_control_port` succeeds."""
    if control_port <= 0:
        return None
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((bind_host, int(control_port)))
        srv.listen(4)
    except OSError as exc:
        logging.debug("Control port %d already bound (%s)", control_port, exc)
        srv.close()
        return None

    def _serve() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                try:
                    req = _recv_json_msg(conn)
                    op = req.get("op")
                    if op == "ping":
                        _send_json_msg(conn, {"status": "ok", "mode": "oneshot"})
                    elif op == "prepare_model":
                        files = req.get("files") or []
                        host_present = req.get("host_present")
                        receive_missing_files_on_worker(
                            conn,
                            target_dir,
                            files,
                            host_present=host_present,
                        )
                        _send_json_msg(
                            conn,
                            {"status": "ready", "target_dir": str(target_dir)},
                        )
                except Exception as exc:
                    logging.warning("One-shot control responder error: %s", exc)

    threading.Thread(target=_serve, daemon=True, name="mlx-ctrl-oneshot").start()
    return srv


def run_daemon(
    *,
    bind_host: str,
    control_port: int,
    models_root: Path,
    backend: str = "any",
) -> int:
    """Run the persistent multi-model MLX worker supervisor."""
    models_root = Path(models_root).expanduser().resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_host, int(control_port)))
    srv.listen(8)

    active_session_id: Optional[str] = None
    child_proc: Optional[subprocess.Popen[str]] = None

    def _stop_child() -> None:
        nonlocal child_proc, active_session_id
        if child_proc is not None:
            if child_proc.poll() is None:
                try:
                    child_proc.terminate()
                    child_proc.wait(timeout=3.0)
                except Exception:
                    try:
                        child_proc.kill()
                    except Exception:
                        pass
            child_proc = None
        active_session_id = None

    def _handle_sig(signum: int, _frame: Any) -> None:
        logging.info("Shutting down MLX worker daemon (signal %d)...", signum)
        _stop_child()
        try:
            srv.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    logging.info(
        "MLX pipeline worker daemon listening on %s:%d (models_root=%s)",
        bind_host,
        control_port,
        models_root,
    )

    try:
        while True:
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            with conn:
                try:
                    req = _recv_json_msg(conn)
                    op = req.get("op")
                    if op == "ping":
                        _send_json_msg(
                            conn,
                            {
                                "status": "ok",
                                "mode": "daemon",
                                "models_root": str(models_root),
                                "active_session_id": active_session_id,
                            },
                        )
                        continue
                    if handle_sync_control_request(conn, req, models_root):
                        continue
                    if op != "prepare_model":
                        _send_json_msg(
                            conn,
                            {"status": "error", "error": f"unknown op {op!r}"},
                        )
                        continue

                    host_session_id = str(req.get("host_session_id") or "default")
                    model_rel_path = str(req["model_rel_path"]).lstrip("/")
                    ratios = [float(r) for r in req["ratios"]]
                    peers = list(req["peers"])
                    rank = int(req.get("rank", 1))
                    files = list(req.get("files") or [])
                    host_present = req.get("host_present")

                    target_dir = (models_root / model_rel_path).resolve()
                    if not str(target_dir).startswith(str(models_root)):
                        _send_json_msg(
                            conn,
                            {"status": "error", "error": "invalid model_rel_path"},
                        )
                        continue

                    streamed = receive_missing_files_on_worker(
                        conn,
                        target_dir,
                        files,
                        host_present=host_present,
                    )
                    if streamed:
                        logging.info(
                            "Cached %d file(s) for %s in %s",
                            len(streamed),
                            model_rel_path,
                            target_dir,
                        )

                    if (
                        child_proc is None
                        or child_proc.poll() is not None
                        or active_session_id != host_session_id
                    ):
                        if child_proc is not None:
                            logging.info(
                                "Host session changed (%s -> %s); restarting session worker...",
                                active_session_id,
                                host_session_id,
                            )
                            _stop_child()
                        child_proc = subprocess.Popen(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--internal-session-worker",
                                "--backend",
                                backend,
                            ],
                            stdin=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                        )
                        active_session_id = host_session_id

                    assert child_proc.stdin is not None
                    cmd_line = json.dumps(
                        {
                            "op": "load_model",
                            "model_path": str(target_dir),
                            "ratios": ratios,
                            "peers": peers,
                            "rank": rank,
                        }
                    )
                    child_proc.stdin.write(cmd_line + "\n")
                    child_proc.stdin.flush()

                    _send_json_msg(
                        conn,
                        {
                            "status": "ready",
                            "target_dir": str(target_dir),
                            "streamed_files": streamed,
                        },
                    )
                except Exception as exc:
                    logging.error(
                        "Error handling control request from %s: %s",
                        addr,
                        exc,
                        exc_info=True,
                    )
    finally:
        _stop_child()
        srv.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a persistent multi-model worker daemon (default when --model-path is omitted).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind IP for the worker daemon control port (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help=f"TCP control port for preflight and shard streaming (default: {DEFAULT_CONTROL_PORT}).",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="Root directory on the worker Mac for caching streamed model shards (default: [scan].folders[0] from heylook.toml, else ~/.cache/heylook/models).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a single MLX model directory (omit to run in persistent --daemon mode).",
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
        "--peers",
        default=None,
        help="Optional comma-separated peer IPs (e.g. '<primary-ip>,<worker-ip>') for SSH-free ring launch.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank index (0 on primary Mac, 1 on worker Mac) when using --peers.",
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
    parser.add_argument(
        "--internal-session-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.internal_session_worker:
        return _run_internal_session_worker(backend=args.backend)

    if args.daemon or args.model_path is None:
        return run_daemon(
            bind_host=args.host,
            control_port=args.control_port,
            models_root=resolve_default_models_root(args.models_root),
            backend=args.backend,
        )

    enable_fast_metal_synch()
    if args.peers is not None and args.rank is None:
        parser.error("--rank is required when --peers is set")

    oneshot_srv = None
    if args.peers is not None and (args.rank or 0) > 0:
        oneshot_srv = _start_oneshot_control_responder(
            Path(args.model_path).expanduser().resolve(),
            control_port=args.control_port,
            bind_host=args.host,
        )

    try:
        ratios = _parse_ratios(args.ratios)
        if args.peers is not None and (args.rank or 0) == 0:
            # Rank 0 with --peers: let selective_pipeline_load run the control handshake
            # (if the worker has its control port open) and initialize the ring.
            model, processor = selective_pipeline_load(
                args.model_path,
                ratios=ratios,
                peers=args.peers,
                rank=0,
                skip_audio=True,
                control_port=args.control_port,
            )
            group = getattr(model, "_pipeline_group", None) or init_pipeline_group()
        else:
            group = init_pipeline_group(
                peers=args.peers,
                rank=args.rank or 0,
                backend=args.backend,
            )
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
                control_port=0,
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
    finally:
        if oneshot_srv is not None:
            oneshot_srv.close()


if __name__ == "__main__":
    sys.exit(main())
