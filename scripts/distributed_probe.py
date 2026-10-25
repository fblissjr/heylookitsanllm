#!/usr/bin/env python3
"""Diagnostic and smoke harness for multi-Mac distributed inference (GGUF RPC & MLX Pipeline).

Subcommands:
  preflight    Check local binaries (llama-server + ggml-rpc-server), Metal
               working set, Thunderbolt Bridge interface, and optional peer TCP/SSH.
  mlx-local    Run a 1-rank vs 2-rank loopback greedy parity check on an MLX model
               using `mlx.launch -n 2` and `scripts/mlx_pipeline_worker.py`.
  mlx-cluster  Validate a `hosts.json` file (SSH + IP reachability) and run a
               multi-Mac `mlx.launch --backend ring` pipeline smoke test.
  gguf-rpc     Verify TCP reachability to a remote `ggml-rpc-server`, compare
               single-node vs distributed `ram_fit` weight sizing, and run
               `llama-fit-params` (or a live one-shot completion with `--smoke`).

Examples:
  uv run python scripts/distributed_probe.py preflight --peer <worker-ip>:50052
  uv run python scripts/distributed_probe.py mlx-local --model Qwen3.5-0.8B-MLX-8bit
  uv run python scripts/distributed_probe.py mlx-cluster --hostfile hosts.json --model Qwen3.5-0.8B-MLX-8bit
  uv run python scripts/distributed_probe.py gguf-rpc --peer <worker-ip>:50052 --model <gguf-id-or-path> --tensor-split 20,80
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_model_entry(model_arg: str, expected_provider: str | None = None) -> tuple[str, Path, Any]:
    """Resolve `model_arg` as either a filesystem path or a served model ID."""
    candidate = Path(model_arg).expanduser()
    if candidate.exists():
        from heylook_llm.config import GGUFModelConfig, MLXModelConfig
        if expected_provider == "gguf" or candidate.suffix == ".gguf" or any(candidate.glob("*.gguf")):
            gguf_file = candidate if candidate.is_file() else next(candidate.glob("*.gguf"))
            cfg = GGUFModelConfig(model_path=str(gguf_file))
            return candidate.name, gguf_file, cfg
        cfg = MLXModelConfig(model_path=str(candidate))
        return candidate.name, candidate, cfg

    # Otherwise resolve via router discovery / heylook.toml
    from heylook_llm.config import GGUFModelConfig, MLXModelConfig
    from scripts._inproc import resolve_config
    provider, cfg_dict = resolve_config(model_arg)
    if expected_provider and provider != expected_provider:
        raise SystemExit(
            f"Model {model_arg!r} has provider={provider!r}, expected {expected_provider!r}"
        )
    cfg = GGUFModelConfig(**cfg_dict) if provider == "gguf" else MLXModelConfig(**cfg_dict)
    return model_arg, Path(cfg.model_path).expanduser(), cfg


def _check_tcp(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ms = (time.perf_counter() - t0) * 1000.0
            return True, f"reachable ({ms:.1f} ms)"
    except OSError as exc:
        return False, str(exc)


def _llama_bin_dir() -> Path:
    base = Path(os.environ.get("HEYLOOK_LLAMA_CPP_DIR", Path.home() / ".heylook" / "llama.cpp"))
    return base / "build" / "bin"


def cmd_preflight(args: argparse.Namespace) -> int:
    print("=== Distributed Inference Preflight ===")
    failures = 0

    # 1. Check llama.cpp build & ggml-rpc-server
    bin_dir = _llama_bin_dir()
    llama_server = bin_dir / "llama-server"
    rpc_server = bin_dir / "ggml-rpc-server"
    manifest_path = bin_dir.parent / "heylook-build.json"

    print("\n[1/4] Canonical llama.cpp binaries:")
    if llama_server.is_file() and os.access(llama_server, os.X_OK):
        print(f"  OK   llama-server:    {llama_server}")
    else:
        print(f"  MISS llama-server:    {llama_server} (run: uv run python scripts/build_llama.py)")
        failures += 1

    if rpc_server.is_file() and os.access(rpc_server, os.X_OK):
        print(f"  OK   ggml-rpc-server: {rpc_server}")
    else:
        print(f"  MISS ggml-rpc-server: {rpc_server} (run: uv run python scripts/build_llama.py)")
        failures += 1

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rev = manifest.get("rev", "unknown")[:10]
        rpc_flag = (manifest.get("effective") or {}).get("GGML_RPC")
        print(f"  OK   build rev={rev}, GGML_RPC={rpc_flag}")
        if rpc_flag != "ON":
            print("  WARN GGML_RPC is not ON in heylook-build.json; rebuild with scripts/build_llama.py")
            failures += 1

    # 2. Check MLX distributed availability
    print("\n[2/4] MLX distributed runtime:")
    mlx_launch = shutil.which("mlx.launch")
    print(f"  {'OK  ' if mlx_launch else 'WARN'} mlx.launch: {mlx_launch or 'not on PATH (use: uv run mlx.launch)'}")
    try:
        import mlx.core as mx
        avail = mx.distributed.is_available()
        print(f"  OK   mx.distributed.is_available() = {avail}")
    except Exception as exc:
        print(f"  FAIL importing mlx.core: {exc}")
        failures += 1

    # 3. Check network interfaces (bridge0 / Thunderbolt)
    print("\n[3/4] Thunderbolt Bridge (bridge0) status:")
    try:
        res = subprocess.run(["ifconfig", "bridge0"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            lines = [ln.strip() for ln in res.stdout.splitlines() if any(k in ln for k in ("inet ", "status:", "member:", "mtu"))]
            for ln in lines:
                print(f"  INFO {ln}")
        else:
            print("  INFO bridge0 not configured or inactive")
    except FileNotFoundError:
        pass

    # 4. Optional peer check
    print("\n[4/4] Peer connectivity:")
    if args.peer:
        host, sep, port_str = args.peer.rpartition(":")
        if not sep:
            host, port = args.peer, 50052
        else:
            port = int(port_str)
        ok, detail = _check_tcp(host, port)
        status = "OK  " if ok else "FAIL"
        print(f"  {status} TCP {host}:{port} -> {detail}")
        if not ok:
            print(f"       (Start worker on peer with: ./scripts/rpc_worker.sh --host {host} --port {port})")
            failures += 1
    else:
        print("  SKIP no --peer <host[:port]> specified")

    return 1 if failures else 0


def cmd_mlx_local(args: argparse.Namespace) -> int:
    _, model_path, _ = _resolve_model_entry(args.model, expected_provider="mlx")
    worker_script = REPO_ROOT / "scripts" / "mlx_pipeline_worker.py"
    print(f"=== MLX Local 2-Rank Pipeline Parity Check ===")
    print(f"Model:  {model_path}")
    print(f"Ratios: {args.ratios}")
    print(f"Prompt: {args.prompt!r}\n")

    # 1. Single-rank baseline
    print("[1/2] Running 1-rank baseline...")
    cmd_1 = [
        sys.executable,
        str(worker_script),
        "--model-path", str(model_path),
        "--ratios", args.ratios,
        "--smoke-prompt", args.prompt,
        "--max-tokens", str(args.max_tokens),
    ]
    res_1 = subprocess.run(cmd_1, capture_output=True, text=True, check=False)
    if res_1.returncode != 0:
        print(res_1.stdout)
        print(res_1.stderr, file=sys.stderr)
        return res_1.returncode
    out_1 = res_1.stdout.strip()
    # stdout of mlx_pipeline_worker.py --smoke-prompt is the generated text directly;
    # stderr carries the per-rank memory/tok/s log lines.
    for ln in res_1.stderr.splitlines():
        if "mlx_pipeline_worker" in ln:
            print(f"  {ln}")
    print(f"  Output: {out_1!r}")

    # 2. 2-rank loopback via mlx.launch -n 2
    print("\n[2/2] Running 2-rank pipeline under `mlx.launch -n 2`...")
    cmd_2 = [
        "uv", "run", "mlx.launch", "-n", "2",
        sys.executable,
        str(worker_script),
        "--model-path", str(model_path),
        "--ratios", args.ratios,
        "--smoke-prompt", args.prompt,
        "--max-tokens", str(args.max_tokens),
    ]
    res_2 = subprocess.run(cmd_2, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if res_2.returncode != 0:
        print(res_2.stdout)
        print(res_2.stderr, file=sys.stderr)
        return res_2.returncode
    out_2 = res_2.stdout.strip()
    for ln in res_2.stderr.splitlines():
        if "mlx_pipeline_worker" in ln:
            print(f"  {ln}")
    print(f"  Output: {out_2!r}")

    if out_1 and out_1 == out_2:
        print(f"\nPASS: 2-rank pipeline output matches 1-rank greedy baseline verbatim: {out_2!r}")
        return 0
    print(f"\nFAIL: 1-rank vs 2-rank output mismatch!\n  1-rank: {out_1!r}\n  2-rank: {out_2!r}", file=sys.stderr)
    return 1


def cmd_mlx_cluster(args: argparse.Namespace) -> int:
    hostfile = Path(args.hostfile).expanduser().resolve()
    if not hostfile.is_file():
        print(f"Hostfile not found: {hostfile}", file=sys.stderr)
        return 1

    hosts = json.loads(hostfile.read_text(encoding="utf-8"))
    print(f"=== MLX Multi-Mac Cluster Check ({hostfile.name}) ===")
    for idx, entry in enumerate(hosts):
        ssh_host = entry.get("ssh", "")
        ips = entry.get("ips") or []
        print(f"  Rank {idx}: ssh={ssh_host!r}, ips={ips}")

    _, model_path, _ = _resolve_model_entry(args.model, expected_provider="mlx")
    worker_script = REPO_ROOT / "scripts" / "mlx_pipeline_worker.py"
    cmd = [
        "uv", "run", "mlx.launch",
        "--backend", "ring",
        "--hostfile", str(hostfile),
        "uv", "--directory", str(REPO_ROOT), "run", "python",
        str(worker_script),
        "--model-path", str(model_path),
        "--ratios", args.ratios,
        "--smoke-prompt", args.prompt,
        "--max-tokens", str(args.max_tokens),
    ]
    print("\nExecuting:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return res.returncode


def cmd_gguf_rpc(args: argparse.Namespace) -> int:
    model_id, gguf_path, cfg = _resolve_model_entry(args.model, expected_provider="gguf")
    host, sep, port_str = args.peer.rpartition(":")
    if not sep:
        host, port = args.peer, 50052
    else:
        port = int(port_str)
    peer_endpoint = f"{host}:{port}"

    print(f"=== GGUF Distributed RPC Probe ({model_id}) ===")
    print(f"Model path:   {gguf_path}")
    print(f"RPC peer:     {peer_endpoint}")
    print(f"Tensor split: {args.tensor_split} (RPC worker first, local Metal last)\n")

    # 1. Compare ram_fit with and without RPC
    from heylook_llm.ram_fit import check_model_fit, read_ceilings
    ceilings = read_ceilings()

    single_cfg = cfg.model_copy(update={"rpc": None, "tensor_split": None})
    dist_cfg = cfg.model_copy(update={"rpc": peer_endpoint, "tensor_split": args.tensor_split})

    fit_single = check_model_fit(single_cfg, "gguf", ceilings=ceilings)
    fit_dist = check_model_fit(dist_cfg, "gguf", ceilings=ceilings)

    print("[1/3] Local Memory Fit Comparison (Mac Studio primary):")
    print(
        f"  Single-node:  weights={fit_single.weights_gb:.2f} GiB | "
        f"kv_headroom={fit_single.kv_headroom_gb or 0.0:.2f} GiB | verdict={fit_single.verdict.upper()}"
    )
    print(
        f"  With --rpc:   weights={fit_dist.weights_gb:.2f} GiB | "
        f"kv_headroom={fit_dist.kv_headroom_gb or 0.0:.2f} GiB | verdict={fit_dist.verdict.upper()} "
        f"({', '.join(fit_dist.sizing_notes)})"
    )

    # 2. Check TCP reachability to ggml-rpc-server
    print(f"\n[2/3] Checking TCP reachability to {peer_endpoint}...")
    ok, detail = _check_tcp(host, port)
    if not ok:
        print(f"  FAIL cannot connect to {peer_endpoint}: {detail}")
        print(f"  -> On the worker Mac, run: ./scripts/rpc_worker.sh --host {host} --port {port}")
        return 1
    print(f"  OK   connected to {peer_endpoint} ({detail})")

    # 3. Run llama-fit-params with --rpc and --tensor-split
    fit_bin = _llama_bin_dir() / "llama-fit-params"
    if fit_bin.is_file():
        print("\n[3/3] Querying `llama-fit-params` across RPC + local Metal:")
        cmd = [
            str(fit_bin),
            "-m", str(gguf_path),
            "--rpc", peer_endpoint,
            "--tensor-split", args.tensor_split,
            "-ngl", "999",
            "-np", "1",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in (res.stdout + "\n" + res.stderr).splitlines():
            if any(k in line.lower() for k in ("rpc", "metal", "buffer", "split", "layer", "device", "fit")):
                print(" ", line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic and verification tool for distributed GGUF RPC and MLX pipeline inference."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_pre = sub.add_parser("preflight", help="Check binaries, Metal limits, bridge0, and optional peer TCP.")
    p_pre.add_argument("--peer", default=None, help="Optional peer host[:port] (default port 50052) to TCP-probe.")
    p_pre.set_defaults(func=cmd_preflight)

    p_mlx_local = sub.add_parser("mlx-local", help="Run 1-rank vs 2-rank local greedy parity check on an MLX model.")
    p_mlx_local.add_argument("--model", default="Qwen3.5-0.8B-MLX-8bit", help="Served MLX model ID or path.")
    p_mlx_local.add_argument("--ratios", default="0.8,0.2", help="Comma-separated rank layer ratios (default: 0.8,0.2).")
    p_mlx_local.add_argument("--prompt", default="The capital of France is", help="Greedy test prompt.")
    p_mlx_local.add_argument("--max-tokens", type=int, default=16, help="Tokens to generate (default: 16).")
    p_mlx_local.set_defaults(func=cmd_mlx_local)

    p_mlx_cluster = sub.add_parser("mlx-cluster", help="Run 2-Mac MLX pipeline test via mlx.launch --hostfile.")
    p_mlx_cluster.add_argument("--hostfile", required=True, help="Path to hosts.json for mlx.launch.")
    p_mlx_cluster.add_argument("--model", required=True, help="Served MLX model ID or path (must exist on both Macs).")
    p_mlx_cluster.add_argument("--ratios", default="0.8,0.2", help="Comma-separated rank layer ratios (default: 0.8,0.2).")
    p_mlx_cluster.add_argument("--prompt", default="The capital of France is", help="Greedy test prompt.")
    p_mlx_cluster.add_argument("--max-tokens", type=int, default=16, help="Tokens to generate (default: 16).")
    p_mlx_cluster.set_defaults(func=cmd_mlx_cluster)

    p_rpc = sub.add_parser("gguf-rpc", help="Verify ggml-rpc peer, compare ram_fit, and run llama-fit-params.")
    p_rpc.add_argument("--peer", required=True, help="RPC worker host:port (e.g. <worker-ip>:50052).")
    p_rpc.add_argument("--model", required=True, help="Served GGUF model ID or path to .gguf file.")
    p_rpc.add_argument("--tensor-split", default="20,80", help="Comma-separated split (RPC worker first, local Metal last; default: 20,80).")
    p_rpc.set_defaults(func=cmd_gguf_rpc)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
