#!/usr/bin/env bash
# Start the llama.cpp ggml-rpc worker daemon (`ggml-rpc-server`) on a peer Mac
# (e.g. MacBook Pro M4 Pro over Thunderbolt Bridge).
#
# Usage:
#   scripts/rpc_worker.sh [--host <worker-ip>] [--port 50052] [--no-cache]
#
# By default `--cache` (`-c`) is enabled so weight tensors > 10 MB are cached
# by 64-bit FNV-1a hash on the worker's local SSD across model reloads.
set -euo pipefail

HOST="${HEYLOOK_RPC_HOST:-0.0.0.0}"
PORT="${HEYLOOK_RPC_PORT:-50052}"
CACHE_FLAG="-c"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --host|-H) HOST="$2"; shift 2 ;;
    --port|-p) PORT="$2"; shift 2 ;;
    --no-cache) CACHE_FLAG=""; shift ;;
    --cache|-c) CACHE_FLAG="-c"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

RPC_BIN="$(python3 -c '
import os, pathlib
base = pathlib.Path(os.environ.get("HEYLOOK_LLAMA_CPP_DIR") or pathlib.Path.home().joinpath(".heylook", "llama.cpp")).expanduser().resolve() / "build" / "bin"
cand = base / "ggml-rpc-server"
if not cand.exists() and (base / "rpc-server").exists():
    cand = base / "rpc-server"
print(cand)
')"

if [ ! -x "$RPC_BIN" ]; then
  echo "error: ggml-rpc-server binary not found at $RPC_BIN" >&2
  echo "Build it first with: uv run scripts/build_llama.py" >&2
  exit 1
fi

CMD=("$RPC_BIN" -H "$HOST" -p "$PORT")
if [ -n "$CACHE_FLAG" ]; then
  CMD+=("$CACHE_FLAG")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting ggml-rpc worker: ${CMD[*]}"
exec "${CMD[@]}"
