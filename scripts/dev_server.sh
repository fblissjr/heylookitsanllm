#!/usr/bin/env bash
# Isolated heylookllm dev server for live verification.
#   server.sh start [--port N] [--model ID] [--headroom-gb N] [--no-warm]
#   server.sh stop  [--port N]
#   server.sh status [--port N]
#
# Guarantees:
#   - Isolated DB (HEYLOOK_DB_PATH under the state dir) -- never touches real data.
#   - Logs to a file, never a pipe (piping to head SIGPIPE-wedges the server).
#   - Readiness is SERVER-OWNED: POST /v1/admin/models/{id}/load?warm=true is
#     the one canonical load+warm call (tests/e2e/lib/server.mjs uses the same
#     contract) -- this script never re-invents poll/warm semantics.
#   - RAM pre-flight: refuses to start if the model + headroom exceeds what the
#     machine has AVAILABLE RIGHT NOW (so a model already resident in another
#     server/agent's process is automatically accounted for).
#   - Only ever kills the PID it spawned itself (recorded in the state dir).
#
# Must run UNSANDBOXED: needs Metal, localhost, and modelzoo traversal.
#
# Usage discipline (for agents and humans):
#   - Reuse first: run `status` before `start`. If ANY heylookllm process is
#     already running (this script's or not), prefer driving it -- its resident
#     model is RAM someone paid for. NEVER kill a server you did not spawn
#     (port 8080 is typically the owner's daily server).
#   - Default model for behavior checks: the fast MoE gemma-4-26B-A4B variant
#     (~90 tok/s, the discriminating model per CLAUDE.md); ids in models.toml
#     carry quant suffixes, so list exact ids first.
#   - Always `stop` a server you started; keep it up across a series of checks
#     (model load is the expensive part), then stop once at the end.
set -euo pipefail

CMD="${1:-status}"; shift || true
PORT=8991
MODEL=""
HEADROOM_GB=12
WARM=1

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --headroom-gb) HEADROOM_GB="$2"; shift 2 ;;
    --no-warm) WARM=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="${TMPDIR:-/tmp}/heylook-devserver-${PORT}"
PIDFILE="$STATE/pid"
LOG="$STATE/server.log"
BASE="http://127.0.0.1:${PORT}"

avail_gb() {
  vm_stat | awk '
    /page size of/ {ps=$8}
    /Pages free/ {gsub(/\./,"",$3); free=$3}
    /Pages inactive/ {gsub(/\./,"",$3); inact=$3}
    /Pages purgeable/ {gsub(/\./,"",$3); purg=$3}
    END {printf "%.0f", (free+inact+purg)*ps/1073741824}'
}

model_size_gb() {
  # Resolve model id -> weights dir via models.toml, then du. Empty on failure.
  ( cd "$REPO_ROOT" && uv run python - "$1" <<'PY' 2>/dev/null ) || true
import sys, tomllib, pathlib
mid = sys.argv[1]
cfg = tomllib.loads(pathlib.Path("models.toml").read_text())
for m in cfg.get("models", []):
    if m.get("id") == mid:
        p = pathlib.Path(m.get("config", {}).get("model_path", ""))
        if p.is_dir():
            print(round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9))
        break
PY
}

probe() {
  # Sandboxed curl can't reach localhost; python urllib works everywhere.
  # cd first: `uv run` outside the repo fails silently and would turn the
  # foreign-server port guard into a no-op.
  ( cd "$REPO_ROOT" && uv run python - "$BASE" <<'PY' 2>/dev/null ) || true
import sys, json, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/v1/models", timeout=5) as r:
        print(",".join(m["id"] for m in json.load(r).get("data", [])))
except Exception:
    pass
PY
}

has_model() {
  # Exact-id membership test on a probe() result (grep on the raw list would
  # substring-match prefix ids and treat dots as regex metachars).
  printf '%s' "$1" | tr ',' '\n' | grep -Fxq "$2"
}

abort_start() {
  # Failed start: kill the pid WE spawned (and only that), clear state so a
  # retry doesn't see a live pidfile and report false success.
  local reason="$1"
  echo "$reason; tail of log:" >&2
  tail -20 "$LOG" >&2 || true
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    echo "killed spawned pid $(cat "$PIDFILE") (failed start must not leave an orphan)" >&2
  fi
  rm -rf "$STATE"
  exit 1
}

case "$CMD" in
  status)
    OWN_PIDS=""
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      OWN=$(cat "$PIDFILE")
      # Ours = the recorded wrapper pid plus its children (uv spawns the
      # actual server as a child); everything else is foreign.
      OWN_PIDS="$OWN $(pgrep -P "$OWN" || true)"
      echo "devserver RUNNING pid=$OWN port=$PORT models=$(probe)"
    else
      echo "devserver NOT RUNNING on port $PORT"
    fi
    OTHERS=$(pgrep -fl "heylookllm" | grep -v "grep" || true)
    for pid in $OWN_PIDS; do
      OTHERS=$(printf '%s\n' "$OTHERS" | awk -v p="$pid" '$1 != p') || true
    done
    if [ -n "$OTHERS" ]; then
      echo "other heylookllm processes (NOT ours -- never kill these):"; echo "$OTHERS"
    fi
    ;;

  start)
    [ -n "$MODEL" ] || { echo "start requires --model <id>" >&2; exit 2; }
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"), models=$(probe)) -- reuse it or stop first"; exit 0
    fi
    SERVING=$(probe)
    if [ -n "$SERVING" ]; then
      echo "port $PORT already serving (models=$SERVING) but not started by this script -- refusing. Reuse it read-only or pick another --port." >&2
      exit 1
    fi

    AVAIL=$(avail_gb)
    SIZE=$(model_size_gb "$MODEL")
    if [ -n "$SIZE" ]; then
      NEED=$((SIZE + HEADROOM_GB))
      if [ "$AVAIL" -lt "$NEED" ]; then
        echo "RAM pre-flight FAILED: model ~${SIZE}GB + ${HEADROOM_GB}GB headroom = ${NEED}GB, but only ~${AVAIL}GB available now (another server/agent may hold a model). Not starting." >&2
        exit 1
      fi
      echo "RAM pre-flight OK: need ~${NEED}GB, ~${AVAIL}GB available"
    else
      echo "WARNING: could not size model '$MODEL' from models.toml; skipping RAM check (~${AVAIL}GB available)" >&2
    fi

    mkdir -p "$STATE"
    : > "$LOG"
    cd "$REPO_ROOT"
    HEYLOOK_DB_PATH="$STATE/db.duckdb" nohup uv run heylookllm \
      --host 127.0.0.1 --port "$PORT" --model-id "$MODEL" --log-level WARNING \
      >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "spawned pid $(cat "$PIDFILE"), waiting for readiness (log: $LOG)"

    # Readiness = HTTP up + model id CONFIGURED (exact match). /v1/models
    # lists enabled models.toml entries regardless of load state -- actual
    # model LOAD is absorbed by the warm request below, not this wait.
    DEADLINE=$(( $(date +%s) + 120 ))
    while :; do
      LISTED=$(probe)
      [ -n "$LISTED" ] && break
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || abort_start "server exited during startup"
      [ "$(date +%s)" -lt "$DEADLINE" ] || abort_start "timed out waiting for /v1/models to answer"
      sleep 2
    done
    if ! has_model "$LISTED" "$MODEL"; then
      abort_start "model '$MODEL' is not in the server's enabled model list (check the exact id in models.toml; got: $LISTED)"
    fi

    # Load (+ optionally warm) via the ONE canonical server-side call --
    # POST /v1/admin/models/{id}/load?warm= owns readiness semantics (weights
    # in LRU + first-forward-pass Metal JIT through the real generation
    # path). tests/e2e/lib/server.mjs is the node client of the same
    # contract; keep the two in sync only via that endpoint, never by
    # re-inventing poll/warm logic here.
    echo "loading model $MODEL (server-side load$([ "$WARM" = 1 ] && echo '+warm')) ..."
    if ! ( cd "$REPO_ROOT" && uv run python - "$BASE" "$MODEL" "$WARM" <<'PY' )
import json, sys, urllib.parse, urllib.request
base, model, warm = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
url = f"{base}/v1/admin/models/{urllib.parse.quote(model, safe='')}/load?warm={'true' if warm else 'false'}"
with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=900) as r:
    data = json.load(r)
if warm and data.get("warmed"):
    print(f"warm generation OK ({data.get('warm_ms', '?')} ms, Metal kernels JIT'd)")
elif warm:
    # Loaded but warm failed: server is usable; surface as a warning.
    print(f"WARNING: warm failed: {data.get('warm_error')}", file=sys.stderr)
PY
    then
      abort_start "load request failed for '$MODEL'"
    fi
    echo "READY: $BASE ($MODEL loaded)"
    ;;

  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      PID=$(cat "$PIDFILE")
      kill "$PID"
      for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
      kill -0 "$PID" 2>/dev/null && kill -9 "$PID" || true
      echo "stopped pid $PID"
    else
      echo "nothing to stop on port $PORT (pidfile absent or stale)"
    fi
    rm -rf "$STATE"
    ;;

  *) echo "usage: server.sh start|stop|status [--port N] [--model ID] [--headroom-gb N] [--no-warm]" >&2; exit 2 ;;
esac
