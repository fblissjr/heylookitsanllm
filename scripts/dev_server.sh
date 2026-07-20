#!/usr/bin/env bash
# Isolated heylookllm dev server for live verification.
#   server.sh start [--port N] [--model ID] [--headroom-gb N] [--no-warm]
#   server.sh stop  [--port N]
#   server.sh status [--port N]
#
# Guarantees:
#   - Isolated DB (HEYLOOK_DB_PATH under the state dir) -- never touches real data.
#   - Logs to a file, never a pipe (piping to head SIGPIPE-wedges the server).
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
  uv run python - "$1" <<'PY' 2>/dev/null || true
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
  uv run python - "$BASE" <<'PY' 2>/dev/null || true
import sys, json, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/v1/models", timeout=5) as r:
        print(",".join(m["id"] for m in json.load(r).get("data", [])))
except Exception:
    pass
PY
}

case "$CMD" in
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "devserver RUNNING pid=$(cat "$PIDFILE") port=$PORT models=$(probe)"
    else
      echo "devserver NOT RUNNING on port $PORT"
    fi
    OTHERS=$(pgrep -fl "heylookllm" | grep -v "grep" || true)
    if [ -n "$OTHERS" ]; then
      echo "other heylookllm processes (NOT ours -- never kill these):"; echo "$OTHERS"
    fi
    ;;

  start)
    [ -n "$MODEL" ] || { echo "start requires --model <id>" >&2; exit 2; }
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"), models=$(probe)) -- reuse it or stop first"; exit 0
    fi
    if [ -n "$(probe)" ]; then
      echo "port $PORT already serving (models=$(probe)) but not started by this script -- refusing. Reuse it read-only or pick another --port." >&2
      exit 1
    fi

    AVAIL=$(avail_gb)
    SIZE=$(cd "$REPO_ROOT" && model_size_gb "$MODEL")
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

    DEADLINE=$(( $(date +%s) + 300 ))
    until probe | grep -q "$MODEL"; do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server exited during startup; tail of log:" >&2; tail -20 "$LOG" >&2; exit 1; }
      [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "timed out waiting for /v1/models; tail of log:" >&2; tail -20 "$LOG" >&2; exit 1; }
      sleep 2
    done
    echo "READY: $BASE (model $MODEL listed)"

    if [ "$WARM" = 1 ]; then
      uv run python - "$BASE" "$MODEL" <<'PY'
import sys, json, urllib.request
req = urllib.request.Request(
    sys.argv[1] + "/v1/chat/completions",
    data=json.dumps({"model": sys.argv[2], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    json.load(r)
print("warm generation OK (Metal kernels JIT'd)")
PY
    fi
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
