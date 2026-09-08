#!/usr/bin/env bash
# Isolated heylookllm dev server for live verification.
#   server.sh start [--port N] [--model ID] [--headroom-gb N] [--no-warm]
#   server.sh stop  [--port N]
#   server.sh status [--port N]
#
# Guarantees:
#   - Isolated DB (HEYLOOK_DB_PATH under the state dir) -- never touches real data.
#   - Logs to a file, never a pipe (piping to head SIGPIPE-wedges the server).
#   - Readiness is SERVER-OWNED: POST /v1/models/{id}/load?warm=true is
#     the one canonical load+warm call (tests/e2e/lib/server.mjs uses the same
#     contract) -- this script never re-invents poll/warm semantics.
#   - RAM pre-flight: refuses to start if the model + headroom exceeds what the
#     machine has AVAILABLE RIGHT NOW (so a model already resident in another
#     server/agent's process is automatically accounted for) OR what the GPU
#     will hold. Delegated to `scripts/ram_report.py --quiet` -- the sizing has
#     traps (a GGUF model_path names ONE shard of a set; sidecars load into the
#     same process; the Metal working-set ceiling is well below total RAM) and
#     they are worth getting wrong in exactly zero places.
#   - Only ever kills the PID it spawned itself (recorded in the state dir).
#
# Must run UNSANDBOXED: needs Metal, localhost, and modelzoo traversal.
#
# Usage discipline (for agents and humans):
#   - Reuse first: run `status` before `start`. If ANY heylookllm process is
#     already running (this script's or not), prefer driving it -- its resident
#     model is RAM someone paid for. NEVER kill a server you did not spawn
#     (an already-running heylookllm on any port may be the owner's daily
#     server).
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

ram_preflight() {
  # Delegates to scripts/ram_report.py: prints one line, exits non-zero when
  # the model won't fit. Sizing a GGUF entry here by hand is what made the
  # old inline version clear a 155 GB model as needing 10 GB -- model_path
  # names one shard of a set.
  ( cd "$REPO_ROOT" && uv run python scripts/ram_report.py \
      --model "$1" --headroom "$2" --quiet )
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
      # "no live pidfile" is NOT "nothing is running": the recorded pid is the
      # `uv run` wrapper and it can exit while the server it spawned keeps the
      # port. Saying NOT RUNNING on that evidence alone is how an orphan hides
      # in plain sight (2026-09-08). Ask the port too.
      HOLDER=$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)
      if [ -n "$HOLDER" ]; then
        echo "devserver ORPHANED on port $PORT: pidfile is stale but pid(s)$HOLDER hold the port (models=$(probe))"
        echo "  \`$0 stop --port $PORT\` resolves the target from the port and will clean it up"
      else
        echo "devserver NOT RUNNING on port $PORT"
      fi
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

    if PREFLIGHT=$(ram_preflight "$MODEL" "$HEADROOM_GB"); then
      echo "$PREFLIGHT"
    else
      case "$PREFLIGHT" in
        # A model that could not be SIZED exits 2 with no verdict line; that
        # is a bad --model, not a memory refusal, so say which it was. The id
        # is resolved through the server's own registry merge, so "unknown"
        # means neither a models.toml entry NOR discovered under
        # [scan].folders -- models.toml alone was never the right question.
        "") echo "RAM pre-flight: could not size model '$MODEL' (unknown id, or its files no longer read -- reason above). Not starting." >&2 ;;
        *)  echo "$PREFLIGHT (another server/agent may hold a model). Not starting." >&2 ;;
      esac
      echo "  run: uv run python scripts/ram_report.py --model $MODEL   # for the full breakdown" >&2
      exit 1
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
    # POST /v1/models/{id}/load?warm= owns readiness semantics (weights
    # in LRU + first-forward-pass Metal JIT through the real generation
    # path). tests/e2e/lib/server.mjs is the node client of the same
    # contract; keep the two in sync only via that endpoint, never by
    # re-inventing poll/warm logic here.
    echo "loading model $MODEL (server-side load$([ "$WARM" = 1 ] && echo '+warm')) ..."
    if ! ( cd "$REPO_ROOT" && uv run python - "$BASE" "$MODEL" "$WARM" <<'PY' )
import json, sys, urllib.parse, urllib.request
base, model, warm = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
url = f"{base}/v1/models/{urllib.parse.quote(model, safe='')}/load?warm={'true' if warm else 'false'}"
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
    # THE PORT IS THE AUTHORITY, the pidfile only a hint. What gets recorded is
    # the `uv run` WRAPPER pid, and that wrapper can exit while the python
    # server it spawned keeps running and keeps the port. Observed 2026-09-08:
    # `kill -0 $(cat pid)` failed on the dead wrapper, stop reported "nothing
    # to stop", and a server was live on the port the whole time -- with its
    # llama-server subprocess still resident. Worse, the old `rm -rf "$STATE"`
    # ran OUTSIDE the if, so that miss deleted the pidfile and every later stop
    # missed for the same reason. Self-perpetuating, and exactly the orphan
    # class CLAUDE.md documents for the e2e harness.
    TARGETS=""
    add_target() {
      case " $TARGETS " in *" $1 "*) ;; *) TARGETS="$TARGETS $1" ;; esac
    }
    if [ -f "$PIDFILE" ]; then
      HINT=$(cat "$PIDFILE" 2>/dev/null || true)
      if [ -n "$HINT" ] && kill -0 "$HINT" 2>/dev/null; then
        add_target "$HINT"
        for c in $(pgrep -P "$HINT" 2>/dev/null || true); do add_target "$c"; done
      fi
    fi
    for p in $(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true); do
      add_target "$p"
    done

    # Refuse to kill something that is not ours. `start` already declines to
    # take a port a foreign server holds; stop must be equally careful, or a
    # mistyped --port turns this into a process killer for unrelated software.
    FOREIGN=""
    for p in $TARGETS; do
      case "$(ps -o args= -p "$p" 2>/dev/null || true)" in
        *heylookllm*|*heylook_llm*|*llama-server*) ;;
        *) FOREIGN="$FOREIGN $p" ;;
      esac
    done
    if [ -n "$FOREIGN" ]; then
      echo "refusing: port $PORT is held by a process that is not a heylook server:" >&2
      for p in $FOREIGN; do ps -o pid=,args= -p "$p" 2>/dev/null | cut -c1-120 >&2; done
      exit 2
    fi

    if [ -z "$TARGETS" ]; then
      echo "nothing running on port $PORT"
      rm -rf "$STATE"          # safe: the port is free, so no handle is lost
      exit 0
    fi

    # Capture the llama-server subprocesses BEFORE killing their parent. A
    # graceful shutdown reaps them itself (lifespan -> unload_all -> SIGTERM),
    # but a SIGKILL escalation would orphan them, and they are the expensive
    # thing to leave behind. Collected by parentage, so this can never reach
    # another session's subprocess.
    KIDS=""
    for p in $TARGETS; do
      for c in $(pgrep -P "$p" 2>/dev/null || true); do
        case "$(ps -o args= -p "$c" 2>/dev/null || true)" in
          *llama-server*) KIDS="$KIDS $c" ;;
        esac
      done
    done

    for p in $TARGETS; do kill "$p" 2>/dev/null || true; done
    for _ in $(seq 1 30); do
      [ -z "$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ] && break
      sleep 1
    done
    for p in $TARGETS $KIDS; do
      kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true
    done
    sleep 1

    STILL=$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
    LEFT=""
    for p in $KIDS; do kill -0 "$p" 2>/dev/null && LEFT="$LEFT $p"; done
    if [ -n "$STILL" ] || [ -n "$LEFT" ]; then
      # Say so and KEEP the state dir: its pidfile is the only handle a human
      # has left, and deleting it is what made this unrecoverable last time.
      echo "FAILED to fully stop port $PORT -- still listening:${STILL:- none}, llama-server left:${LEFT:- none}" >&2
      echo "state kept at $STATE" >&2
      exit 1
    fi
    echo "stopped$TARGETS${KIDS:+ (and llama-server$KIDS)}"
    rm -rf "$STATE"
    ;;

  *) echo "usage: server.sh start|stop|status [--port N] [--model ID] [--headroom-gb N] [--no-warm]" >&2; exit 2 ;;
esac
