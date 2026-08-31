---
name: dev-server
description: Spawn, reuse, or tear down an isolated heylookllm server for live verification (temp DB, safe port, RAM pre-flight)
---

# dev-server

Thin wrapper: the harness lives GIT-TRACKED at `scripts/dev_server.sh` (repo
root). Read its header comment for the full usage discipline; the short form:

```bash
# All must run UNSANDBOXED (Metal + localhost + modelzoo traversal).
bash scripts/dev_server.sh status [--port 8991]
bash scripts/dev_server.sh start --model <exact-model-id> [--port 8991] [--headroom-gb 12] [--no-warm]
bash scripts/dev_server.sh stop   [--port 8991]
```

Non-negotiables (enforced by the script, but they apply to you too):

- **Reuse first**: `status` before `start`; drive an already-running server
  when one exists. NEVER kill a heylookllm process this script did not spawn
  (the default port may be a long-running daily instance; others may belong
  to concurrent agents). The script only ever kills its own recorded PID.
- **RAM pre-flight**: `start` sizes the model from models.toml and requires
  size + headroom of currently-available memory (vm_stat), so models resident
  in other processes are automatically accounted for. On failure, reuse or
  downsize -- never free RAM by killing other servers.
- Isolated `HEYLOOK_DB_PATH`, log-to-file (never pipe), probes via
  `uv run python` urllib (sandboxed curl can't reach localhost) -- all inside
  the script; do not re-derive them. Load+warm readiness is SERVER-owned:
  the script (and tests/e2e/lib/server.mjs) call the one canonical
  `POST /v1/models/{id}/load?warm=true` endpoint; never hand-roll
  poll-the-model-list or warm-generation logic.
- Model ids carry quant suffixes; list exact ids from models.toml first and
  default to the fast MoE gemma-4-26B-A4B variant for behavior checks.
- Always `stop` what you started, once, at the end of the whole check series.
