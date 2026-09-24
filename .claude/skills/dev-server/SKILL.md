---
name: dev-server
description: Spawn, reuse, or tear down an isolated heylookllm server for live verification (temp DB, safe port, RAM pre-flight)
---

# dev-server

Thin wrapper: the harness lives GIT-TRACKED at `scripts/dev_server.sh` (repo
root). Read its header comment for the full usage discipline; the short form:

```bash
# All must run UNSANDBOXED (Metal + localhost + model-folder traversal).
bash scripts/dev_server.sh status [--port 8991]
bash scripts/dev_server.sh start --model <exact-model-id> [--port 8991] [--headroom-gb 12] [--no-warm]
bash scripts/dev_server.sh stop   [--port 8991]
```

Non-negotiables (enforced by the script, but they apply to you too):

- **Reuse first.** Run `status` before `start`, and drive an already-running
  server when one exists. NEVER kill a heylookllm process this script did not
  spawn: the default port may be a long-running daily instance, and others may
  belong to concurrent agents. The script only ever kills its own recorded PID.
- **RAM pre-flight.** `start` sizes the model through `scripts/ram_report.py`,
  which resolves it with the same `discover()` + `merge_discovered()` the router
  uses. A DISCOVERED model (no models.toml entry) therefore sizes like an
  explicit one. It requires size + headroom of reclaimable memory, so models
  resident in other processes are accounted for. On failure, reuse or
  downsize; never free RAM by killing other servers.
- **Do not re-derive the plumbing.** The isolated `HEYLOOK_DB_PATH`,
  log-to-file (never pipe), and probes via `uv run python` urllib (sandboxed
  curl can't reach localhost) all live inside the script.
- **Readiness is server-owned.** The script (and tests/e2e/lib/server.mjs)
  call the one canonical `POST /v1/models/{id}/load?warm=true` endpoint. Never
  hand-roll poll-the-model-list or warm-generation logic.
- **Model ids.** Take them from `GET /v1/models` on a running server, or from
  `merge_discovered(data, discover(data))`, NOT from models.toml alone. Most
  models are discovered and have no entry there.
  - Plumbing and contract checks: the small Qwen3.5-0.8B.
  - Quality-dependent checks: the fast MoE gemma-4-26B-A4B.
- Always `stop` what you started, once, at the end of the whole check series.

Below the server, for one GGUF model with no FastAPI or DB in the way,
`scripts/gguf_probe.py` drives llama-server directly.
