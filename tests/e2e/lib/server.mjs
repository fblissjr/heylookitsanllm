// Spawn heylookllm with an ISOLATED conversation DB (HEYLOOK_DB_PATH) so the
// suites -- which create and clear conversations/notebooks -- never touch real
// data. Readiness = /v1/models responds, then ONE canonical server-side call:
// POST /v1/models/{id}/load?warm=true (loads weights + runs a 1-token
// generation through the real generation path, paying the Metal-kernel JIT).
// The server owns load/warm semantics; scripts/dev_server.sh is the bash
// client of the same contract -- never re-invent poll/warm logic here.

import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { connect } from 'node:net';
import { sleep } from './harness.mjs';

/** Refuse to start when something already listens on `port`.
 *
 * Without this the harness silently tests the WRONG SERVER. The spawned
 * `heylookllm` loses the bind and exits, but that takes seconds (Python
 * import), while Phase 1's first iteration is immediate: it sees `exited`
 * still null, fetches /v1/models from the STRANGER, finds the model listed
 * and returns. loadAndWarm then warms the stranger and every suite runs
 * against it. Teardown makes it self-perpetuating -- `stop()` returns early
 * because our child really did exit, so the stranger survives to capture the
 * next run too. An orphan from an earlier run was found squatting this port
 * on 2026-09-06 running a build four releases stale; nothing in the output
 * said so, and the run would have reported a full green for code that was
 * not executing.
 *
 * Checked by CONNECT rather than by binding: a bind test races (we would have
 * to release the port before the child claims it) and answers a different
 * question -- what matters is whether a peer is there to answer, not whether
 * the address is momentarily free.
 */
async function refuseIfPortBusy(port) {
  const busy = await new Promise((resolve) => {
    const sock = connect({ host: '127.0.0.1', port });
    const done = (v) => { sock.destroy(); resolve(v); };
    sock.setTimeout(2000);
    sock.on('connect', () => done(true));
    sock.on('timeout', () => done(false));
    sock.on('error', () => done(false));
  });
  if (busy) {
    throw new Error(
      `port ${port} is already in use -- refusing to start.\n` +
      `  Something is listening on 127.0.0.1:${port}. The harness would have ` +
      `silently run every suite against THAT server instead of a fresh one.\n` +
      `  Most likely an orphaned heylookllm from an earlier run. Find it with ` +
      `\`lsof -nP -iTCP:${port} -sTCP:LISTEN\` and kill it, or set E2E_PORT to a free port.`);
  }
}

async function fetchJson(url, opts = {}, timeoutMs = 10000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    const body = await res.json().catch(() => null);
    return { status: res.status, body };
  } finally {
    clearTimeout(t);
  }
}

/** Canonical server-side load + warm. Blocks until weights are in the LRU and
 * a 1-token generation has run (Metal kernels JIT'd).
 *
 * Exported because a multi-arm run loads a model per arm, and the one thing a
 * harness must never do is re-invent poll/warm semantics -- they are
 * server-owned. Same call `scripts/dev_server.sh` makes.
 */
export async function loadAndWarm(base, modelId, { timeoutMs = 300000, note = () => '' } = {}) {
  const url = `${base}/v1/models/${encodeURIComponent(modelId)}/load?warm=true`;
  const { status, body } = await fetchJson(url, { method: 'POST' }, timeoutMs);
  if (status !== 200) {
    throw new Error(`load?warm=true returned ${status} for ${modelId}: ${JSON.stringify(body)}${note()}`);
  }
  if (!body?.warmed) {
    throw new Error(`load succeeded but warm failed for ${modelId}: ${body?.warm_error}${note()}`);
  }
}

export async function startServer({ port, dbPath, modelId, repoRoot, logPath }) {
  await refuseIfPortBusy(port);

  const log = createWriteStream(logPath, { flags: 'a' });
  const args = [
    'run', 'heylookllm',
    '--host', '127.0.0.1',
    '--port', String(port),
    '--model-id', modelId,
    '--log-level', 'WARNING',
  ];
  const proc = spawn('uv', args, {
    cwd: repoRoot,
    env: { ...process.env, HEYLOOK_DB_PATH: dbPath },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.pipe(log);
  proc.stderr.pipe(log);

  let exited = null;
  proc.on('exit', (code, signal) => { exited = { code, signal }; });

  const base = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 300000; // model load can be slow (large weights)

  // Phase 1: server socket answers /v1/models with our model present.
  await (async () => {
    while (Date.now() < deadline) {
      if (exited) throw new Error(`server exited during startup (code=${exited.code} signal=${exited.signal}); see ${logPath}`);
      try {
        const { status, body } = await fetchJson(`${base}/v1/models`, {}, 5000);
        if (status === 200 && Array.isArray(body?.data) && body.data.some((m) => m.id === modelId)) return;
      } catch { /* not up yet */ }
      await sleep(500);
    }
    throw new Error(`server /v1/models never listed ${modelId}; see ${logPath}`);
  })();

  // Phase 2: canonical server-side load+warm.
  await loadAndWarm(base, modelId, {
    timeoutMs: deadline - Date.now(),
    note: () => (exited ? ` (server exited, code=${exited.code}); see ${logPath}` : `; see ${logPath}`),
  });

  return {
    proc,
    base,
    async clearData() {
      await fetchJson(`${base}/v1/data/clear`, { method: 'POST' }, 10000);
    },
    async stop() {
      if (exited) return;
      proc.kill('SIGTERM');
      for (let i = 0; i < 40 && !exited; i++) await sleep(250);
      if (!exited) proc.kill('SIGKILL');
      log.end();
    },
  };
}
