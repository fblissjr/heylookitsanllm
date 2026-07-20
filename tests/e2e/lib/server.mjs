// Spawn heylookllm with an ISOLATED conversation DB (HEYLOOK_DB_PATH) so the
// suites -- which create and clear conversations/notebooks -- never touch real
// data. Readiness = /v1/models responds, then ONE canonical server-side call:
// POST /v1/admin/models/{id}/load?warm=true (loads weights + runs a 1-token
// generation through the real generation path, paying the Metal-kernel JIT).
// The server owns load/warm semantics; scripts/dev_server.sh is the bash
// client of the same contract -- never re-invent poll/warm logic here.

import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { sleep } from './harness.mjs';

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

export async function startServer({ port, dbPath, modelId, repoRoot, logPath }) {
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

  // Phase 2: canonical server-side load+warm. Blocks until weights are in the
  // LRU and a 1-token generation has run (Metal kernels JIT'd).
  await (async () => {
    const url = `${base}/v1/admin/models/${encodeURIComponent(modelId)}/load?warm=true`;
    const { status, body } = await fetchJson(url, { method: 'POST' }, deadline - Date.now());
    if (exited) throw new Error(`server exited during load/warm (code=${exited.code}); see ${logPath}`);
    if (status !== 200) throw new Error(`load?warm=true returned ${status} for ${modelId}: ${JSON.stringify(body)}; see ${logPath}`);
    if (!body?.warmed) throw new Error(`load succeeded but warm failed for ${modelId}: ${body?.warm_error}; see ${logPath}`);
  })();

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
