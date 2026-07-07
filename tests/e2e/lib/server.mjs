// Spawn heylookllm with an ISOLATED conversation DB (HEYLOOK_DB_PATH) so the
// suites -- which create and clear conversations/notebooks -- never touch real
// data. Readiness = /v1/models responds AND a warm generation completes (which
// forces the preloaded model through its first real forward pass).

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

  // Phase 2: warm generation -- proves the preloaded model can actually decode
  // before the browser suites start (first forward pass JITs Metal kernels).
  await (async () => {
    while (Date.now() < deadline) {
      if (exited) throw new Error(`server exited during warmup (code=${exited.code}); see ${logPath}`);
      try {
        const { status, body } = await fetchJson(`${base}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelId, messages: [{ role: 'user', content: 'hi' }], max_tokens: 1, stream: false }),
        }, 240000);
        if (status === 200 && body?.choices) return;
      } catch { /* model still loading */ }
      await sleep(1000);
    }
    throw new Error(`warm generation never succeeded for ${modelId}; see ${logPath}`);
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
