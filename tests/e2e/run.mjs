// E2E orchestrator: spawn heylookllm with an isolated DB, launch system Chrome,
// run the chat + pages suites against the frontend, tear everything down, exit non-zero on
// any failure.
//
//   bun run e2e             # both suites, ONE arm
//   bun run e2e:chat        # chat suite only
//   bun run e2e:pages       # pages suite only
//   E2E_ARMS=all bun run e2e        # every engine arm (mlx-lm, mlx-vlm, gguf)
//   E2E_ARMS=gguf,mlx-lm bun run e2e
//
// ARMS ARE ENGINES, not providers: "mlx" is TWO upstream repos (mlx-lm text /
// mlx-vlm vision, separate release trains), so a text arm and a vision arm are
// different code. The mapping is the SERVER's answer (`effective_loader`), read
// through tests/helpers/engines.py -- the same module tests/smoke and
// tests/eval use. The JS side shells out to it rather than re-deriving "which
// engine is this model" in a second language.
//
// ONE ARM BY DEFAULT, on purpose. max_loaded_models = 1, so every extra arm
// pays a full weight load plus Metal warm and evicts the previous one: arm
// count is the cost driver here, not tokens. A matrix run is a deliberate act.
//
// (The package.json script bodies intentionally shell out to `node run.mjs`
// -- the harness runs under node by design; invoke it VIA bun, whose
// non-interactive script shell resolves the real node binary and dodges
// the interactive shell's nvm lazy-load function.)
//
// Config via env:
//   E2E_MODEL      model id to preload + drive        (default: gemma-4-26b-a4b-it-8bit-mlx)
//   E2E_ARMS       engine arms to run, or "all"        (default: the single E2E_MODEL arm)
//   E2E_PORT       server port                        (default: 1264 -- NOT 8000, the daily server; the harness spawns its own)
//   E2E_MAX_TOKENS default per-generation token cap   (default: 24)
//   E2E_CHROME     path to Chrome binary              (default: /Applications/Google Chrome.app/...)
//   E2E_HEADFUL    set to run Chrome with a window (debugging)
//   E2E_BASE_URL   drive an ALREADY-RUNNING server instead of spawning one.
//                  DANGEROUS: that server's real DB gets cleared -- requires
//                  E2E_ALLOW_SHARED_DB=1 to proceed.

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

import { startServer, loadAndWarm } from './lib/server.mjs';
import { launchBrowser, createPageContext } from './lib/browser.mjs';
import { Suite, printSummary } from './lib/harness.mjs';
import { runChatSuite } from './suites/chat.mjs';
import { runPagesSuite } from './suites/pages.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..', '..');

const execFileAsync = promisify(execFile);

const CONFIG = {
  model: process.env.E2E_MODEL || 'gemma-4-26b-a4b-it-8bit-mlx',
  arms: (process.env.E2E_ARMS || '').trim(),
  port: Number(process.env.E2E_PORT || 1264),
  maxTokens: Number(process.env.E2E_MAX_TOKENS || 24),
  baseUrl: process.env.E2E_BASE_URL || null,
};

const which = (process.argv[2] || 'all').toLowerCase();
const runChat = which === 'all' || which === 'chat';
const runPages = which === 'all' || which === 'pages';

// The engine taxonomy is Python's (tests/helpers/engines.py) and stays that
// way: it reads `effective_loader` off the admin row, so the SERVER names the
// engine. Re-deriving that here would be the hand-copied-list defect this repo
// keeps paying for, in a second language where it could drift silently.
async function resolveArms(base, spec) {
  const wanted = spec === 'all' ? [] : spec.split(',').map((a) => a.trim()).filter(Boolean);
  const args = ['run', 'python', '-m', 'helpers.engines', '--server', base, '--json'];
  for (const a of wanted) args.push('--arm', a);
  const { stdout } = await execFileAsync('uv', args, { cwd: join(REPO_ROOT, 'tests'), maxBuffer: 1 << 20 });
  return JSON.parse(stdout);
}

// A red is only attributable if you know what produced it. Chrome updates
// itself underneath this harness and nothing recorded which one ran.
async function versionLine(browser) {
  let chrome = 'unknown';
  try { chrome = await browser.version(); } catch { /* keep unknown */ }
  let puppeteer = 'unknown';
  try {
    const pkg = await import('puppeteer-core/package.json', { with: { type: 'json' } });
    puppeteer = pkg.default?.version ?? 'unknown';
  } catch { /* keep unknown */ }
  return `[e2e] browser: ${chrome} | puppeteer-core ${puppeteer}`;
}

async function main() {
  let server = null;
  let browser = null;
  let tmp = null;
  const suites = [];

  try {
    let base;
    if (CONFIG.baseUrl) {
      if (process.env.E2E_ALLOW_SHARED_DB !== '1') {
        throw new Error(
          'E2E_BASE_URL is set but E2E_ALLOW_SHARED_DB != 1. The suites CLEAR ALL ' +
          'conversations & notebooks on the target server. Spawn a fresh isolated ' +
          'server (unset E2E_BASE_URL) or acknowledge with E2E_ALLOW_SHARED_DB=1.');
      }
      base = CONFIG.baseUrl.replace(/\/$/, '');
      console.log(`[e2e] driving existing server at ${base} (SHARED DB -- data will be cleared)`);
      server = {
        base,
        clearData: async () => { await fetch(`${base}/v1/data/clear`, { method: 'POST' }); },
        stop: async () => {},
      };
    } else {
      tmp = await mkdtemp(join(tmpdir(), 'heylook-e2e-'));
      const dbPath = join(tmp, 'e2e.db');
      const logPath = join(tmp, 'server.log');
      console.log(`[e2e] spawning heylookllm (model=${CONFIG.model} port=${CONFIG.port})`);
      console.log(`[e2e] isolated DB: ${dbPath}`);
      console.log(`[e2e] server log:  ${logPath}`);
      console.log('[e2e] waiting for model load + warm generation…');
      server = await startServer({
        port: CONFIG.port,
        dbPath,
        modelId: CONFIG.model,
        repoRoot: REPO_ROOT,
        logPath,
      });
      base = server.base;
      console.log('[e2e] server ready.');
    }

    browser = await launchBrowser();
    console.log(await versionLine(browser));

    // One arm by default: whatever E2E_MODEL already is, no extra load. Naming
    // arms resolves each one's model from the server and loads it in turn.
    let armPlan = [{ arm: null, model: CONFIG.model }];
    if (CONFIG.arms) {
      const resolved = await resolveArms(base, CONFIG.arms);
      armPlan = Object.entries(resolved.arms).map(([arm, info]) => ({ arm, model: info.model }));
      for (const arm of resolved.absent) {
        // UNCOVERED, never green. "Served but not run" and "no model of this
        // engine exists" are different facts and must not print the same.
        console.log(`[e2e] arm ${arm}: NO MODEL SERVED -- uncovered, not passed`);
      }
      if (!armPlan.length) throw new Error(`no arm resolved for E2E_ARMS=${CONFIG.arms}`);
      console.log(`[e2e] arms: ${armPlan.map((a) => `${a.arm}=${a.model}`).join(' ')}`);
    }

    const plan = [
      { name: 'chat', enabled: runChat, run: runChatSuite },
      { name: 'pages', enabled: runPages, run: runPagesSuite },
    ];
    for (const { arm, model } of armPlan) {
      if (arm) {
        console.log(`\n[e2e] === arm ${arm}: loading ${model} ===`);
        // Server-owned load+warm, the same call startServer makes. Each arm
        // evicts the previous one -- max_loaded_models is 1.
        await loadAndWarm(base, model);
      }
      const armConfig = { ...CONFIG, model, arm };
      for (const { name, enabled, run } of plan) {
        if (!enabled) continue;
        // Suite names are keyed in printSummary, so arms must not collide.
        const suite = new Suite(arm ? `${name} [${arm}]` : name);
        console.log(`\n${suite.name} suite`);
        await server.clearData();
        const page = await browser.newPage();
        const ctx = createPageContext(page, { base, maxTokens: CONFIG.maxTokens });
        try {
          await run({ suite, ctx, config: armConfig });
        } finally {
          await page.close();
        }
        suites.push(suite);
      }
    }
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (server) await server.stop().catch(() => {});
    if (tmp) await rm(tmp, { recursive: true, force: true }).catch(() => {});
  }

  const failed = printSummary(suites);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(`\n[e2e] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
