// E2E orchestrator: spawn heylookllm with an isolated DB, launch system Chrome,
// run the chat + pages suites against /v3, tear everything down, exit non-zero on
// any failure.
//
//   node run.mjs            # both suites
//   node run.mjs chat       # chat suite only
//   node run.mjs pages      # pages suite only
//
// Config via env:
//   E2E_MODEL      model id to preload + drive        (default: gemma-4-26b-a4b-it-8bit-mlx)
//   E2E_PORT       server port                        (default: 8080)
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

import { startServer } from './lib/server.mjs';
import { launchBrowser, createPageContext } from './lib/browser.mjs';
import { Suite, printSummary } from './lib/harness.mjs';
import { runChatSuite } from './suites/chat.mjs';
import { runPagesSuite } from './suites/pages.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..', '..');

const CONFIG = {
  model: process.env.E2E_MODEL || 'gemma-4-26b-a4b-it-8bit-mlx',
  port: Number(process.env.E2E_PORT || 8080),
  maxTokens: Number(process.env.E2E_MAX_TOKENS || 24),
  baseUrl: process.env.E2E_BASE_URL || null,
};

const which = (process.argv[2] || 'all').toLowerCase();
const runChat = which === 'all' || which === 'chat';
const runPages = which === 'all' || which === 'pages';

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

    if (runChat) {
      const suite = new Suite('chat');
      console.log(`\n${suite.name} suite`);
      await server.clearData();
      const page = await browser.newPage();
      const ctx = createPageContext(page, { base, maxTokens: CONFIG.maxTokens });
      try {
        await runChatSuite({ suite, ctx, config: CONFIG });
      } finally {
        await page.close();
      }
      suites.push(suite);
    }

    if (runPages) {
      const suite = new Suite('pages');
      console.log(`\n${suite.name} suite`);
      await server.clearData();
      const page = await browser.newPage();
      const ctx = createPageContext(page, { base, maxTokens: CONFIG.maxTokens });
      try {
        await runPagesSuite({ suite, ctx, config: CONFIG });
      } finally {
        await page.close();
      }
      suites.push(suite);
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
