// Render-layer E2E: drives the REAL /v3 chat page (real js/, real css/) against
// a STUBBED /v1 API. No server, no model, no DB -- a few seconds, and it runs
// anywhere Chrome does.
//
//   bun run e2e:render
//
// Deliberately NOT part of `bun run e2e`: that harness spawns heylookllm and
// generates with a real model, and mixing a model-free suite into it would make
// its prerequisites (Metal, models.toml, RAM) look optional. Same entry style
// as run.mjs -- invoke it VIA bun so the script shell finds the real node.
//
// What it guards: the chat message list is RECONCILED, not rebuilt. `.message`
// carries `content-visibility: auto`, so a row's laid-out height lives on the
// NODE; rebuilding the list collapses scrollHeight for the rest of the tick and
// every pixel-based scroll computed against it aims at a list about to grow
// underneath (v1.62.5: send dumped a long thread near the top). Server-side
// telemetry cannot see any of this, and the model-driven suites never scroll a
// long thread, so this is the only automated check for that class.
//
// Config: E2E_CHROME (Chrome binary), E2E_HEADFUL (show the window).

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { launchBrowser } from './lib/browser.mjs';
import { Suite, printSummary, assert } from './lib/harness.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// E2E_V3_ROOT points the suite at a DIFFERENT copy of the frontend. It exists
// so these checks can be shown to fail: copy the tree, restore the pre-fix
// renderMessages into it, and confirm the scroll checks go red. A check that
// has never been seen failing is decoration.
const V3_ROOT = process.env.E2E_V3_ROOT
  || path.join(__dirname, '..', '..', 'apps', 'heylook-frontend-v3');

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml',
};

// --- static file server for the v3 tree ------------------------------------
function serveV3() {
  const server = http.createServer((req, res) => {
    let rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/v3\/?/, '');
    if (rel === '' || rel === '/') rel = 'index.html';
    const file = path.join(V3_ROOT, rel);
    if (!file.startsWith(V3_ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end('not found');
      return;
    }
    res.setHeader('Content-Type', TYPES[path.extname(file)] || 'application/octet-stream');
    res.writeHead(200);
    res.end(fs.readFileSync(file));
  });
  return new Promise((resolve) => {
    server.listen(0, () => resolve({ server, base: `http://127.0.0.1:${server.address().port}` }));
  });
}

// --- stub conversation: long enough to scroll ------------------------------
const LONG = 'lorem ipsum dolor sit amet consectetur adipiscing elit '.repeat(25);

function makeMessages() {
  const msgs = Array.from({ length: 30 }, (_, i) => ({
    id: `m${i}`,
    role: i % 2 ? 'assistant' : 'user',
    content: `message ${i}\n\n${LONG}`,
    position: i,
    thinking: null,
    content_blocks: null,
  }));
  // The shape finishStream pushes when persisting the reply FAILED: on screen,
  // no id. A row like this must render as a message -- `editingId === msg.id`
  // matching null to null once turned it into an open editor whose Save would
  // PUT to /messages/null.
  msgs.push({ id: null, role: 'assistant', content: 'unsaved reply',
    position: msgs.length, thinking: null, content_blocks: null });
  return msgs;
}

function stubRoutes(messages) {
  const conv = {
    id: 'c1', title: 'render suite', model_id: 'test-model',
    system_prompt: null, applied_preset_id: null, params: {}, messages,
  };
  return (url, method) => {
    if (url.endsWith('/v1/models')) return { data: [{ id: 'test-model' }] };
    if (url.endsWith('/v1/conversations') && method === 'GET') {
      return { conversations: [{ id: 'c1', title: 'render suite', model_id: 'test-model' }] };
    }
    if (url.includes('/v1/conversations/c1/messages')) {
      if (method === 'POST') {
        return { id: `m${messages.length}`, role: 'user', content: 'new message',
          position: messages.length, thinking: null, content_blocks: null };
      }
      if (method === 'PUT') return messages[0];
      return {}; // DELETE (truncate)
    }
    if (url.includes('/v1/conversations/c1')) return conv;
    if (url.endsWith('/v1/presets')) return { presets: [] };
    if (url.endsWith('/v1/admin/models')) {
      return { models: [{ id: 'test-model', loaded: true, provider: 'mlx' }] };
    }
    if (url.endsWith('/v1/capabilities')) return { samplers: [], server_version: 'stub' };
    if (url.endsWith('/v1/admin/model-options')) return { fields: [] };
    return {};
  };
}

// A page with the stub API wired in. `residencyDelayMs` holds back
// /v1/admin/models so the FIRST render happens with the model's provider still
// unknown -- the window in which a signature that keys every row on the current
// model invalidates the whole list on the next render.
async function openChat(browser, base, { residencyDelayMs = 0 } = {}) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  const routes = stubRoutes(makeMessages());

  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const url = req.url();
    if (!url.includes('/v1/')) return req.continue();
    if (url.includes('/chat/completions')) {
      const sse = ['data: {"choices":[{"delta":{"content":"stub reply"}}]}', 'data: [DONE]', '']
        .join('\n\n');
      return req.respond({ status: 200, contentType: 'text/event-stream', body: sse });
    }
    const body = JSON.stringify(routes(url, req.method()));
    const send = () => req.respond({ status: 200, contentType: 'application/json', body });
    if (residencyDelayMs && url.endsWith('/v1/admin/models')) {
      setTimeout(send, residencyDelayMs);
      return;
    }
    send();
  });

  await page.goto(`${base}/v3/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.chat__messages .message', { timeout: 15000 });
  return { page, pageErrors };
}

const settle = (page) => page.evaluate(() =>
  new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

const scroll = (page) => page.evaluate(() => {
  const el = document.querySelector('.chat__messages');
  return { top: Math.round(el.scrollTop), height: el.scrollHeight, client: el.clientHeight };
});

const atBottom = (s) => s.height - s.top - s.client < 100;

// Park the view mid-thread, the way a user reading scrollback sits.
async function parkMidThread(page) {
  await settle(page);
  await page.evaluate(() => {
    const el = document.querySelector('.chat__messages');
    el.scrollTop = Math.round(el.scrollHeight * 0.6);
  });
  await settle(page);
  return scroll(page);
}

async function send(page, text) {
  await page.evaluate((t) => {
    const ta = document.querySelector('.chat__composer textarea');
    ta.value = t;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('.chat__composer button')]
      .find((b) => b.textContent === 'Send').click();
  }, text);
  await new Promise((r) => setTimeout(r, 600));
  await settle(page);
}

async function main() {
  const { server, base } = await serveV3();
  const browser = await launchBrowser();
  const suite = new Suite('render');

  try {
    const { page, pageErrors } = await openChat(browser, base);

    await suite.check('an id-less row renders as a message, not an open editor', async () => {
      const row = await page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('unsaved reply'));
        return { found: Boolean(el), isEditor: Boolean(el?.querySelector('.message-edit')) };
      });
      assert(row.found, 'the unsaved row did not render at all');
      assert(!row.isEditor, 'the unsaved row rendered as an open editor');
    });

    await suite.check('send from mid-thread lands at the bottom', async () => {
      const before = await parkMidThread(page);
      assert(!atBottom(before), `setup: expected to be parked mid-thread, got ${JSON.stringify(before)}`);
      await send(page, 'new message');
      const after = await scroll(page);
      assert(atBottom(after), `send left the view at ${after.top} of ${after.height} (bottom is ${after.height - after.client})`);
    });

    await suite.check('opening and cancelling an edit holds the scroll position', async () => {
      await parkMidThread(page);
      const before = await scroll(page);
      await page.evaluate(() => {
        const el = document.querySelector('.chat__messages');
        const row = [...document.querySelectorAll('.message')].find((m) => m.offsetTop > el.scrollTop + 40);
        [...row.querySelectorAll('button')].find((b) => b.textContent === 'Edit').click();
      });
      await settle(page);
      await page.evaluate(() => {
        [...document.querySelectorAll('.message-edit button')]
          .find((b) => b.textContent === 'Cancel')?.click();
      });
      await settle(page);
      const after = await scroll(page);
      // The edited row changes height while the editor is open, so allow a
      // row's worth of drift -- what this rejects is the list jumping.
      const drift = Math.abs(after.top - before.top);
      assert(drift < 600, `edit/cancel moved the view ${drift}px (from ${before.top} to ${after.top})`);
    });

    await suite.check('no uncaught page errors', () => {
      assert(pageErrors.length === 0, `page errors: ${pageErrors.join(' | ')}`);
    });

    await page.close();

    // Second boot: residency lands AFTER first paint. The next render must not
    // invalidate every row (it did when caps+provider were one shared key, and
    // the first send after load jumped to the top all over again).
    const late = await openChat(browser, base, { residencyDelayMs: 1200 });
    await suite.check('a late residency refresh reuses every unchanged row', async () => {
      const before = await parkMidThread(late.page);
      assert(!atBottom(before), `setup: expected to be parked mid-thread, got ${JSON.stringify(before)}`);
      // Stamp the live nodes: a REUSED row carries the stamp across the
      // re-render, a rebuilt one cannot. This measures node identity rather
      // than the scroll symptom, because Chrome's scroll anchoring often
      // papers over a full rebuild -- until a forced scroll-to-bottom is the
      // first thing computed against the collapsed height, and then it looks
      // like the list jumped to the top for no reason.
      const total = await late.page.evaluate(() => {
        const rows = [...document.querySelectorAll('.message')];
        rows.forEach((el, i) => { el.dataset.reuseProbe = String(i); });
        return rows.length;
      });
      await new Promise((r) => setTimeout(r, 1600)); // let the held response land
      await settle(late.page);
      const kept = await late.page.evaluate(() =>
        [...document.querySelectorAll('.message')].filter((el) => el.dataset.reuseProbe !== undefined).length);
      assert(kept === total,
        `residency arriving rebuilt ${total - kept} of ${total} rows -- nothing about those messages changed`);
      const after = await scroll(late.page);
      assert(Math.abs(after.top - before.top) < 50,
        `a background residency refresh moved the view from ${before.top} to ${after.top}`);
    });
    await suite.check('send still lands at the bottom after a late residency refresh', async () => {
      await send(late.page, 'new message');
      const after = await scroll(late.page);
      assert(atBottom(after), `send left the view at ${after.top} of ${after.height} (bottom is ${after.height - after.client})`);
    });
    await suite.check('no uncaught page errors (late residency)', () => {
      assert(late.pageErrors.length === 0, `page errors: ${late.pageErrors.join(' | ')}`);
    });
    await late.page.close();

    // Third boot: an editor opened BEFORE residency lands is missing Save &
    // Continue (it fails closed while the provider is unknown). The residency
    // render is what repairs it -- and it must repair it without eating the
    // text sitting in the box.
    const dur = await openChat(browser, base, { residencyDelayMs: 1500 });
    await suite.check('the residency render repairs an open editor without eating the draft', async () => {
      const opened = await dur.page.evaluate(() => {
        const row = [...document.querySelectorAll('.message--user')][2];
        [...row.querySelectorAll('button')].find((b) => b.textContent === 'Edit').click();
        return true;
      });
      assert(opened, 'could not open an editor');
      await settle(dur.page);
      const beforeBtns = await dur.page.evaluate(() =>
        [...document.querySelectorAll('.message-edit button')].map((b) => b.textContent));
      assert(!beforeBtns.includes('Save & Continue'),
        `setup: Save & Continue should be absent while the provider is unknown, got ${JSON.stringify(beforeBtns)}`);

      await dur.page.evaluate(() => {
        const ta = document.querySelector('.message-edit textarea');
        ta.value = 'a draft nobody may eat';
        ta.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await new Promise((r) => setTimeout(r, 1800)); // residency lands here
      await settle(dur.page);

      const after = await dur.page.evaluate(() => ({
        value: document.querySelector('.message-edit textarea')?.value,
        buttons: [...document.querySelectorAll('.message-edit button')].map((b) => b.textContent),
      }));
      assert(after.value === 'a draft nobody may eat',
        `the residency render ate the draft (textarea now holds ${JSON.stringify(after.value)})`);
      assert(after.buttons.includes('Save & Continue'),
        `the editor was not repaired: buttons are ${JSON.stringify(after.buttons)}`);
    });
    await suite.check('no uncaught page errors (editor repair)', () => {
      assert(dur.pageErrors.length === 0, `page errors: ${dur.pageErrors.join(' | ')}`);
    });
    await dur.page.close();
  } finally {
    await browser.close();
    server.close();
  }

  const failed = printSummary([suite]); // a COUNT, not a boolean
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
