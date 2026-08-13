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
// What it guards:
// - The chat message list is RECONCILED, not rebuilt. `.message` carries
//   `content-visibility: auto`, so a row's laid-out height lives on the NODE;
//   rebuilding the list collapses scrollHeight for the rest of the tick and
//   every pixel-based scroll computed against it aims at a list about to grow
//   underneath (v1.62.5: send dumped a long thread near the top).
// - The Phase 0 chat-reliability contract (plan_chat_orchestration.md):
//   refusals are LOUD (status line, never a silent dead button), the unsaved
//   fallback row carries Retry save/Discard and locks destructive ops, stream
//   completion reconciles against the store, and the editor offers thinking.
// - A phone-emulation pass (iPhone viewport + touch + hover:none via CDP):
//   the new affordances must be reachable without hover (DESIGN.md §7).
//   Emulation is a PROXY -- the hidden-row WebKit investigation still needs a
//   real device (plan Phase 0.5); Chrome checks structure, not WebKit paint.
//
// The stub /v1 is a STATEFUL mini-store (POST appends, PUT merges, DELETE
// truncates, GET returns live rows): the reconcile-on-completion path reads
// back what the sagas wrote, so a canned-response stub would test fiction.
//
// Config: E2E_CHROME (Chrome binary), E2E_HEADFUL (show the window).
// E2E_V3_ROOT points the suite at a DIFFERENT copy of the frontend. It exists
// so these checks can be shown to fail: point it at a pre-fix copy of the tree
// and confirm the guarded checks go red. A check that has never been seen
// failing is decoration.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { launchBrowser } from './lib/browser.mjs';
import { Suite, printSummary, assert, waitFor, sleep } from './lib/harness.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
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

function makeMessages({ unsaved = false } = {}) {
  const msgs = Array.from({ length: 30 }, (_, i) => ({
    id: `m${i}`,
    role: i % 2 ? 'assistant' : 'user',
    content: `message ${i}\n\n${LONG}`,
    position: i,
    thinking: null,
    content_blocks: null,
  }));
  // One assistant row with thinking, for the two-box editor checks.
  msgs[5].thinking = 'original thinking trace for message 5';
  if (unsaved) {
    // The shape finishStream pushes when persisting the reply FAILED: on
    // screen, no id. Served as part of the conversation so the state exists
    // without simulating a failed save. Must render as a message (not an
    // editor -- `editingId === msg.id` once matched null to null) and must
    // carry Retry save/Discard while locking destructive ops.
    msgs.push({ id: null, role: 'assistant', content: 'unsaved reply',
      position: msgs.length, thinking: null, content_blocks: null });
  }
  return msgs;
}

// Stateful stub store. `handle` mutates `messages` the way the real store
// would, so post-saga reconciliation reads back a consistent thread.
function makeStubStore({ unsaved = false } = {}) {
  const messages = makeMessages({ unsaved });
  let nextId = messages.length;
  const handle = (url, method, postData) => {
    const body = postData ? JSON.parse(postData) : {};
    if (url.endsWith('/v1/models')) return { data: [{ id: 'test-model' }] };
    if (url.endsWith('/v1/conversations') && method === 'GET') {
      return { conversations: [{ id: 'c1', title: 'render suite', model_id: 'test-model' }] };
    }
    if (url.includes('/v1/conversations/c1/messages')) {
      if (method === 'POST') {
        const row = {
          id: `m${nextId++}`, role: body.role,
          content: typeof body.content === 'string' ? body.content : '',
          thinking: body.thinking ?? null,
          position: (messages[messages.length - 1]?.position ?? -1) + 1,
          content_blocks: null,
        };
        messages.push(row);
        return row;
      }
      if (method === 'PUT') {
        const msgId = url.match(/\/messages\/([^/?]+)/)?.[1];
        const row = messages.find((m) => m.id === msgId);
        if (!row) return {};
        if ('content' in body) row.content = body.content;
        if ('thinking' in body) row.thinking = body.thinking;
        return { ...row };
      }
      if (method === 'DELETE') {
        const after = Number(url.match(/after=(-?\d+)/)?.[1]);
        const keep = messages.filter((m) => !(m.position > after));
        messages.length = 0;
        messages.push(...keep);
        return { deleted: 0, after_position: after };
      }
    }
    if (url.includes('/v1/conversations/c1')) {
      return { id: 'c1', title: 'render suite', model_id: 'test-model',
        system_prompt: null, applied_preset_id: null, params: {}, messages };
    }
    if (url.endsWith('/v1/presets')) return { presets: [] };
    if (url.endsWith('/v1/admin/models')) {
      return { models: [{ id: 'test-model', loaded: true, provider: 'mlx' }] };
    }
    if (url.endsWith('/v1/capabilities')) return { samplers: [], server_version: 'stub' };
    if (url.endsWith('/v1/admin/model-options')) return { fields: [] };
    return {};
  };

  // The generate endpoint's stub half (Phase 2 wire): append a canned
  // assistant turn to the store exactly as the server would, and return the
  // Messages-grammar SSE ending in heylook_saved with the stored row.
  const genReply = (postData) => {
    let mode = 'append';
    try { mode = JSON.parse(postData || '{}').mode ?? 'append'; } catch { /* keep default */ }
    const row = {
      id: `mgen${nextId++}`, role: 'assistant', content: 'stub reply',
      thinking: null, content_blocks: null,
      position: (messages[messages.length - 1]?.position ?? -1) + 1,
    };
    messages.push(row);
    const ev = (type, data) => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
    return [
      ev('message_start', { type: 'message_start', message: { id: row.id, content: [] } }),
      ev('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
      ev('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'stub reply' } }),
      ev('content_block_stop', { type: 'content_block_stop', index: 0 }),
      ev('message_delta', { type: 'message_delta', delta: { stop_reason: 'stop' }, usage: { input_tokens: 1, output_tokens: 2 } }),
      ev('message_stop', { type: 'message_stop', performance: {} }),
      ev('heylook_saved', { type: 'heylook_saved', conversation_id: 'c1', mode, end_reason: 'complete', messages: [row], dropped_media: { images: 0, audio: 0 }, timing: {} }),
    ].join('');
  };

  return { messages, handle, genReply };
}

// A page with the stub API wired in.
// - residencyDelayMs holds back /v1/admin/models so the FIRST render happens
//   with the provider unknown (the whole-list-invalidation window).
// - sseDelayMs holds back the chat/completions response, opening a
//   pre-first-token window to probe the loud guards in.
// - mobile emulates an iPhone (viewport + touch + hover:none/pointer:coarse
//   via CDP -- puppeteer's emulateMediaFeatures whitelist rejects hover).
async function openChat(browser, base, {
  residencyDelayMs = 0, sseDelayMs = 0, unsaved = false, mobile = false,
} = {}) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  const store = makeStubStore({ unsaved });
  const reqs = [];

  if (mobile) {
    await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true, deviceScaleFactor: 3 });
    const cdp = await page.createCDPSession();
    await cdp.send('Emulation.setEmulatedMedia', {
      features: [
        { name: 'hover', value: 'none' }, { name: 'any-hover', value: 'none' },
        { name: 'pointer', value: 'coarse' }, { name: 'any-pointer', value: 'coarse' },
      ],
    });
  }

  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const url = req.url();
    if (!url.includes('/v1/')) return req.continue();
    reqs.push({ method: req.method(), url, postData: req.postData() });
    // Conversation-scoped generation (Phase 2 wire): typed-event SSE. The
    // store mutation happens at RESPOND time so a delayed stream mutates
    // when the "server" answers, not when the request leaves.
    if (url.includes('/generate')) {
      if (req.method() === 'POST') {
        const respond = () => req.respond({
          status: 200, contentType: 'text/event-stream',
          body: store.genReply(req.postData()),
        });
        if (sseDelayMs) setTimeout(respond, sseDelayMs);
        else respond();
        return;
      }
      // DELETE (Stop) -- nothing tracked in the stub; say "nothing active"
      return req.respond({ status: 404, contentType: 'application/json', body: '{}' });
    }
    const body = JSON.stringify(store.handle(url, req.method(), req.postData()));
    const send = () => req.respond({ status: 200, contentType: 'application/json', body });
    if (residencyDelayMs && url.endsWith('/v1/admin/models')) {
      setTimeout(send, residencyDelayMs);
      return;
    }
    send();
  });

  await page.goto(`${base}/v3/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.chat__messages .message', { timeout: 15000 });
  return { page, pageErrors, reqs, store };
}

const settle = (page) => page.evaluate(() =>
  new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

const scroll = (page) => page.evaluate(() => {
  const el = document.querySelector('.chat__messages');
  return { top: Math.round(el.scrollTop), height: el.scrollHeight, client: el.clientHeight };
});

const atBottom = (s) => s.height - s.top - s.client < 100;

const statusText = (page) => page.evaluate(
  () => document.querySelector('.chat__status')?.textContent ?? '');

// Click the named action button on the row matching `rowFilter` (evaluated in
// the page). Buttons are click()able even when hover-hidden -- visibility is
// asserted separately where it matters (the touch checks).
const clickRowButton = (page, rowSelector, label, rowIndex = 0) =>
  page.evaluate((sel, lbl, idx) => {
    const row = [...document.querySelectorAll(sel)][idx];
    const btn = row && [...row.querySelectorAll('button')].find((b) => b.textContent === lbl);
    if (!btn) return false;
    btn.click();
    return true;
  }, rowSelector, label, rowIndex);

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
    // ---- boot 1: the unsaved fallback row ---------------------------------
    const un = await openChat(browser, base, { unsaved: true });

    await suite.check('an id-less row renders as a message, not an open editor', async () => {
      const row = await un.page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('unsaved reply'));
        return { found: Boolean(el), isEditor: Boolean(el?.querySelector('.message-edit')) };
      });
      assert(row.found, 'the unsaved row did not render at all');
      assert(!row.isEditor, 'the unsaved row rendered as an open editor');
    });

    await suite.check('the unsaved row says so and offers Retry save / Discard', async () => {
      const row = await un.page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('unsaved reply'));
        const note = el?.querySelector('.message-unsaved-note');
        const labels = [...(el?.querySelectorAll('.message__actions button') ?? [])].map((b) => b.textContent);
        return { note: Boolean(note && note.offsetHeight > 0), labels };
      });
      assert(row.note, 'no visible not-saved note on the unsaved row');
      assert(row.labels.includes('Retry save'), `no Retry save button (got ${JSON.stringify(row.labels)})`);
      assert(row.labels.includes('Discard'), `no Discard button (got ${JSON.stringify(row.labels)})`);
      assert(!row.labels.includes('Edit') && !row.labels.includes('Regenerate') && !row.labels.includes('Delete'),
        `position-anchored actions offered on an unsaved row: ${JSON.stringify(row.labels)}`);
    });

    await suite.check('send refuses loudly while an unsaved row exists', async () => {
      const posts = () => un.reqs.filter((r) => r.method === 'POST' && r.url.includes('/messages')).length;
      const before = posts();
      await send(un.page, 'should not go through');
      assert(posts() === before, 'send POSTed a message despite the unsaved row');
      const status = await statusText(un.page);
      assert(/not saved/i.test(status), `no loud refusal in the status line (got ${JSON.stringify(status)})`);
    });

    await suite.check('regenerate refuses loudly while an unsaved row exists', async () => {
      const dels = () => un.reqs.filter((r) => r.method === 'DELETE').length;
      const before = dels();
      const clicked = await clickRowButton(un.page, '.message--assistant', 'Regenerate', 3);
      assert(clicked, 'no Regenerate button found on a saved assistant row');
      await sleep(300);
      assert(dels() === before, 'regenerate truncated the thread despite the unsaved row');
      const status = await statusText(un.page);
      assert(/not saved/i.test(status), `no loud refusal in the status line (got ${JSON.stringify(status)})`);
    });

    await suite.check('Retry save persists the row and restores normal actions', async () => {
      const clicked = await un.page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('unsaved reply'));
        const btn = el && [...el.querySelectorAll('button')].find((b) => b.textContent === 'Retry save');
        if (!btn) return false;
        btn.click();
        return true;
      });
      assert(clicked, 'no Retry save button to click');
      await waitFor(() => un.page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('unsaved reply'));
        return el && !el.querySelector('.message-unsaved-note');
      }), { message: 'the unsaved note never cleared after Retry save' });
      const post = un.reqs.find((r) => r.method === 'POST' && r.url.includes('/messages')
        && r.postData?.includes('unsaved reply'));
      assert(post, 'Retry save never POSTed the row');
      const labels = await un.page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('unsaved reply'));
        return [...el.querySelectorAll('.message__actions button')].map((b) => b.textContent);
      });
      assert(labels.includes('Edit'), `saved row did not regain normal actions (got ${JSON.stringify(labels)})`);
    });

    await suite.check('no uncaught page errors (unsaved boot)', () => {
      assert(un.pageErrors.length === 0, `page errors: ${un.pageErrors.join(' | ')}`);
    });
    await un.page.close();

    // ---- boot 2: scroll pins + thinking editor + reconcile ----------------
    const { page, pageErrors, reqs } = await openChat(browser, base);

    await suite.check('send from mid-thread lands at the bottom', async () => {
      const before = await parkMidThread(page);
      assert(!atBottom(before), `setup: expected to be parked mid-thread, got ${JSON.stringify(before)}`);
      await send(page, 'new message');
      const after = await scroll(page);
      assert(atBottom(after), `send left the view at ${after.top} of ${after.height} (bottom is ${after.height - after.client})`);
    });

    await suite.check('stream completion reconciles against the store', async () => {
      const convGets = () => reqs.filter((r) => r.method === 'GET' && /\/v1\/conversations\/c1(\?|$)/.test(r.url)).length;
      const before = convGets();
      await send(page, 'another message');
      await waitFor(() => convGets() > before,
        { message: 'no conversation re-fetch followed the stream completion' });
      // The reconcile must be an adoption, not a wipe: the rows this saga
      // wrote come back from the (stateful) store and stay on screen.
      const onScreen = await page.evaluate(() =>
        [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('another message')));
      assert(onScreen, 'the reconcile dropped the just-sent message');
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

    await suite.check('the editor offers a thinking box and Save persists both', async () => {
      const opened = await page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('message 5'));
        el.scrollIntoView({ block: 'center' });
        const btn = [...el.querySelectorAll('button')].find((b) => b.textContent === 'Edit');
        if (!btn) return false;
        btn.click();
        return true;
      });
      assert(opened, 'could not open the editor on the thinking row');
      await settle(page);
      const boxes = await page.evaluate(() => ({
        all: document.querySelectorAll('.message-edit textarea').length,
        thinking: document.querySelectorAll('.message-edit__thinking').length,
      }));
      assert(boxes.all === 2 && boxes.thinking === 1,
        `expected a thinking box + a response box, got ${boxes.all} textareas (${boxes.thinking} thinking)`);
      await page.evaluate(() => {
        const think = document.querySelector('.message-edit__thinking');
        think.value = 'edited thinking text';
        think.dispatchEvent(new Event('input', { bubbles: true }));
        const [, response] = document.querySelectorAll('.message-edit textarea');
        response.value = 'edited response text';
        response.dispatchEvent(new Event('input', { bubbles: true }));
        [...document.querySelectorAll('.message-edit button')]
          .find((b) => b.textContent === 'Save').click();
      });
      await waitFor(() => page.evaluate(() =>
        [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('edited response text'))
        && !document.querySelector('.message-edit')),
      { message: 'the edited row never re-rendered with the saved text' });
      const put = reqs.find((r) => r.method === 'PUT' && r.url.includes('/messages/')
        && r.postData?.includes('edited thinking text'));
      assert(put, 'the PUT never carried the edited thinking');
      const thinkingShown = await page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('edited response text'));
        return el?.querySelector('.thinking__body')?.textContent ?? '';
      });
      assert(thinkingShown.includes('edited thinking text'),
        `the rendered thinking block does not show the edit (got ${JSON.stringify(thinkingShown)})`);
    });

    await suite.check('a saved edit leaves the row painted (Chrome pin for the iOS bug)', async () => {
      // Chrome half of plan Phase 0.5: after edit->Save the row must still
      // paint. The reported iOS symptom (row hidden until interaction) is
      // WebKit-side; this pins the structure Chrome can see.
      const painted = await page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('edited response text'));
        if (!el) return { found: false };
        const rect = el.getBoundingClientRect();
        return { found: true, height: rect.height, visible: getComputedStyle(el).visibility };
      });
      assert(painted.found, 'the saved row is gone from the DOM');
      assert(painted.height > 0, 'the saved row has zero height');
      assert(painted.visible !== 'hidden', 'the saved row is visibility:hidden');
    });

    await suite.check('no uncaught page errors', () => {
      assert(pageErrors.length === 0, `page errors: ${pageErrors.join(' | ')}`);
    });
    await page.close();

    // ---- boot 3: loud guard during the pre-first-token window -------------
    const slow = await openChat(browser, base, { sseDelayMs: 1500 });
    await suite.check('actions refuse loudly during the pre-first-token window', async () => {
      await slow.page.evaluate((t) => {
        const ta = document.querySelector('.chat__composer textarea');
        ta.value = t;
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        [...document.querySelectorAll('.chat__composer button')]
          .find((b) => b.textContent === 'Send').click();
      }, 'slow send');
      await sleep(400); // user message persisted, stream waiting on first token
      const clicked = await clickRowButton(slow.page, '.message--user', 'Edit', 1);
      assert(clicked, 'no Edit button found to probe the guard with');
      await sleep(100);
      const status = await statusText(slow.page);
      assert(/streaming|Stop/i.test(status),
        `no loud refusal during the wait (status: ${JSON.stringify(status)})`);
      const editorOpen = await slow.page.evaluate(() => Boolean(document.querySelector('.message-edit')));
      assert(!editorOpen, 'the editor opened mid-stream -- the guard did not hold');
      await waitFor(() => slow.page.evaluate(() =>
        !document.querySelector('.message--streaming')),
      { message: 'the delayed stream never completed' });
    });
    await suite.check('no uncaught page errors (slow stream)', () => {
      assert(slow.pageErrors.length === 0, `page errors: ${slow.pageErrors.join(' | ')}`);
    });
    await slow.page.close();

    // ---- boot 4: residency lands AFTER first paint ------------------------
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

    // ---- boot 5: editor repair across the residency render ----------------
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

    // ---- boot 6: iPhone emulation (touch reachability) --------------------
    const mob = await openChat(browser, base, { unsaved: true, mobile: true });
    await suite.check('touch: actions (incl. Retry save) are visible without hover', async () => {
      const vis = await mob.page.evaluate(() => {
        const rows = [...document.querySelectorAll('.message')];
        const unsavedRow = rows.find((m) => m.textContent.includes('unsaved reply'));
        const normalRow = rows.find((m) => !m.textContent.includes('unsaved reply'));
        const v = (el) => el && getComputedStyle(el.querySelector('.message__actions')).visibility;
        return { unsaved: v(unsavedRow), normal: v(normalRow) };
      });
      assert(vis.normal === 'visible', `hover-gated actions unreachable on touch (visibility: ${vis.normal})`);
      assert(vis.unsaved === 'visible', `Retry save unreachable on touch (visibility: ${vis.unsaved})`);
    });
    await suite.check('touch: the unsaved note is visible at phone width', async () => {
      const note = await mob.page.evaluate(() => {
        const el = document.querySelector('.message-unsaved-note');
        if (!el) return { found: false };
        const rect = el.getBoundingClientRect();
        return { found: true, height: rect.height, fits: rect.right <= window.innerWidth };
      });
      assert(note.found, 'no unsaved note rendered');
      assert(note.height > 0, 'the unsaved note has zero height');
      assert(note.fits, 'the unsaved note overflows the phone viewport');
    });
    await suite.check('touch: the thinking editor fits the phone viewport', async () => {
      const opened = await mob.page.evaluate(() => {
        const el = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('message 5'));
        el.scrollIntoView({ block: 'center' });
        const btn = [...el.querySelectorAll('button')].find((b) => b.textContent === 'Edit');
        if (!btn) return false;
        btn.click();
        return true;
      });
      assert(opened, 'could not open the thinking editor');
      await settle(mob.page);
      const fit = await mob.page.evaluate(() => {
        const boxes = [...document.querySelectorAll('.message-edit textarea')];
        return {
          count: boxes.length,
          overflow: boxes.some((b) => b.getBoundingClientRect().right > window.innerWidth + 1),
          pageOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
        };
      });
      assert(fit.count === 2, `expected 2 textareas on the thinking row, got ${fit.count}`);
      assert(!fit.overflow, 'an editor textarea overflows the phone viewport');
      assert(!fit.pageOverflow, 'the page scrolls horizontally at phone width');
    });
    await suite.check('no uncaught page errors (mobile)', () => {
      assert(mob.pageErrors.length === 0, `page errors: ${mob.pageErrors.join(' | ')}`);
    });
    await mob.page.close();
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
