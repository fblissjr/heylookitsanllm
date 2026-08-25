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
//   completion adopts the heylook_saved rows in one reconcile pass (no
//   wholesale re-fetch, no frame with the response missing), and the editor
//   offers thinking.
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
import { openDrawer, closeDrawer, clickByText } from './lib/dom.mjs';

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

// 1x1 transparent PNG -- the byte payload the stub media endpoint serves.
const PNG_1PX = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64');
const MEDIA_ID = 'f'.repeat(64); // sha256-shaped, like the server mints

function makeMessages({ unsaved = false, withMedia = false } = {}) {
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
  if (withMedia) {
    // A schema-v7 row: the store externalizes base64 into a blob and returns
    // a url source with the media_id marker -- this is the shape the page
    // actually receives now, and the shape the media checks must exercise.
    msgs.push({
      id: 'mimg', role: 'user', content: 'look at this picture',
      position: msgs.length, thinking: null,
      content_blocks: [
        { type: 'image',
          source: { type: 'url', url: `/v1/conversations/c1/media/${MEDIA_ID}`,
            media_type: 'image/png', media_id: MEDIA_ID } },
        { type: 'text', text: 'look at this picture' },
      ],
    });
  }
  return msgs;
}

// Stateful stub store. `handle` mutates `messages` the way the real store
// would, so post-saga reconciliation reads back a consistent thread.
function makeStubStore({ unsaved = false, caps = [], secondModel = null, withMedia = false } = {}) {
  const messages = makeMessages({ unsaved, withMedia });
  let nextId = messages.length;
  // State "another session" can change under the page: mutate from a check,
  // then resume the page and watch it adopt. Read at RESPOND time.
  const remote = { system_prompt: null, presets: [], updated_at: 't1' };
  const handle = (url, method, postData) => {
    const body = postData ? JSON.parse(postData) : {};
    if (url.endsWith('/v1/models')) {
      const data = [{ id: 'test-model', capabilities: caps }];
      if (secondModel) data.push({ id: secondModel.id, capabilities: secondModel.caps });
      return { data };
    }
    if (url.endsWith('/v1/conversations') && method === 'GET') {
      const convs = [{ id: 'c1', title: 'render suite', model_id: 'test-model', updated_at: remote.updated_at }];
      if (secondModel) convs.push({ id: 'c2', title: 'other model', model_id: secondModel.id });
      return { conversations: convs };
    }
    // A second conversation living on a DIFFERENT model, so the capability-
    // driven chrome can be checked across a conversation switch (not just a
    // model-select switch -- the two take different code paths).
    if (secondModel && url.includes('/v1/conversations/c2')) {
      return { id: 'c2', title: 'other model', model_id: secondModel.id,
        system_prompt: null, applied_preset_id: null, params: {}, messages };
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
        // Single-message delete (v1.73.0): /messages/{id}. The ?after
        // truncation form stays for API users; the UI no longer calls it.
        const one = url.match(/\/messages\/([^/?]+)$/)?.[1];
        if (one) {
          const i = messages.findIndex((m) => m.id === one);
          if (i !== -1) messages.splice(i, 1);
          return { deleted: 1, id: one };
        }
        const after = Number(url.match(/after=(-?\d+)/)?.[1]);
        const keep = messages.filter((m) => !(m.position > after));
        messages.length = 0;
        messages.push(...keep);
        return { deleted: 0, after_position: after };
      }
    }
    if (url.includes('/v1/conversations/c1')) {
      return { id: 'c1', title: 'render suite', model_id: 'test-model',
        system_prompt: remote.system_prompt, applied_preset_id: null, params: {}, messages };
    }
    if (url.endsWith('/v1/presets') && method === 'POST') {
      const row = { id: `p${remote.presets.length + 1}`, name: body.name,
        system_prompt: body.system_prompt ?? null, params: body.params ?? {}, updated_at: 'y' };
      remote.presets.push(row);
      return row;
    }
    if (url.endsWith('/v1/presets')) return { presets: remote.presets };
    if (url.endsWith('/v1/admin/models')) {
      const models = [{ id: 'test-model', loaded: true, provider: 'mlx' }];
      if (secondModel) models.push({ id: secondModel.id, loaded: true, provider: 'mlx' });
      return { models };
    }
    if (url.endsWith('/v1/capabilities')) return { samplers: [], server_version: 'stub' };
    if (url.endsWith('/v1/admin/model-options')) return { fields: [] };
    return {};
  };

  // The generate endpoint's stub half (Phase 2 wire), all three modes: mutate
  // the store the way the server-side saga would -- regenerate and continue
  // TRUNCATE by the anchor and commit together with their row -- and return
  // the Messages-grammar SSE ending in heylook_saved with the stored row(s).
  // Modeling the truncation matters: the client's adoption path merges saved
  // rows over a mirror that visually dropped its tail, and a stub that only
  // ever appends would leave that merge unexercised.
  const genReply = (postData) => {
    let body = {};
    try { body = JSON.parse(postData || '{}'); } catch { /* keep defaults */ }
    const mode = body.mode ?? 'append';
    let row;
    let delta = 'stub reply';
    if (mode === 'regenerate' || mode === 'continue') {
      const anchor = messages.find((m) => m.id === body.message_id);
      const cutoff = mode === 'continue' ? anchor.position : anchor.position - 1;
      const keep = messages.filter((m) => m.position <= cutoff);
      messages.length = 0;
      messages.push(...keep);
      if (mode === 'continue') {
        anchor.content += ' continued tail';
        delta = ' continued tail';
        row = anchor;
      } else {
        delta = 'regenerated reply';
        row = {
          id: `mgen${nextId++}`, role: 'assistant', content: delta,
          thinking: null, content_blocks: null,
          position: (messages[messages.length - 1]?.position ?? -1) + 1,
        };
        messages.push(row);
      }
    } else {
      row = {
        id: `mgen${nextId++}`, role: 'assistant', content: 'stub reply',
        thinking: null, content_blocks: null,
        position: (messages[messages.length - 1]?.position ?? -1) + 1,
      };
      messages.push(row);
    }
    const ev = (type, data) => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
    return [
      ev('message_start', { type: 'message_start', message: { id: row.id, content: [] } }),
      ev('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
      ev('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta } }),
      ev('content_block_stop', { type: 'content_block_stop', index: 0 }),
      ev('message_delta', { type: 'message_delta', delta: { stop_reason: 'stop' }, usage: { input_tokens: 1, output_tokens: 2 } }),
      ev('message_stop', { type: 'message_stop', performance: {} }),
      ev('heylook_saved', { type: 'heylook_saved', conversation_id: 'c1', mode, end_reason: 'complete', messages: [{ ...row }], dropped_media: { images: 0, audio: 0 }, timing: {} }),
    ].join('');
  };

  return { messages, remote, handle, genReply };
}

// A page with the stub API wired in.
// - residencyDelayMs holds back /v1/admin/models so the FIRST render happens
//   with the provider unknown (the whole-list-invalidation window).
// - sseDelayMs holds back the chat/completions response, opening a
//   pre-first-token window to probe the loud guards in.
// - mobile emulates an iPhone (viewport + touch + hover:none/pointer:coarse
//   via CDP -- puppeteer's emulateMediaFeatures whitelist rejects hover).
// - caps is the stub model's capability list. Attachment staging is gated on
//   it, so the drop/paste checks need a vision model and the refusal check
//   needs the text-only default.
async function openChat(browser, base, {
  residencyDelayMs = 0, sseDelayMs = 0, unsaved = false, mobile = false, caps = [],
  secondModel = null, withMedia = false,
} = {}) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  const store = makeStubStore({ unsaved, caps, secondModel, withMedia });
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
    // One failed body fetch on demand (a phone waking with the radio half up).
    if (store.remote.failNextBody && req.method() === 'GET' && /\/v1\/conversations\/c1(\?|$)/.test(url)) {
      store.remote.failNextBody = false;
      return req.respond({ status: 503, contentType: 'application/json', body: '{"detail":"stub outage"}' });
    }
    // The v7 media serve endpoint: bytes, not JSON.
    if (url.includes('/media/')) {
      return req.respond({ status: 200, contentType: 'image/png', body: PNG_1PX });
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

// A GET of the active conversation's body (with or without a query string).
const isConvGet = (r) => r.method === 'GET' && /\/v1\/conversations\/c1(\?|$)/.test(r.url);

// Node-identity probe for "did this re-render REUSE the rows": stamp every
// live row, do the thing, count the stamps that survived. A reused row
// carries the stamp across the re-render, a rebuilt one cannot. This is
// measured instead of the scroll symptom because Chrome's scroll anchoring
// often papers over a full rebuild -- until a forced scroll-to-bottom is the
// first thing computed against the collapsed height, and then the list
// looks like it jumped to the top for no reason.
const stampRows = (page) => page.evaluate(() => {
  const rows = [...document.querySelectorAll('.message')];
  rows.forEach((el, i) => { el.dataset.reuseProbe = String(i); });
  return rows.length;
});
const stampedRows = (page) => page.evaluate(() =>
  [...document.querySelectorAll('.message')].filter((el) => el.dataset.reuseProbe !== undefined).length);

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

// Build a DataTransfer carrying one file, in page context. `items.add` is what
// populates `types` with 'Files' -- the signal both the drop and paste paths
// gate on -- and `files` alongside it.
//
// Drop and paste are event-only affordances: no click can reach them, so these
// synthetic events are the only automated check that they reach addFiles at
// all. Each helper reports whether the event really carried a file, so a check
// cannot pass because the handler bailed early on an empty one.
const mkFile = `const dt = new DataTransfer();
  dt.items.add(new File([new Uint8Array([1, 2, 3, 4])], name, { type }));`;

const dropFile = (page, { name = 'shot.png', type = 'image/png', leaveInstead = false } = {}) =>
  page.evaluate(new Function('name', 'type', 'leave', `
    ${mkFile}
    const el = document.querySelector('.chat__thread');
    const fire = (kind) => el.dispatchEvent(
      new DragEvent(kind, { dataTransfer: dt, bubbles: true, cancelable: true }));
    fire('dragenter');
    fire('dragover');
    const overlaid = el.classList.contains('chat__thread--dragover');
    fire(leave ? 'dragleave' : 'drop');
    return {
      overlaid,
      stillOverlaid: el.classList.contains('chat__thread--dragover'),
      carried: dt.files.length === 1,
    };
  `), name, type, leaveInstead);

// Paste, dispatched where Chrome ACTUALLY targets one. This helper takes a
// real mouse click on a message first, because that is the scenario the
// feature claims to support, and then dispatches at document.activeElement --
// which that click leaves on document.body, OUTSIDE the chat root. Dispatching
// at .chat__messages instead (the first version of this helper) passes with
// the listener mounted on a node no real paste ever reaches; the returned
// targetOutsideRoot is asserted so the check cannot quietly degrade to that.
//
// ClipboardEvent's constructor takes clipboardData in Chrome, but define it
// defensively -- a null clipboardData makes the handler bail, which would pass
// the "nothing staged" checks for entirely the wrong reason.
async function pasteFile(page, { name = 'shot.png', type = 'image/png', withText = false } = {}) {
  await page.click('.chat__messages .message');
  return page.evaluate(new Function('name', 'type', 'withText', `
    ${mkFile}
    if (withText) dt.items.add('some text', 'text/plain');
    const ev = new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true });
    if (!ev.clipboardData) Object.defineProperty(ev, 'clipboardData', { value: dt });
    const target = document.activeElement;
    target.dispatchEvent(ev);
    return {
      carried: Boolean(ev.clipboardData?.files?.length),
      targetTag: target.tagName,
      targetOutsideRoot: !document.querySelector('.chat').contains(target),
      defaultPrevented: ev.defaultPrevented,
    };
  `), name, type, withText);
}

const thumbCount = (page) => page.evaluate(
  () => document.querySelectorAll('.chat__attach .attach-thumb').length);

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
      const body = JSON.parse(post.postData);
      assert(body.content === 'unsaved reply', `expected content string on wire, got ${JSON.stringify(body.content)}`);
      assert(body.content_blocks === undefined, 'content_blocks must not be sent on root wire');
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

    // FIRST send of this boot, deliberately: the blank-frame probe matches
    // any row containing the stub reply text, so a reply minted by an earlier
    // check would satisfy it during the very gap it exists to catch.
    await suite.check('stream completion adopts the saved rows -- no re-fetch, no blank frame', async () => {
      const convGets = () => reqs.filter(isConvGet).length;
      const before = convGets();
      // Watch the swap itself: the streaming node's removal and the saved
      // row's insertion must land in the SAME synchronous reconcile pass.
      // MutationObserver callbacks run after each batch, so if the removal is
      // ever observable with the saved reply absent, there was a frame with
      // the response missing -- the disappear-then-jump this check pins out
      // (the old order removed the node, then awaited a wholesale GET).
      await page.evaluate(() => {
        window.__streamSwap = [];
        const obs = new MutationObserver((records) => {
          for (const rec of records) {
            for (const node of rec.removedNodes) {
              if (node.classList?.contains('message--streaming')) {
                window.__streamSwap.push([...document.querySelectorAll('.message')]
                  .some((m) => m.textContent.includes('stub reply')));
              }
            }
          }
        });
        obs.observe(document.querySelector('.chat__messages-inner'), { childList: true });
      });
      await send(page, 'another message');
      // Adoption, not a wipe: the user turn and the saved reply both stand.
      const onScreen = await page.evaluate(() => ({
        user: [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('another message')),
        reply: [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('stub reply')),
      }));
      assert(onScreen.user, 'the adoption dropped the just-sent message');
      assert(onScreen.reply, 'the saved reply never rendered');
      // The swap asserts come first: "a frame with the response missing" is
      // the user-visible defect; the re-fetch is its cause.
      const swaps = await page.evaluate(() => window.__streamSwap);
      assert(swaps.length >= 1, 'the streaming node was never removed -- no swap happened');
      assert(swaps.every(Boolean),
        'the streaming node was removed while the saved reply was not on screen -- a blank frame');
      assert(convGets() === before,
        'the happy path re-fetched the conversation -- heylook_saved already carried the rows');
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

    // ---- boot 2b: regenerate + continue adoption --------------------------
    // The saved-rows merge over a mirror that dropped its tail -- append mode
    // (boot 2) never exercises it. Fresh boot: no prior sends in the store.
    const gen = await openChat(browser, base);
    const genGets = () => gen.reqs.filter(isConvGet).length;

    await suite.check('regenerate replaces the tail from the saved rows, no re-fetch', async () => {
      const before = genGets();
      // assistant row index 5 = m11: a real tail (m12..m29) exists after it
      const clicked = await clickRowButton(gen.page, '.message--assistant', 'Regenerate', 5);
      assert(clicked, 'no Regenerate button on the anchor row');
      await waitFor(() => gen.page.evaluate(() =>
        !document.querySelector('.message--streaming')
        && [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('regenerated reply'))),
      { message: 'the regenerated row never rendered' });
      const tailGone = await gen.page.evaluate(() =>
        ![...document.querySelectorAll('.message')].some((m) => m.textContent.includes('message 29')));
      assert(tailGone, 'rows after the regenerate anchor are still on screen');
      assert(genGets() === before, 'regenerate completion re-fetched the conversation');
    });

    await suite.check('save & continue merges the continuation onto the anchor row', async () => {
      const before = genGets();
      // the regenerated row is now the thread tail; continue from it
      const opened = await gen.page.evaluate(() => {
        const rows = [...document.querySelectorAll('.message--assistant')];
        const btn = [...rows[rows.length - 1].querySelectorAll('button')]
          .find((b) => b.textContent === 'Edit');
        if (!btn) return false;
        btn.click();
        return true;
      });
      assert(opened, 'could not open the editor on the tail assistant row');
      await settle(gen.page);
      const clicked = await gen.page.evaluate(() => {
        const btn = [...document.querySelectorAll('.message-edit button')]
          .find((b) => b.textContent === 'Save & Continue');
        if (!btn) return false;
        btn.click();
        return true;
      });
      assert(clicked, 'no Save & Continue button (provider unknown?)');
      await waitFor(() => gen.page.evaluate(() =>
        !document.querySelector('.message-edit') && !document.querySelector('.message--streaming')
        && [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('regenerated reply continued tail'))),
      { message: 'the merged continuation never rendered' });
      // Adoption must REPLACE the anchor, not sit a merged copy beside it.
      const carriers = await gen.page.evaluate(() =>
        [...document.querySelectorAll('.message')].filter((m) => m.textContent.includes('regenerated reply')).length);
      assert(carriers === 1, `the anchor row duplicated on adoption (${carriers} rows carry the text)`);
      assert(genGets() === before, 'continue completion re-fetched the conversation');
    });

    await suite.check('Delete removes exactly one message -- the tail survives', async () => {
      // v1.73.0: the old spelling truncated everything after the row. The
      // tail-survives assert is what pins the new meaning of the button.
      const target = await gen.page.evaluate(() => {
        const row = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('message 4'));
        row.scrollIntoView({ block: 'center' });
        return Boolean(row);
      });
      assert(target, 'no message 4 row to delete');
      const click = (label) => gen.page.evaluate((lbl) => {
        const row = [...document.querySelectorAll('.message')]
          .find((m) => m.textContent.includes('message 4'));
        const btn = row && [...row.querySelectorAll('.message__actions button')]
          .find((b) => b.textContent === lbl);
        if (!btn) return false;
        btn.click();
        return true;
      }, label);
      assert(await click('Delete'), 'no Delete button on the row');
      assert(await click('Confirm?'), 'the armed confirm never appeared');
      await waitFor(() => gen.page.evaluate(() =>
        ![...document.querySelectorAll('.message')].some((m) => m.textContent.includes('message 4'))),
      { message: 'the deleted row never left the screen' });
      const survivors = await gen.page.evaluate(() => ({
        before: [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('message 3')),
        after: [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('message 5')),
        tail: [...document.querySelectorAll('.message')].some((m) => m.textContent.includes('continued tail')),
      }));
      assert(survivors.before && survivors.after,
        `the deleted row's neighbors did not survive (${JSON.stringify(survivors)})`);
      assert(survivors.tail, 'Delete took the rest of the thread with it -- the truncation behavior is back');
    });

    await suite.check('no uncaught page errors (regenerate + continue)', () => {
      assert(gen.pageErrors.length === 0, `page errors: ${gen.pageErrors.join(' | ')}`);
    });
    await gen.page.close();

    // ---- boot 2c: schema-v7 media rows render from the blob endpoint ------
    // The store now returns url-source blocks (media by reference); every
    // other boot serves content_blocks: null, so without this the page's
    // media render path is only ever exercised against pre-v7 fixtures.
    const med = await openChat(browser, base, { withMedia: true, caps: ['vision'] });

    await suite.check('a url-source media row renders and its blob request loads', async () => {
      await waitFor(() => med.page.evaluate(() => {
        const img = [...document.querySelectorAll('.message-image')]
          .find((i) => i.src.includes('/media/'));
        // complete + naturalWidth: the request went out AND came back as a
        // decodable image. NB <img> ignores HTTP status and decodes whatever
        // bytes arrive, so this catches an undecodable body (the real
        // server's 404 is JSON -- undecodable), NOT a 404 carrying valid
        // image bytes. Shown red against a garbage body, not a bare 404.
        return Boolean(img && img.complete && img.naturalWidth > 0);
      }), { message: 'the blob-backed image never rendered or never loaded' });
      const row = await med.page.evaluate(() => {
        const img = document.querySelector('.message-image');
        const msg = img.closest('.message');
        return {
          text: msg.textContent.includes('look at this picture'),
          dropNote: Boolean(msg.querySelector('.message-drop-note')),
        };
      });
      assert(row.text, 'the media row lost its text block');
      assert(!row.dropNote, 'a vision-capable model shows a drop disclosure it should not');
    });

    await suite.check('no uncaught page errors (media rows)', () => {
      assert(med.pageErrors.length === 0, `page errors: ${med.pageErrors.join(' | ')}`);
    });
    await med.page.close();

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
      const total = await stampRows(late.page); // see stampRows: identity, not scroll
      await sleep(1600); // let the held response land
      await settle(late.page);
      const kept = await stampedRows(late.page);
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

    // ---- boot 4b: the tab comes back after the store changed under it ----
    // iOS Safari resumes a backgrounded tab with the heap it had; without a
    // resume sync the page would mirror (and write back) hours-old state.
    const resumed = await openChat(browser, base);
    const resumeVia = {
      pageshow: (page) => page.evaluate(() =>
        window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }))),
      visibility: (page) => page.evaluate(() => {
        Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'visible' });
        document.dispatchEvent(new Event('visibilitychange'));
      }),
    };
    const waitForResumeFetch = (reqs, from) => waitFor(() => reqs.slice(from).some(isConvGet),
      { timeout: 5000, interval: 50, message: 'resume did not re-fetch the active conversation' });
    await suite.check('a resume with nothing changed fetches the list, not the conversation body', async () => {
      const { page, reqs } = resumed;
      const from = reqs.length;
      await resumeVia.pageshow(page);
      await waitFor(() => reqs.slice(from).some((r) => r.url.endsWith('/v1/conversations')),
        { timeout: 5000, interval: 50, message: 'resume did not re-fetch the conversation list' });
      await sleep(300);
      assert(!reqs.slice(from).some(isConvGet),
        'resume re-downloaded the conversation body although its updated_at had not moved');
    });
    await suite.check('resume adopts prompt, presets and new rows without rebuilding unchanged ones', async () => {
      const { page, reqs, store } = resumed;
      const total = await stampRows(page);
      // "another session" edits the prompt, saves a preset and appends a row
      store.remote.system_prompt = 'Set on another device.';
      store.remote.updated_at = 't2';
      store.remote.presets = [{ id: 'p-remote', name: 'remote', system_prompt: 'Set on another device.', params: {}, updated_at: 'x' }];
      store.messages.push({ id: 'm-remote', role: 'assistant', content: 'appended elsewhere',
        position: store.messages[store.messages.length - 1].position + 1 });
      const from = reqs.length;
      await resumeVia.pageshow(page);
      await waitForResumeFetch(reqs, from);
      await settle(page);
      const state = await page.evaluate(() => ({
        chip: document.querySelector('.chat__sysprompt-chip')?.textContent,
        presetChip: document.querySelector('.preset-chip:not(.chat__sysprompt-chip)')?.textContent,
        appended: !!document.querySelector('.message--assistant:last-of-type')?.textContent.includes('appended elsewhere'),
      }));
      const kept = await stampedRows(page);
      assert(state.chip !== 'No system prompt', `prompt chip still "${state.chip}" after resume`);
      assert(state.presetChip === 'remote', `preset chip "${state.presetChip}" -- the remote preset was not adopted`);
      assert(state.appended, 'the row appended elsewhere is not rendered after resume');
      assert(kept === total, `resume rebuilt ${total - kept} of ${total} unchanged rows`);
    });
    await suite.check('resume never overwrites a prompt being typed in the drawer', async () => {
      const { page, reqs, store } = resumed;
      await openDrawer(page);
      // focus + keyboard, not a coordinate click: the resume guard keys on
      // focus, which is what is under test
      await page.evaluate(() => { const t = document.querySelector('.drawer--open .sysprompt-input'); t.focus(); t.select(); });
      await page.keyboard.type('local draft, unsaved');
      store.remote.system_prompt = 'Set on another device, again.';
      store.remote.updated_at = 't3';
      const from = reqs.length;
      await resumeVia.visibility(page); // the app-switch spelling this time
      await waitForResumeFetch(reqs, from);
      await settle(page);
      const value = await page.$eval('.drawer--open .sysprompt-input', (el) => el.value);
      assert(value === 'local draft, unsaved', `resume replaced the typed prompt with "${value}"`);
      // The textarea keeping its text is not the whole claim: the drawer
      // rebuild is focus-guarded on its own, so the box can look right while
      // the page's prompt STATE was silently replaced -- and the next preset
      // Save snapshots state, not the box. Save one and read what went out.
      await page.evaluate(() => { const n = document.querySelector('.drawer--open .preset-section .input'); n.focus(); n.select(); });
      await page.keyboard.type('after-resume');
      const beforeSave = reqs.length;
      await clickByText(page, '.drawer--open .preset-row button', 'Save');
      const post = await waitFor(() => reqs.slice(beforeSave).find((r) => r.method === 'POST' && r.url.endsWith('/v1/presets')),
        { timeout: 5000, interval: 50, message: 'preset Save sent no POST' });
      const saved = JSON.parse(post.postData);
      assert(saved.system_prompt === 'local draft, unsaved',
        `preset saved after a resume carried "${saved.system_prompt}" -- the resume overwrote the prompt state under the editor`);
      await closeDrawer(page);
    });
    await suite.check('a resume whose body fetch fails retries on the next resume', async () => {
      const { page, reqs, store } = resumed;
      store.remote.system_prompt = 'Set after an outage.';
      store.remote.updated_at = 't4';
      store.remote.failNextBody = true;
      let from = reqs.length;
      await resumeVia.pageshow(page);
      await waitForResumeFetch(reqs, from); // the one that fails
      await settle(page);
      // Second resume, nothing changed server-side since: the page must not
      // have recorded t4 as adopted, so it fetches the body again.
      from = reqs.length;
      await resumeVia.pageshow(page);
      await waitFor(() => reqs.slice(from).some(isConvGet),
        { timeout: 5000, interval: 50, message: 'after a failed body fetch the next resume did not retry it' });
      await settle(page);
      const chip = await page.$eval('.chat__sysprompt-chip', (el) => el.title);
      assert(chip.includes('Set after an outage.'), `prompt not adopted on the retry (chip title: ${JSON.stringify(chip)})`);
    });
    await suite.check('no uncaught page errors (resume)', () => {
      assert(resumed.pageErrors.length === 0, `page errors: ${resumed.pageErrors.join(' | ')}`);
    });
    await resumed.page.close();

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

    // ---- boot 7: drag/drop + paste attachment staging ---------------------
    // The composer's picker is the only attach path a click can reach; drop
    // and paste are event-only, so this is the only automated check that they
    // reach addFiles at all.
    const vis = await openChat(browser, base, { caps: ['vision'] });

    await suite.check('dragging files over the thread shows the drop overlay', async () => {
      const r = await dropFile(vis.page, { leaveInstead: true });
      assert(r.overlaid, 'no drop overlay while dragging files over the thread');
      assert(!r.stillOverlaid, 'the drop overlay survived dragleave');
    });

    await suite.check('the drop overlay names what the model accepts', async () => {
      const label = await vis.page.evaluate(
        () => document.querySelector('.chat__thread').dataset.dropLabel);
      assert(label === 'Drop images to attach', `drop label reads "${label}"`);
    });

    await suite.check('dropping an image stages it', async () => {
      const r = await dropFile(vis.page);
      assert(r.carried, 'the synthetic drop carried no files -- the check would be vacuous');
      assert(!r.stillOverlaid, 'the drop overlay survived the drop');
      await waitFor(async () => (await thumbCount(vis.page)) === 1,
        { timeout: 3000, message: 'dropped image never reached the attach strip' });
    });

    await suite.check('pasting after clicking in the thread stages the image', async () => {
      const r = await pasteFile(vis.page);
      assert(r.carried, 'the synthetic paste carried no files -- the check would be vacuous');
      assert(r.targetOutsideRoot,
        `paste targeted ${r.targetTag}, INSIDE the chat root -- a real paste does not land there, `
        + 'so this check would pass with the listener on the wrong node');
      await waitFor(async () => (await thumbCount(vis.page)) === 2,
        { timeout: 3000, message: 'pasted image never reached the attach strip' });
    });

    await suite.check('a non-image drop stages nothing', async () => {
      const before = await thumbCount(vis.page);
      await dropFile(vis.page, { name: 'notes.txt', type: 'text/plain' });
      await settle(vis.page);
      const after = await thumbCount(vis.page);
      assert(after === before, `a text/plain drop changed the strip (${before} -> ${after})`);
    });

    await suite.check('a drop carrying nothing attachable says so', async () => {
      const before = await thumbCount(vis.page);
      await dropFile(vis.page, { name: 'paper.pdf', type: 'application/pdf' });
      await waitFor(async () => (await statusText(vis.page)).includes('Nothing attachable'),
        { timeout: 3000, message: 'an unattachable drop vanished without a word' });
      assert((await thumbCount(vis.page)) === before, 'an unattachable drop changed the strip');
    });

    // Deterministic, not a timing gamble: addPendingFiles awaits a FileReader,
    // so a send dispatched in the SAME synchronous block always lands first and
    // always replaces the pending array out from under the in-flight read.
    // Thumb count cannot distinguish the two behaviors (the orphaned push
    // renders nothing either way) -- the status line is the only observable
    // that separates "discarded, and said so" from "vanished".
    await suite.check('a send during the attachment read does not lose it silently', async () => {
      await vis.page.evaluate(() => {
        const dt = new DataTransfer();
        dt.items.add(new File([new Uint8Array(1024)], 'race.png', { type: 'image/png' }));
        const ta = document.querySelector('.chat__composer textarea');
        ta.value = 'text with a racing attachment';
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        document.querySelector('.chat__thread').dispatchEvent(
          new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
        [...document.querySelectorAll('.chat__composer button')]
          .find((b) => b.textContent === 'Send').click();
      });
      await waitFor(async () => (await statusText(vis.page)).includes('discarded'),
        { timeout: 3000, interval: 25,
          message: async () => `the racing attachment vanished without a word (status: "${await statusText(vis.page)}")` });
    });

    await suite.check('no uncaught page errors (attach staging)', () => {
      assert(vis.pageErrors.length === 0, `page errors: ${vis.pageErrors.join(' | ')}`);
    });
    await vis.page.close();

    // Text-only model: refusing at STAGING time is the point -- a silently
    // staged image the send guard rejects later is the behavior this replaced.
    const txt = await openChat(browser, base);
    await suite.check('a drop onto a text-only model refuses loudly and stages nothing', async () => {
      await dropFile(txt.page);
      await waitFor(async () => (await statusText(txt.page)).includes('does not take images'),
        { timeout: 3000, message: 'a drop onto a text-only model said nothing' });
      const n = await thumbCount(txt.page);
      assert(n === 0, `a text-only model staged ${n} image(s)`);
    });

    // A clipboard payload can carry text AND an image. Cancelling the paste on
    // a model that refuses the image would eat the text too, leaving an error
    // and an empty composer.
    await suite.check('a refused paste does not swallow the text half', async () => {
      const r = await pasteFile(txt.page, { withText: true });
      assert(r.carried, 'the synthetic paste carried no files -- the check would be vacuous');
      assert(!r.defaultPrevented,
        'the paste was cancelled on a model that stages nothing -- the text was eaten too');
      assert((await thumbCount(txt.page)) === 0, 'a text-only model staged an image');
    });
    await suite.check('no uncaught page errors (text-only drop)', () => {
      assert(txt.pageErrors.length === 0, `page errors: ${txt.pageErrors.join(' | ')}`);
    });
    await txt.page.close();

    // ---- boot 8: the overlay label must survive a CONVERSATION switch -----
    // Distinct code path from the model-select switch: selectConversation used
    // to refresh the capability-gated chrome BEFORE moving the select, so the
    // label described the conversation being left.
    const two = await openChat(browser, base,
      { caps: ['vision'], secondModel: { id: 'text-model', caps: [] } });

    await suite.check('the drop label follows a conversation switch', async () => {
      const dropLabel = () => two.page.evaluate(
        () => document.querySelector('.chat__thread').dataset.dropLabel);
      await waitFor(async () => (await dropLabel()) === 'Drop images to attach',
        { timeout: 3000, message: async () => `opened on the vision conversation but the label reads "${await dropLabel()}"` });
      const switched = await two.page.evaluate(() => {
        const row = [...document.querySelectorAll('.conv-item')]
          .find((el) => el.textContent.includes('other model'));
        if (!row) return false;
        row.querySelector('.conv-item__title').click();
        return true;
      });
      assert(switched, 'could not find the second conversation row');
      await waitFor(async () => (await dropLabel()) === 'This model takes text only',
        { timeout: 3000, message: async () => `after switching to the text-only conversation the label still reads "${await dropLabel()}"` });
    });

    await suite.check('no uncaught page errors (conversation switch)', () => {
      assert(two.pageErrors.length === 0, `page errors: ${two.pageErrors.join(' | ')}`);
    });
    await two.page.close();

    // ---- boot 9: the "Show special tokens" display pref reaches the wire --
    // DESIGN.md §6. The strip is server-side (before the text is streamed AND
    // before it is persisted), so the pref can only work by being ASKED FOR on
    // the generate body -- a toggle that flips a local flag and sends nothing
    // would look identical in the drawer and do nothing at all.
    const disp = await openChat(browser, base);
    const dispBox = () => disp.page.$('.drawer--open #disp-show_special_tokens');

    await suite.check('the display toggle is offered and defaults to shown', async () => {
      await openDrawer(disp.page);
      const box = await dispBox();
      assert(box, 'no Show special tokens control in the drawer');
      assert(await (await box.getProperty('checked')).jsonValue(),
        'the toggle defaults to OFF -- DESIGN.md §6 says shown by default');
      await closeDrawer(disp.page);
    });

    const sentFlag = async (text) => {
      const before = disp.reqs.length;
      await send(disp.page, text);
      const post = await waitFor(
        () => disp.reqs.slice(before).find((r) => r.method === 'POST' && r.url.includes('/generate')),
        { timeout: 5000, interval: 50, message: 'send made no generate POST' });
      return JSON.parse(post.postData).show_special_tokens;
    };

    await suite.check('generate asks for the specials while the toggle is on', async () => {
      assert((await sentFlag('with specials')) === true,
        'the generate body did not ask for special tokens');
    });

    await suite.check('unchecking it asks for the strip instead', async () => {
      await openDrawer(disp.page);
      await (await dispBox()).click();
      await closeDrawer(disp.page);
      assert((await sentFlag('without specials')) === false,
        'the generate body still asked for special tokens after unchecking');
    });

    await suite.check('a page that ignores the pref does not offer it', async () => {
      // The drawer is an app-shell singleton rendered on every page, so a
      // globally-`wired` pref would appear on explore/jspace too -- which read
      // token ids, not this. Same lie the `wired` gate exists to prevent.
      await disp.page.evaluate(() => { location.hash = '#/explore'; });
      await waitFor(async () => (await disp.page.$('.explore__strip, .page--explore, main')) !== null,
        { timeout: 5000, message: 'explore never mounted' });
      await openDrawer(disp.page, '.drawer-gear');
      assert(!(await dispBox()), 'explore offered a display pref it does not honor');
      await closeDrawer(disp.page);
      await disp.page.evaluate(() => { location.hash = '#/chat'; });
      await waitFor(async () => (await disp.page.$('.chat__thread')) !== null,
        { timeout: 5000, message: 'chat never came back' });
    });

    await suite.check('the pref never rides in the sampler bag', async () => {
      // `overrides` is layered over the conversation's stored params, which is
      // the sampler bag -- a display pref landing there would be persisted as
      // generation state and would reach the model.
      const post = [...disp.reqs].reverse()
        .find((r) => r.method === 'POST' && r.url.includes('/generate'));
      const body = JSON.parse(post.postData);
      assert(!('show_special_tokens' in (body.overrides ?? {})),
        'the display pref was merged into overrides');
    });

    await suite.check('no uncaught page errors (display pref)', () => {
      assert(disp.pageErrors.length === 0, `page errors: ${disp.pageErrors.join(' | ')}`);
    });
    // setDisplayPref PERSISTS, and every boot in this run shares one browser
    // profile and origin -- leaving it unchecked would silently seed every
    // later boot with a non-default pref (code review finding, 2026-08-23).
    await disp.page.evaluate(() => localStorage.removeItem('heylook-v3-display'));
    await disp.page.close();

    // ---- boot 9: raw HTML in model text is SHOWN, never rendered ---------
    // marked passes raw HTML through and DOMPurify then deletes any tag
    // outside its allowlist while KEEPING the tag's content, so a model
    // writing `<d>tag</d>` rendered as "tag" with the tags silently gone --
    // while Copy, which reads the stored text, still showed them. Module-level
    // check (no chat page needed): renderMarkdown is the only text->HTML path.
    const md = await browser.newPage();
    await md.goto(`${base}/v3/`, { waitUntil: 'domcontentloaded' });
    const render = (src) => md.evaluate(async (b, text) => {
      const { renderMarkdown } = await import(`${b}/v3/js/markdown.js`);
      return renderMarkdown(text);
    }, base, src);

    await suite.check('an unknown tag survives rendering as text', async () => {
      const html = await render('Use the <d>tag</d> here.');
      assert(html.includes('&lt;d&gt;tag&lt;/d&gt;'),
        `the tags did not survive rendering: ${JSON.stringify(html)}`);
    });

    await suite.check('an allowlisted tag is shown, not rendered', async () => {
      // The narrow fix (teach DOMPurify to keep <d>) would leave this red:
      // render would still disagree with Copy for every allowlisted tag.
      const html = await render('bold <b>works</b>');
      assert(html.includes('&lt;b&gt;works&lt;/b&gt;'),
        `raw HTML was rendered instead of shown: ${JSON.stringify(html)}`);
    });

    await suite.check('angle-bracketed prose is not mangled into a tag', async () => {
      const html = await render('math a <b and c> d');
      assert(html.includes('math a &lt;b and c&gt; d'),
        `prose was parsed as inline HTML: ${JSON.stringify(html)}`);
    });

    await suite.check('code and autolinks are untouched by the escape', async () => {
      const fenced = await render('```\n<d>fenced</d>\n```');
      assert(fenced.includes('<pre><code>&lt;d&gt;fenced&lt;/d&gt;'),
        `fenced code changed shape: ${JSON.stringify(fenced)}`);
      const span = await render('inline `<d>` code');
      assert(span.includes('<code>&lt;d&gt;</code>'),
        `inline code changed shape: ${JSON.stringify(span)}`);
      const link = await render('<https://example.com>');
      assert(link.includes('<a href="https://example.com">'),
        `an autolink stopped being a link: ${JSON.stringify(link)}`);
    });

    await md.close();
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
