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
//   `content-visibility: auto` until v1.79.18, so a row's laid-out height
//   lived on the NODE and rebuilding collapsed scrollHeight for the rest of
//   the tick (v1.62.5: send dumped a long thread near the top). That feature
//   is gone, but a rebuild still drops open editors and their drafts.
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
// E2E_V3_ROOT points the suite at a DIFFERENT copy of the frontend -- handy for
// running against a pre-fix tree, or for satisfying yourself that a check you
// doubt can really fail. Available when you want it, not a required step.

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
//
// It also carries ONE dynamic route: a drip-fed /generate stream. puppeteer's
// request interception can only answer a request in a single shot, so every
// other stub here delivers a whole SSE body at once -- which is exactly the
// shape that cannot show a painter repainting over time. The streaming-cost
// checks let that one request through to this server instead, and it writes
// the deltas out with real delays. `drip` is mutated by the check that is
// about to run (checks are sequential).
function serveV3() {
  const drip = { text: '', chunkChars: 8, delayMs: 4, rowId: 'mdrip', position: 2, tailPauseMs: 0, omitSaved: false };
  const server = http.createServer(async (req, res) => {
    if (/\/v1\/conversations\/[^/]+\/generate$/.test(req.url) && req.method === 'POST') {
      await streamDrip(req, res, drip);
      return;
    }
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
    server.listen(0, () => resolve({ server, base: `http://127.0.0.1:${server.address().port}`, drip }));
  });
}

// Write `drip.text` out as Messages-grammar SSE, a chunk at a time, ending in
// the heylook_saved event the client adopts from.
async function streamDrip(req, res, drip) {
  for await (const _ of req) { /* drain the POST body */ }
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
  const ev = (type, data) => res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
  ev('message_start', { type: 'message_start', message: { id: drip.rowId, content: [] } });
  ev('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (let i = 0; i < drip.text.length; i += drip.chunkChars) {
    ev('content_block_delta', {
      type: 'content_block_delta', index: 0,
      delta: { type: 'text_delta', text: drip.text.slice(i, i + drip.chunkChars) },
    });
    await sleep(drip.delayMs);
  }
  // Hold the terminal events back on request. The painter is rate-limited, so
  // the LAST delta is normally still unpainted when the saved row lands and
  // replaces the streaming node -- harmless in the app (the adopted row is
  // complete) but it makes "what was on screen" a moving target for a check.
  if (drip.tailPauseMs) await sleep(drip.tailPauseMs);
  ev('content_block_stop', { type: 'content_block_stop', index: 0 });
  ev('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 1, output_tokens: 2 } });
  ev('message_stop', { type: 'message_stop', performance: {} });
  if (drip.omitSaved) { res.end(); return; }  // transport died before the last word
  ev('heylook_saved', {
    type: 'heylook_saved', conversation_id: 'c1', mode: 'append', end_reason: 'complete',
    messages: [{ id: drip.rowId, role: 'assistant', content: drip.text, thinking: null,
      content_blocks: null, position: drip.position }],
    dropped_media: { images: 0, audio: 0 }, timing: {},
  });
  res.end();
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
function makeStubStore({ unsaved = false, caps = [], secondModel = null, withMedia = false,
  presets = [], appliedPresetId = null } = {}) {
  const messages = makeMessages({ unsaved, withMedia });
  let nextId = messages.length;
  // State "another session" can change under the page: mutate from a check,
  // then resume the page and watch it adopt. Read at RESPOND time.
  // `c2Generating` is the SECOND conversation's flag, separate from
  // `generating` (which is c1's): the interesting state is one conversation
  // generating while the page is subscribed to a stream in the OTHER.
  const remote = { system_prompt: null, presets: [...presets], updated_at: 't1', generating: false,
    c2Generating: false, stopDelayMs: 0, applied_preset_id: appliedPresetId };
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
        system_prompt: null, applied_preset_id: null, params: {},
        generating: remote.c2Generating, messages };
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
        system_prompt: remote.system_prompt, applied_preset_id: remote.applied_preset_id, params: {},
        generating: remote.generating, messages };
    }
    if (url.endsWith('/v1/presets') && method === 'POST') {
      const row = { id: `p${remote.presets.length + 1}`, name: body.name,
        system_prompt: body.system_prompt ?? null, params: body.params ?? {}, updated_at: 'y' };
      remote.presets.push(row);
      return row;
    }
    // The overwrite half of the preset store. Modelling it matters: the guard
    // under test is about a PUT that must NOT be sent, and a stub that only
    // answered POST would let a real overwrite fall through to `{}` and look
    // like nothing happened either way.
    if (url.includes('/v1/presets/') && method === 'PUT') {
      const id = url.match(/\/v1\/presets\/([^/?]+)/)?.[1];
      const row = remote.presets.find((p) => p.id === id);
      if (!row) return {};
      if ('name' in body) row.name = body.name;
      if ('system_prompt' in body) row.system_prompt = body.system_prompt ?? null;
      if ('params' in body) row.params = body.params ?? {};
      return { ...row };
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
      ev('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
      ev('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta } }),
      ev('content_block_stop', { type: 'content_block_stop', index: 0 }),
      ev('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 1, output_tokens: 2 } }),
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
  secondModel = null, withMedia = false, dripGenerate = false, presets = [], appliedPresetId = null,
} = {}) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  const store = makeStubStore({ unsaved, caps, secondModel, withMedia, presets, appliedPresetId });
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
    reqs.push({ method: req.method(), url, postData: req.postData(), at: Date.now() });
    // Conversation-scoped generation (Phase 2 wire): typed-event SSE. The
    // store mutation happens at RESPOND time so a delayed stream mutates
    // when the "server" answers, not when the request leaves.
    if (url.includes('/generate')) {
      // Let the real (drip-feeding) server answer -- see serveV3.
      if (dripGenerate && req.method() === 'POST') return req.continue();
      if (req.method() === 'POST') {
        const respond = () => req.respond({
          status: 200, contentType: 'text/event-stream',
          body: store.genReply(req.postData()),
        });
        if (sseDelayMs) setTimeout(respond, sseDelayMs);
        else respond();
        return;
      }
      // DELETE (Stop) -- nothing tracked in the stub; say "nothing active".
      // `stopDelayMs` holds back the ANSWER, which is the only way to see
      // whether a caller waited for it or merely fired it.
      const answer = () => req.respond({ status: 404, contentType: 'application/json', body: '{}' });
      if (store.remote.stopDelayMs) setTimeout(answer, store.remote.stopDelayMs);
      else answer();
      return;
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

// Record every value the status line takes, for checks about what the user was
// TOLD. A status is a sequence of announcements, not a state to sample: an
// announcement can be correct and still be gone before the next poll.
const watchStatus = (page) => page.evaluate(() => {
  const el = document.querySelector('.chat__status');
  window.__statusLog = [el.textContent];
  window.__statusObs?.disconnect();
  window.__statusObs = new MutationObserver(() => window.__statusLog.push(el.textContent));
  window.__statusObs.observe(el, { childList: true, subtree: true, characterData: true });
});
const statusLog = (page) => page.evaluate(() => window.__statusLog ?? []);

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

// A realistic assistant reply: enough distinct BLOCKS that a rebuild-the-
// subtree painter and an incremental one are told apart by how many nodes a
// single paint removes, and enough shapes (fence with a blank line inside,
// loose list, table, quote) that a bad split boundary shows up as a diff.
const STREAM_DOC = [
  'Here is the short answer, followed by the detail you asked for.',
  '## What is happening',
  'The painter re-parsed the whole message on every frame, which is why the '
    + 'cost grew with the response rather than staying flat.',
  '- the parse is superlinear in length',
  '- the DOM was rebuilt each time',
  '- the scroll write forced a layout right after',
  '### The shape of the fix',
  'Split at a boundary no construct can span, render each segment once, and '
    + 're-render only the tail.',
  '```js\nfunction paint(text) {\n  const b = boundary(text);\n\n  commit(text.slice(0, b));\n  return render(text.slice(b));\n}\n```',
  '> The seam rule still holds: the cut is chosen so nothing spans it.',
  '| case | before | after |\n| --- | --- | --- |\n| long reply | whole doc | tail only |',
  'One more paragraph with `inline code`, **bold** and *emphasis* so the '
    + 'inline lexer has something to do.',
  '1. first numbered item',
  '2. second numbered item',
  '---',
  'A closing paragraph that runs on for a while so the tail has some real '
    + 'length to it before the stream ends and the row is adopted.',
].join('\n\n');

// Send, then wait for the stream to actually END. send()'s fixed pause is
// sized for the single-shot stubs; a drip-fed stream outlives it.
const streaming = (page) => page.evaluate(() => Boolean(document.querySelector('.message--streaming')));

// Send and return as soon as the stream is UNDER WAY, so a check can intervene
// mid-stream (resize the viewport, scroll away) rather than only inspect the end.
async function startSend(page, text = 'go') {
  await page.evaluate((t) => {
    const ta = document.querySelector('.chat__composer textarea');
    ta.value = t;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('.chat__composer button')]
      .find((b) => b.textContent === 'Send').click();
  }, text);
  await waitFor(async () => await streaming(page), { timeout: 10000, message: 'the stream never started' });
}

async function waitStreamEnd(page) {
  await waitFor(async () => !(await streaming(page)), { timeout: 30000, message: 'the stream never finished' });
  await settle(page);
}

async function sendAndWait(page, text = 'go') {
  await startSend(page, text);
  await waitStreamEnd(page);
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

// Drop a real, DECODABLE JPEG of given pixel dimensions. The staging path now
// decodes what it is handed in order to cap its resolution, so the 4-byte
// placeholder above cannot exercise it -- that file fails to decode and is
// passed through, which is the fallback path, not the one under test.
const dropRealImage = (page, { w, h, name = 'photo.jpg' }) =>
  page.evaluate(new Function('w', 'h', 'name', `
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d');
    // Noise, not a flat fill: a flat image compresses to almost nothing, which
    // would make every size assertion below pass for the wrong reason.
    const img = cx.createImageData(w, h);
    let seed = 1;
    for (let i = 0; i < img.data.length; i += 4) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      img.data[i] = seed & 0xff;
      img.data[i + 1] = (seed >> 8) & 0xff;
      img.data[i + 2] = (seed >> 16) & 0xff;
      img.data[i + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    return new Promise((resolve) => {
      c.toBlob((blob) => {
        const dt = new DataTransfer();
        dt.items.add(new File([blob], name, { type: 'image/jpeg' }));
        const el = document.querySelector('.chat__thread');
        const fire = (k) => el.dispatchEvent(new DragEvent(k, { dataTransfer: dt, bubbles: true, cancelable: true }));
        fire('dragenter'); fire('dragover'); fire('drop');
        resolve(blob.size);
      }, 'image/jpeg', 0.92);
    });
  `), w, h, name);

// Send whatever is staged and return the resulting message POST.
async function sendAndCapturePost(ctx, text) {
  const before = ctx.reqs.length;
  await ctx.page.evaluate((t) => {
    const ta = document.querySelector('.chat__composer textarea');
    ta.value = t;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('.chat__composer button')].find((b) => b.textContent === 'Send').click();
  }, text);
  return waitFor(
    () => ctx.reqs.slice(before).find((r) => r.method === 'POST' && /\/messages$/.test(r.url)),
    { timeout: 30000, message: 'the message POST never fired' });
}

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
  const { server, base, drip } = await serveV3();
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
      await clickByText(page, '.drawer--open .preset-row button', 'Save as new');
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

    // ---- boot 4c: Save must not silently overwrite a preset's prompt ------
    // The 2026-08-28 loss: the preset <select> pre-fills the save-as name, so
    // picking a preset to LOOK at it arms Save on that preset -- and Save
    // writes the DOCUMENT's prompt over the stored one with an UPDATE that
    // keeps no history. Apply (which overwrites the recoverable side) was
    // armed; Save (which overwrites the unrecoverable side) was not.
    //
    // These assert on THE WIRE, not on the DOM. The arm changes the button's
    // label, so a DOM assertion would pass whether or not the request was
    // actually held back -- which is the whole claim.
    const guard = await openChat(browser, base);
    {
      const { page, store } = guard;
      store.remote.presets.push({
        id: 'p-owned', name: 'owned', system_prompt: 'THE PRESET PROMPT', params: {}, updated_at: 'z',
      });
      store.remote.presets.push({
        id: 'p-bare', name: 'bare', system_prompt: null, params: {}, updated_at: 'z',
      });
      // The iterate-loop fixture: the loop check applies it, edits, and saves
      // BACK onto it, so its stored prompt is deliberately rewritten mid-block.
      store.remote.presets.push({
        id: 'p-pristine', name: 'pristine', system_prompt: 'PRISTINE PRESET PROMPT', params: {}, updated_at: 'z',
      });
      // The one fixture NOTHING in this block writes to, so a check can state
      // its stored prompt as a constant and stay true under reordering.
      store.remote.presets.push({
        id: 'p-stable', name: 'stable', system_prompt: 'STABLE PRESET PROMPT', params: {}, updated_at: 'z',
      });
      // Likewise promptless and likewise never written to -- "bare" acquires a
      // prompt from the arming checks, so it cannot answer "is a promptless
      // preset labelled?" by the time that check runs.
      store.remote.presets.push({
        id: 'p-samplers', name: 'samplers-only', system_prompt: null, params: {}, updated_at: 'z',
      });
      // A second arm-worthy target, so a pending arm can be shown NOT to
      // carry across a change of selection.
      store.remote.presets.push({
        id: 'p-other', name: 'other', system_prompt: 'SOME OTHER PRESET PROMPT', params: {}, updated_at: 'z',
      });
      await openDrawer(page);
      await settle(page);
    }
    // ONE spelling of "find the preset controls". Three had grown -- a select
    // query, a row filtered by its select, a row filtered by its input -- and
    // the two row variants would silently match the WRONG row after a markup
    // change rather than throw.
    const PRESET = {
      select: '.drawer--open .preset-section select',
      // By TITLE, not by position. `nth-of-type` silently RE-AIMS at whatever
      // button ends up in that slot, so reordering the row -- or inserting one
      // -- would leave every check below green while pressing the wrong
      // control, and Update is the destructive one. The titles are stable
      // anchors: armedConfirm relabels `textContent` when armed and never
      // touches `title`, which is also why a text lookup is not the answer.
      applyBtn: '.drawer--open .preset-section .preset-row button[title^="Copy this preset here"]',
      updateBtn: '.drawer--open .preset-section .preset-row button[title^="Overwrite this preset"]',
      nameInput: '.drawer--open .preset-section .preset-row:has(input.input) input.input',
      saveNewBtn: '.drawer--open .preset-section .preset-row:has(input.input) button',
      preview: '.drawer--open .preset-preview__body',
    };
    // Matches data-name, NOT the option text: a preset carrying no prompt is
    // labelled "<name> — settings only", so a text match would miss exactly
    // the promptless fixture these checks lean on.
    const selectPreset = async (name) => {
      await guard.page.evaluate((sel, n) => {
        const el = document.querySelector(sel);
        const opt = [...el.options].find((o) => o.dataset.name === n);
        if (!opt) throw new Error(`no preset option named ${n}`);
        el.value = opt.value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }, PRESET.select, name);
      await settle(guard.page);
    };
    const typeDocPrompt = async (text) => {
      await guard.page.evaluate((t) => {
        const box = document.querySelector('.drawer--open .sysprompt-input');
        if (!box) throw new Error('no system-prompt box in the open drawer');
        box.value = t;
        box.dispatchEvent(new Event('input', { bubbles: true }));
        box.dispatchEvent(new Event('change', { bubbles: true }));
      }, text);
      await settle(guard.page);
    };
    // By SELECTOR, not by text: an armed button relabels itself ("Overwrite
    // prompt?"), so a text lookup would miss exactly the second click these
    // checks need to make.
    const clickApply = async () => {
      await guard.page.click(PRESET.applyBtn);
      await settle(guard.page);
    };
    const saveLabel = () => guard.page.$eval(PRESET.updateBtn, (el) => el.textContent);
    // Hand-type a save-as name, the way a user renaming the target would --
    // this is the one control that can aim Save somewhere the select is not
    // pointing.
    // Drive a sampler control the way the drawer's own listeners see it.
    const setSampler = async (key, value) => {
      await guard.page.evaluate((k, v) => {
        const el = document.getElementById(`set-${k}`);
        if (!el) throw new Error(`no sampler control for ${k}`);
        el.value = v;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }, key, value);
      await settle(guard.page);
    };
    const typePresetName = async (name) => {
      await guard.page.evaluate((sel, n) => {
        const box = document.querySelector(sel);
        box.value = n;
        box.dispatchEvent(new Event('input', { bubbles: true }));
      }, PRESET.nameInput, name);
      await settle(guard.page);
    };
    // Counts BOTH preset writes, not just PUT. A check asserting "zero
    // requests" must not be blind to the shape where the save landed as a
    // CREATE -- a name absent from the local list makes save() POST, and a
    // PUT-only filter would report that unconfirmed write as nothing at all.
    const clickSave = async () => {
      const from = guard.reqs.length;
      await guard.page.click(PRESET.updateBtn);
      await settle(guard.page);
      await sleep(250); // save() awaits a refresh before it would write
      return guard.reqs.slice(from).filter((r) =>
        (r.method === 'PUT' && r.url.includes('/v1/presets/'))
        || (r.method === 'POST' && r.url.endsWith('/v1/presets')));
    };

    await suite.check('the first Save over a preset that owns a prompt sends nothing', async () => {
      await typeDocPrompt('the document prompt, not the preset\'s');
      await selectPreset('owned');
      const puts = await clickSave();
      assert(puts.length === 0,
        `Save overwrote preset "owned" on the first click -- sent ${puts.length} PUT(s): ${JSON.stringify(puts.map((p) => p.postData))}`);
      const stored = guard.store.remote.presets.find((p) => p.id === 'p-owned').system_prompt;
      assert(stored === 'THE PRESET PROMPT', `the stored prompt changed to ${JSON.stringify(stored)}`);
    });

    await suite.check('the armed second Save goes through', async () => {
      const puts = await clickSave();
      assert(puts.length === 1, `a confirmed Save sent ${puts.length} PUT(s), expected 1`);
      assert(JSON.parse(puts[0].postData).system_prompt === 'the document prompt, not the preset\'s',
        'the confirmed Save did not carry the document prompt');
    });

    await suite.check('a Save that would blank a preset\'s prompt also arms', async () => {
      // The quieter half of the same loss: an empty document prompt writes
      // NULL over a stored one, and an override-box preset with no prompt is
      // inert -- so the preset does not vanish from the list, it just stops
      // doing anything. That reads as "my preset disappeared".
      await selectPreset('bare'); // park the selection off the target first
      await typeDocPrompt('');
      await selectPreset('owned');
      const puts = await clickSave();
      assert(puts.length === 0,
        'Save blanked a preset\'s stored prompt on the first click');
    });

    await suite.check('saving a preset that owns no prompt does not arm', async () => {
      // Nothing to lose, so the guard must stay out of the way -- a confirm
      // that fires when nothing is at stake only trains click-through.
      await typeDocPrompt('anything');
      await selectPreset('bare');
      const puts = await clickSave();
      assert(puts.length === 1,
        `Save over a promptless preset asked for confirmation (sent ${puts.length} PUTs, expected 1)`);
    });

    await suite.check('the preset section shows the preset\'s own prompt, not the document\'s', async () => {
      await typeDocPrompt('DOCUMENT TEXT');
      await selectPreset('stable');
      const body = await guard.page.$eval(PRESET.preview, (el) => el.textContent);
      assert(body.includes('STABLE PRESET PROMPT'),
        `the preview shows ${JSON.stringify(body.slice(0, 80))} -- not the selected preset's own prompt`);
      assert(!body.includes('DOCUMENT TEXT'),
        'the preview is showing the document prompt, the exact confusion this fixes');
    });

    // ---- the iterate loop stays one click (v1.79.21) ---------------------
    // Guarding Save cost the loop this bar exists for: apply a preset, edit
    // the prompt, save it back. That armed every time, and an arm the user
    // meets dozens of times a day is an arm they stop reading -- which is
    // what makes the one protecting them worthless. The exemption is narrow:
    // the document must already be RUNNING the preset being saved onto.
    await suite.check('saving back onto the preset the document runs does not arm', async () => {
      await selectPreset('pristine');
      await clickApply(); await clickApply(); // armed: it replaces a differing prompt
      await settle(guard.page);
      const applied = await guard.page.$eval('.drawer--open .sysprompt-input', (el) => el.value);
      assert(applied === 'PRISTINE PRESET PROMPT', `Apply did not carry the prompt (got ${JSON.stringify(applied.slice(0, 40))})`);
      await typeDocPrompt('PRISTINE PRESET PROMPT -- plus my edit');
      const puts = await clickSave();
      assert(puts.length === 1,
        `the apply -> edit -> save-back loop asked for confirmation (sent ${puts.length} PUTs, expected 1)`);
      assert(JSON.parse(puts[0].postData).system_prompt === 'PRISTINE PRESET PROMPT -- plus my edit',
        'the loop save did not carry the edited prompt');
    });

    await suite.check('a refusal does not eat the name you typed', async () => {
      // The refusal used to force a drawer rebuild, which replaces the section
      // -- so the name was gone (and on a phone the keyboard closed) at the
      // exact moment the user was told to go do something else with it.
      await sleep(8200);
      await typePresetName('owned');
      await guard.page.click(PRESET.saveNewBtn);
      await settle(guard.page);
      await sleep(250);
      const kept = await guard.page.$eval(PRESET.nameInput, (el) => el.value);
      assert(kept === 'owned', `the refusal cleared the name box (now ${JSON.stringify(kept)})`);
      const status = await guard.page.$eval('.chat__status', (el) => el.textContent);
      assert(/already exists/i.test(status), `no refusal was shown: ${JSON.stringify(status)}`);
    });

    await suite.check('a name already in use is refused, never overwritten', async () => {
      // The accident class, removed rather than guarded: there is no typing
      // path to an overwrite any more. Save as new creates or refuses; only
      // Update (beside the select, under the preview) overwrites, and only
      // ever the preset it is showing.
      await sleep(8200);
      await typeDocPrompt('a prompt that is not what pristine stores');
      await selectPreset('other');          // Update is aimed at "other"
      await typePresetName('pristine');     // ...and the name box names another
      const from = guard.reqs.length;
      await guard.page.click(PRESET.saveNewBtn);
      await settle(guard.page);
      await sleep(250);
      const writes = guard.reqs.slice(from).filter((r) =>
        (r.method === 'PUT' && r.url.includes('/v1/presets/'))
        || (r.method === 'POST' && r.url.endsWith('/v1/presets')));
      assert(writes.length === 0,
        `"Save as new" wrote to a name already in use (${writes.length} writes: `
        + `${JSON.stringify(writes.map((w) => [w.method, w.postData]))})`);
      const status = await guard.page.$eval('.chat__status', (el) => el.textContent);
      assert(/already exists/i.test(status) && /update/i.test(status),
        `the refusal did not name the other action: ${JSON.stringify(status)}`);
    });

    await suite.check('but saving onto a preset the document does NOT run still arms', async () => {
      // The shape that caused the loss: running one preset, saving onto
      // another. The exemption
      // must not generalise into "any Save is fine once something is stamped".
      await selectPreset('owned');
      const puts = await clickSave();
      assert(puts.length === 0,
        `Save onto an unrelated preset went through unconfirmed (${puts.length} PUTs)`);
    });

    await suite.check('and blanking the running preset still arms', async () => {
      // Clearing the box is not editing it, so the running-preset exemption
      // does not cover it -- a NULL write leaves the preset inert.
      await selectPreset('pristine');
      await typeDocPrompt('');
      const puts = await clickSave();
      assert(puts.length === 0,
        `Save blanked the running preset's prompt unconfirmed (${puts.length} PUTs)`);
    });

    await suite.check('changing the target cancels a pending confirm', async () => {
      // An arm is a promise about ONE preset. Arming on one and then picking
      // another left the next click confirming something never previewed --
      // the same accident in two steps. Both targets here own a prompt and
      // neither is the one the document runs, so both must arm on their own.
      //
      // Start from a KNOWN disarmed state rather than inheriting one: a
      // previous check leaves the button armed, and letting the feature under
      // test be what clears it makes the first assertion fail with a message
      // about arming when the real regression is in cancelling.
      await sleep(8200); // armedConfirm's own timeout
      assert((await saveLabel()) === 'Update', `Update was still armed at the start of the check: ${await saveLabel()}`);
      await typeDocPrompt('a prompt that differs from both');
      await selectPreset('owned');
      let writes = await clickSave();
      assert(writes.length === 0, 'the first Save on "owned" should have armed');
      assert((await saveLabel()) !== 'Update', 'the first Update did not visibly arm');
      await selectPreset('other');
      assert((await saveLabel()) === 'Update', 'picking another preset left the button visibly armed');
      writes = await clickSave();
      assert(writes.length === 0,
        `the arm carried across the selection change and overwrote "other" unconfirmed (${writes.length} writes)`);
    });

    await suite.check('editing the prompt under an armed Save voids the confirm', async () => {
      // The hole a consumer cannot close by wiring its own controls: Save's
      // PAYLOAD is the document prompt, edited in a different drawer section
      // this bar gets no events from. Arm on "replace it with this text", then
      // clear the box -- the second click used to fire whatever the button was
      // now aimed at, blanking the preset with no confirm and skipping the
      // blanking guard entirely.
      await sleep(8200);
      await typeDocPrompt('text that will be cleared in a moment');
      await selectPreset('owned');
      let writes = await clickSave();
      assert(writes.length === 0, 'the first Save on "owned" should have armed');
      await typeDocPrompt(''); // the payload moves under the armed button
      writes = await clickSave();
      assert(writes.length === 0,
        `the armed Save fired against a payload the user never previewed and blanked "owned" (${writes.length} writes)`);
      const stored = guard.store.remote.presets.find((p) => p.id === 'p-owned').system_prompt;
      assert(stored, 'the stored prompt was blanked without a confirmation');
    });

    await suite.check('a Save that changes no prompt at all does not arm', async () => {
      // Ordinary traffic: nudge a sampler, Save it back, prompt untouched.
      // Nothing is at stake, so an arm here is pure click-through training --
      // the failure this whole release is about.
      await sleep(8200);
      await typeDocPrompt('STABLE PRESET PROMPT'); // exactly what "stable" stores
      await selectPreset('stable');
      const writes = await clickSave();
      assert(writes.length === 1,
        `Save asked for confirmation although it would not change the stored prompt (${writes.length} writes)`);
    });

    // ---- the drawer says what it is doing (v1.79.25) ---------------------
    await suite.check('the sampler panel names what it applies to', async () => {
      // It is the active conversation's stored params, it is also the seed for
      // the next new one, and selecting a conversation silently replaces every
      // value in it. Nothing said which.
      //
      // The no-document wording ("Defaults for new conversations.") is NOT
      // checked here and is not reachable in this harness: the stub always
      // serves a conversation and chat auto-selects it. Left uncovered rather
      // than faked.
      const note = await guard.page.$eval('.drawer--open .settings-panel .settings-note',
        (el) => el.textContent);
      assert(/applies to this conversation/i.test(note),
        `the sampling panel's scope note reads ${JSON.stringify(note)}`);
    });

    await suite.check('the reset button says it clears overrides', async () => {
      // It sets every value to null, handing each back to the backend cascade,
      // and writes that onto the open conversation -- "Reset to defaults" read
      // as something global.
      const label = await guard.page.$eval('.drawer--open .settings-panel button',
        (el) => el.textContent);
      assert(label === 'Clear all overrides', `the reset button reads ${JSON.stringify(label)}`);
    });

    await suite.check('a preset carrying no prompt says so where you pick it', async () => {
      const opts = await guard.page.$eval(PRESET.select, (el) =>
        [...el.options].map((o) => [o.dataset.name ?? null, o.textContent]));
      const bare = opts.find(([n]) => n === 'samplers-only');
      const owned = opts.find(([n]) => n === 'owned');
      assert(bare && /settings only/i.test(bare[1]),
        `the promptless preset renders as ${JSON.stringify(bare?.[1])}`);
      assert(owned && owned[1] === 'owned',
        `a preset that DOES carry a prompt was decorated: ${JSON.stringify(owned?.[1])}`);
    });

    await suite.check('the drift line names which half drifted', async () => {
      const drift = () => guard.page.$eval('.drawer--open .preset-drift', (el) => el.textContent);
      // Start from an exact match, then break one half at a time.
      await selectPreset('stable');
      await typeDocPrompt('STABLE PRESET PROMPT');
      await setSampler('temperature', '');
      assert(/matches current settings/i.test(await drift()),
        `expected a match, got ${JSON.stringify(await drift())}`);

      await typeDocPrompt('STABLE PRESET PROMPT, edited');
      let line = await drift();
      assert(/^Prompt differs/.test(line), `prompt-only edit reads ${JSON.stringify(line)}`);

      await typeDocPrompt('STABLE PRESET PROMPT');
      await setSampler('temperature', '0.7');
      line = await drift();
      assert(/^Settings differ/.test(line), `sampler-only edit reads ${JSON.stringify(line)}`);

      await typeDocPrompt('STABLE PRESET PROMPT, edited again');
      line = await drift();
      assert(/^Prompt and settings differ/.test(line), `both edited reads ${JSON.stringify(line)}`);
      await setSampler('temperature', ''); // leave the panel as we found it
    });

    await suite.check('no uncaught page errors (preset guard)', () => {
      assert(guard.pageErrors.length === 0, `page errors: ${guard.pageErrors.join(' | ')}`);
    });
    await guard.page.close();

    // ---- boot 4d: the select reports what the document is running ---------
    // With no explicit pick the select showed "Presets…" even on a
    // conversation carrying an applied_preset_id -- so the one place that
    // could have answered "which preset is this?" said nothing, and finding
    // out meant clicking through the dropdown. That is the click that lands
    // on the Save button with a preset name already in the box.
    const stamped = await openChat(browser, base, {
      presets: [{ id: 'p-run', name: 'running-one', system_prompt: 'STAMPED PRESET PROMPT', params: {}, updated_at: 'z' }],
      appliedPresetId: 'p-run',
    });
    await suite.check('the select opens on the preset the document is running', async () => {
      await openDrawer(stamped.page);
      await settle(stamped.page);
      const shown = await stamped.page.evaluate(() => {
        const sel = document.querySelector('.drawer--open .preset-section select');
        return sel.options[sel.selectedIndex]?.textContent ?? null;
      });
      assert(shown === 'running-one',
        `the select shows ${JSON.stringify(shown)} on a conversation stamped with "running-one"`);
    });
    await suite.check('and its preview names that preset\'s own prompt', async () => {
      const body = await stamped.page.$eval('.drawer--open .preset-preview__body', (el) => el.textContent);
      assert(body.includes('STAMPED PRESET PROMPT'),
        `the preview shows ${JSON.stringify(body.slice(0, 60))}`);
    });
    await suite.check('no uncaught page errors (stamped select)', () => {
      assert(stamped.pageErrors.length === 0, `page errors: ${stamped.pageErrors.join(' | ')}`);
    });
    await stamped.page.close();

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
      // Save & Continue is OFFERED but disabled while the provider is unknown
      // (v1.79.27) -- fail-closed as before, but a button that silently comes
      // and goes with the model reads as arbitrary. The claim that matters is
      // that it cannot FIRE: the truncate it performs is destructive and would
      // land before the 400.
      const beforeBtns = await dur.page.evaluate(() =>
        [...document.querySelectorAll('.message-edit button')]
          .map((b) => [b.textContent, b.disabled, b.title]));
      const cont = beforeBtns.find(([label]) => label === 'Save & Continue');
      assert(cont, `setup: no Save & Continue button at all, got ${JSON.stringify(beforeBtns)}`);
      assert(cont[1] === true,
        'Save & Continue is enabled while the provider is unknown -- it can fire a destructive truncate on a guess');
      assert(cont[2] && cont[2].length > 0,
        'the disabled Save & Continue gives no reason, which is what made it read as arbitrary');

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
    await suite.check('touch: the preset row fits the phone, armed or not', async () => {
      // Update joined a row that already held a select, Apply and Del, and an
      // armed button relabels itself to "Overwrite prompt?". At 390px a
      // nowrap row would either overflow the drawer or crush the select --
      // the one control that names what Update is aimed at.
      await openDrawer(mob.page);
      await settle(mob.page);
      const measure = () => mob.page.evaluate(() => {
        const row = document.querySelector('.drawer--open .preset-section .preset-row');
        const sel = row.querySelector('select');
        return {
          overflow: row.scrollWidth - row.clientWidth,
          selectWidth: sel.getBoundingClientRect().width,
        };
      });
      let m = await measure();
      assert(m.overflow <= 1, `the preset row overflows by ${m.overflow}px at phone width`);
      assert(m.selectWidth >= 100, `the preset select is crushed to ${Math.round(m.selectWidth)}px`);

      // Arm the longest label the row can show and measure again.
      await mob.page.evaluate(() => {
        const row = document.querySelector('.drawer--open .preset-section .preset-row');
        [...row.querySelectorAll('button')].forEach((b) => { b.textContent = 'Overwrite prompt?'; });
      });
      await settle(mob.page);
      m = await measure();
      assert(m.overflow <= 1, `the armed preset row overflows by ${m.overflow}px at phone width`);
      assert(m.selectWidth >= 100, `the armed preset row crushes the select to ${Math.round(m.selectWidth)}px`);
      await closeDrawer(mob.page);
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
      await watchStatus(vis.page);
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
      // Observe the sequence rather than polling for a value. "Was the user
      // told?" is a question about what was WRITTEN, and the send that raced
      // this attachment overwrites the line within milliseconds of it
      // appearing -- polling for it passes or fails on timing alone. (It was
      // passing on a window a few ms wide until an unrelated change to the
      // staging path moved it.) The observer is installed before the drop, so
      // nothing can slip past between writes.
      await waitFor(async () => (await statusLog(vis.page)).some((t) => t.includes('discarded')),
        { timeout: 3000, interval: 25,
          message: async () => `the racing attachment vanished without a word (status line only ever said: ${JSON.stringify(await statusLog(vis.page))})` });
    });

    await suite.check('an oversized image is capped before it reaches the wire', async () => {
      // Measured before this fix, through this same path: eight 3MB camera-roll
      // photos left as a single 32MB POST. Nothing between the file picker and
      // the request reduced anything, and base64 added 33% on top.
      const sourceBytes = await dropRealImage(vis.page, { w: 4032, h: 3024 });
      await waitFor(async () => (await thumbCount(vis.page)) === 1,
        { timeout: 20000, message: 'the image never staged' });
      const post = await sendAndCapturePost(vis, 'describe this');
      const block = JSON.parse(post.postData).content.find((b) => b.type === 'image');
      assert(block, `no image block on the wire: ${post.postData.slice(0, 200)}`);
      // Base64 is 1.33x, so an uncapped image can never come in under source.
      assert(post.postData.length < sourceBytes,
        `the POST body (${post.postData.length}B) is no smaller than the source image (${sourceBytes}B) -- nothing capped it`);
      // And it is the PIXELS that shrank, not merely the JPEG re-compressing.
      const dims = await vis.page.evaluate((b64, type) => new Promise((resolve) => {
        const i = new Image();
        i.onload = () => resolve({ w: i.naturalWidth, h: i.naturalHeight });
        i.onerror = () => resolve(null);
        i.src = `data:${type};base64,${b64}`;
      }), block.source.data, block.source.media_type);
      assert(dims && Math.max(dims.w, dims.h) <= 2048,
        `the image reached the wire at ${dims ? `${dims.w}x${dims.h}` : 'an unreadable size'} -- the resolution cap did not apply`);
    });

    await suite.check('an image already within the cap is passed through, not re-encoded', async () => {
      // A lossy round-trip that saves nothing is worse than doing nothing.
      const sourceBytes = await dropRealImage(vis.page, { w: 800, h: 600, name: 'small.jpg' });
      await waitFor(async () => (await thumbCount(vis.page)) === 1,
        { timeout: 20000, message: 'the small image never staged' });
      const post = await sendAndCapturePost(vis, 'and this one');
      const block = JSON.parse(post.postData).content.find((b) => b.type === 'image');
      // Untouched bytes reach the wire as base64: 4/3 of the source, plus padding.
      const ratio = block.source.data.length / sourceBytes;
      assert(ratio > 1.3 && ratio < 1.4,
        `a within-cap image was re-encoded (base64 is ${ratio.toFixed(2)}x the source, expected ~1.33x)`);
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

    // ---- boot 8b: losing a capability is disclosed, not gated -------------
    const think = await openChat(browser, base,
      { caps: ['thinking'], secondModel: { id: 'text-model', caps: [] } });
    await suite.check('losing thinking is announced even with no media to warn about', async () => {
      // The note used to be gated on `lines.length` -- it only ever rode an
      // existing media-drop warning, so in a TEXT-ONLY conversation switching
      // from a thinking model to a plain one said nothing at all and the
      // toggle just vanished. It discloses rather than gates: no Cancel /
      // Switch-anyway prompt, because losing a capability destroys nothing.
      const { page } = think;
      await openDrawer(page);
      await page.evaluate(() => {
        const box = document.querySelector('.drawer--open #set-enable_thinking');
        if (!box) throw new Error('no thinking control on a thinking-capable model');
        box.checked = true;
        box.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await closeDrawer(page);
      await settle(page);
      await page.evaluate(() => {
        const sel = [...document.querySelectorAll('select')].find((s) => s.title === 'Model');
        sel.value = 'text-model';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await settle(page);
      const status = await page.$eval('.chat__status', (el) => el.textContent);
      assert(/thinking is unavailable/i.test(status),
        `switching away from a thinking model said ${JSON.stringify(status)}`);
      assert(!(await page.$('.chat__switch-warning')),
        'losing thinking raised a blocking warning -- it should disclose, not gate');
    });
    await suite.check('no uncaught page errors (thinking loss)', () => {
      assert(think.pageErrors.length === 0, `page errors: ${think.pageErrors.join(' | ')}`);
    });
    await think.page.close();

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


    // ---- boot 10: the streaming painter is incremental --------------------
    // A streaming message used to be re-parsed WHOLE and assigned to innerHTML
    // on every animation frame. marked's parse is superlinear in document
    // length, so the per-frame cost grew faster than the response did and a
    // long generation saturated the main thread -- which on a phone is heat and
    // battery, invisible to every check that only renders a FINISHED document.
    // The two properties that make the fix real are guarded here, and the
    // boundary rules they rest on are proven by the growth check further down.

    await suite.check('a growing message re-renders the tail, never the whole document', async () => {
      // A rebuild-the-subtree painter removes EVERY child on every paint; an
      // incremental one removes only the tail. The largest single removal is
      // what separates them, and it does not depend on timing.
      drip.text = STREAM_DOC;
      drip.chunkChars = 6;
      drip.delayMs = 3;
      const st = await openChat(browser, base, { dripGenerate: true });
      await st.page.evaluate(() => {
        window.__paint = { records: 0, maxRemoved: 0, maxChildren: 0 };
        const obs = new MutationObserver((records) => {
          const el = document.querySelector('.message--streaming .message-content');
          if (!el) return;
          const p = window.__paint;
          for (const r of records) {
            if (r.type !== 'childList' || !r.target.closest?.('.message--streaming')) continue;
            p.records++;
            // Only judge removals once the message HAS a prefix to preserve.
            if (p.maxChildren > 8) p.maxRemoved = Math.max(p.maxRemoved, r.removedNodes.length);
          }
          p.maxChildren = Math.max(p.maxChildren, el.childNodes.length);
        });
        obs.observe(document.querySelector('.chat__messages'), { childList: true, subtree: true });
        window.__paintObs = obs;
      });
      await sendAndWait(st.page);
      const p = await st.page.evaluate(() => { window.__paintObs.disconnect(); return window.__paint; });
      await st.page.close();
      assert(p.maxChildren > 8,
        `setup: the streamed message only reached ${p.maxChildren} child nodes -- too short to tell`);
      // The old painter's removals equal the whole child count (20+ here);
      // the incremental one only ever drops the trailing block or two.
      assert(p.maxRemoved <= 6,
        `a single paint removed ${p.maxRemoved} of ${p.maxChildren} nodes -- the message is being rebuilt, not extended`);
    });

    await suite.check('switching MODELS mid-run says the answer survives, never "Stopped"', async () => {
      // A client-side abort ends our SUBSCRIPTION; the run detaches, finishes
      // and commits the whole reply. The end used to report that as
      // "Stopped." -- a false statement about the server, and the opposite of
      // what the reader needs before walking away.
      //
      // The stub does not model a detached run: its GET returns
      // generating:false the instant our fetch dies, whereas a real server
      // keeps the run alive and says generating:true (verified per engine in
      // tests/smoke/). Since the client now believes the SERVER over its own
      // proxies -- correctly -- the stub has to report what a real one does or
      // this check tests the wrong world.
      drip.text = STREAM_DOC;
      drip.chunkChars = 4;
      drip.delayMs = 25;
      const away = await openChat(browser, base, {
        dripGenerate: true, secondModel: { id: 'text-model', caps: [] },
      });
      away.store.remote.generating = true;
      await startSend(away.page, 'go');
      await waitFor(() => streaming(away.page), { timeout: 8000, message: 'the stream never started' });
      await away.page.evaluate(() => {
        const sel = [...document.querySelectorAll('select')].find((s) => s.title === 'Model');
        sel.value = 'text-model';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await waitStreamEnd(away.page);
      const line = await away.page.$eval('.chat__status', (el) => el.textContent);
      assert(!/stopped/i.test(line), `switching models mid-run reported: ${JSON.stringify(line)}`);
      assert(/keeps generating/i.test(line) && /text-model/.test(line),
        `switching models mid-run said ${JSON.stringify(line)}`);
      assert(away.pageErrors.length === 0, `page errors: ${away.pageErrors.join(' | ')}`);
      await away.page.close();
    });

    await suite.check('switching CONVERSATIONS mid-run names the one being left', async () => {
      // Different path: finishGenerate early-returns once activeId has moved,
      // so selectConversation has to say this itself. Its own boot, because
      // the check above leaves the composer in its remote-generating state.
      drip.text = STREAM_DOC;
      drip.chunkChars = 4;
      drip.delayMs = 25;
      const swap = await openChat(browser, base, {
        dripGenerate: true, secondModel: { id: 'text-model', caps: [] },
      });
      await startSend(swap.page, 'go');
      await waitFor(() => streaming(swap.page), { timeout: 8000, message: 'the stream never started' });
      await swap.page.evaluate(() => {
        const row = [...document.querySelectorAll('.conv-item__title')]
          .find((el) => el.textContent === 'other model');
        if (!row) throw new Error('the second conversation is not in the sidebar');
        row.click();
      });
      await waitStreamEnd(swap.page);
      const line = await swap.page.$eval('.chat__status', (el) => el.textContent);
      assert(!/stopped/i.test(line), `switching conversations mid-run reported: ${JSON.stringify(line)}`);
      assert(/keeps generating/i.test(line) && /render suite/.test(line),
        `switching conversations mid-run said ${JSON.stringify(line)}`);
      assert(swap.pageErrors.length === 0, `page errors: ${swap.pageErrors.join(' | ')}`);
      await swap.page.close();
    });

    await suite.check('switching INTO a generating conversation offers Stop, and keeps it', async () => {
      // Two mechanisms, one symptom: the composer says Send for a run that is
      // running. Written as ONE property because the user cannot tell them
      // apart and neither should the check.
      //
      //   1. `setRemoteGenerating` returned early on ANY live `s.stream`. But
      //      the stream we are leaving belongs to the conversation we LEFT --
      //      it owns that conversation's button, not this one's. The guard's
      //      own reason ("a LOCAL stream owns the button") is about our own run
      //      in the CURRENT document, so the test is targetConvId, not mere
      //      existence.
      //   2. `releaseStream` then restored the button to the literal 'Send'.
      //      That is the half with no race in it: the old stream's unwind is
      //      GUARANTEED to land after the switch, so fixing (1) alone converts
      //      a narrow timing bug into a certain one.
      //
      // Hence "and keeps it": the second assertion is the load-bearing one.
      drip.text = STREAM_DOC;
      drip.chunkChars = 4;
      drip.delayMs = 25;
      const sw = await openChat(browser, base, {
        dripGenerate: true, secondModel: { id: 'text-model', caps: [] },
      });
      sw.store.remote.c2Generating = true;   // the OTHER conversation is running
      await startSend(sw.page, 'go');
      await waitFor(() => streaming(sw.page), { timeout: 8000, message: 'the stream never started' });
      await sw.page.evaluate(() => {
        const row = [...document.querySelectorAll('.conv-item__title')]
          .find((el) => el.textContent === 'other model');
        if (!row) throw new Error('the second conversation is not in the sidebar');
        row.click();
      });
      // TEXT AND TITLE. The word alone cannot tell the two Stops apart, and a
      // LOCAL stream's Stop is on screen for the whole abandoned-stream
      // window -- so a text-only assertion passes on the button belonging to
      // the conversation being left, which is the vacuous reading of exactly
      // this check. Only the remote wording means "there is a way to stop the
      // run that is actually going".
      const composer = () => sw.page.evaluate(
        () => {
          const b = [...document.querySelectorAll('.chat__composer button')]
            .find((x) => x.textContent === 'Stop' || x.textContent === 'Send');
          return `${b?.textContent}: ${b?.title || '(no title)'}`;
        });
      const REMOTE_STOP = 'Stop: Stop the run finishing on the server for this conversation';
      await waitFor(async () => (await composer()) === REMOTE_STOP,
        { timeout: 8000,
          message: async () => `the composer read "${await composer()}" for a conversation the `
            + 'server says is generating' });
      // Past the abandoned stream's unwind: that is when releaseStream runs.
      await waitStreamEnd(sw.page);
      await settle(sw.page);
      const after = await composer();
      const errs = sw.pageErrors.slice();
      await sw.page.close();
      assert(after === REMOTE_STOP,
        `the composer fell back to "${after}" when the abandoned stream released, `
        + 'leaving no way to stop the run that is actually going');
      assert(errs.length === 0, `page errors: ${errs.join(' | ')}`);
    });

    await suite.check('deleting a streaming conversation waits for the stop to be ANSWERED', async () => {
      // The delete path fired `stopGenerate` and immediately asserted, via
      // ABANDON.DELETE, that the run had "genuinely ended server-side" -- on a
      // request whose outcome it had not waited for and would never look at.
      // Two things wrong with that, and the ordering is the checkable one: the
      // conversation DELETE raced the stop it depends on, so the 409-then-retry
      // branch below it was covering for an ordering this function could just
      // have had.
      //
      // Only a DELAYED answer can show the difference. Dispatch order was
      // already stop-then-delete; what changed is whether the second waits for
      // the first's response.
      drip.text = STREAM_DOC;
      drip.chunkChars = 4;
      drip.delayMs = 25;
      const del = await openChat(browser, base, { dripGenerate: true });
      del.store.remote.stopDelayMs = 400;
      await startSend(del.page, 'go');
      await waitFor(() => streaming(del.page), { timeout: 8000, message: 'the stream never started' });
      await del.page.evaluate(() => {
        const b = document.querySelector('.conv-item--active .conv-item__delete');
        if (!b) throw new Error('no delete control on the active conversation');
        b.click();   // armedConfirm: first click arms,
        b.click();   // second confirms. Both synchronous.
      });
      const convDel = await waitFor(
        () => del.reqs.find((r) => r.method === 'DELETE' && /\/v1\/conversations\/c1$/.test(r.url)),
        { timeout: 8000, message: 'the conversation was never deleted' });
      const stop = del.reqs.find((r) => r.method === 'DELETE' && /\/generate$/.test(r.url));
      const errs = del.pageErrors.slice();
      await del.page.close();
      assert(stop, 'no stop was sent for the streaming conversation being deleted');
      assert(convDel.at - stop.at >= 300,
        `the conversation DELETE went out ${convDel.at - stop.at}ms after the stop, whose answer `
        + 'was held for 400ms -- it did not wait for the outcome it then asserted');
      assert(errs.length === 0, `page errors: ${errs.join(' | ')}`);
    });

    await suite.check('an idle server is never described as still generating', async () => {
      // The mirror of the check above, and the reason the refactor exists:
      // when the transport dies but the server reports the conversation IDLE,
      // the run is over and "the reply keeps generating" would be a fresh
      // false claim in the opposite direction. The client derives this from
      // the server's own answer rather than from `aborted`.
      drip.text = STREAM_DOC;
      drip.chunkChars = 4;
      drip.delayMs = 25;
      const idle = await openChat(browser, base, {
        dripGenerate: true, secondModel: { id: 'text-model', caps: [] },
      });
      idle.store.remote.generating = false; // the server says: nothing running
      await startSend(idle.page, 'go');
      await waitFor(() => streaming(idle.page), { timeout: 8000, message: 'the stream never started' });
      await idle.page.evaluate(() => {
        const sel = [...document.querySelectorAll('select')].find((s) => s.title === 'Model');
        sel.value = 'text-model';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await waitStreamEnd(idle.page);
      const line = await idle.page.$eval('.chat__status', (el) => el.textContent);
      assert(!/keeps generating/i.test(line) && !/still generating/i.test(line),
        `an idle conversation was described as generating: ${JSON.stringify(line)}`);
      assert(idle.pageErrors.length === 0, `page errors: ${idle.pageErrors.join(' | ')}`);
      await idle.page.close();
    });

    await suite.check('a Stop that lands after the run finished does not claim "Stopped"', async () => {
      // The stub answers the stop DELETE with 404 -- exactly the real shape
      // when the server has already finished and released its claim while
      // heylook_saved is still in flight. sawEvent is true by then, so
      // stopStream must NOT abort locally AND must not mark the run stopped:
      // the stream completes carrying the whole answer.
      //
      // Setting userStopped at the TOP of stopStream re-opened the regression
      // first fixed 2026-08-13 from the other side -- a COMPLETE response
      // reported as "Stopped -- partial response saved.", with the token
      // line lost. Two independent reviews found it; this is the check that
      // keeps it shut.
      drip.text = STREAM_DOC;
      drip.chunkChars = 4;
      drip.delayMs = 20;
      const late = await openChat(browser, base, { dripGenerate: true });
      await startSend(late.page, 'go');
      await waitFor(() => streaming(late.page), { timeout: 8000, message: 'the stream never started' });
      await late.page.evaluate(() => {
        [...document.querySelectorAll('.chat__composer button')]
          .find((b) => b.textContent === 'Stop').click();
      });
      await waitStreamEnd(late.page);
      const line = await late.page.$eval('.chat__status', (el) => el.textContent);
      assert(!/stopped/i.test(line),
        `a run that finished completely reported ${JSON.stringify(line)} because Stop had been pressed`);
      assert(late.pageErrors.length === 0, `page errors: ${late.pageErrors.join(' | ')}`);
      await late.page.close();
    });

    await suite.check('the streaming repaint rate is bounded, not one per frame', async () => {
      // Deltas arrive far faster than anyone reads. A per-frame painter paints
      // once per delta (they are slower than a frame); a rate-limited one
      // paints on the order of duration/PAINT_INTERVAL_MS.
      drip.text = STREAM_DOC;
      drip.chunkChars = 6;
      drip.delayMs = 3;
      const st = await openChat(browser, base, { dripGenerate: true });
      await st.page.evaluate(() => {
        window.__rate = { paints: 0, deltas: 0, start: performance.now(), end: 0 };
        const obs = new MutationObserver(() => {
          const el = document.querySelector('.message--streaming .message-content');
          if (!el) return;
          window.__rate.paints++;
          window.__rate.end = performance.now();
        });
        obs.observe(document.querySelector('.chat__messages'), { childList: true, subtree: true, characterData: true });
        window.__rateObs = obs;
      });
      await sendAndWait(st.page);
      const r = await st.page.evaluate(() => { window.__rateObs.disconnect(); return window.__rate; });
      await st.page.close();
      const deltas = Math.ceil(STREAM_DOC.length / 6);
      const ms = r.end - r.start;
      // The ceiling is the CONFIGURED rate (PAINT_INTERVAL_MS = 66) with slack
      // for observer batching and the structural renders that bracket a stream.
      // The point is the ORDER: bounded by elapsed time, not by how many
      // deltas arrived. Measured either side of the fix on this same document
      // -- rate-limited ~10, one-per-frame 45, ceiling ~22.
      const ceiling = Math.max(15, (ms / 66) * 2.5);
      assert(r.paints < ceiling,
        `${r.paints} repaints for ${deltas} deltas over ${Math.round(ms)}ms (ceiling ${Math.round(ceiling)}) -- the painter is not rate-limited`);
    });

    await suite.check('the streamed message matches a whole-document render', async () => {
      // Speed is worthless if the seam shows. What was on screen at the last
      // paint must equal what a single render of the same text produces.
      drip.text = STREAM_DOC;
      drip.chunkChars = 5;
      drip.delayMs = 2;
      drip.tailPauseMs = 400; // let the last rate-limited paint land
      // A REALISTIC tail position, overriding the stub default of 2. That
      // default makes adoptSavedRows truncate the mirror from 32 rows to 3 at
      // stream end, and the resulting collapse is a harness artifact that
      // masks the bug this check is for: measured, the real strand is caught
      // ~17% of the time at position 2 and ~79% at a realistic append.
      drip.position = 31;
      const st = await openChat(browser, base, { dripGenerate: true });
      await st.page.evaluate(() => {
        window.__last = '';
        const obs = new MutationObserver(() => {
          const el = document.querySelector('.message--streaming .message-content');
          if (el) window.__last = el.innerHTML;
        });
        obs.observe(document.querySelector('.chat__messages'), { childList: true, subtree: true, characterData: true });
        window.__lastObs = obs;
      });
      await sendAndWait(st.page);
      const got = await st.page.evaluate(() => { window.__lastObs.disconnect(); return window.__last; });
      const want = await st.page.evaluate(async (b, text) => {
        const { renderMarkdown } = await import(`${b}/v3/js/markdown.js`);
        const el = document.createElement('div');
        el.innerHTML = renderMarkdown(text);
        return el.innerHTML;
      }, base, STREAM_DOC);
      await st.page.close();
      drip.tailPauseMs = 0;
      if (got !== want) {
        // Report WHERE they diverge -- the heads are identical for hundreds of
        // characters, so a head-of-string dump names nothing.
        let i = 0;
        while (i < got.length && i < want.length && got[i] === want[i]) i++;
        assert(false,
          `the streamed render differs from a whole-document render at offset ${i} `
          + `(streamed ${got.length} chars, whole ${want.length}):\n`
          + `  got:  ${JSON.stringify(got.slice(Math.max(0, i - 80), i + 160))}\n`
          + `  want: ${JSON.stringify(want.slice(Math.max(0, i - 80), i + 160))}`);
      }
    });

    await suite.check('the view still follows the tail after a mid-stream viewport change', async () => {
      // A resize changes clientHeight without the reader ever scrolling. This
      // check exists because the first version of the fix cached "at the tail"
      // as a flag written by scroll events -- and pinning coalesces those to a
      // handful across a whole generation, so the flag went stale exactly here.
      // On a phone the keyboard resizes the viewport constantly, so this is the
      // common case, not an exotic one.
      drip.text = STREAM_DOC;
      drip.chunkChars = 6;
      drip.delayMs = 7;
      drip.tailPauseMs = 400;
      const st = await openChat(browser, base, { dripGenerate: true });
      await st.page.setViewport({ width: 390, height: 844 });
      await startSend(st.page);
      await sleep(250);
      // Shrink, then grow, WITHOUT ever scrolling.
      await st.page.setViewport({ width: 390, height: 600 });
      await sleep(200);
      await st.page.setViewport({ width: 390, height: 844 });
      await waitStreamEnd(st.page);
      // Condition, not a snapshot: the final rate-limited paint and WebKit's
      // own post-resize scroll restore both land after the stream ends, and a
      // single read right at that boundary made this check flaky. What is
      // being asserted is where the view SETTLES.
      const gap = await waitFor(async () => {
        const at = await scroll(st.page);
        const g = at.height - at.top - at.client;
        return g < 120 ? { g } : null;
      }, { timeout: 5000, interval: 100,
           message: async () => {
             const at = await scroll(st.page);
             return `the view stranded ${at.height - at.top - at.client}px above the tail after a mid-stream resize`;
           } }).then((r) => r.g);
      await st.page.close();
      drip.tailPauseMs = 0;
      drip.position = 2;
      assert(gap < 120, `settled ${gap}px above the tail`);
    });

    await suite.check('a reader who scrolls up mid-stream is left alone', async () => {
      // The other half of the same flag: it must also go FALSE and stay false.
      drip.text = STREAM_DOC;
      drip.chunkChars = 6;
      drip.delayMs = 7;
      const st = await openChat(browser, base, { dripGenerate: true });
      await startSend(st.page);
      await sleep(300);
      const parked = await st.page.evaluate(() => {
        const el = document.querySelector('.chat__messages');
        el.scrollTop = 0;
        return el.scrollTop;
      });
      await waitStreamEnd(st.page);
      const end = await scroll(st.page);
      await st.page.close();
      assert(end.top < parked + 200,
        `the view was yanked back to the tail (${parked} -> ${end.top}) while the reader was scrolled up`);
    });

    await suite.check('a run started elsewhere shows as generating, with a way to stop it', async () => {
      // A generation now outlives the response that started it, so a
      // conversation can be generating with no local stream object: you left
      // mid-generation and came back. Without a surface that says so, a
      // runaway has no off switch from the page that shows it running.
      const gen = await openChat(browser, base);
      gen.store.remote.generating = true;
      gen.store.remote.updated_at = 't2';           // make resume refetch the body
      await gen.page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
      await waitFor(async () => (await gen.page.evaluate(
        () => [...document.querySelectorAll('.chat__composer button')]
          .find((b) => b.textContent === 'Stop' || b.textContent === 'Send')?.textContent)) === 'Stop',
        { timeout: 8000, message: 'the composer never offered Stop for a run started elsewhere' });
      const status = await statusText(gen.page);
      assert(/still generating/i.test(status),
        `no disclosure that the server is still generating (status: ${JSON.stringify(status)})`);

      // Sending is refused LOUDLY rather than racing the running generation.
      //
      // This has to reach the real send path. The first version set the
      // textarea via evaluate (which never focuses it) and then pressed Enter,
      // so the key went to document.body and the assertion below was vacuously
      // true whether or not the guard existed. Clicking is no good either --
      // the button reads Stop here by design. So: focus the field for real and
      // press Enter, which is the keystroke a user would actually use.
      const beforePosts = gen.reqs.filter((r) => r.method === 'POST').length;
      await gen.page.focus('.chat__composer textarea');
      await gen.page.type('.chat__composer textarea', 'me too');
      await gen.page.keyboard.press('Enter');
      await settle(gen.page);
      assert(gen.reqs.filter((r) => r.method === 'POST').length === beforePosts,
        'a send went out while the conversation was still generating');
      const refusal = await statusText(gen.page);
      assert(/still generating/i.test(refusal),
        `the refusal was silent (status: ${JSON.stringify(refusal)})`);

      // Stop reaches it as a DELETE, the same contract the local Stop uses.
      await gen.page.evaluate(() => [...document.querySelectorAll('.chat__composer button')]
        .find((b) => b.textContent === 'Stop').click());
      const del = await waitFor(
        () => gen.reqs.find((r) => r.method === 'DELETE' && /\/generate$/.test(r.url)),
        { timeout: 8000, message: 'Stop never sent a DELETE for the remote run' });
      assert(del, 'no DELETE observed');
      await gen.page.close();
    });

    await suite.check('a stream that dies mid-run does not claim the answer is complete', async () => {
      // Two things used to look identical when heylook_saved never arrived:
      // the server FINISHED and its commit raced our resync, or the server is
      // STILL GENERATING. Since a run outlives the response that started it,
      // the second is now the common one -- and announcing "Recovered what the
      // server saved" there states that a partial answer is the whole one.
      drip.text = STREAM_DOC;
      // Slow enough that the streaming node is OBSERVABLE: at 40 chars/1ms the
      // whole stream finished inside one poll interval and startSend timed out
      // waiting for a node that had already come and gone.
      drip.chunkChars = 8;
      drip.delayMs = 8;
      drip.omitSaved = true;
      const dead = await openChat(browser, base, { dripGenerate: true });
      await watchStatus(dead.page);
      await startSend(dead.page);
      // Flipped AFTER the send: with it set beforehand the composer already
      // reads Stop on mount and there is nothing to send -- the setup would
      // defeat the check rather than arrange it.
      dead.store.remote.generating = true;   // the run continues server-side
      // The recovery backoff is 1s, 2.5s, 6.25s -- outlast the first two.
      await sleep(5000);
      const seen = await statusLog(dead.page);
      const btn = await dead.page.evaluate(
        () => [...document.querySelectorAll('.chat__composer button')]
          .find((b) => b.textContent === 'Stop' || b.textContent === 'Send')?.textContent);
      await dead.page.close();
      drip.omitSaved = false;
      assert(!seen.some((t) => /Recovered what the server saved/.test(t)),
        `claimed recovery while the server was still generating: ${JSON.stringify(seen)}`);
      assert(seen.some((t) => /[Ss]till generating/.test(t)),
        `never said the server was still generating: ${JSON.stringify(seen)}`);
      assert(btn === 'Stop', `the composer offered "${btn}" for a run that is still going`);
    });

    await suite.check('no uncaught page errors (streaming paint)', async () => {
      const st = await openChat(browser, base, { dripGenerate: true });
      drip.text = STREAM_DOC;
      await sendAndWait(st.page);
      const errs = st.pageErrors.slice();
      await st.page.close();
      assert(errs.length === 0, `page errors: ${errs.join(' | ')}`);
    });


    // ---- the boundary rules that make incremental rendering equivalent ----
    // Every speed property above rests on ONE claim: the cut MarkdownStream
    // chooses is a place no markdown construct can span. That claim is not
    // provable by example -- it is a property, so check it as one, over
    // generated documents grown one chunk at a time, exactly the way a stream
    // arrives. Chunk-boundary invariance is the same technique the backend's
    // parser tests use (TestParserInvariants) and for the same reason: both
    // 2026-07-23 parser bugs were invisible to example-based tests.
    await suite.check('an incrementally grown document renders identically to a whole one', async () => {
      const bad = await md.evaluate(async (b) => {
        const { MarkdownStream } = await import(`${b}/v3/js/markdown-stream.js`);
        const { renderMarkdown } = await import(`${b}/v3/js/markdown.js`);

        let seed = 20260826;
        const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
        const pick = (a) => a[Math.floor(rnd() * a.length)];
        const W = ('the model returns a value when parsing tokens across layers and caches '
          + 'inputs for later reuse in generation which is why latency matters here').split(' ');
        const words = (n) => Array.from({ length: n }, () => pick(W)).join(' ');

        // Every block shape that could plausibly straddle a split.
        const BLOCKS = [
          () => words(25) + '.',
          () => '## ' + words(4),
          () => '### ' + words(3),
          () => '- ' + words(5) + '\n- ' + words(4) + '\n- ' + words(6),
          () => '1. ' + words(5) + '\n2. ' + words(4),
          () => '- ' + words(5) + '\n\n- ' + words(4),                  // LOOSE list across a blank line
          () => '- ' + words(4) + '\n\n  continued paragraph in the item', // indented continuation
          () => '```js\nconst x = 1;\n\nfunction f() { return x; }\n```', // blank line INSIDE a fence
          () => '~~~\nplain ~~ fence body\n~~~',
          () => '````\n```\nnested ticks\n```\n````',                    // longer fence wins
          () => '```\nunclosed fence body\n' + words(6),                 // never closed
          () => '> ' + words(8) + '\n> ' + words(6),
          () => '> ' + words(5) + '\n\n> ' + words(5),                   // adjacent quotes
          () => '    indented code one\n    indented code two',
          () => '    code a\n\n    code b',                             // one code block across a blank line
          () => '| a | b |\n| --- | --- |\n| ' + words(2) + ' | ' + words(2) + ' |',
          () => '---',
          () => words(4) + '\n' + '='.repeat(5),                        // setext h1
          () => words(4) + '\n' + '-'.repeat(5),                        // setext h2
          () => words(8) + ' `inline` and **bold** and *em*.',
          () => '<div>literal html</div>',
          () => 'A line\nwith a soft break.',                           // breaks: true
          () => '[a link](https://example.com) mid-paragraph ' + words(6),
        ];
        // Reference and footnote definitions reach forward arbitrarily far, so
        // no split is safe once one appears -- the fallback must keep these
        // CORRECT, which is the whole point of including them here.
        const UNSAFE = [
          () => '[ref]: https://example.com\n\nSee [the doc][ref] for more.',
          () => 'Text with a note[^1].\n\n[^1]: the footnote body.',
          // CommonMark HTML blocks of types 1-5 end on a CLOSING CONDITION,
          // not on a blank line, so each of these renders as one token whole
          // and as two if cut at the inner blank line -- and a committed
          // prefix is never revisited, so the seam would be permanent. The
          // generator never emitted one, which is how the boundary rule stayed
          // provably-wrong-but-green: <div> (type 6) DOES end at a blank line
          // and was the only html shape in BLOCKS.
          () => '<pre>\nfoo\n\nbar\n</pre>',
          () => '<script>\nvar a = 1;\n\nvar b = 2;\n</script>',
          () => '<style>\n.a { color: red }\n\n.b { color: blue }\n</style>',
          () => '<textarea>\nline one\n\nline two\n</textarea>',
          () => '<!--\na comment\n\nstill the comment\n-->',
          () => '<![CDATA[\nraw\n\nmore raw\n]]>',
        ];

        const CHUNKS = [1, 5, 23, 97];
        for (let d = 0; d < 60; d++) {
          const n = 4 + Math.floor(rnd() * 6);
          const parts = Array.from({ length: n }, () => pick(BLOCKS)());
          if (d % 7 === 0) parts.splice(Math.floor(rnd() * parts.length), 0, pick(UNSAFE)());
          const doc = parts.join('\n\n');

          const ref = document.createElement('div');
          ref.innerHTML = renderMarkdown(doc);
          const want = ref.innerHTML;

          for (const step of CHUNKS) {
            const el = document.createElement('div');
            const ms = new MarkdownStream(el);
            try {
              for (let i = step; i < doc.length; i += step) ms.render(doc.slice(0, i));
              ms.render(doc);
            } catch (err) {
              return { doc, step, threw: String(err && err.message) };
            }
            if (el.innerHTML !== want) return { doc, step, got: el.innerHTML, want };
          }
        }
        return null;
      }, base);
      if (bad) {
        const detail = bad.threw
          ? `threw: ${bad.threw}`
          : `\n  got:  ${JSON.stringify(String(bad.got).slice(0, 500))}\n  want: ${JSON.stringify(String(bad.want).slice(0, 500))}`;
        assert(false,
          `growing this document ${bad.step} chars at a time diverged from a whole-document render.\n`
          + `  source: ${JSON.stringify(bad.doc.slice(0, 500))}\n  ${detail}`);
      }
    });

    await suite.check('a document with no safe split still renders correctly', async () => {
      // One giant block (no blank line anywhere) can never advance the
      // boundary. It must stay CORRECT -- the rate limit is what bounds its
      // cost, not the boundary machinery.
      const ok = await md.evaluate(async (b) => {
        const { MarkdownStream } = await import(`${b}/v3/js/markdown-stream.js`);
        const { renderMarkdown } = await import(`${b}/v3/js/markdown.js`);
        const doc = 'one enormous paragraph with no blank line anywhere in it '.repeat(40).trim();
        const el = document.createElement('div');
        const ms = new MarkdownStream(el);
        for (let i = 7; i < doc.length; i += 7) ms.render(doc.slice(0, i));
        ms.render(doc);
        const ref = document.createElement('div');
        ref.innerHTML = renderMarkdown(doc);
        return el.innerHTML === ref.innerHTML;
      }, base);
      assert(ok, 'a split-free document did not survive incremental rendering');
    });


    // ---- the shared document writer's ordering rule ----------------------
    // chat and notebook each carried a byte-identical copy of this, and what
    // the copies duplicated was not boilerplate: it is the keepalive rule
    // below, hand-maintained in two files and guarded in neither.
    await suite.check('a keepalive prompt write is dispatched ahead of the PUT chain', async () => {
      const out = await md.evaluate(async (b) => {
        const { createDocumentWriter } = await import(`${b}/v3/js/document-writer.js`);
        const calls = [];
        let releaseFirst;
        const update = (docId, body, opts) => {
          calls.push({ body, keepalive: Boolean(opts?.keepalive) });
          // The first PUT never settles: it is the in-flight request a
          // hide-time flush must not queue behind.
          if (calls.length === 1) return new Promise((r) => { releaseFirst = r; });
          return Promise.resolve();
        };
        const w = createDocumentWriter({ update, onError: () => {} });
        w.putSystemPrompt('d1', 'typed');
        // Let the ordinary write actually REACH the network before flushing --
        // that is the real sequence (a keystroke PUT in flight, then the page
        // hides), and the invariant is about not queueing behind it.
        await new Promise((r) => setTimeout(r, 0));
        const inFlight = calls.length;
        w.putSystemPrompt('d1', 'flushed', { keepalive: true });
        const dispatched = calls.length;  // read SYNCHRONOUSLY: no await to hide behind
        releaseFirst?.();
        return { inFlight, dispatched, calls };
      }, base);
      // A page may be UNLOADING: a request still queued behind an in-flight
      // PUT is never sent at all, so the newest value would be lost.
      assert(out.inFlight === 1, `setup: expected one in-flight PUT, saw ${out.inFlight}`);
      assert(out.dispatched === 2,
        `the keepalive write queued behind the in-flight PUT (${out.dispatched} dispatched, expected 2)`);
      assert(out.calls[1].keepalive === true, 'the flush lost its keepalive flag');
      assert(out.calls[1].body.system_prompt === 'flushed',
        `the flush carried ${JSON.stringify(out.calls[1].body)}`);
    });

    await suite.check('ordinary prompt writes stay serialised', async () => {
      // The other half: without keepalive they must chain, or an older value
      // can land after a newer one.
      const out = await md.evaluate(async (b) => {
        const { createDocumentWriter } = await import(`${b}/v3/js/document-writer.js`);
        const calls = [];
        let releaseFirst;
        const update = (docId, body) => {
          calls.push(body.system_prompt);
          if (calls.length === 1) return new Promise((r) => { releaseFirst = r; });
          return Promise.resolve();
        };
        const w = createDocumentWriter({ update, onError: () => {} });
        w.putSystemPrompt('d1', 'one');
        w.putSystemPrompt('d1', 'two');
        await new Promise((r) => setTimeout(r, 0));
        const beforeRelease = calls.length;
        releaseFirst?.();
        await new Promise((r) => setTimeout(r, 0));
        return { beforeRelease, after: calls };
      }, base);
      assert(out.beforeRelease === 1,
        `a second ordinary write went out while the first was in flight (${out.beforeRelease})`);
      assert(out.after.join(',') === 'one,two', `writes landed as ${out.after.join(',')}`);
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
