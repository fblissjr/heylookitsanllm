// Chat suite: the most-verified surface. Covers streaming, position-based
// edit/regenerate/delete truncation, stop=partial-saved, post-abort health,
// settings + the localStorage sampler seed, conversation CRUD, and a 390px
// mobile pass. Data is cleared by the orchestrator before this runs.

import { assert, waitFor, sleep, skip } from '../lib/harness.mjs';
import { serverGet } from '../lib/server-state.mjs';
import { clickByText, armedClick, count, textOf, waitForLabel, settingsInputValue, setSettingsInput, noHorizontalOverflow, openDrawer, closeDrawer, driftText } from '../lib/dom.mjs';

const COMPOSER = '.chat__composer textarea';
const SEND_BTN = '.chat__composer .btn--primary';
const MODEL_SELECT = '.chat__bar select';

// Stop-check generations reopen with a large cap so there's time to click Stop
// before generation finishes.
const STOP_TEST_MAX_TOKENS = 400;

// Streaming-cadence regression thresholds. Old poll ceiling was ~10/s / ~100ms
// gaps; the fix delivers ~90/s / ~11ms on the MoE. These sit ~2-3x inside the
// regression signature so a fast model passes comfortably (see README).
const CADENCE_MIN_CHUNKS = 8;
const CADENCE_MAX_MEDIAN_MS = 50;
const CADENCE_MIN_RATE = 30;

async function sendText(page, text) {
  await page.click(COMPOSER);
  await page.type(COMPOSER, text);
  await page.keyboard.press('Enter');
}

async function sendBtnLabel(page) {
  return textOf(page, SEND_BTN);
}

// Wait until the composer send button reads "Send" again (stream released).
async function waitIdle(page, timeout = 30000) {
  await waitForLabel(page, SEND_BTN, 'Send', { timeout, message: 'stream never returned to idle' });
}

async function assistantCount(page) {
  return count(page, '.message--assistant:not(.message--streaming)');
}
async function userCount(page) {
  return count(page, '.message--user');
}

async function lastAssistantText(page) {
  const els = await page.$$('.message--assistant:not(.message--streaming) .message-content');
  if (!els.length) return '';
  const t = await els[els.length - 1].evaluate((e) => e.textContent.trim());
  return t;
}

// One reader for the persisted server-side shape of the (single) conversation
// in this suite: message counts + the last assistant message id. Lets
// outcome-based checks avoid racing the transient Stop-button label -- a
// one-word reply on the fast MoE can start AND finish inside a single poll
// interval (seen live 2026-07-23 -- deterministic timeout once the machine
// warms up), so waiting to OBSERVE "Stop" is unsound; waiting for the
// server-persisted outcome is not.
async function conversationStateServerSide(page) {
  const list = await serverGet(page, '/v1/conversations');
  const first = list?.conversations?.[0];
  if (!first) return { userCount: 0, assistantCount: 0, lastAssistantId: null };
  const msgs = (await serverGet(page, `/v1/conversations/${first.id}`))?.messages ?? [];
  return {
    userCount: msgs.filter((m) => m.role === 'user').length,
    assistantCount: msgs.filter((m) => m.role === 'assistant').length,
    lastAssistantId: msgs.filter((m) => m.role === 'assistant').at(-1)?.id ?? null,
  };
}

// By-id reader for the capability/thinking/image section below, where several
// checks each seed their OWN fresh conversation -- conversationStateServerSide's
// "conversations[0] is the one conversation this suite has" assumption no
// longer holds once more than one exists, so reads there target an explicit id.
async function conversationStateById(page, id) {
  const msgs = (await serverGet(page, `/v1/conversations/${id}`))?.messages ?? [];
  return {
    userCount: msgs.filter((m) => m.role === 'user').length,
    assistantCount: msgs.filter((m) => m.role === 'assistant').length,
    lastUser: msgs.filter((m) => m.role === 'user').at(-1) ?? null,
    lastAssistant: msgs.filter((m) => m.role === 'assistant').at(-1) ?? null,
  };
}

// Click New and resolve the fresh conversation's id server-side -- by MAX
// created_at, never by list position: the list orders by updated_at, and a
// PRIOR check's trailing debounced params PUT (bindDocumentParams, 400ms)
// can bump its old conversation past the fresh one inside this window,
// silently handing back the wrong id (reproduced live 2026-07-23: the image
// round-trip check read an old text conversation and found 0 image blocks).
// created_at is immutable, so the newest-created conversation is always the
// one New just made.
async function newFreshConversation(page) {
  await clickByText(page, '.chat__convs-head button', 'New');
  // The conv-item paints at the START of selectConversation; its async
  // hydrate is still in flight then, and hydrateDocParams silently resets
  // the sampler cache from the new conversation's params -- a settings
  // change (e.g. clicking the thinking toggle) made in that window gets
  // reverted (seen live 2026-07-23). newConversation() focuses the composer
  // as its LAST act, after selection fully completes -- wait for that.
  await waitFor(async () => page.evaluate(
    () => document.activeElement?.matches('.chat__composer textarea') ?? false),
  { message: 'new conversation not fully selected (composer never focused)' });
  const id = await page.evaluate(async () => {
    const { conversations } = await (await fetch('/v1/conversations')).json();
    return conversations.reduce(
      (a, b) => (a && a.created_at > b.created_at ? a : b), null)?.id ?? null;
  });
  assert(id, 'could not resolve the fresh conversation id server-side');
  return id;
}

// The suite's current conversation, fetched SINGLY.
//
// `GET /v1/conversations` deliberately omits system_prompt and params (3b44c61,
// 2026-08-26: the sidebar reads neither and both are unbounded on a response
// that ships on page load and on every foreground). Two checks below asserted
// `conversations.some(c => c.system_prompt === ...)` against that list and had
// been unsatisfiable since -- written 2026-07-09 and correct for seven weeks.
// The body carries both; that is what "fetch the conversation to get either"
// means. Resolved by newest created_at, the same rule newFreshConversation
// uses, because these checks run against the one conversation the suite made.
async function currentConversation(page) {
  return page.evaluate(async () => {
    const { conversations } = await (await fetch('/v1/conversations')).json();
    const newest = conversations.reduce(
      (a, b) => (a && a.created_at > b.created_at ? a : b), null);
    if (!newest) return null;
    return (await fetch(`/v1/conversations/${newest.id}`)).json();
  });
}

// Client-observed streaming cadence, measured INSIDE the page. The Phase 1 fix
// (asyncio.wait instead of a 0.1s poll in async_generator_with_abort) is
// invisible to server-side telemetry -- only a client timing the stream can
// catch a regression back to the ~100ms poll ceiling. Returns per-delta
// inter-arrival gaps.
//
// Measured against /v1/messages: the delivery machinery under test is shared by
// every streaming route, and this is a wire v3 actually speaks (notebook and
// explore) -- it used to probe /v1/chat/completions, which no page had used
// since v1.74.0 (the route itself was removed in v1.79.66), so the comment
// claiming it was "the path the app uses" had quietly stopped being true. Deliberately NOT chat's own
// /v1/conversations/{id}/generate: that path PERSISTS, and an extra assistant
// row mid-suite would break the sidebar-title, survives-a-reload and
// second-turn checks that run after this one.
//
// Counts any content_block_delta carrying text, thinking deltas included: the
// question is when BYTES reach the client, and counting only the answer text
// measures a sparse subset whose gaps span the thinking phase. Measured on one
// gemma-4-E4B generation, same stream counted both ways: 61 deltas at 19.8ms
// median (all) vs 18 at 28.5ms (content only) -- and the old content-only count
// on the OpenAI wire yielded ELEVEN samples, one unlucky run above the >= 8
// floor. That is how the check false-failed at a 197ms median on a thinking
// run while the wire itself was delivering at 19ms. Quantization would slow
// every delta alike, so counting more of them cannot hide it -- it only removes
// the phase-transition confound and gives ~5x the samples.
async function measureStreamCadence(page, model, maxTokens) {
  return page.evaluate(async (model, maxTokens) => {
    const marks = [];
    let usage = null;
    const res = await fetch('/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: 'Write several full sentences about the sea and the sky.' }],
        max_tokens: maxTokens,
        stream: true,
        stream_options: { include_usage: true },
      }),
    });
    if (!res.ok) return { ok: false, status: res.status };
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf('\n\n')) !== -1) {
        const evt = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of evt.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const d = line.slice(5).trim();
          if (!d) continue;
          let c;
          try { c = JSON.parse(d); } catch { continue; }
          // Messages grammar: text rides content_block_delta; usage lands on
          // message_delta (and again on message_stop).
          if (c.type === 'content_block_delta' && c.delta?.text) marks.push(performance.now());
          if (c.usage) usage = c.usage;
          else if (c.delta?.usage) usage = c.delta.usage;
          else if (c.message?.usage) usage = c.message.usage;
        }
      }
    }
    const gaps = [];
    for (let i = 1; i < marks.length; i++) gaps.push(marks[i] - marks[i - 1]);
    gaps.sort((a, b) => a - b);
    const median = gaps.length ? gaps[Math.floor(gaps.length / 2)] : null;
    const spanMs = marks.length > 1 ? marks[marks.length - 1] - marks[0] : null;
    const rate = spanMs ? (marks.length - 1) / (spanMs / 1000) : null;
    return { ok: true, chunks: marks.length, median, rate, usage };
  }, model, maxTokens);
}

export async function runChatSuite({ suite, ctx, config }) {
  const { page } = ctx;
  await ctx.open('#/chat');

  await suite.check('app boots with 6 nav routes', async () => {
    await page.waitForSelector('#nav-desktop .nav-item');
    // The settings gear is also an #nav-desktop .nav-item but has no data-route;
    // filter to real routes (defined dataset.route) before counting.
    const routes = await page.$$eval('#nav-desktop .nav-item', (els) =>
      [...new Set(els.map((e) => e.dataset.route).filter(Boolean))]);
    assert(routes.length === 6, `expected 6 routes, got ${routes.join(',')}`);
    assert(['chat', 'notebook', 'explore', 'jspace', 'models', 'perf'].every((r) => routes.includes(r)),
      `missing route in ${routes.join(',')}`);
  });

  await suite.check('default page is chat', async () => {
    // ASSERT WHAT THE ROUTER ACTUALLY WRITES. This read `document.body.dataset
    // .page` for as long as anyone can trace, and that attribute exists nowhere
    // in the frontend -- so the check reported `undefined !== 'chat'` on a
    // perfectly healthy page and took the rest of the suite down behind it.
    // app.js's navigate() marks the active route by toggling
    // `nav-item--active` and setting aria-current="page" on the nav link; that
    // is the observable, and it is load-bearing a11y (DESIGN.md §7) rather
    // than a test-only hook, so it is the right thing to pin.
    const active = await page.$eval('#nav-desktop .nav-item[aria-current="page"]',
      (el) => el.dataset.route).catch(() => null);
    assert(active === 'chat', `active nav route=${active}`);
    await page.waitForSelector('.chat');
  });

  await suite.check('composer, send button, model select present', async () => {
    await page.waitForSelector(COMPOSER);
    await page.waitForSelector(SEND_BTN);
    await page.waitForSelector(MODEL_SELECT);
    assert((await sendBtnLabel(page)) === 'Send', 'send button label');
  });

  await suite.check('model select contains the E2E model', async () => {
    // POLL, never one-shot: the select exists at #app but fills only when
    // the page's listModels lands -- measured ~1.7s on a 29-model registry
    // (cold Chrome, live server). The old single $$eval raced that window
    // and took the next six checks down with it ("No models available").
    await waitFor(async () => {
      const opts = await page.$$eval(`${MODEL_SELECT} option`, (els) => els.map((e) => e.value));
      return opts.includes(config.model);
    }, { message: `model ${config.model} never appeared in the select` });
    await page.select(MODEL_SELECT, config.model);
  });

  await suite.check('empty state before any conversation', async () => {
    // Same race as the select: rendered by the same async setup.
    await waitFor(async () => {
      const empty = await textOf(page, '.chat__messages .empty-state');
      return Boolean(empty && empty.length > 0);
    }, { message: 'empty-state prompt never appeared' });
  });

  await suite.check('send streams an assistant reply that persists', async () => {
    await page.select(MODEL_SELECT, config.model);
    await sendText(page, 'Say hello in one short sentence.');
    // user bubble shows optimistically
    await waitFor(async () => (await userCount(page)) === 1, { message: 'user bubble' });
    // streaming placeholder appears at some point
    await waitFor(async () => (await sendBtnLabel(page)) === 'Stop', { message: 'never entered streaming' });
    await waitIdle(page);
    // finishStream flips the Send button to idle (releaseStream) BEFORE it awaits
    // the save and renders the persisted (non-streaming) message, so poll for the
    // message rather than racing the assert against that gap.
    await waitFor(async () => (await assistantCount(page)) === 1, { message: 'assistant reply not persisted/rendered' });
    const reply = await lastAssistantText(page);
    assert(reply.length > 0, 'assistant reply is empty');
  });

  await suite.check('status line reports token usage after completion', async () => {
    const status = await textOf(page, '.chat__status');
    assert(status && /token/i.test(status), `status="${status}"`);
  });

  await suite.check('streaming delivery is not poll-quantized (client cadence)', async () => {
    // Guards the Phase 1 delivery fix. The old poll capped delivery at ~10/s
    // (~100ms gaps); the fix delivers as fast as the model decodes (~90 tok/s /
    // ~11ms on the MoE). Thresholds sit ~2-3x inside the regression signature so
    // a fast model passes comfortably. Requires a fast E2E_MODEL (the default
    // MoE); a natively-slow model would false-fail -- see README.
    const c = await measureStreamCadence(page, config.model, 64);
    assert(c.ok, `probe request failed (status=${c.status})`);
    assert(c.chunks >= CADENCE_MIN_CHUNKS, `too few content chunks to measure (${c.chunks})`);
    // The quantization SIGNATURE is chunks that carry several tokens each at
    // ~100ms spacing. A natively slow model (a 27B Q8 on llama-server decodes
    // ~19 tok/s, one token every ~52ms) sends ONE token per chunk at the
    // model's own pace, which the gap/rate thresholds alone cannot tell from
    // quantization (v1.79.64: Qwen3.8 read 52ms and false-failed). So when
    // usage shows ~1 token per chunk, delivery is per-token by construction
    // and the pace is the model's; the thresholds still bite whenever chunks
    // carry more than one token.
    const perChunk = c.usage?.output_tokens && c.chunks ? c.usage.output_tokens / c.chunks : null;
    const perToken = perChunk != null && perChunk <= 1.5;
    if (perToken && (c.median >= CADENCE_MAX_MEDIAN_MS || c.rate <= CADENCE_MIN_RATE)) {
      console.log(`      cadence: ${c.chunks} chunks, median gap ${c.median.toFixed(1)}ms, ${c.rate.toFixed(1)}/s -- `
        + `${perChunk.toFixed(2)} tok/chunk: per-token delivery at the model's own pace (slow model, not quantized)`);
      return;
    }
    assert(c.median != null && c.median < CADENCE_MAX_MEDIAN_MS,
      `median inter-chunk gap ${c.median?.toFixed(1)}ms >= ${CADENCE_MAX_MEDIAN_MS}ms (poll-quantization signature is ~100ms)`);
    assert(c.rate > CADENCE_MIN_RATE,
      `client decode rate ${c.rate?.toFixed(1)}/s <= ${CADENCE_MIN_RATE}/s (old poll ceiling was ~10/s)`);
    console.log(`      cadence: ${c.chunks} chunks, median gap ${c.median.toFixed(1)}ms, ${c.rate.toFixed(1)}/s`
      + (perChunk != null ? `, ${perChunk.toFixed(2)} tok/chunk` : ''));
  });

  await suite.check('conversation appears in sidebar with derived title', async () => {
    await waitFor(async () => (await count(page, '.conv-item')) === 1, { message: 'conv-item' });
    const title = await textOf(page, '.conv-item__title');
    assert(title && title.startsWith('Say hello'), `title="${title}"`);
  });

  await suite.check('assistant reply survives a reload', async () => {
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.chat');
    await waitFor(async () => (await assistantCount(page)) === 1, { message: 'reply gone after reload' });
    const reply = await lastAssistantText(page);
    assert(reply.length > 0, 'reply empty after reload');
  });

  await suite.check('second turn appends to the same conversation', async () => {
    await page.select(MODEL_SELECT, config.model);
    await sendText(page, 'Now say goodbye briefly.');
    await waitFor(async () => (await userCount(page)) === 2, { message: 'second user bubble' });
    await waitIdle(page);
    assert((await assistantCount(page)) === 2, 'two assistant messages');
    assert((await count(page, '.conv-item')) === 1, 'still one conversation');
  });

  await suite.check('edit opens an inline editor on a user message', async () => {
    // Was: captured a handle to the first .message--user and asserted it
    // truthy -- true regardless of whether Edit/Cancel did anything (the
    // handle exists before either is clicked). Assert what the check's name
    // claims instead: the editor prefills with the message's own text, and
    // Cancel restores that exact text (a Cancel that silently cleared it
    // would have passed the old version).
    const originalText = await page.$eval('.message--user .message-content', (e) => e.textContent);
    await clickByText(page, '.message--user .message__actions button', 'Edit');
    await page.waitForSelector('.message-edit textarea', { timeout: 5000 });
    const editorValue = await page.$eval('.message-edit textarea', (e) => e.value);
    assert(editorValue === originalText, `editor prefilled "${editorValue}", expected "${originalText}"`);
    // Cancel to restore
    await clickByText(page, '.message-edit__buttons button', 'Cancel');
    await waitFor(async () => (await count(page, '.message-edit')) === 0, { message: 'editor did not close' });
    const restoredText = await page.$eval('.message--user .message-content', (e) => e.textContent);
    assert(restoredText === originalText, 'Cancel did not restore the original text');
  });

  await suite.check('save & regenerate truncates then regenerates', async () => {
    // Edit the FIRST user message -> truncates everything after position, then
    // streams a single fresh assistant reply.
    await clickByText(page, '.message--user .message__actions button', 'Edit');
    await page.waitForSelector('.message-edit textarea');
    await page.click('.message-edit textarea', { clickCount: 3 });
    // "one short sentence" (not "single word"): terse prompts raise the
    // empty-EOS odds, and this check + the regenerate check below strictly
    // assert a persisted reply -- an empty completion legally persists
    // nothing (finishStream). Sentence-shaped asks have been reliably
    // non-empty; the max_tokens seed keeps the run fast regardless.
    await page.type('.message-edit textarea', 'Reply with one short sentence.');
    await clickByText(page, '.message-edit__buttons button', 'Save & Regenerate');
    // Outcome-based, not the transient Stop label: a short reply on the
    // fast MoE can start AND finish inside a single poll interval (the same
    // class of flake fixed on the regenerate check below). The click handler
    // awaits the truncation DELETE and only THEN unconditionally calls
    // startStream, so a server-confirmed truncation is a safe proxy for
    // "generation began" without racing the button label.
    await waitFor(async () => (await conversationStateServerSide(page)).userCount === 1,
      { message: 'truncation to 1 user message never landed server-side' });
    await waitIdle(page);
    const u = await userCount(page);
    const a = await assistantCount(page);
    assert(u === 1, `expected 1 user msg after truncation, got ${u}`);
    assert(a === 1, `expected 1 assistant msg, got ${a}`);
  });

  await suite.check('regenerate on an assistant message replaces it', async () => {
    const before = (await conversationStateServerSide(page)).lastAssistantId;
    assert(before !== null, 'had a persisted assistant message');
    await clickByText(page, '.message--assistant .message__actions button', 'Regenerate');
    // Don't wait to OBSERVE the transient Stop label: a short reply on the
    // fast MoE can start AND finish inside a single poll interval (seen
    // live 2026-07-23 -- deterministic timeout once the machine warms up).
    // The outcome is what matters: the regenerated reply persists under a
    // NEW message id, and the thread settles back to one assistant message.
    await waitFor(async () => {
      const id = (await conversationStateServerSide(page)).lastAssistantId;
      return id !== null && id !== before;
    }, { message: 'regenerated reply not persisted under a new id' });
    await waitIdle(page);
    await waitFor(async () => (await assistantCount(page)) === 1,
      { message: 'thread did not settle to one assistant message' });
    assert((await userCount(page)) === 1, 'user message preserved');
  });

  await suite.check('save & continue extends the message in place', async () => {
    // Continuation (v1.61.0): edit the assistant reply down to a distinctive
    // partial, Save & Continue -- the model FINISHES the same message
    // (prefill). Contract under test: same message id (updated in place,
    // never a new row), edited text as the verbatim prefix, and the reply
    // grew past it. Leaves the thread at 1 user + 1 assistant, which the
    // delete check below expects.
    const before = (await conversationStateServerSide(page)).lastAssistantId;
    assert(before !== null, 'had a persisted assistant message');
    const PREFIX = 'Once upon a time there was a';
    await clickByText(page, '.message--assistant .message__actions button', 'Edit');
    await page.waitForSelector('.message-edit textarea');
    await page.evaluate((p) => {
      const ta = document.querySelector('.message-edit textarea');
      ta.value = p;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }, PREFIX);
    await clickByText(page, '.message-edit__buttons button', 'Save & Continue');
    await waitFor(async () => {
      const list = await serverGet(page, '/v1/conversations');
      const first = list?.conversations?.[0];
      if (!first) return false;
      const msgs = (await serverGet(page, `/v1/conversations/${first.id}`))?.messages ?? [];
      const last = msgs.filter((m) => m.role === 'assistant').at(-1);
      return Boolean(last && last.id === before
        && (last.content ?? '').startsWith(PREFIX)
        && (last.content ?? '').length > PREFIX.length);
    }, { message: 'continuation did not extend the same message in place' });
    await waitIdle(page);
    await waitFor(async () => (await assistantCount(page)) === 1,
      { message: 'continuation must not add a message row' });
    assert((await userCount(page)) === 1, 'user message preserved');
  });

  await suite.check('delete (armed) removes exactly that message -- the tail survives', async () => {
    // v1.74.0: Delete means delete. The old spelling truncated everything
    // after the row, so deleting the FIRST user message (mid-thread) is the
    // discriminating case: under truncation the assistant rows vanish too.
    const before = await conversationStateServerSide(page);
    assert(before.userCount >= 1 && before.assistantCount >= 1,
      `need a user row with messages after it (got ${JSON.stringify(before)})`);
    const delBtn = await page.evaluateHandle(() => {
      const msg = document.querySelector('.message--user');
      return [...msg.querySelectorAll('.message__actions button')].find((b) => b.textContent.trim() === 'Delete');
    });
    await armedClick(delBtn.asElement());
    await delBtn.dispose();
    await waitFor(async () => (await userCount(page)) === before.userCount - 1,
      { message: 'user message not deleted' });
    const after = await conversationStateServerSide(page);
    assert(after.userCount === before.userCount - 1,
      `server kept ${after.userCount} user rows, expected ${before.userCount - 1}`);
    assert(after.assistantCount === before.assistantCount,
      `assistant rows changed (${before.assistantCount} -> ${after.assistantCount}) `
      + '-- Delete truncated the tail again');
  });

  // ---- stop = partial saved (needs a long generation) --------------------
  await suite.check('stop mid-stream saves the partial reply', async () => {
    await ctx.open('#/chat');
    await page.select(MODEL_SELECT, config.model);
    // Phase 2: generation params come from the CONVERSATION's params bag
    // (the server builds the request from the store), so the budget must be
    // raised through the panel on a fresh conversation -- a localStorage
    // seed alone never reaches the store (same trap documented on the
    // thinking-block check below). Big enough that a fast model is still
    // mid-stream when Stop lands.
    const convId = await newFreshConversation(page);
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', '4000');
    await closeDrawer(page);
    await waitFor(async () => page.evaluate(async (id) => {
      const conv = await (await fetch(`/v1/conversations/${id}`)).json();
      return (conv.params ?? {}).max_tokens === 4000;
    }, convId), { message: 'max_tokens params PUT never landed server-side' });
    await sendText(page, 'Write a long detailed paragraph about the ocean.');
    // wait until partial content is visibly streaming
    await waitFor(async () => {
      const t = await textOf(page, '.message--streaming .message-content');
      return t && t.length > 0;
    }, { message: 'no partial content appeared' });
    await page.click(SEND_BTN); // Stop
    // finishStream flips the button to idle (releaseStream) BEFORE it awaits the
    // partial-save and sets the status, so wait for the status itself -- its
    // presence also proves the save resolved (matters for the reload check next).
    await waitFor(async () => /stopped/i.test((await textOf(page, '.chat__status')) || ''),
      { message: 'stop status never appeared' });
    const status = await textOf(page, '.chat__status');
    assert(/partial/i.test(status), `expected partial-saved status, got "${status}"`);
    const reply = await lastAssistantText(page);
    assert(reply.length > 0, 'partial reply not on screen');
    // Contain the 4000: it lives in THIS conversation's params AND the
    // panel cache, and this conversation stays the newest for a while --
    // hydration would push 4000 into every later check (and a fresh
    // conversation would snapshot it). Restore the suite's default seed
    // value (browser.mjs open() seeds max_tokens=24), NOT '' -- an empty
    // value means backend-cascade, whose budget is model-sized and blows
    // the waitIdle windows of later thinking checks.
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', '24');
    await closeDrawer(page);
    await waitFor(async () => page.evaluate(async (id) => {
      const conv = await (await fetch(`/v1/conversations/${id}`)).json();
      return (conv.params ?? {}).max_tokens === 24;
    }, convId), { message: 'max_tokens never restored on the stop-test conversation' });
  });

  await suite.check('partial reply persisted across reload', async () => {
    const before = await assistantCount(page);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.chat');
    await waitFor(async () => (await assistantCount(page)) === before, { message: 'partial lost on reload' });
    assert(before >= 1, 'had a partial reply');
  });

  // ---- disconnect persistence (TODO P2, plan Phase 1's server-owned half) --
  await suite.check('a mid-stream disconnect still persists the partial (server task)', async () => {
    // The contract: a phone locking mid-stream loses nothing. No Stop is
    // pressed -- the fetch dies with the page (reload), the server notices
    // the disconnect, and its DETACHED task persists whatever was generated.
    // Since v1.79.26 the run OUTLIVES the response: the reload ends the
    // subscription, the server task runs to completion and commits the
    // WHOLE answer, and the reloaded page shows it generating until then.
    // So the cap is the stop-test one, not 4000 -- large enough to still
    // be streaming when the reload lands, small enough that the run
    // finishes inside this check's window instead of 45s later.
    const convId = await newFreshConversation(page);
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', String(STOP_TEST_MAX_TOKENS));
    await closeDrawer(page);
    await waitFor(async () => page.evaluate(async (id, cap) => {
      const conv = await (await fetch(`/v1/conversations/${id}`)).json();
      return (conv.params ?? {}).max_tokens === cap;
    }, convId, STOP_TEST_MAX_TOKENS), { message: 'max_tokens params PUT never landed server-side' });
    await sendText(page, 'Write a long detailed story about a lighthouse.');
    await waitFor(async () => {
      const t = await textOf(page, '.message--streaming .message-content');
      return t && t.length > 0;
    }, { message: 'no partial content appeared before the disconnect' });
    // LIVENESS GUARD: if the stream finished before the reload, this check
    // silently degenerates into the plain reload check above and passes with
    // the disconnect path deleted. The Send button reads "Stop" exactly
    // while a stream is live -- assert it in the same breath as the reload.
    // (The 4000-token budget makes completion this early implausible, but
    // implausible is not a guard.)
    assert((await sendBtnLabel(page)) === 'Stop',
      'stream already finished before the reload -- this run never tested the disconnect path');
    // Reload mid-stream = the disconnect. The in-stream beforeunload guard
    // raises a confirm dialog -- accept it or the reload (and the suite) hangs.
    page.once('dialog', (d) => d.accept().catch(() => {}));
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.chat');
    // The reloaded page finds the run still going and SAYS so: the status
    // names it and the Send button reads Stop (setRemoteGenerating). It does
    // not poll (page.js: two invalidation points, select and resume), so the
    // finished row lands on the next reselect -- which is what the second
    // half exercises. A live re-attach to the running stream is the open
    // improvement here, not a check to fake around.
    await waitFor(async () => /still generating/i.test((await textOf(page, '.chat__status')) || '')
      || (await conversationStateById(page, convId)).assistantCount === 1,
    { timeout: 15000, message: 'the reloaded page neither reported the run nor found it finished' });
    // The detached task keeps generating after the disconnect and commits
    // when it finishes -- give it the whole budget's worth of time.
    await waitFor(async () => {
      const state = await conversationStateById(page, convId);
      return state.assistantCount === 1;
    }, { timeout: 90000, message: 'the detached run never persisted its reply' });
    const msgs = (await serverGet(page, `/v1/conversations/${convId}`))?.messages ?? [];
    const partial = msgs.find((m) => m.role === 'assistant');
    assert(partial?.content?.length > 0, 'the persisted reply has no content');
    // Reselecting the conversation is the refresh the design offers: click
    // another row, then this one, and the saved reply must be there. Clicks
    // happen IN-PAGE by title: the list re-renders on every store bump, so
    // an element handle taken a tick earlier is a detached node by the time
    // it is clicked ("Node is either not clickable").
    const clickRow = (matchActive, title) => page.evaluate((wantActive, t) => {
      const row = [...document.querySelectorAll('.conv-item')].find((r) => {
        const active = r.classList.contains('conv-item--active');
        const text = r.querySelector('.conv-item__title')?.textContent.trim() ?? '';
        return t ? text === t : active === wantActive;
      });
      if (!row) return false;
      row.querySelector('.conv-item__title').click();
      return true;
    }, matchActive, title);
    const storyTitle = await page.evaluate(async (id) =>
      (await (await fetch(`/v1/conversations/${id}`)).json()).title, convId);
    assert(await clickRow(false, null), 'no other conversation row to switch to');
    await waitFor(async () => page.evaluate((t) =>
      document.querySelector('.conv-item--active .conv-item__title')?.textContent.trim() !== t, storyTitle),
    { message: 'switching away never took' });
    assert(await clickRow(null, storyTitle), `no sidebar row titled ${JSON.stringify(storyTitle)}`);
    await waitFor(async () => (await lastAssistantText(page)).length > 0,
      { timeout: 15000, message: 'the reselected conversation never rendered the persisted reply' });
    assert(!/still generating/i.test((await textOf(page, '.chat__status')) || ''),
      'the status still claims the run is generating after it finished and was reselected');
    // Contain the 4000 (same hydration-leak reasoning as the stop check).
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', '24');
    await closeDrawer(page);
    await waitFor(async () => page.evaluate(async (id) => {
      const conv = await (await fetch(`/v1/conversations/${id}`)).json();
      return (conv.params ?? {}).max_tokens === 24;
    }, convId), { message: 'max_tokens never restored on the disconnect-test conversation' });
  });

  await suite.check('post-abort health: a new send completes normally', async () => {
    // Was: asserted assistantCount grows by exactly 1. But finishStream()
    // only calls addMessage when content || thinking is non-empty (chat.js)
    // -- an immediate empty-EOS reply is a legal model output (documented
    // model-flake, terse prompts especially) and legitimately saves nothing.
    // The actual claim ("a new send completes normally" after a prior abort)
    // is about pipeline health, not reply length: the stream must reach
    // completion without error and return the composer to idle; IF content
    // was produced, THAT must persist.
    await ctx.open('#/chat'); // back to small max_tokens
    await page.select(MODEL_SELECT, config.model);
    // conversations load async after mount -- wait for the list before clicking.
    await waitFor(async () => (await count(page, '.conv-item')) >= 1, { message: 'conv list' });
    await page.click('.conv-item');
    await waitFor(async () => (await userCount(page)) >= 1, { message: 'messages loaded' });
    const beforeUsers = await userCount(page);
    const beforeAssistants = await assistantCount(page);
    await sendText(page, 'Reply with one short word.');
    // Outcome-based, not the transient Stop label: wait for the new user
    // bubble (renderMessages runs, then startStream sets Stop synchronously
    // in the same tick right after -- by the time this resolves, streaming
    // has already begun, so there is no window where it can be missed).
    await waitFor(async () => (await userCount(page)) === beforeUsers + 1,
      { message: 'new message not sent after prior abort' });
    await waitIdle(page); // proves the stream reached completion, not stuck from the prior abort
    const status = await textOf(page, '.chat__status');
    assert(!/failed/i.test(status || ''), `generation failed after prior abort: "${status}"`);
    const afterAssistants = await assistantCount(page);
    assert(afterAssistants === beforeAssistants || afterAssistants === beforeAssistants + 1,
      `assistant count changed unexpectedly after prior abort: ${beforeAssistants} -> ${afterAssistants}`);
    if (afterAssistants === beforeAssistants + 1) {
      const reply = await lastAssistantText(page);
      assert(reply.length > 0, 'persisted assistant reply is empty');
    }
  });

  // ---- settings (in the app-shell drawer) --------------------------------
  // The drawer opens once here and stays open through the settings + preset +
  // sysprompt checks; it is closed before the conversation-management checks
  // (a modal drawer makes #app inert, so the page is not interactable while open).
  await suite.check('settings drawer opens with sampling controls', async () => {
    await openDrawer(page);
    await page.waitForSelector('.drawer--open .settings-panel', { timeout: 5000 });
    const labels = await page.$$eval('.settings-panel .settings-row label', (els) => els.map((e) => e.textContent.trim()));
    assert(labels.includes('Temperature'), `labels: ${labels.join(', ')}`);
    assert(labels.includes('Max tokens'), 'no Max tokens control');
  });

  await suite.check('chat bar gear opens the same drawer', async () => {
    await openDrawer(page, '.chat__settings-btn');
    await page.waitForSelector('.drawer--open .settings-panel', { timeout: 5000 });
  });

  await suite.check('the DOCUMENT\'s params win over the localStorage seed', async () => {
    // This asserted that `ctx.open()`'s localStorage seed showed in the panel,
    // which stopped being true at v1.65-66: chat hydrates the panel from the
    // conversation (`hydrateDocParams` -> `applySettings(doc.params)`), and
    // `mergeKnown` rebuilds from empty, so adopting a document REPLACES the
    // seeded cache rather than merging into it. The check was describing the
    // pre-v1.65 architecture and could not pass.
    //
    // The rule worth having is the one that replaced it, and it is stronger:
    // the document is authoritative for sampler params, so a page load that
    // re-seeds localStorage must STILL show the conversation's value.
    const conv = await currentConversation(page);
    assert(conv?.id, 'no conversation to seed params on');
    await page.evaluate(async (id) => {
      await fetch(`/v1/conversations/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: { max_tokens: 4321 } }),
      });
    }, conv.id);
    await ctx.open();            // reload; re-seeds localStorage with config.maxTokens
    await openDrawer(page, '.chat__settings-btn');
    await waitFor(async () => (await settingsInputValue(page, 'Max tokens')) === '4321',
      { message: `panel did not adopt the document's max_tokens (seed was ${config.maxTokens})` });
  });

  await suite.check('settings edit writes through to localStorage', async () => {
    await setSettingsInput(page, 'Temperature', '0.42');
    await waitFor(async () => {
      const s = await ctx.readSettings();
      return s.temperature === 0.42;
    }, { message: 'temperature not saved to localStorage' });
    // restore so it doesn't leak into later generations
    await setSettingsInput(page, 'Temperature', '');
  });

  // ---- system prompt + presets --------------------------------------------
  // ORDER COUPLING: this and the next check stamp 'Answer in exactly one
  // word. Be terse.' onto the suite's one conversation, and nothing resets
  // it. That's safe ONLY because every check between here and the capability/
  // thinking/image section below is settings/preset/conversation-CRUD --
  // none of them send a message. The capability/thinking/image section DOES
  // generate again: every check there that sends a message first creates its
  // OWN fresh conversation via newFreshConversation() (the 'New' button) so
  // the terse system prompt never silently degrades those generations.
  const SYS_PROMPT = 'Answer in exactly one word.';

  // One owner for the "find a preset <option> by its label" lookup.
  // By data-name: the option LABEL is decorated for a promptless preset
  // ("<name> — settings only", v1.79.25), the raw name rides data-name.
  const presetOptionValue = (name) => page.evaluate((n) =>
    [...document.querySelectorAll('.preset-row select option')]
      .find((o) => o.dataset.name === n || o.textContent === n)?.value ?? null, name);

  await suite.check('system prompt edit persists without blur', async () => {
    // drawer is open from the checks above; the sysprompt editor is one of chat's
    // contributed sections inside it (its details is always expanded now).
    await page.click('.sysprompt-input');
    await page.type('.sysprompt-input', SYS_PROMPT);
    // Deliberately NO blur: state commits per keystroke and the PUT is
    // debounced -- the old blur-only commit is the bug this regression guards.
    await waitFor(async () => {
      const conv = await currentConversation(page);
      return conv?.system_prompt === SYS_PROMPT;
    }, { message: 'system prompt not saved server-side' });
  });

  await suite.check('sysprompt typed text survives Escape-close', async () => {
    // Escape with focus still in the textarea removes the field before any
    // change event can fire -- the exact path that used to lose the prompt.
    await page.click('.sysprompt-input');
    await page.type('.sysprompt-input', ' Be terse.');
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.drawer--open'), { timeout: 5000 });
    await waitFor(async () => {
      const conv = await currentConversation(page);
      return (conv?.system_prompt ?? '').includes('Be terse.');
    }, { message: 'text typed before Escape-close never reached the server' });
    await openDrawer(page);  // leave the drawer open for the preset checks
    const val = await page.$eval('.sysprompt-input', (el) => el.value);
    assert(val.includes('Be terse.'), 'reopened drawer lost the typed text');
  });

  await suite.check('preset save + apply round-trips sampler state', async () => {
    await setSettingsInput(page, 'Temperature', '0.31');
    await page.click('.preset-section .input');
    await page.type('.preset-section .input', 'e2e-preset');
    await clickByText(page, '.preset-row button', 'Save as new');
    await waitFor(async () => (await presetOptionValue('e2e-preset')) !== null,
      { message: 'saved preset not listed in the select' });
    // fresh save selects the preset and matches by construction
    await waitFor(async () => (await driftText(page))?.includes('Matches'),
      { message: 'drift line not "Matches" right after save' });
    // drift the panel: the line must flip live, and selection alone must NOT
    // have touched the panel (apply is an explicit button now)
    await setSettingsInput(page, 'Temperature', '1.9');
    await waitFor(async () => /differ/i.test((await driftText(page)) ?? ''),
      { message: 'drift line did not flip to "Differs" after a sampler edit' });
    await page.select('.preset-row select', await presetOptionValue('e2e-preset'));
    assert((await settingsInputValue(page, 'Temperature')) === '1.9',
      'selecting a preset applied it (selection must be inert)');
    // explicit Apply restores the pin (no arming: the prompt is unchanged)
    await clickByText(page, '.preset-row button', 'Apply');
    await waitFor(async () => (await settingsInputValue(page, 'Temperature')) === '0.31',
      { message: 'applying the preset did not restore temperature' });
    await waitFor(async () => (await driftText(page))?.includes('Matches'),
      { message: 'drift line not back to "Matches" after apply' });
    // back to cascade so nothing leaks into later generations
    await setSettingsInput(page, 'Temperature', '');
  });

  await suite.check('system-prompt chip states what is in force and opens it', async () => {
    // "What prompt am I running?" must be answerable from the bar, including
    // the negative answer -- a hidden chip reads the same as a broken one.
    const label = await page.$eval('.chat__sysprompt-chip', (el) => el.textContent);
    const prompt = await page.$eval('.sysprompt-input', (el) => el.value);
    if (prompt.trim()) {
      assert(label.startsWith('System prompt:'), `chip="${label}" with a prompt set`);
    } else {
      assert(label === 'No system prompt', `chip="${label}" with no prompt`);
    }
    // click-through: opens the drawer AND lands in the field
    await closeDrawer(page);
    await page.evaluate(() => document.querySelector('.chat__sysprompt-chip').click());
    await page.waitForSelector('.sysprompt-input', { timeout: 5000 });
    await waitFor(async () => page.evaluate(() =>
      (document.activeElement?.className ?? '').includes('sysprompt-input')),
    { message: 'chip click did not focus the system-prompt editor' });
  });

  await suite.check('a preset carrying no system prompt overrides nothing', async () => {
    // The prompt is an OVERRIDE box (v1.62.3): a preset OWNS a prompt and
    // carries it, but an EMPTY one means "does not speak for the prompt" --
    // applying it must leave the conversation's prompt alone, not blank it.
    // The old behavior blanked it, and a Save in that window then stored the
    // blank over a good prompt, which is how two presets lost their prompts
    // at once. Data loss, so this is pinned rather than left to inspection.
    //
    // Reopen first: the chip check above scrolls the drawer down to the
    // system-prompt textarea, which puts the preset row under the sticky
    // header -- every hit-tested click here then fails with "Node is either
    // not clickable". openDrawer() closes and reopens, resetting the scroll.
    await openDrawer(page);
    const before = await page.$eval('.sysprompt-input', (el) => el.value);
    assert(before.trim(), 'precondition: this conversation should have a prompt by now');

    // save a promptless preset: clear the box, save, restore the prompt
    await page.$eval('.sysprompt-input', (el) => {
      el.value = '';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    // Set the name directly rather than click+type: the check above leaves
    // the drawer scrolled to the system-prompt textarea (the chip focuses
    // it), so a real click on the preset row can hit-test onto another
    // element -- "Node is either not clickable". The value + input event is
    // what the field's own listeners consume anyway.
    await page.$eval('.preset-section .input', (el) => {
      el.value = 'e2e-promptless';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await clickByText(page, '.preset-row button', 'Save as new');
    await waitFor(async () => (await presetOptionValue('e2e-promptless')) !== null,
      { message: 'promptless preset not listed in the select' });

    await page.$eval('.sysprompt-input', (el, v) => {
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }, before);

    await page.select('.preset-row select', await presetOptionValue('e2e-promptless'));
    // no arming: a promptless preset replaces nothing, so Apply fires at once
    await clickByText(page, '.preset-row button', 'Apply');
    await sleep(400);
    const after = await page.$eval('.sysprompt-input', (el) => el.value);
    assert(after === before,
      `promptless preset changed the prompt: "${before}" -> "${after}"`);

    // ORDER COUPLING -- leave the world exactly as found, on BOTH axes:
    //
    //  (1) the preset LIST. A promptless preset left behind is visible to
    //      every later check's provenance inference, and the delete check
    //      downstream asserts the chip goes empty. Delete it here.
    //  (2) the STAMP. Save AND Apply both write applied_preset_id on the
    //      active conversation, so this check has re-pointed the chip at
    //      e2e-promptless; the two chip checks below assert e2e-preset WITH
    //      "(edited)". Re-apply, then re-drift the panel.
    await page.select('.preset-row select', await presetOptionValue('e2e-promptless'));
    const delPromptless = await page.$('.preset-section .btn--ghost');
    await armedClick(delPromptless);
    await delPromptless.dispose();
    await waitFor(async () => (await presetOptionValue('e2e-promptless')) === null,
      { message: 'promptless preset not cleaned up' });

    await page.select('.preset-row select', await presetOptionValue('e2e-preset'));
    await clickByText(page, '.preset-row button', 'Apply');
    await waitFor(async () => (await settingsInputValue(page, 'Temperature')) === '0.31',
      { message: 're-applying e2e-preset did not restore temperature' });
    await setSettingsInput(page, 'Temperature', '');   // drift it again
    await waitFor(async () => /differ/i.test((await driftText(page)) ?? ''),
      { message: 'panel not drifted off e2e-preset again' });
  });

  await suite.check('applied-preset chip shows in the chat bar', async () => {
    // The prior check saved+applied e2e-preset, then reset Temperature to
    // cascade -- the panel is drifted, so the chip must carry "(edited)".
    const txt = await page.$eval('.chat__preset-chip', (el) => (el.hidden ? null : el.textContent));
    assert(txt?.includes('e2e-preset'), `chip="${txt}"`);
    assert(txt.includes('(edited)'), `chip should be marked (edited), got "${txt}"`);
  });

  await suite.check('applied-preset chip provenance survives a reload', async () => {
    // The stamp lives on the CONVERSATION (applied_preset_id), so "(edited)"
    // has to come back after a reload. Before that field existed this check
    // could not pass at all: the panel is drifted, so exact-match inference
    // finds nothing and the chip would simply be hidden.
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.chat');
    await waitFor(async () => {
      const t = await page.$eval('.chat__preset-chip', (el) => (el.hidden ? null : el.textContent));
      return Boolean(t?.includes('e2e-preset') && t.includes('(edited)'));
    }, { message: 'chip lost its preset provenance across a reload' });
    // Restore what the next check needs: the reload closed the drawer and
    // cleared the select-box selection (that IS session state -- only the
    // stamp is durable), and Del is disabled without a selected preset.
    await openDrawer(page);
    await page.select('.preset-row select', await presetOptionValue('e2e-preset'));
  });

  await suite.check('new conversation starts as the selected preset', async () => {
    // v1.59.0 inheritance: e2e-preset is selected (drawer state from the
    // check above) and the panel is deliberately drifted from it -- a NEW
    // conversation must start as the PRESET (prompt + params + stamp), not
    // as the drifted panel. Server-side assertions: the stamp is written at
    // create, not inferred client-side.
    await closeDrawer(page); // the drawer is modal; the New button lives in #app
    const convId = await newFreshConversation(page);
    const conv = await page.evaluate(async (id) =>
      await (await fetch(`/v1/conversations/${id}`)).json(), convId);
    assert(conv.applied_preset_id, 'new conversation was not stamped with the preset');
    assert((conv.system_prompt ?? '').includes('Be terse.'),
      `preset prompt not inherited (got ${JSON.stringify(conv.system_prompt)})`);
    assert(String(conv.params?.temperature) === '0.31',
      `preset params not inherited (got ${JSON.stringify(conv.params)})`);
    // The inherited state hydrates the panel, so the chip reads un-edited.
    await waitFor(async () => {
      const chip = await page.$eval('.chat__preset-chip', (el) => (el.hidden ? null : el.textContent));
      return Boolean(chip?.includes('e2e-preset') && !chip.includes('(edited)'));
    }, { message: 'chip should show the inherited preset un-edited' });
    // Leave the world as found: later checks index into the conv list
    // ('the second item is the older one with messages'), so the inherited
    // conversation must not survive this check. UI delete re-selects the
    // suite's original conversation, restoring the pre-check active state.
    const before = await count(page, '.conv-item');
    const delConv = await page.$('.conv-item--active .conv-item__delete');
    await armedClick(delConv);
    await delConv.dispose();
    await waitFor(async () => (await count(page, '.conv-item')) === before - 1,
      { message: 'inherited conversation not cleaned up' });
    // Restore the delete check's preconditions (drawer open + selection).
    await openDrawer(page);
    await page.select('.preset-row select', await presetOptionValue('e2e-preset'));
  });

  await suite.check('preset delete (armed) removes it from the select', async () => {
    const delBtn = await page.$('.preset-section .btn--ghost');
    await armedClick(delBtn);
    await delBtn.dispose();
    await waitFor(async () => (await presetOptionValue('e2e-preset')) === null,
      { message: 'deleted preset still listed' });
    await waitFor(async () => page.$eval('.chat__preset-chip', (el) => el.hidden),
      // Name what it still claims: the chip can survive a delete either by a
      // stamp another check left behind or by provenance inference matching
      // some other preset, and those need different fixes.
      { message: async () => 'applied-preset chip did not clear after delete; still shows '
          + JSON.stringify(await page.$eval('.chat__preset-chip', (el) => el.textContent)) });
    // done with the drawer -- close it so #app is interactable again.
    await closeDrawer(page);
  });

  // ---- conversation management -------------------------------------------
  // Defensive: if any settings/preset check above failed before its closeDrawer,
  // the drawer would still be open (#app inert) and cascade-fail every check
  // here. closeDrawer is a no-op when already closed.
  await closeDrawer(page);

  await suite.check('New button creates an additional conversation', async () => {
    const before = await count(page, '.conv-item');
    await clickByText(page, '.chat__convs-head button', 'New');
    await waitFor(async () => (await count(page, '.conv-item')) === before + 1, { message: 'no new conversation' });
    // A brand-new conversation has an active id but zero messages -- renderMessages
    // draws an empty inner (no .empty-state, which is only the no-active-conv case).
    await waitFor(async () => (await count(page, '.conv-item--active')) === 1, { message: 'new conv not active' });
    assert((await userCount(page)) === 0 && (await assistantCount(page)) === 0, 'new conversation is not empty');
  });

  await suite.check('switching conversations loads the right messages', async () => {
    const items = await page.$$('.conv-item');
    assert(items.length >= 2, 'need >= 2 conversations');
    // An older conversation WITH messages: the rows are ordered by updated_at
    // and the checks above leave empty "New conversation" rows behind, so
    // choose by title (a titled row has at least a first message) rather
    // than by position.
    // Ask the server which older conversation actually has a user message,
    // then click the row carrying that title.
    const wanted = await page.evaluate(async () => {
      const { conversations } = await (await fetch('/v1/conversations')).json();
      for (const c of conversations) {
        const body = await (await fetch(`/v1/conversations/${c.id}`)).json();
        if ((body.messages ?? []).some((m) => m.role === 'user')) return { id: c.id, title: c.title };
      }
      return null;
    });
    assert(wanted, 'no conversation with a user message exists server-side');
    let target = null;
    for (const item of items) {
      const info = await item.evaluate((el) => ({
        active: el.classList.contains('conv-item--active'),
        title: el.querySelector('.conv-item__title')?.textContent.trim() ?? el.textContent.trim(),
      }));
      if (!info.active && info.title === wanted.title) { target = item; break; }
    }
    assert(target, `no inactive sidebar row titled ${JSON.stringify(wanted.title)}`);
    await target.click();
    await waitFor(async () => (await userCount(page)) >= 1,
      { message: `older conv ${JSON.stringify(wanted.title)} (${wanted.id}) messages not loaded` });
    const active = await count(page, '.conv-item--active');
    assert(active === 1, `exactly one active conv, got ${active}`);
  });

  await suite.check('delete conversation (armed) removes it from the sidebar', async () => {
    const before = await count(page, '.conv-item');
    const delBtn = await page.$('.conv-item .conv-item__delete');
    await armedClick(delBtn);
    await delBtn.dispose();
    await waitFor(async () => (await count(page, '.conv-item')) === before - 1, { message: 'conversation not removed' });
  });

  await suite.check('mobile 390px: Chats toggle reveals conversations, no overflow', async () => {
    await ctx.setViewport(390, 780);
    await ctx.open('#/chat');
    await page.waitForSelector('.chat');
    // the convs pane is off-canvas until toggled
    await clickByText(page, '.chat__bar button', 'Chats');
    await waitFor(async () => page.evaluate(() =>
      document.querySelector('.chat').classList.contains('chat--convs-open')),
      { message: 'convs pane did not open' });
    assert(await noHorizontalOverflow(page), 'horizontal overflow at 390px');
    await ctx.setViewport(1280, 900);
  });

  // ---- capability gating, thinking wiring, image attach/round-trip --------
  // The mobile check's ctx.open() reload just above reset localStorage settings
  // to the default seed (small max_tokens, no thinking/temperature overrides)
  // and the viewport back to desktop, so this section starts from a clean
  // baseline. Every check here that GENERATES seeds its own fresh conversation
  // (see the ORDER COUPLING note above) so the terse system prompt stamped on
  // the suite's original conversation never reaches these generations.
  const THINK_BTN = '.chat__composer button[aria-label="Toggle thinking"]';

  // Switching models can present a pre-switch warning when the switch would
  // LOSE something: incompatible history media naming what gets dropped.
  // (Load cost no longer warns -- v1.62.3 made it a disclosure, so a switch
  // to an unloaded-but-compatible target commits silently.) The switch does
  // not COMMIT until confirmed, so cap-gated UI deliberately keeps tracking
  // the old model while the warning is up. Tests that switch act like a user
  // who means it: confirm when asked.
  async function selectModelConfirming(id) {
    await page.select(MODEL_SELECT, id);
    const confirmed = await page.evaluate(() => {
      const btn = [...document.querySelectorAll('.chat__switch-actions button')]
        .find((b) => b.textContent.trim() === 'Switch anyway');
      if (btn) { btn.click(); return true; }
      return false;
    });
    if (confirmed) {
      await waitFor(async () => (await count(page, '.chat__switch-warning')) === 0,
        { timeout: 5000, message: 'switch warning never cleared after confirm' });
    }
  }

  await closeDrawer(page); // defensive: a prior failure could have left it open

  await suite.check('capability gating: thinking toggle and vision_tokens track the selected model', async () => {
    // Selecting a model in the chat bar is pure metadata (fillModelSelect +
    // the change listener) -- it does NOT load the model, so probing an
    // unloaded model's gating here is cheap and safe.
    const models = await page.evaluate(async () => (await (await fetch('/v1/models')).json()).data ?? []);
    const hasCap = (m, cap) => (m.capabilities ?? []).includes(cap);

    await page.select(MODEL_SELECT, config.model);
    const posThinkHidden = await page.$eval(THINK_BTN, (b) => b.hidden);
    assert(posThinkHidden === false, `thinking toggle hidden for ${config.model} (expected thinking-capable per E2E config)`);
    await openDrawer(page);
    const posVision = await page.$('#set-vision_tokens');
    assert(posVision, `#set-vision_tokens absent for ${config.model} (expected vision-capable per E2E config)`);
    const [min, max] = await page.$eval('#set-vision_tokens', (el) => [el.min, el.max]);
    assert(min === '16' && max === '16384', `#set-vision_tokens min/max = ${min}/${max}, expected 16/16384`);
    await closeDrawer(page);

    // Prefer a model missing BOTH caps for the strongest negative signal;
    // fall back to one missing either.
    const negative = models.find((m) => m.id !== config.model && !hasCap(m, 'thinking') && !hasCap(m, 'vision'))
      ?? models.find((m) => m.id !== config.model && (!hasCap(m, 'thinking') || !hasCap(m, 'vision')));

    if (!negative) {
      console.log('      no model in /v1/models lacks thinking and/or vision -- skipping the negative half of this check');
      return;
    }
    const negCaps = negative.capabilities ?? [];
    console.log(`      negative model: ${negative.id} (capabilities: ${negCaps.join(', ') || 'none'})`);

    // The model select lives in #app, which is inert while the drawer is
    // open -- change the model with the drawer CLOSED, then open it to read
    // the (force-rebuilt) panel, matching the drawer's actual re-render gate
    // (it only rebuilds while open).
    await selectModelConfirming(negative.id);
    if (!hasCap(negative, 'thinking')) {
      const negThinkHidden = await page.$eval(THINK_BTN, (b) => b.hidden);
      assert(negThinkHidden === true, `thinking toggle still visible for non-thinking model ${negative.id}`);
    }
    await openDrawer(page);
    if (!hasCap(negative, 'vision')) {
      const negVision = await page.$('#set-vision_tokens');
      assert(!negVision, `#set-vision_tokens still present for non-vision model ${negative.id}`);
    }
    await closeDrawer(page);

    // restore the capable model for the checks below
    await selectModelConfirming(config.model);
  });

  await suite.check('model switch warns before committing and Cancel reverts', async () => {
    // G3: switching to a model that cannot read this conversation is a
    // decision, not a discovery -- the warning must appear BEFORE the switch
    // commits, and Cancel must put the select back on the committed model.
    // NB this suite's conversation is often text-only, in which case there is
    // nothing to lose and the silent-switch branch below is the real path;
    // the loss warning itself is exercised by the image checks above.
    const models = await page.evaluate(async () => (await (await fetch('/v1/models')).json()).data ?? []);
    const other = models.find((m) => m.id !== config.model);
    if (!other) {
      console.log('      only one model configured -- skipping');
      return;
    }
    await page.select(MODEL_SELECT, other.id);
    if ((await count(page, '.chat__switch-warning')) === 0) {
      // Nothing in this conversation the target cannot read: a clean switch
      // commits silently by design (residency no longer factors in).
      await selectModelConfirming(config.model);
      return;
    }
    await clickByText(page, '.chat__switch-actions button', 'Cancel');
    const val = await page.$eval(MODEL_SELECT, (el) => el.value);
    assert(val === config.model, `Cancel did not revert the select (now on ${val})`);
    assert((await count(page, '.chat__switch-warning')) === 0, 'warning still up after Cancel');
  });

  await suite.check('vision_tokens control round-trips through localStorage', async () => {
    await page.select(MODEL_SELECT, config.model);
    await openDrawer(page);
    await page.waitForSelector('#set-vision_tokens', { timeout: 5000 });
    await page.evaluate(() => {
      const el = document.querySelector('#set-vision_tokens');
      el.value = '512';
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await waitFor(async () => (await ctx.readSettings()).vision_tokens === 512,
      { message: 'vision_tokens=512 never landed in localStorage' });
    // clear back to the cascade default so it doesn't leak into later generations
    await page.evaluate(() => {
      const el = document.querySelector('#set-vision_tokens');
      el.value = '';
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await waitFor(async () => (await ctx.readSettings()).vision_tokens === null,
      { message: 'vision_tokens never cleared back to null (cascade)' });
    await closeDrawer(page);
  });

  await suite.check('thinking button reflects the model default and writes an explicit value', async () => {
    // v1.79.62: thinking is a tri-state. Unset follows the server's answer
    // for the model (`thinking_default` on the /v1/models row: ON for a
    // thinking-capable model since v1.79.62), the drawer's "Model default"
    // option names that value, and the composer button shows the EFFECTIVE
    // state -- each tap writes an explicit true/false to the conversation's
    // params, which is what generate builds from.
    await page.select(MODEL_SELECT, config.model);
    const convId = await newFreshConversation(page);
    await page.waitForSelector(THINK_BTN, { timeout: 5000 });
    const models = await page.evaluate(async () => (await (await fetch('/v1/models')).json()).data ?? []);
    const def = models.find((m) => m.id === config.model)?.thinking_default;
    assert(typeof def === 'boolean', `/v1/models row for ${config.model} carries no boolean thinking_default`);
    const pressed = async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'true';
    const convParams = () => page.evaluate(async (id) => {
      const conv = await (await fetch(`/v1/conversations/${id}`)).json();
      return conv.params ?? {};
    }, convId);

    // Start from "Model default": an earlier check may have left an explicit
    // value in the panel cache, and a fresh conversation snapshots it.
    await openDrawer(page);
    await page.evaluate(() => {
      const sel = document.querySelector('#set-enable_thinking');
      if (!(sel instanceof HTMLSelectElement)) throw new Error('the thinking control is not the tri-state select');
      sel.value = '';
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    });
    const label = await page.$eval('#set-enable_thinking option[value=""]', (o) => o.textContent);
    assert(label === `Model default (${def ? 'on' : 'off'})`,
      `the default option reads ${JSON.stringify(label)} while thinking_default is ${def}`);
    await closeDrawer(page);
    await waitFor(async () => !('enable_thinking' in await convParams()),
      { message: 'clearing the thinking value never landed server-side' });
    assert((await pressed()) === def, `button pressed=${await pressed()} while the model default is ${def}`);

    await page.click(THINK_BTN);
    await waitFor(async () => (await pressed()) === !def, { message: 'thinking button did not flip' });
    // the PUT is debounced -- wait for the store to show it before sending
    await waitFor(async () => (await convParams()).enable_thinking === !def,
      { message: 'the explicit thinking value never landed server-side' });

    // The generate wire itself must stay sampler-free: nothing cap-gated
    // can ride it, which is the structural close of the old leak class.
    const genBodies = [];
    const captureGen = (req) => {
      if (req.method() === 'POST' && req.url().includes('/generate')) genBodies.push(req.postData());
    };
    page.on('request', captureGen);
    await sendText(page, 'Say hi in one word.');
    await waitFor(async () => (await conversationStateById(page, convId)).userCount === 1,
      { message: 'user message never persisted' });
    await waitIdle(page);
    page.off('request', captureGen);
    assert(genBodies.length > 0, 'no /generate request captured');
    const genBody = JSON.parse(genBodies.at(-1));
    // v1.67.0 contract: the TOP-LEVEL body stays sampler-free; the panel's
    // live intent rides in `overrides`, so the explicit value must be there.
    assert(!('enable_thinking' in genBody),
      `enable_thinking rode the top-level generate body: ${genBodies.at(-1)}`);
    assert(genBody.overrides?.enable_thinking === !def,
      `overrides did not carry the explicit value: ${genBodies.at(-1)}`);

    // flip back: a second explicit value, the model default's own
    await page.click(THINK_BTN);
    await waitFor(async () => (await pressed()) === def, { message: 'thinking button did not flip back' });
    await waitFor(async () => (await convParams()).enable_thinking === def,
      { message: 'the second explicit value never landed server-side' });
    // Leave the suite's baseline (thinking OFF, see browser.mjs open()) for
    // the content-asserting checks that follow.
    await openDrawer(page);
    await page.evaluate(() => {
      const sel = document.querySelector('#set-enable_thinking');
      sel.value = 'off';
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await closeDrawer(page);
    await waitFor(async () => (await convParams()).enable_thinking === false,
      { message: 'restoring thinking off never landed server-side' });
  });

  await suite.check('the composer eye button previews the next prompt', async () => {
    // v1.79.62: the exact string the next Send would feed the model, rendered
    // by the model's own engine, with the draft as the next user turn and
    // special tokens highlighted. Persists nothing; needs the model resident
    // (it is -- the harness loaded it).
    await page.select(MODEL_SELECT, config.model);
    await newFreshConversation(page);
    await page.type(COMPOSER, 'draft text for the preview');
    await page.click('.chat__composer button[aria-label="Preview prompt"]');
    await page.waitForSelector('.chat__prompt-preview .prompt-preview__text', { timeout: 20000 });
    const rendered = await textOf(page, '.chat__prompt-preview .prompt-preview__text');
    assert(rendered.includes('draft text for the preview'), 'the draft was not rendered as the next user turn');
    assert((await count(page, '.chat__prompt-preview mark.prompt-preview__tok')) > 0,
      'no special tokens highlighted in the render');
    const head = await textOf(page, '.chat__prompt-preview .prompt-preview__head');
    assert(head.includes(config.model), `preview head does not name the model: ${JSON.stringify(head)}`);
    await clickByText(page, '.chat__prompt-preview button', 'Close');
    assert(await page.$eval('.chat__prompt-preview', (el) => el.hidden), 'preview did not close');
    await page.evaluate(() => {
      const ta = document.querySelector('.chat__composer textarea');
      ta.value = '';
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    });
  });

  await suite.check('cap-gated params are dropped for a model lacking the capability', async () => {
    // The settings cache legitimately KEEPS enable_thinking/vision_tokens
    // when the panel hides their controls (switch back and they return);
    // what must not happen is those values riding a request to a model
    // that lacks the cap. Phase 2 moved the gate SERVER-side
    // (conversation_generate_api._CAP_GATED, unit-pinned): the client wire
    // carries no sampler keys at all. This check pins that structural
    // property on the wire the browser actually sends.
    const models = await page.evaluate(async () => (await (await fetch('/v1/models')).json()).data ?? []);
    const negative = models.find((m) => m.id !== config.model && !(m.capabilities ?? []).includes('thinking'));
    if (!negative) {
      console.log('      no non-thinking model registered -- skipping the cap-filter wire check');
      return;
    }

    // arm the pin: thinking ON while the capable model is selected
    await page.select(MODEL_SELECT, config.model);
    await newFreshConversation(page);
    await waitFor(async () => page.evaluate(() =>
      document.querySelector('.chat__composer button[aria-label="Toggle thinking"]')?.hidden === false),
    { message: 'thinking toggle never visible on the capable model' });
    if ((await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) !== 'true') {
      await page.click(THINK_BTN);
    }
    await waitFor(async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'true',
      { message: 'thinking toggle did not arm' });

    await page.select(MODEL_SELECT, negative.id);
    // Intercept + ABORT the generate request: only the request SHAPE is
    // under test -- the (unloaded) negative model must never actually load.
    await page.setRequestInterception(true);
    const bodies = [];
    const intercept = (req) => {
      if (req.method() === 'POST' && req.url().includes('/generate')) {
        bodies.push(req.postData());
        req.abort();
      } else {
        req.continue();
      }
    };
    page.on('request', intercept);
    try {
      await sendText(page, 'Hello.');
      await waitFor(async () => bodies.length > 0, { message: 'no generate request captured' });
    } finally {
      page.off('request', intercept);
      await page.setRequestInterception(false);
    }
    const body = JSON.parse(bodies.at(-1));
    // v1.67.0 contract: sampler keys never ride top-level; the panel
    // snapshot rides in `overrides`, CLIENT-cap-filtered for the selected
    // model -- so on a non-thinking/non-vision target the gated keys must
    // be absent even from overrides (and the server gate behind that is
    // unit-pinned in test_conversation_generate.py::TestCapGating).
    for (const key of ['enable_thinking', 'vision_tokens', 'temperature', 'max_tokens']) {
      assert(!(key in body), `${key} rode the top-level generate body: ${bodies.at(-1)}`);
    }
    for (const key of ['enable_thinking', 'vision_tokens']) {
      assert(!(body.overrides && key in body.overrides),
        `cap-gated ${key} rode overrides to ${negative.id}: ${bodies.at(-1)}`);
    }
    assert(body.overrides?.model === negative.id,
      `overrides.model should pin the selected target: ${bodies.at(-1)}`);
    // the aborted stream surfaces as a failed generation on this throwaway
    // conversation -- expected; wait for the composer to release
    await waitIdle(page);

    // restore: capable model + toggle off (the cache still holds true --
    // that persistence is the FEATURE half of this behavior)
    await page.select(MODEL_SELECT, config.model);
    await waitFor(async () => page.evaluate(() =>
      document.querySelector('.chat__composer button[aria-label="Toggle thinking"]')?.hidden === false),
    { message: 'toggle did not return on the capable model' });
    if ((await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'true') {
      await page.click(THINK_BTN);
    }
    await waitFor(async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'false',
      { message: 'thinking toggle did not disarm' });
    await waitFor(async () => (await ctx.readSettings()).enable_thinking !== true,
      { message: 'enable_thinking still true in localStorage after restore' });
  });

  await suite.check('thinking block renders in the UI when the model produces thinking content', async () => {
    // NO reload for the token budget: a reload's localStorage seed is dead on
    // arrival -- setup auto-selects the newest conversation and
    // hydrateDocParams replaces the seeded cache with that conversation's
    // stored params (per-document params win by design; seen live
    // 2026-07-23). Instead, seed a fresh conversation and raise Max tokens
    // through the PANEL, which PUTs to that conversation's params for real.
    await page.select(MODEL_SELECT, config.model);
    const convId = await newFreshConversation(page);
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', String(STOP_TEST_MAX_TOKENS));
    await closeDrawer(page);

    // MAKE the toggle state, don't assume it: prior state can legally be
    // either (hydration may have resurrected an earlier true).
    await waitFor(async () => page.evaluate(() =>
      document.querySelector('.chat__composer button[aria-label="Toggle thinking"]')?.hidden === false),
    { message: 'thinking toggle never became visible for the capable model' });
    if ((await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) !== 'true') {
      await page.click(THINK_BTN);
    }
    await waitFor(async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'true',
      { message: 'thinking toggle did not turn on' });

    await sendText(page, 'What is 17 times 24? Reason through the multiplication step by step, then give the final number.');
    // Wait on the STREAM finishing, not on a persisted row. With thinking on
    // at temperature 1.0 this model sometimes spends the entire token budget
    // inside the thinking block and emits no content at all (measured
    // 2026-08-07: 1 run in 6, finish_reason=length, 1513 chars of thinking and
    // 0 of content). finishStream deliberately drops an empty completion, so
    // nothing ever persists and waiting on lastAssistant times out -- a red
    // bar for legal model output, which is exactly what the README says this
    // suite must not do.
    await waitIdle(page);
    // finishStream releases the Send button BEFORE it awaits the save, so a
    // bounded poll separates "not saved yet" from "never saved".
    let state = await conversationStateById(page, convId);
    if (!state.lastAssistant) {
      try {
        await waitFor(async () => (await conversationStateById(page, convId)).lastAssistant !== null,
          { timeout: 5000 });
        state = await conversationStateById(page, convId);
      } catch { /* nothing arrived: the empty-completion case below */ }
    }
    if (!state.lastAssistant) {
      console.log('      model produced thinking but no content (legal, ~1 run in 6) -- nothing to persist, UI half skipped');
    } else if (!state.lastAssistant.thinking) {
      // Empty/absent thinking is legal model output (README: never flake on
      // it) -- the request/persistence pipeline is what this half proves.
      console.log('      model produced no thinking content for this prompt (legal) -- pipeline half verified, UI half skipped');
    } else {
      await waitFor(async () => (await count(page, '.thinking')) >= 1,
        { message: 'assistant persisted non-empty thinking but no .thinking block rendered' });
      const body = await textOf(page, '.thinking .thinking__body');
      assert(body && body.trim().length > 0, '.thinking__body rendered empty despite a non-empty persisted thinking field');
    }

    // Restore: toggle off + Max tokens back to the fast default. No reload --
    // the settings cache updates synchronously, and the next generating check
    // snapshots the cache at conversation-create time.
    await page.click(THINK_BTN);
    await waitFor(async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'false',
      { message: 'thinking toggle did not turn back off' });
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', String(config.maxTokens));
    await closeDrawer(page);
  });

  await suite.check('stop mid-thought, preview the resume, then Save & Continue resumes the same trace', async () => {
    // v1.79.62-64 end to end through the real UI. A reply stopped inside its
    // thinking persists as a thinking-only row; the editor's Preview prompt
    // shows the engine's own render ending inside the OPEN thinking block;
    // Save & Continue resumes THAT trace -- prefix exactly once, then more,
    // with the seam space intact -- instead of starting a second thought.
    await page.select(MODEL_SELECT, config.model);
    const convId = await newFreshConversation(page);
    await openDrawer(page);
    await setSettingsInput(page, 'Max tokens', String(STOP_TEST_MAX_TOKENS));
    await page.evaluate(() => {
      const sel = document.querySelector('#set-enable_thinking');
      sel.value = 'on';
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await closeDrawer(page);
    await waitFor(async () => page.evaluate(async (id, cap) => {
      const p = (await (await fetch(`/v1/conversations/${id}`)).json()).params ?? {};
      return p.max_tokens === cap && p.enable_thinking === true;
    }, convId, STOP_TEST_MAX_TOKENS), { message: 'params never landed server-side' });
    const restore = async () => {
      // back to the suite's baseline: fast cap, thinking explicitly OFF
      await openDrawer(page);
      await setSettingsInput(page, 'Max tokens', String(config.maxTokens));
      await page.evaluate(() => {
        const sel = document.querySelector('#set-enable_thinking');
        sel.value = 'off';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await closeDrawer(page);
    };

    await sendText(page, 'Reason step by step, at length, about why the sky is blue before you answer.');
    await waitFor(async () => {
      const t = await textOf(page, '.message--streaming .thinking__body');
      return Boolean(t && t.trim().length > 60);
    }, { message: 'no thinking streamed before the stop', timeout: 60000 });
    await page.click(SEND_BTN); // Stop
    await waitFor(async () => /stopped/i.test((await textOf(page, '.chat__status')) || ''),
      { message: 'stop status never appeared' });
    let before = null;
    await waitFor(async () => {
      before = (await conversationStateById(page, convId)).lastAssistant;
      return Boolean(before?.thinking);
    }, { message: 'the stopped reply did not persist its thinking' });
    if ((before.content ?? '').trim()) {
      // Legal, but the resume half did not run: report it as such rather
      // than as a pass. On a fast model the stop can land after the block
      // closes; rerun (or lower STOP_TEST_MAX_TOKENS) to reach the resume.
      await restore();
      skip('the stop landed after the thinking block closed; the resume half did not run');
    }
    const prefix = before.thinking;
    const tail = prefix.trimEnd().slice(-40);

    await clickByText(page, '.message--assistant .message__actions button', 'Edit');
    await page.waitForSelector('.message-edit__thinking');
    await clickByText(page, '.message-edit__buttons button', 'Preview prompt');
    await page.waitForSelector('.message-edit__preview .prompt-preview__text', { timeout: 20000 });
    const head = await textOf(page, '.message-edit__preview .prompt-preview__head');
    const rendered = await textOf(page, '.message-edit__preview .prompt-preview__text');
    assert(/resumes inside/i.test(head), `preview head reads ${JSON.stringify(head)}`);
    assert(rendered.trimEnd().endsWith(tail),
      `the rendered prompt does not end inside the stopped thinking: ...${JSON.stringify(rendered.slice(-80))}`);

    await clickByText(page, '.message-edit__buttons button', 'Save & Continue');
    await waitFor(async () => {
      const s = await conversationStateById(page, convId);
      const t = s.lastAssistant?.thinking ?? '';
      return s.lastAssistant?.id === before.id && t.startsWith(prefix) && t.length > prefix.length;
    }, { message: 'continuation did not extend the same thinking in place', timeout: 120000 });
    await waitIdle(page, 180000);
    const after = (await conversationStateById(page, convId)).lastAssistant;
    assert(after.thinking.split(prefix.slice(0, 40)).length === 2,
      'the stopped prefix appears more than once -- a second thought was glued on');
    assert((await assistantCount(page)) === 1, 'the resume must not add a message row');
    await restore();
  });

  await suite.check('image attach caps at 8 with an aria-live status message', async () => {
    await page.waitForSelector('.chat__composer input[type="file"]', { timeout: 5000 });
    await page.evaluate(async () => {
      const canvas = document.createElement('canvas');
      canvas.width = 8;
      canvas.height = 8;
      const c2d = canvas.getContext('2d');
      const files = [];
      for (let i = 0; i < 9; i++) {
        c2d.fillStyle = `rgb(${(i * 25) % 256},${(255 - i * 25) % 256},${(i * 10) % 256})`;
        c2d.fillRect(0, 0, 8, 8);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        files.push(new File([blob], `attach-cap-${i}.png`, { type: 'image/png' }));
      }
      const dt = new DataTransfer();
      for (const f of files) dt.items.add(f);
      const input = document.querySelector('.chat__composer input[type="file"]');
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });

    await waitFor(async () => (await count(page, '.attach-thumb')) === 8,
      { message: 'expected exactly 8 attach-thumb elements after attaching 9 images' });
    const status = await textOf(page, '.chat__status');
    // chat.js addImages: `${MAX_ATTACH_IMAGES} image max -- ${overflow} not attached.`
    assert(status && /8 image max/.test(status) && /not attached/i.test(status),
      `status line did not mention the 8-image cap: "${status}"`);

    // Clear all staged images via the per-thumb remove buttons so check 6
    // starts from an empty composer.
    let remaining = await count(page, '.attach-thumb');
    while (remaining > 0) {
      await page.click('.attach-thumb__remove');
      remaining -= 1;
      await waitFor(async () => (await count(page, '.attach-thumb')) === remaining,
        { message: `attach-thumb count did not drop to ${remaining} after a remove click` });
    }
    assert(await page.$eval('.chat__attach', (el) => el.hidden), 'attach strip did not hide once all images were removed');
  });

  await suite.check('pasting an image into the composer stages it', async () => {
    // The paste path is a distinct entry point from the picker (chat.js
    // paste listener filters clipboard items to image files) -- exercise it
    // with a synthetic ClipboardEvent carrying a real File.
    await page.evaluate(async () => {
      const canvas = document.createElement('canvas');
      canvas.width = 8;
      canvas.height = 8;
      const c2d = canvas.getContext('2d');
      c2d.fillStyle = 'rgb(30,180,90)';
      c2d.fillRect(0, 0, 8, 8);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
      const dt = new DataTransfer();
      dt.items.add(new File([blob], 'pasted.png', { type: 'image/png' }));
      const ta = document.querySelector('.chat__composer textarea');
      ta.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }));
    });
    await waitFor(async () => (await count(page, '.attach-thumb')) === 1,
      { message: 'pasted image never staged as a thumbnail' });
    // clear so the round-trip check below stages exactly its own images
    await page.click('.attach-thumb__remove');
    await waitFor(async () => (await count(page, '.attach-thumb')) === 0,
      { message: 'staged pasted image did not clear' });
  });

  await suite.check('an attached image round-trips: send, persist, render, survive reload', async () => {
    await page.select(MODEL_SELECT, config.model);
    const convId = await newFreshConversation(page);

    await page.waitForSelector('.chat__composer input[type="file"]', { timeout: 5000 });
    await page.evaluate(async () => {
      const canvas = document.createElement('canvas');
      canvas.width = 8;
      canvas.height = 8;
      const c2d = canvas.getContext('2d');
      const colors = ['rgb(200,40,40)', 'rgb(40,40,200)'];
      const files = [];
      for (const color of colors) {
        c2d.fillStyle = color;
        c2d.fillRect(0, 0, 8, 8);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        files.push(new File([blob], `roundtrip-${color}.png`, { type: 'image/png' }));
      }
      const dt = new DataTransfer();
      for (const f of files) dt.items.add(f);
      const input = document.querySelector('.chat__composer input[type="file"]');
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await waitFor(async () => (await count(page, '.attach-thumb')) === 2, { message: 'images did not stage for send' });

    await sendText(page, 'Briefly describe these images.');
    await waitFor(async () => (await conversationStateById(page, convId)).lastUser !== null,
      { message: 'user message with images never persisted' });

    const state = await conversationStateById(page, convId);
    const imageBlocks = (state.lastUser.content_blocks ?? []).filter((b) => b.type === 'image');
    assert(imageBlocks.length === 2, `expected 2 image content_blocks server-side, got ${imageBlocks.length}`);

    await waitFor(async () => (await count(page, '.message--user .message-image')) === 2,
      { message: 'user bubble did not render 2 .message-image elements' });

    // The assistant reply may legally be empty-EOS and persist nothing
    // (finishStream) -- only the user-side image round-trip is asserted here.
    await waitIdle(page);

    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.chat');
    await waitFor(async () => (await count(page, '.message--user .message-image')) === 2,
      { message: 'images did not survive a reload (store round-trip)' });
  });
}
