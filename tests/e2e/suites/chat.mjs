// Chat suite: the most-verified surface. Covers streaming, position-based
// edit/regenerate/delete truncation, stop=partial-saved, post-abort health,
// settings + the localStorage sampler seed, conversation CRUD, and a 390px
// mobile pass. Data is cleared by the orchestrator before this runs.

import { assert, waitFor, sleep } from '../lib/harness.mjs';
import { clickByText, armedClick, count, textOf, waitForLabel, settingsInputValue, setSettingsInput, noHorizontalOverflow } from '../lib/dom.mjs';

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

// Client-observed streaming cadence, measured INSIDE the page against the same
// /v1/chat/completions path the app uses. The Phase 1 fix (asyncio.wait instead
// of a 0.1s poll in async_generator_with_abort) is invisible to server-side
// telemetry -- only a client timing this stream can catch a regression back to
// the ~100ms poll ceiling. Returns per-content-delta inter-arrival gaps.
async function measureStreamCadence(page, model, maxTokens) {
  return page.evaluate(async (model, maxTokens) => {
    const marks = [];
    let usage = null;
    const res = await fetch('/v1/chat/completions', {
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
          if (!d || d === '[DONE]') continue;
          let c;
          try { c = JSON.parse(d); } catch { continue; }
          if (c.choices?.[0]?.delta?.content) marks.push(performance.now());
          if (c.usage) usage = c.usage;
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

  await suite.check('app boots with 5 nav routes', async () => {
    await page.waitForSelector('#nav-desktop .nav-item');
    const routes = await page.$$eval('#nav-desktop .nav-item', (els) =>
      [...new Set(els.map((e) => e.dataset.route))]);
    assert(routes.length === 5, `expected 5 routes, got ${routes.join(',')}`);
    assert(['chat', 'notebook', 'explore', 'models', 'perf'].every((r) => routes.includes(r)),
      `missing route in ${routes.join(',')}`);
  });

  await suite.check('default page is chat', async () => {
    const page_ = await page.evaluate(() => document.body.dataset.page);
    assert(page_ === 'chat', `body[data-page]=${page_}`);
    await page.waitForSelector('.chat');
  });

  await suite.check('composer, send button, model select present', async () => {
    await page.waitForSelector(COMPOSER);
    await page.waitForSelector(SEND_BTN);
    await page.waitForSelector(MODEL_SELECT);
    assert((await sendBtnLabel(page)) === 'Send', 'send button label');
  });

  await suite.check('model select contains the E2E model', async () => {
    const opts = await page.$$eval(`${MODEL_SELECT} option`, (els) => els.map((e) => e.value));
    assert(opts.includes(config.model), `model ${config.model} not in [${opts.join(', ')}]`);
    await page.select(MODEL_SELECT, config.model);
  });

  await suite.check('empty state before any conversation', async () => {
    const empty = await textOf(page, '.chat__messages .empty-state');
    assert(empty && empty.length > 0, 'expected empty-state prompt');
  });

  await suite.check('send streams an assistant reply that persists', async () => {
    await page.select(MODEL_SELECT, config.model);
    await sendText(page, 'Say hello in one short sentence.');
    // user bubble shows optimistically
    await waitFor(async () => (await userCount(page)) === 1, { message: 'user bubble' });
    // streaming placeholder appears at some point
    await waitFor(async () => (await sendBtnLabel(page)) === 'Stop', { message: 'never entered streaming' });
    await waitIdle(page);
    assert((await assistantCount(page)) === 1, 'one assistant message');
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
    assert(c.median != null && c.median < CADENCE_MAX_MEDIAN_MS,
      `median inter-chunk gap ${c.median?.toFixed(1)}ms >= ${CADENCE_MAX_MEDIAN_MS}ms (poll-quantization signature is ~100ms)`);
    assert(c.rate > CADENCE_MIN_RATE,
      `client decode rate ${c.rate?.toFixed(1)}/s <= ${CADENCE_MIN_RATE}/s (old poll ceiling was ~10/s)`);
    console.log(`      cadence: ${c.chunks} chunks, median gap ${c.median.toFixed(1)}ms, ${c.rate.toFixed(1)}/s`);
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
    const firstUser = (await page.$$('.message--user'))[0];
    await clickByText(page, '.message--user .message__actions button', 'Edit');
    await page.waitForSelector('.message-edit textarea', { timeout: 5000 });
    // Cancel to restore
    await clickByText(page, '.message-edit__buttons button', 'Cancel');
    await waitFor(async () => (await count(page, '.message-edit')) === 0, { message: 'editor did not close' });
    assert(firstUser, 'had a user message');
  });

  await suite.check('save & regenerate truncates then regenerates', async () => {
    // Edit the FIRST user message -> truncates everything after position, then
    // streams a single fresh assistant reply.
    await clickByText(page, '.message--user .message__actions button', 'Edit');
    await page.waitForSelector('.message-edit textarea');
    await page.click('.message-edit textarea', { clickCount: 3 });
    await page.type('.message-edit textarea', 'Reply with the single word: ready.');
    await clickByText(page, '.message-edit__buttons button', 'Save & Regenerate');
    await waitFor(async () => (await sendBtnLabel(page)) === 'Stop', { message: 'regenerate did not start' });
    await waitIdle(page);
    const u = await userCount(page);
    const a = await assistantCount(page);
    assert(u === 1, `expected 1 user msg after truncation, got ${u}`);
    assert(a === 1, `expected 1 assistant msg, got ${a}`);
  });

  await suite.check('regenerate on an assistant message replaces it', async () => {
    const before = await lastAssistantText(page);
    await clickByText(page, '.message--assistant .message__actions button', 'Regenerate');
    await waitFor(async () => (await sendBtnLabel(page)) === 'Stop', { message: 'regen did not start' });
    await waitIdle(page);
    assert((await assistantCount(page)) === 1, 'still one assistant message');
    assert((await userCount(page)) === 1, 'user message preserved');
    assert(typeof before === 'string', 'had prior text');
  });

  await suite.check('delete (armed) removes a message via truncation', async () => {
    // Delete the assistant message -> truncates from its position, leaving the
    // single user message.
    const delBtn = await page.evaluateHandle(() => {
      const msg = document.querySelector('.message--assistant');
      return [...msg.querySelectorAll('.message__actions button')].find((b) => b.textContent.trim() === 'Delete');
    });
    await armedClick(delBtn.asElement());
    await delBtn.dispose();
    await waitFor(async () => (await assistantCount(page)) === 0, { message: 'assistant not deleted' });
    assert((await userCount(page)) === 1, 'user message remains');
  });

  // ---- stop = partial saved (needs a long generation) --------------------
  await suite.check('stop mid-stream saves the partial reply', async () => {
    await ctx.open('#/chat', { max_tokens: STOP_TEST_MAX_TOKENS });
    await page.select(MODEL_SELECT, config.model);
    // open the existing conversation (first in the list)
    await waitFor(async () => (await count(page, '.conv-item')) >= 1, { message: 'conv list' });
    await page.click('.conv-item');
    await waitFor(async () => (await userCount(page)) >= 1, { message: 'messages loaded' });
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
  });

  await suite.check('partial reply persisted across reload', async () => {
    const before = await assistantCount(page);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.chat');
    await waitFor(async () => (await assistantCount(page)) === before, { message: 'partial lost on reload' });
    assert(before >= 1, 'had a partial reply');
  });

  await suite.check('post-abort health: a new send completes normally', async () => {
    await ctx.open('#/chat'); // back to small max_tokens
    await page.select(MODEL_SELECT, config.model);
    await page.click('.conv-item');
    await waitFor(async () => (await userCount(page)) >= 1, { message: 'messages loaded' });
    const before = await assistantCount(page);
    await sendText(page, 'Reply with one short word.');
    await waitFor(async () => (await sendBtnLabel(page)) === 'Stop', { message: 'did not start' });
    await waitIdle(page);
    assert((await assistantCount(page)) === before + 1, 'new reply not added after prior abort');
  });

  // ---- settings ----------------------------------------------------------
  await suite.check('settings panel toggles open with sampling controls', async () => {
    await clickByText(page, '.chat__bar button', 'Settings');
    await page.waitForSelector('.chat__settings .settings-panel', { timeout: 5000 });
    const labels = await page.$$eval('.settings-panel .settings-row label', (els) => els.map((e) => e.textContent.trim()));
    assert(labels.includes('Temperature'), `labels: ${labels.join(', ')}`);
    assert(labels.includes('Max tokens'), 'no Max tokens control');
  });

  await suite.check('seeded max_tokens is reflected in the settings panel', async () => {
    const val = await settingsInputValue(page, 'Max tokens');
    assert(val === String(config.maxTokens), `max_tokens input="${val}", expected ${config.maxTokens}`);
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
  const SYS_PROMPT = 'Answer in exactly one word.';

  // One owner for the "find a preset <option> by its label" lookup.
  const presetOptionValue = (name) => page.evaluate((n) =>
    [...document.querySelectorAll('.preset-row select option')]
      .find((o) => o.textContent === n)?.value ?? null, name);

  await suite.check('system prompt edit persists to the conversation', async () => {
    // settings panel is open from the checks above
    await page.evaluate(() => { document.querySelector('.chat__sysprompt').open = true; });
    await page.click('.chat__sysprompt-input');
    await page.type('.chat__sysprompt-input', SYS_PROMPT);
    await page.click(COMPOSER); // blur -> change -> PUT
    await waitFor(async () => page.evaluate(async (sys) => {
      const res = await fetch('/v1/conversations');
      const { conversations } = await res.json();
      return conversations.some((c) => c.system_prompt === sys);
    }, SYS_PROMPT), { message: 'system prompt not saved server-side' });
  });

  await suite.check('preset save + apply round-trips sampler state', async () => {
    await setSettingsInput(page, 'Temperature', '0.31');
    await page.click('.preset-section .input');
    await page.type('.preset-section .input', 'e2e-preset');
    await clickByText(page, '.preset-row button', 'Save');
    await waitFor(async () => (await presetOptionValue('e2e-preset')) !== null,
      { message: 'saved preset not listed in the select' });
    // drift the panel, then re-apply the preset and expect the pin back
    await setSettingsInput(page, 'Temperature', '1.9');
    await page.select('.preset-row select', await presetOptionValue('e2e-preset'));
    await waitFor(async () => (await settingsInputValue(page, 'Temperature')) === '0.31',
      { message: 'applying the preset did not restore temperature' });
    // back to cascade so nothing leaks into later generations
    await setSettingsInput(page, 'Temperature', '');
  });

  await suite.check('preset delete (armed) removes it from the select', async () => {
    const delBtn = await page.$('.preset-section .btn--ghost');
    await armedClick(delBtn);
    await delBtn.dispose();
    await waitFor(async () => (await presetOptionValue('e2e-preset')) === null,
      { message: 'deleted preset still listed' });
  });

  // ---- conversation management -------------------------------------------
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
    // the second item is the older one with messages
    await items[1].click();
    await waitFor(async () => (await userCount(page)) >= 1, { message: 'older conv messages not loaded' });
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
}
