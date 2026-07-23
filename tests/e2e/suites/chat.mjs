// Chat suite: the most-verified surface. Covers streaming, position-based
// edit/regenerate/delete truncation, stop=partial-saved, post-abort health,
// settings + the localStorage sampler seed, conversation CRUD, and a 390px
// mobile pass. Data is cleared by the orchestrator before this runs.

import { assert, waitFor } from '../lib/harness.mjs';
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
  return page.evaluate(async () => {
    const { conversations } = await (await fetch('/v1/conversations')).json();
    if (!conversations.length) return { userCount: 0, assistantCount: 0, lastAssistantId: null };
    const conv = await (await fetch(`/v1/conversations/${conversations[0].id}`)).json();
    const msgs = conv.messages ?? [];
    return {
      userCount: msgs.filter((m) => m.role === 'user').length,
      assistantCount: msgs.filter((m) => m.role === 'assistant').length,
      lastAssistantId: msgs.filter((m) => m.role === 'assistant').at(-1)?.id ?? null,
    };
  });
}

// By-id reader for the capability/thinking/image section below, where several
// checks each seed their OWN fresh conversation -- conversationStateServerSide's
// "conversations[0] is the one conversation this suite has" assumption no
// longer holds once more than one exists, so reads there target an explicit id.
async function conversationStateById(page, id) {
  return page.evaluate(async (convId) => {
    const conv = await (await fetch(`/v1/conversations/${convId}`)).json();
    const msgs = conv.messages ?? [];
    return {
      userCount: msgs.filter((m) => m.role === 'user').length,
      assistantCount: msgs.filter((m) => m.role === 'assistant').length,
      lastUser: msgs.filter((m) => m.role === 'user').at(-1) ?? null,
      lastAssistant: msgs.filter((m) => m.role === 'assistant').at(-1) ?? null,
    };
  }, id);
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
  const presetOptionValue = (name) => page.evaluate((n) =>
    [...document.querySelectorAll('.preset-row select option')]
      .find((o) => o.textContent === n)?.value ?? null, name);

  await suite.check('system prompt edit persists without blur', async () => {
    // drawer is open from the checks above; the sysprompt editor is one of chat's
    // contributed sections inside it (its details is always expanded now).
    await page.click('.chat__sysprompt-input');
    await page.type('.chat__sysprompt-input', SYS_PROMPT);
    // Deliberately NO blur: state commits per keystroke and the PUT is
    // debounced -- the old blur-only commit is the bug this regression guards.
    await waitFor(async () => page.evaluate(async (sys) => {
      const res = await fetch('/v1/conversations');
      const { conversations } = await res.json();
      return conversations.some((c) => c.system_prompt === sys);
    }, SYS_PROMPT), { message: 'system prompt not saved server-side' });
  });

  await suite.check('sysprompt typed text survives Escape-close', async () => {
    // Escape with focus still in the textarea removes the field before any
    // change event can fire -- the exact path that used to lose the prompt.
    await page.click('.chat__sysprompt-input');
    await page.type('.chat__sysprompt-input', ' Be terse.');
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.drawer--open'), { timeout: 5000 });
    await waitFor(async () => page.evaluate(async () => {
      const res = await fetch('/v1/conversations');
      const { conversations } = await res.json();
      return conversations.some((c) => (c.system_prompt ?? '').includes('Be terse.'));
    }), { message: 'text typed before Escape-close never reached the server' });
    await openDrawer(page);  // leave the drawer open for the preset checks
    const val = await page.$eval('.chat__sysprompt-input', (el) => el.value);
    assert(val.includes('Be terse.'), 'reopened drawer lost the typed text');
  });

  await suite.check('preset save + apply round-trips sampler state', async () => {
    await setSettingsInput(page, 'Temperature', '0.31');
    await page.click('.preset-section .input');
    await page.type('.preset-section .input', 'e2e-preset');
    await clickByText(page, '.preset-row button', 'Save');
    await waitFor(async () => (await presetOptionValue('e2e-preset')) !== null,
      { message: 'saved preset not listed in the select' });
    // fresh save selects the preset and matches by construction
    await waitFor(async () => (await driftText(page))?.includes('Matches'),
      { message: 'drift line not "Matches" right after save' });
    // drift the panel: the line must flip live, and selection alone must NOT
    // have touched the panel (apply is an explicit button now)
    await setSettingsInput(page, 'Temperature', '1.9');
    await waitFor(async () => (await driftText(page))?.includes('Differs'),
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

  await suite.check('applied-preset chip shows in the chat bar', async () => {
    // The prior check saved+applied e2e-preset, then reset Temperature to
    // cascade -- the panel is drifted, so the chip must carry "(edited)".
    const txt = await page.$eval('.chat__preset-chip', (el) => (el.hidden ? null : el.textContent));
    assert(txt?.includes('e2e-preset'), `chip="${txt}"`);
    assert(txt.includes('(edited)'), `chip should be marked (edited), got "${txt}"`);
  });

  await suite.check('preset delete (armed) removes it from the select', async () => {
    const delBtn = await page.$('.preset-section .btn--ghost');
    await armedClick(delBtn);
    await delBtn.dispose();
    await waitFor(async () => (await presetOptionValue('e2e-preset')) === null,
      { message: 'deleted preset still listed' });
    await waitFor(async () => page.$eval('.chat__preset-chip', (el) => el.hidden),
      { message: 'applied-preset chip did not clear after delete' });
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

  // ---- capability gating, thinking wiring, image attach/round-trip --------
  // The mobile check's ctx.open() reload just above reset localStorage settings
  // to the default seed (small max_tokens, no thinking/temperature overrides)
  // and the viewport back to desktop, so this section starts from a clean
  // baseline. Every check here that GENERATES seeds its own fresh conversation
  // (see the ORDER COUPLING note above) so the terse system prompt stamped on
  // the suite's original conversation never reaches these generations.
  const THINK_BTN = '.chat__composer button[aria-label="Toggle thinking"]';

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
    await page.select(MODEL_SELECT, negative.id);
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
    await page.select(MODEL_SELECT, config.model);
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

  await suite.check('thinking toggle wires enable_thinking (absent when off, true when on)', async () => {
    await page.select(MODEL_SELECT, config.model);
    const convId = await newFreshConversation(page);
    await page.waitForSelector(THINK_BTN, { timeout: 5000 });
    assert((await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'false',
      'thinking toggle already pressed at the start of this check');

    // Capture every POST to /v1/chat/completions this send makes -- streamChat
    // retries once on a 503 model_overloaded, so the request actually SENT is
    // the LAST one captured, not necessarily the first.
    const offBodies = [];
    const captureOff = (req) => {
      if (req.method() === 'POST' && req.url().includes('/v1/chat/completions')) offBodies.push(req.postData());
    };
    page.on('request', captureOff);
    await sendText(page, 'Say hi in one word.');
    await waitFor(async () => (await conversationStateById(page, convId)).userCount === 1,
      { message: 'off-toggle user message never persisted' });
    await waitIdle(page);
    page.off('request', captureOff);
    assert(offBodies.length > 0, 'no /v1/chat/completions request captured with thinking off');
    const offBody = JSON.parse(offBodies.at(-1));
    assert(!('enable_thinking' in offBody), `enable_thinking present while toggle is off: ${JSON.stringify(offBody.enable_thinking)}`);

    await page.click(THINK_BTN);
    await waitFor(async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'true',
      { message: 'thinking toggle did not flip to pressed' });

    const onBodies = [];
    const captureOn = (req) => {
      if (req.method() === 'POST' && req.url().includes('/v1/chat/completions')) onBodies.push(req.postData());
    };
    page.on('request', captureOn);
    await sendText(page, 'Say hi in one word again.');
    await waitFor(async () => (await conversationStateById(page, convId)).userCount === 2,
      { message: 'on-toggle user message never persisted' });
    await waitIdle(page);
    page.off('request', captureOn);
    assert(onBodies.length > 0, 'no /v1/chat/completions request captured with thinking on');
    const onBody = JSON.parse(onBodies.at(-1));
    assert(onBody.enable_thinking === true, `enable_thinking not true while toggle is on: ${JSON.stringify(onBody.enable_thinking)}`);

    // toggle back off so it doesn't leak into later checks
    await page.click(THINK_BTN);
    await waitFor(async () => (await page.$eval(THINK_BTN, (b) => b.getAttribute('aria-pressed'))) === 'false',
      { message: 'thinking toggle did not flip back off' });
    // Persisted outcome, not just localStorage: the toggle-off fires a
    // DEBOUNCED params PUT to this conversation; if the next check reloads
    // before it lands, hydration resurrects enable_thinking=true (seen
    // live 2026-07-23). Wait for the server-side params to show it off.
    await waitFor(async () => page.evaluate(async (id) => {
      const conv = await (await fetch(`/v1/conversations/${id}`)).json();
      return (conv.params ?? {}).enable_thinking !== true;
    }, convId), { message: 'off-toggle params PUT never landed server-side' });
  });

  await suite.check('cap-gated params are dropped for a model lacking the capability', async () => {
    // The settings cache legitimately KEEPS enable_thinking/vision_tokens
    // when the panel hides their controls (switch back and they return);
    // what must not happen is those values riding a request to a model
    // that lacks the cap (samplerParams(caps) filter, v1.39.10).
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
    // Intercept + ABORT the completion request: only the request SHAPE is
    // under test -- the (unloaded) negative model must never actually load.
    await page.setRequestInterception(true);
    const bodies = [];
    const intercept = (req) => {
      if (req.method() === 'POST' && req.url().includes('/v1/chat/completions')) {
        bodies.push(req.postData());
        req.abort();
      } else {
        req.continue();
      }
    };
    page.on('request', intercept);
    try {
      await sendText(page, 'Hello.');
      await waitFor(async () => bodies.length > 0, { message: 'no completion request captured' });
    } finally {
      page.off('request', intercept);
      await page.setRequestInterception(false);
    }
    const body = JSON.parse(bodies.at(-1));
    assert(!('enable_thinking' in body),
      `enable_thinking rode a request to non-thinking ${negative.id}: ${JSON.stringify(body.enable_thinking)}`);
    if (!(negative.capabilities ?? []).includes('vision')) {
      assert(!('vision_tokens' in body), `vision_tokens rode a request to non-vision ${negative.id}`);
    }
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
    await waitFor(async () => (await conversationStateById(page, convId)).lastAssistant !== null,
      { timeout: 30000, message: 'assistant reply never persisted' });

    const state = await conversationStateById(page, convId);
    if (!state.lastAssistant.thinking) {
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
