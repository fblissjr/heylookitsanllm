// Pages suite: notebook (autosave + generate-at-cursor tail preservation),
// perf (no-polling proof + ranges),
// models (list/load+warm/
// unload + folder & HF scan + danger-zone clear). Data is cleared by the
// orchestrator before this runs; the danger-zone clear check runs LAST.

import { assert, waitFor, sleep, skip, proveQuiet } from '../lib/harness.mjs';
import { serverGet } from '../lib/server-state.mjs';
import { clickByText, armedClick, count, textOf, waitForLabel, findModelRow, modelRowState, noHorizontalOverflow, openDrawer, closeDrawer, driftText, handleByText } from '../lib/dom.mjs';

// Record requests whose URL matches `regex` from the moment this is called until
// stop(). Used to prove the perf page does NOT poll.
function watchRequests(page, regex) {
  const urls = [];
  const handler = (req) => { if (regex.test(req.url())) urls.push(req.url()); };
  page.on('request', handler);
  return { urls, stop: () => page.off('request', handler) };
}

// Is the beforeunload guard armed? Dispatched at `window`, which is where the
// handler is registered and where the browser fires the real one -- not a
// convenient node that happened to be in scope. What it cannot see is whether
// Chrome paints the dialog; chat.mjs's mid-stream reload observes that end for
// real, by having to accept one.
const unloadGuardArmed = (page) => page.evaluate(() =>
  !window.dispatchEvent(new Event('beforeunload', { cancelable: true })));

// Empty the template editor, which is also how a check hands back a DISARMED
// guard. Leaving one armed is not a tidy-up nit: a later ctx.open() does a real
// reload, puppeteer auto-dismisses the dialog, dismissing CANCELS the
// navigation, and the reload times out -- so an earlier failure resurfaces as
// an unrelated navigation timeout in a different check.
async function clearTemplateBody(page) {
  await page.$eval('.cfg-tmpl__body', (el) => { el.focus(); el.select(); });
  await page.keyboard.press('Backspace');
  await waitFor(async () => (await page.$eval('.cfg-tmpl__body', (el) => el.value)) === '',
    { timeout: 5000, message: 'the template body never cleared' });
}

// Open a model's config panel and its (lazy) template section, resolved.
async function openTemplatePanel(page, modelId) {
  const row = await findModelRow(page, modelId);
  assert(row, `no model row titled ${modelId}`);
  const handle = await row.evaluateHandle((r) =>
    [...r.querySelectorAll('.model-row__actions button')].find((b) => b.textContent.trim() === 'Configure'));
  const btn = handle.asElement();
  assert(btn, `no Configure button on the row for ${modelId}`);
  await btn.click();
  await handle.dispose();
  await page.waitForSelector('.model-config', { timeout: 10000 });
  if (await page.$eval('.cfg-tmpl', (el) => !el.open)) {
    await page.$eval('.cfg-tmpl > summary', (el) => el.click());
  }
  await waitFor(async () => Boolean((await textOf(page, '.cfg-tmpl__origin') || '').trim()),
    { timeout: 10000, message: 'template panel never resolved an origin' });
}

// Server-side notebook state, read straight from the API inside the page --
// used to wait on a PERSISTED autosave instead of sleeping past the debounce
// window (the list view omits `content` for efficiency; use notebookFull for
// that). Assumes a single active notebook (true throughout this suite).
async function notebookListRow(page) {
  const body = await serverGet(page, '/v1/notebooks');
  return body?.notebooks?.[0] ?? null;
}

async function notebookFull(page, id) {
  return serverGet(page, `/v1/notebooks/${id}`);
}

// Stop-check reopens with a large cap so there's a window to click Stop before
// generation finishes on its own (chat suite parity, same constant value).
const STOP_TEST_MAX_TOKENS = 400;

export async function runPagesSuite({ suite, ctx, config }) {
  const { page } = ctx;

  // =========================== NOTEBOOK ==================================
  await ctx.open('#/notebook');

  await suite.check('notebook page mounts', async () => {
    await page.waitForSelector('.notebook');
    // POLL: the empty state renders when the page's async setup lands (the
    // same ~1.7s /v1/models+list window the chat suite's early checks race).
    await waitFor(async () => {
      const empty = await textOf(page, '.notebook__empty');
      return Boolean(empty && empty.length > 0);
    }, { message: 'notebook empty-state never appeared' });
  });

  await suite.check('New notebook creates an entry and opens the editor', async () => {
    await clickByText(page, '.notebook__list-head button', 'New');
    await waitFor(async () => (await count(page, '.notebook-item')) === 1, { message: 'notebook not created' });
    await page.waitForSelector('.notebook__form', { timeout: 5000 });
    await page.waitForSelector('.notebook__content');
  });

  await suite.check('title autosaves and survives reload', async () => {
    await page.click('.notebook__title');
    // See the note at the config-editor inputs below: puppeteer 25 changed
    // triple-click, so select-all goes through the native input API.
    await page.$eval('.notebook__title', (el) => el.select());
    await page.type('.notebook__title', 'Ocean Notes');
    // Outcome-based: wait for the debounced PUT to actually land server-side
    // (list view carries title) before reloading, rather than sleeping past
    // the nominal 500ms debounce window (F: condition exists, don't sleep).
    await waitFor(async () => (await notebookListRow(page))?.title === 'Ocean Notes',
      { message: 'title not saved server-side before reload' });
    await ctx.open('#/notebook');
    await page.waitForSelector('.notebook__title');
    await waitFor(async () => (await page.$eval('.notebook__title', (e) => e.value)) === 'Ocean Notes',
      { message: 'title not persisted' });
  });

  await suite.check('content autosaves and survives reload', async () => {
    await page.click('.notebook__content');
    await page.type('.notebook__content', 'The sea is wide.');
    // Same outcome-based wait as the title check above -- content is omitted
    // from the list view, so read the full record.
    const row = await notebookListRow(page);
    await waitFor(async () => (await notebookFull(page, row.id))?.content?.includes('The sea is wide.'),
      { message: 'content not saved server-side before reload' });
    await ctx.open('#/notebook');
    await page.waitForSelector('.notebook__content');
    await waitFor(async () => (await page.$eval('.notebook__content', (e) => e.value)).includes('The sea is wide.'),
      { message: 'content not persisted' });
  });

  await suite.check('generate-at-cursor preserves the tail after the insertion point', async () => {
    await page.select('.notebook__model', config.model);
    await page.$eval('.notebook__content', (el) => {
      el.value = 'HEAD_MARKER\n\nTAIL_MARKER';
      el.dispatchEvent(new Event('input', { bubbles: true }));
      const pos = 'HEAD_MARKER\n\n'.length;
      el.setSelectionRange(pos, pos);
      el.focus();
    });
    await clickByText(page, '.notebook__actions button', 'Generate');
    await waitForLabel(page, '.notebook__actions button', 'Stop', { message: 'generation did not start' });
    await waitForLabel(page, '.notebook__actions button', 'Generate', { timeout: 30000, message: 'generation did not finish' });
    const value = await page.$eval('.notebook__content', (e) => e.value);
    // The claim under test is head/tail PRESERVATION, which the pipeline must
    // honor regardless of what the model produced. An immediate-EOS empty
    // completion is a legal model outcome (same lesson as the chat suite's
    // empty-reply fix) -- it must not fail this check, so "something was
    // inserted" is logged, not asserted.
    assert(value.startsWith('HEAD_MARKER'), `head lost: "${value.slice(0, 20)}"`);
    assert(value.endsWith('TAIL_MARKER'), `tail lost: "${value.slice(-20)}"`);
    if (value.length === 'HEAD_MARKER\n\nTAIL_MARKER'.length) {
      console.log('    (note: model produced an empty completion -- head/tail preservation still verified)');
    }
  });

  await suite.check('system prompt autosaves and reopens expanded', async () => {
    // The per-notebook system-prompt editor is a contributed section of the
    // app-shell settings drawer now, so it only exists in the DOM while the
    // drawer is open (a notebook must already be active from prior checks).
    await openDrawer(page);
    // Set value + fire the input event directly (the sysprompt autosaves on
    // 'input'); avoids depending on the field's clickability inside the drawer.
    await page.evaluate((val) => {
      const ta = document.querySelector('.sysprompt .sysprompt-input');
      ta.value = val;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }, 'You are a marine biologist.');
    // Outcome-based: wait for the debounced PUT server-side (list view
    // carries system_prompt) before reloading, instead of sleeping past the
    // nominal 500ms debounce window.
    await waitFor(async () => (await notebookListRow(page))?.system_prompt?.includes('marine biologist'),
      { message: 'system prompt not saved server-side before reload' });
    await ctx.open('#/notebook');  // reload closes the drawer
    await page.waitForSelector('.notebook__content'); // notebook re-selected + editor ready
    await openDrawer(page);         // reopen to reach the contributed sysprompt section
    await page.waitForSelector('.sysprompt-input');
    await waitFor(async () => (await page.$eval('.sysprompt-input', (e) => e.value)).includes('marine biologist'),
      { message: 'system prompt not persisted' });
    // The shared prompt section (prompt-section.js) is ALWAYS open by design --
    // a collapsed-when-empty field read as "my prompt disappeared". Asserted
    // deliberately so a regression to collapse-on-reopen is caught here, not
    // as a vacuous pass.
    const open = await page.$eval('.sysprompt', (e) => e.open);
    assert(open, 'system prompt details did not reopen expanded');
    await closeDrawer(page);
  });

  await suite.check('notebook preset bar: save, drift, armed apply', async () => {
    // Shared preset bar (preset-bar.js) contributed by notebook too; same
    // grammar as chat: inert select, live drift line, explicit armed Apply.
    // ORDER-COUPLED: relies on the notebook's system prompt still being "You
    // are a marine biologist." from the prior check -- do not reorder or
    // isolate without updating the drift-flip assertions below.
    await openDrawer(page);
    await page.waitForSelector('.preset-section');
    // save the current notebook state (marine-biologist prompt) as a preset
    await page.click('.preset-section .input');
    await page.type('.preset-section .input', 'nb-preset');
    await clickByText(page, '.preset-section button', 'Save as new');
    await waitFor(async () => (await driftText(page))?.includes('Matches'),
      { message: 'drift line not "Matches" right after save' });
    // NOTEBOOK CHIP: saving stamps the association -- the chip in the editor
    // row (behind the drawer backdrop, but readable) must name the preset.
    const chipText = () => page.$eval('.notebook__row .preset-chip', (el) => (el.hidden ? null : el.textContent));
    await waitFor(async () => (await chipText()) === 'nb-preset',
      { message: 'notebook chip did not show the saved preset' });
    // drift the prompt -- the line must flip live, without a rebuild
    await page.evaluate(() => {
      const ta = document.querySelector('.sysprompt-input');
      ta.value = 'You are a physicist.';
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    });
    // The line reads "Prompt differs from <name> ..." since v1.79.62 (the chat
    // suite's twin was updated then; this one still matched the old lead).
    await waitFor(async () => /differs/i.test((await driftText(page)) ?? ''),
      { message: 'drift line did not flip after a prompt edit' });
    await waitFor(async () => (await chipText())?.includes('(edited)'),
      { message: 'notebook chip did not gain (edited) after the prompt drift' });
    // Apply arms first here: it would replace a differing non-empty prompt
    const applyBtn = await handleByText(page, '.preset-section button', 'Apply');
    await armedClick(applyBtn);
    await applyBtn.dispose();
    await waitFor(async () => (await page.$eval('.sysprompt-input', (e) => e.value)).includes('marine biologist'),
      { message: 'apply did not restore the preset prompt' });
    // cleanup so the preset doesn't leak (presets are excluded from /v1/data/clear)
    await armedClick(await page.$('.preset-section .btn--ghost'));
    await waitFor(async () => page.$eval('.preset-row select',
      (s) => ![...s.options].some((o) => o.textContent === 'nb-preset')),
      { message: 'preset not deleted' });
    await waitFor(async () => (await chipText()) === null,
      { message: 'notebook chip did not clear after preset delete' });
    await closeDrawer(page);
  });

  await suite.check('stop mid-generation keeps partial text', async () => {
    await ctx.open('#/notebook', { max_tokens: STOP_TEST_MAX_TOKENS });
    await page.waitForSelector('.notebook__content');
    await page.select('.notebook__model', config.model);
    const seed = 'Begin: ';
    await page.$eval('.notebook__content', (el, seedText) => {
      el.value = seedText;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.setSelectionRange(el.value.length, el.value.length);
      el.focus();
    }, seed);
    await clickByText(page, '.notebook__actions button', 'Generate');
    // startGenerate() flips the button label to 'Stop' SYNCHRONOUSLY in the
    // click handler, before any network call -- so that transition itself
    // isn't the race. The race is the WHOLE generation (start->finish)
    // completing before we ever get a chance to click Stop: at
    // STOP_TEST_MAX_TOKENS=400 this is unlikely but not impossible (a model
    // can legally hit EOS after a handful of tokens). Poll for whichever
    // actionable state arrives first, and decide what to click from the
    // observed state -- never assume 'Stop' is still showing.
    const outcome = await waitFor(async () => {
      const label = await textOf(page, '.notebook__actions button');
      if (label === 'Generate') return { finishedFast: true };
      const val = await page.$eval('.notebook__content', (e) => e.value);
      if (val.length > seed.length + 3) return { finishedFast: false };
      return null;
    }, { message: 'generation neither streamed partial content nor completed' });

    if (outcome.finishedFast) {
      // Generation finished on its own before Stop was clickable -- a legal
      // outcome (short completion), not a pipeline bug. Stop-discipline
      // itself goes unverified this run; still assert the pipeline produced
      // content rather than silently passing on nothing.
      console.log('    (note: generation completed before Stop could be clicked -- stop-discipline not exercised this run)');
      const value = await page.$eval('.notebook__content', (e) => e.value);
      assert(value.length > seed.length, 'no content after a fast-finished generation');
      return;
    }

    await clickByText(page, '.notebook__actions button', 'Stop');
    await waitForLabel(page, '.notebook__actions button', 'Generate', { message: 'did not stop' });
    const status = await textOf(page, '.notebook__status');
    assert(/stopped/i.test(status), `status="${status}"`);
    const value = await page.$eval('.notebook__content', (e) => e.value);
    assert(value.length > seed.length, 'partial text discarded');
  });

  await suite.check('delete notebook (armed) removes it from the list', async () => {
    await ctx.open('#/notebook');
    await waitFor(async () => (await count(page, '.notebook-item')) >= 1, { message: 'no notebooks' });
    const before = await count(page, '.notebook-item');
    const del = await page.$('.notebook-item__delete');
    await armedClick(del);
    await del.dispose();
    await waitFor(async () => (await count(page, '.notebook-item')) === before - 1, { message: 'notebook not removed' });
  });

  // ============================= PERF ====================================
  await suite.check('perf system metrics render', async () => {
    await ctx.open('#/perf');
    await page.waitForSelector('.perf');
    await waitFor(async () => {
      const v = await textOf(page, '.perf-row__value');
      return v && /GB/.test(v);
    }, { message: 'RAM value never populated' });
    const err = await page.$eval('.perf .error-note', (e) => e.hidden).catch(() => true);
    assert(err === true, 'perf error-note is visible');
  });

  await suite.check('perf page does NOT poll (no requests while idle)', async () => {
    // let the mount fetches settle first
    await sleep(500);
    const watch = watchRequests(page, /\/v1\/(system\/metrics|performance\/profile)/);
    await proveQuiet(watch, { quiet: 2500, message: 'perf page background requests while idle' });
  });

  await suite.check('Refresh triggers exactly one metrics fetch', async () => {
    const watch = watchRequests(page, /\/v1\/system\/metrics/);
    await clickByText(page, '.perf__header-actions button', 'Refresh');
    // Condition-wait for the fetch to fire, then a quiet window to prove no
    // SECOND fetch follows -- that absence is the actual claim (proveQuiet
    // carries the rationale for the bounded sleep).
    await proveQuiet(watch, { atLeast: 1, quiet: 800, message: 'metrics fetches on Refresh' });
  });

  await suite.check('switching time range loads a new profile', async () => {
    const watch = watchRequests(page, /\/v1\/performance\/profile\/6h/);
    await clickByText(page, '.perf__range-buttons button', '6h');
    await waitFor(async () => page.$eval('.perf__range-buttons button:nth-child(2)', (e) => e.classList.contains('perf__range-btn--active')),
      { message: '6h did not become active' });
    await waitFor(async () => watch.urls.length >= 1, { message: 'no 6h profile request' });
    watch.stop();
  });

  await suite.check('profile section renders a table or a resolved empty state', async () => {
    // loadProfile() writes a '.empty-state' "Loading..." placeholder BEFORE
    // the fetch resolves, and renderProfileEmpty() also uses '.empty-state'
    // for the real "no data yet" outcome -- checking for either class alone
    // would vacuously pass on a stuck/never-resolved fetch (rubric C).
    // Exclude the loading placeholder explicitly so this only passes once
    // the range switch actually resolved to a real state.
    await waitFor(async () => {
      const text = await textOf(page, '.perf__profile-body');
      return text !== null && text !== 'Loading…';
    }, { message: 'profile body never left the loading placeholder' });
    const hasTable = (await count(page, '.perf__profile-body .perf-table')) > 0;
    const hasEmpty = (await count(page, '.perf__profile-body .empty-state')) > 0;
    assert(hasTable || hasEmpty, 'profile body neither table nor empty-state');
  });

  // ============================ MODELS ===================================
  await suite.check('models page lists the E2E model', async () => {
    await ctx.open('#/models');
    await page.waitForSelector('.models');
    await waitFor(async () => (await count(page, '.model-row')) > 0, { message: 'no model rows' });
    const ids = await page.$$eval('.model-row__title strong', (els) => els.map((e) => e.textContent.trim()));
    assert(ids.includes(config.model), `${config.model} not listed`);
  });

  await suite.check('preloaded model shows a Loaded badge', async () => {
    const st = await modelRowState(page, config.model);
    assert(st?.loaded, 'E2E model is not marked Loaded');
  });

  await suite.check('unload then reload toggles the model state', async () => {
    const rowActionBtn = async () => {
      const row = await findModelRow(page, config.model);
      return row && row.$('.model-row__actions button');
    };

    let btn = await rowActionBtn();
    assert((await btn.evaluate((e) => e.textContent.trim())) === 'Unload', 'expected Unload button');
    await btn.click();
    await waitFor(async () => (await modelRowState(page, config.model))?.badge === 'Idle',
      { timeout: 30000, message: 'model never became Idle' });

    // reload it so the box returns to its prior state
    btn = await rowActionBtn();
    await btn.click();
    await waitFor(async () => (await modelRowState(page, config.model))?.loaded,
      { timeout: 120000, message: 'model never reloaded' });
  });

  await suite.check('Load warms the model, not just resides it', async () => {
    // The Load button sends ?warm=true, so "Loaded" means the Metal kernels
    // are JIT'd too and the first real message doesn't pay for it. The warm
    // TIMING note is the only observable difference between a warm load and
    // a bare one -- without asserting it, a silently-dropped `warm` param
    // would leave every other check on this page passing.
    const note = await textOf(page, '.models__list-note');
    assert(/loaded and warmed in [\d.]+s/.test(note || ''),
      `expected a warm-timing note after the reload above, got ${JSON.stringify(note)}`);
  });

  // Drive one scan and wait for THAT scan to finish. Waiting on "the panel
  // has rows" instead is a trap: the previous scan's rows are still there, so
  // the wait returns instantly and the assertions read pre-click state.
  // Returns the request body the page actually sent, or null if it refused to
  // send one.
  async function runScan({ paths, hf }) {
    await page.$eval('#scan-paths', (el) => { el.value = ''; });
    if (paths) await page.type('#scan-paths', paths);
    await page.$eval('#scan-hf', (el, want) => { if (el.checked !== want) el.click(); }, hf);

    let sentBody = null;
    let responded = false;
    const onRequest = (req) => {
      if (req.url().endsWith('/v1/admin/models/scan')) sentBody = JSON.parse(req.postData() || '{}');
    };
    const onResponse = (res) => {
      if (res.url().endsWith('/v1/admin/models/scan')) responded = true;
    };
    page.on('request', onRequest);
    page.on('response', onResponse);
    try {
      await clickByText(page, '.models__section-head button', 'Scan');
      if (paths || hf) {
        await waitFor(async () => responded, { timeout: 60000, message: 'scan never responded' });
        // the click handler renders after the await; give it the same tick
        await waitFor(async () => (await textOf(page, '.models__section-head button')) === 'Scan',
          { timeout: 10000, message: 'scan button never returned to idle' });
      }
    } finally {
      page.off('request', onRequest);
      page.off('response', onResponse);
    }
    return sentBody;
  }

  await suite.check('a failed Load stays on screen after the list refreshes', async () => {
    // The handler paints its failure, then refetches the model list to update
    // badges -- and that refetch used to clear the status area on success,
    // wiping the message ~200ms later. Consequence: this page has never once
    // shown a load failure. Asserting AFTER the row has re-rendered is the
    // whole point; asserting immediately passes even with the bug.
    await page.setRequestInterception(true);
    const fail = (req) => {
      // Path moved off the admin gate in v1.79.48. This is a REGEX, so the
      // escaped `admin\/models` form did not match the plain-string sweep that
      // moved the other six callers -- and the stale interceptor made the
      // synthetic 500 never fire, so Load SUCCEEDED and the test reported
      // "load failure raised no error note at all", which reads like a product
      // bug in the page rather than a stale harness.
      if (req.method() === 'POST' && /\/v1\/models\/.*\/load/.test(req.url())) {
        req.respond({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Failed to load model: synthetic e2e failure' }),
        });
      } else {
        req.continue();
      }
    };
    page.on('request', fail);
    try {
      const row = await findModelRow(page, config.model);
      const btn = await row.$('.model-row__actions button');
      const wasLoaded = (await btn.evaluate((e) => e.textContent.trim())) === 'Unload';
      if (wasLoaded) {
        // Unload first (real call), so the next click is a Load we can fail.
        await btn.click();
        await waitFor(async () => (await modelRowState(page, config.model))?.badge === 'Idle',
          { timeout: 30000, message: 'model never became Idle' });
      }
      await (await (await findModelRow(page, config.model)).$('.model-row__actions button')).click();
      await waitFor(async () => !!(await textOf(page, '.models__status .error-note')),
        { timeout: 15000, message: 'load failure raised no error note at all' });
      // Let the trailing refetch land, then re-assert.
      await sleep(1000);
      const err = await textOf(page, '.models__status .error-note');
      assert(err && /Load failed/.test(err),
        `the load failure was wiped by the list refresh (status now ${JSON.stringify(err)})`);
    } finally {
      page.off('request', fail);
      await page.setRequestInterception(false);
    }

    // Restore: really load it again for the checks that follow. The
    // interception was JUST torn down, and a request issued inside that
    // teardown window can be silently dropped (puppeteer interception race)
    // -- seen live 2026-08-18 as a 120s "never reloaded" timeout. Retry the
    // click once if the first attempt visibly never lands.
    await (await (await findModelRow(page, config.model)).$('.model-row__actions button')).click();
    try {
      await waitFor(async () => (await modelRowState(page, config.model))?.loaded,
        { timeout: 60000, message: 'first reload attempt never landed' });
    } catch {
      await (await (await findModelRow(page, config.model)).$('.model-row__actions button')).click();
      await waitFor(async () => (await modelRowState(page, config.model))?.loaded,
        { timeout: 120000, message: 'model never reloaded after the failure check (retried)' });
    }
  });

  await suite.check('scan reaches local folders, not just the HF cache', async () => {
    // The whole GGUF import path targets local model folders. The page used
    // to hardcode {scan_hf_cache: true} with no paths, so nothing on disk
    // outside the HF cache was reachable from the UI at all.
    const body = await runScan({ paths: 'modelzoo', hf: false });
    assert(body?.paths?.includes('modelzoo'),
      `scan body carried no local path: ${JSON.stringify(body)}`);
    assert(body.scan_hf_cache === false, 'unchecking the HF cache did not reach the request');
    const err = await textOf(page, '.models__status .error-note');
    assert(!err, `scan raised an error: ${err}`);
  });

  await suite.check('scanning nothing at all is refused, not sent', async () => {
    // Both sources off can only return nothing; the page has to say so rather
    // than round-trip an empty scan and render "No new models found", which
    // reads as "your folder is empty".
    const body = await runScan({ paths: '', hf: false });
    assert(body === null, `empty scan was still sent: ${JSON.stringify(body)}`);
    await waitFor(async () => !!(await textOf(page, '.models__status .error-note')),
      { timeout: 5000, message: 'empty scan raised no error note' });
  });

  await suite.check('HF cache scan still works and rows report what was found', async () => {
    const body = await runScan({ paths: '', hf: true });
    assert(body?.scan_hf_cache === true, `HF scan not requested: ${JSON.stringify(body)}`);
    const err = await textOf(page, '.models__status .error-note');
    assert(!err, `scan raised an error: ${err}`);

    // Every rendered row must carry a meta line whose first field is a size
    // and a provider. The importer's findings (modalities, thinking, a paired
    // drafter) are appended to that same line, so a bare meta means the row
    // regressed to id-only.
    const metas = await page.$$eval('.scan-row .scan-row__meta',
      (els) => els.map((e) => e.textContent.trim()));
    for (const meta of metas) {
      assert(/^\d+\.\d+ GB · (mlx|mlx_embedding|gguf)/.test(meta),
        `scan row meta is not size + provider: ${JSON.stringify(meta)}`);
    }
  });

  await suite.check('Configure opens a schema-driven config editor', async () => {
    // The panel is generated from GET /v1/admin/model-options, so the honest
    // assertion is against that schema: every field the server declares for
    // this model's provider must render a control (advanced ones live inside
    // a <details>, but they are in the DOM either way). A hand-picked field
    // list here would rot the first time the backend adds one -- schema-driven
    // is the feature, so schema-driven is the check.
    const row = await findModelRow(page, config.model);
    const cfgBtn = await row.evaluateHandle((r) =>
      [...r.querySelectorAll('.model-row__actions button')].find((b) => b.textContent.trim() === 'Configure'));
    await cfgBtn.asElement().click();
    await page.waitForSelector('.model-config', { timeout: 10000 });

    const models = await serverGet(page, '/v1/admin/models');
    const provider = models?.models?.find((m) => m.id === config.model)?.provider;
    const schema = await serverGet(page, '/v1/admin/model-options');
    // Hidden fields are declared IN the schema (ui:"hidden"), so the check
    // derives them from the same source the editor reads -- no hand-copied
    // mirror to rot.
    const schemaFields = schema?.providers?.[provider]?.fields ?? [];
    const expected = schemaFields
      .filter((f) => f.ui !== 'hidden').map((f) => f.name).sort();
    assert(expected.length > 0, `option schema has no fields for provider ${provider}`);

    const rendered = (await page.$$eval('.model-config .cfg-field__label',
      (els) => els.map((e) => e.textContent.trim()))).sort();
    assert(JSON.stringify(rendered) === JSON.stringify(expected),
      `rendered fields diverge from the option schema:\n  schema: ${expected.join(',')}\n  rendered: ${rendered.join(',')}`);

    // load_time_only fields must be disabled and say why.
    for (const f of schemaFields) {
      if (f.effect !== 'load_time_only' || f.ui === 'hidden') continue;
      const disabled = await page.$eval(`#mcfg-${config.model.replace(/[^a-zA-Z0-9_-]/g, '-')}-${f.name}`,
        (el) => el.disabled);
      assert(disabled, `load_time_only field ${f.name} is editable`);
    }

    // The fit meter (v1.60.0): server-computed, renders a weights row and a
    // verdict. E2E's model is real and loaded-able, so the verdict resolves
    // (any of the three states -- the machine's RAM is not the check's
    // business; "fit unavailable" IS a failure, it means the POST broke).
    await page.waitForSelector('.cfg-fit', { timeout: 5000 });
    await waitFor(async () => {
      const v = await textOf(page, '.cfg-fit__verdict');
      return Boolean(v && v.trim() && !/unavailable/.test(v));
    }, { timeout: 10000, message: 'fit verdict never resolved (or came back unavailable)' });
    const weights = await textOf(page, '.cfg-fit__value');
    assert(/GiB/.test(weights || ''), `weights row missing, got "${weights}"`);
  });

  await suite.check('config save PATCHes typed values and null resets a cleared field', async () => {
    // Intercepted end to end: the E2E server runs on the REAL models.toml
    // (only the DB is isolated), and a landed PATCH would rewrite it. The
    // check is about what the page SENDS -- typed JSON, not strings, and an
    // explicit null for a cleared field (the wire spelling of "back to the
    // default") -- plus the reload affordance rendered from the response.
    const input = await page.$('.model-config input[id$="-max_tokens"]');
    assert(input, 'no max_tokens control in the open panel');

    const bodies = [];
    await page.setRequestInterception(true);
    const fake = (req) => {
      if (req.method() === 'PATCH' && req.url().includes('/v1/admin/models/')) {
        bodies.push(JSON.parse(req.postData() || '{}'));
        // Echo a post-save model like the real route would: the page rebuilds
        // the panel from response.model.config, so the fake must carry the
        // saved key for the follow-up clear-to-null step to be dirty.
        const cfg = bodies.length === 1 ? { max_tokens: 512 } : {};
        req.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            model: { config: cfg, stale_reload_fields: [] },
            reload_required_fields: [],
          }),
        });
      } else {
        req.continue();
      }
    };
    page.on('request', fake);
    try {
      // Select-all via the native input API rather than clickCount: 3.
      // puppeteer 25 changed triple-click semantics and the selection no
      // longer lands, so the type() APPENDED instead of replacing and the
      // check failed on the harness, not the app (verified: identical tree
      // passes 28/28 on puppeteer 24.10.2). el.select() is what a real
      // triple-click invokes natively, so this is more faithful, not less.
      await input.click();
      await input.evaluate((el) => el.select());
      await input.type('512');
      await clickByText(page, '.model-config .cfg-actions button', 'Save');
      await waitFor(async () => bodies.length === 1, { timeout: 10000, message: 'PATCH never sent' });
      assert(bodies[0]?.config?.max_tokens === 512,
        `expected typed integer 512, got ${JSON.stringify(bodies[0])}`);
      await waitFor(async () => /Saved\./.test(await textOf(page, '.model-config .cfg-note') || ''),
        { timeout: 5000, message: 'save note never appeared' });

      // The save rebuilds the panel (that is how the row marker/chip
      // repaint), so the old input handle is detached -- re-query it.
      const input2 = await page.$('.model-config input[id$="-max_tokens"]');
      assert(input2, 'max_tokens control missing after the post-save rebuild');
      assert(await input2.evaluate((el) => el.value) === '512',
        'rebuilt panel did not show the saved value');

      // Clearing the just-saved value must re-arm Save (the rebuilt panel's
      // baseline is the response config) and send an explicit null.
      await input2.click();
      await input2.evaluate((el) => el.select());
      await page.keyboard.press('Backspace');
      await clickByText(page, '.model-config .cfg-actions button', 'Save');
      await waitFor(async () => bodies.length === 2, { timeout: 10000, message: 'null-reset PATCH never sent' });
      assert(bodies[1]?.config?.max_tokens === null,
        `expected explicit null for the cleared field, got ${JSON.stringify(bodies[1])}`);
    } finally {
      page.off('request', fake);
      await page.setRequestInterception(false);
    }
  });

  await suite.check('the chat template panel shows what the model resolves to', async () => {
    // Read-only: a GET cannot touch the model directory. The WRITE half is
    // intercepted in the next check for the same reason the config PATCH is --
    // the E2E server runs on the REAL models.toml and real model folders, and
    // a landed PUT would drop a chat_template.heylook.jinja beside the weights.
    const details = await page.$('.cfg-tmpl');
    assert(details, 'no chat template panel in the open config panel');

    // Lazy by design: nothing is fetched until the section is opened.
    assert(await details.evaluate((el) => !el.open), 'template panel starts open');
    await page.$eval('.cfg-tmpl > summary', (el) => el.click());

    await waitFor(async () => Boolean((await textOf(page, '.cfg-tmpl__origin') || '').trim()),
      { timeout: 10000, message: 'template panel never resolved an origin' });
    const origin = await textOf(page, '.cfg-tmpl__origin');
    assert(/In force:/.test(origin), `origin line reads ${JSON.stringify(origin)}`);

    const body = await page.$eval('.cfg-tmpl__body', (el) => el.value);
    assert(body && body.trim().length > 0,
      'the resolved template came back empty -- the panel would paint a blank editor');

    // Save is disabled until something is actually edited. This is also the
    // CRLF guard: a template with \r\n used to read as edited the instant it
    // was painted, because a textarea's value getter normalizes line endings
    // while the compared server string does not.
    const disabled = await page.$eval('.cfg-tmpl .cfg-actions button', (el) => el.disabled);
    assert(disabled, 'Save was enabled on a freshly loaded, unedited template');
  });

  await suite.check('unsaved template text survives a models-list rebuild', async () => {
    // The regression this exists for: the textarea was panel-local state, so
    // any renderModelList rebuild (Load, unload, a config save, a reload)
    // silently discarded typed text and re-collapsed the section. This file's
    // own header states unsaved edits live in the caller's `draft` object for
    // exactly this reason; the template panel did not honour it.
    //
    // A config save is the cheapest rebuild trigger, and it is intercepted so
    // nothing reaches models.toml.
    const TYPED = '{# E2E-DRAFT-MARKER #}';
    await page.$eval('.cfg-tmpl__body', (el) => { el.focus(); });
    await page.type('.cfg-tmpl__body', TYPED);

    const bodies = [];
    const templateWrites = [];
    await page.setRequestInterception(true);
    const fake = (req) => {
      if (req.method() === 'PATCH' && req.url().includes('/v1/admin/models/')) {
        bodies.push(1);
        req.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            model: { config: { max_tokens: 512 }, stale_reload_fields: [] },
            reload_required_fields: [],
          }),
        });
      } else if (req.method() === 'PUT' && req.url().includes('/chat-template')) {
        // Never let a template write land on a real model folder. RECORD and
        // ABORT -- do not assert here. `assert` throws, and a throw inside a
        // puppeteer request listener is raised on the emitter, not on the
        // check body awaiting below: it can never turn this check red. The
        // branch also has to dispose of the request, or the PUT hangs and the
        // safety property holds only by accident of the stall. The recorded
        // flag is asserted after the awaits, where a failure is real.
        templateWrites.push(req.url());
        req.abort();
      } else {
        req.continue();
      }
    };
    page.on('request', fake);
    try {
      // STAMP the live panel first. Waiting on "the PATCH was sent" is not
      // enough and made this check vacuous: the request fires before
      // renderModelList runs, so the poll caught the OLD textarea still
      // holding the typed text and passed even with the draft restore
      // deleted. The stamp is gone the moment a fresh node replaces it, so
      // its absence is the rebuild actually having happened.
      await page.$eval('.cfg-tmpl', (el) => { el.dataset.e2eStamp = '1'; });

      const input = await page.$('.model-config input[id$="-max_tokens"]');
      await input.click();
      await input.evaluate((el) => el.select());
      await input.type('512');
      await clickByText(page, '.model-config .cfg-actions button', 'Save');
      await waitFor(async () => bodies.length === 1,
        { timeout: 10000, message: 'PATCH never sent' });

      await waitFor(async () => {
        const el = await page.$('.cfg-tmpl');
        if (!el) return false;
        return !(await el.evaluate((n) => n.dataset.e2eStamp === '1'));
      }, { timeout: 10000, message: 'the models list never rebuilt the template panel' });

      const stillOpen = await page.$eval('.cfg-tmpl', (el) => el.open);
      assert(stillOpen, 'the rebuild re-collapsed the template section');

      await waitFor(async () => {
        const v = await page.$eval('.cfg-tmpl__body', (el) => el.value).catch(() => '');
        return v.includes(TYPED);
      }, { timeout: 10000, message: 'typed template text was discarded by the rebuild' });

      // Asserted HERE, on the check body, which is the only place a failure
      // can be seen. A draft is not a save: nothing in this check should ever
      // put a template on disk.
      assert(templateWrites.length === 0,
        `the template panel wrote a template during a rebuild check: ${templateWrites.join(', ')}`);
    } finally {
      page.off('request', fake);
      await page.setRequestInterception(false);
    }
  });

  await suite.check('an unsaved template body arms the unload guard', async () => {
    // Runs on what the PREVIOUS check left: typed text that has already
    // survived a models-list rebuild, which is the whole point of the
    // ordering. The guard underneath is refcounted process-wide, and a
    // rebuild constructs a NEW editor over the SAME draft object with no
    // destroy hook -- so a per-panel enable() leaks one count per rebuild and
    // the first disarm below could never reach zero.
    try {
      // Assert the precondition rather than assuming it: the text typed in
      // the previous check must ALREADY have the guard armed here, which is
      // also the only proof that a rebuilt panel re-arms through
      // render()/syncDirty rather than only on a keystroke. Without this the
      // disarm below could pass on a guard that was never armed at all.
      assert(await unloadGuardArmed(page),
        'the draft left by the rebuild check did not arm the guard -- everything below would pass vacuously');

      await clearTemplateBody(page);
      assert(!(await unloadGuardArmed(page)),
        'the guard stayed armed after the draft was cleared -- a refcount leaked on the rebuild, and the dialog now fires on every page');

      await page.type('.cfg-tmpl__body', '{# E2E-GUARD-MARKER #}');
      assert(await unloadGuardArmed(page), 'unsaved template text did not arm the unload guard');

      // A blank box is not pending work -- Save refuses it -- so the guard
      // must agree with the button rather than warn about losing nothing.
      await clearTemplateBody(page);
      assert(!(await unloadGuardArmed(page)), 'the guard stayed armed with nothing left to save');
    } finally {
      await clearTemplateBody(page).catch(() => {});
    }
  });

  await suite.check('a draft holds the guard from a panel that is no longer on screen', async () => {
    // syncUnsavedGuard asks EVERY draft on the page, and this is the check
    // that says so: narrow it to the open panel's draft and every other guard
    // check here stays green while the real regression ships -- type into one
    // model, open another, reload, and the first model's text is gone with no
    // warning. Opening the second panel is what forces the recomputation; a
    // closed panel alone would leave a stale armed guard and prove nothing.
    const otherId = await page.evaluate((mine) =>
      [...document.querySelectorAll('.model-row__title strong')]
        .map((el) => el.textContent.trim())
        .find((id) => id && id !== mine) || null, config.model);
    if (!otherId) skip('only one model is served -- nothing to switch panels to');

    try {
      await page.type('.cfg-tmpl__body', '{# E2E-OTHER-PANEL #}');
      assert(await unloadGuardArmed(page), 'typed template text did not arm the guard');

      await openTemplatePanel(page, otherId);
      assert(await unloadGuardArmed(page),
        `opening ${otherId}'s panel disarmed the guard -- the first model's unsaved template is now silently losable`);
    } finally {
      // Hand back both a disarmed guard AND the E2E model's panel, which the
      // next check expects to find open.
      await openTemplatePanel(page, config.model).catch(() => {});
      await clearTemplateBody(page).catch(() => {});
    }
    assert(!(await unloadGuardArmed(page)), 'clearing the original draft left the guard armed');
  });

  await suite.check('open config panel fits a phone viewport', async () => {
    await ctx.setViewport(390, 780);
    assert(await noHorizontalOverflow(page), 'horizontal overflow at 390px with the config panel open');
    await ctx.setViewport(1280, 900);
    // Close the panel so later checks see the page in its default state.
    // Find-and-click retried as one unit: the PREVIOUS check's second save
    // resolves asynchronously (its check only waits for the request to be
    // SENT), and the resulting list re-render can detach a handle grabbed
    // in the gap -- seen once the fit meter widened the rebuild window.
    await waitFor(async () => {
      try {
        const row = await findModelRow(page, config.model);
        const closeBtn = await row.evaluateHandle((r) =>
          [...r.querySelectorAll('.model-row__actions button')].find((b) => b.textContent.trim() === 'Close'));
        const el = closeBtn.asElement();
        if (!el) return false;
        await el.click();
        return true;
      } catch { return false; }
    }, { timeout: 5000, message: 'Close button never clickable' });
    await waitFor(async () => (await count(page, '.model-config')) === 0,
      { timeout: 5000, message: 'config panel never closed' });
  });

  await suite.check('models page has no horizontal overflow at 390px', async () => {
    await ctx.setViewport(390, 780);
    await ctx.open('#/models');
    await page.waitForSelector('.models');
    await waitFor(async () => (await count(page, '.model-row')) > 0, { message: 'rows' });
    assert(await noHorizontalOverflow(page), 'horizontal overflow at 390px on models page');
    await ctx.setViewport(1280, 900);
  });

  await suite.check('leaving the models page disarms the unload guard', async () => {
    // The exit that hurts: page teardown. An enable() with no matching
    // disable() leaves the dialog armed over chat, notebook and perf, which
    // own no unsaved work at all and would never clear it. createUnloadGuard
    // registers that disarm itself; this checks the page routes through it.
    await ctx.open('#/models');
    await page.waitForSelector('.models');
    await waitFor(async () => (await count(page, '.model-row')) > 0, { message: 'no model rows' });
    await openTemplatePanel(page, config.model);

    await page.type('.cfg-tmpl__body', '{# E2E-TEARDOWN-MARKER #}');
    assert(await unloadGuardArmed(page), 'typed template text did not arm the guard');

    // Hash nav, not a reload: the path beforeunload cannot see, and the one
    // that discards the draft. The draft dying here is by design; the dialog
    // outliving the page that raised it is not.
    await ctx.goHash('#/chat');
    await page.waitForSelector('.chat', { timeout: 15000 });
    assert(!(await unloadGuardArmed(page)),
      'the guard survived the models page teardown -- the dialog is armed on every other page now');
  });

  await suite.check('a template save failing AFTER the page is gone cannot re-arm the guard', async () => {
    // The leak a review found in the first version of this feature. commit()
    // optimistically deletes the draft key before its await, and its finally
    // re-writes it from the textarea when the PUT rejects. Landing that after
    // teardown re-armed a guard nobody owns: no teardown left to disarm it,
    // and the next mount's guard is a different closure whose set(false)
    // early-returns -- so the leave-site dialog stuck to every page for the
    // rest of the session. load()'s GET has the same shape (no signal).
    //
    // The PUT is HELD and answered with a failure, so nothing is ever written
    // beside the weights.
    let held = null;
    await page.setRequestInterception(true);
    const hold = (req) => {
      if (req.method() === 'PUT' && req.url().includes('/chat-template')) held = req;
      else req.continue();
    };
    page.on('request', hold);
    try {
      await ctx.open('#/models');
      await page.waitForSelector('.models');
      await waitFor(async () => (await count(page, '.model-row')) > 0, { message: 'no model rows' });
      await openTemplatePanel(page, config.model);

      await page.type('.cfg-tmpl__body', '{# E2E-LATE-FAILURE #}');
      assert(await unloadGuardArmed(page), 'typed template text did not arm the guard');
      await clickByText(page, '.cfg-tmpl .cfg-actions button', 'Save template');
      await waitFor(async () => held !== null,
        { timeout: 10000, message: 'the template PUT was never sent' });

      await ctx.goHash('#/chat');
      await page.waitForSelector('.chat', { timeout: 15000 });
      assert(!(await unloadGuardArmed(page)), 'teardown did not disarm the guard');

      // Now let the save fail, on a page that no longer exists.
      const failing = held;
      held = null;
      await failing.respond({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'e2e: template rejected' }),
      });
      await sleep(300);
      assert(!(await unloadGuardArmed(page)),
        'a save rejecting after teardown re-armed the guard -- it is now stuck on every page with nothing able to clear it');
    } finally {
      if (held) await held.abort().catch(() => {});
      page.off('request', hold);
      await page.setRequestInterception(false);
    }
  });

  await suite.check('no uncaught page errors during the suite', async () => {
    assert(ctx.pageErrors.length === 0, `page errors: ${ctx.pageErrors.join(' | ')}`);
  });

  // ---- LAST: danger zone wipes the isolated DB --------------------------
  await suite.check('danger-zone clear reports deleted counts', async () => {
    await ctx.open('#/models');
    await page.waitForSelector('.models__danger');
    const btn = await page.$('.models__danger button');
    await armedClick(btn);
    await btn.dispose();
    await waitFor(async () => {
      const t = await textOf(page, '.models__danger-result');
      return t && /Deleted \d+ conversations, \d+ notebooks/.test(t);
    }, { message: 'clear result not reported' });
  });
}
