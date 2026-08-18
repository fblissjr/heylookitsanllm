// Pages suite: notebook (autosave + generate-at-cursor tail preservation),
// explore (logprob chips + keyboard nav), perf (no-polling proof + ranges),
// jspace (Jacobian-lens workspace strip, lens-gated), models (list/load+warm/
// unload + folder & HF scan + danger-zone clear). Data is cleared by the
// orchestrator before this runs; the danger-zone clear check runs LAST.

import { assert, waitFor, sleep, proveQuiet } from '../lib/harness.mjs';
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
    await page.click('.notebook__title', { clickCount: 3 });
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
    await clickByText(page, '.preset-section button', 'Save');
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
    await waitFor(async () => (await driftText(page))?.includes('Differs'),
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

  // =========================== EXPLORE ===================================
  await suite.check('explore page shows its prompt when idle', async () => {
    await ctx.open('#/explore');
    await page.waitForSelector('.explore');
    const hint = await textOf(page, '.explore__strip .empty-state');
    assert(hint && hint.length > 0, 'no explore empty-state hint');
  });

  await suite.check('explore model select contains the E2E model', async () => {
    // POLL, never one-shot -- same fill-race as the chat suite's select.
    await waitFor(async () => {
      const opts = await page.$$eval('.explore__bar select option', (els) => els.map((e) => e.value));
      return opts.includes(config.model);
    }, { message: 'model never appeared in the explore select' });
    await page.select('.explore__bar select', config.model);
  });

  await suite.check('generating produces per-token logprob chips', async () => {
    // Waits on the FINAL 'Generate' label (idle), not a transient 'Stop' --
    // correct even if start+finish land inside one poll interval.
    // "Count: one two three" is a strong completion cue (unlike a
    // conversational prompt) specifically to keep an immediate-EOS empty
    // completion vanishingly unlikely -- checks 13-16 below all depend on
    // this producing >=1 token (order-coupled: explore builds up one
    // continuous result, not independent per-check state).
    await page.click('.explore__composer textarea');
    await page.type('.explore__composer textarea', 'Count: one two three');
    await clickByText(page, '.explore__composer button', 'Generate');
    await waitForLabel(page, '.explore__composer button', 'Generate', { timeout: 30000, message: 'explore generation did not finish' });
    await waitFor(async () => (await count(page, '.explore__strip .tok')) > 0, { message: 'no token chips' });
  });

  await suite.check('clicking a token opens its detail panel', async () => {
    await page.click('.explore__strip .tok');
    await waitFor(async () => (await count(page, '.tok--selected')) === 1, { message: 'token not selected' });
    const detail = await textOf(page, '.explore__detail');
    assert(/Logprob/i.test(detail) && /Probability/i.test(detail) && /Position/i.test(detail),
      `detail panel incomplete: "${detail?.slice(0, 80)}"`);
  });

  await suite.check('detail panel lists top alternatives', async () => {
    const bars = await count(page, '.explore__detail .explore-bar');
    assert(bars > 0, 'no alternative bars rendered');
  });

  await suite.check('arrow keys move the token selection', async () => {
    await page.focus('.explore');
    const idxOf = () => page.$$eval('.explore__strip .tok', (els) => els.findIndex((e) => e.classList.contains('tok--selected')));
    const start = await idxOf();
    await page.keyboard.press('ArrowRight');
    await waitFor(async () => (await idxOf()) !== start, { message: 'ArrowRight did not move selection' });
    const afterRight = await idxOf();
    await page.keyboard.press('ArrowLeft');
    await waitFor(async () => (await idxOf()) !== afterRight, { message: 'ArrowLeft did not move selection' });
  });

  await suite.check('Escape clears the selection', async () => {
    await page.focus('.explore');
    await page.keyboard.press('Escape');
    await waitFor(async () => (await count(page, '.tok--selected')) === 0, { message: 'selection not cleared' });
    const detail = await textOf(page, '.explore__detail');
    assert(/Click a token/i.test(detail), `detail did not reset: "${detail?.slice(0, 60)}"`);
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
      if (req.method() === 'POST' && /\/v1\/admin\/models\/.*\/load/.test(req.url())) {
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
      await input.click({ clickCount: 3 });
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
      await input2.click({ clickCount: 3 });
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

  // =========================== JSPACE ====================================
  // Lens-gated: only asserts the analyze flow when a lens for the E2E model is
  // installed at adapters/jspace/<model_id>/ (registry default).
  let jspaceHasLens = false;
  await suite.check('jspace page mounts (lens model or empty-state)', async () => {
    await ctx.open('#/jspace');
    await page.waitForSelector('.jspace');
    // setup() renders the select options OR the empty-state only AFTER the async
    // /v1/jspace/models fetch resolves -- wait for one of them before asserting.
    await waitFor(async () =>
      (await count(page, '.jspace__bar select option')) > 0 ||
      (await count(page, '.jspace .empty-state')) > 0,
      { message: 'jspace models never resolved (no options and no empty-state)' });
    const opts = await page.$$eval('.jspace__bar select option', (els) => els.map((e) => e.value));
    jspaceHasLens = opts.includes(config.model);
    const hasEmpty = (await count(page, '.jspace .empty-state')) > 0;
    assert(jspaceHasLens || hasEmpty, 'jspace: neither the E2E lens model nor an empty-state');
  });

  // ORDER-COUPLED (checks below through "heatmap-off analyze"): one continuous
  // jspace session -- pin/unpin/scope checks read the s.data from THIS
  // Analyze call, and the heatmap-off check reuses the composer text this
  // check types in. Do not reorder or run any of these in isolation.
  await suite.check('jspace analyze renders the workspace strip + heatmap', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    await page.select('.jspace__bar select', config.model);
    // The heatmap toggle is a drawer extra now; flip it on there, then close the
    // drawer (its checked state persists) before driving the page's Analyze.
    await openDrawer(page);
    await page.evaluate(() => document.querySelector('#jspace-heatmap').click()); // heatmap on
    await closeDrawer(page);
    await page.click('.jspace__composer textarea');
    await page.type('.jspace__composer textarea', 'The Eiffel Tower is located in the city of');
    await clickByText(page, '.jspace__composer button', 'Analyze');
    await waitFor(async () => (await count(page, '.jspace__strip .jspace__row')) > 0,
      { timeout: 90000, message: 'workspace strip never rendered' });
    assert((await count(page, '.jspace__chip')) > 0, 'no workspace chips rendered');
    assert((await count(page, '.jspace__hcell')) > 0, 'no heatmap cells rendered');
    assert((await count(page, '.jspace__hpos--onset')) === 1, 'answer-onset column marker missing');
  });

  await suite.check('jspace: clicking a workspace row pins the top-N readout', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    await page.click('.jspace__strip .jspace__row');
    assert((await count(page, '.jspace__row--pinned')) === 1, 'strip row not marked pinned');
    assert((await count(page, '.jspace__detail .jspace-bar')) > 0, 'pinned panel has no top-N bars');
  });

  await suite.check('jspace: Escape unpins the readout', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    await page.keyboard.press('Escape');
    assert((await count(page, '.jspace__row--pinned')) === 0, 'pin survived Escape');
    // Unpinned detail = the aggregation view, not the cell readout.
    assert((await count(page, '.jspace__detail .jspace__agg')) === 1, 'detail panel did not reset');
    assert((await count(page, '.jspace__detail .jspace-bar')) === 0, 'cell readout survived Escape');
  });

  await suite.check('jspace: non-onset heatmap cell pins its per-cell top-N', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    // First data row (nth-of-type 2: row 1 is the token header), first column
    // -- not the onset column; per-cell top-k comes from heatmap_top_k.
    await page.click('.jspace__heatmap .jspace__hrow:nth-of-type(2) .jspace__hcell');
    assert((await count(page, '.jspace__hcell--pinned')) === 1, 'heatmap cell not marked pinned');
    assert((await count(page, '.jspace__detail .jspace-bar')) > 0,
      'pinned non-onset cell has no top-N bars (heatmap_top_k data missing)');
    // Clicking the pinned cell again unpins.
    await page.click('.jspace__hcell--pinned');
    assert((await count(page, '.jspace__hcell--pinned')) === 0, 'cell pin did not toggle off');
  });

  await suite.check('jspace: unpinned detail panel aggregates common tokens', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    assert((await count(page, '.jspace__detail .jspace__agg-row')) > 0,
      'aggregation list empty while unpinned');
  });

  await suite.check('jspace: layer slider slot click scopes the rows; reset restores', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    const slots = await count(page, '.jspace__slot');
    if (slots < 2) { console.log('    (skipped: single-layer band, no slider)'); return; }
    await page.click('.jspace__slot'); // first slot -> single-layer range
    assert((await count(page, '.jspace__row--out')) === slots - 1,
      'expected all but one strip row scoped out');
    assert((await count(page, '.jspace__hrow--out')) === slots - 1,
      'expected all but one heatmap row scoped out');
    // Arrow-walk must respect the scope: pin the one visible row, walk, and
    // the pin must never land on a hidden (--out) row.
    await page.click('.jspace__row:not(.jspace__row--out)');
    await page.keyboard.press('ArrowUp');
    await page.keyboard.press('ArrowDown');
    assert((await count(page, '.jspace__row--pinned')) === 1, 'scoped pin lost during arrow walk');
    assert((await count(page, '.jspace__row--out.jspace__row--pinned')) === 0,
      'arrow walk moved the pin onto a scoped-out row');
    await page.keyboard.press('Escape');
    await clickByText(page, '.jspace__slider button', 'reset');
    assert((await count(page, '.jspace__row--out')) === 0, 'reset did not restore the rows');
  });

  await suite.check('jspace: heatmap-off analyze renders strip-only and pins from onset_strip', async () => {
    if (!jspaceHasLens) { console.log('    (skipped: no lens installed for the E2E model)'); return; }
    await openDrawer(page);
    await page.evaluate(() => document.querySelector('#jspace-heatmap').click()); // toggle heatmap back OFF
    await closeDrawer(page);
    await clickByText(page, '.jspace__composer button', 'Analyze');
    await waitFor(async () => (await count(page, '.jspace__detail')) > 0 &&
      (await count(page, '.jspace__heatmap')) === 0,
      { timeout: 90000, message: 'heatmap-off result never rendered' });
    assert((await count(page, '.jspace__strip .jspace__row')) > 0, 'strip missing');
    assert((await count(page, '.jspace__detail .jspace__agg-row')) > 0,
      'aggregation (onset_strip fallback) empty');
    await page.click('.jspace__strip .jspace__row');
    assert((await count(page, '.jspace__row--pinned')) === 1, 'strip row not pinned');
    assert((await count(page, '.jspace__detail .jspace-bar')) > 0,
      'onset pin has no bars without a heatmap');
    await page.keyboard.press('Escape');
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
