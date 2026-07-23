// Pages suite: notebook (autosave + generate-at-cursor tail preservation),
// explore (logprob chips + keyboard nav), perf (no-polling proof + ranges),
// jspace (Jacobian-lens workspace strip, lens-gated), models (list/load/unload
// + HF scan + danger-zone clear). Data is cleared by the orchestrator before
// this runs; the danger-zone clear check runs LAST.

import { assert, waitFor, sleep } from '../lib/harness.mjs';
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
  return page.evaluate(async () => {
    const res = await fetch('/v1/notebooks');
    const { notebooks } = await res.json();
    return notebooks[0] ?? null;
  });
}

async function notebookFull(page, id) {
  return page.evaluate(async (nbId) => {
    const res = await fetch(`/v1/notebooks/${nbId}`);
    return res.ok ? res.json() : null;
  }, id);
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
    // empty state until a notebook exists
    const empty = await textOf(page, '.notebook__empty');
    assert(empty && empty.length > 0, 'expected notebook empty-state');
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
      const d = document.querySelector('.notebook__sysprompt');
      d.open = true;
      const ta = d.querySelector('.notebook__sysprompt-input');
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
    await page.waitForSelector('.notebook__sysprompt-input');
    await waitFor(async () => (await page.$eval('.notebook__sysprompt-input', (e) => e.value)).includes('marine biologist'),
      { message: 'system prompt not persisted' });
    const open = await page.$eval('.notebook__sysprompt', (e) => e.open);
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
    // drift the prompt -- the line must flip live, without a rebuild
    await page.evaluate(() => {
      const ta = document.querySelector('.notebook__sysprompt-input');
      ta.value = 'You are a physicist.';
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFor(async () => (await driftText(page))?.includes('Differs'),
      { message: 'drift line did not flip after a prompt edit' });
    // Apply arms first here: it would replace a differing non-empty prompt
    const applyBtn = await handleByText(page, '.preset-section button', 'Apply');
    await armedClick(applyBtn);
    await applyBtn.dispose();
    await waitFor(async () => (await page.$eval('.notebook__sysprompt-input', (e) => e.value)).includes('marine biologist'),
      { message: 'apply did not restore the preset prompt' });
    // cleanup so the preset doesn't leak (presets are excluded from /v1/data/clear)
    await armedClick(await page.$('.preset-section .btn--ghost'));
    await waitFor(async () => page.$eval('.preset-row select',
      (s) => ![...s.options].some((o) => o.textContent === 'nb-preset')),
      { message: 'preset not deleted' });
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
    const opts = await page.$$eval('.explore__bar select option', (els) => els.map((e) => e.value));
    assert(opts.includes(config.model), `model missing from explore select`);
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
    await sleep(2500);
    watch.stop();
    assert(watch.urls.length === 0, `expected 0 background requests, saw ${watch.urls.length}: ${watch.urls.join(', ')}`);
  });

  await suite.check('Refresh triggers exactly one metrics fetch', async () => {
    const watch = watchRequests(page, /\/v1\/system\/metrics/);
    await clickByText(page, '.perf__header-actions button', 'Refresh');
    // Condition-wait for the fetch to fire (F: don't sleep for something
    // observable), then a short quiet window to prove no SECOND fetch
    // follows -- that absence is the actual claim, and there's no DOM
    // condition to wait on for "nothing happened", so a bounded sleep here
    // is correct (same technique as the load-bearing no-polling check below).
    await waitFor(async () => watch.urls.length >= 1, { message: 'Refresh never triggered a metrics fetch' });
    await sleep(800);
    watch.stop();
    assert(watch.urls.length === 1, `expected 1 metrics fetch on Refresh, saw ${watch.urls.length}`);
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
      { timeout: 90000, message: 'model never reloaded' });
  });

  await suite.check('HF cache scan renders results panel', async () => {
    await clickByText(page, '.models__section-head button', 'Scan HF cache');
    await waitFor(async () => {
      const hasRows = (await count(page, '.scan-panel .scan-row')) > 0;
      const hasEmpty = (await count(page, '.scan-panel .empty-state')) > 0;
      const hasNote = (await count(page, '.scan-panel .muted')) > 0;
      return hasRows || hasEmpty || hasNote;
    }, { timeout: 30000, message: 'scan panel never populated' });
    const err = await textOf(page, '.models__status .error-note');
    assert(!err, `scan raised an error: ${err}`);
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
