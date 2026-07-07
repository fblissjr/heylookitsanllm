// Pages suite: notebook (autosave + generate-at-cursor tail preservation),
// explore (logprob chips + keyboard nav), perf (no-polling proof + ranges),
// models (list/load/unload + HF scan + danger-zone clear). Data is cleared by
// the orchestrator before this runs; the danger-zone clear check runs LAST.

import { assert, waitFor, sleep } from '../lib/harness.mjs';
import { clickByText, armedClick, count, textOf, waitForLabel, findModelRow, modelRowState, noHorizontalOverflow } from '../lib/dom.mjs';

// Record requests whose URL matches `regex` from the moment this is called until
// stop(). Used to prove the perf page does NOT poll.
function watchRequests(page, regex) {
  const urls = [];
  const handler = (req) => { if (regex.test(req.url())) urls.push(req.url()); };
  page.on('request', handler);
  return { urls, stop: () => page.off('request', handler) };
}

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
    await sleep(700); // debounce is 500ms
    await ctx.open('#/notebook');
    await page.waitForSelector('.notebook__title');
    await waitFor(async () => (await page.$eval('.notebook__title', (e) => e.value)) === 'Ocean Notes',
      { message: 'title not persisted' });
  });

  await suite.check('content autosaves and survives reload', async () => {
    await page.click('.notebook__content');
    await page.type('.notebook__content', 'The sea is wide.');
    await sleep(700);
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
    assert(value.startsWith('HEAD_MARKER'), `head lost: "${value.slice(0, 20)}"`);
    assert(value.endsWith('TAIL_MARKER'), `tail lost: "${value.slice(-20)}"`);
    assert(value.length > 'HEAD_MARKER\n\nTAIL_MARKER'.length, 'nothing was inserted');
  });

  await suite.check('system prompt autosaves and reopens expanded', async () => {
    await page.evaluate(() => {
      const d = document.querySelector('.notebook__sysprompt');
      if (!d.open) d.querySelector('summary').click();
    });
    await page.click('.notebook__sysprompt-input');
    await page.type('.notebook__sysprompt-input', 'You are a marine biologist.');
    await sleep(700);
    await ctx.open('#/notebook');
    await page.waitForSelector('.notebook__sysprompt-input');
    await waitFor(async () => (await page.$eval('.notebook__sysprompt-input', (e) => e.value)).includes('marine biologist'),
      { message: 'system prompt not persisted' });
    const open = await page.$eval('.notebook__sysprompt', (e) => e.open);
    assert(open, 'system prompt details did not reopen expanded');
  });

  await suite.check('stop mid-generation keeps partial text', async () => {
    await ctx.open('#/notebook', { max_tokens: 400 });
    await page.waitForSelector('.notebook__content');
    await page.select('.notebook__model', config.model);
    await page.$eval('.notebook__content', (el) => {
      el.value = 'Begin: ';
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.setSelectionRange(el.value.length, el.value.length);
      el.focus();
    });
    await clickByText(page, '.notebook__actions button', 'Generate');
    await waitForLabel(page, '.notebook__actions button', 'Stop', { message: 'gen did not start' });
    // wait until content grew beyond the seed
    await waitFor(async () => (await page.$eval('.notebook__content', (e) => e.value)).length > 'Begin: '.length + 3,
      { message: 'no partial text appeared' });
    await clickByText(page, '.notebook__actions button', 'Stop');
    await waitForLabel(page, '.notebook__actions button', 'Generate', { message: 'did not stop' });
    const status = await textOf(page, '.notebook__status');
    assert(/stopped/i.test(status), `status="${status}"`);
    const value = await page.$eval('.notebook__content', (e) => e.value);
    assert(value.length > 'Begin: '.length, 'partial text discarded');
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
    await sleep(1200);
    watch.stop();
    assert(watch.urls.length === 1, `expected 1 metrics fetch on Refresh, saw ${watch.urls.length}`);
  });

  await suite.check('switching time range loads a new profile', async () => {
    const watch = watchRequests(page, /\/v1\/performance\/profile\/6h/);
    await clickByText(page, '.perf__range-buttons button', '6h');
    await waitFor(async () => page.$eval('.perf__range-buttons button:nth-child(2)', (e) => e.classList.contains('perf__range-btn--active')),
      { message: '6h did not become active' });
    await sleep(800);
    watch.stop();
    assert(watch.urls.length >= 1, 'no 6h profile request');
  });

  await suite.check('profile section renders a table or an empty state', async () => {
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
