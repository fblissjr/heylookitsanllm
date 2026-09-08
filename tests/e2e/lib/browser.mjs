// System-Chrome launch via puppeteer-core (claude-in-chrome refuses localhost,
// so puppeteer is the only path to drive the frontend against a real server).
//
// Generation length is capped by seeding the sampler PANEL through the drawer --
// small max_tokens = fast, deterministic runs. It used to be seeded by writing
// `localStorage['heylook-v3-settings']` before boot, which v2.0.38 stopped
// persisting.
//
// SEED THE PANEL, NOT THE DOCUMENT. The first replacement PUT params onto the
// document and was wrong in four ways that all trace to one fact: the panel is
// upstream of everything. `startStream` sends `overrides = {...samplerParams()}`
// from the panel and the server layers those ON TOP of the stored params
// (`{**conv.params, **overrides}`), so a dirty panel BEATS a seeded document; a
// new conversation is created with `params: snapshotSettings()`, i.e. the panel;
// and `bindDocumentParams` PUTs the whole panel snapshot, erasing a seed written
// behind its back. Worse, at the first `open()` of a cleared run there is no
// document to seed at all, so the run's first conversation was created uncapped
// and the miss was silent -- the exact failure this comment used to claim had
// been eliminated. Priming the panel is what the localStorage seed actually did,
// and everything downstream inherits from it.

import puppeteer from 'puppeteer-core';
import { openDrawer, closeDrawer } from './dom.mjs';

const DEFAULT_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

export async function launchBrowser() {
  return puppeteer.launch({
    executablePath: process.env.E2E_CHROME || DEFAULT_CHROME,
    headless: process.env.E2E_HEADFUL ? false : true,
    defaultViewport: { width: 1280, height: 900 },
    args: ['--no-first-run', '--no-default-browser-check', '--disable-gpu'],
  });
}

// A page wired for a suite: tracks uncaught page errors (app.js mount try/catch
// invariant) and knows how to seed sampler settings + navigate the hash router.
export function createPageContext(page, { base, maxTokens }) {
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));

  // enable_thinking: false is part of the seed since v1.79.62: a thinking-
  // capable model now THINKS by default, and at the suite's small max_tokens
  // the whole reply is thinking with no content -- the fast checks assert
  // content. The thinking checks turn it on themselves, explicitly.
  const DEFAULT_PARAMS = { max_tokens: maxTokens, enable_thinking: false };

  // How a panel value is spelled in its control. Mirrors settings.js's
  // bindControl, which is the only other place that knows this.
  const controlValue = (v) => (v === true ? 'on' : v === false ? 'off' : v === null ? '' : String(v));

  const ctx = {
    page,
    base,
    pageErrors,

    // THE seeding primitive: set sampler values through the real drawer, so the
    // panel holds them and the create path, the overrides and the params PUT all
    // inherit.
    //
    // `required` is the half that took a correction. Throwing on ANY missing
    // control looked like the loud-over-silent answer and was wrong: several
    // sampler keys are CAPABILITY-GATED (`requiresCap` in PARAM_META), so
    // `#set-enable_thinking` legitimately does not exist on a model without the
    // thinking capability, and capabilities land asynchronously after mount so
    // it can be absent for a moment on any model. That made the throw fire
    // deterministically on a non-thinking E2E_MODEL and intermittently on a slow
    // boot. Only `max_tokens` is genuinely always there, and it is the one whose
    // absence would leave the run uncapped -- so that is what is required, and a
    // gated key that is missing is skipped and reported in the return value.
    async seedPanel(params, { required = ['max_tokens'] } = {}) {
      await openDrawer(page);
      // The panel is built from capabilities; wait for the required control
      // rather than racing it.
      for (const key of required) {
        await page.waitForSelector(`.drawer--open #set-${key}`, { timeout: 10000 });
      }
      const missing = await page.evaluate((p, spell) => {
        const absent = [];
        for (const key of Object.keys(p)) {
          const el = document.getElementById(`set-${key}`);
          if (!el) { absent.push(key); continue; }
          el.value = spell[key];
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return absent;
      }, params, Object.fromEntries(
        Object.entries(params).map(([k, v]) => [k, controlValue(v)])));
      await closeDrawer(page);
      const fatal = missing.filter((k) => required.includes(k));
      if (fatal.length) {
        throw new Error(`seedPanel: no control for ${fatal.join(', ')} -- `
          + 'the run would have been uncapped. Did a sampler key get renamed?');
      }
      return missing;
    },

    // Boot the app fresh, then seed the panel. `settings: null` means reload and
    // seed NOTHING, for a check that has written params of its own and needs
    // them to survive; a PARTIAL bag is merged over the defaults rather than
    // replacing them, so passing `{max_tokens: N}` cannot silently re-enable
    // thinking (it did, once the seed became a whole-bag write).
    // Pages with no sampler panel (models, perf) seed nothing and need nothing.
    async open(hash = '#/chat', settings = DEFAULT_PARAMS) {
      await page.goto(`${base}/${hash}`, { waitUntil: 'domcontentloaded' });
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.waitForSelector('#app', { timeout: 15000 });
      if (settings && (hash.includes('chat') || hash.includes('notebook'))) {
        await page.waitForSelector('.drawer-gear', { timeout: 15000 });
        await ctx.seedPanel({ ...DEFAULT_PARAMS, ...settings });
        // SIDE EFFECT, stated because it is not obvious and cannot be avoided:
        // seeding drives the REAL controls, so it fires onSettingsChange and
        // bindDocumentParams debounce-PUTs the whole snapshot onto whichever
        // document the app restored. Reopening therefore REWRITES that
        // document's stored params to the seed -- which is what you want for a
        // suite that needs short generations, and is why `settings: null`
        // exists for the one check that has written params it needs kept.
        // Settled here rather than left in flight so it cannot land in the
        // middle of the next check's assertions.
        await ctx.settleParams(hash.includes('notebook') ? 'notebooks' : 'conversations');
      }
    },

    // Wait for a pending debounced params PUT to land. Bounded and quiet: it is
    // removing a RACE, not asserting a fact, so a document that never appears
    // (no conversation exists yet -- the normal case at the first open of a
    // cleared run) is not a failure.
    async settleParams(kind = 'conversations') {
      const deadline = Date.now() + 3000;
      while (Date.now() < deadline) {
        try {
          const params = await ctx.readSettings(kind);
          if (params.max_tokens === maxTokens) return;
        } catch { return; }   // nothing to settle against
        await new Promise((r) => setTimeout(r, 100));
      }
    },

    // SPA navigation without a reload (keeps in-page and browser-local state).
    async goHash(hash) {
      await page.evaluate((h) => { location.hash = h; }, hash);
    },

    // The panel's committed state, read where it LANDS: the params of the
    // document the panel is bound to -- the ACTIVE conversation, not the newest
    // one, since chat restores whichever this browser was last in.
    //
    // Resolved by the row's OWN id, never by its position. The sidebar renders
    // the client's locally-mutated array while `GET /v1/conversations` orders by
    // updated_at DESC, and every caller polls this right after a panel edit
    // whose PUT bumps updated_at -- so the two orders diverge exactly when this
    // is called, and an index mapping reads a different conversation. That is
    // the same updated_at trap `newFreshConversation` documents as reproduced
    // live 2026-07-23; it was reintroduced here and caught in review.
    //
    // THROWS rather than returning {} when it cannot resolve. Two of its callers
    // assert ABSENCE (`!('vision_tokens' in ...)`), which an empty answer
    // satisfies for free -- an oracle that returns empty on failure turns those
    // into checks that cannot fail. Debounced (400ms), so callers poll.
    async readSettings(kind = 'conversations') {
      const found = await page.evaluate(async (k) => {
        if (k !== 'conversations') {
          const res = await fetch(`/v1/${k}`);
          if (!res.ok) return { error: `GET /v1/${k} -> ${res.status}` };
          const rows = (await res.json())[k] ?? [];
          return rows.length ? { id: rows[0].id } : { error: `no ${k} exist` };
        }
        const active = document.querySelector('.conv-item--active');
        if (!active) return { error: 'no active conversation in the sidebar' };
        const id = active.dataset.id;
        if (!id) return { error: 'the active conversation row carries no data-id' };
        return { id };
      }, kind);
      if (found.error) throw new Error(`readSettings: ${found.error}`);
      const body = await page.evaluate(async (k, i) => {
        const res = await fetch(`/v1/${k}/${i}`);
        return res.ok ? (await res.json()).params ?? {} : { __error: res.status };
      }, kind, found.id);
      if (body.__error) throw new Error(`readSettings: GET /v1/${kind}/${found.id} -> ${body.__error}`);
      return body;
    },

    async setViewport(width, height) {
      await page.setViewport({ width, height });
    },
  };
  return ctx;
}
