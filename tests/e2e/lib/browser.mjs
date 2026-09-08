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
    // inherit. Throws if a control is missing rather than seeding nothing --
    // a silent no-op here is a whole suite running uncapped.
    async seedPanel(params) {
      await openDrawer(page);
      const missing = await page.evaluate((p, spell) => {
        const absent = [];
        for (const [key, value] of Object.entries(p)) {
          const el = document.getElementById(`set-${key}`);
          if (!el) { absent.push(key); continue; }
          el.value = spell[key];
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return absent;
      }, params, Object.fromEntries(
        Object.entries(params).map(([k, v]) => [k, controlValue(v)])));
      await closeDrawer(page);
      if (missing.length) {
        throw new Error(`seedPanel: no control for ${missing.join(', ')} -- `
          + 'the run would have been uncapped. Did a sampler key get renamed?');
      }
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
      }
    },

    // SPA navigation without a reload (keeps in-page and browser-local state).
    async goHash(hash) {
      await page.evaluate((h) => { location.hash = h; }, hash);
    },

    // The panel's committed state, read where it LANDS: the params of the
    // document the panel is bound to. That is the ACTIVE conversation, not the
    // newest one -- chat restores the conversation this browser was last in, so
    // newest-by-created_at and newest-by-updated_at can both be someone else.
    // Resolved by the sidebar's active row, whose render order is the list's.
    //
    // THROWS rather than returning {} when it cannot resolve. Two of its callers
    // assert ABSENCE (`!('vision_tokens' in ...)`), which an empty answer
    // satisfies for free -- an oracle that returns empty on failure turns those
    // into checks that cannot fail. Debounced (400ms), so callers poll.
    async readSettings(kind = 'conversations') {
      const id = await page.evaluate(async (k) => {
        const items = [...document.querySelectorAll('.conv-item')];
        const idx = items.findIndex((el) => el.classList.contains('conv-item--active'));
        const res = await fetch(`/v1/${k}`);
        if (!res.ok) return { error: `GET /v1/${k} -> ${res.status}` };
        const rows = (await res.json())[k] ?? [];
        if (k !== 'conversations') {
          return rows.length ? { id: rows[0].id } : { error: `no ${k} exist` };
        }
        if (idx < 0) return { error: 'no active conversation in the sidebar' };
        if (!rows[idx]) return { error: `sidebar row ${idx} has no matching ${k} row` };
        return { id: rows[idx].id };
      }, kind);
      if (id.error) throw new Error(`readSettings: ${id.error}`);
      const body = await page.evaluate(async (k, i) => {
        const res = await fetch(`/v1/${k}/${i}`);
        return res.ok ? (await res.json()).params ?? {} : { __error: res.status };
      }, kind, id.id);
      if (body.__error) throw new Error(`readSettings: GET /v1/${kind}/${id.id} -> ${body.__error}`);
      return body;
    },

    async setViewport(width, height) {
      await page.setViewport({ width, height });
    },
  };
  return ctx;
}
