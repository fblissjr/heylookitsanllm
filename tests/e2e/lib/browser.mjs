// System-Chrome launch via puppeteer-core (claude-in-chrome refuses localhost,
// so puppeteer is the only path to drive the frontend against a real server).
//
// Generation length is capped by seeding the DOCUMENT's `params` -- small
// max_tokens = fast, deterministic runs. It used to be seeded through
// `localStorage['heylook-v3-settings']`, which v2.0.38 stopped persisting: the
// sampler panel is a VIEW of a document now and nothing carries it across a
// reload. That is not a workaround for the removal -- the document was ALREADY
// the authoritative half. `hydrateDocParams` replaced the seeded cache on every
// select, so the old seed only ever primed the FIRST conversation, and
// `chat.mjs`'s "the DOCUMENT's params win over the localStorage seed" check
// exists because someone found that out the hard way. Seeding the document
// primes the same thing directly, and fails LOUDLY (an HTTP error) where a
// missed cache seed failed silently.

import puppeteer from 'puppeteer-core';

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

  // Newest document of `kind` by created_at, or null. created_at rather than
  // list position for the reason chat.mjs's own helper gives: the list orders
  // by updated_at, and a trailing debounced params PUT can reorder it.
  const newestId = (kind) => page.evaluate(async (k) => {
    const res = await fetch(`/v1/${k}`);
    if (!res.ok) return null;
    const body = await res.json();
    const rows = body[k] ?? body.data ?? [];
    return rows.reduce((a, b) => (a && a.created_at > b.created_at ? a : b), null)?.id ?? null;
  }, kind);

  const ctx = {
    page,
    base,
    pageErrors,
    defaultParams: DEFAULT_PARAMS,

    // Write sampler params onto a document. THE seeding primitive: every other
    // path here goes through it, so there is one spelling of "make this run
    // short" rather than one per surface.
    async seedParams(kind, id, params = DEFAULT_PARAMS) {
      if (!id) return false;
      const ok = await page.evaluate(async (k, i, p) => {
        const res = await fetch(`/v1/${k}/${i}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ params: p }),
        });
        return res.ok;
      }, kind, id, params);
      if (!ok) throw new Error(`seedParams: PUT /v1/${kind}/${id} params failed`);
      return true;
    },

    // Seed the ACTIVE document's params, then boot the app onto it. The reload
    // is still what makes the seed real: the page hydrates the panel from the
    // document at select, so the PUT has to land before that select runs.
    // A page with no document of its own (models, perf) seeds nothing and needs
    // nothing -- there is no generation to cap there.
    // `settings: null` = reload and seed NOTHING, for a check that has just
    // written params of its own and needs them to survive the reload. Distinct
    // from `{}`, which would PUT an empty bag and erase them -- the difference
    // matters because the seed is a real write now, not a cache poke.
    async open(hash = '#/chat', settings = DEFAULT_PARAMS) {
      await page.goto(`${base}/${hash}`, { waitUntil: 'domcontentloaded' });
      const kind = hash.includes('notebook') ? 'notebooks' : 'conversations';
      if (settings && (hash.includes('notebook') || hash.includes('chat'))) {
        await ctx.seedParams(kind, await newestId(kind), settings);
      }
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.waitForSelector('#app', { timeout: 15000 });
    },

    // SPA navigation without a reload (keeps localStorage/session state).
    async goHash(hash) {
      await page.evaluate((h) => { location.hash = h; }, hash);
    },

    // The panel's committed state, read where it actually LIVES: the document's
    // stored `params`. This used to read localStorage, which was a proxy for the
    // same claim and stopped existing in v2.0.38 -- and the document is the
    // stronger assertion anyway, because it is what the server layers a
    // generation over. Debounced (bindDocumentParams, 400ms), so callers poll.
    async readSettings(kind = 'conversations') {
      const id = await newestId(kind);
      if (!id) return {};
      return page.evaluate(async (k, i) => {
        const res = await fetch(`/v1/${k}/${i}`);
        if (!res.ok) return {};
        return (await res.json()).params ?? {};
      }, kind, id);
    },

    async setViewport(width, height) {
      await page.setViewport({ width, height });
    },
  };
  return ctx;
}
