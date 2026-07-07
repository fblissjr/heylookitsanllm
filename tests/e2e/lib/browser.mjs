// System-Chrome launch via puppeteer-core (claude-in-chrome refuses localhost,
// so puppeteer is the only path to drive /v3 against a real server). Pages are
// seeded with sampler settings through localStorage BEFORE the app boots, which
// is how we cap generation length (small max_tokens = fast, deterministic runs).

import puppeteer from 'puppeteer-core';

const SETTINGS_KEY = 'heylook-v3-settings';
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

  const ctx = {
    page,
    base,
    pageErrors,

    // Seed sampler settings, then boot the app with them applied. settings.js
    // caches localStorage ONCE at module import, so the seed only takes effect
    // on a full document load. A hash-only goto is same-document (no reload), so
    // we write localStorage and then force an explicit reload -- that reload is
    // what makes the seed real, regardless of the prior URL.
    async open(hash = '#/chat', settings = { max_tokens: maxTokens }) {
      await page.goto(`${base}/v3/${hash}`, { waitUntil: 'domcontentloaded' });
      await page.evaluate((key, val) => {
        localStorage.setItem(key, JSON.stringify(val));
      }, SETTINGS_KEY, settings);
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.waitForSelector('#app', { timeout: 15000 });
    },

    // SPA navigation without a reload (keeps localStorage/session state).
    async goHash(hash) {
      await page.evaluate((h) => { location.hash = h; }, hash);
    },

    async readSettings() {
      return page.evaluate((key) => {
        try { return JSON.parse(localStorage.getItem(key) || '{}'); } catch { return {}; }
      }, SETTINGS_KEY);
    },

    async setViewport(width, height) {
      await page.setViewport({ width, height });
    },
  };
  return ctx;
}
