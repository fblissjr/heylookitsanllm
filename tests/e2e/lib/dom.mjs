// Shared DOM interaction helpers for the suites. Kept intentionally small; page-
// specific logic lives in the suite files.

import { waitFor } from './harness.mjs';

// First element matching `selector` whose trimmed text === `text`, as an
// ElementHandle (for armedClick etc.). Throws if absent; caller disposes.
export async function handleByText(page, selector, text) {
  const handle = await page.evaluateHandle((sel, txt) => {
    return [...document.querySelectorAll(sel)].find((e) => e.textContent.trim() === txt) || null;
  }, selector, text);
  const el = handle.asElement();
  if (!el) {
    await handle.dispose();
    throw new Error(`no <${selector}> with text "${text}"`);
  }
  return el;
}

// Click the first element matching `selector` whose trimmed text === `text`.
export async function clickByText(page, selector, text) {
  const el = await handleByText(page, selector, text);
  await el.click();
  await el.dispose();
}

// The shared preset bar's drift line: its text while visible, null while hidden.
export async function driftText(page) {
  return page.$eval('.preset-drift', (el) => (el.hidden ? null : el.textContent));
}

// Two-tap destructive confirm (utils.armedConfirm): first click arms the button
// (adds .btn--armed, text -> "Confirm?"), second click within 3s runs the action.
export async function armedClick(elHandle) {
  await elHandle.click();
  await waitFor(() => elHandle.evaluate((e) => e.classList.contains('btn--armed')),
    { timeout: 2000, interval: 30, message: 'button never armed' });
  await elHandle.click();
}

// Count elements matching a selector.
export async function count(page, selector) {
  return page.$$eval(selector, (els) => els.length);
}

// Trimmed textContent of the first match, or null.
export async function textOf(page, selector) {
  return page.$eval(selector, (e) => e.textContent.trim()).catch(() => null);
}

// True when the page has no horizontal overflow at the current viewport width.
export async function noHorizontalOverflow(page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
}

// Wait until the element at `selector` shows exactly `label` (trimmed). The
// toggle-button-label idiom (Send<->Stop, Generate<->Stop) used across suites.
export async function waitForLabel(page, selector, label, opts = {}) {
  await waitFor(async () => (await textOf(page, selector)) === label,
    { message: `"${selector}" never showed label "${label}"`, ...opts });
}

// ElementHandle of the models-page row whose title === `id`, or null. Centralizes
// the "find .model-row by its title text" lookup. Use ONLY where an ElementHandle
// is needed (e.g. clicking the row's button) -- for state polled in a loop use
// modelRowState (a plain-value read that doesn't leak a handle per poll).
export async function findModelRow(page, id) {
  const handle = await page.evaluateHandle((modelId) =>
    [...document.querySelectorAll('.model-row')].find(
      (r) => r.querySelector('.model-row__title strong')?.textContent.trim() === modelId) || null,
    id);
  return handle.asElement();
}

// { badge, loaded } for the model row titled `id`, or null if absent. A pure
// value read (no ElementHandle) -- safe to call inside a waitFor poll.
export async function modelRowState(page, id) {
  return page.evaluate((modelId) => {
    const row = [...document.querySelectorAll('.model-row')].find(
      (r) => r.querySelector('.model-row__title strong')?.textContent.trim() === modelId);
    if (!row) return null;
    const badge = row.querySelector('.model-badge');
    return { badge: badge?.textContent.trim() ?? null, loaded: badge?.classList.contains('model-badge--loaded') ?? false };
  }, id);
}

// Read / write a sampler-settings row's <input> by its label (chat settings panel).
export async function settingsInputValue(page, label) {
  return page.evaluate((lbl) => {
    const row = [...document.querySelectorAll('.settings-panel .settings-row')]
      .find((r) => r.querySelector('label')?.textContent.trim() === lbl);
    return row?.querySelector('input')?.value ?? null;
  }, label);
}

export async function setSettingsInput(page, label, value) {
  await page.evaluate((lbl, val) => {
    const row = [...document.querySelectorAll('.settings-panel .settings-row')]
      .find((r) => r.querySelector('label')?.textContent.trim() === lbl);
    const input = row.querySelector('input');
    input.value = val;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, label, value);
}

// The settings/presets/sysprompt/jspace-toggles all live in the app-shell
// settings drawer now (js/settings-drawer.js), not inline on the page. The
// drawer is a MODAL: while open it makes #app `inert`, and its backdrop covers
// the page -- so a puppeteer click aimed at the (inert) sidebar gear lands on
// the backdrop and closes it instead. It also survives a same-document (hash)
// navigation. Both make a naive "click the gear" flaky, so these helpers reset
// to a known-closed, #app-live state first.

// Close the drawer if open and GUARANTEE #app is interactable again. Clicks the
// drawer's own Close button (it's inside the drawer, never inert), then clears
// inert defensively so a leaked-open drawer never seals the page for the next click.
// Waits for BOTH the drawer to go and the backdrop to actually hide -- the
// backdrop's visibility transition is *delayed* ~140ms (reduced-motion doesn't
// cancel a delay), and until it hides it still covers #app, so a too-early click
// on page content lands on the backdrop instead of the button.
export async function closeDrawer(page) {
  await page.evaluate(() => {
    document.querySelector('.drawer--open .drawer__close')?.click();
    const app = document.getElementById('app');
    if (app) app.inert = false;
  });
  await page.waitForFunction(() => {
    if (document.querySelector('.drawer--open')) return false;
    const bd = document.querySelector('.drawer-backdrop');
    return !bd || getComputedStyle(bd).visibility === 'hidden';
  }, { timeout: 5000 });
}

// Open the drawer cleanly for the CURRENT page: reset first (handles a leaked or
// stale open drawer + inert #app), then fire the gear's handler. We use
// evaluate().click() rather than page.click() on purpose: right after
// closeDrawer the backdrop is still fading out, so a hit-tested click would land
// on it and re-close; dispatching the handler directly is immune to that. Then
// wait for the panel to finish sliding in -- a click on drawer content mid-slide
// misses (the element is still off-screen right).
// `gear` picks the opener (default: the sidebar gear; pass e.g. chat's
// in-context '.chat__settings-btn' to exercise that entry point).
export async function openDrawer(page, gear = '.drawer-gear') {
  await closeDrawer(page);
  await page.evaluate((sel) => document.querySelector(sel)?.click(), gear);
  await page.waitForSelector('.drawer--open .drawer__body', { timeout: 5000 });
  await page.waitForFunction(() => {
    const p = document.querySelector('.drawer--open');
    if (!p) return false;
    const r = p.getBoundingClientRect();
    return r.right <= window.innerWidth + 1 && r.left >= 0; // fully slid in
  }, { timeout: 5000 });
}
