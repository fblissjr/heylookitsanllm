// Shared DOM interaction helpers for the suites. Kept intentionally small; page-
// specific logic lives in the suite files.

import { waitFor } from './harness.mjs';

// Click the first element matching `selector` whose trimmed text === `text`.
export async function clickByText(page, selector, text) {
  const handle = await page.evaluateHandle((sel, txt) => {
    return [...document.querySelectorAll(sel)].find((e) => e.textContent.trim() === txt) || null;
  }, selector, text);
  const el = handle.asElement();
  if (!el) {
    await handle.dispose();
    throw new Error(`no <${selector}> with text "${text}"`);
  }
  await el.click();
  await handle.dispose();
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
