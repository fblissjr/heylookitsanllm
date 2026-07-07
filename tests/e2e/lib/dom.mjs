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
