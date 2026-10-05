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

// ---- the drawer's preset section (js/preset-bar.js) -------------------------
// Every helper here needs the drawer OPEN, and dispatches the control's own
// handler instead of hit-testing a click: checks leave the drawer scrolled
// (the system-prompt chip focuses its textarea), and a coordinate click on a
// row under the sticky header fails with "Node is either not clickable".

// The provenance line: "No preset", "From preset X", or "X, modified: ...".
export async function provenanceText(page) {
  return page.$eval('.drawer--open .preset-provenance', (el) => el.textContent);
}

// Open the Apply picker (a no-op when it is already open) and wait for its
// entries. Opening it copies nothing anywhere.
export async function openPresetPicker(page) {
  await page.evaluate(() => {
    const root = document.querySelector('.drawer--open .preset-section');
    if (!root.querySelector('.preset-picker').hidden) return;
    [...root.querySelectorAll('.preset-row button')]
      .find((b) => b.textContent.trim() === 'Apply preset…').click();
  });
  await page.waitForFunction(() => {
    const picker = document.querySelector('.drawer--open .preset-picker');
    return picker && !picker.hidden && picker.children.length > 0;
  }, { timeout: 5000 });
}

// One click on a picker entry's Apply button. Returns the button's label
// AFTER the click, or null when the click applied (the section rebuilds and
// the entry is gone) -- so a caller can tell "armed" from "fired".
export async function clickPresetApply(page, name) {
  return page.evaluate((n) => {
    const entry = [...document.querySelectorAll('.drawer--open .preset-option')]
      .find((o) => o.dataset.name === n);
    if (!entry) throw new Error(`no picker entry named ${n}`);
    const btn = entry.querySelector('button');
    btn.click();
    return btn.isConnected ? btn.textContent.trim() : null;
  }, name);
}

// Apply a preset through the picker. `armed` STATES THE EXPECTATION rather
// than adapting to what happens: true = the entry must arm first ("Replace
// prompt?") and is then confirmed; false = it must fire on the first click.
// Either mismatch throws, because both are the behaviour under test.
export async function applyPreset(page, name, { armed = false } = {}) {
  await openPresetPicker(page);
  const after = await clickPresetApply(page, name);
  if (armed) {
    if (after !== 'Replace prompt?') {
      throw new Error(`Apply "${name}" was expected to arm, but ${after === null ? 'fired at once' : `reads "${after}"`}`);
    }
    await clickPresetApply(page, name);
  } else if (after !== null) {
    throw new Error(`Apply "${name}" was expected to fire at once, but it reads "${after}"`);
  }
}

// Open the Save sheet (a no-op when it is already open).
export async function openSaveSheet(page) {
  await page.evaluate(() => {
    const root = document.querySelector('.drawer--open .preset-section');
    if (!root.querySelector('.preset-save').hidden) return;
    [...root.querySelectorAll('.preset-row button')]
      .find((b) => b.textContent.trim() === 'Save…').click();
  });
}

// Save as a new preset, through the Save sheet: set the name, press the button.
export async function saveAsNewPreset(page, name) {
  await openSaveSheet(page);
  await page.evaluate((n) => {
    const sheet = document.querySelector('.drawer--open .preset-save');
    const input = sheet.querySelector('input');
    input.value = n;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    [...sheet.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Save as a new preset').click();
  }, name);
}

// The Save sheet's "Overwrite X" option as the sheet offers it right now:
// its label, or null when it is not offered (no stamped preset, or nothing
// differs from it). Opens the sheet.
export async function updateOptionLabel(page) {
  await openSaveSheet(page);
  return page.evaluate(() => {
    const btn = document.querySelector('.drawer--open .preset-save__update');
    return btn && !btn.hidden ? btn.textContent.trim() : null;
  });
}

// One click on the "Overwrite X" option. Returns its label right after the
// click: the armed question ("Overwrite X?") on a first press, its own label
// once a confirming press went ahead (the write is asynchronous, so the
// caller waits on the wire or the store).
export async function clickUpdateOption(page) {
  await openSaveSheet(page);
  return page.evaluate(() => {
    const btn = document.querySelector('.drawer--open .preset-save__update');
    if (!btn || btn.hidden) throw new Error('the Save sheet offers no Overwrite option');
    btn.click();
    return btn.isConnected ? btn.textContent.trim() : null;
  });
}

// Write the document back to the preset it came from: the option arms on
// EVERY press and the second press confirms. Throws if it fires unarmed,
// because that is the behaviour under test.
export async function overwriteStampedPreset(page) {
  const armed = await clickUpdateOption(page);
  if (!/^Overwrite .+\?$/.test(armed ?? '')) {
    throw new Error(`the overwrite option did not arm (it reads ${JSON.stringify(armed)})`);
  }
  await clickUpdateOption(page);
}

// ---- the preset STORE, read and cleaned up directly -------------------------
// What a check asserts about a stored preset comes from the store, never from
// a control that might be showing a cached list.
export async function storedPresets(page) {
  return page.evaluate(async () => (await (await fetch('/v1/presets')).json()).presets ?? []);
}

// Cleanup only (presets survive /v1/data/clear). A check that is ABOUT
// deleting uses the Presets page's own button instead.
export async function deleteStoredPreset(page, name) {
  await page.evaluate(async (n) => {
    const { presets } = await (await fetch('/v1/presets')).json();
    for (const p of presets.filter((x) => x.name === n)) {
      await fetch(`/v1/presets/${p.id}`, { method: 'DELETE' });
    }
  }, name);
}

// ---- the Presets page (js/pages/presets.js) ---------------------------------
// A card's button by its label, as an ElementHandle (caller disposes).
export async function presetCardButton(page, name, label) {
  const handle = await page.evaluateHandle((n, l) => {
    const card = [...document.querySelectorAll('.preset-card')].find((c) => c.dataset.name === n);
    return [...(card?.querySelectorAll('button') ?? [])].find((b) => b.textContent.trim() === l) || null;
  }, name, label);
  const el = handle.asElement();
  if (!el) {
    await handle.dispose();
    throw new Error(`no "${label}" button on the preset card "${name}"`);
  }
  return el;
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
// A settings row's <label> is [name text node, source element, note element]
// since v2.0.175 (settings.js: the blank field's default source sits under
// its name), so the row is found by the label's FIRST text node, never by its
// whole textContent -- that reads "Temperatureheylook default".
export async function settingsInputValue(page, label) {
  return page.evaluate((lbl) => {
    const row = [...document.querySelectorAll('.settings-panel .settings-row')]
      .find((r) => r.querySelector('label')?.firstChild?.textContent.trim() === lbl);
    return row?.querySelector('input')?.value ?? null;
  }, label);
}

export async function setSettingsInput(page, label, value) {
  await page.evaluate((lbl, val) => {
    const row = [...document.querySelectorAll('.settings-panel .settings-row')]
      .find((r) => r.querySelector('label')?.firstChild?.textContent.trim() === lbl);
    const input = row.querySelector('input');
    input.value = val;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, label, value);
}

// The settings/presets/sysprompt controls all live in the app-shell
// settings drawer now (js/settings-drawer.js), not inline on the page. The
// drawer is a MODAL: while open it makes #app `inert`, and its backdrop covers
// the page -- so a puppeteer click aimed at the (inert) sidebar gear lands on
// the backdrop and closes it instead. It also survives a same-document (hash)
// navigation. Both make a naive "click the gear" flaky, so these helpers reset
// to a known-closed, #app-live state first.

// Close the drawer if open and GUARANTEE #app is interactable again. Clicks the
// drawer's own Close button (it's inside the drawer, never inert), then clears
// inert defensively so a leaked-open drawer never seals the page for the next click.
// Waits for BOTH the drawer to go and the backdrop to leave the render tree --
// closed, it is `display: none`, but that switch is held for the length of the
// closing fade (`display ... allow-discrete` in app.css), and until it lands
// the backdrop still covers #app, so a too-early click on page content lands
// on the backdrop instead of the button.
export async function closeDrawer(page) {
  await page.evaluate(() => {
    document.querySelector('.drawer--open .drawer__close')?.click();
    const app = document.getElementById('app');
    if (app) app.inert = false;
  });
  await page.waitForFunction(() => {
    if (document.querySelector('.drawer--open')) return false;
    const bd = document.querySelector('.drawer-backdrop');
    return !bd || getComputedStyle(bd).display === 'none';
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
