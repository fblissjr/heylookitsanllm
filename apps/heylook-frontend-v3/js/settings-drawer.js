// Shared, global settings drawer -- an app-shell singleton, NOT a per-page
// panel. A persistent gear (in the desktop nav + a floating affordance on
// phone widths) opens a right slide-over; backdrop-click or Escape closes it.
//
// The router runs ONE page at a time, so a single "current contribution" is
// correct: each page registers what it wants in the drawer during setup() and
// unregisters on teardown. The drawer composes, top to bottom:
//
//   ...contribution.sections()   -- page-owned lead sections (chat and
//                                   notebook: the shared preset bar + their
//                                   system-prompt editor)
//   Sampling panel               -- buildSettingsPanel({caps}); 'disabled' renders
//                                   it read-only + greedy note; 'hidden' omits it
//   Display panel                -- buildDisplayPanel() (global, model-agnostic)
//   ...contribution.extras()     -- page-owned trailing controls (jspace toggles,
//                                   explore's logprobs note)
//
// Focus guard (migrated from chat's old inline panel): a background/async
// repaint must never destroy uncommitted text in a field the user is editing.
// requestRebuild() honors it -- it skips while focus is inside the render
// target. Discrete, user-initiated actions that must repaint (the preset bar's
// Apply button, etc.) pass { force: true } -- the exact analog of
// the old rebuildSettingsPanel(), which was always unguarded.

import { createEl } from './utils.js';
import { buildSettingsPanel, buildDisplayPanel } from './settings.js';

let mounted = false;
let panelEl = null;   // the slide-over <aside>
let bodyEl = null;    // render target (the ONLY thing render() replaces)
let backdropEl = null;
let closeBtn = null;
let gearBtn = null;
let bottomGearBtn = null;   // phone-width opener: a trailing item in #bottom-nav
let appEl = null;     // #app -- made inert while the modal drawer is open

let isOpen = false;
let lastOpener = null;
let current = null;   // the registered page contribution (or null)

// ---------------------------------------------------------------------------
// mount (once, by app.js)
// ---------------------------------------------------------------------------

export function mountSettingsDrawer(navDesktop, navBottom) {
  if (mounted) return;
  mounted = true;
  appEl = document.getElementById('app');

  gearBtn = createEl('button', {
    type: 'button', class: 'nav-item drawer-gear',
    title: 'Settings', 'aria-label': 'Open settings',
  }, ['⚙ Settings']);
  gearBtn.addEventListener('click', () => open(gearBtn));
  navDesktop.append(gearBtn);

  // Phone widths hide #sidebar (and with it the desktop gear), so the drawer
  // needs its own reachable affordance there -- a trailing gear item in the
  // bottom nav (a bottom composer, e.g. chat, leaves no room for a floating
  // FAB). #bottom-nav rides its own <=767px show/hide, and lives inside #app so
  // the inert focus-trap covers it for free. Same open() target.
  bottomGearBtn = createEl('button', {
    type: 'button', class: 'nav-item drawer-gear-bottom',
    title: 'Settings', 'aria-label': 'Open settings',
  }, ['⚙']);
  bottomGearBtn.addEventListener('click', () => open(bottomGearBtn));
  navBottom.append(bottomGearBtn);

  backdropEl = createEl('div', { class: 'drawer-backdrop' });
  backdropEl.addEventListener('click', () => close());

  bodyEl = createEl('div', { class: 'drawer__body' });
  closeBtn = createEl('button', {
    type: 'button', class: 'btn btn--ghost btn--sm drawer__close',
    'aria-label': 'Close settings',
  }, ['Close']);
  closeBtn.addEventListener('click', () => close());

  panelEl = createEl('aside', {
    class: 'drawer', role: 'dialog', 'aria-label': 'Settings', 'aria-modal': 'true',
  }, [
    createEl('header', { class: 'drawer__head' }, [
      createEl('h2', {}, ['Settings']),
      closeBtn,
    ]),
    bodyEl,
  ]);
  panelEl.inert = true; // not focusable/tabbable while closed

  document.body.append(backdropEl, panelEl);

  // Escape closes. Focus lives inside the drawer while open (modal backdrop),
  // so this never races page-level Escape handlers (jspace unpin, explore
  // deselect), which are bound to their page roots.
  document.addEventListener('keydown', (e) => {
    if (isOpen && e.key === 'Escape') close();
  });

  // A route change (nav link, or browser back/forward -- a common way to dismiss
  // a modal) must not leave #app sealed under the newly-mounted page. Registered
  // before app.js's own hashchange->navigate listener, so inert is cleared before
  // the next page mounts.
  window.addEventListener('hashchange', () => { if (isOpen) close(); });
}

// ---------------------------------------------------------------------------
// registration (one current contribution -- the router runs one page at a time)
// ---------------------------------------------------------------------------

// contribution = {
//   caps?():    string[]                 -- model capabilities (gates enable_thinking)
//   samplers?:  'enabled'|'disabled'|'hidden'   (default 'hidden')
//   sections?():Node[]                   -- lead sections (rendered first)
//   extras?():  Node[]                   -- trailing controls (rendered last)
//   onOpen?():  void                     -- fired each time the drawer opens
//                                           (e.g. lazily refresh presets)
// }
export function registerSettings(contribution) {
  current = contribution;
  if (isOpen) render();
  return () => {
    if (current === contribution) {
      current = null;
      if (isOpen) render(); // fall back to the global Display panel
    }
  };
}

// Page-level entry points (e.g. chat's top-bar gear) open the same singleton
// drawer. Pass the triggering button so close() restores focus to it.
export { open as openSettings };

// Ask the drawer to re-render. No-op while closed (matches the old
// rebuildSettingsPanel, which no-op'd while hidden). Honors the focus guard
// unless force -- see the module header.
export function requestRebuild({ force = false } = {}) {
  if (!mounted || !isOpen) return;
  if (!force && bodyEl.contains(document.activeElement)) return;
  render();
}

// ---------------------------------------------------------------------------
// open / close
// ---------------------------------------------------------------------------

function open(opener) {
  if (!mounted || isOpen) return;
  isOpen = true;
  lastOpener = opener ?? gearBtn;
  render();
  panelEl.inert = false;
  // Modal: seal the rest of the page so Tab can't escape the dialog. The
  // drawer + backdrop are body children (siblings of #app), so they stay live;
  // the bottom-nav gear is inside #app, so it's sealed too (correct: it's the
  // opener, behind the backdrop).
  if (appEl) appEl.inert = true;
  panelEl.classList.add('drawer--open');
  backdropEl.classList.add('drawer-backdrop--open');
  // Focus the close button (in the header, OUTSIDE bodyEl) so the drawer is
  // keyboard-reachable without tripping the focus guard on an onOpen refresh.
  closeBtn.focus();
  current?.onOpen?.();
}

function close() {
  if (!isOpen) return;
  // Flush before teardown: a field the user is still focused in commits on
  // `change`, which never fires once replaceChildren removes the element --
  // an Escape/hashchange close would silently discard the typed value (the
  // bug that ate chat's system prompt). Blurring here fires it for EVERY
  // commit-on-change field (sampler numbers included), at the one place all
  // close paths converge.
  if (bodyEl.contains(document.activeElement)) document.activeElement.blur();
  isOpen = false;
  panelEl.classList.remove('drawer--open');
  backdropEl.classList.remove('drawer-backdrop--open');
  panelEl.inert = true;
  // Un-seal the page BEFORE restoring focus -- the opener lives inside #app.
  if (appEl) appEl.inert = false;
  // Clear the body: open() always re-renders, so a closed drawer holds nothing.
  // Prevents a page's contributed nodes (e.g. jspace's fixed-id toggles) from
  // lingering in the hidden drawer after that page unmounts (stale retention +
  // a transient duplicate-id window on re-entry).
  bodyEl.replaceChildren();
  lastOpener?.focus?.();
}

// ---------------------------------------------------------------------------
// render (replaces ONLY bodyEl's children)
// ---------------------------------------------------------------------------

function render() {
  const children = [];

  if (current?.sections) children.push(...current.sections());

  const samplers = current?.samplers ?? 'hidden';
  if (samplers !== 'hidden') {
    const panel = buildSettingsPanel({ caps: current?.caps?.() ?? [] });
    if (samplers === 'disabled') {
      for (const el of panel.querySelectorAll('input, button')) el.disabled = true;
      panel.append(createEl('div', { class: 'settings-note muted small' },
        ['Sampling is fixed (greedy) here.']));
    }
    children.push(panel);
  }

  const displayPanel = buildDisplayPanel();   // null while no display pref is wired yet
  if (displayPanel) children.push(displayPanel);

  if (current?.extras) children.push(...current.extras());

  bodyEl.replaceChildren(...children);
}
