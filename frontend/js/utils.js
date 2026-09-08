// Shared DOM + formatting primitives. Keep dependency-free.

export function createEl(tag, props = {}, children = []) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') el.className = value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key in el) el[key] = value;
    else el.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    el.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return el;
}

// Coalesce repeated calls into one run per animation frame. `.cancel()`
// drops any pending frame (createPage calls it on teardown).
export function throttleToFrame(fn) {
  let raf = null;
  let lastArgs = null;
  const wrapped = (...args) => {
    lastArgs = args;
    if (raf !== null) return;
    raf = requestAnimationFrame(() => {
      raf = null;
      fn(...lastArgs);
    });
  };
  wrapped.cancel = () => {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
  };
  return wrapped;
}

// Coalesce repeated calls to at most one run per `ms`, on an animation frame.
//
// throttleToFrame is the right tool when a call is cheap; a STREAM painter is
// not. One paint per frame is up to 120/s on a ProMotion phone, and nobody
// reads faster than a fraction of that -- the extra frames are pure heat. The
// run is still aligned to a frame so the DOM write lands with the browser's
// own paint rather than fighting it. Leading-edge (the first call runs
// immediately) so a stream's first token is not held back.
export function throttleToInterval(fn, ms) {
  let timer = null;
  let raf = null;
  let lastRun = 0;
  let lastArgs = null;
  let pending = false;

  const fire = () => {
    if (!pending) return;
    pending = false;
    lastRun = performance.now();
    fn(...lastArgs);
  };

  const schedule = () => {
    timer = null;
    raf = requestAnimationFrame(() => { raf = null; fire(); });
  };

  const wrapped = (...args) => {
    lastArgs = args;
    pending = true;
    if (timer !== null || raf !== null) return;
    timer = setTimeout(schedule, Math.max(0, ms - (performance.now() - lastRun)));
  };
  wrapped.cancel = () => {
    clearTimeout(timer);
    timer = null;
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
    pending = false;
  };
  return wrapped;
}

const beforeUnloadHandler = (e) => {
  e.preventDefault();
  e.returnValue = '';
};

// One global guard; refcounted so overlapping users don't fight. NOT exported:
// every consumer goes through createUnloadGuard below, so the refcount cannot
// be unbalanced from a call site that has no teardown to disarm in.
let unloadGuards = 0;
const beforeUnloadGuard = {
  enable() {
    if (++unloadGuards === 1) window.addEventListener('beforeunload', beforeUnloadHandler);
  },
  disable() {
    if (unloadGuards > 0 && --unloadGuards === 0) {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
    }
  },
};

// Page-level ownership of that guard: one boolean per page, so a caller can
// answer "do I have unsaved work" as often as it likes and the refcount stays
// balanced by construction. Every real consumer has its enable() and its
// disable() structurally far apart -- a models config draft outlives the panel
// showing it (rebuilt on every save), a chat stream outlives the click that
// started it -- so a per-widget enable() leaks a refcount on each rebuild and
// leaves the dialog armed over the whole app, including pages that own no
// unsaved work at all. Teardown disarms, which is the exit hand-written
// bookkeeping forgets.
//
// This is for work only the USER can commit (a button they have not pressed).
// Where a page owns a debounced writer with a server home -- notebook's
// scheduleSave, the prompt sections, bindDocumentParams -- the flush-on-hide
// path is the better answer and a dialog would be a regression: it asks a
// question the app can just answer by saving.
export function createUnloadGuard(ctx) {
  let armed = false;
  let disposed = false;
  const set = (unsaved) => {
    // A page's async tails outlive the page. An unaborted GET landing after
    // teardown, or a PUT that REJECTS after it (the failure path re-writes
    // the draft it optimistically cleared), re-runs whatever computes
    // "unsaved" and would re-arm a guard nobody owns any more: there is no
    // teardown left to disarm it, and the next mount's guard is a different
    // closure whose set(false) early-returns on its own `armed`. That is a
    // leave-site dialog stuck on every page for the rest of the session.
    // The check belongs here rather than in each caller -- a per-caller
    // `ctx.alive` test leaves the hole open for the next consumer.
    if (disposed || Boolean(unsaved) === armed) return;
    armed = !armed;
    if (armed) beforeUnloadGuard.enable();
    else beforeUnloadGuard.disable();
  };
  ctx.onTeardown(() => { set(false); disposed = true; });
  return set;
}

// Page status line: plain text, danger color when it's an error.
export function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.style.color = isError ? 'var(--danger)' : '';
}

// Replace a <select>'s options with one per value.
export function fillOptions(select, values) {
  select.replaceChildren(...values.map((v) => createEl('option', { value: v }, [v])));
}

export function formatBytes(bytes) {
  if (bytes == null || !Number.isFinite(bytes)) return '--';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let i = -1;
  do { v /= 1024; i++; } while (v >= 1024 && i < units.length - 1);
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

// A token count the way context sizes are spoken: 4k, 40k, 1M. ONE speller
// for the chat context select, the models page and the prompt preview, so
// the same ceiling reads the same everywhere (base 1024, like the sizes
// models are published with).
export function formatTokens(n) {
  if (n == null || !Number.isFinite(n)) return '--';
  if (n >= 1048576) return `${(n / 1048576).toFixed(n % 1048576 ? 1 : 0)}M`;
  if (n >= 1024) return `${Math.round(n / 1024)}k`;
  return String(n);
}

export function debounce(fn, ms) {
  let timer = null;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn(...args); }, ms);
  };
  wrapped.flush = (...args) => {
    if (timer !== null) { clearTimeout(timer); timer = null; fn(...args); }
  };
  wrapped.cancel = () => { clearTimeout(timer); timer = null; };
  return wrapped;
}

// Auto-grow a textarea up to maxPx.
//
// NB the early return below hands sizing to CSS entirely -- INCLUDING maxPx.
// A caller passing a cap must have a matching `max-height` inside the
// `@supports (field-sizing: content)` block in app.css, or the cap silently
// disappears on every browser that supports it. Three fields shipped without
// one. A viewport-relative cap belongs there as `dvh`, not `vh`: this function
// reads `window.innerHeight`, which follows the mobile toolbar, and `vh` does
// not.
export function autoGrow(textarea, maxPx = 200) {
  if (!textarea) return;
  if (typeof CSS !== 'undefined' && CSS.supports && CSS.supports('field-sizing', 'content')) return;
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, maxPx)}px`;
}

// Mobile: a slide-in list pane (chat conversations, notebook list) covers most
// of the page; a tap on the visible content outside the pane and its toggle
// dismisses it. Wire on the page root; `insideSelectors` are the elements a
// click must NOT dismiss on (the pane itself, its toggle button).
export function dismissPaneOnOutsideClick(root, openClass, ...insideSelectors) {
  root.addEventListener('click', (e) => {
    if (root.classList.contains(openClass) &&
        !insideSelectors.some((sel) => e.target.closest(sel))) {
      root.classList.remove(openClass);
    }
  });
}

// Two-tap destructive confirm: first click arms the button briefly,
// second click within that window runs the action. Optional `when` predicate:
// arming only happens while it returns true -- otherwise the action runs on
// the first click (for buttons that are only sometimes destructive, e.g.
// preset Apply, which only overwrites a prompt when one would change).
// `target` (optional) returns a comparable value describing WHAT the confirmed
// action would do -- its destination and its payload. It is captured when the
// button arms and re-read on the confirming click: if it moved, the arm was a
// promise about something else, so the click re-arms instead of firing.
//
// This has to live in the primitive. A consumer can wire disarm() to its own
// controls, but the thing that changes the consequence is often OUTSIDE the
// component: the preset bar's Save writes the DOCUMENT's system prompt, which
// is edited in a different section of the drawer, so "click Save (arms on
// 'replace this text'), clear the prompt box, click Save (blanks the preset)"
// confirmed a write nobody previewed and no amount of disarm() wiring in the
// bar could have seen it. `when` alone cannot catch it either -- it answers
// "is something at stake", which was true both times, for different reasons.
export function armedConfirm(btn, action, armedLabel = 'Confirm?', when = null, target = null) {
  // innerHTML, not textContent: an ICON button (static SVG markup we author,
  // e.g. the sidebar's Del) must come back as the icon after a disarm, not
  // as an empty button. The armed label itself is still plain text.
  const original = btn.innerHTML;
  let armed = false;
  let timer = null;
  let armedFor = null; // what the pending arm is a promise ABOUT
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (armed && target && target() !== armedFor) disarm();
    if (armed || (when && !when())) {
      disarm();
      action();
      return;
    }
    arm();
  });

  function arm() {
    armed = true;
    armedFor = target?.() ?? null;
    btn.classList.add('btn--armed');
    btn.textContent = armedLabel;
    // Long enough to READ the confirmation and decide on a phone. At 3s a
    // careful reader's second tap landed as a fresh FIRST tap, which reads as
    // "the button didn't work" -- the cost fell on exactly the people the
    // confirmation is for. Safety does not rest on this: `target` already
    // makes a stale arm refuse to fire, so the timer only clears a visually
    // stale label.
    timer = setTimeout(disarm, 8000);
  }

  // Cancel a pending arm. `target` makes a stale arm HARMLESS; disarm makes it
  // VISIBLY gone, which is a different job -- a button still reading "Overwrite
  // prompt?" while aimed somewhere else is a lie even though clicking it is now
  // safe. Consumers call this from controls that re-aim (the preset bar's
  // select and name box). Exposed on the button, so a caller that never
  // re-aims can ignore it.
  function disarm() {
    if (!armed) return;
    clearTimeout(timer);
    armed = false;
    armedFor = null;
    btn.classList.remove('btn--armed');
    btn.innerHTML = original;
  }
  btn.disarm = disarm;
  return btn;
}


// ---------------------------------------------------------------------------
// Browser-local storage. ONE wrapper, because every raw call needs the same
// try/catch: Safari throws on access in private mode, quota is finite, and iOS
// EVICTS script-writable storage for a site the user has not visited lately.
// Everything stored here must therefore be a convenience whose loss changes
// nothing important -- per-document state lives on the document, on the server.
//
// The `heylook.` prefix is one namespace so a stale-key sweep is a prefix match.
// It replaced two spellings (`heylook-v3-*` dashes, `heylook.v3.*` dots) in
// v2.0.38: "v3" stopped meaning anything when the frontend left that mount in
// v1.79.76, and no single prefix reached both.
// ---------------------------------------------------------------------------

const LS_PREFIX = 'heylook.';

export function lsRead(key) {
  try { return localStorage.getItem(LS_PREFIX + key); } catch { return null; }
}

export function lsWrite(key, value) {
  try {
    if (value === null || value === undefined || value === '') localStorage.removeItem(LS_PREFIX + key);
    else localStorage.setItem(LS_PREFIX + key, value);
  } catch { /* private mode / quota -- in-memory only, by contract above */ }
}

// Keys this app used to write and no longer reads. Swept once at boot so a
// long-lived browser does not carry them forever -- there is no way for the
// user to clear one by hand on a phone, which is the case that decided it.
const RETIRED_KEYS = [
  'heylook-v3-settings',    // sampler bag: de-persisted in v2.0.38
  'heylook-v3-display',     // display prefs: removed with show_special_tokens
  'heylook-v3-scan-paths',  // one-off scan paths: no longer remembered
  'heylook.v3.chat.draft-prompt',  // renamed into the heylook. namespace
];

export function sweepRetiredStorage() {
  for (const key of RETIRED_KEYS) {
    try { localStorage.removeItem(key); } catch { /* nothing to do */ }
  }
}
