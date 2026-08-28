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

// One global guard; refcounted so overlapping users don't fight.
let unloadGuards = 0;
export const beforeUnloadGuard = {
  enable() {
    if (++unloadGuards === 1) window.addEventListener('beforeunload', beforeUnloadHandler);
  },
  disable() {
    if (unloadGuards > 0 && --unloadGuards === 0) {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
    }
  },
};

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

// Two-tap destructive confirm: first click arms the button for 3s,
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
  const original = btn.textContent;
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
    timer = setTimeout(disarm, 3000);
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
    btn.textContent = original;
  }
  btn.disarm = disarm;
  return btn;
}
