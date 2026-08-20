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
export function armedConfirm(btn, action, armedLabel = 'Confirm?', when = null) {
  const original = btn.textContent;
  let armed = false;
  let timer = null;
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (armed || (when && !when())) {
      clearTimeout(timer);
      armed = false;
      btn.classList.remove('btn--armed');
      btn.textContent = original;
      action();
    } else {
      armed = true;
      btn.classList.add('btn--armed');
      btn.textContent = armedLabel;
      timer = setTimeout(() => {
        armed = false;
        btn.classList.remove('btn--armed');
        btn.textContent = original;
      }, 3000);
    }
  });
  return btn;
}
