// Sampler settings. Every key defaults to null = "use backend cascade"
// (global -> thinking -> models.toml -> request). samplerParams() copies
// ONLY non-null keys so omitted params respect the cascade -- this is a
// real integration contract, preserve it exactly.

import { createEl } from './utils.js';

const STORAGE_KEY = 'heylook-v3-settings';

export const PARAM_META = {
  temperature:             { label: 'Temperature', type: 'number', min: 0, max: 2, step: 0.05, section: 'core' },
  max_tokens:              { label: 'Max tokens', type: 'number', min: 1, max: 65536, step: 1, section: 'core' },
  top_p:                   { label: 'Top-p', type: 'number', min: 0, max: 1, step: 0.01, section: 'core' },
  top_k:                   { label: 'Top-k', type: 'number', min: 0, max: 500, step: 1, section: 'core' },
  min_p:                   { label: 'Min-p', type: 'number', min: 0, max: 1, step: 0.01, section: 'advanced' },
  repetition_penalty:      { label: 'Repetition penalty', type: 'number', min: 0.5, max: 2, step: 0.01, section: 'advanced' },
  repetition_context_size: { label: 'Repetition context', type: 'number', min: 1, max: 8192, step: 1, section: 'advanced' },
  presence_penalty:        { label: 'Presence penalty', type: 'number', min: 0, max: 2, step: 0.01, section: 'advanced' },
  seed:                    { label: 'Seed', type: 'number', min: 0, max: Number.MAX_SAFE_INTEGER, step: 1, section: 'advanced' },
  enable_thinking:         { label: 'Enable thinking', type: 'checkbox', section: 'advanced', requiresCap: 'thinking' },
  // Thinking DEPTH, only meaningful with thinking on. The accepted set is
  // per-model (Qwen3.8 takes xhigh/medium/low and RAISES otherwise; harmony
  // models take low/medium/high), so this offers the union and the backend
  // rejects a value the request schema does not know. 'auto' = send nothing,
  // leaving the template's own default -- xhigh on Qwen3.8, which is why the
  // control exists at all.
  reasoning_effort:        { label: 'Thinking depth', type: 'select', options: ['low', 'medium', 'high', 'xhigh'], section: 'advanced', requiresCap: 'reasoning_effort' },
  // Target visual tokens per image; the backend snaps to what the model's
  // processor supports (gemma-4 buckets 70..1120, qwen continuous).
  vision_tokens:           { label: 'Vision tokens / image', type: 'number', min: 16, max: 16384, step: 1, section: 'advanced', requiresCap: 'vision' },
};

function emptySettings() {
  return Object.fromEntries(Object.keys(PARAM_META).map((k) => [k, null]));
}

// The "only known keys, everything else null" invariant in one place --
// load() and applySettings() both funnel through it.
function mergeKnown(src) {
  const out = emptySettings();
  for (const k of Object.keys(out)) if (k in src) out[k] = src[k];
  return out;
}

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? mergeKnown(JSON.parse(raw)) : emptySettings();
  } catch {
    return emptySettings();
  }
}

let cache = load();
let saveTimer = null;

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(cache)); }
    catch { /* storage full/unavailable -- settings stay in-memory */ }
  }, 300);
}

export function getSetting(key) { return cache[key]; }

// Sampler-change listeners -- fired on any panel mutation so a surface can
// persist the panel elsewhere (chat binds this to PUT the active conversation's
// `params`). Return value is an unsubscribe fn (call it
// in the page's teardown so listeners don't outlive the mount).
const samplerListeners = new Set();
export function onSettingsChange(cb) {
  samplerListeners.add(cb);
  return () => samplerListeners.delete(cb);
}
function fireSettingsChange() {
  for (const cb of samplerListeners) { try { cb(); } catch { /* isolate */ } }
}

export function setSetting(key, value) {
  cache[key] = value;
  scheduleSave();
  fireSettingsChange();
}

export function resetSettings() {
  applySettings({});
}

// Preset capture: every non-null key, raw. Unlike samplerParams() it keeps
// zeros -- a preset pinning top_k=0 records the user's panel state even
// though requests omit it.
export function snapshotSettings() {
  const out = {};
  for (const key of Object.keys(PARAM_META)) {
    const v = cache[key];
    if (v !== null && v !== undefined) out[key] = v;
  }
  return out;
}

// Preset apply: the preset's params become the whole panel state -- absent
// keys revert to null (backend cascade), matching "a preset IS the settings".
// `silent`: skip firing listeners -- used when HYDRATING the panel from a
// conversation's stored params, so loading a conversation doesn't immediately
// PUT its own params straight back.
export function applySettings(params, { silent = false } = {}) {
  cache = mergeKnown(params);
  scheduleSave();
  if (!silent) fireSettingsChange();
}

// Request-body params: the snapshot minus the knobs that are only
// meaningful when > 0 (backend treats 0 as unset). Pass the CURRENT model's
// `caps` to also drop capability-gated keys the model doesn't support --
// the panel hides those controls (requiresCap) but the cache keeps their
// values, and without this filter a value set on a capable model rides
// every request to an incapable one invisibly ("pinned") until Reset.
// The cache itself is untouched: switch back to a capable model and the
// value (and its control) return.
export function samplerParams(caps = null) {
  const out = snapshotSettings();
  if (!(out.top_k > 0)) delete out.top_k;
  if (!(out.presence_penalty > 0)) delete out.presence_penalty;
  if (caps) {
    for (const [key, meta] of Object.entries(PARAM_META)) {
      if (meta.requiresCap && !caps.includes(meta.requiresCap)) delete out[key];
    }
  }
  return out;
}

// The same bag spelled for /v1/messages (Phase 3b): DERIVED from
// samplerParams, never a second hand-written copy -- the one wire difference
// is that Messages says `thinking` where the OpenAI wire said
// `enable_thinking` (same tri-state: absent = the model's own default).
export function messagesParams(caps = null) {
  const { enable_thinking, ...out } = samplerParams(caps);
  if (enable_thinking !== undefined) out.thinking = enable_thinking;
  return out;
}

// Display prefs that ride the WIRE rather than the renderer. `show_special_tokens`
// is one: the server strips a model's declared specials before the text is
// streamed AND before it is persisted, so "show" cannot be a render-time choice
// on these surfaces -- the client has to ask for the unstripped text.
//
// Kept OUT of samplerParams/messagesParams on purpose: those are the sampler bag,
// everything in them reaches the model, and a document's stored `params` is that
// same bag (CLAUDE.md). This is display state, so it is spread in beside the bag
// at the two send sites, never merged into it.
//
// Takes the PAGE'S OWN declared list -- the same array it hands the drawer as
// `displayPrefs` -- so declaring a pref and sending it cannot come apart.
// Hardcoding the body here instead let a page offer a checkbox it never sent
// (and would have sent a second pref from every caller regardless of what that
// caller claimed to honor). Keys with no `wire` spelling in DISPLAY_META are
// render-time prefs and are skipped: a page may honor those without any request
// changing shape. Explore speaks the same wire but declares nothing -- it is a
// token-ARRAY surface (§6's other mechanism: flag specials by token id).
export function displayWireFields(prefs = []) {
  const out = {};
  for (const key of prefs) {
    const wire = DISPLAY_META[key]?.wire;
    if (wire) out[wire] = getDisplayPref(key) === true;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Per-DOCUMENT sampler settings. ONE mechanism shared by every page whose doc
// carries `params` (chat conversations, notebooks, ...) so sampler tuning lives
// with the document on the server, not as browser-global state -- and the pages
// don't branch into copies of the same wiring.
// ---------------------------------------------------------------------------

// Bind the sampler panel to a document's `params`: on any panel change,
// debounce-PUT the whole snapshot to the ACTIVE document. `activeId()` -> the
// current doc id (null = no doc yet; the panel rides along until create seeds
// it). `updateDoc(id, body)` = the page's update call. Returns an unsubscribe fn
// (register in the page's teardown). The debounce timer is per-binding (closure),
// and `id` is captured at schedule time so a doc switch mid-debounce still writes
// to the one the edit was for.
// `onHide` is the page's ctx.onHide: the binder registers its own
// last-moment flush there, exactly as it owns its teardown flush -- a
// consumer that had to remember either is how notebook shipped without
// one. `updateDoc(id, body, opts)` receives `{ keepalive }` on those flushes.
export function bindDocumentParams({ activeId, updateDoc, onError, onHide, delay = 400 }) {
  let timer = null;
  let pending = false;
  // Fire the debounced PUT NOW. Exposed as .flush on the returned teardown
  // so a generate can settle the store first: overrides carry SET panel
  // values past the debounce window, but a CLEARED value is expressed by
  // absence, which overrides cannot spell -- only the params PUT can
  // (review finding 2026-08-13: reset temperature + fast Send still
  // generated at the stored value).
  const flush = (opts = {}) => {
    clearTimeout(timer);
    timer = null;
    if (!pending) return Promise.resolve();
    pending = false;
    const id = activeId();
    if (!id) return Promise.resolve();
    return Promise.resolve(updateDoc(id, { params: snapshotSettings() }, opts))
      .catch(onError || (() => {}));
  };
  const unsub = onSettingsChange(() => {
    const id = activeId();
    if (!id) return;
    pending = true;
    clearTimeout(timer);
    timer = setTimeout(flush, delay);
  });
  const offHide = onHide?.(() => flush({ keepalive: true }));
  // Teardown FLUSHES (it used to cancel): leaving the page inside the
  // debounce window is the same typed-and-believed-saved shape as the
  // drawer closing under focus. The id was captured when the edit was
  // scheduled, so the write stays correct after the mount is gone.
  const teardown = () => { offHide?.(); unsub(); flush(); };
  teardown.flush = flush;
  return teardown;
}

// Load a document's stored params into the panel WITHOUT firing listeners, so
// selecting/loading a doc doesn't immediately PUT its own params back.
export function hydrateDocParams(doc) {
  applySettings(doc?.params ?? {}, { silent: true });
}

// ---------------------------------------------------------------------------
// Global display preferences -- render toggles, NOT sampler params. Kept in a
// SEPARATE store (own key, own cache) from PARAM_META so a display flag can
// never leak into samplerParams()/snapshotSettings() and reach the model. These
// are the cross-cutting "how do we render tokens" prefs every surface reads
// (DESIGN.md §6). Display-only, by contract.
// ---------------------------------------------------------------------------

const DISPLAY_STORAGE_KEY = 'heylook-v3-display';

export const DISPLAY_META = {
  show_special_tokens: {
    label: 'Show special tokens',
    type: 'checkbox',
    default: true,   // honesty-first: shown by default (DESIGN.md §6)
    // The request field this pref rides on (v1.79.6): the DECODED-TEXT surfaces
    // (chat, notebook) send it and the server then skips its declared-specials
    // strip (DESIGN.md §6). Absent `wire` = a render-time pref, honored without
    // any request changing shape. WHICH pages honor it is the page's own
    // `displayPrefs` declaration, not a flag here -- "no surface honors this" is
    // exactly "no page lists it", so a second gate here could only disagree with
    // that one, silently. The token-ARRAY surfaces (explore, jspace) are the
    // other half of "one preference, two render mechanisms" and declare nothing.
    wire: 'show_special_tokens',
    help: 'Keep the model\'s special tokens (<|im_end|>, <bos>, role markers) in '
        + 'its replies instead of stripping them. Display-only -- never changes '
        + 'what is sent to the model. Applies to replies generated from now on: '
        + 'chat stores a reply exactly as it was parsed, so this cannot add '
        + 'markers to (or remove them from) a reply that already exists. '
        + 'gguf models are never stripped, so turning this off does nothing there. '
        + 'Chat strips the markers back out when it replays a reply as history, so '
        + 'they never re-enter a prompt; notebook writes the reply into the document, '
        + 'where they are ordinary text you can see and edit -- and send.',
  },
};

function loadDisplay() {
  const out = Object.fromEntries(Object.entries(DISPLAY_META).map(([k, m]) => [k, m.default]));
  try {
    const raw = localStorage.getItem(DISPLAY_STORAGE_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      for (const k of Object.keys(out)) if (typeof saved[k] === 'boolean') out[k] = saved[k];
    }
  } catch { /* fall back to defaults */ }
  return out;
}

let displayCache = loadDisplay();

export function getDisplayPref(key) { return displayCache[key]; }

// Stored only. A display pref reaches the model on the NEXT generation
// (displayWireFields puts it on the wire), so there is nothing to re-render
// when one flips -- which is what each pref's own help text says. An
// onDisplayChange subscription mechanism used to live here promising live
// re-render; it never had a subscriber, so the notify loop always ran over an
// empty set and the promise was never kept.
export function setDisplayPref(key, value) {
  displayCache[key] = value;
  try { localStorage.setItem(DISPLAY_STORAGE_KEY, JSON.stringify(displayCache)); }
  catch { /* in-memory only */ }
}

// ---------------------------------------------------------------------------
// Data-driven panel. `caps` filters params gated on model capabilities
// (e.g. enable_thinking only shows for thinking-capable models).
// ---------------------------------------------------------------------------

function bindControl(key, meta) {
  if (meta.type === 'select') {
    // '' is the empty option and means "don't send the key at all" -- for
    // reasoning_effort that leaves the model's chat template on its own
    // default, which is NOT the same as any of the listed values.
    const sel = createEl('select', { id: `set-${key}`, class: 'input' },
      [createEl('option', { value: '' }, ['auto']),
       ...meta.options.map((o) => createEl('option', { value: o }, [o]))]);
    sel.value = cache[key] ?? '';
    sel.addEventListener('change', () => setSetting(key, sel.value || null));
    return sel;
  }
  if (meta.type === 'checkbox') {
    const box = createEl('input', { id: `set-${key}`, type: 'checkbox', checked: cache[key] === true });
    // unchecking sets null (cascade), NOT false -- false would override the
    // backend's per-model thinking default.
    box.addEventListener('change', () => setSetting(key, box.checked ? true : null));
    return box;
  }
  const input = createEl('input', {
    id: `set-${key}`,
    class: 'input',
    type: 'number',
    min: meta.min, max: meta.max, step: meta.step,
    placeholder: 'auto',
    value: cache[key] ?? '',
  });
  input.addEventListener('change', () => {
    const v = input.value.trim();
    setSetting(key, v === '' ? null : Number(v));
  });
  return input;
}

export function buildSettingsPanel({ caps = [] } = {}) {
  const rows = { core: [], advanced: [] };
  const controls = [];

  for (const [key, meta] of Object.entries(PARAM_META)) {
    if (meta.requiresCap && !caps.includes(meta.requiresCap)) continue;
    const control = bindControl(key, meta);
    controls.push({ key, meta, control });
    rows[meta.section].push(createEl('div', { class: 'settings-row' }, [
      createEl('label', { for: `set-${key}` }, [meta.label]),
      control,
    ]));
  }

  const resetBtn = createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Reset to defaults']);
  resetBtn.addEventListener('click', () => {
    resetSettings();
    for (const { meta, control } of controls) {
      if (meta.type === 'checkbox') control.checked = false;
      else control.value = '';
    }
  });

  return createEl('div', { class: 'settings-panel' }, [
    createEl('h3', {}, ['Sampling']),
    ...rows.core,
    rows.advanced.length
      ? createEl('details', {}, [
          createEl('summary', {}, ['Advanced']),
          createEl('div', {}, rows.advanced),
        ])
      : null,
    resetBtn,
  ]);
}

// Display-prefs section (the second section-kind, alongside Sampling and
// per-page extras). Model-agnostic, so no capability gating -- but NOT
// page-agnostic: `honored` is the PAGE's list of pref keys it actually reads
// (drawer contribution `displayPrefs`), and that ONE list is the gate. It is the
// same array the page passes to displayWireFields(), so a checkbox rendered here
// is a checkbox that does something -- which is the whole rule (a control that
// silently does nothing on the page you are looking at is worse than no control).
// Explore and jspace read token ids rather than this, so they declare nothing.
// Returns null when the page honors none, so the drawer omits the section.
export function buildDisplayPanel(honored = []) {
  const rows = Object.entries(DISPLAY_META).filter(
    ([key]) => honored.includes(key)
  ).map(([key, meta]) => {
    const box = createEl('input', { id: `disp-${key}`, type: 'checkbox', checked: getDisplayPref(key) === true });
    box.addEventListener('change', () => setDisplayPref(key, box.checked));
    return createEl('div', { class: 'settings-row' }, [
      createEl('label', { for: `disp-${key}`, title: meta.help || '' }, [meta.label]),
      box,
    ]);
  });
  if (!rows.length) return null;
  return createEl('div', { class: 'settings-panel' }, [
    createEl('h3', {}, ['Display']),
    ...rows,
  ]);
}
