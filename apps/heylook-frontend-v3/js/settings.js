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

export function setSetting(key, value) {
  cache[key] = value;
  scheduleSave();
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
export function applySettings(params) {
  cache = mergeKnown(params);
  scheduleSave();
}

// Request-body params: the snapshot minus the knobs that are only
// meaningful when > 0 (backend treats 0 as unset).
export function samplerParams() {
  const out = snapshotSettings();
  if (!(out.top_k > 0)) delete out.top_k;
  if (!(out.presence_penalty > 0)) delete out.presence_penalty;
  return out;
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
    help: 'Render <|im_start|>, <think>, role markers etc. as distinct tokens. '
        + 'Display-only -- never changes what is sent to the model. Edit surfaces '
        + 'always expose raw tokens regardless of this toggle.',
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
const displayListeners = new Set();

export function getDisplayPref(key) { return displayCache[key]; }

// Fires listeners so every open surface re-renders live when a pref flips.
export function setDisplayPref(key, value) {
  displayCache[key] = value;
  try { localStorage.setItem(DISPLAY_STORAGE_KEY, JSON.stringify(displayCache)); }
  catch { /* in-memory only */ }
  for (const cb of displayListeners) { try { cb(key, value); } catch { /* isolate */ } }
}

// Subscribe to display-pref changes; returns an unsubscribe fn (call it in a
// page's teardown so listeners don't outlive the mount).
export function onDisplayChange(cb) {
  displayListeners.add(cb);
  return () => displayListeners.delete(cb);
}

// ---------------------------------------------------------------------------
// Data-driven panel. `caps` filters params gated on model capabilities
// (e.g. enable_thinking only shows for thinking-capable models).
// ---------------------------------------------------------------------------

function bindControl(key, meta) {
  if (meta.type === 'checkbox') {
    const box = createEl('input', { type: 'checkbox', checked: cache[key] === true });
    // unchecking sets null (cascade), NOT false -- false would override the
    // backend's per-model thinking default.
    box.addEventListener('change', () => setSetting(key, box.checked ? true : null));
    return box;
  }
  const input = createEl('input', {
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
      createEl('label', {}, [meta.label]),
      control,
    ]));
  }

  const resetBtn = createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Reset to defaults']);
  resetBtn.addEventListener('click', () => {
    resetSettings();
    for (const { key, meta, control } of controls) {
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

// Global display-prefs section (the second section-kind, alongside Sampling and
// per-page extras). Rendered in the shared drawer on every page -- these prefs
// are model-agnostic, so no capability gating.
export function buildDisplayPanel() {
  const rows = Object.entries(DISPLAY_META).map(([key, meta]) => {
    const box = createEl('input', { type: 'checkbox', checked: getDisplayPref(key) === true });
    box.addEventListener('change', () => setDisplayPref(key, box.checked));
    return createEl('div', { class: 'settings-row' }, [
      createEl('label', { title: meta.help || '' }, [meta.label]),
      box,
    ]);
  });
  return createEl('div', { class: 'settings-panel' }, [
    createEl('h3', {}, ['Display']),
    ...rows,
  ]);
}
