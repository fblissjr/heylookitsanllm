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

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptySettings();
    const parsed = JSON.parse(raw);
    const out = emptySettings();
    for (const k of Object.keys(out)) if (k in parsed) out[k] = parsed[k];
    return out;
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
  cache = emptySettings();
  scheduleSave();
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
  cache = emptySettings();
  for (const k of Object.keys(cache)) if (k in params) cache[k] = params[k];
  scheduleSave();
}

// Request-body params: only non-null keys; top_k and presence_penalty are
// only meaningful when > 0 (backend treats 0 as unset).
export function samplerParams() {
  const out = {};
  for (const key of Object.keys(PARAM_META)) {
    const v = cache[key];
    if (v === null || v === undefined) continue;
    if (key === 'top_k' && !(v > 0)) continue;
    if (key === 'presence_penalty' && !(v > 0)) continue;
    out[key] = v;
  }
  return out;
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

// `lead` lets a page prepend its own sections (e.g. chat's preset bar and
// per-conversation system prompt) inside the same panel card.
export function buildSettingsPanel({ caps = [], lead = [] } = {}) {
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
    ...lead,
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
