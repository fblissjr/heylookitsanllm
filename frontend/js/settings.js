// Sampler settings. Every key defaults to null = "use backend cascade"
// (global -> thinking -> models.toml -> request). samplerParams() copies
// ONLY non-null keys so omitted params respect the cascade -- this is a
// real integration contract, preserve it exactly.

import { createEl } from './utils.js';

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
  // THREE states, not a checkbox (v1.79.62): null = the model's own default
  // (the server's cascade answer, labelled with its value when the page
  // knows it), true = on, false = off. The checkbox before it could only say
  // "on" or "unset", so an unset thinking model showed an unticked box while
  // the server had already decided -- and there was no way to ask for OFF.
  enable_thinking:         { label: 'Thinking', type: 'tristate', section: 'advanced', requiresCap: 'thinking',
                             defaultLabel: 'Model default', onLabel: 'On', offLabel: 'Off' },
  // Thinking DEPTH, only meaningful with thinking on. The accepted set is
  // per-model (Qwen3.8 takes xhigh/medium/low and RAISES otherwise; harmony
  // models take low/medium/high), so this offers the union and the backend
  // rejects a value the request schema does not know. 'auto' = send nothing,
  // leaving the template's own default -- xhigh on Qwen3.8, which is why the
  // control exists at all.
  // The offered list is the UNION across model families, so a value valid for
  // one model reaches another's chat template and is rejected there -- for a
  // gguf model a raised jinja exception comes back as a 500. The backend
  // cannot narrow this per model (the accepted set lives in the template, and
  // for gguf inside the GGUF's own metadata), so the honest move is to say so
  // here rather than to imply every value works everywhere.
  reasoning_effort:        { label: 'Thinking depth', type: 'select', options: ['low', 'medium', 'high', 'xhigh'], section: 'advanced', requiresCap: 'reasoning_effort',
                             note: 'Accepted values differ by model — a rejected one fails the request. "auto" always works.' },
  // Target visual tokens per image; the backend snaps to what the model's
  // processor supports (gemma-4 buckets 70..1120, qwen continuous).
  vision_tokens:           { label: 'Vision tokens / image', type: 'number', min: 16, max: 16384, step: 1, section: 'advanced', requiresCap: 'vision' },
};

function emptySettings() {
  return Object.fromEntries(Object.keys(PARAM_META).map((k) => [k, null]));
}

// Is `v` a STRUCTURALLY usable value for `key`? Type and shape only.
//
// It deliberately does NOT range-check against PARAM_META's min/max. Those are
// input-widget hints -- step, slider bounds, what a sane number looks like --
// and they are NOT the backend's validation range: `max_tokens` caps at 65536
// here while a gguf model can hold a context far past that, so a stored 100000
// is legal, useful, and something the panel has no business rejecting.
//
// Rejecting it was worse than useless, because a dropped key does not stay
// dropped: `snapshotSettings()` omits nulls, so the next panel edit PUTs a bag
// without it and the stored value is ERASED from the document or preset, with
// nothing on screen having said so. A filter meant to stop a bad value reaching
// the wire would have silently destroyed a good one. Structural checks have no
// such failure mode -- a string where a number belongs, or a NaN, is garbage
// under every backend range.
function valid(key, v) {
  if (v === null || v === undefined) return false;   // absent IS the cascade
  const meta = PARAM_META[key];
  if (!meta) return false;
  if (meta.type === 'number') return typeof v === 'number' && Number.isFinite(v);
  if (meta.type === 'tristate') return v === true || v === false;
  if (meta.type === 'select') return meta.options.includes(v);
  if (meta.type === 'checkbox') return typeof v === 'boolean';
  return true;
}

// The "only known keys, everything else null" invariant in one place --
// every hydration funnels through it.
//
// It filters KEYS and, since v2.0.38, the SHAPE of values -- localStorage was
// never the only source, since `presets.params` and a document's stored
// `params` hydrate this same panel through this same function. What it cannot
// do is decide a number is out of range: see `valid()` for why that filter was
// removed rather than tuned. A value the BACKEND refuses still returns a 422
// that names the field but not where it was stored; that is the honest limit
// of a client-side check, and destroying the value is not a better answer.
function mergeKnown(src) {
  const out = emptySettings();
  for (const k of Object.keys(out)) if (k in src && valid(k, src[k])) out[k] = src[k];
  return out;
}

// NOT persisted, deliberately (v2.0.38). The panel is a VIEW of a document:
// `hydrateDocParams` overwrites it on every document select, so a stored copy
// was overwritten before anyone could read it in every case but one -- seeding
// the NEXT new document when none is open. Keeping that seed across a reload
// was the entire durable behaviour, and it was already lost the moment a reload
// with any conversation present hydrated conversations[0] over it. What the
// stored copy DID do was cross surfaces: chat and notebook share this module,
// so the last-hydrated document's params seeded the other page's next new
// document, per-browser and invisibly. In-memory keeps that within a session
// (documentScopeNote says so) instead of making it durable.
let cache = emptySettings();

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
// Data-driven panel. `caps` filters params gated on model capabilities
// (e.g. enable_thinking only shows for thinking-capable models).
// ---------------------------------------------------------------------------

// Tri-state <select> value spellings: '' = null (model default), 'on', 'off'.
const TRISTATE_FROM_VALUE = { '': null, on: true, off: false };
const TRISTATE_TO_VALUE = (v) => (v === true ? 'on' : v === false ? 'off' : '');

// How a resolved model default reads in a placeholder. null/undefined means
// the page does not know (no model row yet, or a key the server does not
// report) -- callers fall back to the literal word "auto", which is the
// honest answer for "we cannot tell you".
function defaultText(v) {
  if (v === null || v === undefined) return null;
  if (v === true) return 'on';
  if (v === false) return 'off';
  return String(v);
}

// `lookup(key)` rather than a snapshot object: the caller decides where a
// default comes from, and `enable_thinking` still answers from a different
// source than the rest. (It also used to depend on the live thinking switch,
// which is why the indirection exists at all -- see the lookup itself.)
function bindControl(key, meta, lookup = () => null) {
  if (meta.type === 'tristate') {
    // The default option NAMES the value it stands for when the page knows
    // it (`modelDefaults[key]`, the admin row's thinking_default) -- a
    // "model default" that hides whether it means on or off is the exact
    // mystery the control exists to end. Unknown = plain "Model default".
    const known = lookup(key);
    const suffix = known === true ? ' (on)' : known === false ? ' (off)' : '';
    const sel = createEl('select', { id: `set-${key}`, class: 'input' }, [
      createEl('option', { value: '' }, [`${meta.defaultLabel}${suffix}`]),
      createEl('option', { value: 'on' }, [meta.onLabel]),
      createEl('option', { value: 'off' }, [meta.offLabel]),
    ]);
    sel.value = TRISTATE_TO_VALUE(cache[key]);
    sel.addEventListener('change', () => setSetting(key, TRISTATE_FROM_VALUE[sel.value] ?? null));
    return sel;
  }
  if (meta.type === 'select') {
    // '' is the empty option and means "don't send the key at all" -- for
    // reasoning_effort that leaves the model's chat template on its own
    // default, which is NOT the same as any of the listed values.
    const shown = defaultText(lookup(key));
    const sel = createEl('select', { id: `set-${key}`, class: 'input' },
      [createEl('option', { value: '' }, [shown ? `auto (${shown})` : 'auto']),
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
    // The model's REAL resolved value, not the word "auto" -- a greyed
    // placeholder already reads as "this is what you get if you leave it
    // alone", and "auto" left the reader to generate something to find out.
    placeholder: defaultText(lookup(key)) ?? 'auto',
    value: cache[key] ?? '',
  });
  input.addEventListener('change', () => {
    const v = input.value.trim();
    setSetting(key, v === '' ? null : Number(v));
  });
  return input;
}

// The panel's scope line, composed in ONE place so chat and notebook cannot
// drift apart on the wording (they differ by a noun). `hasActive` is the
// difference that actually matters to the reader: with a document open these
// controls ARE that document's stored params and every edit writes to it;
// without one they are the browser-side seed the next new document seeds from.
// Nothing on screen said which, and selecting a document silently replaces
// every value in the panel -- the root of the "did my settings just change?"
// confusion (v3 user guide, rough edges).
// The no-document half is deliberately explicit about TWO things the panel used
// to leave unsaid: the values are a seed rather than saved state, and the seed
// is whatever was last loaded ANYWHERE -- chat and notebook share one in-memory
// cache, so a notebook's params seed a new conversation. Naming it is the
// chosen answer; scoping the cache per page was priced and declined (it threads
// a scope through every accessor and the drawer).
export function documentScopeNote(noun, hasActive) {
  return hasActive
    ? `Applies to this ${noun} — changes save as you make them.`
    : `Seeds the next ${noun} you start, from whatever was last open. Not saved anywhere yet.`;
}

// `scope` is a resolved string (see documentScopeNote) or null for a surface
// with nothing useful to say. `modelDefaults` is what the current model
// resolves an UNSET key to, keyed like PARAM_META (today: enable_thinking
// from the admin row's thinking_default) -- read by the tri-state control.
export function buildSettingsPanel({ caps = [], scope = null, modelDefaults = {},
                                    samplerDefaults = null } = {}) {
  const rows = { core: [], advanced: [] };
  const controls = [];

  // `samplerDefaults` is ONE bag since v2.0.33. It was `{off,on}` and this
  // read had a resolver to pick a half, because the anti-loop overlay moved
  // presence_penalty off the thinking switch and the panel's thinking control
  // could disagree with the number shown. That overlay is gone, the halves
  // became identical, and the resolver went with them.
  //
  // enable_thinking still answers from `modelDefaults` -- the tri-state labels
  // the MODEL's own default, which does not move when the user picks.
  const lookup = (key) => {
    if (key === 'enable_thinking' || !samplerDefaults) return modelDefaults[key] ?? null;
    return samplerDefaults[key] ?? null;
  };

  for (const [key, meta] of Object.entries(PARAM_META)) {
    if (meta.requiresCap && !caps.includes(meta.requiresCap)) continue;
    const control = bindControl(key, meta, lookup);
    // Shown only while the key is overridden, so its presence IS the "you
    // changed this" signal and there is nothing extra on screen otherwise.
    // Not a hover reveal -- state, not pointer -- so DESIGN.md §7's
    // touch-fallback rule does not apply; the hit area is padded to a real
    // tap target under `hover:none` in app.css.
    // Hidden by VISIBILITY, not by the `hidden` attribute or `display`, and
    // the difference is layout: the button keeps its box either way, so a row
    // does not jog sideways the moment a key becomes overridden -- which it
    // would under `display:none`, by the button's whole width, and by more on
    // a phone where the tap target is larger. `visibility:hidden` also leaves
    // the a11y tree and the tab order, so an inert row exposes nothing.
    const reset = createEl('button', {
      class: 'settings-row__reset', type: 'button',
      title: 'Back to the model default',
      'aria-label': `Reset ${meta.label} to the model default`,
    }, ['\u21ba']);
    const row = createEl('div', { class: 'settings-row' }, [
      createEl('label', { for: `set-${key}` }, [
        meta.label,
        meta.note ? createEl('span', { class: 'settings-row__note muted small' }, [meta.note]) : null,
      ]),
      // The control and its reset share ONE wrapper: .settings-row is a
      // two-child space-between flex row and a third child re-spaces it.
      createEl('div', { class: 'settings-row__control' }, [control, reset]),
    ]);
    const sync = () => {
      // DERIVED from samplerParams, not from `cache[key] != null`. The two
      // disagree: samplerParams drops a top_k or presence_penalty of 0
      // (`!(v > 0)`) and drops any capability-gated key the model lacks, so a
      // typed 0 used to light the row accent and offer a reset for a value the
      // server never sees and never applies. Marking is a CLAIM about what the
      // model is running, so it has to be read off the thing that decides it.
      const overridden = key in samplerParams(caps);
      row.classList.toggle('settings-row--overridden', overridden);
      reset.classList.toggle('settings-row__reset--on', overridden);
    };
    // Registered AFTER bindControl's own handler, so the cache is already
    // updated by the time this reads it.
    control.addEventListener('change', () => { sync(); if (key === 'enable_thinking') syncDefaults(); });
    reset.addEventListener('click', () => {
      setSetting(key, null);
      if (meta.type === 'checkbox') control.checked = false; else control.value = '';
      sync();
      if (key === 'enable_thinking') syncDefaults();
    });
    controls.push({ key, meta, control, sync });
    sync();
    rows[meta.section].push(row);
  }

  // Repaint every placeholder against the current thinking state. Only the
  // keys the overlay moves actually change, but repainting all of them keeps
  // this from having to know WHICH keys those are -- that list lives in the
  // sampler TOML, and a second copy here is the drift this repo names as a
  // defect with a delay.
  function syncDefaults() {
    for (const { key, meta, control } of controls) {
      if (meta.type === 'number') control.placeholder = defaultText(lookup(key)) ?? 'auto';
      else if (meta.type === 'select') {
        const shown = defaultText(lookup(key));
        const blank = control.querySelector('option[value=""]');
        if (blank) blank.textContent = shown ? `auto (${shown})` : 'auto';
      }
    }
  }

  // "Clear all overrides", not "Reset to defaults": this sets every value to
  // null, which hands each one back to the BACKEND cascade (global -> thinking
  // -> models.toml). The behaviour was always right; "defaults" just read as
  // something global while the button in fact rewrites the open document's
  // params. No confirm -- sampler values are trivially recoverable, and a
  // confirm here would train click-through past the ones that protect work.
  const resetBtn = createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Clear all overrides']);
  resetBtn.addEventListener('click', () => {
    resetSettings();
    for (const { meta, control, sync } of controls) {
      if (meta.type === 'checkbox') control.checked = false;
      else control.value = ''; // selects + tristates: '' is the unset option
      sync();
    }
    syncDefaults();
  });

  return createEl('div', { class: 'settings-panel' }, [
    createEl('h3', {}, ['Sampling']),
    scope ? createEl('div', { class: 'settings-note muted small' }, [scope]) : null,
    ...rows.core,
    // OPEN by default (owner ask 2026-09-04): thinking and its depth live in
    // here, and a collapsed group hid the one control people were looking
    // for. Still a <details> so it can be folded on a short phone screen.
    rows.advanced.length
      ? createEl('details', { open: true }, [
          createEl('summary', {}, ['Advanced']),
          createEl('div', {}, rows.advanced),
        ])
      : null,
    resetBtn,
  ]);
}

