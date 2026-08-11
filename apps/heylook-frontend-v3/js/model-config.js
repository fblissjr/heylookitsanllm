// Per-model config editor -- schema-driven from GET /v1/admin/model-options.
//
// Every control is generated from the option schema (type, bounds, enum,
// default, effect class, arg spelling), so a new backend config field appears
// here without a frontend change. The effect classes drive the layout -- they
// exist so a UI can say WHEN a change lands, and this panel is the first
// consumer that distinguishes all of them:
//
//   per_request / applies_live / descriptive -> "Applies immediately"
//   requires_reload                          -> "Requires reload" (the panel
//                                               offers Reload now on a loaded
//                                               model after such a save)
//   load_time_only                           -> disabled, with the reason
//   identity                                 -> never sent by the endpoint
//
// Value contract: an ABSENT models.toml key means "inherit the default", so
// every control has an explicit unset state (empty input / "default" option)
// and clearing one PATCHes null, which removes the key server-side. The saved
// state lives in model.config; unsaved edits live in the caller's `draft`
// object so the panel survives the models page's list re-renders.

import { createEl, armedConfirm, debounce } from './utils.js';
import { api } from './api.js';

const LIVE_EFFECTS = new Set(['per_request', 'applies_live', 'descriptive']);

// Which fields never render is declared ON the field (`ui: "hidden"` in the
// backend schema -- gguf's host/port/server_binary/startup_timeout_s, mlx's
// derived `vision` mirror), not in a frontend name-list: one source, read by
// this editor, the summary chip, and the E2E check alike.
const isHidden = (f) => f.ui === 'hidden';

// The collapsed-row summary chip: which of this entry's stored keys are worth
// announcing on the list ("why is this one configured differently"). Only
// process-shaping scalars -- descriptive fields are written by every import
// (modalities, supports_thinking) and would put a chip on every row, and
// paths/arrays are too long to be a chip.
export function configSummary(config, fields) {
  if (!config || !fields?.length) return null;
  const parts = [];
  for (const f of fields) {
    if (f.effect === 'descriptive' || isHidden(f)) continue;
    const v = config[f.name];
    if (v == null || Array.isArray(v) || typeof v === 'object') continue;
    if (typeof v === 'string' && v.length > 16) continue;
    parts.push(typeof v === 'boolean' ? `${f.name} ${v ? 'on' : 'off'}` : `${f.name} ${v}`);
  }
  if (!parts.length) return null;
  const shown = parts.slice(0, 4);
  if (parts.length > 4) shown.push(`+${parts.length - 4} more`);
  return shown.join(' · ');
}

// models.toml stores toml values; the wire carries JSON. Both are typed, so
// the only stringly parsing here is what the <input> element forces on us.
function parseControlValue(field, raw) {
  if (raw === '' || raw == null) return { value: null };
  switch (field.type) {
    case 'integer': {
      const n = Number(raw);
      if (!Number.isInteger(n)) return { error: `${field.name}: not a whole number` };
      return { value: n };
    }
    case 'number': {
      const n = Number(raw);
      if (!Number.isFinite(n)) return { error: `${field.name}: not a number` };
      return { value: n };
    }
    case 'boolean':
      return { value: raw === 'true' };
    case 'array':
      // One element per LINE, not comma-separated: array elements can
      // legitimately contain commas (extra_args: --tensor-split "3,1",
      // -ot "a=CPU,b=GPU"), and a comma split silently rewrites them into a
      // different argv that only fails at the next spawn.
      return { value: raw.split('\n').map((s) => s.trim()).filter(Boolean) };
    default:
      return { value: raw };
  }
}

// The control's string form of a stored value (inverse of parseControlValue).
function toControlValue(field, value) {
  if (value == null) return '';
  if (field.type === 'array') return Array.isArray(value) ? value.join('\n') : String(value);
  return String(value);
}

function defaultLabel(field) {
  if (field.default == null) return null;
  if (field.type === 'boolean') return field.default ? 'on' : 'off';
  if (field.type === 'array' && Array.isArray(field.default)) return field.default.join(', ');
  return String(field.default);
}

function boundsLabel(field) {
  const parts = [];
  if (field.minimum != null) parts.push(`min ${field.minimum}`);
  if (field.exclusiveMinimum != null) parts.push(`> ${field.exclusiveMinimum}`);
  if (field.maximum != null) parts.push(`max ${field.maximum}`);
  if (field.exclusiveMaximum != null) parts.push(`< ${field.exclusiveMaximum}`);
  return parts.join(', ');
}

function buildHint(field) {
  const bits = [];
  if (field.arg) bits.push(createEl('span', { class: 'cfg-field__arg' }, [field.arg]));
  const def = defaultLabel(field);
  if (def != null) bits.push(`default ${def}`);
  const bounds = boundsLabel(field);
  if (bounds) bits.push(bounds);
  if (field.type === 'array') bits.push('one per line');
  if (field.reason) bits.push(field.reason);
  if (!bits.length) return null;
  const el = createEl('div', { class: 'cfg-field__hint muted small' });
  bits.forEach((b, i) => {
    if (i > 0) el.append(' · ');
    el.append(b);
  });
  return el;
}

// One control. Selects get an explicit "default" option; free inputs use
// empty-means-default with the default as placeholder. Everything reports
// edits into `onEdit(name, rawValue)` immediately -- Save decides what to send.
function buildControl(field, rawValue, inputId, onEdit) {
  const disabled = field.effect === 'load_time_only';

  if (field.type === 'boolean' || field.enum) {
    // Tri-state (default / choices) needs a select; a checkbox cannot say
    // "unset". Boolean choices render as on/off.
    const choices = field.type === 'boolean'
      ? [['true', 'on'], ['false', 'off']]
      : field.enum.map((v) => [String(v), String(v)]);
    const def = defaultLabel(field);
    const select = createEl('select', { id: inputId, disabled }, [
      createEl('option', { value: '' }, [def != null ? `default (${def})` : 'default']),
      ...choices.map(([v, label]) => createEl('option', { value: v }, [label])),
    ]);
    select.value = rawValue;
    select.addEventListener('change', () => onEdit(field.name, select.value));
    return select;
  }

  if (field.type === 'array') {
    const area = createEl('textarea', {
      id: inputId,
      class: 'input cfg-field__lines',
      rows: 3,
      value: rawValue,
      placeholder: defaultLabel(field) ?? 'default',
      disabled,
    });
    area.addEventListener('input', () => onEdit(field.name, area.value));
    return area;
  }

  const isNumeric = field.type === 'integer' || field.type === 'number';
  const input = createEl('input', {
    id: inputId,
    class: 'input',
    type: isNumeric ? 'number' : 'text',
    value: rawValue,
    placeholder: defaultLabel(field) ?? 'default',
    disabled,
  });
  if (isNumeric) {
    input.step = field.type === 'integer' ? '1' : 'any';
    if (field.minimum != null) input.min = field.minimum;
    if (field.maximum != null) input.max = field.maximum;
  }
  input.addEventListener('input', () => onEdit(field.name, input.value));
  return input;
}

function fieldRow(field, rawValue, idPrefix, onEdit) {
  const inputId = `${idPrefix}-${field.name}`;
  const row = createEl('div', {
    class: `cfg-field${field.effect === 'load_time_only' ? ' cfg-field--fixed' : ''}`,
  }, [
    createEl('label', { class: 'cfg-field__label', for: inputId }, [field.name]),
    buildControl(field, rawValue, inputId, onEdit),
  ]);
  const hint = buildHint(field);
  if (hint) row.append(hint);
  return row;
}

function sectionEl(title, note, rows) {
  return createEl('section', { class: 'cfg-section' }, [
    createEl('h3', { class: 'cfg-section__title' }, [title]),
    note ? createEl('div', { class: 'muted small' }, [note]) : null,
    ...rows,
  ]);
}

// ---------------------------------------------------------------------------
// Memory fit meter (design doc §5 -- "the heart of it"). Every number is the
// SERVER's (POST /{id}/fit wraps heylook_llm.ram_fit); this renderer never
// derives fit client-side -- a reimplementation would drift from ram_fit
// immediately, and on any failure it says "fit unavailable" rather than
// guessing. hard_working_set carries the engine asymmetry: over the Metal
// working set is FAIL for MLX (refuses above it) but WARN for gguf
// (llama.cpp loads past it and degrades into paging).

const gib = (v) => `${v.toFixed(1)} GiB`;

function buildFitMeter({ model, overrides, onGate }) {
  const rowsEl = createEl('div', { class: 'cfg-fit__rows' });
  // The observed line (design §5's closing loop): once the model is LOADED,
  // show what it actually holds resident -- the user sees how good the
  // sizing above was and learns whether to trust it. Distinctly labelled
  // measured-after-load; best-effort (the fit rows stand alone without it).
  const observedEl = createEl('div', { class: 'cfg-fit__row cfg-fit__observed', hidden: true });
  // role=status: the verdict flips live as fields are edited -- announced,
  // not just shown (DESIGN.md §7).
  const verdictEl = createEl('div', { class: 'cfg-fit__verdict', role: 'status' });
  const el = createEl('section', { class: 'cfg-section cfg-fit' }, [
    createEl('h3', { class: 'cfg-section__title' }, ['Memory fit']),
    rowsEl, observedEl, verdictEl,
  ]);

  async function refreshObserved() {
    if (!model.loaded) return;
    try {
      const metrics = await api.systemMetrics();
      const mb = metrics?.models?.[model.id]?.memory_mb;
      if (mb == null) return;
      observedEl.replaceChildren(
        createEl('span', { class: 'cfg-fit__label' }, ['Resident now']),
        createEl('span', { class: 'cfg-fit__value' }, [gib(mb / 1024)]),
        createEl('span', { class: 'muted small' }, ['measured after load']),
      );
      observedEl.hidden = false;
    } catch { /* stays hidden */ }
  }

  const row = (label, value, note) => createEl('div', { class: 'cfg-fit__row' }, [
    createEl('span', { class: 'cfg-fit__label' }, [label]),
    createEl('span', { class: 'cfg-fit__value' }, [value]),
    note ? createEl('span', { class: 'muted small' }, [note]) : null,
  ]);

  function render(r) {
    const rows = [row('Weights', gib(r.weights_gb))];
    for (const note of r.sizing_notes) {
      rows.push(createEl('div', { class: 'cfg-fit__note muted small' }, [note]));
    }
    if (r.working_set_gb != null) {
      rows.push(row('Metal working set', gib(r.working_set_gb),
        r.hard_working_set ? 'hard limit — MLX refuses above it'
          : 'advisory — llama.cpp pages past it'));
    }
    if (r.kv_headroom_gb != null) {
      const kv = row('Headroom for KV', gib(r.kv_headroom_gb));
      if (r.kv_headroom_gb < 0) kv.classList.add('cfg-fit__row--danger');
      rows.push(kv);
    }
    rows.push(row('Reclaimable RAM', gib(r.reclaimable_gb),
      'total − anonymous − wired'));
    rowsEl.replaceChildren(...rows);

    const pieces = [];
    const ws = r.lines.find((l) => l.ceiling === 'metal_working_set');
    const ram = r.lines.find((l) => l.ceiling === 'reclaimable_ram');
    const buf = r.lines.find((l) => l.ceiling === 'metal_max_buffer');
    if (ram?.verdict === 'fail') {
      pieces.push(`Won't fit: needs ${gib(ram.need_gb)} (weights + headroom), `
        + `~${gib(ram.have_gb)} reclaimable.`);
    }
    if (ws && ws.verdict !== 'pass') {
      const over = gib(ws.need_gb - ws.have_gb);
      pieces.push(ws.verdict === 'fail'
        ? `Over the Metal working set by ${over} — MLX refuses above it.`
        : `Over the Metal working set by ${over} — llama.cpp will still load `
          + `this; Metal stops guaranteeing residency and you degrade into paging.`);
    }
    if (buf) pieces.push(`Weights exceed the ${gib(buf.have_gb)} per-allocation cap — ${buf.note}.`);

    verdictEl.className = `cfg-fit__verdict cfg-fit__verdict--${r.verdict}`;
    verdictEl.replaceChildren(
      r.verdict === 'pass' ? 'Fits.' : pieces.join(' '),
      // Server-gated actionability: non-null ONLY over the working set while
      // iogpu.wired_limit_mb is at its OS default. Present -> show verbatim.
      ...(r.sysctl_suggest_mb != null ? [createEl('div', { class: 'cfg-fit__sysctl' }, [
        'Raise it (resets on reboot): ',
        createEl('code', {}, [`sudo sysctl iogpu.wired_limit_mb=${r.sysctl_suggest_mb}`]),
      ])] : []),
    );
    // Only a FAIL gates Load (gguf's over-WS is a warn by design), and a
    // loaded model is already past loading -- nothing to gate.
    onGate?.(r.verdict === 'fail' && !model.loaded
      ? 'Does not fit memory (see the Configure panel)' : null);
  }

  let ctl = null;
  async function refresh() {
    ctl?.abort();
    ctl = new AbortController();
    try {
      const report = await api.adminModelFit(
        model.id, { config_overrides: overrides() }, { signal: ctl.signal });
      render(report);
    } catch (err) {
      if (err.name === 'AbortError') return;
      // 422 = the (candidate) model_path doesn't exist -- name it; anything
      // else is just "unavailable". NEVER fall back to a client-side guess.
      verdictEl.className = 'cfg-fit__verdict';
      verdictEl.textContent = err.status === 422
        ? 'fit unavailable — model_path does not exist'
        : 'fit unavailable';
      rowsEl.replaceChildren();
      onGate?.(null);
    }
  }

  return { el, refresh, refreshObserved, scheduleRefresh: debounce(refresh, 300) };
}

// createModelConfigEditor({ model, fields, draft, initialNote, onError, onSaved, onReload, onReset })
//
// - model:  the AdminModelResponse row (id, provider, loaded, config,
//           stale_reload_fields, ...). stale_reload_fields is SERVER truth
//           ("saved but the loaded process still runs the old value") and is
//           what re-arms the Reload offer on every rebuild -- panel-local
//           state cannot survive a remount, and a client-side copy of this
//           fact drifts (it did).
// - fields: the option schema's field list for model.provider (ui:"hidden"
//           fields are filtered here, so callers pass the schema verbatim)
// - draft:  caller-owned {fieldName: rawControlValue} of unsaved edits; the
//           editor mutates it so a page re-render can rebuild without loss
// - initialNote: one-shot status text to show (and announce) after mounting
//           -- the save outcome survives the post-save rebuild through it
// - onError(msg): route a failure to the page's status area (page invariant:
//           errors have ONE home)
// - onSaved(updatedModel, noteText): the PATCH landed; caller swaps its row
//           model, re-renders (this editor is REBUILT -- do nothing after),
//           and hands noteText to the rebuilt panel as initialNote
// - onReload(): caller runs its unload + warm-load cycle (armed-confirmed)
// - onReset(): drafts were dropped; caller rebuilds the panel from saved state
//
// The returned { el, announce } contract: caller invokes announce() AFTER
// appending el to the document -- a live region only announces text written
// into an already-mounted node, so baking the note into the initial DOM
// would render it silent to assistive tech.
// - onFitGate(reason|null): optional -- the fit meter's Load gate. A non-null
//           reason means the model FAILS fit (MLX over the working set, or
//           reclaimable RAM on either provider) and the page should disable
//           its Load affordance with that reason; null lifts the gate
//           (including on fit-unavailable -- never block on missing info).
export function createModelConfigEditor({ model, fields: allFields, draft, initialNote, onError, onSaved, onReload, onReset, onFitGate }) {
  const fields = allFields.filter((f) => !isHidden(f));
  const idPrefix = `mcfg-${model.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  // Editor-local baseline (what dirty is measured against). A copy, updated
  // on each successful save -- reading model.config would go stale after the
  // first save (the panel deliberately isn't rebuilt then), making a revert
  // of a just-saved value read as "no change".
  const saved = { ...(model.config || {}) };
  let busy = false;

  const savedRaw = (field) => toControlValue(field, saved[field.name]);
  const currentRaw = (field) =>
    Object.hasOwn(draft, field.name) ? draft[field.name] : savedRaw(field);

  // Server-derived: which saved requires_reload values the loaded process
  // has not picked up yet. Drives the Reload offer across rebuilds/remounts.
  const staleFields = model.loaded ? (model.stale_reload_fields || []) : [];

  const noteEl = createEl('div', { class: 'cfg-note muted small', role: 'status' });
  const saveBtn = createEl('button', { class: 'btn btn--sm' }, ['Save']);
  const resetBtn = createEl('button', { class: 'btn btn--sm' }, ['Reset']);
  const reloadBtn = createEl('button', { class: 'btn btn--sm', hidden: !staleFields.length }, ['Reload now']);

  const dirtyFields = () =>
    fields.filter((f) => Object.hasOwn(draft, f.name) && draft[f.name] !== savedRaw(f));

  const setNote = (text) => { noteEl.textContent = text; };

  // The pre-save note names what a save will DO, per effect class -- never a
  // reload cost for a live field (that lie trains the user to pay reloads for
  // changes already in effect), never silence about a field that needs one.
  const pendingNote = (dirty) => {
    const reload = dirty.filter((f) => f.effect === 'requires_reload').map((f) => f.name);
    const live = dirty.filter((f) => f.effect !== 'requires_reload').map((f) => f.name);
    const parts = [];
    if (reload.length) {
      parts.push(`${reload.join(', ')} — ${model.loaded
        ? 'needs a reload after save (the loaded model keeps running as-is)'
        : 'applies on next load'}`);
    }
    if (live.length) parts.push(`${live.join(', ')} — applies on save`);
    return `Changed: ${parts.join('; ')}.`;
  };

  const syncSaveState = () => {
    const dirty = dirtyFields();
    saveBtn.disabled = busy || dirty.length === 0;
    resetBtn.disabled = busy || dirty.length === 0;
    if (dirty.length) setNote(pendingNote(dirty));
  };

  // Candidate config for the fit meter: the dirty fields' PARSED values over
  // the stored config (null = reset, the PATCH spelling). Unparseable drafts
  // are skipped -- the meter answers for the closest well-formed candidate
  // rather than going blank on every half-typed number.
  const fitOverrides = () => {
    const overrides = {};
    for (const field of dirtyFields()) {
      const { value, error } = parseControlValue(field, draft[field.name]);
      if (!error) overrides[field.name] = value;
    }
    return overrides;
  };
  const fitMeter = buildFitMeter({ model, overrides: fitOverrides, onGate: onFitGate });

  const onEdit = (name, rawValue) => {
    draft[name] = rawValue;
    syncSaveState();
    fitMeter.scheduleRefresh();
  };

  async function save() {
    const dirty = dirtyFields();
    if (busy || !dirty.length) return;

    const config = {};
    for (const field of dirty) {
      const { value, error } = parseControlValue(field, draft[field.name]);
      if (error) { onError(`Not saved -- ${error}`); return; }
      // null both clears a set key and is skipped for a never-set one; the
      // backend treats null as "remove the key", so sending it is harmless
      // either way and simpler than distinguishing.
      config[field.name] = value;
    }

    busy = true;
    saveBtn.textContent = 'Saving…';
    syncSaveState();
    try {
      const result = await api.adminUpdateModel(model.id, { config });
      const reloadNeeded = result.reload_required_fields || [];
      for (const field of dirty) delete draft[field.name];
      const parts = ['Saved.'];
      if (reloadNeeded.length) {
        parts.push(model.loaded
          ? `Takes effect on reload: ${reloadNeeded.join(', ')}.`
          : `Applies on next load: ${reloadNeeded.join(', ')}.`);
      }
      if (result.warning) parts.push(result.warning);
      // The page swaps the row model and REBUILDS this panel (so the row's
      // reload marker and chip actually repaint); the rebuilt panel restores
      // the outcome via initialNote and the response's stale_reload_fields.
      onSaved(result.model, parts.join(' '));
      return;
    } catch (err) {
      onError(`Save failed: ${err.message}`);
    }
    busy = false;
    saveBtn.textContent = 'Save';
    syncSaveState();
  }

  saveBtn.addEventListener('click', save);
  resetBtn.addEventListener('click', () => {
    // Revert the form to persisted values: drop the drafts and let the page
    // rebuild the panel from the saved config.
    for (const f of fields) delete draft[f.name];
    onReset();
  });
  // Armed confirm (the danger-zone grammar): a reload on a big model is
  // minutes of disk I/O, so the first click arms rather than fires.
  armedConfirm(reloadBtn, () => { reloadBtn.hidden = true; onReload(); }, 'Confirm reload?');

  // --- layout: a strict PARTITION of the field set. `fixed` wins over
  // `advanced` (a load_time_only field must render disabled-with-reason
  // exactly once, never twice with a duplicate id), and the Advanced title
  // only claims a reload when every field in it actually requires one --
  // stamping "(requires reload)" over a live field would be the exact lie
  // pendingNote exists to avoid.
  const fixed = fields.filter((f) => f.effect === 'load_time_only');
  const advanced = fields.filter((f) => f.ui === 'advanced' && f.effect !== 'load_time_only');
  const live = fields.filter((f) => LIVE_EFFECTS.has(f.effect) && f.ui !== 'advanced');
  const reload = fields.filter((f) => f.effect === 'requires_reload' && f.ui !== 'advanced');

  const rows = (list) => list.map((f) => fieldRow(f, currentRaw(f), idPrefix, onEdit));

  const children = [fitMeter.el];
  if (live.length) {
    children.push(sectionEl('Applies immediately', null, rows(live)));
  }
  if (reload.length) {
    children.push(sectionEl(
      'Requires reload',
      model.loaded
        ? 'Saved to models.toml now; the loaded model keeps running as-is until reloaded.'
        : 'Saved to models.toml; applies when the model loads.',
      rows(reload),
    ));
  }
  if (advanced.length) {
    const allReload = advanced.every((f) => f.effect === 'requires_reload');
    children.push(createEl('details', { class: 'cfg-section' }, [
      createEl('summary', { class: 'cfg-section__title' },
        [allReload ? 'Advanced (requires reload)' : 'Advanced']),
      ...rows(advanced),
    ]));
  }
  if (fixed.length) {
    children.push(sectionEl('Fixed for this process', null, rows(fixed)));
  }
  if (children.length === 1) { // only the fit meter -- no editable sections
    children.push(createEl('div', { class: 'muted small' },
      [`No editable options for provider "${model.provider}".`]));
  }

  children.push(createEl('div', { class: 'cfg-actions' }, [saveBtn, resetBtn, reloadBtn, noteEl]));

  syncSaveState();
  fitMeter.refresh(); // first paint; needs no DOM, lands whenever it lands
  fitMeter.refreshObserved(); // loaded models also show the measured resident line
  // The mount note: the save outcome carried across the rebuild, or the
  // standing server-derived reload reminder. Written by announce() AFTER the
  // caller mounts the element, so the live region actually announces it.
  const mountNote = initialNote
    || (staleFields.length ? `Saved changes pending reload: ${staleFields.join(', ')}.` : '');
  return {
    el: createEl('div', { class: 'model-config' }, children),
    announce: () => { if (mountNote && !noteEl.textContent) setNote(mountNote); },
  };
}
