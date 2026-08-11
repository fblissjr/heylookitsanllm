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

import { createEl, armedConfirm } from './utils.js';
import { api } from './api.js';

const LIVE_EFFECTS = new Set(['per_request', 'applies_live', 'descriptive']);

// Deliberately not rendered at all (stricter than disabled -- absence is the
// UI's choice; the effect class is the floor, not the ceiling). Per the
// owner-approved design (internal/research/expert_offload_design_frontend.md
// §7): port 0 = pick-a-free-port is correct and nothing good comes of a UI
// breaking it; a per-model path-to-executable picker in a web form is a
// foot-cannon; host/startup_timeout_s are plumbing. All gguf-only names, so a
// flat set is safe today -- revisit if another provider grows one of these.
const HIDDEN_FIELDS = new Set(['host', 'port', 'server_binary', 'startup_timeout_s']);

// The collapsed-row summary chip: which of this entry's stored keys are worth
// announcing on the list ("why is this one configured differently"). Only
// process-shaping scalars -- descriptive fields are written by every import
// (modalities, supports_thinking) and would put a chip on every row, and
// paths/arrays are too long to be a chip.
export function configSummary(config, fields) {
  if (!config || !fields?.length) return null;
  const parts = [];
  for (const f of fields) {
    if (f.effect === 'descriptive' || HIDDEN_FIELDS.has(f.name)) continue;
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
      return { value: raw.split(',').map((s) => s.trim()).filter(Boolean) };
    default:
      return { value: raw };
  }
}

// The control's string form of a stored value (inverse of parseControlValue).
function toControlValue(field, value) {
  if (value == null) return '';
  if (field.type === 'array') return Array.isArray(value) ? value.join(', ') : String(value);
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
  if (field.type === 'array') bits.push('comma-separated');
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

// createModelConfigEditor({ model, fields, draft, onError, onSaved, onReload, onReset })
//
// - model:  the AdminModelResponse row (id, provider, loaded, config, ...)
// - fields: the option schema's field list for model.provider (HIDDEN_FIELDS
//           are filtered here, so callers pass the schema verbatim)
// - draft:  caller-owned {fieldName: rawControlValue} of unsaved edits; the
//           editor mutates it so a page re-render can rebuild without loss
// - onError(msg): route a failure to the page's status area (page invariant:
//           errors have ONE home)
// - onSaved(updatedModel): the PATCH landed; caller refreshes its list row
// - onReload(): caller runs its unload + warm-load cycle (button appears only
//           on a loaded model after a reload-required save, armed-confirmed)
// - onReset(): drafts were dropped; caller rebuilds the panel from saved state
export function createModelConfigEditor({ model, fields: allFields, draft, onError, onSaved, onReload, onReset }) {
  const fields = allFields.filter((f) => !HIDDEN_FIELDS.has(f.name));
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

  const noteEl = createEl('div', { class: 'cfg-note muted small', role: 'status' });
  const saveBtn = createEl('button', { class: 'btn btn--sm' }, ['Save']);
  const resetBtn = createEl('button', { class: 'btn btn--sm' }, ['Reset']);
  const reloadBtn = createEl('button', { class: 'btn btn--sm', hidden: true }, ['Reload now']);

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

  const onEdit = (name, rawValue) => {
    draft[name] = rawValue;
    syncSaveState();
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
      for (const [key, value] of Object.entries(config)) {
        if (value === null) delete saved[key];
        else saved[key] = value;
      }
      for (const field of dirty) delete draft[field.name];
      // The response's model is the post-save truth (defaults resolved,
      // values normalized); hand it up so the row + next rebuild use it.
      onSaved(result.model);
      const parts = ['Saved.'];
      if (reloadNeeded.length) {
        parts.push(model.loaded
          ? `Takes effect on reload: ${reloadNeeded.join(', ')}.`
          : `Applies on next load: ${reloadNeeded.join(', ')}.`);
      }
      if (result.warning) parts.push(result.warning);
      setNote(parts.join(' '));
      reloadBtn.hidden = !(model.loaded && reloadNeeded.length);
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

  // --- layout: live / reload / advanced / fixed ---
  const advanced = fields.filter((f) => f.ui === 'advanced');
  const live = fields.filter((f) => LIVE_EFFECTS.has(f.effect) && f.ui !== 'advanced');
  const reload = fields.filter((f) => f.effect === 'requires_reload' && f.ui !== 'advanced');
  const fixed = fields.filter((f) => f.effect === 'load_time_only');

  const rows = (list) => list.map((f) => fieldRow(f, currentRaw(f), idPrefix, onEdit));

  const children = [];
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
    children.push(createEl('details', { class: 'cfg-section' }, [
      createEl('summary', { class: 'cfg-section__title' }, ['Advanced (requires reload)']),
      ...rows(advanced),
    ]));
  }
  if (fixed.length) {
    children.push(sectionEl('Fixed for this process', null, rows(fixed)));
  }
  if (!children.length) {
    children.push(createEl('div', { class: 'muted small' },
      [`No editable options for provider "${model.provider}".`]));
  }

  children.push(createEl('div', { class: 'cfg-actions' }, [saveBtn, resetBtn, reloadBtn, noteEl]));

  syncSaveState();
  return { el: createEl('div', { class: 'model-config' }, children) };
}
