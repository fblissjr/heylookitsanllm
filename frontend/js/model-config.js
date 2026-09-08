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

// The chat template in force, and an editor for overriding it.
//
// NOT a schema-driven field: this is a FILE BODY, not a models.toml value, so
// it sits beside the fit meter rather than in the generated form. Saving
// writes one file next to the weights and touches no config at all.
//
// Lazy: the body is fetched when the section is first opened, never on the
// models list paint. A template is kilobytes and there is one per model, so
// eager loading would fetch the lot to show a collapsed heading.
//
// The server is the authority on all of it -- which rung won, whether an edit
// would be inert, whether a reload is owed. This panel renders those answers
// and re-derives none of them; the ladder lives in two providers and a
// client-side second opinion would disagree with them the first time either
// one changed.
// `draft` is the caller's per-model scratch object, the same one the schema
// fields use. Reserved keys (not schema field names, so save()'s dirtyFields
// never sees them) carry the panel's unsaved state across a models-list
// rebuild -- this file's own header states that is what `draft` is for, and
// keeping the textarea in closure state meant clicking Load after typing a
// 300-line template silently discarded it and re-collapsed the section.
//
// It is deliberately PAGE-SCOPED, and the scope is wider than a reload:
// `s.configDrafts` is built fresh on every mount, so an in-app hash
// navigation discards an unsaved body exactly as a reload does. Only the
// RELOAD half is warned (v2.0.35 arms the unload guard while a draft holds
// text); `beforeunload` cannot see a hashchange at all, so the nav half is
// still silent -- as is iOS Safari in EVERY case, which does not fire
// `beforeunload` reliably. That asymmetry is the reason to read this before
// "improving" either half.
// The tempting precedent is chat.js parking its system prompt in
// localStorage, and it does NOT transfer. That draft is typed before a
// conversation exists, so the parking is a bridge to adoption, not a
// durability feature; a template body has a home from the first keystroke,
// one enabled button away. Storing it would also go STALE: `load()` refetches
// on every new panel, and `draft[TMPL_DRAFT] ?? serverText` lets a stored body
// win over whatever came back -- the override's own text where one exists, the
// model's otherwise -- including a template another session wrote. An in-memory draft can only ever be as old as
// the page, which is the property you want here.
const TMPL_DRAFT = '__chat_template_body';
const TMPL_OPEN = '__chat_template_open';

// "Does this draft hold unsaved template text", for a page that owns the
// draft but not the panel. Presence is the whole test because syncDirty
// DELETES the key the moment the body matches the server's -- but a blank
// body is not pending work: Save refuses it, so a warning about losing it
// would fire where no button could have saved anything -- the same CONTENT
// test Save's enabled state makes. It is deliberately asked of drafts whose
// panel is CLOSED or rebuilt, where there is no Save button on screen at all,
// and it ignores `busy`, so an in-flight save stays armed until it lands.
export const hasUnsavedTemplate = (draft) =>
  Boolean(draft && String(draft[TMPL_DRAFT] ?? '').trim());

// A textarea's value getter normalizes CRLF to LF, so comparing it against a
// raw server string makes a Windows-authored template read as edited the
// instant it is painted -- Save enabled, nothing typed, and a PUT that
// differs from disk only in line endings.
const eol = (t) => String(t ?? '').replace(/\r\n?/g, '\n');

function buildChatTemplatePanel({ model, draft, onDraftChange }) {
  const statusEl = createEl('div', { class: 'cfg-tmpl__status', role: 'status' });
  const originEl = createEl('div', { class: 'cfg-tmpl__origin muted small' });
  const areaId = `cfg-tmpl-${model.id}`.replace(/[^a-zA-Z0-9_-]/g, '-');
  // Disabled until the FIRST SUCCESSFUL render, which is the only thing that
  // knows whether this model's template is writable -- or what it currently
  // says. Before that the box is empty because nothing has loaded, not because
  // the model has no template, and typing into it enabled Save: one click then
  // PUT a fragment as the model's ENTIRE template (after a failed load), or
  // threw away the body that arrived mid-typing (during one, where
  // `draft[TMPL_DRAFT] ?? serverText` shows the fragment and hides what came
  // back). render() re-enables it, so both paths are covered by the one flag.
  // A REJECTED SAVE deliberately leaves it enabled: that path never re-renders,
  // and the repair belongs in the box you typed the template into.
  const area = createEl('textarea', {
    class: 'cfg-tmpl__body', id: areaId, spellcheck: 'false', rows: '16',
    disabled: true,
  });
  const label = createEl('label', { class: 'cfg-tmpl__label', for: areaId },
    ['Template body']);
  const saveBtn = createEl('button', { class: 'btn btn--sm', disabled: true }, ['Save template']);
  const revertBtn = createEl('button', { class: 'btn btn--sm', hidden: true }, ['Revert to model default']);
  const actions = createEl('div', { class: 'cfg-actions' }, [saveBtn, revertBtn]);
  const body = createEl('div', { class: 'cfg-tmpl__inner' },
    [originEl, label, area, actions, statusEl]);

  const el = createEl('details', { class: 'cfg-section cfg-tmpl' }, [
    createEl('summary', { class: 'cfg-section__title' }, ['Chat template']),
    body,
  ]);

  let loaded = false;
  let serverText = '';
  let busy = false;

  const say = (text, kind = '') => {
    statusEl.textContent = text || '';
    statusEl.className = `cfg-tmpl__status${kind ? ` cfg-tmpl__status--${kind}` : ''}`;
  };

  const syncDirty = () => {
    const body = eol(area.value);
    // `!loaded` is the half that survives a failed load. The box is re-enabled
    // there so the user can reach and clear their own draft, but Save must stay
    // off: `serverText` is still '' because render() never ran, so any typed
    // text differs from it and would arm a PUT of that fragment as the model's
    // ENTIRE template -- the bug the disable was added for, reachable again the
    // moment the box is editable.
    saveBtn.disabled = busy || !loaded || body === eol(serverText) || !body.trim();
    // Keep only genuinely unsaved text; a pristine panel must not resurrect
    // a stale body over a template someone changed elsewhere.
    if (body === eol(serverText)) delete draft[TMPL_DRAFT];
    else draft[TMPL_DRAFT] = area.value;
    // The draft outlives this panel, so whoever owns it hears every change --
    // including the ones that come from render() and commit(), not just typing.
    onDraftChange?.();
  };

  function render(view) {
    // EDIT the override's own body, not the winner's. They differ exactly
    // when an override exists but lost the ladder -- and painting the
    // winner's body there made the operator's own file unreadable from the
    // surface that wrote it: a rejected template showed as an empty box with
    // Save disabled and no way to repair it.
    serverText = (view.override_present ? view.override_template : view.template) || '';
    area.value = draft[TMPL_DRAFT] ?? serverText;
    area.disabled = !view.supported || !view.writable;
    revertBtn.hidden = !view.override_present;

    const bits = [];
    if (view.supported) bits.push(`In force: ${view.origin}`);
    if (view.override_present) bits.push('your override is on disk');
    originEl.textContent = bits.join(' · ');

    // Ordered worst-first: an edit that cannot land at all matters more than
    // one that has landed but needs a reload.
    if (!view.supported) {
      say(view.notes[0] || 'This provider does not use a chat template.');
    } else if (view.inert_reason) {
      say(view.inert_reason, 'warn');
    } else if (!view.writable) {
      say(view.notes[0] || 'This model folder is not writable.', 'warn');
    } else if (view.stale) {
      // stale is null for an unloaded model, which is NOT "up to date" --
      // only an explicit true means the running process differs from disk.
      say('Edited since this model loaded. Reload it to apply.', 'warn');
    } else {
      say('');
    }
    syncDirty();
  }

  async function load() {
    if (loaded) return;
    loaded = true;
    say('Loading…');
    try {
      render(await api.adminChatTemplate(model.id));
    } catch (e) {
      loaded = false; // let the next open retry
      say(`Could not read the template: ${e.message}`, 'error');
      // Show the draft and let it be edited. `render()` never ran, so without
      // this the box stays EMPTY and DISABLED while `draft[TMPL_DRAFT]` still
      // holds text: unreachable, uneditable, and still arming the leave-site
      // guard -- with no way for the user to see it or clear it. Save stays
      // disabled (`serverText` is still '' and syncDirty has no baseline to
      // compare against, so a blind overwrite is not on offer), which is the
      // half of the disable that was load-bearing. Re-enabling only the box
      // keeps the fix for typing over a template that never loaded, and drops
      // the part that trapped the user's own text behind a failed fetch.
      area.value = draft[TMPL_DRAFT] ?? '';
      area.disabled = false;
      syncDirty();
    }
  }

  async function commit(fn, working) {
    if (busy) return;
    busy = true;
    syncDirty();
    say(working);
    try {
      delete draft[TMPL_DRAFT];
      render(await fn());
    } catch (e) {
      // The server's refusal message names what would have broken -- it is
      // the entire value of validating before the write, so show it verbatim
      // rather than a generic failure.
      say(e.message, 'error');
    } finally {
      busy = false;
      syncDirty();
    }
  }

  el.addEventListener('toggle', () => {
    draft[TMPL_OPEN] = el.open;
    if (el.open) load();
  });
  // Restore the open section AFTER the listener exists, and fetch explicitly:
  // assigning `el.open` programmatically does not fire `toggle`, so setting it
  // earlier reopened the panel onto a blank box that never loaded.
  if (draft[TMPL_OPEN]) { el.open = true; load(); }
  area.addEventListener('input', syncDirty);
  saveBtn.addEventListener('click', () =>
    commit(() => api.adminSetChatTemplate(model.id, { template: area.value }),
           'Saving…'));
  // Armed: reverting discards an override that may be the only copy of work
  // typed here. (What it CANNOT lose is the model's own template -- the
  // override is a separate file and the vendor's was never written.)
  armedConfirm(revertBtn,
    () => commit(() => api.adminDelChatTemplate(model.id), 'Reverting…'),
    'Discard your override?');

  return { el };
}

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
      // force_refresh: the metrics snapshot is 30s-cached, and right after a
      // (re)load the cached entry predates it -- the one moment this line
      // exists for is exactly when the cache is wrong.
      const metrics = await api.systemMetrics(true);
      const mb = metrics?.models?.[model.id]?.memory_mb;
      // 0.0 is the collector's measurement-FAILED sentinel, not a reading --
      // rendering "0.0 GiB measured" would be the opposite of calibration.
      if (!mb) return;
      // Built by the same row() helper as the estimate rows above, so a
      // restyle can't silently fork this line's markup.
      observedEl.replaceChildren(
        ...row('Resident now', gib(mb / 1024), 'measured after load').children);
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
      // A LOADED model is itself holding the memory this line measures
      // against, so the counterfactual "could this be loaded" is depressed by
      // exactly the model asking the question -- and the panel said "Won't
      // fit" about a model that was running at the time. Reclaimable is
      // total minus anonymous minus wired, and Metal wires a resident model's
      // working set, so its own bytes are subtracted from the figure. Say
      // what is true instead; the other ceilings are unaffected and still
      // render. (The Load gate below already ignores the verdict when loaded,
      // so nothing downstream changes.)
      pieces.push(model.loaded
        ? `Loaded now, so this model's own memory is excluded from the `
          + `~${gib(ram.have_gb)} reclaimable figure — unload it to see what a `
          + `fresh load would find.`
        : `Won't fit: needs ${gib(ram.need_gb)} (weights + headroom), `
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
    // Server-derived (ram_fit.THIN_HEADROOM_GB): the model fits, but KV +
    // compute at full context have little room. Two consequences the reader
    // should hear before Load: heylook spawns with llama-server's own
    // micro-batch (slower prefill), and a decode-time Metal OOM is possible.
    if (r.headroom_thin && r.kv_headroom_gb != null) {
      pieces.push(`Thin headroom: ${gib(r.kv_headroom_gb)} left for KV + compute. `
        + `Spawns with llama-server's default micro-batch (512) instead of 2048; `
        + `a Metal out-of-memory at full context is possible.`);
    }

    verdictEl.className = `cfg-fit__verdict cfg-fit__verdict--${r.verdict}`;
    verdictEl.replaceChildren(
      pieces.length ? pieces.join(' ') : 'Fits.',
      // Server-gated actionability: non-null ONLY while iogpu.wired_limit_mb
      // is at its OS default AND either the working set is exceeded or the
      // headroom is thin. Present -> show verbatim.
      ...(r.sysctl_suggest_mb != null ? [createEl('div', { class: 'cfg-fit__sysctl' }, [
        'Raise the GPU wired limit: ',
        createEl('code', {}, [`sudo sysctl iogpu.wired_limit_mb=${r.sysctl_suggest_mb}`]),
        ' (resets at reboot; ',
        createEl('code', {}, ['scripts/gpu_wired_limit.sh install']),
        ' persists it). Restart the server afterwards so it sizes against the new ceiling.',
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
      // 422 = the server could not SIZE this model, and it says why. The
      // reason was hardcoded to "model_path does not exist" here, which was
      // the only 422 the route produced until v1.79.56 widened it to every
      // unsizeable report -- so an interrupted download that leaves a
      // config.json and no weights was told its path did not exist, when it
      // does. Render the server's reason; NEVER fall back to a client-side
      // guess about which one it was.
      verdictEl.className = 'cfg-fit__verdict';
      const why = err.detail?.error;
      verdictEl.textContent = (err.status === 422 && why)
        ? `fit unavailable — ${why}`
        : 'fit unavailable';
      rowsEl.replaceChildren();
      onGate?.(null);
    }
  }

  return { el, refresh, refreshObserved, scheduleRefresh: debounce(refresh, 300) };
}

// createModelConfigEditor({ model, fields, draft, initialNote, onError,
//                            onSaved, onReload, onReset, onFitGate, onDraftChange })
//
// - model:  the AdminModelResponse row (id, provider, loaded, config,
//           stale_reload_fields, ...). stale_reload_fields is SERVER truth
//           ("saved but the loaded process still runs the old value") and is
//           what re-arms the Reload offer on every rebuild -- panel-local
//           state cannot survive a remount, and a client-side copy of this
//           fact drifts (it did).
// - onDraftChange: fires whenever the template panel writes or clears its
//           reserved draft key -- from render() and commit() as well as from
//           typing, so it CAN fire after the caller's page is torn down (an
//           unaborted GET landing late, a PUT that rejects). A consumer that
//           acts on process-global state must tolerate that; createUnloadGuard
//           does it for you.
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
export function createModelConfigEditor({ model, fields: allFields, draft, initialNote, onError, onSaved, onReload, onReset, onFitGate, onDraftChange }) {
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
  const chatTemplate = buildChatTemplatePanel({ model, draft, onDraftChange });

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
  // Appended after the emptiness check on purpose: it is not an "editable
  // option" in the schema sense, and counting it would suppress the
  // no-options message for every provider that has none.
  children.push(chatTemplate.el);

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
