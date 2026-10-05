// Presets page -- where a stored preset is edited, renamed, duplicated or
// deleted (plan_dark_mode.md Phase 3b, owner 2026-10-05).
//
// The settings drawer used to do all of this beside the conversation's own
// prompt and samplers, and that mix is what cost a 35k-character prompt on
// 2026-08-28: a write aimed at a preset nobody was looking at. Here the thing
// you edit IS the thing on screen -- the editor opens on the preset's own
// stored text and knobs, and Save writes exactly that card back. The drawer
// keeps Apply (preset -> document), Save as new (document -> a NEW preset)
// and one overwrite it cannot aim: the Save sheet's "Update X", the document
// written back to the preset it is stamped with (preset-bar.js).
//
// Three rules this page keeps (sharp_edges.md "Presets and the system
// prompt"):
//   - a preset is a COPY, never a link: nothing here touches a conversation
//     or a notebook, and none of them is read. A document started from a
//     preset keeps what it was given
//   - an empty preset prompt makes no claim: such a preset is "settings
//     only", and blanking a stored prompt arms, because the preset stays in
//     the list but stops carrying anything ("my preset disappeared")
//   - a write never lands on a row that moved: Save re-reads the store first
//     and refuses when `updated_at` is not the one the editor opened on
//
// The knob editor is built from PARAM_META and bound to the preset's own
// bag. It deliberately does NOT reuse settings.js's panel: that panel is a
// view of the module-level sampler cache chat and notebook share, so editing
// a preset through it would leak the preset's values into the next document.
// A preset is model-agnostic, so every knob is offered and nothing is judged
// against a model here -- Apply removes what the target model cannot use and
// names it.

import { createPage } from '../page.js';
import { api } from '../api.js';
import { createEl, setStatus, armedConfirm, autoGrow, createUnloadGuard } from '../utils.js';
import { PARAM_META } from '../settings.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.presets = [];
    s.cards = new Map();      // preset id -> its card element
    s.openEditors = new Set(); // ids with an editor or rename box open
    s.setUnloadGuard = createUnloadGuard(ctx);

    s.statusEl = createEl('div', { class: 'presets__status', role: 'status' });
    s.listEl = createEl('div', { class: 'presets__list' }, [
      createEl('div', { class: 'empty-state' }, ['Loading presets…']),
    ]);
    ctx.el.append(createEl('div', { class: 'presets' }, [
      createEl('h1', {}, ['Presets']),
      createEl('p', { class: 'presets__lede muted' }, [
        'Edit, rename, duplicate and delete presets here. A conversation or notebook keeps a '
        + 'copy of the preset it was started from, so nothing done here changes one that is '
        + 'already running.',
      ]),
      s.statusEl,
      s.listEl,
    ]));

    // Another tab or device may have written while this one was away. Cards
    // with an open editor are left alone: their typed text is the user's, and
    // Save's own re-read is what protects the store from it.
    ctx.onResume(() => { if (!s.openEditors.size) load(ctx); });
    await load(ctx);
  },
});

const status = (ctx, text, isError = false) => setStatus(ctx.state.statusEl, text, isError);

async function fetchPresets(ctx) {
  const res = await api.listPresets({ signal: ctx.signal });
  return res.presets ?? [];
}

async function load(ctx) {
  const s = ctx.state;
  try {
    s.presets = await fetchPresets(ctx);
  } catch (err) {
    if (ctx.alive) status(ctx, `Could not load presets: ${err.message}`, true);
    return;
  }
  if (!ctx.alive) return;
  s.cards.clear();
  if (!s.presets.length) {
    s.listEl.replaceChildren(createEl('div', { class: 'empty-state' }, [
      'No presets yet. Open the settings drawer on a chat or a notebook and use Save '
      + 'to make one from what it is running.',
    ]));
    return;
  }
  s.listEl.replaceChildren(...s.presets.map((p) => buildCard(ctx, p)));
}

// ---------------------------------------------------------------------------
// what a preset holds, in words
// ---------------------------------------------------------------------------

const carriesPrompt = (p) => Boolean(p.system_prompt);

function knobText(key, value) {
  const meta = PARAM_META[key];
  const label = (meta?.label ?? key).toLowerCase();
  if (meta?.type === 'thinking') return `${label} ${value ? 'on' : 'off'}`;
  return `${label} ${value}`;
}

function knobSummary(params) {
  const known = Object.keys(PARAM_META).filter((k) => params?.[k] != null);
  const other = Object.keys(params ?? {}).filter((k) => !(k in PARAM_META) && params[k] != null);
  const parts = [...known, ...other].map((k) => knobText(k, params[k]));
  return parts.length ? parts.join(' · ') : 'No settings pinned: the model\'s own defaults apply.';
}

// ---------------------------------------------------------------------------
// cards
// ---------------------------------------------------------------------------

function replaceCard(ctx, preset) {
  const s = ctx.state;
  const idx = s.presets.findIndex((p) => p.id === preset.id);
  if (idx >= 0) s.presets[idx] = preset;
  // Read the old card BEFORE building: buildCard registers the new one under
  // the same id.
  const old = s.cards.get(preset.id);
  const next = buildCard(ctx, preset);
  old?.replaceWith(next);
  return next;
}

function setEditing(ctx, id, on) {
  const s = ctx.state;
  if (on) s.openEditors.add(id); else s.openEditors.delete(id);
  // Typed edits here have no home until Save, so a reload would lose them.
  s.setUnloadGuard(s.openEditors.size > 0);
}

function buildCard(ctx, preset) {
  const s = ctx.state;
  setEditing(ctx, preset.id, false);

  const nameEl = createEl('h2', { class: 'preset-card__name' }, [preset.name]);
  const tag = carriesPrompt(preset) ? null
    : createEl('span', { class: 'preset-card__tag' }, ['settings only']);

  const editBtn = createEl('button', {
    type: 'button', class: 'btn btn--sm', title: 'Edit this preset\'s prompt and settings',
  }, ['Edit']);
  const renameBtn = createEl('button', { type: 'button', class: 'btn btn--sm' }, ['Rename']);
  const dupBtn = createEl('button', {
    type: 'button', class: 'btn btn--sm', title: 'Make a copy under a new name',
  }, ['Duplicate']);
  const delBtn = armedConfirm(
    createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Delete']),
    () => remove(ctx, preset),
    'Delete?',
    null,
    () => preset.id,
  );
  const actions = createEl('div', { class: 'preset-card__actions' }, [editBtn, renameBtn, dupBtn, delBtn]);

  const head = createEl('header', { class: 'preset-card__head' }, [
    createEl('div', { class: 'preset-card__title' }, [nameEl, tag]),
    actions,
  ]);
  const body = createEl('div', { class: 'preset-card__body' }, buildReadout(preset));

  const card = createEl('article', {
    class: 'preset-card', dataset: { id: preset.id, name: preset.name },
    'aria-label': `Preset ${preset.name}`,
  }, [head, body]);
  s.cards.set(preset.id, card);

  editBtn.addEventListener('click', () => {
    delBtn.disarm();
    setEditing(ctx, preset.id, true);
    actions.hidden = true;
    body.replaceChildren(buildEditor(ctx, preset));
  });
  renameBtn.addEventListener('click', () => {
    delBtn.disarm();
    setEditing(ctx, preset.id, true);
    actions.hidden = true;
    head.querySelector('.preset-card__title').replaceChildren(buildRename(ctx, preset));
  });
  dupBtn.addEventListener('click', () => duplicate(ctx, preset));
  return card;
}

function buildReadout(preset) {
  return [
    createEl('div', { class: 'preset-card__knobs muted small' }, [knobSummary(preset.params)]),
    carriesPrompt(preset)
      ? createEl('div', { class: 'preset-card__prompt' }, [preset.system_prompt])
      : createEl('div', { class: 'preset-card__prompt preset-card__prompt--none' }, [
        'Carries no system prompt. Applying it changes the settings and leaves the prompt as it is.',
      ]),
  ];
}

// ---------------------------------------------------------------------------
// edit (prompt + knobs), the one overwrite
// ---------------------------------------------------------------------------

function knobControl(key, meta, value, id) {
  if (meta.type === 'thinking' || meta.type === 'checkbox') {
    const sel = createEl('select', { id, class: 'input' }, [
      createEl('option', { value: '' }, ['not set']),
      createEl('option', { value: 'true' }, ['on']),
      createEl('option', { value: 'false' }, ['off']),
    ]);
    sel.value = value === true ? 'true' : value === false ? 'false' : '';
    return { el: sel, read: () => (sel.value === '' ? null : sel.value === 'true') };
  }
  if (meta.type === 'select') {
    const sel = createEl('select', { id, class: 'input' }, [
      createEl('option', { value: '' }, ['not set']),
      ...meta.options.map((o) => createEl('option', { value: o }, [o])),
    ]);
    sel.value = value ?? '';
    return { el: sel, read: () => sel.value || null };
  }
  if (meta.type === 'number') {
    const input = createEl('input', {
      id, class: 'input', type: 'number', min: meta.min, max: meta.max, step: meta.step,
      placeholder: 'not set', value: value ?? '',
    });
    return { el: input, read: () => (input.value.trim() === '' ? null : Number(input.value)) };
  }
  // depth: the level is a word from one model's own template, so there is no
  // list to offer here. A level the target model lacks is removed on Apply.
  const input = createEl('input', {
    id, class: 'input', type: 'text', placeholder: 'not set', value: value ?? '', maxLength: 32,
  });
  return { el: input, read: () => input.value.trim() || null };
}

function buildEditor(ctx, preset) {
  const openedAt = preset.updated_at;
  const stored = preset.system_prompt || null;

  const promptId = `preset-prompt-${preset.id}`;
  const prompt = createEl('textarea', {
    id: promptId, class: 'input preset-edit__prompt', rows: 6, value: preset.system_prompt ?? '',
    placeholder: 'No system prompt: this preset would carry settings only.',
  });
  prompt.addEventListener('input', () => autoGrow(prompt, 480));
  queueMicrotask(() => autoGrow(prompt, 480));
  const promptValue = () => prompt.value.trim() || null;

  const readers = {};
  const rows = Object.entries(PARAM_META).map(([key, meta]) => {
    const id = `preset-${preset.id}-${key}`;
    const control = knobControl(key, meta, preset.params?.[key] ?? null, id);
    readers[key] = control.read;
    return createEl('div', { class: 'preset-edit__row' }, [
      createEl('label', { for: id }, [meta.label]),
      control.el,
    ]);
  });
  // The bag the editor would store: every key it has a control for, read
  // from the control; any other stored key carried through untouched, because
  // a PUT replaces `params` whole and a key left out is a key deleted.
  const paramsValue = () => {
    const out = {};
    for (const [k, v] of Object.entries(preset.params ?? {})) {
      if (!(k in PARAM_META) && v != null) out[k] = v;
    }
    for (const [k, read] of Object.entries(readers)) {
      const v = read();
      if (v != null) out[k] = v;
    }
    return out;
  };

  const saveBtn = armedConfirm(
    createEl('button', { type: 'button', class: 'btn btn--sm btn--primary' }, ['Save']),
    () => save(ctx, preset, { openedAt, system_prompt: promptValue(), params: paramsValue() }),
    'Remove prompt?',
    // The one write here that can cost typed work with nothing on screen to
    // show for it: emptying a prompt the preset stores.
    () => Boolean(stored && !promptValue()),
    () => JSON.stringify([preset.id, promptValue()]),
  );
  prompt.addEventListener('input', () => saveBtn.disarm());
  const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
  cancelBtn.addEventListener('click', () => replaceCard(ctx, preset));

  return createEl('div', { class: 'preset-edit' }, [
    createEl('label', { for: promptId }, ['System prompt']),
    prompt,
    createEl('div', { class: 'muted small' }, [
      'Empty means this preset carries no prompt: applying it leaves a prompt as it is.',
    ]),
    createEl('div', { class: 'preset-edit__knobs' }, rows),
    createEl('div', { class: 'muted small' }, [
      'Every setting is offered here. One a model cannot use is removed when the preset is '
      + 'applied to it, and named.',
    ]),
    createEl('div', { class: 'preset-edit__buttons' }, [saveBtn, cancelBtn]),
  ]);
}

async function save(ctx, preset, { openedAt, system_prompt, params }) {
  try {
    // Re-read FIRST: the write keeps no history, so it must not land on a row
    // that changed after this editor opened on it.
    const fresh = (await fetchPresets(ctx)).find((p) => p.id === preset.id);
    if (!ctx.alive) return;
    if (!fresh) {
      status(ctx, `"${preset.name}" no longer exists: it was deleted elsewhere. `
        + 'Copy anything you need out of the editor, then Cancel.', true);
      return;
    }
    if (fresh.updated_at !== openedAt) {
      status(ctx, `"${preset.name}" changed elsewhere since you opened it. Nothing was saved: `
        + 'copy anything you need out of the editor, then Cancel and Edit again.', true);
      return;
    }
    // No `name`: PUT patches only the fields it is given, and re-sending a
    // cached name would revert a rename made on another device.
    const saved = await api.updatePreset(preset.id, { system_prompt, params });
    if (!ctx.alive) return;
    replaceCard(ctx, saved);
    status(ctx, `Preset "${saved.name}" saved.`);
  } catch (err) {
    if (ctx.alive) status(ctx, `Preset save failed: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// rename, duplicate, delete
// ---------------------------------------------------------------------------

function buildRename(ctx, preset) {
  const input = createEl('input', {
    class: 'input preset-card__rename', type: 'text', value: preset.name,
    'aria-label': `New name for ${preset.name}`,
  });
  const commit = async () => {
    const name = input.value.trim();
    if (!name || name === preset.name) { replaceCard(ctx, preset); return; }
    try {
      const saved = await api.updatePreset(preset.id, { name });
      if (!ctx.alive) return;
      replaceCard(ctx, saved);
      status(ctx, `Renamed to "${saved.name}".`);
    } catch (err) {
      if (!ctx.alive) return;
      // The box stays, with what was typed: a refusal is not a reason to
      // retype the name.
      status(ctx, err.status === 409
        ? `A preset named "${name}" already exists. Pick another name.`
        : `Rename failed: ${err.message}`, true);
    }
  };
  const saveBtn = createEl('button', { type: 'button', class: 'btn btn--sm' }, ['Save name']);
  const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
  saveBtn.addEventListener('click', commit);
  cancelBtn.addEventListener('click', () => replaceCard(ctx, preset));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') commit();
    else if (e.key === 'Escape') replaceCard(ctx, preset);
  });
  queueMicrotask(() => { input.focus(); input.select(); });
  return createEl('div', { class: 'preset-card__rename-row' }, [input, saveBtn, cancelBtn]);
}

// "<name> copy", then "<name> copy 2", ... -- the first one not in use.
function copyName(name, taken) {
  const base = `${name} copy`;
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) {
    if (!taken.has(`${base} ${n}`)) return `${base} ${n}`;
  }
}

async function duplicate(ctx, preset) {
  const s = ctx.state;
  try {
    // Named against a fresh list; the server's 409 is the backstop.
    const fresh = await fetchPresets(ctx);
    if (!ctx.alive) return;
    const source = fresh.find((p) => p.id === preset.id) ?? preset;
    const created = await api.createPreset({
      name: copyName(source.name, new Set(fresh.map((p) => p.name))),
      system_prompt: source.system_prompt ?? null,
      params: { ...(source.params ?? {}) },
    });
    if (!ctx.alive) return;
    s.presets.push(created);
    const card = buildCard(ctx, created);
    (s.cards.get(preset.id) ?? s.listEl.lastElementChild).after(card);
    status(ctx, `Duplicated "${source.name}" as "${created.name}".`);
  } catch (err) {
    if (ctx.alive) status(ctx, `Duplicate failed: ${err.message}`, true);
  }
}

async function remove(ctx, preset) {
  const s = ctx.state;
  try {
    await api.deletePreset(preset.id);
  } catch (err) {
    if (ctx.alive) status(ctx, `Delete failed: ${err.message}`, true);
    return;
  }
  if (!ctx.alive) return;
  setEditing(ctx, preset.id, false);
  s.presets = s.presets.filter((p) => p.id !== preset.id);
  s.cards.get(preset.id)?.remove();
  s.cards.delete(preset.id);
  // A document stamped with this preset keeps its own prompt and settings and
  // simply reads "No preset" from now on: its stamp resolves to nothing.
  status(ctx, `Deleted "${preset.name}". Anything started from it keeps its own copy.`);
  if (!s.presets.length) load(ctx);
}
