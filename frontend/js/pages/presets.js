// Presets page -- where a stored preset is edited, renamed, duplicated or
// deleted (plan_dark_mode.md Phase 3b, owner 2026-10-05).
//
// The settings drawer used to do all of this beside the conversation's own
// prompt and samplers, and that mix is what cost a 35k-character prompt on
// 2026-08-28: a write aimed at a preset nobody was looking at. Here the thing
// you edit IS the thing on screen -- the editor opens on the preset's own
// stored text and knobs, and Save writes exactly that card back. The drawer
// keeps Apply (preset -> document), Save as new (document -> a NEW preset)
// and one overwrite it cannot aim: the Save sheet's "Overwrite X", the document
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
import { registerPageHotkeys } from '../hotkeys.js';
import {
  buildZip,
  parsePresetFiles,
  planImport,
  serializePresetJson,
  serializePresetMarkdown,
  serializePresetsJson,
  slugifyPresetName,
  uniqueMarkdownFilenames,
} from '../preset-io.js';
import { runPresetAssist, toolCapableModels } from '../preset-assist.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.presets = [];
    s.cards = new Map();      // preset id -> its card element
    s.openEditors = new Set(); // ids with an editor or rename box open
    s.dirtyEditors = new Map(); // preset id -> () => boolean
    s.assistAbort = null;
    s.setUnloadGuard = createUnloadGuard(ctx);

    s.importInput = createEl('input', {
      type: 'file',
      multiple: true,
      accept: '.json,.md,.markdown,application/json,text/markdown,text/plain',
      hidden: true,
      'aria-label': 'Import preset files',
    });
    s.importInput.addEventListener('change', () => {
      const files = [...(s.importInput.files ?? [])];
      s.importInput.value = '';
      if (files.length) handleImportFiles(ctx, files);
    });

    s.assistBtn = createEl('button', {
      type: 'button',
      class: 'btn btn--sm btn--primary',
      title: 'Use a tool-capable model to draft edits across one or more presets and review diffs before applying',
    }, ['Assist with AI']);
    s.assistBtn.addEventListener('click', () => openAssistDialog(ctx));

    s.importBtn = createEl('button', {
      type: 'button',
      class: 'btn btn--sm',
      title: 'Import one or more presets from JSON or Markdown file(s)',
    }, ['Import']);
    s.importBtn.addEventListener('click', () => s.importInput.click());

    s.exportAllJsonBtn = createEl('button', {
      type: 'button',
      class: 'btn btn--sm',
      title: 'Export all presets into one JSON file',
      disabled: true,
    }, ['Export all JSON']);
    s.exportAllJsonBtn.addEventListener('click', () => exportAllJson(ctx));

    s.exportAllMdBtn = createEl('button', {
      type: 'button',
      class: 'btn btn--sm',
      title: 'Export all presets as individual Markdown files (one per preset)',
      disabled: true,
    }, ['Export all Markdown']);
    s.exportAllMdBtn.addEventListener('click', () => exportAllMarkdown(ctx));

    s.toolbarEl = createEl('div', { class: 'presets__toolbar' }, [
      s.assistBtn,
      s.importBtn,
      s.exportAllJsonBtn,
      s.exportAllMdBtn,
      s.importInput,
    ]);

    s.importDialog = buildModalDialog('preset-import-dialog', 'preset-import-title');
    s.assistDialog = buildModalDialog('preset-assist-dialog', 'preset-assist-title');
    s.assistDialog.addEventListener('close', () => {
      s.assistAbort?.abort();
      s.assistAbort = null;
    });
    s.statusEl = createEl('div', { class: 'presets__status', role: 'status' });
    s.listEl = createEl('div', { class: 'presets__list' }, [
      createEl('div', { class: 'empty-state' }, ['Loading presets…']),
    ]);
    ctx.el.append(createEl('div', { class: 'presets' }, [
      createEl('h1', {}, ['Presets']),
      createEl('p', { class: 'presets__lede muted' }, [
        'Edit, rename, duplicate, import, export and delete presets here. A conversation or notebook '
        + 'keeps a copy of the preset it was started from, so nothing done here changes one that is '
        + 'already running.',
      ]),
      s.toolbarEl,
      s.statusEl,
      s.listEl,
      s.importDialog,
      s.assistDialog,
    ]));

    // Close any open per-card Export menu when clicking elsewhere on the page.
    const closeExportMenus = (event) => {
      for (const menu of ctx.el.querySelectorAll('.preset-export[open]')) {
        if (!menu.contains(event.target)) menu.open = false;
      }
    };
    document.addEventListener('click', closeExportMenus);
    ctx.onTeardown(() => document.removeEventListener('click', closeExportMenus));

    const closeDialogsOnHash = () => {
      if (s.importDialog?.open) s.importDialog.close();
      if (s.assistDialog?.open) s.assistDialog.close();
    };
    window.addEventListener('hashchange', closeDialogsOnHash);
    ctx.onTeardown(() => {
      s.assistAbort?.abort();
      s.assistAbort = null;
      window.removeEventListener('hashchange', closeDialogsOnHash);
    });

    // Another tab or device may have written while this one was away. Cards
    // with an open editor are left alone: their typed text is the user's, and
    // Save's own re-read is what protects the store from it.
    ctx.onResume(() => { if (!s.openEditors.size) load(ctx); });
    registerPageHotkeys(ctx, {
      onEscape: () => {
        const openMenu = ctx.el.querySelector('.preset-export[open]');
        if (openMenu) {
          openMenu.open = false;
          return true;
        }
        if (!s.openEditors.size) return false;
        // Never discard an editor with unsaved changes on bare Escape (FB-34).
        let blockedDirty = false;
        for (const id of s.openEditors) {
          if (s.dirtyEditors.get(id)?.()) {
            blockedDirty = true;
            continue;
          }
          const card = s.cards.get(id);
          const cancelBtn = [...(card?.querySelectorAll('.preset-edit__buttons .btn, .preset-card__rename-row .btn') ?? [])]
            .find((b) => b.textContent === 'Cancel');
          if (cancelBtn) {
            cancelBtn.click();
            return true;
          }
        }
        if (blockedDirty) {
          status(ctx, 'Unsaved edits — click Cancel to discard or Save to keep them.');
          return true;
        }
        return false;
      },
    });
    await load(ctx);
  },
});

const status = (ctx, text, isError = false) => setStatus(ctx.state.statusEl, text, isError);

function updateToolbar(ctx) {
  const s = ctx.state;
  const empty = !s.presets.length;
  s.exportAllJsonBtn.disabled = empty;
  s.exportAllMdBtn.disabled = empty;
}

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
  s.openEditors.clear();
  s.dirtyEditors.clear();
  s.setUnloadGuard(false);
  updateToolbar(ctx);
  if (!s.presets.length) {
    s.listEl.replaceChildren(createEl('div', { class: 'empty-state' }, [
      'No presets yet. Open the settings drawer on a chat or a notebook and use Save, '
      + 'or use Import above to load JSON or Markdown preset files.',
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

function setEditing(ctx, id, on, isDirty = null) {
  const s = ctx.state;
  if (on) {
    s.openEditors.add(id);
    if (isDirty) s.dirtyEditors.set(id, isDirty);
    else s.dirtyEditors.delete(id);
  } else {
    s.openEditors.delete(id);
    s.dirtyEditors.delete(id);
  }
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

  const exportJsonItem = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['JSON (.json)']);
  const exportMdItem = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Markdown (.md)']);
  const exportMenu = createEl('details', { class: 'preset-export' }, [
    createEl('summary', { class: 'btn btn--sm', title: 'Export this preset to JSON or Markdown' }, ['Export']),
    createEl('div', { class: 'preset-export__menu' }, [exportJsonItem, exportMdItem]),
  ]);
  exportJsonItem.addEventListener('click', () => {
    exportMenu.open = false;
    exportSingle(ctx, preset, 'json');
  });
  exportMdItem.addEventListener('click', () => {
    exportMenu.open = false;
    exportSingle(ctx, preset, 'md');
  });

  const delBtn = armedConfirm(
    createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Delete']),
    () => remove(ctx, preset),
    'Delete?',
    null,
    () => preset.id,
  );
  const actions = createEl('div', { class: 'preset-card__actions' }, [
    editBtn, renameBtn, dupBtn, exportMenu, delBtn,
  ]);

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
  const touched = new Set();
  const rows = Object.entries(PARAM_META).map(([key, meta]) => {
    const id = `preset-${preset.id}-${key}`;
    const control = knobControl(key, meta, preset.params?.[key] ?? null, id);
    readers[key] = control.read;
    for (const ev of ['input', 'change']) control.el.addEventListener(ev, () => touched.add(key));
    return createEl('div', { class: 'preset-edit__row' }, [
      createEl('label', { for: id }, [meta.label]),
      control.el,
    ]);
  });
  // The bag the editor would store. It STARTS as what the preset stores and
  // changes only for a control the user actually touched, so an editor opened
  // and saved untouched writes the stored bag back by construction: nothing a
  // browser fills in, defaults or restores into a field can pin a setting
  // nobody chose. A PUT replaces `params` whole, so every stored key rides
  // along, including one this page has no control for.
  const paramsValue = () => {
    const out = {};
    for (const [k, v] of Object.entries(preset.params ?? {})) {
      if (v != null) out[k] = v;
    }
    for (const key of touched) {
      const v = readers[key]();
      if (v != null) out[key] = v; else delete out[key];
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
  const isDirty = () => promptValue() !== stored || touched.size > 0;
  setEditing(ctx, preset.id, true, isDirty);
  const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
  // Read the latest preset row from state so a "changed elsewhere" refusal
  // followed by Cancel + Edit again opens on the refreshed row (FB-84).
  cancelBtn.addEventListener('click', () => {
    const latest = ctx.state.presets.find((p) => p.id === preset.id) ?? preset;
    replaceCard(ctx, latest);
  });

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
      // Refresh our cached row so "Cancel and Edit again" opens on the new
      // updated_at instead of failing with the stale timestamp forever (FB-84).
      const idx = ctx.state.presets.findIndex((p) => p.id === preset.id);
      if (idx >= 0) ctx.state.presets[idx] = fresh;
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
    // Consumed: the page hotkey would otherwise click the first Cancel on the
    // page, which can be a different card's open editor.
    else if (e.key === 'Escape') { e.preventDefault(); replaceCard(ctx, preset); }
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
    updateToolbar(ctx);
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
  updateToolbar(ctx);
  s.cards.get(preset.id)?.remove();
  s.cards.delete(preset.id);
  // A document stamped with this preset keeps its own prompt and settings and
  // simply reads "No preset" from now on: its stamp resolves to nothing.
  status(ctx, `Deleted "${preset.name}". Anything started from it keeps its own copy.`);
  if (!s.presets.length) load(ctx);
}

// ---------------------------------------------------------------------------
// export (single & bulk, JSON & Markdown)
// ---------------------------------------------------------------------------

function triggerDownload(filename, content, mimeType) {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = createEl('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportSingle(ctx, preset, format) {
  const slug = slugifyPresetName(preset.name);
  if (format === 'md') {
    triggerDownload(`${slug}.md`, serializePresetMarkdown(preset), 'text/markdown;charset=utf-8');
    status(ctx, `Exported "${preset.name}" as ${slug}.md.`);
    return;
  }
  triggerDownload(`${slug}.json`, serializePresetJson(preset), 'application/json;charset=utf-8');
  status(ctx, `Exported "${preset.name}" as ${slug}.json.`);
}

function exportAllJson(ctx) {
  const presets = ctx.state.presets;
  if (!presets.length) return;
  triggerDownload('heylook-presets.json', serializePresetsJson(presets), 'application/json;charset=utf-8');
  status(ctx, `Exported ${presets.length} preset${presets.length === 1 ? '' : 's'} to heylook-presets.json.`);
}

async function exportAllMarkdown(ctx) {
  const presets = ctx.state.presets;
  if (!presets.length) return;
  if (presets.length === 1) {
    exportSingle(ctx, presets[0], 'md');
    return;
  }
  const filenames = uniqueMarkdownFilenames(presets);
  const files = presets.map((p, i) => ({
    name: filenames[i],
    text: serializePresetMarkdown(p),
  }));

  // Prefer writing individual .md files directly into a chosen folder when supported.
  if (typeof window.showDirectoryPicker === 'function') {
    try {
      const dir = await window.showDirectoryPicker({ mode: 'readwrite' });
      for (const file of files) {
        const handle = await dir.getFileHandle(file.name, { create: true });
        const writable = await handle.createWritable();
        await writable.write(file.text);
        await writable.close();
      }
      if (ctx.alive) {
        status(ctx, `Exported ${files.length} Markdown files to "${dir.name}".`);
      }
      return;
    } catch (err) {
      if (err?.name === 'AbortError') return;
      // Fall through to ZIP archive if folder write failed (e.g. read-only directory).
    }
  }

  const zipBytes = buildZip(files);
  triggerDownload('heylook-presets-markdown.zip', new Blob([zipBytes], { type: 'application/zip' }), 'application/zip');
  status(ctx, `Exported ${files.length} Markdown files in heylook-presets-markdown.zip.`);
}

// ---------------------------------------------------------------------------
// shared modal dialog + AI preset assistant + import review modal
// ---------------------------------------------------------------------------

function buildModalDialog(className, labelledBy) {
  const dialog = createEl('dialog', {
    class: className,
    closedby: 'any',
    'aria-labelledby': labelledBy,
  });
  // Fallback for browsers without native <dialog closedby="any"> support (e.g. Safari).
  if (typeof HTMLDialogElement !== 'undefined' && !('closedBy' in HTMLDialogElement.prototype)) {
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside = (
        rect.top <= event.clientY
        && event.clientY <= rect.top + rect.height
        && rect.left <= event.clientX
        && event.clientX <= rect.left + rect.width
      );
      if (!inside) dialog.close();
    });
  }
  return dialog;
}

async function openAssistDialog(ctx) {
  const s = ctx.state;
  const dialog = s.assistDialog;
  const close = () => {
    s.assistAbort?.abort();
    s.assistAbort = null;
    if (dialog.open) dialog.close();
  };

  let models = [];
  let freshPresets = s.presets;
  try {
    const [mRes, pRes] = await Promise.all([
      api.listModels({ signal: ctx.signal }),
      fetchPresets(ctx),
    ]);
    if (!ctx.alive) return;
    models = toolCapableModels(mRes?.data ?? []);
    freshPresets = pRes;
  } catch (err) {
    if (ctx.alive) status(ctx, `Could not open preset assistant: ${err.message}`, true);
    return;
  }

  const closeBtn = createEl('button', {
    type: 'button', class: 'btn btn--ghost btn--sm', 'aria-label': 'Close preset assistant',
  }, ['Close']);
  closeBtn.addEventListener('click', close);

  const head = createEl('header', { class: 'preset-import__head' }, [
    createEl('h2', { id: 'preset-assist-title' }, ['Assist with AI']),
    closeBtn,
  ]);

  const modelSelectId = 'preset-assist-model';
  const modelSelect = createEl('select', { id: modelSelectId, class: 'input' },
    models.length
      ? models.map((m) => createEl('option', { value: m.id }, [
        m.supportsTools ? m.id : `${m.id} (no tool support detected)`,
      ]))
      : [createEl('option', { value: '' }, ['No generative models available'])],
  );
  const modelHint = createEl('div', { class: 'muted small' });
  const syncModelHint = () => {
    const chosen = models.find((m) => m.id === modelSelect.value);
    if (!chosen) {
      modelHint.textContent = 'Load or configure a tool-capable model first.';
    } else if (!chosen.supportsTools) {
      modelHint.textContent = 'This model\'s chat template did not advertise tool calling; /v1/messages may refuse it.';
    } else {
      modelHint.textContent = 'Uses /v1/messages tool calling. Nothing is saved until you review and confirm the diff.';
    }
  };
  modelSelect.addEventListener('change', syncModelHint);
  syncModelHint();

  const selectedNames = new Set();
  const checkboxes = freshPresets.map((p) => {
    const cb = createEl('input', { type: 'checkbox', value: p.name });
    cb.addEventListener('change', () => {
      if (cb.checked) selectedNames.add(p.name);
      else selectedNames.delete(p.name);
      syncTargetCount();
    });
    return { cb, label: createEl('label', { class: 'preset-assist__target' }, [cb, createEl('span', {}, [p.name])]) };
  });

  const targetCountEl = createEl('span', { class: 'muted small' });
  const syncTargetCount = () => {
    if (!freshPresets.length) {
      targetCountEl.textContent = 'No saved presets yet (the assistant can still propose new ones).';
    } else if (!selectedNames.size) {
      targetCountEl.textContent = `All ${freshPresets.length} presets visible to the model (check specific ones to narrow scope).`;
    } else {
      targetCountEl.textContent = `${selectedNames.size} of ${freshPresets.length} selected.`;
    }
  };
  syncTargetCount();

  const selectAllBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Select all']);
  selectAllBtn.addEventListener('click', () => {
    for (const { cb } of checkboxes) { cb.checked = true; selectedNames.add(cb.value); }
    syncTargetCount();
  });
  const clearSelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Clear']);
  clearSelBtn.addEventListener('click', () => {
    for (const { cb } of checkboxes) cb.checked = false;
    selectedNames.clear();
    syncTargetCount();
  });

  const instructionId = 'preset-assist-instruction';
  const instructionEl = createEl('textarea', {
    id: instructionId,
    class: 'input preset-assist__instruction',
    rows: 5,
    placeholder: 'Describe the wording or settings change — e.g. "Weave this paragraph into each selected preset\'s system prompt naturally, matching its formatting and casing:\n\n..."',
  });
  instructionEl.addEventListener('input', () => autoGrow(instructionEl, 320));

  const progressEl = createEl('div', { class: 'preset-assist__progress small muted', role: 'status', 'aria-live': 'polite' });

  const runBtn = createEl('button', {
    type: 'button',
    class: 'btn btn--sm btn--primary',
    disabled: !models.length,
  }, ['Draft edits']);
  const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
  cancelBtn.addEventListener('click', close);

  runBtn.addEventListener('click', async () => {
    const instruction = instructionEl.value.trim();
    if (!instruction) {
      progressEl.textContent = 'Enter an instruction first.';
      instructionEl.focus();
      return;
    }
    const model = modelSelect.value;
    if (!model) {
      progressEl.textContent = 'Choose a model first.';
      return;
    }

    s.assistAbort?.abort();
    const ac = ctx.linkedController();
    s.assistAbort = ac;
    runBtn.disabled = true;
    modelSelect.disabled = true;
    instructionEl.disabled = true;
    runBtn.textContent = 'Working…';
    progressEl.textContent = 'Contacting model…';

    try {
      const result = await runPresetAssist({
        createMessage: (body, opts) => api.createMessage(body, opts),
        model,
        instruction,
        presets: freshPresets,
        targetNames: [...selectedNames],
        signal: ac.signal,
        onProgress: (ev) => {
          if (!ctx.alive) return;
          if (ev.phase === 'thinking') {
            progressEl.textContent = `Turn ${ev.turn}: model thinking${ev.stagedCount ? ` (${ev.stagedCount} staged)` : ''}…`;
          } else if (ev.phase === 'tool') {
            progressEl.textContent = `Turn ${ev.turn}: ${ev.summary} (${ev.stagedCount} staged)…`;
          }
        },
      });
      if (!ctx.alive) return;
      s.assistAbort = null;
      if (!result.proposals.length) {
        progressEl.textContent = result.assistantText
          ? `Model made no preset edits: ${result.assistantText}`
          : 'Model finished without proposing any preset changes.';
        runBtn.disabled = false;
        modelSelect.disabled = false;
        instructionEl.disabled = false;
        runBtn.textContent = 'Draft edits';
        return;
      }
      if (dialog.open) dialog.close();
      const latestPresets = await fetchPresets(ctx).catch(() => freshPresets);
      if (!ctx.alive) return;
      const plan = planImport(result.proposals, latestPresets);
      openImportDialog(ctx, plan, { source: 'assist', assistantText: result.assistantText });
    } catch (err) {
      if (!ctx.alive) return;
      s.assistAbort = null;
      progressEl.textContent = err?.name === 'AbortError'
        ? 'Cancelled.'
        : `Assistant failed: ${err.message}`;
      runBtn.disabled = false;
      modelSelect.disabled = false;
      instructionEl.disabled = false;
      runBtn.textContent = 'Draft edits';
    }
  });

  const bodyChildren = [
    createEl('div', { class: 'preset-assist__field' }, [
      createEl('label', { for: modelSelectId }, ['Model']),
      modelSelect,
      modelHint,
    ]),
  ];
  if (checkboxes.length) {
    bodyChildren.push(createEl('div', { class: 'preset-assist__field' }, [
      createEl('div', { class: 'preset-assist__target-head' }, [
        createEl('span', { class: 'preset-diff__label small' }, ['Target presets (optional)']),
        createEl('div', { class: 'preset-assist__target-btns' }, [selectAllBtn, clearSelBtn]),
      ]),
      createEl('div', { class: 'preset-assist__targets' }, checkboxes.map((c) => c.label)),
      targetCountEl,
    ]));
  }
  bodyChildren.push(
    createEl('div', { class: 'preset-assist__field' }, [
      createEl('label', { for: instructionId }, ['Instruction']),
      instructionEl,
    ]),
    progressEl,
  );

  const box = createEl('div', { class: 'preset-import__box' }, [
    head,
    createEl('div', { class: 'preset-import__body' }, bodyChildren),
    createEl('footer', { class: 'preset-import__foot' }, [runBtn, cancelBtn]),
  ]);
  dialog.replaceChildren(box);
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
  queueMicrotask(() => instructionEl.focus());
}

function formatParamVal(val) {
  return typeof val === 'string' ? val : JSON.stringify(val);
}

function buildDiffBlock(item) {
  const children = [
    createEl('div', { class: 'preset-diff__head' }, [
      createEl('strong', {}, [item.incoming.name]),
      createEl('span', { class: 'preset-card__tag' }, [item.changed ? 'changed' : 'unchanged']),
    ]),
  ];

  if (!item.changed) {
    children.push(createEl('div', { class: 'muted small' }, ['No differences from the stored preset.']));
    return createEl('section', { class: 'preset-diff' }, children);
  }

  if (item.paramDiffs.length) {
    const rows = item.paramDiffs.map((d) => {
      if (d.kind === 'added') {
        return createEl('div', { class: 'preset-diff__line preset-diff__line--add' }, [
          `+ ${d.key}: ${formatParamVal(d.after)}`,
        ]);
      }
      if (d.kind === 'removed') {
        return createEl('div', { class: 'preset-diff__line preset-diff__line--del' }, [
          `- ${d.key}: ${formatParamVal(d.before)}`,
        ]);
      }
      return createEl('div', { class: 'preset-diff__pair' }, [
        createEl('div', { class: 'preset-diff__line preset-diff__line--del' }, [
          `- ${d.key}: ${formatParamVal(d.before)}`,
        ]),
        createEl('div', { class: 'preset-diff__line preset-diff__line--add' }, [
          `+ ${d.key}: ${formatParamVal(d.after)}`,
        ]),
      ]);
    });
    children.push(
      createEl('div', { class: 'preset-diff__label small muted' }, ['Settings']),
      createEl('div', { class: 'preset-diff__code' }, rows),
    );
  }

  if (item.promptChanged) {
    const label = !item.existing.system_prompt
      ? 'System prompt (added)'
      : !item.incoming.system_prompt
        ? 'System prompt (removed — becomes settings only)'
        : 'System prompt';
    const lines = item.promptDiff.map((line) => {
      const prefix = line.type === 'add' ? '+ ' : line.type === 'del' ? '- ' : line.type === 'hunk' ? '' : '  ';
      const cls = line.type === 'add'
        ? 'preset-diff__line preset-diff__line--add'
        : line.type === 'del'
          ? 'preset-diff__line preset-diff__line--del'
          : line.type === 'hunk'
            ? 'preset-diff__line preset-diff__line--hunk'
            : 'preset-diff__line';
      return createEl('div', { class: cls }, [`${prefix}${line.text}`]);
    });
    children.push(
      createEl('div', { class: 'preset-diff__label small muted' }, [label]),
      createEl('div', { class: 'preset-diff__code' }, lines),
    );
  }

  return createEl('section', { class: 'preset-diff' }, children);
}

async function handleImportFiles(ctx, files) {
  let incoming;
  try {
    const entries = await Promise.all(
      files.map(async (f) => ({ name: f.name, text: await f.text() })),
    );
    incoming = parsePresetFiles(entries);
  } catch (err) {
    if (ctx.alive) status(ctx, `Import failed: ${err.message}`, true);
    return;
  }

  let fresh;
  try {
    fresh = await fetchPresets(ctx);
  } catch (err) {
    if (ctx.alive) status(ctx, `Could not check existing presets: ${err.message}`, true);
    return;
  }
  if (!ctx.alive) return;

  const plan = planImport(incoming, fresh);
  openImportDialog(ctx, plan);
}

function openImportDialog(ctx, plan, { source = 'import', assistantText = '' } = {}) {
  const dialog = ctx.state.importDialog;
  const close = () => { if (dialog.open) dialog.close(); };
  const isAssist = source === 'assist';

  const single = plan.items.length === 1;
  const titleText = isAssist
    ? (single ? 'Review proposed preset edit' : `Review ${plan.items.length} proposed preset edits`)
    : (single
      ? (plan.items[0].existing ? 'Preset already exists' : 'Import preset')
      : `Import ${plan.items.length} presets`);

  const closeBtn = createEl('button', {
    type: 'button', class: 'btn btn--ghost btn--sm', 'aria-label': 'Close import dialog',
  }, ['Close']);
  closeBtn.addEventListener('click', close);

  const head = createEl('header', { class: 'preset-import__head' }, [
    createEl('h2', { id: 'preset-import-title' }, [titleText]),
    closeBtn,
  ]);

  const bodyChildren = [];
  if (isAssist && assistantText) {
    bodyChildren.push(createEl('p', { class: 'preset-import__note muted small' }, [assistantText]));
  }
  const footerBtns = [];
  let diffsWrap = null;

  const removesAnyPrompt = plan.existingItems.some(
    (it) => it.changed && Boolean(it.existing?.system_prompt) && !it.incoming.system_prompt,
  );

  const runApply = async (includeExisting, triggerBtn) => {
    for (const b of footerBtns) b.disabled = true;
    const prevLabel = triggerBtn.textContent;
    triggerBtn.textContent = isAssist ? 'Applying…' : 'Importing…';
    const result = await applyImportPlan(ctx, plan, { includeExisting, label: isAssist ? 'Apply' : 'Import' });
    if (!ctx.alive) return;
    close();
    reconcileAppliedPresets(ctx, result.createdPresets, result.updatedPresets);
    if (result.errors.length) {
      status(ctx, `${result.summary} Failed: ${result.errors.join('; ')}`, true);
    } else {
      status(ctx, result.summary);
    }
    triggerBtn.textContent = prevLabel;
  };

  if (single) {
    const item = plan.items[0];
    if (!item.existing) {
      bodyChildren.push(
        createEl('p', { class: 'preset-import__note' }, [
          `New preset "${item.incoming.name}".`,
        ]),
        createEl('div', { class: 'preset-card__knobs muted small' }, [knobSummary(item.incoming.params)]),
        item.incoming.system_prompt
          ? createEl('div', { class: 'preset-card__prompt' }, [item.incoming.system_prompt])
          : createEl('div', { class: 'preset-card__prompt preset-card__prompt--none' }, [
            'Carries no system prompt (settings only).',
          ]),
      );
      const importBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--primary' }, [
        isAssist ? 'Create preset' : 'Import preset',
      ]);
      importBtn.addEventListener('click', () => runApply(true, importBtn));
      const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
      cancelBtn.addEventListener('click', close);
      footerBtns.push(importBtn, cancelBtn);
    } else {
      bodyChildren.push(
        createEl('p', { class: 'preset-import__note' }, [
          item.changed
            ? `"${item.incoming.name}" already exists. Review the before/after diff below before overwriting.`
            : `"${item.incoming.name}" already exists and has no differences.`,
        ]),
        buildDiffBlock(item),
      );
      if (item.changed) {
        const overwriteBtn = armedConfirm(
          createEl('button', { type: 'button', class: 'btn btn--sm btn--primary' }, [
            isAssist ? 'Apply change' : 'Overwrite preset',
          ]),
          () => runApply(true, overwriteBtn),
          'Remove prompt?',
          // Arm before overwriting a stored prompt with null (FB-86).
          () => removesAnyPrompt,
          () => item.existing.id,
        );
        const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
        cancelBtn.addEventListener('click', close);
        footerBtns.push(overwriteBtn, cancelBtn);
      } else {
        const okBtn = createEl('button', { type: 'button', class: 'btn btn--sm' }, ['Close']);
        okBtn.addEventListener('click', close);
        footerBtns.push(okBtn);
      }
    }
  } else {
    // Bulk import / assist: list New vs Already exist, and let the user
    // either view combined diffs first OR import and apply directly.
    if (plan.newItems.length) {
      bodyChildren.push(
        createEl('div', { class: 'preset-import__group' }, [
          createEl('div', { class: 'preset-import__group-title' }, [`New (${plan.newItems.length})`]),
          createEl('ul', { class: 'preset-import__list' }, plan.newItems.map((it) =>
            createEl('li', {}, [
              createEl('span', {}, [it.incoming.name]),
              createEl('span', { class: 'preset-card__tag preset-import__badge--new' }, ['new']),
            ]))),
        ]),
      );
    }
    if (plan.existingItems.length) {
      bodyChildren.push(
        createEl('div', { class: 'preset-import__group' }, [
          createEl('div', { class: 'preset-import__group-title' }, [`Already exist (${plan.existingItems.length})`]),
          createEl('ul', { class: 'preset-import__list' }, plan.existingItems.map((it) =>
            createEl('li', {}, [
              createEl('span', {}, [it.incoming.name]),
              createEl('span', { class: 'preset-card__tag' }, [it.changed ? 'exists · changed' : 'exists · unchanged']),
            ]))),
        ]),
      );
    }

    if (plan.existingItems.length) {
      bodyChildren.push(
        createEl('p', { class: 'preset-import__note muted small' }, [
          'Inspect all existing preset diffs in a combined view below, or apply now.',
        ]),
      );
      diffsWrap = createEl('div', { class: 'preset-import__diffs', hidden: !isAssist },
        plan.existingItems.map((it) => buildDiffBlock(it)));
      bodyChildren.push(diffsWrap);

      const toggleDiffsBtn = createEl('button', {
        type: 'button', class: 'btn btn--sm', 'aria-expanded': String(isAssist),
      }, [isAssist ? 'Hide combined diffs' : 'View combined diffs']);
      toggleDiffsBtn.addEventListener('click', () => {
        const show = diffsWrap.hidden;
        diffsWrap.hidden = !show;
        toggleDiffsBtn.setAttribute('aria-expanded', String(show));
        toggleDiffsBtn.textContent = show ? 'Hide combined diffs' : 'View combined diffs';
      });

      const applyAllBtn = armedConfirm(
        createEl('button', { type: 'button', class: 'btn btn--sm btn--primary' }, [
          isAssist ? 'Apply all changes' : 'Import and apply',
        ]),
        () => runApply(true, applyAllBtn),
        'Remove stored prompt(s)?',
        () => removesAnyPrompt,
        () => plan.existingItems.map((it) => it.existing.id).join(','),
      );

      footerBtns.push(toggleDiffsBtn, applyAllBtn);
      if (plan.newItems.length) {
        const newOnlyBtn = createEl('button', { type: 'button', class: 'btn btn--sm' }, ['Import new only']);
        newOnlyBtn.addEventListener('click', () => runApply(false, newOnlyBtn));
        footerBtns.push(newOnlyBtn);
      }
      const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
      cancelBtn.addEventListener('click', close);
      footerBtns.push(cancelBtn);
    } else {
      const applyBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--primary' }, [
        `${isAssist ? 'Create' : 'Import'} ${plan.newItems.length} presets`,
      ]);
      applyBtn.addEventListener('click', () => runApply(true, applyBtn));
      const cancelBtn = createEl('button', { type: 'button', class: 'btn btn--sm btn--ghost' }, ['Cancel']);
      cancelBtn.addEventListener('click', close);
      footerBtns.push(applyBtn, cancelBtn);
    }
  }

  const box = createEl('div', { class: 'preset-import__box' }, [
    head,
    createEl('div', { class: 'preset-import__body' }, bodyChildren),
    createEl('footer', { class: 'preset-import__foot' }, footerBtns),
  ]);
  dialog.replaceChildren(box);
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
}

// Update only the cards affected by an import/assist so open editors on
// unrelated cards keep their typed drafts (FB-85).
function reconcileAppliedPresets(ctx, createdPresets, updatedPresets) {
  const s = ctx.state;
  if (!createdPresets.length && !updatedPresets.length) return;
  if (s.listEl.querySelector('.empty-state')) s.listEl.replaceChildren();
  for (const saved of updatedPresets) {
    if (s.cards.has(saved.id)) replaceCard(ctx, saved);
    else {
      s.presets.push(saved);
      s.listEl.append(buildCard(ctx, saved));
    }
  }
  for (const created of createdPresets) {
    s.presets.push(created);
    s.listEl.append(buildCard(ctx, created));
  }
  updateToolbar(ctx);
}

async function applyImportPlan(ctx, plan, { includeExisting, label = 'Import' }) {
  let created = 0;
  let updated = 0;
  let unchanged = 0;
  let skipped = 0;
  const errors = [];
  const createdPresets = [];
  const updatedPresets = [];

  // Re-read before overwriting existing rows so a preset edited elsewhere
  // while the dialog was open is never clobbered silently (FB-86).
  let freshById = null;
  if (includeExisting && plan.changedExistingItems.length) {
    try {
      const fresh = await fetchPresets(ctx);
      freshById = new Map(fresh.map((p) => [p.id, p]));
    } catch {
      freshById = null;
    }
  }

  for (const item of plan.items) {
    if (!ctx.alive) break;
    try {
      if (!item.existing) {
        const createdRow = await api.createPreset({
          name: item.incoming.name,
          system_prompt: item.incoming.system_prompt,
          params: item.incoming.params,
        });
        createdPresets.push(createdRow);
        created += 1;
      } else if (!includeExisting) {
        skipped += 1;
      } else if (!item.changed) {
        unchanged += 1;
      } else {
        if (freshById) {
          const current = freshById.get(item.existing.id);
          if (!current) {
            errors.push(`${item.incoming.name}: deleted elsewhere while reviewing`);
            continue;
          }
          if (current.updated_at !== item.existing.updated_at) {
            errors.push(`${item.incoming.name}: changed elsewhere while reviewing`);
            continue;
          }
        }
        const savedRow = await api.updatePreset(item.existing.id, {
          system_prompt: item.incoming.system_prompt,
          params: item.incoming.params,
        });
        updatedPresets.push(savedRow);
        updated += 1;
      }
    } catch (err) {
      errors.push(`${item.incoming.name}: ${err.message}`);
    }
  }

  const parts = [];
  if (created) parts.push(`${created} created`);
  if (updated) parts.push(`${updated} updated`);
  if (unchanged) parts.push(`${unchanged} unchanged`);
  if (skipped) parts.push(`${skipped} skipped`);
  const summary = parts.length ? `${label} complete: ${parts.join(', ')}.` : 'Nothing changed.';
  return { summary, errors, createdPresets, updatedPresets };
}
