// Models admin: list + load/warm/unload, folder + HF-cache scan and import,
// danger zone.
//
// Invariants:
// - One status/error area at the top. Failed actions write `.error-note`
//   text there; any subsequent successful action clears it. No console-only
//   failures, no native alert()/confirm(). Success NOTES therefore need their
//   own element (`.models__list-note`, `.models__danger-result`) -- writing
//   one into the status area gets it wiped by the next successful action.
// - Busy flags are always per-row (Set keyed by model/result id), never a
//   single page-wide flag -- unrelated rows must stay interactive.

import { createPage } from '../page.js';
import { createEl, armedConfirm, createUnloadGuard, formatTokens } from '../utils.js';
import { api } from '../api.js';
import * as drawer from '../settings-drawer.js';
import { createModelConfigEditor, configSummary, hasUnsavedTemplate } from '../model-config.js';
import { contextCeiling, engineSummary, renderEngine } from '../engine.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.models = [];
    s.loadingIds = new Set();   // model ids mid load/unload
    s.pendingLoadNote = null;   // warm-timing note, flushed after the list refetch
    s.optionsSchema = null;     // /v1/admin/model-options payload (fetched on first Configure)
    s.optionsPromise = null;    // in-flight fetch of the above
    s.configOpenId = null;      // model id with the config editor expanded (single panel)
    s.configDrafts = new Map(); // model id -> {field: rawValue} unsaved edits; survives re-renders
    s.engineOpenIds = new Set(); // model ids whose engine panel is open; survives re-renders
    // Unsaved work on this page is a chat-template body typed into a panel:
    // the one thing here that only a button press can commit, and the one
    // worth thousands of characters. Owned at PAGE level because the draft
    // outlives every panel that shows it -- collapse the panel, or save a
    // config field and watch renderModelList rebuild it, and the text is
    // still unsaved. Schema fields are deliberately NOT counted: their draft
    // keys are written even when the value matches saved, so presence does
    // not mean dirty there, and the loss is a number retyped in seconds.
    s.setUnloadGuard = createUnloadGuard(ctx);
    // model id -> reason string while the fit meter says FAIL (server
    // verdict; see onFitGate below). Only ever set for unloaded models.
    s.fitGates = new Map();
    s.configSaveNote = null;    // {id, text} one-shot: carries the save outcome across the post-save rebuild
    s.savingFolders = false;

    buildSkeleton(ctx);
    // No sampler/page settings here -- register so the drawer still offers the
    // global Display prefs and hides the sampler panel.
    ctx.onTeardown(drawer.registerSettings({ samplers: 'hidden' }));
    // The option schema also feeds the per-row summary chip, so fetch it
    // eagerly but NON-fatally -- the page must work without it (Configure
    // retries the fetch and reports failure itself).
    s.optionsPromise = api.adminModelOptions({ signal: ctx.signal });
    s.optionsPromise
      .then((schema) => {
        if (!ctx.alive) return;
        s.optionsSchema = schema;
        renderModelList(ctx);
      })
      .catch(() => { s.optionsPromise = null; });
    // Non-fatal, like the option schema: the watch-folder editor is a
    // convenience over models.toml, and the model list must render without it.
    loadWatchFolders(ctx);
    await fetchModels(ctx);
  },
});

// ---------------------------------------------------------------------------
// skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.statusEl = createEl('div', { class: 'models__status' });

  s.listEl = createEl('div', { class: 'models__list' }, [
    createEl('div', { class: 'empty-state' }, ['Loading models…']),
  ]);
  s.listNoteEl = createEl('div', { class: 'models__list-note muted small', role: 'status' });
  const listSection = createEl('section', { class: 'models__section' }, [s.listEl, s.listNoteEl]);

  const scanSection = createEl('section', { class: 'models__section' }, [
    createEl('div', { class: 'models__section-head' }, [
      createEl('h2', {}, ['Watch folders']),
    ]),
    buildScanControls(ctx),
  ]);

  const root = createEl('div', { class: 'models' }, [
    createEl('h1', {}, ['Models']),
    s.statusEl,
    listSection,
    scanSection,
    buildDangerZone(ctx),
  ]);

  ctx.el.append(root);
}

// Watch folders are the only way a model is served: everything under one is
// served with no models.toml entry. The one-off scan + Import panel that sat
// here was retired with `heylookllm import` (v2.0.72) -- every model lives in
// a watch folder, so there was nothing left for it to add.
function buildScanControls(ctx) {
  const s = ctx.state;

  // WATCH FOLDERS -- server config ([scan].folders in models.toml), not a
  // browser preference.
  s.foldersInput = createEl('textarea', {
    id: 'scan-folders',
    class: 'input',
    rows: '3',
    placeholder: '/path/to/models\n/path/to/more-models',
  });
  s.foldersSaveBtn = createEl('button', { class: 'btn btn--sm' }, ['Save watch folders']);
  s.foldersSaveBtn.addEventListener('click', () => saveWatchFolders(ctx));
  // role=status: the saved-count line is the only feedback that a folder took
  // effect, and it must reach a screen reader (DESIGN.md §7).
  s.foldersNote = createEl('div', { class: 'muted small', role: 'status' }, ['']);

  return createEl('div', { class: 'scan-controls' }, [
    createEl('div', { class: 'scan-controls__row' }, [
      createEl('label', { for: 'scan-folders' }, ['Watch folders (served automatically)']),
      s.foldersInput,
    ]),
    createEl('div', { class: 'scan-controls__row' }, [s.foldersSaveBtn, s.foldersNote]),
    createEl('div', { class: 'muted small' }, [
      'One per line, read on the SERVER. Every model under a watch folder is '
      + 'served. Its settings live in its own folder (model.heylook.toml), '
      + 'written by the config editor below.',
    ]),
  ]);
}

async function loadWatchFolders(ctx) {
  const s = ctx.state;
  try {
    const cfg = await api.adminScanConfig({ signal: ctx.signal });
    if (!ctx.alive) return;
      s.foldersInput.value = (cfg.folders || []).join('\n');
    if (cfg.scan_interval_seconds === 0) {
      s.foldersNote.textContent = 'Discovery is off (scan_interval_seconds = 0).';
    }
  } catch (err) {
    if (ctx.alive) s.foldersNote.textContent = `Could not read watch folders: ${err.message}`;
  }
}

async function saveWatchFolders(ctx) {
  const s = ctx.state;
  if (s.savingFolders) return;
  s.savingFolders = true;
  s.foldersSaveBtn.disabled = true;
  s.foldersNote.textContent = 'Saving…';
  try {
    const folders = s.foldersInput.value.split('\n').map((f) => f.trim()).filter(Boolean);
    const cfg = await api.adminSetScanConfig({ folders });
    if (!ctx.alive) return;
      // models_served is the POINT of the edit -- naming the consequence beats
    // "Saved", which says nothing about whether the folder found anything.
    s.foldersNote.textContent = cfg.warning
      ? cfg.warning
      : `Saved — ${cfg.models_served} models served.`;
    clearError(ctx);
    await fetchModels(ctx, { keepStatus: true });
  } catch (err) {
    if (ctx.alive) s.foldersNote.textContent = `Save failed: ${err.message}`;
  }
  if (!ctx.alive) return;
  s.savingFolders = false;
  s.foldersSaveBtn.disabled = false;
}

function buildDangerZone(ctx) {
  const s = ctx.state;
  s.dangerResultEl = createEl('div', { class: 'models__danger-result muted small' });
  const clearBtn = armedConfirm(
    createEl('button', { class: 'btn btn--sm btn--danger' }, ['Clear all conversations & notebooks']),
    () => clearAllData(ctx),
  );
  return createEl('section', { class: 'models__section models__danger' }, [
    createEl('h2', {}, ['Danger zone']),
    clearBtn,
    s.dangerResultEl,
  ]);
}

function showError(ctx, message) {
  ctx.state.statusEl.replaceChildren(createEl('div', { class: 'error-note', role: 'alert' }, [message]));
}

function clearError(ctx) {
  ctx.state.statusEl.replaceChildren();
}

// ---------------------------------------------------------------------------
// model list
// ---------------------------------------------------------------------------

// `keepStatus` marks an INTERNAL refetch -- the list refresh that trails a
// load/unload/import. Those must not clear the status area, because the
// action that triggered them may have just written its own failure there and
// this refetch succeeding says nothing about that. Without it, every error
// this page can raise was painted and then wiped ~200ms later: the models
// page has in fact never shown a load failure. A refetch still reports its
// OWN failure either way, since that is news.
async function fetchModels(ctx, { keepStatus = false } = {}) {
  const s = ctx.state;
  try {
    const data = await api.adminListModels({ signal: ctx.signal });
    if (!ctx.alive) return;
    s.models = data.models ?? [];
    if (!keepStatus) clearError(ctx);
  } catch (err) {
    if (!ctx.alive) return;
    showError(ctx, `Could not load models: ${err.message}`);
  }
  renderModelList(ctx);
}

function renderModelList(ctx) {
  const s = ctx.state;
  if (!s.models.length) {
    s.listEl.replaceChildren(
      createEl('div', { class: 'empty-state' }, [
        'No models yet. Add a watch folder below — everything under one is '
        + 'served without a models.toml entry.',
      ]),
    );
    return;
  }
  const children = [];
  let panelAnnounce = null;
  for (const m of s.models) {
    children.push(buildModelRow(ctx, m));
    if (s.configOpenId === m.id) {
      // Disclosure, not a confirm: saving a discovered model's first setting
      // creates its models.toml entry. Nothing is lost by that -- it is how
      // an override comes into existence -- so it gets stated, not gated.
      if (m.source === 'discovered') {
        children.push(createEl('div', { class: 'config-panel__note muted small' }, [
          'Found by a watch folder, so it has no models.toml entry yet. '
          + 'Saving a setting here creates one; everything else keeps being '
          + 'detected at load.',
        ]));
      }
      const panel = buildConfigPanel(ctx, m);
      children.push(panel.el);
      panelAnnounce = panel.announce;
    }
  }
  // An open panel whose model vanished (removed entry) just doesn't render;
  // its draft stays in configDrafts. Not quite harmless since v2.0.35: an
  // unsaved template body in such a draft keeps the unload guard armed with
  // no panel left to save or clear it from. Kept anyway -- a vanished model
  // is usually one rescan from coming back, and dropping the text to quiet a
  // dialog would be the loss the dialog exists to prevent.
  s.listEl.replaceChildren(...children);
  // After mounting: a live region only announces text written into a node
  // already in the document, so the panel's status note is written here, not
  // baked into the DOM it mounted with.
  if (panelAnnounce) setTimeout(panelAnnounce, 0);
}

function modelMetaLine(model) {
  const parts = [model.provider];
  if (model.capabilities?.length) parts.push(model.capabilities.join(', '));
  // The context window, derived server-side for every provider that has one
  // (gguf header / MLX config.json) -- answered for unloaded models too.
  const ceiling = contextCeiling(model);
  if (ceiling) parts.push(`ctx ${formatTokens(ceiling)}`);
  if (model.config?.chat_template_source) {
    parts.push(`template: ${model.config.chat_template_source}`);
  }
  if (model.tags?.length) parts.push(model.tags.join(', '));
  // A discovered model is served exactly like any other -- this is NOT a
  // warning, and it must not read as one. It is here because the config
  // panel behaves differently (the first save writes an entry) and because
  // "why does this have no stored settings" is otherwise unanswerable: an
  // entry with every field defaulted serializes identically to no entry.
  if (model.source === 'discovered') parts.push('no entry');
  return parts.join(' · ');
}

// Not perf.js's `buildModelRow`, which renders a metrics row from a bare id.
// Same name, unrelated output.
function buildModelRow(ctx, model) {
  const s = ctx.state;
  const busy = s.loadingIds.has(model.id);

  const badge = createEl('span', {
    class: `model-badge${model.loaded ? ' model-badge--loaded' : ''}`,
  }, [model.loaded ? 'Loaded' : 'Idle']);

  const main = [
    createEl('div', { class: 'model-row__title' }, [createEl('strong', {}, [model.id]), badge]),
    createEl('div', { class: 'model-row__meta muted small' }, [modelMetaLine(model)]),
  ];
  if (model.description) {
    main.push(createEl('div', { class: 'model-row__desc muted small' }, [model.description]));
  }
  // Non-default load options say so on the list -- the discoverability
  // mechanism for "why is this model configured differently from its twin".
  const summary = configSummary(model.config,
    s.optionsSchema?.providers?.[model.provider]?.fields);
  if (summary) {
    main.push(createEl('div', { class: 'model-row__conf small' }, [summary]));
  }
  // SERVER-derived (router compares the loaded process's snapshot against
  // the saved config): survives page remounts and other tabs, clears itself
  // on the refetch after a reload/unload. Purely visual here -- the panel's
  // live-region note carries the announcement.
  if (model.stale_reload_fields?.length) {
    main.push(createEl('div', { class: 'model-row__stale small' },
      ['config changed — reload to apply']));
  }
  main.push(buildEnginePanel(ctx, model));

  const btn = createEl('button', { class: 'btn btn--sm' }, [
    busy ? (model.loaded ? 'Unloading…' : 'Loading…') : (model.loaded ? 'Unload' : 'Load'),
  ]);
  const fitGate = !model.loaded ? s.fitGates.get(model.id) : null;
  btn.disabled = busy || Boolean(fitGate);
  if (fitGate) btn.title = fitGate;
  btn.addEventListener('click', () => toggleLoad(ctx, model));

  const open = s.configOpenId === model.id;
  const cfgBtn = createEl('button', {
    class: 'btn btn--sm',
    'aria-expanded': open ? 'true' : 'false',
  }, [open ? 'Close' : 'Configure']);
  cfgBtn.addEventListener('click', () => toggleConfig(ctx, model));

  // Load/Unload stays the FIRST button in the actions cell -- it is the
  // primary action, and the E2E helpers address it positionally (the fit
  // gate's in-place button update below relies on the same invariant).
  return createEl('div', { class: 'model-row', dataset: { modelId: model.id } }, [
    createEl('div', { class: 'model-row__main' }, main),
    createEl('div', { class: 'model-row__actions' }, [btn, cfgBtn]),
  ]);
}

// The engine contract, rendered by the shared renderer (js/engine.js): what
// runs this model, with what, and why -- every setting included, per-request
// defaults too, since a default quietly set in models.toml is exactly what
// this panel exists to show. Built only when opened (the list re-renders
// often), and its open state outlives the re-render.
function buildEnginePanel(ctx, model) {
  const s = ctx.state;
  const panel = createEl('details', { class: 'engine-panel', dataset: { modelId: model.id } }, [
    createEl('summary', { class: 'engine-panel__summary small' }, [
      createEl('span', { class: 'engine-panel__label' }, ['Engine']),
      ' ',
      createEl('span', { class: 'muted' }, [engineSummary(model.engine)]),
    ]),
  ]);
  const fill = () => {
    if (panel.dataset.filled) return;
    panel.dataset.filled = '1';
    panel.append(renderEngine(model.engine,
      { fields: s.optionsSchema?.providers?.[model.provider]?.fields }));
  };
  if (s.engineOpenIds.has(model.id)) {
    panel.open = true;
    fill();
  }
  panel.addEventListener('toggle', () => {
    if (panel.open) {
      s.engineOpenIds.add(model.id);
      fill();
    } else {
      s.engineOpenIds.delete(model.id);
    }
  });
  return panel;
}

async function toggleLoad(ctx, model) {
  const s = ctx.state;
  if (s.loadingIds.has(model.id)) return;
  const wasLoaded = model.loaded;
  // Belt to the disabled-button suspender: the gate must hold even from a
  // stale row (e.g. a render raced the fit response).
  if (!wasLoaded && s.fitGates.get(model.id)) {
    showError(ctx, `Not loading: ${s.fitGates.get(model.id)}`);
    return;
  }

  s.loadingIds.add(model.id);
  renderModelList(ctx);

  try {
    if (wasLoaded) {
      await api.adminUnloadModel(model.id);
      if (!ctx.alive) return;
      clearError(ctx);
    } else {
      // warm=true: load, then run a 1-token generation through the real
      // path so the Metal kernel JIT is paid here rather than by whoever
      // sends the first message. Same readiness call the dev-server script
      // and the E2E harness use -- "Loaded" should mean ready, not merely
      // resident. A warm failure still leaves the model loaded and the
      // server usable, so it reports as a note, not an error.
      const result = await api.adminLoadModel(model.id, true);
      if (!ctx.alive) return;
      if (result?.warm_error) {
        showError(ctx, `Loaded, but the warm-up generation failed: ${result.warm_error}`);
      } else {
        clearError(ctx);
        setLoadNote(ctx, model.id, result?.warm_ms);
      }
    }
  } catch (err) {
    if (!ctx.alive) return;
    showError(ctx, `${wasLoaded ? 'Unload' : 'Load'} failed: ${err.message}`);
  }

  s.loadingIds.delete(model.id);
  // keepStatus: this handler already wrote its outcome (cleared, or an
  // error); the refresh must not overwrite it. The note is flushed after,
  // because the list re-render is what it annotates.
  if (ctx.alive) await fetchModels(ctx, { keepStatus: true });
  if (ctx.alive) flushLoadNote(ctx);
}

// Warm timing is the only evidence that "Loaded" meant ready rather than
// merely resident. It gets its own element rather than the shared status
// area, which the page invariant reserves for errors (same reason the
// danger zone reports into its own result line).
function setLoadNote(ctx, modelId, warmMs) {
  ctx.state.pendingLoadNote = warmMs == null
    ? null
    : `${modelId} loaded and warmed in ${(warmMs / 1000).toFixed(1)}s.`;
}

function flushLoadNote(ctx) {
  const s = ctx.state;
  s.listNoteEl.textContent = s.pendingLoadNote || '';
  s.pendingLoadNote = null;
}

// ---------------------------------------------------------------------------
// per-model config editor (schema-driven -- see model-config.js)
// ---------------------------------------------------------------------------

async function toggleConfig(ctx, model) {
  const s = ctx.state;
  if (s.configOpenId === model.id) {
    s.configOpenId = null;
    renderModelList(ctx);
    return;
  }
  // The option schema is one static payload for all models; fetch once per
  // page mount, before first paint of the panel (it decides every control).
  // The promise is cached so a double-click doesn't fetch twice.
  if (!s.optionsSchema) {
    if (!s.optionsPromise) s.optionsPromise = api.adminModelOptions({ signal: ctx.signal });
    try {
      s.optionsSchema = await s.optionsPromise;
    } catch (err) {
      s.optionsPromise = null;
      if (!ctx.alive) return;
      showError(ctx, `Could not load option schema: ${err.message}`);
      return;
    }
    if (!ctx.alive) return;
  }
  s.configOpenId = model.id;
  renderModelList(ctx);
}

// Asks every draft on the page, never just the open panel's: the answer has
// to stay right for a model whose panel was closed with text still in it.
function syncUnsavedGuard(ctx) {
  const s = ctx.state;
  s.setUnloadGuard([...s.configDrafts.values()].some(hasUnsavedTemplate));
}

function buildConfigPanel(ctx, model) {
  const s = ctx.state;
  const fields = s.optionsSchema?.providers?.[model.provider]?.fields ?? [];
  let draft = s.configDrafts.get(model.id);
  if (!draft) {
    draft = {};
    s.configDrafts.set(model.id, draft);
  }
  // One-shot: consume the save note stashed for this model by the last save.
  let initialNote = null;
  if (s.configSaveNote?.id === model.id) {
    initialNote = s.configSaveNote.text;
    s.configSaveNote = null;
  }
  return createModelConfigEditor({
    model,
    fields,
    draft,
    initialNote,
    onError: (msg) => showError(ctx, msg),
    onSaved: (updatedModel, noteText) => {
      clearError(ctx);
      // Swap the row's model in place (the PATCH response carries the
      // post-save config AND stale_reload_fields) and re-render, so the
      // row's marker/chip repaint and the rebuilt panel restores the
      // outcome via the stashed note. A full refetch would say nothing
      // this response doesn't already say.
      const idx = s.models.findIndex((m) => m.id === model.id);
      if (idx >= 0 && updatedModel) s.models[idx] = { ...s.models[idx], ...updatedModel };
      s.configSaveNote = { id: model.id, text: noteText };
      renderModelList(ctx);
    },
    onDraftChange: () => syncUnsavedGuard(ctx),
    onReload: () => reloadModel(ctx, model),
    onReset: () => renderModelList(ctx),
    // The fit meter's Load gate (design §5: MLX FAIL disables Load with the
    // reason; gguf's over-working-set is a warn and never gates). Updates the
    // live row button IN PLACE -- a full re-render here would rebuild the
    // editor under the user's cursor on every debounced fit response.
    onFitGate: (reason) => {
      if (!ctx.alive) return;
      if (reason) s.fitGates.set(model.id, reason);
      else s.fitGates.delete(model.id);
      const row = s.listEl?.querySelector(
        `.model-row[data-model-id="${CSS.escape(model.id)}"]`);
      const loadBtn = row?.querySelector('.model-row__actions button');
      if (loadBtn && !model.loaded && !s.loadingIds.has(model.id)) {
        loadBtn.disabled = Boolean(reason);
        loadBtn.title = reason || '';
      }
    },
  });
}

// The "Reload now" cycle after a reload-required save: teardown + fresh load
// through the same warm path the Load button uses, so "Loaded" keeps meaning
// ready. Reuses the per-row busy set -- the row renders Loading… while the
// cycle runs.
async function reloadModel(ctx, model) {
  const s = ctx.state;
  if (s.loadingIds.has(model.id)) return;
  s.loadingIds.add(model.id);
  renderModelList(ctx);

  try {
    // Server-owned reload (v1.62.0): one call, so a dying browser can no
    // longer strand the model unloaded between an unload and a load.
    const result = await api.adminReloadModel(model.id, true);
    if (!ctx.alive) return;
    if (result?.warm_error) {
      showError(ctx, `Reloaded, but the warm-up generation failed: ${result.warm_error}`);
    } else {
      clearError(ctx);
      setLoadNote(ctx, model.id, result?.warm_ms);
    }
  } catch (err) {
    if (!ctx.alive) return;
    showError(ctx, `Reload failed: ${err.message}`);
  }

  s.loadingIds.delete(model.id);
  if (ctx.alive) await fetchModels(ctx, { keepStatus: true });
  if (ctx.alive) flushLoadNote(ctx);
}

// ---------------------------------------------------------------------------
// danger zone
// ---------------------------------------------------------------------------

async function clearAllData(ctx) {
  const s = ctx.state;
  try {
    const result = await api.clearAllData();
    if (!ctx.alive) return;
    clearError(ctx);
    s.dangerResultEl.textContent =
      `Deleted ${result.conversations_deleted} conversations, ${result.notebooks_deleted} notebooks.`;
  } catch (err) {
    if (!ctx.alive) return;
    showError(ctx, `Clear failed: ${err.message}`);
  }
}
