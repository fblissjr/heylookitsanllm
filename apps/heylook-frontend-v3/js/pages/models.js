// Models admin: list + load/unload, HF cache scan + import, danger zone.
//
// Invariants:
// - One status/error area at the top. Failed actions write `.error-note`
//   text there; any subsequent successful action clears it. No console-only
//   failures, no native alert()/confirm().
// - Busy flags are always per-row (Set keyed by model/result id), never a
//   single page-wide flag -- unrelated rows must stay interactive.

import { createPage } from '../page.js';
import { createEl, armedConfirm } from '../utils.js';
import { api } from '../api.js';
import * as drawer from '../settings-drawer.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.models = [];
    s.loadingIds = new Set();   // model ids mid load/unload
    s.scanResults = null;       // null until a scan has run
    s.scanning = false;
    s.importingIds = new Set(); // scan result ids mid import

    buildSkeleton(ctx);
    // No sampler/page settings here -- register so the drawer still offers the
    // global Display prefs and hides the sampler panel.
    ctx.onTeardown(drawer.registerSettings({ samplers: 'hidden' }));
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
  const listSection = createEl('section', { class: 'models__section' }, [s.listEl]);

  s.scanBtn = createEl('button', { class: 'btn btn--sm' }, ['Scan HF cache']);
  s.scanBtn.addEventListener('click', () => handleScan(ctx));
  s.scanResultsEl = createEl('div', { class: 'scan-panel' });
  const scanSection = createEl('section', { class: 'models__section' }, [
    createEl('div', { class: 'models__section-head' }, [
      createEl('h2', {}, ['Find models']),
      s.scanBtn,
    ]),
    s.scanResultsEl,
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

async function fetchModels(ctx) {
  const s = ctx.state;
  try {
    const data = await api.adminListModels({ signal: ctx.signal });
    if (!ctx.alive) return;
    s.models = data.models ?? [];
    clearError(ctx);
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
        'models.toml has no entries yet. Use "Scan HF cache" below to find models to import.',
      ]),
    );
    return;
  }
  s.listEl.replaceChildren(...s.models.map((m) => buildModelRow(ctx, m)));
}

function modelMetaLine(model) {
  const parts = [model.provider];
  if (model.capabilities?.length) parts.push(model.capabilities.join(', '));
  if (model.config?.chat_template_source) {
    parts.push(`template: ${model.config.chat_template_source}`);
  }
  if (model.tags?.length) parts.push(model.tags.join(', '));
  if (!model.enabled) parts.push('disabled');
  return parts.join(' · ');
}

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

  const btn = createEl('button', { class: 'btn btn--sm' }, [
    busy ? (model.loaded ? 'Unloading…' : 'Loading…') : (model.loaded ? 'Unload' : 'Load'),
  ]);
  btn.disabled = busy;
  btn.addEventListener('click', () => toggleLoad(ctx, model));

  return createEl('div', { class: 'model-row' }, [
    createEl('div', { class: 'model-row__main' }, main),
    createEl('div', { class: 'model-row__actions' }, [btn]),
  ]);
}

async function toggleLoad(ctx, model) {
  const s = ctx.state;
  if (s.loadingIds.has(model.id)) return;
  const wasLoaded = model.loaded;

  s.loadingIds.add(model.id);
  renderModelList(ctx);

  try {
    if (wasLoaded) await api.adminUnloadModel(model.id);
    else await api.adminLoadModel(model.id);
    if (!ctx.alive) return;
    clearError(ctx);
  } catch (err) {
    if (!ctx.alive) return;
    showError(ctx, `${wasLoaded ? 'Unload' : 'Load'} failed: ${err.message}`);
  }

  s.loadingIds.delete(model.id);
  if (ctx.alive) await fetchModels(ctx);
}

// ---------------------------------------------------------------------------
// scan + import
// ---------------------------------------------------------------------------

async function handleScan(ctx) {
  const s = ctx.state;
  if (s.scanning) return;

  s.scanning = true;
  s.scanBtn.disabled = true;
  s.scanBtn.textContent = 'Scanning…';

  try {
    const data = await api.adminScan({ scan_hf_cache: true });
    if (!ctx.alive) return;
    s.scanResults = data.models ?? [];
    clearError(ctx);
  } catch (err) {
    if (!ctx.alive) return;
    showError(ctx, `Scan failed: ${err.message}`);
  }

  s.scanning = false;
  if (!ctx.alive) return;
  s.scanBtn.disabled = false;
  s.scanBtn.textContent = 'Scan HF cache';
  renderScanResults(ctx);
}

function renderScanResults(ctx) {
  const s = ctx.state;
  if (s.scanResults === null) {
    s.scanResultsEl.replaceChildren();
    return;
  }

  const newOnes = s.scanResults.filter((r) => !r.already_configured);
  const configuredCount = s.scanResults.length - newOnes.length;

  const children = [];
  if (configuredCount > 0) {
    children.push(createEl('div', { class: 'muted small' }, [`${configuredCount} already configured.`]));
  }
  if (!newOnes.length) {
    children.push(createEl('div', { class: 'empty-state' }, ['No new models found.']));
  } else {
    children.push(...newOnes.map((r) => buildScanRow(ctx, r)));
  }
  s.scanResultsEl.replaceChildren(...children);
}

function scanMetaLine(result) {
  const parts = [`${result.size_gb.toFixed(1)} GB`, result.provider];
  if (result.vision) parts.push('vision');
  return parts.join(' · ');
}

function buildScanRow(ctx, result) {
  const s = ctx.state;
  const busy = s.importingIds.has(result.id);

  const main = createEl('div', { class: 'scan-row__main' }, [
    createEl('strong', {}, [result.id]),
    createEl('span', { class: 'scan-row__meta muted small' }, [scanMetaLine(result)]),
  ]);

  const btn = createEl('button', { class: 'btn btn--sm' }, [busy ? 'Importing…' : 'Import']);
  btn.disabled = busy;
  btn.addEventListener('click', () => importModel(ctx, result));

  return createEl('div', { class: 'scan-row' }, [main, createEl('div', { class: 'scan-row__actions' }, [btn])]);
}

async function importModel(ctx, result) {
  const s = ctx.state;
  if (s.importingIds.has(result.id)) return;

  s.importingIds.add(result.id);
  renderScanResults(ctx);

  let ok = false;
  try {
    await api.adminImport({ models: [{ id: result.id, path: result.path, provider: result.provider }] });
    ok = true;
  } catch (err) {
    if (ctx.alive) showError(ctx, `Import failed: ${err.message}`);
  }
  if (!ctx.alive) return;

  s.importingIds.delete(result.id);
  if (!ok) {
    renderScanResults(ctx);
    return;
  }

  clearError(ctx);
  s.scanResults = s.scanResults.filter((r) => r.id !== result.id);
  renderScanResults(ctx);
  await fetchModels(ctx);
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
