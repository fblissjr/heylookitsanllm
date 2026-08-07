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
    s.pendingLoadNote = null;   // warm-timing note, flushed after the list refetch

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
  s.listNoteEl = createEl('div', { class: 'models__list-note muted small', role: 'status' });
  const listSection = createEl('section', { class: 'models__section' }, [s.listEl, s.listNoteEl]);

  s.scanBtn = createEl('button', { class: 'btn btn--sm' }, ['Scan']);
  s.scanBtn.addEventListener('click', () => handleScan(ctx));
  s.scanResultsEl = createEl('div', { class: 'scan-panel' });
  const scanSection = createEl('section', { class: 'models__section' }, [
    createEl('div', { class: 'models__section-head' }, [
      createEl('h2', {}, ['Find models']),
      s.scanBtn,
    ]),
    buildScanControls(ctx),
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

// Scan sources. The HF cache is ONE source, not the only one: local folders
// (a modelzoo of GGUFs, a converted checkpoint anywhere on disk) are found by
// the same endpoint's `paths`, which the UI simply never sent -- so every
// locally-downloaded model was unreachable from this page. The folder list
// persists locally because re-typing a path on every scan is the thing that
// makes an operator go edit models.toml by hand instead.
function buildScanControls(ctx) {
  const s = ctx.state;

  s.pathsInput = createEl('input', {
    id: 'scan-paths',
    type: 'text',
    class: 'input',
    placeholder: 'modelzoo/gguf, modelzoo',
    value: loadScanPaths(),
  });
  s.pathsInput.addEventListener('change', () => saveScanPaths(s.pathsInput.value));

  s.hfBox = createEl('input', { id: 'scan-hf', type: 'checkbox', checked: true });

  return createEl('div', { class: 'scan-controls' }, [
    createEl('div', { class: 'scan-controls__row' }, [
      createEl('label', { for: 'scan-paths' }, ['Folders to scan']),
      s.pathsInput,
    ]),
    createEl('div', { class: 'scan-controls__row scan-controls__row--check' }, [
      s.hfBox,
      createEl('label', { for: 'scan-hf' }, ['Also scan the HuggingFace cache']),
    ]),
    createEl('div', { class: 'muted small' }, [
      'Comma- or newline-separated. Paths are read on the SERVER, so a relative '
      + 'one resolves against the server\'s working directory, not the browser\'s.',
    ]),
  ]);
}

const SCAN_PATHS_KEY = 'heylook-v3-scan-paths';

function loadScanPaths() {
  try { return localStorage.getItem(SCAN_PATHS_KEY) || ''; }
  catch { return ''; }
}

function saveScanPaths(value) {
  try { localStorage.setItem(SCAN_PATHS_KEY, value); }
  catch { /* private mode / quota -- scanning still works, it just won't stick */ }
}

function parsePaths(raw) {
  return raw.split(/[,\n]/).map((p) => p.trim()).filter(Boolean);
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
  if (ctx.alive) await fetchModels(ctx);
  // After fetchModels, never before: it clears the error area on success,
  // which would swallow a note written earlier in this handler.
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
// scan + import
// ---------------------------------------------------------------------------

async function handleScan(ctx) {
  const s = ctx.state;
  if (s.scanning) return;

  const paths = parsePaths(s.pathsInput.value);
  const scanHf = s.hfBox.checked;
  if (!paths.length && !scanHf) {
    showError(ctx, 'Nothing to scan: add a folder or enable the HuggingFace cache.');
    return;
  }
  saveScanPaths(s.pathsInput.value);

  s.scanning = true;
  s.scanBtn.disabled = true;
  s.scanBtn.textContent = 'Scanning…';

  try {
    const data = await api.adminScan({ paths, scan_hf_cache: scanHf });
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
  s.scanBtn.textContent = 'Scan';
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

// What the scan actually found. The importer reads projector headers, the
// model's own chat template and the drafter's name prefix; all of it used to
// stop at the server log, so this row could only ever say "vision" -- and
// after the thin-entry change it could not honestly say even that. Modalities
// beyond text are what change the chat UI (attach button, thinking toggle),
// so they are the facts worth showing before deciding to import.
function scanMetaLine(result) {
  const parts = [`${result.size_gb.toFixed(1)} GB`, result.provider];
  if (result.quantization) parts.push(result.quantization);
  const extraModalities = (result.modalities || []).filter((m) => m !== 'text');
  parts.push(...extraModalities);
  if (result.supports_thinking) parts.push('thinking');
  return parts.join(' · ');
}

// The drafter line is deliberately separate and phrased as a fact plus the
// setting it enables: import pairs the drafter PATH but never turns
// speculative decoding on, because whether it pays off is a per-model
// measurement. Naming the required --spec-type here is the difference
// between "do I want this on" and "what is this drafter called".
function scanDraftLine(result) {
  if (!result.draft_model_path) return null;
  const name = result.draft_model_path.split('/').pop();
  const text = result.draft_spec_type
    ? `drafter ${name} — set spec_type = "${result.draft_spec_type}" to enable speculative decoding`
    : `drafter ${name} — unrecognised prefix, spec_type must be set by hand`;
  return createEl('div', { class: 'scan-row__note muted small' }, [text]);
}

function buildScanRow(ctx, result) {
  const s = ctx.state;
  const busy = s.importingIds.has(result.id);

  const main = createEl('div', { class: 'scan-row__main' }, [
    createEl('strong', {}, [result.id]),
    createEl('span', { class: 'scan-row__meta muted small' }, [scanMetaLine(result)]),
    scanDraftLine(result),
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
