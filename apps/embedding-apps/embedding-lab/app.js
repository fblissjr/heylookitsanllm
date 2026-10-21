import {
  MAX_FILE_BYTES, cancel, embedWithRetry, kindOf, listEmbeddingModels, mediaBlock, newRequestId,
  planBatches, textBlock,
} from '../kit/embed-kit.js';
import { buildPlan, evaluateCase, summarize } from './metrics.js';

const $ = (id) => document.getElementById(id);
const ui = {
  model: $('model'), dims: $('dims'), queryPrompt: $('queryPrompt'), docPrompt: $('docPrompt'),
  add: $('add'), configs: $('configs'), run: $('run'), stop: $('stop'), download: $('download'),
  progress: $('progress'), status: $('status'), summary: $('summary'), detail: $('detail'), setInfo: $('setInfo'),
};
const state = { models: [], setName: '', files: new Map(), plan: null, configs: [], results: [], abort: null, requestId: null, busy: false };

const say = (t) => { ui.status.textContent = t; };
const model = () => state.models.find((m) => m.id === ui.model.value);
const label = (c) => `${c.model} · ${c.dimensions}d · query prompt: ${c.queryPrompt || 'none'} · document prompt: ${c.docPrompt || 'none'}`;

function option(value, text = value) { const o = document.createElement('option'); o.value = value; o.textContent = text; return o; }
function cell(tag, text, cls) { const c = document.createElement(tag); c.textContent = text; if (cls) c.className = cls; return c; }
const signed = (n) => (n == null ? '–' : `${n < 0 ? '−' : '+'}${Math.abs(n).toFixed(3)}`);

function fillPrompts(select, preferred) {
  const names = Object.keys(model()?.engine.embedding.prompts ?? {});
  select.replaceChildren(option('', 'No task prompt'), ...names.map((n) => option(n)));
  if (names.includes(preferred)) select.value = preferred;
}

function refreshModel() {
  const info = model()?.engine.embedding;
  if (!info) return;
  ui.dims.replaceChildren(...info.dimensions.map((n) => option(String(n))));
  ui.dims.value = String(info.native_dimensions);
  fillPrompts(ui.queryPrompt, 'query');
  fillPrompts(ui.docPrompt, 'document');
}

function renderConfigs() {
  ui.configs.replaceChildren(...state.configs.map((c, i) => {
    const li = document.createElement('li');
    const text = document.createElement('span'); text.textContent = label(c);
    const rm = document.createElement('button'); rm.type = 'button'; rm.textContent = 'Remove';
    rm.disabled = state.busy;
    rm.addEventListener('click', () => { state.configs.splice(i, 1); renderConfigs(); });
    li.append(text, rm);
    return li;
  }));
  ui.run.disabled = state.busy || !state.plan || !state.configs.length;
}

function addConfig() {
  const c = { model: ui.model.value, dimensions: Number(ui.dims.value), queryPrompt: ui.queryPrompt.value, docPrompt: ui.docPrompt.value };
  if (state.configs.some((o) => label(o) === label(c))) { say('That configuration is already listed.'); return; }
  state.configs.push(c);
  renderConfigs();
}

async function loadFolder(fileList) {
  const files = new Map();
  for (const f of fileList) {
    const rel = (f.webkitRelativePath || f.name).split('/').slice(1).join('/') || f.name;
    files.set(rel, f);
  }
  state.plan = null; state.files = files;
  try {
    const setFile = files.get('set.json');
    if (!setFile) throw new Error('No set.json at the top of that folder.');
    const set = JSON.parse(await setFile.text());
    const plan = buildPlan(set, (path) => {
      const f = files.get(path);
      return f && kindOf(f) && f.size <= MAX_FILE_BYTES;
    });
    state.plan = plan; state.setName = set.name || 'test set';
    const media = plan.pool.filter((p) => p.file).length + plan.queries.filter((q) => q.file).length;
    ui.setInfo.textContent = `${state.setName}: ${plan.cases.length} cases, ${plan.pool.length} documents, ${media} media references.`;
  } catch (error) {
    ui.setInfo.textContent = `Cannot use that folder: ${error.message}`;
  }
  renderConfigs();
}

const blocks = new Map();
function mediaFor(path) {
  if (!blocks.has(path)) blocks.set(path, mediaBlock(state.files.get(path)));
  return blocks.get(path);
}
async function contentFor(item) {
  const content = [];
  if (item.text) content.push(textBlock(item.text));
  if (item.file) content.push(await mediaFor(item.file));
  return content;
}

/** Embed items in order under one prompt. Atomic: any failure aborts the configuration. */
async function embedAll(items, prompt, cfg) {
  const info = state.models.find((m) => m.id === cfg.model).engine.embedding;
  for (const item of items) {
    const kind = item.file && kindOf(state.files.get(item.file));
    if (kind && !info.modalities.includes(kind)) throw new Error(`${cfg.model} does not take ${kind} (${item.file}).`);
  }
  const entries = items.map((item) => ({ item, bytes: item.file ? state.files.get(item.file).size : 1 }));
  const vectors = [];
  for (const batch of planBatches(entries)) {
    if (state.abort.signal.aborted) throw new DOMException('stopped', 'AbortError');
    const input = [];
    for (const e of batch) input.push({ content: await contentFor(e.item), ...(prompt ? { prompt_name: prompt } : {}) });
    state.requestId = newRequestId();
    const result = await embedWithRetry({
      model: cfg.model, dimensions: cfg.dimensions, items: input, requestId: state.requestId, signal: state.abort.signal,
    });
    if (result.data.length !== batch.length) throw new Error('The server returned the wrong number of vectors.');
    for (const row of result.data) vectors.push(row.embedding);
  }
  return vectors;
}

async function runConfig(cfg) {
  const { plan } = state;
  const started = performance.now();
  const pool = await embedAll(plan.pool, cfg.docPrompt, cfg);
  const queries = await embedAll(plan.queries, cfg.queryPrompt, cfg);
  const cases = plan.cases.map((c, i) => ({ id: c.id, ...evaluateCase(queries[i], pool, new Set(c.positives), new Set(c.excluded)) }));
  return { cfg, label: label(cfg), cases, summary: summarize(cases), seconds: (performance.now() - started) / 1000 };
}

function setBusy(on) {
  state.busy = on;
  ui.stop.hidden = !on; ui.progress.hidden = !on;
  for (const el of [ui.add, ui.model, ui.dims, ui.queryPrompt, ui.docPrompt]) el.disabled = on;
  renderConfigs();
}

async function runAll() {
  if (state.busy || !state.plan) return;
  // Group by model so a model that has to load (or an engine switch) happens once.
  const order = [...state.configs].sort((a, b) => a.model.localeCompare(b.model));
  state.results = []; render();
  setBusy(true);
  state.abort = new AbortController();
  ui.progress.max = order.length; ui.progress.value = 0;
  try {
    for (const [i, cfg] of order.entries()) {
      say(`Running ${i + 1} of ${order.length}: ${label(cfg)}. The model loads automatically if needed.`);
      state.results.push(await runConfig(cfg));
      ui.progress.value = i + 1;
      render();
    }
    say(`Done. ${order.length} configuration${order.length === 1 ? '' : 's'}. Scores are cosine similarity.`);
  } catch (error) {
    say(error.name === 'AbortError' ? 'Stopped. Finished configurations are shown.' : `Failed: ${error.message}`);
  } finally {
    setBusy(false);
    ui.download.disabled = !state.results.length;
  }
}

function describe(item) { return [item.text && item.text.slice(0, 50), item.file].filter(Boolean).join(' + '); }

function bestIndexes(values, higherIsBetter) {
  const usable = values.filter((v) => v != null);
  if (usable.length < 2) return new Set();
  const best = higherIsBetter ? Math.max(...usable) : Math.min(...usable);
  return new Set(values.map((v, i) => (v === best ? i : -1)).filter((i) => i >= 0));
}

function render() {
  ui.summary.replaceChildren(); ui.detail.replaceChildren();
  if (!state.results.length) return;
  const cols = [
    ['Recall@1', (s) => s.r1, (v) => `${(v * 100).toFixed(0)}%`, true],
    ['Recall@3', (s) => s.r3, (v) => `${(v * 100).toFixed(0)}%`, true],
    ['MRR', (s) => s.mrr, (v) => v.toFixed(3), true],
    ['Mean margin', (s) => s.meanMargin, signed, true],
    ['Worst margin', (s) => s.minMargin, signed, true],
  ];
  const table = document.createElement('table');
  const cap = document.createElement('caption'); cap.textContent = `Summary: ${state.plan.cases.length} cases (underlined = best in column)`;
  const head = document.createElement('tr'); head.append(cell('th', 'Configuration'), ...cols.map((c) => cell('th', c[0])), cell('th', 'Seconds'));
  table.append(cap, head);
  const bests = cols.map(([, get, , up]) => bestIndexes(state.results.map((r) => get(r.summary)), up));
  state.results.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.append(cell('th', r.label));
    cols.forEach(([, get, fmt], j) => { const v = get(r.summary); tr.append(cell('td', v == null ? '–' : fmt(v), `num${bests[j].has(i) ? ' best' : ''}`)); });
    tr.append(cell('td', r.seconds.toFixed(1), 'num'));
    table.append(tr);
  });
  ui.summary.append(table);

  const d = document.createElement('table');
  const dcap = document.createElement('caption');
  dcap.textContent = 'Per case: rank of the best right answer among all documents, and its margin over the best wrong one. ✗ = a wrong document ranked first.';
  const dh = document.createElement('tr'); dh.append(cell('th', 'Case'), ...state.results.map((r) => cell('th', r.label)));
  d.append(dcap, dh);
  state.plan.cases.forEach((c, i) => {
    const tr = document.createElement('tr');
    tr.append(cell('th', c.id));
    for (const r of state.results) {
      const x = r.cases[i];
      const td = cell('td', `${x.rank === 1 ? '✓' : '✗'} #${x.rank}  ${signed(x.margin)}`, x.rank === 1 ? '' : 'fail');
      if (x.rank !== 1 && x.bestNegIndex >= 0) {
        const note = document.createElement('div'); note.className = 'muted';
        note.textContent = `beaten by: ${describe(state.plan.pool[x.bestNegIndex])}`;
        td.append(note);
      }
      tr.append(td);
    }
    d.append(tr);
  });
  ui.detail.append(d);
}

function downloadResults() {
  const payload = { set: state.setName, cases: state.plan.cases.map((c) => c.id), results: state.results };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = 'embedding-lab-results.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function init() {
  try { state.models = await listEmbeddingModels(); }
  catch (error) { say(`Cannot reach the heylook server: ${error.message}`); return; }
  if (!state.models.length) { say('No embedding models are registered on the server.'); return; }
  ui.model.replaceChildren(...state.models.map((m) => option(m.id)));
  ui.model.addEventListener('change', refreshModel);
  ui.add.addEventListener('click', addConfig);
  ui.run.addEventListener('click', runAll);
  ui.stop.addEventListener('click', () => { state.abort?.abort(); cancel(state.requestId); say('Stopping…'); });
  ui.download.addEventListener('click', downloadResults);
  $('folder').addEventListener('change', (e) => { loadFolder([...e.target.files]); e.target.value = ''; });
  window.addEventListener('pagehide', () => { if (state.busy) cancel(state.requestId); });
  refreshModel();
  say('Choose a test folder and add at least one configuration.');
}

init();
