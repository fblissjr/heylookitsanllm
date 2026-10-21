import {
  MAX_FILE_BYTES, cancel, embedWithRetry, kindOf, listEmbeddingModels, mediaBlock, newRequestId,
  openStore, planBatches, spaceId, textBlock, thumbnail, topK,
} from '../kit/embed-kit.js';

const $ = (id) => document.getElementById(id);
const ui = {
  model: $('model'), dims: $('dims'), docPrompt: $('docPrompt'), queryPrompt: $('queryPrompt'),
  status: $('status'), progress: $('progress'), stop: $('stop'), clear: $('clear'),
  problems: $('problems'), problemList: $('problemList'),
  query: $('query'), example: $('example'), exampleName: $('exampleName'), go: $('go'),
  count: $('count'), results: $('results'),
};
const KIND_ICON = { image: '🖼', audio: '🔊', video: '🎞' };

const state = { models: [], store: null, files: new Map(), abort: null, requestId: null, busy: false, indexed: 0 };

const model = () => state.models.find((m) => m.id === ui.model.value);
const space = () => spaceId({ model: ui.model.value, dimensions: Number(ui.dims.value), prompt: ui.docPrompt.value });
const say = (text) => { ui.status.textContent = text; };

function option(value, label = value) {
  const o = document.createElement('option');
  o.value = value; o.textContent = label;
  return o;
}

/** Fill a prompt picker from the model's own catalog; pre-pick `hint` when one is named like it. */
function fillPrompts(select, hint) {
  const names = Object.keys(model()?.engine.embedding.prompts ?? {});
  select.replaceChildren(option('', 'No task prompt'), ...names.map((n) => option(n)));
  const guess = names.find((n) => n.toLowerCase().includes(hint));
  if (guess) select.value = guess;
}

function refreshModel() {
  const info = model()?.engine.embedding;
  if (!info) return;
  ui.dims.replaceChildren(...info.dimensions.map((n) => option(String(n))));
  const preferred = info.dimensions.includes(512) ? 512 : info.native_dimensions;
  ui.dims.value = String(preferred);
  fillPrompts(ui.docPrompt, 'document');
  fillPrompts(ui.queryPrompt, 'query');
  refreshCount();
}

async function refreshCount() {
  state.indexed = (await state.store.ids(space())).size;
  ui.go.disabled = state.indexed === 0;
  say(state.indexed ? `${state.indexed} files indexed in this configuration.` : 'Nothing indexed yet. Choose a folder or files.');
}

function problem(name, reason) {
  ui.problems.hidden = false;
  const li = document.createElement('li');
  li.textContent = `${name}: ${reason}`;
  ui.problemList.append(li);
}

function setBusy(on) {
  state.busy = on;
  ui.stop.hidden = !on;
  ui.progress.hidden = !on;
  for (const el of [ui.model, ui.dims, ui.docPrompt, ui.clear]) el.disabled = on;
}

/** Index files not already stored in this space. Resumable: a re-run skips what is done. */
async function indexFiles(fileList) {
  if (state.busy) return;
  const info = model().engine.embedding;
  const sp = space();
  const done = await state.store.ids(sp);
  const entries = [];
  ui.problemList.replaceChildren(); ui.problems.hidden = true;
  for (const file of fileList) {
    const path = file.webkitRelativePath || file.name;
    const kind = kindOf(file);
    if (!kind) continue; // folders hold plenty of non-media files; not worth listing
    if (!info.modalities.includes(kind)) { problem(path, `this model does not take ${kind}`); continue; }
    if (file.size > MAX_FILE_BYTES) { problem(path, 'larger than 32 MB'); continue; }
    const id = `${sp}|${path}|${file.size}|${file.lastModified}`;
    state.files.set(id, file);
    if (!done.has(id)) entries.push({ id, path, kind, file, bytes: file.size });
  }
  if (!entries.length) { say('Nothing new to index.'); return; }

  setBusy(true);
  state.abort = new AbortController();
  const batches = planBatches(entries);
  let processed = 0;
  ui.progress.max = entries.length; ui.progress.value = 0;
  try {
    for (const batch of batches) {
      if (state.abort.signal.aborted) break;
      say(`Embedding ${processed + 1}–${processed + batch.length} of ${entries.length}…`);
      const records = await embedBatch(batch, sp);
      await state.store.put(records);
      processed += batch.length;
      ui.progress.value = processed;
    }
    say(state.abort.signal.aborted ? 'Stopped. Progress is saved; choose the folder again to resume.' : 'Done.');
  } catch (error) {
    say(`Failed: ${error.message}. Progress is saved; choose the folder again to resume.`);
  } finally {
    setBusy(false);
    await refreshCount();
  }
}

/** Embed one planned batch; if the server rejects it, retry each file alone so one bad file only skips itself. */
async function embedBatch(batch, sp) {
  const args = { model: ui.model.value, dimensions: Number(ui.dims.value), signal: state.abort.signal };
  const prompt = ui.docPrompt.value ? { prompt_name: ui.docPrompt.value } : {};
  const run = async (entries) => {
    const items = [];
    for (const e of entries) items.push({ content: [await mediaBlock(e.file)], ...prompt });
    state.requestId = newRequestId();
    return embedWithRetry({ ...args, items, requestId: state.requestId });
  };
  const build = async (entries, result) => Promise.all(entries.map(async (e, i) => ({
    id: e.id, space: sp, path: e.path, kind: e.kind, size: e.file.size,
    vec: Float32Array.from(result.data[i].embedding),
    thumb: await thumbnail(e.file, e.kind),
  })));
  try {
    return await build(batch, await run(batch));
  } catch (error) {
    if (batch.length === 1 || state.abort.signal.aborted || error.status !== 400) throw error;
    const records = [];
    for (const e of batch) {
      try { records.push(...await build([e], await run([e]))); }
      catch (single) {
        if (single.status !== 400) throw single;
        problem(e.path, single.message);
      }
    }
    return records;
  }
}

async function search(event) {
  event.preventDefault();
  if (state.busy) return;
  const text = ui.query.value.trim();
  const example = ui.example.files[0];
  if (!text && !example) { say('Type something or choose an example file.'); return; }
  const content = [];
  if (text) content.push(textBlock(text));
  try {
    if (example) {
      if (!kindOf(example) || example.size > MAX_FILE_BYTES) throw new Error('The example must be an image, audio or video file under 32 MB.');
      content.push(await mediaBlock(example));
    }
    say('Searching…');
    state.requestId = newRequestId();
    const result = await embedWithRetry({
      model: ui.model.value, dimensions: Number(ui.dims.value), requestId: state.requestId,
      items: [{ content, ...(ui.queryPrompt.value ? { prompt_name: ui.queryPrompt.value } : {}) }],
    });
    const hits = topK(result.data[0].embedding, await state.store.all(space()));
    render(hits);
    ui.count.textContent = `Top ${hits.length} of ${state.indexed}. Scores are cosine similarity, not probabilities.`;
    say('Done.');
  } catch (error) { say(`Search failed: ${error.message}`); }
}

function render(hits) {
  ui.results.replaceChildren(...hits.map(({ record, score }) => {
    const li = document.createElement('li');
    li.className = 'card';
    const button = document.createElement('button');
    button.type = 'button';
    button.addEventListener('click', () => preview(record));
    let thumb;
    if (record.thumb) { thumb = document.createElement('img'); thumb.src = record.thumb; thumb.alt = ''; }
    else { thumb = document.createElement('div'); thumb.textContent = KIND_ICON[record.kind]; thumb.setAttribute('aria-hidden', 'true'); }
    thumb.classList.add('thumb');
    const meta = document.createElement('div');
    meta.className = 'meta';
    const name = document.createElement('div'); name.textContent = record.path;
    const sc = document.createElement('div'); sc.className = 'score'; sc.textContent = `${record.kind} · ${score.toFixed(3)}`;
    meta.append(name, sc);
    button.append(thumb, meta);
    li.append(button);
    return li;
  }));
}

/** Files are only playable while this tab still holds them; a reloaded index shows thumbnails and paths. */
function preview(record) {
  const file = state.files.get(record.id);
  if (!file) { say(`${record.path}: re-choose its folder to open the original (the index keeps vectors and thumbnails only).`); return; }
  const url = URL.createObjectURL(file);
  const dialog = document.createElement('dialog');
  const media = document.createElement(record.kind === 'image' ? 'img' : record.kind);
  media.src = url;
  if (record.kind !== 'image') media.controls = true;
  if (record.kind === 'image') media.alt = record.path;
  const close = document.createElement('button');
  close.type = 'button'; close.textContent = 'Close';
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { URL.revokeObjectURL(url); dialog.remove(); });
  dialog.append(media, document.createElement('br'), close);
  document.body.append(dialog);
  dialog.showModal();
}

async function init() {
  state.store = await openStore('media-finder');
  try {
    state.models = await listEmbeddingModels();
  } catch (error) { say(`Cannot reach the heylook server: ${error.message}`); return; }
  if (!state.models.length) { say('No embedding models are registered on the server.'); return; }
  ui.model.replaceChildren(...state.models.map((m) => option(m.id)));
  ui.model.addEventListener('change', refreshModel);
  for (const el of [ui.dims, ui.docPrompt]) el.addEventListener('change', refreshCount);
  $('folder').addEventListener('change', (e) => { indexFiles([...e.target.files]); e.target.value = ''; });
  $('files').addEventListener('change', (e) => { indexFiles([...e.target.files]); e.target.value = ''; });
  ui.example.addEventListener('change', () => { ui.exampleName.textContent = ui.example.files[0]?.name ?? ''; });
  $('searchForm').addEventListener('submit', search);
  ui.stop.addEventListener('click', () => { state.abort?.abort(); cancel(state.requestId); say('Stopping…'); });
  ui.clear.addEventListener('click', async () => {
    if (!confirm('Delete every vector in this configuration?')) return;
    await state.store.clear(space());
    ui.results.replaceChildren(); ui.count.textContent = '';
    await refreshCount();
  });
  window.addEventListener('pagehide', () => { if (state.busy) cancel(state.requestId); });
  refreshModel();
}

init();
