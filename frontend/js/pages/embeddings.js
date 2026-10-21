// A local workbench: one ordered content list per vector, one shared space per run.
import { createPage } from '../page.js';
import { createEl as el } from '../utils.js';
import { api, requestId, httpError } from '../api.js';
import * as drawer from '../settings-drawer.js';

function button(text, action) {
  return el('button', { type: 'button', class: 'btn btn--sm', onclick: action }, [text]);
}
function field(label, control) {
  return el('label', { class: 'embeddings__field' }, [el('span', {}, [label]), control]);
}
function download(value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const a = el('a', { href: url, download: 'embeddings.json' });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function fileData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1]);
    reader.onerror = () => reject(new Error(`Cannot read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export default createPage({
  async setup(ctx) {
    ctx.onTeardown(drawer.registerSettings({ samplers: 'hidden' }));
    const s = ctx.state;
    s.items = [];
    s.models = [];
    s.runId = null;
    const model = el('select', { 'aria-label': 'Embedding model' });
    const dimensions = el('select', { 'aria-label': 'Embedding dimensions' });
    const status = el('p', { role: 'status', 'aria-live': 'polite' }, ['Loading embedding models…']);
    const note = el('p', { class: 'muted small' });
    const cards = el('div', { class: 'embeddings__items' });
    const results = el('section', { class: 'embeddings__results', 'aria-label': 'Embedding results' });
    const controls = el('fieldset', { class: 'embeddings__controls' });
    const run = button('Embed & compare', execute);
    const stop = button('Stop', async () => {
      if (!s.runId) return;
      const id = s.runId;
      try {
        const response = await fetch(`/v1/requests/${encodeURIComponent(id)}`, { method: 'DELETE' });
        if (!response.ok) throw await httpError(response);
        if (ctx.alive && s.runId === id) status.textContent = 'Cancellation requested. Waiting for the current operation to finish…';
      } catch (error) { if (ctx.alive) status.textContent = error.message; }
    });
    stop.hidden = true;
    const add = button('Add item', () => { addItem(); invalidate(); });
    controls.append(el('div', { class: 'embeddings__toolbar' }, [field('Model', model), field('Dimensions', dimensions)]),
      note, cards, el('div', { class: 'embeddings__toolbar' }, [add, run]));
    ctx.el.append(el('div', { class: 'embeddings' }, [
      el('h1', {}, ['Embeddings']),
      el('p', { class: 'muted' }, ['Compare meaning across text, images, audio and video. Each item becomes one vector; combine media within an item to describe it together.']),
      el('details', {}, [el('summary', {}, ['Ways to use this']), el('p', {}, [
        'Search: put a query first and candidates after it. Classification: compare an input with label descriptions. Similarity: compare paraphrases, code, or media. Select a model task prompt when available. Scores are cosine similarity, not probabilities. Video uses visual frames; add audio separately to include sound.',
      ])]), controls, stop, status, results,
    ]));
    const selected = () => s.models.find((m) => m.id === model.value);
    function invalidate() { results.replaceChildren(); }
    function prompts(select, previous = '') {
      select.replaceChildren(el('option', { value: '' }, ['No task prefix']),
        ...Object.keys(selected()?.engine.embedding.prompts ?? {}).map((name) => el('option', { value: name }, [name])));
      if ([...select.options].some((o) => o.value === previous)) select.value = previous;
    }
    function refreshModel() {
      const info = selected()?.engine.embedding;
      dimensions.replaceChildren(...(info?.dimensions ?? []).map((n) => el('option', { value: n }, [String(n)])));
      if (info) dimensions.value = info.native_dimensions;
      note.textContent = info ? `Inputs: ${info.modalities.join(', ')}. Each item must fit ${info.max_input_tokens.toLocaleString()} tokens including media. ${Object.keys(info.prompts).length ? 'Task prompts come from the checkpoint.' : 'This checkpoint has no named prompt catalog; task instructions can be entered as text.'}` : '';
      for (const item of s.items) prompts(item.prompt, item.prompt.value);
      run.disabled = !info;
      invalidate();
    }
    function addItem(text = '') {
      if (s.items.length >= 16) return;
      const item = { blocks: [] };
      item.name = el('input', { type: 'text', class: 'input', value: `Item ${s.items.length + 1}`, maxLength: 80, 'aria-label': 'Item label', oninput: invalidate });
      item.prompt = el('select', { 'aria-label': 'Task prompt', onchange: invalidate });
      prompts(item.prompt);
      const blocks = el('div', { class: 'embeddings__blocks' });
      const paintBlocks = () => {
        blocks.replaceChildren(...item.blocks.map((block, index) => {
          const content = block.file ? el('span', {}, [`${block.kind}: ${block.file.name}`])
            : el('textarea', { value: block.text, rows: 3, 'aria-label': 'Text content', placeholder: 'Text, code, query or label description',
              oninput: (event) => { block.text = event.target.value; invalidate(); } });
          return el('div', { class: 'embeddings__block' }, [content, el('div', { class: 'embeddings__toolbar' }, [
            button('Move up', () => { if (index) { [item.blocks[index-1], item.blocks[index]] = [block, item.blocks[index-1]]; paintBlocks(); invalidate(); } }),
            button('Remove block', () => { item.blocks.splice(index, 1); paintBlocks(); invalidate(); }),
          ])]);
        }));
      };
      item.blocks.push({ kind: 'text', text });
      const files = el('input', { type: 'file', class: 'input', multiple: true, accept: 'image/*,audio/*,video/*', 'aria-label': 'Add images, audio or video' });
      files.addEventListener('change', () => {
        for (const file of files.files) {
          const kind = file.type.split('/')[0];
          if (!['image', 'audio', 'video'].includes(kind)) { status.textContent = `Unsupported file type: ${file.name}`; continue; }
          if (file.size > 32 * 1024 * 1024) { status.textContent = `${file.name} exceeds the media size limit.`; continue; }
          if (item.blocks.length >= 32) { status.textContent = 'An item can contain at most 32 blocks.'; break; }
          item.blocks.push({ kind, file });
        }
        files.value = '';
        paintBlocks(); invalidate();
      });
      item.card = el('section', { class: 'embeddings__item' }, [
        el('div', { class: 'embeddings__toolbar' }, [field('Label', item.name), field('Task', item.prompt),
          button('Remove item', () => { s.items.splice(s.items.indexOf(item), 1); item.card.remove(); add.disabled = false; invalidate(); })]),
        blocks, el('div', { class: 'embeddings__toolbar' }, [button('Add text', () => {
          if (item.blocks.length < 32) { item.blocks.push({ kind: 'text', text: '' }); paintBlocks(); invalidate(); }
        }), field('Attach media', files)]),
      ]);
      s.items.push(item); cards.append(item.card); paintBlocks();
      add.disabled = s.items.length >= 16;
    }
    async function execute() {
      controls.disabled = true; invalidate();
      status.textContent = 'Preparing inputs…';
      try {
        if (!s.items.length) throw new Error('Add at least one item.');
        const info = selected().engine.embedding;
        let bytes = 0;
        const input = [];
        const labels = s.items.map((item, i) => item.name.value.trim() || `Item ${i + 1}`);
        for (const [index, item] of s.items.entries()) {
          const content = [];
          for (const block of item.blocks) {
            if (!info.modalities.includes(block.kind)) throw new Error(`This model does not support ${block.kind}.`);
            if (!block.file) { if (block.text.trim()) content.push({ type: 'text', text: block.text }); continue; }
            bytes += block.file.size;
            if (bytes > 64 * 1024 * 1024) throw new Error('Combined media exceeds the request size limit.');
            const data = await fileData(block.file);
            if (!ctx.alive) return;
            content.push({ type: block.kind, source: { type: 'base64', media_type: block.file.type, data } });
          }
          if (!content.length) throw new Error(`${labels[index]} is empty.`);
          input.push({ content, ...(item.prompt.value ? { prompt_name: item.prompt.value } : {}) });
        }
        if (!ctx.alive) return;
        const body = { model: model.value, dimensions: Number(dimensions.value), input };
        s.runId = requestId(); stop.hidden = false;
        status.textContent = 'Embedding… The model loads automatically if needed.';
        const response = await fetch('/v1/embeddings', { method: 'POST', signal: ctx.signal,
          headers: { 'Content-Type': 'application/json', 'X-Request-ID': s.runId }, body: JSON.stringify(body) });
        if (!response.ok) throw await httpError(response);
        const result = await response.json();
        if (!ctx.alive) return;
        const vectors = result.data.map((row) => row.embedding);
        const cosine = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
        const table = el('table', { class: 'embeddings__matrix' }, [
          el('caption', {}, ['Cosine similarity — higher means more similar']),
          el('thead', {}, [el('tr', {}, [el('th', { scope: 'col' }, ['Item']), ...labels.map((label) => el('th', { scope: 'col' }, [label]))])]),
          el('tbody', {}, vectors.map((vector, i) => el('tr', {}, [el('th', { scope: 'row' }, [labels[i]]),
            ...vectors.map((other) => el('td', {}, [cosine(vector, other).toFixed(3)]))]))),
        ]);
        const ranking = vectors.slice(1).map((v, i) => ({ label: labels[i+1], score: cosine(vectors[0], v) })).sort((a, b) => b.score - a.score);
        results.append(el('h2', {}, ['Results']), el('p', {}, [`${result.model} · ${result.dimensions} dimensions · ${result.data.length} vectors`]),
          ...(ranking.length ? [el('h3', {}, [`Closest to ${labels[0]}`]), el('ol', {}, ranking.map((r) => el('li', {}, [`${r.label}: ${r.score.toFixed(3)}`])))] : []),
          el('div', { class: 'embeddings__table-scroll', tabIndex: 0, 'aria-label': 'Similarity table' }, [table]),
          button('Download vectors', () => download({ labels, ...result })),
          el('details', {}, [el('summary', {}, ['Inspect vectors']), el('pre', {}, [JSON.stringify(result.data, null, 2)])]));
        status.textContent = `Complete${result.elapsed_ms != null ? ` in ${(result.elapsed_ms / 1000).toFixed(2)}s` : ''}. ${result.input_tokens == null ? 'Token usage unavailable.' : `${result.input_tokens} input tokens.`}`;
      } catch (error) { if (ctx.alive) status.textContent = error.message; }
      finally { s.runId = null; if (ctx.alive) { controls.disabled = false; stop.hidden = true; } }
    }
    model.addEventListener('change', refreshModel);
    dimensions.addEventListener('change', invalidate);
    ctx.onTeardown(() => { if (s.runId) fetch(`/v1/requests/${encodeURIComponent(s.runId)}`, { method: 'DELETE', keepalive: true }).catch(() => {}); });
    try {
      const data = await api.listModels();
      if (!ctx.alive) return;
      s.models = (data.data ?? []).filter((m) => m.engine?.embedding);
      model.replaceChildren(...s.models.map((m) => el('option', { value: m.id }, [m.id])));
      refreshModel();
      addItem('A dog playing on a beach.'); addItem('A puppy running beside the ocean.'); addItem('A database query joining two tables.');
      status.textContent = s.models.length ? 'Ready. Edit the examples or attach your own media.' : 'No embedding models found. Add a supported checkpoint to a scanned model folder and rescan on Models.';
    } catch (error) { if (ctx.alive) { status.textContent = error.message; run.disabled = true; } }
  },
});
