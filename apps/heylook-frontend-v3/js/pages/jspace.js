// J-Space: read a model's verbalizable "workspace" (Jacobian lens). For a
// prompt, the model greedily answers and we show, per network-depth band layer,
// the top-k "silent" tokens it was disposed toward before answering -- plus an
// optional layer x position heatmap and a hallucination-risk score.
//
// Non-streaming: one POST /v1/jspace/analyze, then a full render. Only models
// with a fitted+converted lens appear in the picker (GET /v1/jspace/models).

import { createPage } from '../page.js';
import { createEl, autoGrow, setStatus, fillOptions } from '../utils.js';
import { api } from '../api.js';

// Probability/strength in [0,1] -> OKLCH fill (25=red .. 145=green), matching
// the explore-page chip formula.
function strengthColor(t) {
  const hue = 25 + Math.max(0, Math.min(1, t)) * 120;
  return `oklch(0.86 0.11 ${hue})`;
}

function glyph(token) {
  if (token === '') return '∅';
  return token.replace(/\n/g, '⏎').replace(/\t/g, '⇥').replace(/ /g, '·');
}

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.models = [];
    s.busy = false;

    buildSkeleton(ctx);

    const res = await api.jspaceModels({ signal: ctx.signal }).catch(() => ({ models: [] }));
    if (!ctx.alive) return;
    s.models = res.models ?? [];
    if (s.models.length) {
      fillOptions(s.modelSelect, s.models);
      setStatus(s.statusEl, `${s.models.length} model(s) with a lens. Enter a prompt and analyze.`);
    } else {
      s.modelSelect.disabled = true;
      s.analyzeBtn.disabled = true;
      s.resultEl.replaceChildren(createEl('div', { class: 'empty-state' }, [
        'No j-space lenses installed. Convert a lens offline and place it at ',
        createEl('code', {}, ['HEYLOOK_JSPACE_DIR/<model_id>/lens.safetensors']),
        '.',
      ]));
    }
  },
});

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.modelSelect = createEl('select', { title: 'Model' });

  s.textarea = createEl('textarea', { rows: 2, placeholder: 'A prompt to probe, e.g. "The Eiffel Tower is in the city of"…' });
  s.textarea.addEventListener('input', () => autoGrow(s.textarea));
  s.textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); analyze(ctx); }
  });

  s.heatmapToggle = createEl('input', { type: 'checkbox', id: 'jspace-heatmap' });
  const heatmapLabel = createEl('label', { class: 'jspace__opt', for: 'jspace-heatmap' },
    [s.heatmapToggle, ' layer×token heatmap']);

  // Raw completion (default) reads the workspace at a content token -> crisp
  // "silent words". Chat mode formats an instruct turn (needed for risk).
  s.chatToggle = createEl('input', { type: 'checkbox', id: 'jspace-chat' });
  const chatLabel = createEl('label', { class: 'jspace__opt', for: 'jspace-chat' },
    [s.chatToggle, ' chat mode']);

  s.analyzeBtn = createEl('button', { class: 'btn btn--primary' }, ['Analyze']);
  s.analyzeBtn.addEventListener('click', () => analyze(ctx));

  s.statusEl = createEl('div', { class: 'jspace__status small' });
  s.resultEl = createEl('div', { class: 'jspace__result' });

  s.rootEl = createEl('div', { class: 'jspace' }, [
    createEl('header', { class: 'jspace__bar' }, [s.modelSelect, heatmapLabel, chatLabel]),
    createEl('div', { class: 'jspace__composer' }, [s.textarea, s.analyzeBtn]),
    s.statusEl,
    s.resultEl,
  ]);
  ctx.el.append(s.rootEl);
}

async function analyze(ctx) {
  const s = ctx.state;
  if (s.busy) return;
  const prompt = s.textarea.value.trim();
  const model = s.modelSelect.value;
  if (!prompt || !model) { setStatus(s.statusEl, 'Pick a model and enter a prompt.', true); return; }

  s.busy = true;
  s.analyzeBtn.disabled = true;
  setStatus(s.statusEl, 'Analyzing… (generating + reading the workspace)');
  try {
    const data = await api.jspaceAnalyze(
      { model, prompt, heatmap: s.heatmapToggle.checked, chat: s.chatToggle.checked,
        max_answer_tokens: 8, top_k: 8 },
      { signal: ctx.signal });
    if (!ctx.alive) return;
    renderResult(ctx, data);
    setStatus(s.statusEl, `Done — ${data.band_layers?.length ?? 0} band layers read.`);
  } catch (err) {
    if (!ctx.alive) return;
    setStatus(s.statusEl, `Analyze failed: ${err.message}`, true);
  } finally {
    if (ctx.alive) { s.busy = false; s.analyzeBtn.disabled = false; }
  }
}

function renderResult(ctx, data) {
  const s = ctx.state;
  const parts = [];

  // Answer + optional risk badge.
  const head = createEl('div', { class: 'jspace__answer' }, [
    createEl('span', { class: 'jspace__answer-label' }, ['answer ']),
    createEl('span', { class: 'jspace__answer-text' }, [data.answer || '(empty)']),
  ]);
  if (typeof data.risk === 'number') {
    const pct = Math.round(data.risk * 100);
    head.append(createEl('span', {
      class: 'jspace__risk',
      style: `background:${strengthColor(1 - data.risk)}`,
      title: 'Hallucination-risk router: P(answer wrong)',
    }, [`risk ${pct}%`]));
  }
  parts.push(head);

  // Answer-onset workspace strip: one row per band layer (deep -> shallow),
  // each the top-k silent tokens colored by within-layer rank.
  const strip = createEl('div', { class: 'jspace__strip' });
  const rows = [...(data.onset_strip ?? [])].reverse();
  for (const row of rows) {
    const k = row.top_k.length;
    const chips = row.top_k.map((c, i) => createEl('span', {
      class: 'jspace__chip',
      style: `background:${strengthColor(k > 1 ? 1 - i / (k - 1) : 1)}`,
      title: `logit ${c.logit.toFixed(2)}`,
    }, [glyph(c.token)]));
    strip.append(createEl('div', { class: 'jspace__row' }, [
      createEl('span', { class: 'jspace__layer' }, [`L${row.layer}`]),
      createEl('span', { class: 'jspace__ent', title: 'lens entropy (nats)' }, [row.entropy.toFixed(1)]),
      createEl('span', { class: 'jspace__chips' }, chips),
    ]));
  }
  parts.push(createEl('div', { class: 'jspace__section-label small muted' }, ['workspace at answer-onset (deep → shallow)']));
  parts.push(strip);

  // Optional layer x position heatmap (top-1 token per cell, colored by entropy).
  if (Array.isArray(data.heatmap) && data.heatmap.length) {
    parts.push(buildHeatmap(data));
  }

  s.resultEl.replaceChildren(...parts);
}

function buildHeatmap(data) {
  const flat = data.heatmap.flatMap((r) => r.cells.map((c) => c.entropy));
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const norm = (e) => (hi > lo ? 1 - (e - lo) / (hi - lo) : 1); // low entropy = confident = green

  const grid = createEl('div', { class: 'jspace__heatmap' });
  for (const rowData of [...data.heatmap].reverse()) {
    const cells = rowData.cells.map((c) => createEl('span', {
      class: 'jspace__hcell',
      style: `background:${strengthColor(norm(c.entropy))}`,
      title: `${glyph(c.token)}  ·  entropy ${c.entropy.toFixed(2)}`,
    }, [glyph(c.token).slice(0, 4)]));
    grid.append(createEl('div', { class: 'jspace__hrow' }, [
      createEl('span', { class: 'jspace__layer' }, [`L${rowData.layer}`]),
      ...cells,
    ]));
  }
  return createEl('div', {}, [
    createEl('div', { class: 'jspace__section-label small muted' }, ['layer × position (top-1 token, color = confidence)']),
    grid,
  ]);
}
