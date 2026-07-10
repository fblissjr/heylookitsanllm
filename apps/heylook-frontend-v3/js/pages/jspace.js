// J-Space: read a model's verbalizable "workspace" (Jacobian lens). For a
// prompt, the model greedily answers and we show, per network-depth band layer,
// the top-k "silent" tokens it was disposed toward before answering -- plus an
// optional layer x position heatmap and a hallucination-risk score.
//
// One POST /v1/jspace/analyze, then a full render. Only models with a
// fitted+converted lens appear in the picker (GET /v1/jspace/models).
//
// Interaction (DESIGN.md §3-4): click a workspace row or heatmap cell to PIN
// its readout in the detail panel; Esc unpins; arrows walk layers/positions.
// The answer-onset column (the strip; the heatmap's last column) is the only
// place the API returns full top-k today -- other cells pin a reduced
// top-1+entropy readout until the per-cell top-N analyze extension lands.

import { createPage } from '../page.js';
import { createEl, autoGrow, setStatus, fillOptions } from '../utils.js';
import { api } from '../api.js';

// Probability/strength in [0,1] -> OKLCH fill (25=red .. 145=green), matching
// the explore-page chip formula (DESIGN.md §2: fixed L/C, hue carries data).
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
    s.data = null;
    s.pinned = null; // { layer, posIdx } -- posIdx null = answer onset (no heatmap)

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

  s.rootEl = createEl('div', { class: 'jspace', tabindex: '0' }, [
    createEl('header', { class: 'jspace__bar' }, [s.modelSelect, heatmapLabel, chatLabel]),
    createEl('div', { class: 'jspace__composer' }, [s.textarea, s.analyzeBtn]),
    s.statusEl,
    s.resultEl,
  ]);
  s.rootEl.addEventListener('keydown', (e) => handleKeydown(ctx, e));
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
    setStatus(s.statusEl, `Done — ${data.band_layers?.length ?? 0} band layers read. Click a row or cell to pin its readout.`);
  } catch (err) {
    if (!ctx.alive) return;
    setStatus(s.statusEl, `Analyze failed: ${err.message}`, true);
  } finally {
    if (ctx.alive) { s.busy = false; s.analyzeBtn.disabled = false; }
  }
}

// ---------------------------------------------------------------------------
// pin state (single selection; strip rows and heatmap cells share it)
// ---------------------------------------------------------------------------

// The strip reads positions=[-1]; the heatmap covers the LAST N prompt
// positions -- so the heatmap's last column IS the answer onset.
function onsetPosIdx(data) {
  return data.heatmap_positions ? data.heatmap_positions.length - 1 : null;
}

function isOnsetPin(data, pin) {
  return pin.posIdx == null || pin.posIdx === onsetPosIdx(data);
}

function togglePin(ctx, pin) {
  const s = ctx.state;
  const same = s.pinned && s.pinned.layer === pin.layer && s.pinned.posIdx === pin.posIdx;
  setPin(ctx, same ? null : pin);
  s.rootEl.focus({ preventScroll: true });
}

// The token a pinned cell "wants" (its top-1) -- drives the echo highlight.
function pinnedTopToken(data, pin) {
  if (isOnsetPin(data, pin)) {
    return data.onset_strip.find((r) => r.layer === pin.layer)?.top_k[0]?.token ?? null;
  }
  return data.heatmap.find((r) => r.layer === pin.layer)?.cells[pin.posIdx]?.token ?? null;
}

function setPin(ctx, pin) {
  const s = ctx.state;
  s.pinned = pin;
  const d = s.data;
  const onset = pin && isOnsetPin(d, pin);
  const echoTok = pin ? pinnedTopToken(d, pin) : null;

  for (const [layer, row] of s.rowEls) {
    row.classList.toggle('jspace__row--pinned', !!pin && onset && layer === pin.layer);
  }
  for (const [key, cell] of s.cellEls) {
    const [layer, posIdx] = key.split(':').map(Number);
    const isPinned = !!pin && layer === pin.layer &&
      (pin.posIdx == null ? posIdx === onsetPosIdx(d) : posIdx === pin.posIdx);
    cell.classList.toggle('jspace__hcell--pinned', isPinned);
    cell.classList.toggle('jspace__hcell--echo',
      !!echoTok && !isPinned && cell.dataset.tok === echoTok);
    if (isPinned) cell.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  renderDetail(ctx);
}

function handleKeydown(ctx, e) {
  const s = ctx.state;
  if (e.target.closest('textarea, input, select')) return;
  if (!s.data) return;
  if (e.key === 'Escape') { setPin(ctx, null); return; }
  if (!s.pinned) return;

  const band = s.data.band_layers ?? [];
  if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
    // Rows render deep -> shallow (descending layer), so Up = deeper.
    e.preventDefault();
    const li = band.indexOf(s.pinned.layer) + (e.key === 'ArrowUp' ? 1 : -1);
    if (li >= 0 && li < band.length) setPin(ctx, { layer: band[li], posIdx: s.pinned.posIdx });
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    const count = s.data.heatmap_positions?.length ?? 0;
    if (!count) return;
    e.preventDefault();
    const cur = s.pinned.posIdx ?? onsetPosIdx(s.data);
    const np = cur + (e.key === 'ArrowRight' ? 1 : -1);
    if (np >= 0 && np < count) setPin(ctx, { layer: s.pinned.layer, posIdx: np });
  }
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

function renderResult(ctx, data) {
  const s = ctx.state;
  s.data = data;
  s.pinned = null;
  s.rowEls = new Map();   // layer -> strip row element
  s.cellEls = new Map();  // "layer:posIdx" -> heatmap cell element

  // Answer + optional risk badge.
  const head = createEl('div', { class: 'jspace__answer' }, [
    createEl('span', { class: 'jspace__answer-label' }, ['answer ']),
    createEl('span', { class: 'jspace__answer-text' }, [data.answer || '(empty)']),
  ]);
  if (Number.isFinite(data.risk)) {
    const pct = Math.round(data.risk * 100);
    head.append(createEl('span', {
      class: 'jspace__risk',
      style: `background:${strengthColor(1 - data.risk)}`,
      title: 'Hallucination-risk router: P(answer wrong)',
    }, [`risk ${pct}%`]));
  }

  // Answer-onset workspace strip: one row per band layer (deep -> shallow),
  // each the top-k silent tokens colored by within-layer rank. Rows pin the
  // layer's full top-k readout.
  const strip = createEl('div', { class: 'jspace__strip' });
  for (const row of [...(data.onset_strip ?? [])].reverse()) {
    const k = row.top_k.length;
    const chips = row.top_k.map((c, i) => createEl('span', {
      class: 'jspace__chip',
      style: `background:${strengthColor(k > 1 ? 1 - i / (k - 1) : 1)}`,
      title: `logit ${c.logit.toFixed(2)}`,
    }, [glyph(c.token)]));
    const rowEl = createEl('div', { class: 'jspace__row jspace__row--selectable' }, [
      createEl('span', { class: 'jspace__layer' }, [`L${row.layer}`]),
      createEl('span', { class: 'jspace__ent', title: 'lens entropy (nats)' }, [row.entropy.toFixed(1)]),
      createEl('span', { class: 'jspace__chips' }, chips),
    ]);
    rowEl.addEventListener('click', () =>
      togglePin(ctx, { layer: row.layer, posIdx: onsetPosIdx(data) }));
    s.rowEls.set(row.layer, rowEl);
    strip.append(rowEl);
  }

  const main = createEl('div', { class: 'jspace__main' }, [
    createEl('div', { class: 'jspace__section-label small muted' }, ['workspace at answer-onset (deep → shallow)']),
    strip,
  ]);
  if (Array.isArray(data.heatmap) && data.heatmap.length) {
    main.append(buildHeatmap(ctx, data));
  }

  s.detailEl = createEl('aside', { class: 'jspace__detail' });
  s.resultEl.replaceChildren(head,
    createEl('div', { class: 'jspace__layout' }, [main, s.detailEl]));
  renderDetail(ctx);
}

function buildHeatmap(ctx, data) {
  const s = ctx.state;
  const flat = data.heatmap.flatMap((r) => r.cells.map((c) => c.entropy));
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const norm = (e) => (hi > lo ? 1 - (e - lo) / (hi - lo) : 1); // low entropy = confident = green

  const grid = createEl('div', { class: 'jspace__heatmap' });

  // Header row: the prompt token at each position; the last column is the
  // answer onset (where the strip reads).
  const onsetIdx = onsetPosIdx(data);
  const headCells = data.heatmap_positions.map((pos, i) => {
    const onset = i === onsetIdx;
    return createEl('span', {
      class: `jspace__hpos${onset ? ' jspace__hpos--onset' : ''}`,
      title: onset ? `position ${pos} — answer onset` : `position ${pos}`,
    }, [glyph(data.prompt_tokens?.[pos] ?? '').slice(0, 4)]);
  });
  grid.append(createEl('div', { class: 'jspace__hrow' }, [
    createEl('span', { class: 'jspace__layer' }, ['']),
    ...headCells,
  ]));

  for (const rowData of [...data.heatmap].reverse()) {
    const cells = rowData.cells.map((c, i) => {
      const cell = createEl('span', {
        class: 'jspace__hcell jspace__hcell--selectable',
        style: `background:${strengthColor(norm(c.entropy))}`,
        title: `${glyph(c.token)}  ·  entropy ${c.entropy.toFixed(2)}`,
        dataset: { tok: c.token },
      }, [glyph(c.token).slice(0, 4)]);
      cell.addEventListener('click', () => togglePin(ctx, { layer: rowData.layer, posIdx: i }));
      s.cellEls.set(`${rowData.layer}:${i}`, cell);
      return cell;
    });
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

// ---------------------------------------------------------------------------
// pinned detail panel
// ---------------------------------------------------------------------------

function renderDetail(ctx) {
  ctx.state.detailEl.replaceChildren(buildDetailPanel(ctx));
}

function detailRow(label, valueEl) {
  return createEl('div', { class: 'jspace-detail__row' }, [
    createEl('span', { class: 'muted small' }, [label]),
    valueEl,
  ]);
}

// Ranked logit bars, min-max normalized within the list (bars encode rank by
// length; hue stays out of it -- DESIGN.md §2). The row matching the model's
// actual first answer token gets the "chosen" emphasis.
function buildBars(topK, firstAnswerToken) {
  const logits = topK.map((c) => c.logit);
  const lo = Math.min(...logits), hi = Math.max(...logits);
  const width = (l) => (hi > lo ? 4 + 96 * (l - lo) / (hi - lo) : 100);
  return topK.map((c) => {
    const isAnswer = c.token === firstAnswerToken;
    return createEl('div', {
      class: `jspace-bar${isAnswer ? ' jspace-bar--answer' : ''}`,
      title: isAnswer ? 'matches the first answer token' : undefined,
    }, [
      createEl('span', { class: 'jspace-bar__label' }, [glyph(c.token)]),
      createEl('div', { class: 'jspace-bar__track' }, [
        createEl('div', { class: 'jspace-bar__fill', style: `width: ${width(c.logit).toFixed(1)}%` }),
      ]),
      createEl('span', { class: 'jspace-bar__logit small muted' }, [c.logit.toFixed(2)]),
    ]);
  });
}

function buildDetailPanel(ctx) {
  const s = ctx.state;
  const d = s.data;
  const pin = s.pinned;
  if (!pin) {
    return createEl('div', { class: 'empty-state' }, [
      'Click a workspace row or a heatmap cell to pin its silent-token readout. Esc unpins; arrow keys walk layers and positions.',
    ]);
  }

  const onset = isOnsetPin(d, pin);
  const posIdx = pin.posIdx ?? onsetPosIdx(d);
  const absPos = posIdx != null ? d.heatmap_positions?.[posIdx] : null;
  const posLabel = onset
    ? `answer onset${absPos != null ? ` (${absPos})` : ''}`
    : `${absPos} "${glyph(d.prompt_tokens?.[absPos] ?? '')}"`;

  const parts = [
    detailRow('Layer', createEl('span', { class: 'jspace-detail__mono' }, [`L${pin.layer}`])),
    detailRow('Position', createEl('span', { class: 'jspace-detail__mono' }, [posLabel])),
  ];

  if (onset) {
    const row = d.onset_strip.find((r) => r.layer === pin.layer);
    parts.push(detailRow('Entropy', createEl('span', { class: 'jspace-detail__mono' },
      [`${row.entropy.toFixed(2)} nats`])));
    parts.push(createEl('h3', { class: 'jspace-detail__heading' }, [`Silent tokens (top ${row.top_k.length})`]));
    parts.push(createEl('div', { class: 'jspace-bars' }, buildBars(row.top_k, d.first_answer_token)));
  } else {
    const cell = d.heatmap.find((r) => r.layer === pin.layer)?.cells[posIdx];
    parts.push(detailRow('Entropy', createEl('span', { class: 'jspace-detail__mono' },
      [`${cell.entropy.toFixed(2)} nats`])));
    parts.push(createEl('h3', { class: 'jspace-detail__heading' }, ['Top token']));
    parts.push(createEl('div', { class: 'jspace-detail__mono' }, [`"${glyph(cell.token)}"`]));
    parts.push(createEl('div', { class: 'muted small jspace-detail__note' }, [
      'The API returns top-1 + entropy for non-onset cells today; the full per-cell top-N readout lands with the analyze extension.',
    ]));
  }

  return createEl('div', { class: 'jspace-detail__content' }, parts);
}
