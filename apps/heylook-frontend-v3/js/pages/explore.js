// Token Explorer: stream a completion with per-token logprobs and let the
// user inspect what the model considered at each step.
//
// Invariants:
// - state.tokens holds {token, logprob, prob, top} -- built fresh from each
//   onLogprobs batch, never mutated in place once appended.
// - The token strip is a single throttled FULL re-render (ctx.throttle), not
//   incremental DOM patching -- v2's dual incremental/full path is gone.
// - Keyboard nav (ArrowLeft/ArrowRight/Escape) is bound to the page
//   container, not document -- it dies with the DOM, no leaked listener.

import { createPage } from '../page.js';
import { createEl, autoGrow, setStatus, fillOptions } from '../utils.js';
import { api } from '../api.js';
import { streamChat } from '../streaming.js';
import { samplerParams } from '../settings.js';
import * as drawer from '../settings-drawer.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.models = [];
    s.tokens = [];
    s.selectedIndex = null;
    s.stream = null; // { controller }

    buildSkeleton(ctx);
    s.paintStrip = ctx.throttle(() => { if (ctx.alive) renderStrip(ctx); });

    // Explore consumes samplerParams(); logprobs are hard-wired on for this
    // view, so the drawer just states that honestly as an extra (no control).
    const unregisterSettings = drawer.registerSettings({
      caps: () => exploreCaps(ctx),
      samplers: 'enabled',
      extras: () => [createEl('div', { class: 'settings-note muted small' },
        ['Logprobs on: top-5 alternatives per token (fixed for this view).'])],
    });
    ctx.onTeardown(unregisterSettings);

    renderStrip(ctx);
    renderDetail(ctx);

    const models = await api.listModels({ signal: ctx.signal }).catch(() => ({ data: [] }));
    if (!ctx.alive) return;
    s.models = models.data ?? [];
    fillModelSelect(ctx);
  },
});

// ---------------------------------------------------------------------------
// skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.modelSelect = createEl('select', { title: 'Model' });
  // capability-gated sampler controls (enable_thinking) track the model
  s.modelSelect.addEventListener('change', () => drawer.requestRebuild({ force: true }));

  s.textarea = createEl('textarea', { rows: 2, placeholder: 'Write a prompt…' });
  s.textarea.addEventListener('input', () => autoGrow(s.textarea));
  s.textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!s.stream) generate(ctx);
    }
  });

  s.genBtn = createEl('button', { class: 'btn btn--primary' }, ['Generate']);
  s.genBtn.addEventListener('click', () => (s.stream ? stopStream(ctx) : generate(ctx)));

  s.statusEl = createEl('div', { class: 'explore__status small', role: 'status' });

  s.thinkingBody = createEl('div', { class: 'thinking__body' });
  s.thinkingEl = createEl('details', { class: 'thinking explore__thinking', hidden: true }, [
    createEl('summary', {}, ['Thinking']),
    s.thinkingBody,
  ]);

  s.stripEl = createEl('div', { class: 'explore__strip' });
  s.detailEl = createEl('aside', { class: 'explore__detail' });

  s.rootEl = createEl('div', { class: 'explore', tabindex: '0' }, [
    createEl('header', { class: 'explore__bar' }, [s.modelSelect]),
    createEl('div', { class: 'explore__composer' }, [s.textarea, s.genBtn]),
    s.statusEl,
    createEl('div', { class: 'explore__body' }, [
      createEl('div', { class: 'explore__main' }, [s.thinkingEl, s.stripEl]),
      s.detailEl,
    ]),
  ]);
  s.rootEl.addEventListener('keydown', (e) => handleKeydown(ctx, e));

  ctx.el.append(s.rootEl);
}

function fillModelSelect(ctx) {
  const s = ctx.state;
  fillOptions(s.modelSelect, s.models.map((m) => m.id));
}

function exploreCaps(ctx) {
  const model = ctx.state.models.find((m) => m.id === ctx.state.modelSelect.value);
  return model?.capabilities ?? [];
}

function showStatus(ctx, text, isError = false) {
  setStatus(ctx.state.statusEl, text, isError);
}

// ---------------------------------------------------------------------------
// keyboard + selection
// ---------------------------------------------------------------------------

function handleKeydown(ctx, e) {
  const s = ctx.state;
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    if (!s.tokens.length) return;
    selectToken(ctx, s.selectedIndex == null ? s.tokens.length - 1 : Math.max(0, s.selectedIndex - 1));
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    if (!s.tokens.length) return;
    selectToken(ctx, s.selectedIndex == null ? 0 : Math.min(s.tokens.length - 1, s.selectedIndex + 1));
  } else if (e.key === 'Escape') {
    selectToken(ctx, null);
  }
}

function selectToken(ctx, idx) {
  ctx.state.selectedIndex = idx;
  renderStrip(ctx);
  renderDetail(ctx);
}

// ---------------------------------------------------------------------------
// token strip
// ---------------------------------------------------------------------------

// Renders whitespace as visible muted glyphs; '\n' also emits a real <br>
// so the strip actually wraps at that point (chips are inline-block, not
// flex, specifically so an internal <br> is honored).
function appendGlyphs(el, text) {
  for (const ch of text) {
    if (ch === ' ') el.append(createEl('span', { class: 'ws-glyph' }, ['·']));
    else if (ch === '\t') el.append(createEl('span', { class: 'ws-glyph' }, ['→']));
    else if (ch === '\n') {
      el.append(createEl('span', { class: 'ws-glyph' }, ['↵']));
      el.append(createEl('br'));
    } else {
      el.append(document.createTextNode(ch));
    }
  }
}

function buildChip(ctx, tok, idx) {
  const hue = 25 + tok.prob * 120;
  const selected = ctx.state.selectedIndex === idx;
  // Probability is titled, not color-only (DESIGN.md §2): the chip hue encodes
  // it, but a tooltip + a11y readout must carry it too.
  const chip = createEl('span', {
    class: `tok${selected ? ' tok--selected' : ''}`,
    title: `p ${(tok.prob * 100).toFixed(1)}%`,
  });
  chip.style.background = `oklch(0.86 0.11 ${hue})`;
  appendGlyphs(chip, tok.token);
  chip.addEventListener('click', () => {
    selectToken(ctx, idx);
    ctx.state.rootEl.focus({ preventScroll: true });
  });
  return chip;
}

function renderStrip(ctx) {
  const s = ctx.state;
  if (!s.tokens.length) {
    s.stripEl.replaceChildren(
      createEl('div', { class: 'empty-state' }, [
        'Pick a model, write a prompt — every generated token shows its probability; click one to see what else the model considered.',
      ]),
    );
    return;
  }
  s.stripEl.replaceChildren(...s.tokens.map((tok, i) => buildChip(ctx, tok, i)));
}

// ---------------------------------------------------------------------------
// detail panel
// ---------------------------------------------------------------------------

function buildBarRow(tok, alt) {
  const pct = Math.max(0, Math.min(100, Math.exp(alt.logprob) * 100));
  const label = createEl('span', { class: 'explore-bar__label' });
  appendGlyphs(label, alt.token);
  const chosen = alt.token === tok.token;
  return createEl('div', { class: `explore-bar${chosen ? ' explore-bar--chosen' : ''}` }, [
    label,
    createEl('div', { class: 'explore-bar__track' }, [
      createEl('div', { class: 'explore-bar__fill', style: `width: ${pct.toFixed(2)}%` }),
    ]),
    createEl('span', { class: 'explore-bar__pct small muted' }, [`${pct.toFixed(1)}%`]),
  ]);
}

function buildDetailPanel(ctx) {
  const s = ctx.state;
  const idx = s.selectedIndex;
  const tok = idx == null ? null : s.tokens[idx];
  if (!tok) {
    return createEl('div', { class: 'empty-state' }, [
      'Click a token (or select one with the arrow keys) to inspect its probability and alternatives.',
    ]);
  }

  const tokenLabel = createEl('span', { class: 'explore-detail__token' }, ['"']);
  appendGlyphs(tokenLabel, tok.token);
  tokenLabel.append('"');

  const row = (label, valueEl) => createEl('div', { class: 'explore-detail__row' }, [
    createEl('span', { class: 'muted small' }, [label]),
    valueEl,
  ]);

  const bars = (tok.top ?? []).slice(0, 5).map((alt) => buildBarRow(tok, alt));

  return createEl('div', { class: 'explore-detail__content' }, [
    row('Token', tokenLabel),
    row('Logprob', createEl('span', { class: 'explore-detail__mono' }, [tok.logprob.toFixed(3)])),
    row('Probability', createEl('span', {}, [`${(tok.prob * 100).toFixed(2)}%`])),
    row('Position', createEl('span', {}, [String(idx)])),
    createEl('h3', { class: 'explore-detail__heading' }, ['Top alternatives']),
    createEl('div', { class: 'explore-bars' },
      bars.length ? bars : [createEl('div', { class: 'muted small' }, ['No alternatives recorded.'])]),
  ]);
}

function renderDetail(ctx) {
  ctx.state.detailEl.replaceChildren(buildDetailPanel(ctx));
}

// ---------------------------------------------------------------------------
// generate + stream
// ---------------------------------------------------------------------------

function buildRequestBody(ctx) {
  const s = ctx.state;
  return {
    model: s.modelSelect.value,
    messages: [{ role: 'user', content: s.textarea.value.trim() }],
    logprobs: true,
    top_logprobs: 5,
    ...samplerParams(exploreCaps(ctx)),
  };
}

function generate(ctx) {
  const s = ctx.state;
  const prompt = s.textarea.value.trim();
  if (!prompt || s.stream) return;
  if (!s.modelSelect.value) {
    showStatus(ctx, 'No models available.', true);
    return;
  }

  s.tokens = [];
  s.selectedIndex = null;
  s.thinkingEl.hidden = true;
  s.thinkingBody.textContent = '';
  renderStrip(ctx);
  renderDetail(ctx);
  showStatus(ctx, '');

  const controller = ctx.linkedController();
  const stream = { controller };
  s.stream = stream;
  s.genBtn.textContent = 'Stop';
  s.rootEl.focus({ preventScroll: true });

  const isCurrent = () => s.stream === stream;

  streamChat(buildRequestBody(ctx), {
    signal: controller.signal,
    onThinking: (_, full) => {
      if (!ctx.alive || !isCurrent()) return;
      s.thinkingEl.hidden = false;
      s.thinkingBody.textContent = full;
    },
    onLogprobs: (items) => {
      if (!ctx.alive || !isCurrent()) return;
      for (const item of items) {
        s.tokens.push({
          token: item.token,
          logprob: item.logprob,
          prob: Math.exp(item.logprob),
          top: item.top_logprobs ?? [],
        });
      }
      s.paintStrip();
    },
    onRetryWait: (wait) => {
      if (ctx.alive && isCurrent()) showStatus(ctx, `Server busy -- retrying in ${wait}s…`);
    },
    onComplete: (result) => finishStream(ctx, stream, result),
    onError: (err) => handleStreamError(ctx, stream, err),
  });
}

function stopStream(ctx) {
  ctx.state.stream?.controller.abort();
}

function releaseStream(ctx, stream) {
  const s = ctx.state;
  // No-op after normal completion; drops the linkedController chain listener.
  stream.controller.abort();
  if (s.stream !== stream) return;
  s.stream = null;
  if (ctx.alive) s.genBtn.textContent = 'Generate';
}

function finishStream(ctx, stream, { aborted }) {
  releaseStream(ctx, stream);
  if (!ctx.alive) return;
  if (aborted) showStatus(ctx, ctx.state.tokens.length ? 'Stopped.' : 'Stopped -- no tokens generated.');
}

function handleStreamError(ctx, stream, err) {
  releaseStream(ctx, stream);
  if (!ctx.alive) return;
  showStatus(ctx, `Generation failed: ${err.message}`, true);
}
