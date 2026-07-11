// Notebook: freeform writing surface with generate-at-cursor completion.
// Plain text only -- content lives in textarea.value, never innerHTML/markdown.
//
// Invariants:
// - Auto-save is debounced (utils debounce): event handlers update state
//   FIRST, then schedule a save that reads back FROM state -- never from the
//   DOM at save time. `dirty` tracks whether there's anything worth saving.
// - Generation is keyed to the notebook it started in (gen.targetId): if the
//   user switches notebooks mid-generation the stream is stopped, but the
//   partial text still lands in the notebook it belonged to.
// - Stop is normal completion: partial text is kept, not discarded.

import { createPage } from '../page.js';
import { createEl, autoGrow, armedConfirm, debounce, setStatus, fillOptions, dismissPaneOnOutsideClick } from '../utils.js';
import { api } from '../api.js';
import { streamChat } from '../streaming.js';
import { samplerParams, snapshotSettings, bindDocumentParams, hydrateDocParams } from '../settings.js';
import * as drawer from '../settings-drawer.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.notebooks = [];
    s.models = [];
    s.activeId = null;
    s.title = '';
    s.content = '';
    s.systemPrompt = '';
    s.modelId = '';
    s.dirty = false;
    s.gen = null; // { controller, targetId, head, tail, content }

    buildSkeleton(ctx);
    s.scheduleSave = debounce(() => { doSave(ctx); }, 500);
    ctx.onTeardown(() => s.scheduleSave.flush());

    // Notebook consumes samplerParams() for generate-at-cursor, so it gets full
    // sampler controls; its per-notebook system-prompt editor leads the panel.
    const unregisterSettings = drawer.registerSettings({
      caps: () => notebookCaps(ctx),
      samplers: 'enabled',
      sections: () => [s.sysPromptDetails],
    });
    ctx.onTeardown(unregisterSettings);
    // Per-NOTEBOOK sampler settings via the SAME shared binding chat uses --
    // one mechanism, no branched copy. Panel change -> debounced PUT to the
    // active notebook's `params`; hydrate on select is silent.
    ctx.onTeardown(bindDocumentParams({
      activeId: () => ctx.state.activeId,
      updateDoc: (id, body) => api.updateNotebook(id, body),
      onError: (err) => showStatus(ctx, `Settings save failed: ${err.message}`, true),
    }));
    // One throttle for the whole mount (reads s.gen) -- a per-generation
    // throttle would pin each generation's head/tail copies until unmount.
    s.paint = ctx.throttle(() => paintGen(ctx));

    const [models, list] = await Promise.all([
      api.listModels({ signal: ctx.signal }).catch(() => ({ data: [] })),
      api.listNotebooks({ signal: ctx.signal }).catch((err) => {
        if (ctx.alive) showStatus(ctx, `Could not load notebooks: ${err.message}`, true);
        return { notebooks: [] };
      }),
    ]);
    if (!ctx.alive) return;

    s.models = models.data ?? [];
    s.notebooks = list.notebooks ?? [];
    fillModelSelect(ctx);
    renderList(ctx);

    if (s.notebooks.length) {
      await selectNotebook(ctx, s.notebooks[0].id);
    } else {
      renderEditor(ctx);
    }
  },
});

// ---------------------------------------------------------------------------
// skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.listEl = createEl('div', { class: 'notebook__list' });
  const newBtn = createEl('button', { class: 'btn btn--sm' }, ['New']);
  newBtn.addEventListener('click', () => newNotebook(ctx));

  const listPane = createEl('aside', { class: 'notebook__list-pane' }, [
    createEl('div', { class: 'notebook__list-head' }, [
      createEl('h2', {}, ['Notebooks']),
      newBtn,
    ]),
    s.listEl,
  ]);

  s.listToggleBtn = createEl('button', { class: 'btn btn--sm notebook__list-toggle' }, ['Notebooks']);
  s.listToggleBtn.addEventListener('click', () => s.rootEl.classList.toggle('notebook--list-open'));
  const toolbar = createEl('header', { class: 'notebook__toolbar' }, [s.listToggleBtn]);

  s.titleInput = createEl('input', {
    class: 'input notebook__title', type: 'text', placeholder: 'Untitled',
  });
  s.titleInput.addEventListener('input', () => {
    s.title = s.titleInput.value;
    s.dirty = true;
    const nb = s.notebooks.find((n) => n.id === s.activeId);
    if (nb) { nb.title = s.title; renderList(ctx); }
    s.scheduleSave();
  });

  s.modelSelect = createEl('select', { class: 'notebook__model', title: 'Model' });
  s.modelSelect.addEventListener('change', () => {
    s.modelId = s.modelSelect.value;
    s.dirty = true;
    s.scheduleSave();
    // capability-gated sampler controls (enable_thinking) track the model
    drawer.requestRebuild({ force: true });
  });

  const row = createEl('div', { class: 'notebook__row' }, [s.titleInput, s.modelSelect]);

  s.sysPromptInput = createEl('textarea', {
    class: 'notebook__sysprompt-input', rows: 3,
    placeholder: 'Optional system prompt for generation…',
  });
  s.sysPromptInput.addEventListener('input', () => {
    s.systemPrompt = s.sysPromptInput.value;
    s.dirty = true;
    s.scheduleSave();
  });
  // Lives in the shared settings drawer (registered as a section), not inline
  // in the form -- but state (value/open) is still driven by populateFields.
  s.sysPromptDetails = createEl('details', { class: 'notebook__sysprompt' }, [
    createEl('summary', {}, ['System prompt']),
    s.sysPromptInput,
  ]);

  s.contentTextarea = createEl('textarea', {
    class: 'notebook__content', placeholder: 'Start writing…',
  });
  s.contentTextarea.addEventListener('input', () => {
    s.content = s.contentTextarea.value;
    s.dirty = true;
    autoGrow(s.contentTextarea, Infinity);
    s.scheduleSave();
  });
  const contentWrap = createEl('div', { class: 'notebook__content-wrap' }, [s.contentTextarea]);

  s.generateBtn = createEl('button', { class: 'btn btn--primary' }, ['Generate']);
  s.generateBtn.addEventListener('click', () => (s.gen ? stopGenerate(ctx) : startGenerate(ctx)));
  const actions = createEl('div', { class: 'notebook__actions' }, [s.generateBtn]);

  s.formEl = createEl('div', { class: 'notebook__form' }, [row, contentWrap, actions]);
  s.emptyEl = createEl('div', { class: 'empty-state notebook__empty' }, [
    'Create a notebook to draft with the model — Generate continues from your cursor.',
  ]);

  s.editorBody = createEl('div', { class: 'notebook__body' });
  s.statusEl = createEl('div', { class: 'notebook__status', role: 'status' });

  const editorSection = createEl('section', { class: 'notebook__editor' }, [
    toolbar,
    s.editorBody,
    s.statusEl,
  ]);

  s.rootEl = createEl('div', { class: 'notebook' }, [listPane, editorSection]);
  // Mobile: tapping the visible editor (outside the list pane + toggle) dismisses
  // the slide-in pane.
  dismissPaneOnOutsideClick(s.rootEl, 'notebook--list-open', '.notebook__list-pane', '.notebook__list-toggle');
  ctx.el.append(s.rootEl);
}

function fillModelSelect(ctx) {
  const s = ctx.state;
  fillOptions(s.modelSelect, s.models.map((m) => m.id));
}

function notebookCaps(ctx) {
  const model = ctx.state.models.find((m) => m.id === ctx.state.modelSelect.value);
  return model?.capabilities ?? [];
}

function showStatus(ctx, text, isError = false) {
  setStatus(ctx.state.statusEl, text, isError);
}

function renderEditor(ctx) {
  const s = ctx.state;
  s.editorBody.replaceChildren(s.activeId ? s.formEl : s.emptyEl);
}

function populateFields(ctx) {
  const s = ctx.state;
  s.titleInput.value = s.title;
  s.sysPromptInput.value = s.systemPrompt;
  s.sysPromptDetails.open = Boolean(s.systemPrompt);
  s.contentTextarea.value = s.content;
  autoGrow(s.contentTextarea, Infinity);
  if (s.modelId && s.models.some((m) => m.id === s.modelId)) s.modelSelect.value = s.modelId;
  s.modelId = s.modelSelect.value; // reflect whatever the widget actually landed on
  renderEditor(ctx);
  renderList(ctx);
}

// ---------------------------------------------------------------------------
// notebooks sidebar
// ---------------------------------------------------------------------------

function renderList(ctx) {
  const s = ctx.state;
  if (!s.notebooks.length) {
    s.listEl.replaceChildren(
      createEl('div', { class: 'empty-state' }, ['No notebooks yet.']),
    );
    return;
  }
  s.listEl.replaceChildren(...s.notebooks.map((nb) => {
    const label = nb.title || 'Untitled';
    const title = createEl('span', { class: 'notebook-item__title', title: label }, [label]);
    const del = armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost notebook-item__delete' }, ['Del']),
      () => deleteNotebook(ctx, nb.id),
    );
    const item = createEl('div', {
      class: `notebook-item${nb.id === s.activeId ? ' notebook-item--active' : ''}`,
    }, [title, del]);
    item.addEventListener('click', () => {
      selectNotebook(ctx, nb.id);
      s.rootEl.classList.remove('notebook--list-open');
    });
    return item;
  }));
}

async function newNotebook(ctx) {
  const s = ctx.state;
  s.scheduleSave.flush();
  try {
    const nb = await api.createNotebook({ title: 'Untitled', content: '', params: snapshotSettings() });
    if (!ctx.alive) return;
    s.notebooks.unshift(nb);
    renderList(ctx);
    await selectNotebook(ctx, nb.id);
    s.rootEl.classList.remove('notebook--list-open');
    s.titleInput.focus();
    s.titleInput.select();
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Create failed: ${err.message}`, true);
  }
}

async function deleteNotebook(ctx, id) {
  const s = ctx.state;
  try {
    await api.deleteNotebook(id);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Delete failed: ${err.message}`, true);
    return;
  }
  if (!ctx.alive) return;
  s.notebooks = s.notebooks.filter((n) => n.id !== id);
  if (s.activeId === id) {
    if (s.gen) stopGenerate(ctx); // partial still persists to its own notebook
    s.scheduleSave.cancel();
    s.dirty = false;
    s.activeId = null;
    s.title = '';
    s.content = '';
    s.systemPrompt = '';
    s.modelId = '';
    if (s.notebooks.length) await selectNotebook(ctx, s.notebooks[0].id);
    else renderEditor(ctx);
  }
  if (ctx.alive) renderList(ctx);
}

async function selectNotebook(ctx, id) {
  const s = ctx.state;
  if (s.activeId === id) return;
  if (s.gen) stopGenerate(ctx); // partial still persists to its own notebook
  s.scheduleSave.flush();
  s.activeId = id;
  showStatus(ctx, '');
  renderList(ctx);
  try {
    const nb = await api.getNotebook(id, { signal: ctx.signal });
    if (!ctx.alive || s.activeId !== id) return;
    s.title = nb.title ?? '';
    s.content = nb.content ?? '';
    s.systemPrompt = nb.system_prompt ?? '';
    s.modelId = nb.model_id ?? '';
    hydrateDocParams(nb);  // sampler panel <- this notebook (silent, no re-PUT)
    s.dirty = false;
    populateFields(ctx);
    // an open drawer shows the previous notebook's params/sysprompt otherwise
    drawer.requestRebuild({ force: true });
  } catch (err) {
    if (ctx.alive && s.activeId === id) {
      showStatus(ctx, `Could not load notebook: ${err.message}`, true);
    }
  }
}

// ---------------------------------------------------------------------------
// auto-save (debounced; always reads FROM ctx.state, never the DOM)
// ---------------------------------------------------------------------------

async function doSave(ctx) {
  const s = ctx.state;
  const id = s.activeId;
  if (!id || !s.dirty) return;
  s.dirty = false;
  try {
    await api.updateNotebook(id, {
      title: s.title,
      content: s.content,
      system_prompt: s.systemPrompt || null,
      model_id: s.modelId || null,
    });
  } catch (err) {
    if (ctx.alive && s.activeId === id) {
      s.dirty = true; // retry on the next edit/flush
      showStatus(ctx, `Save failed: ${err.message}`, true);
    }
  }
}

// ---------------------------------------------------------------------------
// generate-at-cursor
// ---------------------------------------------------------------------------

function startGenerate(ctx) {
  const s = ctx.state;
  if (s.gen || !s.activeId) return;
  if (!s.modelSelect.value) {
    showStatus(ctx, 'No models available.', true);
    return;
  }

  const insertPos = s.contentTextarea.selectionStart ?? s.content.length;
  const head = s.content.slice(0, insertPos);
  const tail = s.content.slice(insertPos);

  const messages = [];
  if (s.systemPrompt) messages.push({ role: 'system', content: s.systemPrompt });
  messages.push({ role: 'user', content: head.trim() ? head : 'Continue writing.' });

  const controller = ctx.linkedController();

  const gen = { controller, targetId: s.activeId, head, tail, content: '' };
  s.gen = gen;
  s.generateBtn.textContent = 'Stop';
  // Honest affordance: the painter overwrites textarea.value every frame, so
  // mid-generation keystrokes would be silently destroyed. Lock the surface
  // instead of eating input.
  s.contentTextarea.readOnly = true;
  showStatus(ctx, '');

  streamChat({ model: s.modelSelect.value, messages, ...samplerParams() }, {
    signal: controller.signal,
    onToken: (_, full) => { gen.content = full; if (ctx.alive) s.paint(); },
    onRetryWait: (wait) => {
      if (ctx.alive && s.gen === gen) showStatus(ctx, `Server busy -- retrying in ${wait}s…`);
    },
    onComplete: (result) => finishGenerate(ctx, gen, result),
    onError: (err) => handleGenerateError(ctx, gen, err),
  });
}

// Throttled painter (one per mount, created in setup).
function paintGen(ctx) {
  const s = ctx.state;
  const gen = s.gen;
  if (!gen || s.activeId !== gen.targetId) return;
  s.contentTextarea.value = gen.head + gen.content + gen.tail;
  const caret = gen.head.length + gen.content.length;
  s.contentTextarea.setSelectionRange(caret, caret);
  autoGrow(s.contentTextarea, Infinity);
}

function stopGenerate(ctx) {
  ctx.state.gen?.controller.abort();
}

function releaseGen(ctx, gen) {
  const s = ctx.state;
  // No-op after normal completion; drops the linkedController chain listener.
  gen.controller.abort();
  if (s.gen !== gen) return;
  s.gen = null;
  if (ctx.alive) {
    s.generateBtn.textContent = 'Generate';
    s.contentTextarea.readOnly = false;
  }
}

async function finishGenerate(ctx, gen, { content, aborted }) {
  const s = ctx.state;
  releaseGen(ctx, gen);

  const full = gen.head + content + gen.tail;
  const isTarget = s.activeId === gen.targetId;

  if (isTarget && ctx.alive) {
    s.content = full;
    s.dirty = true;
    s.contentTextarea.value = full;
    autoGrow(s.contentTextarea, Infinity);
    doSave(ctx); // save now -- no need to arm the debounce just to flush it
    if (aborted) showStatus(ctx, content ? 'Stopped -- partial text kept.' : 'Stopped.');
  } else if (content) {
    // partial still persists to the notebook it belonged to, even if the
    // user switched away or the page is tearing down mid-generation.
    try {
      await api.updateNotebook(gen.targetId, { content: full });
    } catch (err) {
      console.warn('background notebook save failed', err);
    }
  }
}

function handleGenerateError(ctx, gen, err) {
  const s = ctx.state;
  releaseGen(ctx, gen);
  if (ctx.alive && s.activeId === gen.targetId) {
    showStatus(ctx, `Generation failed: ${err.message}`, true);
  }
}
