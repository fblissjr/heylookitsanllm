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
import { streamMessages } from '../streaming.js';
import { messagesParams, displayWireFields, snapshotSettings, bindDocumentParams, hydrateDocParams, documentScopeNote } from '../settings.js';
import * as drawer from '../settings-drawer.js';
import { createPresetBar, paintPresetChip } from '../preset-bar.js';
import { createPromptSection } from '../prompt-section.js';
import { createDocumentWriter } from '../document-writer.js';

// Match chat's streaming repaint rate: one paint per animation frame is up to
// 120/s on a ProMotion phone and nobody reads at that rate.
const PAINT_INTERVAL_MS = 66;

// The display prefs this page HONORS -- one array, two uses: it declares what the
// drawer may offer (registerSettings `displayPrefs`) and it selects what goes on
// the wire (displayWireFields). Same list for both, so offering a control and
// sending it cannot come apart.
const DISPLAY_PREFS = ['show_special_tokens'];

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.notebooks = [];
    s.models = [];
    s.activeId = null;
    s.title = '';
    s.content = '';
    s.systemPrompt = '';
    s.appliedPresetId = null;
    s.modelId = '';
    s.dirty = false;
    s.gen = null; // { controller, targetId, head, tail, content }

    buildSkeleton(ctx);
    s.scheduleSave = debounce((opts) => { doSave(ctx, opts); }, 500);
    ctx.onTeardown(() => s.scheduleSave.flush());
    // The largest typed body on any page: flush it before the tab goes away.
    ctx.onHide(() => s.scheduleSave.flush({ keepalive: true }));

    // Shared preset bar (preset-bar.js), adapted to the active notebook.
    // Apply DOES write the notebook's system prompt -- a preset is a
    // prompt+sampler bundle everywhere, and the armed confirm guards the
    // overwrite. Notebook state uses '' for "no prompt"; the bar speaks null.
    // Created AFTER buildSkeleton (chat parity) so the chip exists before
    // any indicator callback can fire.
    s.presetBar = createPresetBar(ctx, {
      getPrompt: () => s.systemPrompt || null,
      setPrompt: (v) => setSystemPrompt(ctx, v),
      onStatus: (text, isError) => showStatus(ctx, text, isError),
      docId: () => s.activeId,
      onIndicator: (info) => paintPresetChip(s.presetChip, info),
      getStamp: () => s.appliedPresetId,
      setStamp: (id) => setAppliedPreset(ctx, id),
    });
    // The chip needs preset names before the drawer's first lazy fetch.
    s.presetBar.refresh().then(() => { if (ctx.alive) s.presetBar.syncIndicator(); });

    // Shared prompt-section factory (prompt-section.js), ONE instance for the
    // page's lifetime -- unlike chat, which rebuilds one per drawer build,
    // the notebook's widget is reused across notebook switches: owner() is a
    // live read (s.activeId), and every switch calls setValue() (in
    // populateFields) to resync the field AND re-anchor which notebook new
    // edits target. Created after the preset bar so onEdit can reach it.
    s.promptSection = createPromptSection(ctx, {
      owner: () => s.activeId,
      get: () => s.systemPrompt,
      set: (v) => { s.systemPrompt = v ?? ''; },
      persist: (v, id, opts) => putSystemPrompt(ctx, id, v, opts),
      onEdit: () => s.presetBar.updateDrift(), // prompt edits drift the selected preset live
    label: 'System prompt for this notebook',
    });

    // Notebook consumes messagesParams() for generate-at-cursor, so it gets full
    // sampler controls; the preset bar + per-notebook system-prompt editor
    // lead the panel.
    const unregisterSettings = drawer.registerSettings({
      caps: () => notebookCaps(ctx),
      samplers: 'enabled',
      scope: () => documentScopeNote('notebook', Boolean(s.activeId)),
      sections: () => [s.presetBar.buildSection(), s.promptSection.element],
      onOpen: s.presetBar.onDrawerOpen,
      displayPrefs: DISPLAY_PREFS,
    });
    ctx.onTeardown(unregisterSettings);
    // Per-NOTEBOOK sampler settings via the SAME shared binding chat uses --
    // one mechanism, no branched copy. Panel change -> debounced PUT to the
    // active notebook's `params`; hydrate on select is silent.
    ctx.onTeardown(bindDocumentParams({
      activeId: () => ctx.state.activeId,
      updateDoc: (id, body, opts) => api.updateNotebook(id, body, opts),
      onError: (err) => showStatus(ctx, `Settings save failed: ${err.message}`, true),
      onHide: ctx.onHide,
    }));
    ctx.onResume(ctx.guard(() => refreshAfterResume(ctx)));
    // One throttle for the whole mount (reads s.gen) -- a per-generation
    // throttle would pin each generation's head/tail copies until unmount.
    s.docWriter = createDocumentWriter({
      update: api.updateNotebook,
      onError: (msg) => showStatus(ctx, msg, true),
    });
    s.paint = ctx.throttleTime(() => paintGen(ctx), PAINT_INTERVAL_MS);

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

  // Applied-preset chip beside the model select -- same grammar as chat's.
  s.presetChip = createEl('button', {
    class: 'btn preset-chip', hidden: true,
    title: 'Preset applied to this notebook -- open settings',
  });
  s.presetChip.addEventListener('click', () => drawer.openSettings(s.presetChip));

  const row = createEl('div', { class: 'notebook__row' }, [s.titleInput, s.modelSelect, s.presetChip]);

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

// The preset-apply write path (the bar's setPrompt adapter): state + widget
// sync now, then an immediate field-scoped PUT if a notebook is active --
// chat.js's setSystemPrompt has the same shape. The textarea has its own
// path -- per-keystroke state + debounced PUT via the shared prompt-section
// factory; both converge on putSystemPrompt.

// Thin delegates to the shared writer (document-writer.js). What stays here is
// only what is genuinely this page's: which api function to call, and the
// pre-create guard below.
function putSystemPrompt(ctx, docId, value, opts) {
  ctx.state.docWriter.putSystemPrompt(docId, value, opts);
}

function setAppliedPreset(ctx, presetId) {
  const s = ctx.state;
  s.appliedPresetId = presetId;
  if (!s.activeId) return; // pre-create draft: the stamp rides in state only
  s.docWriter.setAppliedPreset(s.activeId, presetId);
}

function setSystemPrompt(ctx, value) {
  const s = ctx.state;
  const next = value ?? '';
  const changed = next !== s.systemPrompt;
  s.systemPrompt = next;
  s.promptSection.setValue(next);
  if (!s.activeId || !changed) return; // no-op PUTs skipped (preset re-apply)
  putSystemPrompt(ctx, s.activeId, next);
}

// All system-prompt PUTs serialize through one per-mount chain -- same
// ordering guard as chat.js's putSystemPrompt (see its comment): the
// textarea's blur-flush and a preset-apply's write are separate fetches
// issued milliseconds apart, and without ordering the stale one could land
// last server-side. Field-scoped: only system_prompt moves here, never the
// rest of the document (doSave owns title/content/model_id).
// Explicit preset stamp for the active notebook -- chat parity (see
// chat.js's setAppliedPreset and the provenance note in preset-bar.js).



function populateFields(ctx) {
  const s = ctx.state;
  s.titleInput.value = s.title;
  s.promptSection.setValue(s.systemPrompt);
  s.contentTextarea.value = s.content;
  autoGrow(s.contentTextarea, Infinity);
  if (s.modelId && s.models.some((m) => m.id === s.modelId)) s.modelSelect.value = s.modelId;
  s.modelId = s.modelSelect.value; // reflect whatever the widget actually landed on
  s.presetBar.syncIndicator(); // chip follows the (possibly new) active notebook
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
    // Same new-document preset inheritance as chat's newConversation: the
    // selected (or stamped) preset is the unit of continuity, and starting-as
    // is an explicit apply, so it stamps at create.
    const preset = s.presetBar.presetForNewDoc();
    const nb = await api.createNotebook({
      title: 'Untitled', content: '',
      system_prompt: preset?.system_prompt || undefined,
      params: preset ? { ...(preset.params ?? {}) } : snapshotSettings(),
      applied_preset_id: preset?.id,
    });
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
    else {
      s.presetBar.syncIndicator(); // no active doc -> chip clears (chat parity)
      // The prompt widget outlives any single notebook (one instance per
      // mount), so clearing state is not enough -- resync it too, or the
      // deleted notebook's prompt is still sitting in the drawer.
      s.promptSection.setValue('');
      renderEditor(ctx);
    }
  }
  if (ctx.alive) renderList(ctx);
}

async function selectNotebook(ctx, id) {
  const s = ctx.state;
  if (s.activeId === id) return;
  if (s.gen) stopGenerate(ctx); // partial still persists to its own notebook
  s.scheduleSave.flush();
  s.promptSection.flush(); // pending prompt write lands against the OLD notebook first
  s.activeId = id;
  showStatus(ctx, '');
  renderList(ctx);
  try {
    const nb = await api.getNotebook(id, { signal: ctx.signal });
    if (!ctx.alive || s.activeId !== id) return;
    s.title = nb.title ?? '';
    s.content = nb.content ?? '';
    s.systemPrompt = nb.system_prompt ?? '';
    s.appliedPresetId = nb.applied_preset_id ?? null;
    s.modelId = nb.model_id ?? '';
    hydrateDocParams(nb);  // sampler panel <- this notebook (silent, no re-PUT)
    s.dirty = false;
    populateFields(ctx);
    // an open drawer shows the previous notebook's params/sysprompt otherwise
    drawer.requestRebuild({ force: true });
  } catch (err) {
    if (ctx.alive && s.activeId === id) {
      showStatus(ctx, `Could not load notebook: ${err.message}`, true);
      // The doc never loaded: drop back to "no active notebook" so the
      // stale editor state can't be edited/saved under the wrong id, and so
      // re-clicking this notebook retries -- the select guard
      // (activeId === id) would otherwise no-op forever.
      s.activeId = null;
      s.appliedPresetId = null;
      renderEditor(ctx);
      renderList(ctx);
      // resync so the chip doesn't keep claiming the previous notebook's
      // preset for the one that failed to load
      s.presetBar.syncIndicator();
    }
  }
}

// ---------------------------------------------------------------------------
// auto-save (debounced; always reads FROM ctx.state, never the DOM)
// ---------------------------------------------------------------------------

async function doSave(ctx, { keepalive = false } = {}) {
  const s = ctx.state;
  const id = s.activeId;
  if (!id || !s.dirty) return;
  s.dirty = false;
  try {
    // system_prompt has its own field-scoped writer (putSystemPrompt, via the
    // shared prompt-section factory) -- omitted here, not just null-guarded,
    // so a slow whole-document save can never clobber a newer prompt PUT.
    await api.updateNotebook(id, {
      title: s.title,
      content: s.content,
      model_id: s.modelId || null,
    }, { keepalive });
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

  const controller = ctx.linkedController();

  const gen = { controller, targetId: s.activeId, head, tail, content: '' };
  s.gen = gen;
  s.generateBtn.textContent = 'Stop';
  // Honest affordance: the painter overwrites textarea.value every frame, so
  // mid-generation keystrokes would be silently destroyed. Lock the surface
  // instead of eating input.
  s.contentTextarea.readOnly = true;
  showStatus(ctx, '');

  // Phase 3b wire: /v1/messages -- system rides as the top-level `system`
  // parameter (Messages has no system role in the messages array).
  streamMessages({
    model: s.modelSelect.value,
    system: s.systemPrompt || undefined,
    messages: [{ role: 'user', content: head.trim() ? head : 'Continue writing.' }],
    ...messagesParams(notebookCaps(ctx)),
    // Display pref, not a sampler -- spread AFTER the bag and never inside it.
    ...displayWireFields(DISPLAY_PREFS),
  }, {
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
//
// Every line here is O(document): the value write, the selection move and
// autoGrow's forced layout all scale with what has been generated so far. This
// is the same shape of loop the chat painter had, one order of magnitude
// cheaper per run (plain text, no markdown) -- which is why the rate limit is
// the whole fix here and no incremental renderer is warranted.
function paintGen(ctx) {
  const s = ctx.state;
  const gen = s.gen;
  if (!gen || s.activeId !== gen.targetId) return;
  s.contentTextarea.value = gen.head + gen.content + gen.tail;
  // Moving the caret is how the view follows the generation point, which only
  // means anything while this textarea has focus. It is readOnly during
  // generation, so usually it does not -- and the call is not free on a long
  // value.
  if (document.activeElement === s.contentTextarea) {
    const caret = gen.head.length + gen.content.length;
    s.contentTextarea.setSelectionRange(caret, caret);
  }
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

// Re-adopt the store after the tab comes back (ctx.onResume). The page is a
// mirror with no other invalidation: nothing polls, and the select guard
// deliberately skips re-fetching the active notebook.
async function refreshAfterResume(ctx) {
  const s = ctx.state;
  if (s.resumeSync) return;
  s.resumeSync = true;
  try {
    const nbId = s.activeId;
    const [presetsChanged, list] = await Promise.all([
      s.presetBar.refresh(),
      api.listNotebooks({ signal: ctx.signal }).catch(() => null),
    ]);
    if (!ctx.alive) return;
    let docChanged = false;
    if (list?.notebooks) {
      const held = s.notebooks.find((n) => n.id === nbId)?.updated_at;
      const fresh = list.notebooks.find((n) => n.id === nbId)?.updated_at;
      const unchanged = Boolean(held && fresh && held === fresh);
      let adopted = unchanged;
      if (nbId && !unchanged) {
        const nb = await api.getNotebook(nbId, { signal: ctx.signal }).catch(() => null);
        if (!ctx.alive) return;
        if (nb && s.activeId === nbId) {
          docChanged = true;
          const typingPrompt = document.activeElement?.classList.contains('sysprompt-input') ?? false;
          const typingDoc = s.dirty || Boolean(s.gen) || document.activeElement === s.contentTextarea || document.activeElement === s.titleInput;
          if (!typingDoc) {
            s.title = nb.title ?? '';
            s.content = nb.content ?? '';
            s.modelId = nb.model_id ?? '';
            s.titleInput.value = s.title;
            s.contentTextarea.value = s.content;
            autoGrow(s.contentTextarea, Infinity);
            if (s.modelId && s.models.some((m) => m.id === s.modelId)) s.modelSelect.value = s.modelId;
            s.modelId = s.modelSelect.value;
          }
          if (!typingPrompt) {
            s.systemPrompt = nb.system_prompt ?? '';
            s.promptSection.setValue(s.systemPrompt);
          }
          s.appliedPresetId = nb.applied_preset_id ?? null;
          hydrateDocParams(nb);
          adopted = !typingPrompt && !typingDoc;
        }
      }
      s.notebooks = list.notebooks;
      renderList(ctx);
      if (!adopted && nbId) {
        const cur = s.notebooks.find((n) => n.id === nbId);
        if (cur && held) cur.updated_at = held;
      }
    }
    if (presetsChanged || docChanged) {
      s.presetBar.syncIndicator();
      drawer.requestRebuild();
    }
  } finally {
    s.resumeSync = false;
  }
}
