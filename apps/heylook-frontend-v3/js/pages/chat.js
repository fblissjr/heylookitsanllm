// Chat: conversations sidebar + streaming thread. Markdown render path only
// (marked + DOMPurify via renderMarkdown -- no other text->HTML path).
//
// Invariants:
// - Edit / regenerate / delete all use position-based truncation:
//   DELETE /messages?after=P removes every message with position > P.
// - Stream callbacks are keyed to the stream object AND target conversation;
//   switching conversations aborts the stream, but the partial assistant
//   message is still persisted to the conversation it belonged to.
// - Abort (Stop button) is normal completion: partial content is saved.

import { createPage } from '../page.js';
import { createEl, autoGrow, armedConfirm, beforeUnloadGuard, formatBytes, setStatus, fillOptions } from '../utils.js';
import { api } from '../api.js';
import { streamChat } from '../streaming.js';
import { renderMarkdown } from '../markdown.js';
import { buildSettingsPanel, samplerParams, snapshotSettings, applySettings } from '../settings.js';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.conversations = [];
    s.models = [];
    s.activeId = null;
    s.messages = [];
    s.systemPrompt = null;
    s.stream = null;      // { controller, targetConvId, content, thinking, els, retries }
    s.editingId = null;
    s.presets = [];       // saved system-prompt + sampler bundles (server-side)
    s.presetId = null;    // select-box state only -- applying copies, not binds

    buildSkeleton(ctx);
    // One throttle for the whole mount (it reads s.stream), not one per
    // stream -- per-stream throttles would pin each stream's closure in the
    // page's cleanup list for the mount lifetime.
    s.paint = ctx.throttle(() => paintStream(ctx));
    ctx.onTeardown(() => {
      if (s.stream) beforeUnloadGuard.disable();
    });

    const [models, convList] = await Promise.all([
      api.listModels({ signal: ctx.signal }).catch(() => ({ data: [] })),
      api.listConversations({ signal: ctx.signal }).catch((err) => {
        if (ctx.alive) showStatus(ctx, `Could not load conversations: ${err.message}`, true);
        return { conversations: [] };
      }),
    ]);
    if (!ctx.alive) return;

    s.models = models.data ?? [];
    s.conversations = convList.conversations ?? [];
    fillModelSelect(ctx);
    renderConvList(ctx);

    if (s.conversations.length) {
      await selectConversation(ctx, s.conversations[0].id);
    } else {
      renderMessages(ctx);
    }
  },
});

// ---------------------------------------------------------------------------
// skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.convListEl = createEl('div', { class: 'chat__convs-list' });
  const newBtn = createEl('button', { class: 'btn btn--sm' }, ['New']);
  newBtn.addEventListener('click', () => newConversation(ctx));

  const convPane = createEl('aside', { class: 'chat__convs' }, [
    createEl('div', { class: 'chat__convs-head' }, [
      createEl('h2', {}, ['Conversations']),
      newBtn,
    ]),
    s.convListEl,
  ]);

  s.modelSelect = createEl('select', { title: 'Model' });
  s.modelSelect.addEventListener('change', () => {
    if (ctx.state.activeId) {
      api.updateConversation(ctx.state.activeId, { model_id: s.modelSelect.value })
        .catch((err) => console.warn('model_id save failed', err));
      const conv = s.conversations.find((c) => c.id === ctx.state.activeId);
      if (conv) conv.model_id = s.modelSelect.value;
    }
    // Capability-gated controls (enable_thinking) must track the model: an
    // open panel rebuilt here, not only on next open.
    if (!s.settingsHost.hidden) rebuildSettingsPanel(ctx);
  });

  const convsToggle = createEl('button', { class: 'btn btn--sm chat__convs-toggle' }, ['Chats']);
  convsToggle.addEventListener('click', () => s.rootEl.classList.toggle('chat--convs-open'));

  s.gearBtn = createEl('button', { class: 'btn btn--sm btn--ghost', title: 'Sampling settings' }, ['Settings']);
  s.gearBtn.addEventListener('click', () => toggleSettings(ctx));
  s.settingsHost = createEl('div', { class: 'chat__settings', hidden: true });

  s.messagesInner = createEl('div', { class: 'chat__messages-inner' });
  s.messagesEl = createEl('div', { class: 'chat__messages' }, [s.messagesInner]);

  s.statusEl = createEl('div', { class: 'chat__status' });

  s.textarea = createEl('textarea', { rows: 1, placeholder: 'Message… (Enter to send)' });
  s.textarea.addEventListener('input', () => autoGrow(s.textarea));
  s.textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(ctx);
    }
  });
  // Images: attach via picker (mobile camera roll comes free) or paste.
  s.pendingImages = [];
  s.fileInput = createEl('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true });
  s.fileInput.addEventListener('change', () => {
    addImages(ctx, s.fileInput.files);
    s.fileInput.value = '';
  });
  s.attachBtn = createEl('button', { class: 'btn', title: 'Attach images' }, ['+ Img']);
  s.attachBtn.addEventListener('click', () => s.fileInput.click());
  s.textarea.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData?.items ?? [])]
      .filter((it) => it.kind === 'file' && it.type.startsWith('image/'))
      .map((it) => it.getAsFile())
      .filter(Boolean);
    if (files.length) {
      e.preventDefault();
      addImages(ctx, files);
    }
  });
  s.attachStrip = createEl('div', { class: 'chat__attach', hidden: true });
  s.sendBtn = createEl('button', { class: 'btn btn--primary' }, ['Send']);
  s.sendBtn.addEventListener('click', () => (s.stream ? stopStream(ctx) : send(ctx)));

  const thread = createEl('section', { class: 'chat__thread' }, [
    createEl('header', { class: 'chat__bar' }, [
      convsToggle,
      s.modelSelect,
      createEl('div', { class: 'chat__bar-spacer' }),
      s.gearBtn,
    ]),
    s.settingsHost,
    s.messagesEl,
    s.statusEl,
    s.attachStrip,
    createEl('div', { class: 'chat__composer' }, [s.attachBtn, s.fileInput, s.textarea, s.sendBtn]),
  ]);

  s.rootEl = createEl('div', { class: 'chat' }, [convPane, thread]);
  ctx.el.append(s.rootEl);
}

function fillModelSelect(ctx) {
  const s = ctx.state;
  fillOptions(s.modelSelect, s.models.map((m) => m.id));
}

function currentCaps(ctx) {
  const model = ctx.state.models.find((m) => m.id === ctx.state.modelSelect.value);
  return model?.capabilities ?? [];
}

async function toggleSettings(ctx) {
  const host = ctx.state.settingsHost;
  if (!host.hidden) {
    host.hidden = true;
    return;
  }
  await refreshPresets(ctx);
  if (!ctx.alive) return;
  rebuildSettingsPanel(ctx);
  host.hidden = false;
}

function rebuildSettingsPanel(ctx) {
  ctx.state.settingsHost.replaceChildren(buildSettingsPanel({
    caps: currentCaps(ctx),
    lead: [buildPresetSection(ctx), buildSystemPromptSection(ctx)],
  }));
}

// ---------------------------------------------------------------------------
// system prompt (per-conversation) + presets (saved prompt/sampler bundles)
// ---------------------------------------------------------------------------

// One writer for the conversation's system prompt: state now, PUT if a
// conversation exists (explicit null clears server-side). With no active
// conversation the value rides along until send() creates one.
function setSystemPrompt(ctx, value) {
  const s = ctx.state;
  s.systemPrompt = value;
  if (!s.activeId) return;
  api.updateConversation(s.activeId, { system_prompt: value })
    .catch((err) => showStatus(ctx, `System prompt save failed: ${err.message}`, true));
}

function buildSystemPromptSection(ctx) {
  const s = ctx.state;
  const input = createEl('textarea', {
    class: 'chat__sysprompt-input', rows: 3,
    placeholder: 'Optional system prompt for this conversation…',
    value: s.systemPrompt ?? '',
  });
  // save on blur/commit, not per keystroke -- one PUT per edit session
  input.addEventListener('change', () => setSystemPrompt(ctx, input.value.trim() || null));
  return createEl('details', { class: 'chat__sysprompt', open: Boolean(s.systemPrompt) }, [
    createEl('summary', {}, ['System prompt']),
    input,
  ]);
}

async function refreshPresets(ctx) {
  const s = ctx.state;
  try {
    const res = await api.listPresets({ signal: ctx.signal });
    s.presets = res.presets ?? [];
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Could not load presets: ${err.message}`, true);
  }
  if (!s.presets.some((p) => p.id === s.presetId)) s.presetId = null;
}

// Applying a preset COPIES its fields (LM Studio semantics): params become
// the whole sampler panel state, system_prompt lands on the conversation.
// There is no live binding -- later edits don't touch the preset until Save.
function applyPreset(ctx, presetId) {
  const s = ctx.state;
  s.presetId = presetId || null;
  const preset = s.presets.find((p) => p.id === presetId);
  if (preset) {
    applySettings(preset.params ?? {});
    setSystemPrompt(ctx, preset.system_prompt ?? null);
    showStatus(ctx, `Preset "${preset.name}" applied.`);
  }
  rebuildSettingsPanel(ctx);
}

async function savePreset(ctx, name) {
  const s = ctx.state;
  name = name.trim();
  if (!name) return;
  const body = { name, system_prompt: s.systemPrompt, params: snapshotSettings() };
  const existing = s.presets.find((p) => p.name === name);
  try {
    const saved = existing
      ? await api.updatePreset(existing.id, body)
      : await api.createPreset(body);
    if (!ctx.alive) return;
    await refreshPresets(ctx);
    if (!ctx.alive) return;
    s.presetId = saved.id;
    rebuildSettingsPanel(ctx);
    showStatus(ctx, `Preset "${name}" ${existing ? 'updated' : 'saved'}.`);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Preset save failed: ${err.message}`, true);
  }
}

async function deletePreset(ctx) {
  const s = ctx.state;
  if (!s.presetId) return;
  try {
    await api.deletePreset(s.presetId);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Preset delete failed: ${err.message}`, true);
    return;
  }
  if (!ctx.alive) return;
  s.presetId = null;
  await refreshPresets(ctx);
  if (!ctx.alive) return;
  rebuildSettingsPanel(ctx);
}

function buildPresetSection(ctx) {
  const s = ctx.state;
  const selected = s.presets.find((p) => p.id === s.presetId);

  const select = createEl('select', { title: 'Apply a saved preset' }, [
    createEl('option', { value: '' }, ['Presets…']),
    ...s.presets.map((p) => createEl('option', { value: p.id }, [p.name])),
  ]);
  select.value = s.presetId ?? '';
  select.addEventListener('change', () => applyPreset(ctx, select.value));

  const delBtn = armedConfirm(
    createEl('button', { class: 'btn btn--sm btn--ghost', disabled: !selected }, ['Del']),
    () => deletePreset(ctx),
  );

  // Save under the typed name: matches an existing preset -> overwrite it,
  // new name -> create. Picking a preset pre-fills its name for overwrite.
  const nameInput = createEl('input', {
    class: 'input', placeholder: 'Save as…', value: selected?.name ?? '',
  });
  const saveBtn = createEl('button', { class: 'btn btn--sm' }, ['Save']);
  saveBtn.addEventListener('click', () => savePreset(ctx, nameInput.value));
  nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') savePreset(ctx, nameInput.value);
  });

  return createEl('div', { class: 'preset-section' }, [
    createEl('h3', {}, ['Preset']),
    createEl('div', { class: 'preset-row' }, [select, delBtn]),
    createEl('div', { class: 'preset-row' }, [nameInput, saveBtn]),
  ]);
}

function showStatus(ctx, text, isError = false) {
  setStatus(ctx.state.statusEl, text, isError);
}

// ---------------------------------------------------------------------------
// conversations sidebar
// ---------------------------------------------------------------------------

function renderConvList(ctx) {
  const s = ctx.state;
  if (!s.conversations.length) {
    s.convListEl.replaceChildren(
      createEl('div', { class: 'empty-state' }, ['No conversations yet.']),
    );
    return;
  }
  s.convListEl.replaceChildren(...s.conversations.map((conv) => {
    const title = createEl('span', { class: 'conv-item__title', title: conv.title }, [conv.title]);
    title.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      startRename(ctx, conv, title);
    });
    const del = armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost conv-item__delete' }, ['Del']),
      () => deleteConversation(ctx, conv.id),
    );
    const item = createEl('div', {
      class: `conv-item${conv.id === s.activeId ? ' conv-item--active' : ''}`,
    }, [title, del]);
    item.addEventListener('click', () => {
      selectConversation(ctx, conv.id);
      s.rootEl.classList.remove('chat--convs-open');
    });
    return item;
  }));
}

function startRename(ctx, conv, titleEl) {
  const input = createEl('input', { class: 'input conv-item__rename', value: conv.title });
  const commit = ctx.guard(async () => {
    const next = input.value.trim();
    input.replaceWith(titleEl);
    if (!next || next === conv.title) return;
    try {
      await api.updateConversation(conv.id, { title: next });
      if (!ctx.alive) return;
      conv.title = next;
      renderConvList(ctx);
    } catch (err) {
      showStatus(ctx, `Rename failed: ${err.message}`, true);
    }
  });
  input.addEventListener('blur', commit);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') input.blur();
    if (e.key === 'Escape') { input.value = conv.title; input.blur(); }
  });
  input.addEventListener('click', (e) => e.stopPropagation());
  titleEl.replaceWith(input);
  input.focus();
  input.select();
}

async function newConversation(ctx) {
  const s = ctx.state;
  clearPendingImages(ctx); // staged images belong to the conv they were picked in
  try {
    const conv = await api.createConversation({
      title: 'New conversation',
      model_id: s.modelSelect.value || undefined,
    });
    if (!ctx.alive) return;
    s.conversations.unshift(conv);
    await selectConversation(ctx, conv.id);
    s.rootEl.classList.remove('chat--convs-open');
    s.textarea.focus();
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Create failed: ${err.message}`, true);
  }
}

async function deleteConversation(ctx, convId) {
  const s = ctx.state;
  try {
    await api.deleteConversation(convId);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Delete failed: ${err.message}`, true);
    return;
  }
  if (!ctx.alive) return;
  s.conversations = s.conversations.filter((c) => c.id !== convId);
  if (s.activeId === convId) {
    stopStream(ctx);
    s.activeId = null;
    s.messages = [];
    s.systemPrompt = null;
    if (s.conversations.length) await selectConversation(ctx, s.conversations[0].id);
    else {
      if (!s.settingsHost.hidden) rebuildSettingsPanel(ctx);
      renderMessages(ctx);
    }
  }
  if (ctx.alive) renderConvList(ctx);
}

async function selectConversation(ctx, convId) {
  const s = ctx.state;
  if (s.activeId === convId && s.messages.length) return;
  if (s.stream) stopStream(ctx); // partial still persists to its own conv
  s.activeId = convId;
  s.editingId = null;
  clearPendingImages(ctx); // staged images belong to the conv they were picked in
  showStatus(ctx, '');
  renderConvList(ctx);
  try {
    const conv = await api.getConversation(convId, { signal: ctx.signal });
    if (!ctx.alive || s.activeId !== convId) return;
    s.messages = conv.messages ?? [];
    s.systemPrompt = conv.system_prompt ?? null;
    if (conv.model_id && s.models.some((m) => m.id === conv.model_id)) {
      s.modelSelect.value = conv.model_id;
    }
    // an open panel shows the previous conversation's system prompt otherwise
    if (!s.settingsHost.hidden) rebuildSettingsPanel(ctx);
    renderMessages(ctx);
    scrollMessages(ctx, true);
  } catch (err) {
    if (ctx.alive && s.activeId === convId) {
      showStatus(ctx, `Could not load conversation: ${err.message}`, true);
    }
  }
}

// ---------------------------------------------------------------------------
// message rendering
// ---------------------------------------------------------------------------

function renderMessages(ctx) {
  const s = ctx.state;
  if (!s.activeId) {
    s.messagesInner.replaceChildren(
      createEl('div', { class: 'empty-state' },
        ['Send a message below to start a new conversation.']),
    );
    return;
  }
  s.messagesInner.replaceChildren(...s.messages.map((msg) =>
    s.editingId === msg.id ? buildEditEl(ctx, msg) : buildMessageEl(ctx, msg)));
}

function buildThinkingEl(thinking, open = false) {
  const details = createEl('details', { class: 'thinking', open }, [
    createEl('summary', {}, ['Thinking']),
    createEl('div', { class: 'thinking__body' }, [thinking ?? '']),
  ]);
  return details;
}

function buildMessageEl(ctx, msg) {
  const content = createEl('div', { class: 'message-content' });
  if (msg.role === 'assistant') content.innerHTML = renderMarkdown(msg.content);
  else content.textContent = msg.content;

  const bubbleChildren = [];
  if (hasImageBlocks(msg)) {
    bubbleChildren.push(createEl('div', { class: 'message-images' },
      msg.content_blocks
        .filter((b) => b.type === 'image')
        .map((b) => createEl('img', {
          class: 'message-image',
          src: imageBlockUrl(b),
          alt: 'attached image',
        }))));
  }
  bubbleChildren.push(content);
  const bubble = createEl('div', { class: 'message-bubble' }, bubbleChildren);
  const children = [];
  if (msg.role === 'assistant' && msg.thinking) children.push(buildThinkingEl(msg.thinking));
  children.push(bubble);
  children.push(buildActions(ctx, msg));

  return createEl('div', { class: `message message--${msg.role}` }, children);
}

function buildActions(ctx, msg) {
  const btn = (label, fn) => {
    const b = createEl('button', { class: 'btn btn--sm btn--ghost' }, [label]);
    b.addEventListener('click', fn);
    return b;
  };
  const actions = [];
  // Copy copies the flattened text; an image-only message has none to copy.
  if (msg.content) {
    actions.push(btn('Copy', () => navigator.clipboard?.writeText(msg.content).catch(() => {})));
  }
  // Editing is text-only: the editor would replace content and silently drop
  // the image blocks. Image messages get delete/regenerate, not edit.
  if (!hasImageBlocks(msg)) {
    actions.push(btn('Edit', () => {
      if (ctx.state.stream) return; // renderMessages would orphan the stream placeholder
      ctx.state.editingId = msg.id;
      renderMessages(ctx);
    }));
  }
  if (msg.role === 'assistant') {
    actions.push(btn('Regenerate', () => regenerate(ctx, msg)));
  }
  actions.push(armedConfirm(
    createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Delete']),
    () => deleteMessage(ctx, msg),
  ));
  return createEl('div', { class: 'message__actions' }, actions);
}

function buildEditEl(ctx, msg) {
  const s = ctx.state;
  const textarea = createEl('textarea', { value: msg.content });
  const cancel = () => { s.editingId = null; renderMessages(ctx); };

  const save = async (regenerateAfter) => {
    const next = textarea.value;
    try {
      if (next !== msg.content) {
        const updated = await api.updateMessage(s.activeId, msg.id, { content: next });
        if (!ctx.alive) return;
        // keep content AND content_blocks in sync with the server's view
        msg.content = updated.content;
        msg.content_blocks = updated.content_blocks;
      }
      s.editingId = null;
      if (regenerateAfter) {
        await truncateAfter(ctx, msg.position);
        if (!ctx.alive) return;
        renderMessages(ctx);
        startStream(ctx);
      } else {
        renderMessages(ctx);
      }
    } catch (err) {
      if (ctx.alive) showStatus(ctx, `Save failed: ${err.message}`, true);
    }
  };

  const buttons = [
    createEl('button', { class: 'btn btn--sm' }, ['Cancel']),
    createEl('button', { class: 'btn btn--sm btn--primary' }, ['Save']),
  ];
  buttons[0].addEventListener('click', cancel);
  buttons[1].addEventListener('click', () => save(false));
  if (msg.role === 'user') {
    const saveRegen = createEl('button', { class: 'btn btn--sm btn--primary' }, ['Save & Regenerate']);
    saveRegen.addEventListener('click', () => save(true));
    buttons.push(saveRegen);
  }

  const el = createEl('div', { class: `message message--${msg.role}` }, [
    createEl('div', { class: 'message-edit' }, [
      textarea,
      createEl('div', { class: 'message-edit__buttons' }, buttons),
    ]),
  ]);
  queueMicrotask(() => { autoGrow(textarea, 400); textarea.focus(); });
  return el;
}

function scrollMessages(ctx, force = false) {
  const el = ctx.state.messagesEl;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
  if (force || nearBottom) el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// message mutations (position-based truncation)
// ---------------------------------------------------------------------------

async function truncateAfter(ctx, position) {
  const s = ctx.state;
  await api.deleteMessagesAfter(s.activeId, position);
  if (!ctx.alive) return;
  s.messages = s.messages.filter((m) => m.position <= position);
}

async function regenerate(ctx, msg) {
  if (ctx.state.stream) return;
  try {
    await truncateAfter(ctx, msg.position - 1);
    if (!ctx.alive) return;
    renderMessages(ctx);
    scrollMessages(ctx, true);
    startStream(ctx);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Regenerate failed: ${err.message}`, true);
  }
}

async function deleteMessage(ctx, msg) {
  if (ctx.state.stream) return;
  try {
    await truncateAfter(ctx, msg.position - 1);
    if (!ctx.alive) return;
    renderMessages(ctx);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Delete failed: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// image attachments
// ---------------------------------------------------------------------------

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function addImages(ctx, files) {
  const s = ctx.state;
  const reads = await Promise.all([...files]
    .filter((f) => f.type.startsWith('image/'))
    .map((f) => fileToDataUrl(f)
      .then((dataUrl) => ({ dataUrl, mediaType: f.type }))
      .catch(() => null)));  /* unreadable file -- skip */
  if (!ctx.alive) return;
  s.pendingImages.push(...reads.filter(Boolean));
  renderAttachStrip(ctx);
}

function clearPendingImages(ctx) {
  ctx.state.pendingImages = [];
  renderAttachStrip(ctx);
}

function renderAttachStrip(ctx) {
  const s = ctx.state;
  s.attachStrip.hidden = !s.pendingImages.length;
  s.attachStrip.replaceChildren(...s.pendingImages.map((img, i) => {
    const remove = createEl('button', { class: 'attach-thumb__remove', title: 'Remove' }, ['×']);
    remove.addEventListener('click', () => {
      s.pendingImages.splice(i, 1);
      renderAttachStrip(ctx);
    });
    return createEl('div', { class: 'attach-thumb' }, [
      createEl('img', { src: img.dataUrl, alt: '' }),
      remove,
    ]);
  }));
}

// Stored shape is Messages-style content blocks (what the server persists);
// hasImageBlocks gates the flows that only make sense for text.
function hasImageBlocks(msg) {
  return Boolean(msg.content_blocks?.some((b) => b.type === 'image'));
}

// One place that knows how an image block becomes a URL (render AND wire use
// it -- they must never disagree). Handles both stored source types.
function imageBlockUrl(b) {
  return b.source.type === 'url'
    ? b.source.url
    : `data:${b.source.media_type};base64,${b.source.data}`;
}

function buildContentBlocks(text, images) {
  const blocks = images.map((img) => ({
    type: 'image',
    source: {
      type: 'base64',
      media_type: img.mediaType,
      data: img.dataUrl.slice(img.dataUrl.indexOf(',') + 1),
    },
  }));
  if (text) blocks.push({ type: 'text', text });
  return blocks;
}

// ---------------------------------------------------------------------------
// send + stream
// ---------------------------------------------------------------------------

async function send(ctx) {
  const s = ctx.state;
  const text = s.textarea.value.trim();
  const images = s.pendingImages;
  if ((!text && !images.length) || s.stream) return;
  if (!s.modelSelect.value) {
    showStatus(ctx, 'No models available.', true);
    return;
  }

  const content = images.length ? buildContentBlocks(text, images) : text;
  const title = (text || 'Image message').slice(0, 50);
  s.textarea.value = '';
  autoGrow(s.textarea);
  s.pendingImages = [];
  renderAttachStrip(ctx);
  showStatus(ctx, '');

  try {
    if (!s.activeId) {
      const conv = await api.createConversation({
        title,
        model_id: s.modelSelect.value,
        // a prompt typed (or preset applied) before the first send
        system_prompt: s.systemPrompt || undefined,
      });
      if (!ctx.alive) return;
      s.conversations.unshift(conv);
      s.activeId = conv.id;
      s.messages = [];
      s.systemPrompt = conv.system_prompt ?? null;
      renderConvList(ctx);
    } else {
      const conv = s.conversations.find((c) => c.id === s.activeId);
      if (conv && conv.title === 'New conversation' && !s.messages.length) {
        conv.title = title;
        api.updateConversation(s.activeId, { title: conv.title }).catch(() => {});
        renderConvList(ctx);
      }
    }

    const msg = await api.addMessage(s.activeId, { role: 'user', content });
    if (!ctx.alive) return;
    s.messages.push(msg);
    renderMessages(ctx);
    scrollMessages(ctx, true);
    startStream(ctx);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Send failed: ${err.message}`, true);
  }
}

// Wire shape for generation: v3 currently speaks the OpenAI chat-completions
// format, so stored image blocks convert to image_url parts (data URLs).
// When v3 migrates to /v1/messages (Phase 3b) the stored blocks ARE the wire
// shape and this conversion disappears.
function toWireContent(msg) {
  if (!hasImageBlocks(msg)) return msg.content;
  return msg.content_blocks.map((b) => (
    b.type === 'image'
      ? { type: 'image_url', image_url: { url: imageBlockUrl(b) } }
      : { type: 'text', text: b.text ?? '' }
  ));
}

function buildRequestBody(ctx) {
  const s = ctx.state;
  const messages = [];
  if (s.systemPrompt) messages.push({ role: 'system', content: s.systemPrompt });
  for (const m of s.messages) messages.push({ role: m.role, content: toWireContent(m) });
  return { model: s.modelSelect.value, messages, ...samplerParams() };
}

function startStream(ctx) {
  const s = ctx.state;
  if (s.stream || !s.activeId) return;

  const controller = ctx.linkedController();

  // streaming placeholder message
  const contentEl = createEl('div', { class: 'message-content' });
  const thinkingEl = buildThinkingEl('', true);
  thinkingEl.hidden = true;
  const msgEl = createEl('div', { class: 'message message--assistant message--streaming' }, [
    thinkingEl,
    createEl('div', { class: 'message-bubble' }, [contentEl]),
  ]);
  s.messagesInner.append(msgEl);
  scrollMessages(ctx, true);

  const stream = {
    controller,
    targetConvId: s.activeId,
    content: '',
    thinking: '',
    contentDirty: false,
    thinkingDirty: false,
    els: { msgEl, contentEl, thinkingBody: thinkingEl.querySelector('.thinking__body'), thinkingEl },
  };
  s.stream = stream;
  s.sendBtn.textContent = 'Stop';
  beforeUnloadGuard.enable();
  showStatus(ctx, '');

  const isCurrent = () => s.stream === stream && s.activeId === stream.targetConvId;

  streamChat(buildRequestBody(ctx), {
    signal: controller.signal,
    onToken: (_, full) => { stream.content = full; stream.contentDirty = true; if (ctx.alive) s.paint(); },
    onThinking: (_, full) => { stream.thinking = full; stream.thinkingDirty = true; if (ctx.alive) s.paint(); },
    onRetryWait: (wait) => {
      if (ctx.alive && isCurrent()) showStatus(ctx, `Server busy -- retrying in ${wait}s…`);
    },
    onComplete: (result) => finishStream(ctx, stream, result),
    onError: (err) => handleStreamError(ctx, stream, err),
  });
}

// Throttled painter (one per mount, created in setup): renders only the
// halves that changed since the last frame.
function paintStream(ctx) {
  const s = ctx.state;
  const stream = s.stream;
  if (!stream || s.activeId !== stream.targetConvId) return;
  if (stream.contentDirty) {
    stream.contentDirty = false;
    stream.els.contentEl.innerHTML = renderMarkdown(stream.content);
  }
  if (stream.thinkingDirty) {
    stream.thinkingDirty = false;
    stream.els.thinkingEl.hidden = false;
    stream.els.thinkingBody.textContent = stream.thinking;
  }
  scrollMessages(ctx);
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
  beforeUnloadGuard.disable();
  if (ctx.alive) s.sendBtn.textContent = 'Send';
}

async function finishStream(ctx, stream, { content, thinking, usage, timing, aborted }) {
  const s = ctx.state;
  releaseStream(ctx, stream);

  // Persist to the conversation the stream belonged to, even if the user
  // switched away or the page is tearing down mid-stream.
  let saved = null;
  if (content || thinking) {
    try {
      saved = await api.addMessage(stream.targetConvId, {
        role: 'assistant',
        content,
        thinking: thinking || undefined,
      });
    } catch (err) {
      if (ctx.alive) showStatus(ctx, `Could not save response: ${err.message}`, true);
    }
  }

  if (!ctx.alive || s.activeId !== stream.targetConvId) return;

  stream.els.msgEl.remove();
  if (saved) {
    s.messages.push(saved);
  } else if (content || thinking) {
    // save failed -- keep it on screen unsaved rather than vanishing text
    s.messages.push({ id: null, role: 'assistant', content, thinking,
      position: (s.messages[s.messages.length - 1]?.position ?? -1) + 1 });
  }
  renderMessages(ctx);
  scrollMessages(ctx);

  if (aborted) {
    showStatus(ctx, content ? 'Stopped -- partial response saved.' : 'Stopped.');
  } else if (usage) {
    const parts = [`${usage.completion_tokens ?? '?'} tokens`];
    if (timing?.peak_memory_gb != null) parts.push(`${timing.peak_memory_gb.toFixed(2)} GB peak`);
    if (timing?.kv_cache_bytes != null) parts.push(`${formatBytes(timing.kv_cache_bytes)} KV`);
    showStatus(ctx, parts.join(' · '));
  }
}

function handleStreamError(ctx, stream, err) {
  const s = ctx.state;
  releaseStream(ctx, stream);
  if (!ctx.alive || s.activeId !== stream.targetConvId) return;
  stream.els.msgEl.remove();
  renderMessages(ctx);
  showStatus(ctx, `Generation failed: ${err.message}`, true);
}
