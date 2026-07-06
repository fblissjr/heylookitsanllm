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
import { buildSettingsPanel, samplerParams } from '../settings.js';

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
    createEl('div', { class: 'chat__composer' }, [s.textarea, s.sendBtn]),
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

function toggleSettings(ctx) {
  const host = ctx.state.settingsHost;
  if (host.hidden) {
    host.replaceChildren(buildSettingsPanel({ caps: currentCaps(ctx) }));
    host.hidden = false;
  } else {
    host.hidden = true;
  }
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
    else renderMessages(ctx);
  }
  if (ctx.alive) renderConvList(ctx);
}

async function selectConversation(ctx, convId) {
  const s = ctx.state;
  if (s.activeId === convId && s.messages.length) return;
  if (s.stream) stopStream(ctx); // partial still persists to its own conv
  s.activeId = convId;
  s.editingId = null;
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

  const bubble = createEl('div', { class: 'message-bubble' }, [content]);
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
  const actions = [
    btn('Copy', () => navigator.clipboard?.writeText(msg.content).catch(() => {})),
    btn('Edit', () => {
      if (ctx.state.stream) return; // renderMessages would orphan the stream placeholder
      ctx.state.editingId = msg.id;
      renderMessages(ctx);
    }),
  ];
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
        await api.updateMessage(s.activeId, msg.id, { content: next });
        if (!ctx.alive) return;
        msg.content = next;
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
// send + stream
// ---------------------------------------------------------------------------

async function send(ctx) {
  const s = ctx.state;
  const text = s.textarea.value.trim();
  if (!text || s.stream) return;
  if (!s.modelSelect.value) {
    showStatus(ctx, 'No models available.', true);
    return;
  }

  s.textarea.value = '';
  autoGrow(s.textarea);
  showStatus(ctx, '');

  try {
    if (!s.activeId) {
      const conv = await api.createConversation({
        title: text.slice(0, 50),
        model_id: s.modelSelect.value,
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
        conv.title = text.slice(0, 50);
        api.updateConversation(s.activeId, { title: conv.title }).catch(() => {});
        renderConvList(ctx);
      }
    }

    const msg = await api.addMessage(s.activeId, { role: 'user', content: text });
    if (!ctx.alive) return;
    s.messages.push(msg);
    renderMessages(ctx);
    scrollMessages(ctx, true);
    startStream(ctx);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Send failed: ${err.message}`, true);
  }
}

function buildRequestBody(ctx) {
  const s = ctx.state;
  const messages = [];
  if (s.systemPrompt) messages.push({ role: 'system', content: s.systemPrompt });
  for (const m of s.messages) messages.push({ role: m.role, content: m.content });
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
