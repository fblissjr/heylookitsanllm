// Chat page -- conversation display, streaming, edit+regenerate
// NOTE: All user-generated HTML goes through DOMPurify via renderMarkdown().
// Only the structural HTML (layout, buttons) uses innerHTML with hardcoded templates.

import * as api from '../api.js'
import bus from '../bus.js'
import { streamChat } from '../streaming.js'
import { renderMarkdown, ensureMarked } from '../components/markdown.js'
import { createEl, beforeUnloadGuard } from '../utils.js'
import { samplerParams } from '../settings.js'
import { buildSettingsPanel } from '../components/settings_panel.js'

let container = null
let state = null
let _streamRafPending = false

function freshState() {
  return {
    conversations: [],
    activeId: null,
    messages: [],
    models: [],
    selectedModel: null,
    streaming: { active: false, content: '', thinking: '', controller: null },
    editingMsgId: null,
  }
}

export function mount(el) {
  container = el
  state = freshState()
  buildShell()

  const input = container.querySelector('#chat-input')
  const sendBtn = container.querySelector('#send-btn')
  const modelSelect = container.querySelector('#model-select')
  const sidebarBtn = container.querySelector('#sidebar-btn')

  input.addEventListener('input', () => {
    input.style.height = 'auto'
    input.style.height = Math.min(input.scrollHeight, 200) + 'px'
  })

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  })

  sendBtn.addEventListener('click', handleSend)

  modelSelect.addEventListener('change', () => {
    state.selectedModel = modelSelect.value
    if (state.activeId) {
      api.updateConversation(state.activeId, { model_id: state.selectedModel }).catch(() => {})
    }
  })

  sidebarBtn.addEventListener('click', () => {
    const sidebar = document.getElementById('sidebar')
    sidebar.classList.toggle('sidebar--open')
    toggleOverlay(sidebar.classList.contains('sidebar--open'))
  })

  settingsBtn.addEventListener('click', () => {
    const sc = container.querySelector('#settings-container')
    if (!sc) return
    if (sc.style.display === 'none') {
      // Build panel lazily with current model capabilities
      const model = state.models.find(m => m.id === state.selectedModel)
      const capabilities = model?.capabilities || []
      sc.replaceChildren(buildSettingsPanel({ capabilities }))
      sc.style.display = ''
    } else {
      sc.style.display = 'none'
    }
  })

  init()
  return { teardown }
}

function buildShell() {
  // Structural HTML only -- no user content
  const page = document.createElement('div')
  page.className = 'chat-page'

  const header = document.createElement('div')
  header.className = 'chat-header'
  const settingsBtn = createEl('button', { class: 'settings-toggle-btn', id: 'settings-btn', type: 'button', title: 'Sampler settings' }, '\u2699')
  header.append(
    createEl('button', { class: 'sidebar-toggle', id: 'sidebar-btn', type: 'button' }, '\u2630'),
    createEl('select', { id: 'model-select' }),
    settingsBtn,
    createEl('span', { id: 'chat-status', style: 'font-size:12px;color:var(--text-dim)' }),
  )

  // Settings panel (hidden by default)
  const settingsPanelContainer = createEl('div', { id: 'settings-container', style: 'display:none' })

  const messages = createEl('div', { class: 'chat-messages', id: 'chat-messages' })

  const inputArea = document.createElement('div')
  inputArea.className = 'chat-input-area'
  const inputRow = document.createElement('div')
  inputRow.className = 'chat-input-row'
  inputRow.append(
    createEl('textarea', { class: 'chat-input', id: 'chat-input', placeholder: 'Type a message...', rows: '1' }),
    createEl('button', { class: 'btn btn-primary', id: 'send-btn', type: 'button' }, 'Send'),
  )
  inputArea.append(inputRow)

  page.append(header, settingsPanelContainer, messages, inputArea)
  container.append(page)
}

function teardown() {
  stopStream()
  _streamRafPending = false
  const list = document.getElementById('conversation-list')
  if (list) list.onclick = null
  const newBtn = document.getElementById('new-chat-btn')
  if (newBtn) newBtn.onclick = null
  state = null
  container = null
}

async function init() {
  await ensureMarked()
  await Promise.all([loadModels(), loadConversations()])
  renderConversationList()

  if (state.conversations.length > 0) {
    await selectConversation(state.conversations[0].id)
  }
}

async function loadModels() {
  try {
    const data = await api.listModels()
    state.models = data.data || []
    renderModelSelect()
  } catch {
    setStatus('Failed to connect')
  }
}

async function loadConversations() {
  try {
    const data = await api.listConversations()
    state.conversations = data.conversations || []
  } catch {
    state.conversations = []
  }
}

function renderModelSelect() {
  const select = container?.querySelector('#model-select')
  if (!select) return

  select.replaceChildren()
  for (const m of state.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.id
    if (m.id === state.selectedModel) opt.selected = true
    select.append(opt)
  }

  if (!state.selectedModel && state.models.length > 0) {
    state.selectedModel = state.models[0].id
  }
}

function renderConversationList() {
  const list = document.getElementById('conversation-list')
  if (!list) return

  list.replaceChildren()
  for (const c of state.conversations) {
    const item = createEl('div', {
      class: `conversation-item${c.id === state.activeId ? ' conversation-item--active' : ''}`,
      'data-id': c.id,
    }, c.title)
    list.append(item)
  }

  list.onclick = (e) => {
    const item = e.target.closest('.conversation-item')
    if (item) {
      selectConversation(item.dataset.id)
      document.getElementById('sidebar')?.classList.remove('sidebar--open')
      toggleOverlay(false)
    }
  }

  const newBtn = document.getElementById('new-chat-btn')
  if (newBtn) {
    newBtn.onclick = async () => {
      const conv = await api.createConversation({
        title: 'New Conversation',
        model_id: state.selectedModel,
      })
      state.conversations.unshift(conv)
      await selectConversation(conv.id)
      renderConversationList()
    }
  }
}

async function selectConversation(id) {
  stopStream()
  state.activeId = id
  state.editingMsgId = null

  try {
    const conv = await api.getConversation(id)
    state.messages = conv.messages || []
    if (conv.model_id) {
      state.selectedModel = conv.model_id
      renderModelSelect()
    }
  } catch {
    state.messages = []
  }

  renderMessages()
  renderConversationList()
  scrollToBottom()
}

// -- Message rendering (DOM-based, user content goes through DOMPurify) --

function renderMessages() {
  const el = container?.querySelector('#chat-messages')
  if (!el) return

  const fragment = document.createDocumentFragment()
  for (const msg of state.messages) {
    fragment.append(buildMessageEl(msg))
  }

  if (state.streaming.active) {
    fragment.append(buildStreamingEl())
  }

  el.replaceChildren(fragment)
}

function buildMessageEl(msg) {
  const isEditing = state.editingMsgId === msg.id
  const row = document.createElement('div')
  row.className = `message message--${msg.role}`
  row.dataset.msgId = msg.id

  const bubble = document.createElement('div')
  bubble.className = 'message-bubble'

  if (isEditing) {
    bubble.style.maxWidth = '100%'
    const area = document.createElement('textarea')
    area.className = 'message-edit-area'
    area.id = 'edit-area'
    area.value = msg.content

    const actions = document.createElement('div')
    actions.className = 'edit-actions'
    const saveBtn = createEl('button', { class: 'btn btn-sm btn-primary', 'data-action': 'save-edit', 'data-msg-id': msg.id }, 'Save')
    const regenBtn = createEl('button', { class: 'btn btn-sm btn-primary', 'data-action': 'save-regenerate', 'data-msg-id': msg.id }, 'Save & Regenerate')
    const cancelBtn = createEl('button', { class: 'btn btn-sm btn-ghost', 'data-action': 'cancel-edit' }, 'Cancel')
    actions.append(saveBtn, regenBtn, cancelBtn)
    ;[saveBtn, regenBtn, cancelBtn].forEach(b => b.addEventListener('click', handleMessageAction))

    bubble.append(area, actions)
  } else {
    if (msg.thinking) {
      bubble.append(buildThinkingEl(msg.thinking))
    }

    const contentDiv = document.createElement('div')
    contentDiv.className = 'message-content'
    // User content sanitized through DOMPurify via renderMarkdown
    contentDiv.innerHTML = renderMarkdown(msg.content)
    bubble.append(contentDiv)

    if (msg.role !== 'system') {
      bubble.append(buildActionsEl(msg))
    }
  }

  row.append(bubble)
  return row
}

function buildThinkingEl(thinking, isStreaming = false) {
  const block = document.createElement('div')
  block.className = `thinking-block${isStreaming ? ' thinking-block--open' : ''}`

  const toggle = document.createElement('button')
  toggle.className = 'thinking-toggle'
  toggle.type = 'button'
  toggle.textContent = isStreaming ? 'Thinking...' : 'Thinking'
  toggle.addEventListener('click', () => block.classList.toggle('thinking-block--open'))

  const content = document.createElement('div')
  content.className = 'thinking-content'
  content.textContent = thinking

  block.append(toggle, content)
  return block
}

function buildActionsEl(msg) {
  const div = document.createElement('div')
  div.className = 'message-actions'

  const editBtn = createEl('button', { class: 'message-action', 'data-action': 'edit', 'data-msg-id': msg.id }, 'Edit')
  const copyBtn = createEl('button', { class: 'message-action', 'data-action': 'copy', 'data-msg-id': msg.id }, 'Copy')
  const deleteBtn = createEl('button', { class: 'message-action message-action--danger', 'data-action': 'delete', 'data-msg-id': msg.id, 'data-position': msg.position }, 'Delete')
  div.append(editBtn, copyBtn)

  if (msg.role === 'assistant') {
    const regenBtn = createEl('button', { class: 'message-action', 'data-action': 'regenerate', 'data-msg-id': msg.id, 'data-position': msg.position }, 'Regenerate')
    div.append(regenBtn)
    regenBtn.addEventListener('click', handleMessageAction)
  }

  div.append(deleteBtn)
  ;[editBtn, copyBtn, deleteBtn].forEach(b => b.addEventListener('click', handleMessageAction))

  return div
}

function buildStreamingEl() {
  const row = document.createElement('div')
  row.className = 'message message--assistant'
  row.dataset.streaming = ''

  const bubble = document.createElement('div')
  bubble.className = 'message-bubble'

  if (state.streaming.thinking) {
    bubble.append(buildThinkingEl(state.streaming.thinking, true))
  }

  const contentDiv = document.createElement('div')
  contentDiv.className = 'message-content'
  // Streaming content sanitized through DOMPurify via renderMarkdown
  contentDiv.innerHTML = state.streaming.content
    ? renderMarkdown(state.streaming.content)
    : ''
  const cursor = document.createElement('span')
  cursor.className = 'streaming-cursor'
  contentDiv.append(cursor)
  bubble.append(contentDiv)

  const actions = document.createElement('div')
  actions.className = 'message-actions message-actions--visible'
  const stopBtn = createEl('button', { class: 'message-action', 'data-action': 'stop' }, 'Stop')
  stopBtn.addEventListener('click', handleMessageAction)
  actions.append(stopBtn)
  bubble.append(actions)

  row.append(bubble)
  return row
}

// -- Send & stream --

async function handleSend() {
  const input = container?.querySelector('#chat-input')
  if (!input) return
  const content = input.value.trim()
  if (!content || state.streaming.active) return

  if (!state.activeId) {
    const conv = await api.createConversation({
      title: content.slice(0, 50),
      model_id: state.selectedModel,
    })
    state.activeId = conv.id
    state.conversations.unshift(conv)
    renderConversationList()
  }

  const userMsg = await api.appendMessage(state.activeId, { role: 'user', content })
  state.messages.push(userMsg)
  input.value = ''
  input.style.height = 'auto'

  renderMessages()
  scrollToBottom()

  if (state.messages.length === 1) {
    api.updateConversation(state.activeId, { title: content.slice(0, 50) }).catch(() => {})
    const conv = state.conversations.find(c => c.id === state.activeId)
    if (conv) conv.title = content.slice(0, 50)
    renderConversationList()
  }

  await startStream()
}

async function startStream() {
  if (!state.selectedModel || !state.activeId) return

  const controller = new AbortController()
  state.streaming = { active: true, content: '', thinking: '', controller }
  window.addEventListener('beforeunload', beforeUnloadGuard)
  renderMessages()
  scrollToBottom()

  const request = {
    model: state.selectedModel,
    messages: buildApiMessages(),
    ...samplerParams(),
  }

  const targetConvId = state.activeId

  await streamChat(request, {
    onToken(token) {
      if (state.activeId !== targetConvId) return
      state.streaming.content += token
      updateStreamingDisplay()
    },
    onThinking(text) {
      if (state.activeId !== targetConvId) return
      state.streaming.thinking += text
      updateStreamingDisplay()
    },
    async onComplete(data) {
      if (state.activeId !== targetConvId) return
      const content = state.streaming.content
      const thinking = state.streaming.thinking || null
      state.streaming = { active: false, content: '', thinking: '', controller: null }
      window.removeEventListener('beforeunload', beforeUnloadGuard)

      if (content) {
        const msg = await api.appendMessage(targetConvId, {
          role: 'assistant', content, thinking,
        })
        state.messages.push(msg)
      }

      renderMessages()
      scrollToBottom()
      setStatus(data?.usage ? `${data.usage.completion_tokens || 0} tokens` : '')
    },
    onError(error) {
      state.streaming = { active: false, content: '', thinking: '', controller: null }
      window.removeEventListener('beforeunload', beforeUnloadGuard)
      renderMessages()
      setStatus(`Error: ${error.message}`)
    },
  }, controller.signal)
}

function updateStreamingDisplay() {
  // Throttle to display frame rate -- tokens arrive faster than 60fps
  if (_streamRafPending) return
  _streamRafPending = true
  requestAnimationFrame(() => {
    _streamRafPending = false
    if (!state.streaming.active) return

    const msgEl = container?.querySelector('[data-streaming]')
    if (!msgEl) {
      renderMessages()
      scrollToBottom()
      return
    }

    const contentEl = msgEl.querySelector('.message-content')
    if (contentEl) {
      // Content sanitized through DOMPurify via renderMarkdown
      const rendered = renderMarkdown(state.streaming.content)
      contentEl.innerHTML = rendered
      const cursor = document.createElement('span')
      cursor.className = 'streaming-cursor'
      contentEl.append(cursor)
    }

    const thinkingEl = msgEl.querySelector('.thinking-content')
    if (thinkingEl && state.streaming.thinking) {
      thinkingEl.textContent = state.streaming.thinking
    } else if (!thinkingEl && state.streaming.thinking) {
      renderMessages()
    }

    scrollToBottom()
  })
}

function stopStream() {
  if (state.streaming.controller) {
    state.streaming.controller.abort()
    state.streaming.controller = null
    window.removeEventListener('beforeunload', beforeUnloadGuard)
  }
}

function buildApiMessages() {
  const conv = state.conversations.find(c => c.id === state.activeId)
  const msgs = []
  if (conv?.system_prompt) msgs.push({ role: 'system', content: conv.system_prompt })
  for (const m of state.messages) msgs.push({ role: m.role, content: m.content })
  return msgs
}

async function handleMessageAction(e) {
  const btn = e.currentTarget
  const action = btn.dataset.action
  const msgId = btn.dataset.msgId

  switch (action) {
    case 'stop':
      stopStream()
      break

    case 'edit':
      state.editingMsgId = msgId
      renderMessages()
      break

    case 'cancel-edit':
      state.editingMsgId = null
      renderMessages()
      break

    case 'save-edit': {
      const area = container.querySelector('#edit-area')
      if (!area) break
      await api.updateMessage(state.activeId, msgId, { content: area.value })
      const msg = state.messages.find(m => m.id === msgId)
      if (msg) msg.content = area.value
      state.editingMsgId = null
      renderMessages()
      break
    }

    case 'save-regenerate': {
      const area = container.querySelector('#edit-area')
      if (!area) break
      const msg = state.messages.find(m => m.id === msgId)
      if (!msg) break
      await api.updateMessage(state.activeId, msgId, { content: area.value })
      msg.content = area.value
      state.editingMsgId = null
      await api.truncateMessages(state.activeId, msg.position)
      state.messages = state.messages.filter(m => m.position <= msg.position)
      renderMessages()
      await startStream()
      break
    }

    case 'regenerate': {
      const msg = state.messages.find(m => m.id === msgId)
      if (!msg) break
      const prevPos = msg.position - 1
      await api.truncateMessages(state.activeId, prevPos)
      state.messages = state.messages.filter(m => m.position <= prevPos)
      renderMessages()
      await startStream()
      break
    }

    case 'delete': {
      if (!confirm('Delete this message and everything after it?')) break
      const msg = state.messages.find(m => m.id === msgId)
      if (!msg) break
      const prevPos = msg.position - 1
      await api.truncateMessages(state.activeId, prevPos)
      state.messages = state.messages.filter(m => m.position <= prevPos)
      renderMessages()
      break
    }

    case 'copy': {
      const msg = state.messages.find(m => m.id === msgId)
      if (msg) navigator.clipboard.writeText(msg.content).catch(() => {})
      break
    }
  }
}

function scrollToBottom() {
  const el = container?.querySelector('#chat-messages')
  if (el) requestAnimationFrame(() => { el.scrollTop = el.scrollHeight })
}

function setStatus(text) {
  const el = container?.querySelector('#chat-status')
  if (el) el.textContent = text
}

function toggleOverlay(show) {
  let overlay = document.querySelector('.sidebar-overlay')
  if (show && !overlay) {
    overlay = document.createElement('div')
    overlay.className = 'sidebar-overlay sidebar-overlay--visible'
    overlay.onclick = () => {
      document.getElementById('sidebar')?.classList.remove('sidebar--open')
      toggleOverlay(false)
    }
    document.getElementById('app').prepend(overlay)
  } else if (!show && overlay) {
    overlay.remove()
  }
}
