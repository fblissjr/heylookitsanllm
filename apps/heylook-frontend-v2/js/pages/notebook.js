// Notebook page -- text scratchpad with LLM generation
// NOTE: Notebook content is plain text (not markdown). No innerHTML with user content.

import * as api from '../api.js'
import { streamChat } from '../streaming.js'
import { createEl, beforeUnloadGuard, throttleToFrame } from '../utils.js'
import { samplerParams } from '../settings.js'

let container = null
let state = null
let saveTimeout = null
let _throttledGenUpdate = null

function freshState() {
  return {
    notebooks: [],
    activeId: null,
    models: [],
    selectedModel: null,
    generating: false,
    controller: null,
    dirty: false,
  }
}

export function mount(el) {
  container = el
  state = freshState()
  buildShell()
  init()
  return { teardown }
}

function teardown() {
  stopGeneration()
  flushSave()
  if (saveTimeout) { clearTimeout(saveTimeout); saveTimeout = null }
  _throttledGenUpdate?.reset()
  state = null
  container = null
}

async function init() {
  await Promise.all([loadModels(), loadNotebooks()])
  if (!state) return
  renderList()
  if (state.notebooks.length > 0) {
    await selectNotebook(state.notebooks[0].id)
  }
}

async function loadModels() {
  try {
    const data = await api.listModels()
    if (!state) return
    state.models = data.data || []
  } catch { /* unavailable */ }
}

async function loadNotebooks() {
  try {
    const data = await api.listNotebooks()
    if (!state) return
    state.notebooks = data.notebooks || []
  } catch {
    if (state) state.notebooks = []
  }
}

function buildShell() {
  const page = createEl('div', { class: 'page-shell notebook-page' })

  const header = createEl('div', { class: 'page-header' })
  header.append(createEl('h2', { class: 'page-title' }, 'Notebooks'))

  const headerActions = createEl('div', { class: 'header-actions' })
  const newBtn = createEl('button', { class: 'btn btn-sm', id: 'new-nb-btn' }, 'New')
  newBtn.addEventListener('click', handleNew)
  headerActions.append(newBtn)
  header.append(headerActions)

  const body = createEl('div', { class: 'notebook-body' })
  const list = createEl('div', { class: 'notebook-list', id: 'nb-list' })
  const editor = createEl('div', { class: 'notebook-editor', id: 'nb-editor' })
  editor.append(createEl('div', { class: 'empty-text' }, 'Select or create a notebook'))
  body.append(list, editor)

  page.append(header, body)
  container.append(page)
}

function renderList() {
  const list = container?.querySelector('#nb-list')
  if (!list || !state) return

  list.replaceChildren()
  for (const nb of state.notebooks) {
    const item = createEl('div', {
      class: `notebook-item${nb.id === state.activeId ? ' notebook-item--active' : ''}`,
      'data-id': nb.id,
    })
    const title = createEl('span', { class: 'notebook-item-title' }, nb.title)
    const delBtn = createEl('button', { class: 'notebook-item-delete', 'data-id': nb.id }, 'x')
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation()
      handleDelete(nb.id)
    })
    item.append(title, delBtn)
    item.addEventListener('click', () => selectNotebook(nb.id))
    list.append(item)
  }
}

async function selectNotebook(id) {
  flushSave()
  if (!state) return
  state.activeId = id
  state.dirty = false
  stopGeneration()

  try {
    const nb = await api.getNotebook(id)
    if (!state || state.activeId !== id) return
    renderEditor(nb)
  } catch {
    renderEditor(null)
  }
  renderList()
}

function renderEditor(nb) {
  const editor = container?.querySelector('#nb-editor')
  if (!editor || !state) return

  if (!nb) {
    editor.replaceChildren(createEl('div', { class: 'empty-text' }, 'Notebook not found'))
    return
  }

  editor.replaceChildren()

  const titleInput = createEl('input', {
    class: 'notebook-title-input', type: 'text', placeholder: 'Title', value: nb.title,
  })
  titleInput.addEventListener('input', () => scheduleSave())

  const modelRow = createEl('div', { class: 'notebook-model-row' })
  const select = createEl('select', { class: 'form-select' })
  for (const m of state.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.id
    if (m.id === (nb.model_id || state.selectedModel)) opt.selected = true
    select.append(opt)
  }
  select.addEventListener('change', () => {
    state.selectedModel = select.value
    scheduleSave()
  })
  if (nb.model_id) state.selectedModel = nb.model_id
  modelRow.append(createEl('label', { class: 'form-label form-label--sm' }, 'Model'), select)

  const sysBlock = createEl('details', { class: 'notebook-system-block' })
  const sysSummary = createEl('summary', {}, 'System prompt')
  const sysInput = createEl('textarea', {
    class: 'form-textarea', rows: '2', placeholder: 'Optional system prompt...',
  })
  sysInput.value = nb.system_prompt || ''
  sysInput.addEventListener('input', () => scheduleSave())
  sysBlock.append(sysSummary, sysInput)

  const contentArea = createEl('textarea', {
    class: 'notebook-content', id: 'nb-content',
    placeholder: 'Start writing...',
  })
  contentArea.value = nb.content || ''
  contentArea.addEventListener('input', () => {
    autoResize(contentArea)
    scheduleSave()
  })

  const actions = createEl('div', { class: 'notebook-actions' })
  const genBtn = createEl('button', { class: 'btn btn-primary', id: 'gen-btn' }, 'Generate')
  genBtn.addEventListener('click', handleGenerate)
  const stopBtn = createEl('button', { class: 'btn btn-sm', id: 'stop-btn', style: 'display:none' }, 'Stop')
  stopBtn.addEventListener('click', stopGeneration)
  actions.append(genBtn, stopBtn)

  editor.append(titleInput, modelRow, sysBlock, actions, contentArea)
  requestAnimationFrame(() => autoResize(contentArea))
}

function autoResize(textarea) {
  textarea.style.height = 'auto'
  textarea.style.height = textarea.scrollHeight + 'px'
}

function scheduleSave() {
  if (!state) return
  state.dirty = true
  if (saveTimeout) clearTimeout(saveTimeout)
  saveTimeout = setTimeout(() => flushSave(), 500)
}

function flushSave() {
  if (!state?.activeId || !state.dirty) return
  const id = state.activeId

  const titleEl = container?.querySelector('.notebook-title-input')
  const contentEl = container?.querySelector('#nb-content')
  const sysEl = container?.querySelector('.notebook-system-block textarea')
  const selectEl = container?.querySelector('.notebook-model-row select')
  const fields = {}
  if (titleEl) fields.title = titleEl.value
  if (contentEl) fields.content = contentEl.value
  if (sysEl) fields.system_prompt = sysEl.value
  if (selectEl) fields.model_id = selectEl.value

  state.dirty = false
  api.updateNotebook(id, fields).catch(() => {})

  const nb = state.notebooks.find(n => n.id === id)
  if (nb && fields.title) {
    nb.title = fields.title
    renderList()
  }
}

async function handleNew() {
  if (!state) return
  const nb = await api.createNotebook({ title: 'Untitled', model_id: state.selectedModel })
  state.notebooks.unshift(nb)
  await selectNotebook(nb.id)
}

async function handleDelete(id) {
  if (!state) return
  if (!confirm('Delete this notebook?')) return
  if (state.activeId === id) {
    stopGeneration()
    if (saveTimeout) { clearTimeout(saveTimeout); saveTimeout = null }
    state.dirty = false
  }
  await api.deleteNotebook(id)
  if (!state) return
  state.notebooks = state.notebooks.filter(n => n.id !== id)
  if (state.activeId === id) {
    state.activeId = null
    const editor = container?.querySelector('#nb-editor')
    if (editor) editor.replaceChildren(createEl('div', { class: 'empty-text' }, 'Select or create a notebook'))
  }
  renderList()
}

async function handleGenerate() {
  if (!state?.activeId || state.generating) return
  const contentArea = container?.querySelector('#nb-content')
  if (!contentArea) return

  const model = state.selectedModel || state.models[0]?.id
  if (!model) return

  state.generating = true
  const controller = new AbortController()
  state.controller = controller
  window.addEventListener('beforeunload', beforeUnloadGuard)

  const genBtn = container?.querySelector('#gen-btn')
  const stopBtn = container?.querySelector('#stop-btn')
  if (genBtn) genBtn.style.display = 'none'
  if (stopBtn) stopBtn.style.display = ''

  const cursorPos = contentArea.selectionStart
  const beforeCursor = contentArea.value.slice(0, cursorPos)
  const afterCursor = contentArea.value.slice(cursorPos)

  const sysEl = container?.querySelector('.notebook-system-block textarea')
  const systemPrompt = sysEl?.value || null

  const messages = []
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt })
  messages.push({ role: 'user', content: beforeCursor || 'Continue writing.' })

  let generated = ''

  _throttledGenUpdate = throttleToFrame(() => {
    if (!state) return
    contentArea.value = beforeCursor + generated + afterCursor
    autoResize(contentArea)
    contentArea.selectionStart = contentArea.selectionEnd = cursorPos + generated.length
  })

  await streamChat(
    { model, messages, ...samplerParams() },
    {
      onToken(token) {
        if (!state) return
        generated += token
        _throttledGenUpdate()
      },
      onThinking() {},
      onComplete() {
        if (!state) return
        contentArea.value = beforeCursor + generated + afterCursor
        state.generating = false
        state.controller = null
        _throttledGenUpdate?.reset()
        window.removeEventListener('beforeunload', beforeUnloadGuard)
        if (genBtn) genBtn.style.display = ''
        if (stopBtn) stopBtn.style.display = 'none'
        scheduleSave()
      },
      onError(err) {
        if (!state) return
        state.generating = false
        state.controller = null
        _throttledGenUpdate?.reset()
        window.removeEventListener('beforeunload', beforeUnloadGuard)
        if (genBtn) genBtn.style.display = ''
        if (stopBtn) stopBtn.style.display = 'none'
        console.error('Generation error:', err)
      },
    },
    controller.signal,
  )
}

function stopGeneration() {
  if (state?.controller) {
    state.controller.abort()
    state.controller = null
    state.generating = false
    _throttledGenUpdate?.reset()
    window.removeEventListener('beforeunload', beforeUnloadGuard)
    const genBtn = container?.querySelector('#gen-btn')
    const stopBtn = container?.querySelector('#stop-btn')
    if (genBtn) genBtn.style.display = ''
    if (stopBtn) stopBtn.style.display = 'none'
  }
}
