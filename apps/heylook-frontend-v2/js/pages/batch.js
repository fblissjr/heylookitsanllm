// Batch page -- multi-prompt batch completions
// NOTE: innerHTML usage is exclusively through renderMarkdown() which sanitizes via DOMPurify.

import * as api from '../api.js'
import { renderMarkdown, ensureMarked } from '../components/markdown.js'
import { createEl, statCard } from '../utils.js'
import { getSettings, updateSettings } from '../settings.js'

let container = null
let state = null

function freshState() {
  const s = getSettings()
  return {
    models: [],
    selectedModel: null,
    prompts: ['', ''],
    temperature: s.temperature,
    maxTokens: s.max_tokens,
    running: false,
    results: null,
    error: null,
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
  container = null
}

async function init() {
  await ensureMarked()
  try {
    const data = await api.listModels()
    state.models = data.data || []
    if (state.models.length > 0) state.selectedModel = state.models[0].id
    renderForm()
  } catch {
    // Models unavailable
  }
}

function buildShell() {
  const page = createEl('div', { class: 'page-shell' })
  const header = createEl('div', { class: 'page-header' })
  header.append(createEl('h2', { class: 'page-title' }, 'Batch'))
  const content = createEl('div', { id: 'batch-content', class: 'page-content' })
  page.append(header, content)
  container.append(page)
}

function renderForm() {
  const el = container?.querySelector('#batch-content')
  if (!el) return
  const fragment = document.createDocumentFragment()

  // Model selector
  const modelRow = createEl('div', { class: 'form-row' })
  modelRow.append(createEl('label', { class: 'form-label' }, 'Model'))
  const select = createEl('select', { class: 'form-select', id: 'batch-model' })
  for (const m of state.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.id
    if (m.id === state.selectedModel) opt.selected = true
    select.append(opt)
  }
  select.addEventListener('change', () => { state.selectedModel = select.value })
  modelRow.append(select)
  fragment.append(modelRow)

  // Prompts
  const promptsLabel = createEl('div', { class: 'form-row' })
  promptsLabel.append(
    createEl('label', { class: 'form-label' }, 'Prompts'),
    createEl('button', { class: 'btn btn-sm', id: 'add-prompt-btn' }, '+ Add'),
  )
  fragment.append(promptsLabel)

  const promptsDiv = createEl('div', { id: 'prompts-list' })
  for (let i = 0; i < state.prompts.length; i++) {
    const row = createEl('div', { class: 'prompt-row' })
    const textarea = createEl('textarea', {
      class: 'form-textarea',
      placeholder: `Prompt ${i + 1}`,
      rows: '3',
    })
    textarea.value = state.prompts[i]
    const idx = i
    textarea.addEventListener('input', () => { state.prompts[idx] = textarea.value })

    row.append(textarea)
    if (state.prompts.length > 2) {
      const removeBtn = createEl('button', { class: 'btn btn-sm btn-ghost' }, 'x')
      removeBtn.addEventListener('click', () => { state.prompts.splice(idx, 1); renderForm() })
      row.append(removeBtn)
    }
    promptsDiv.append(row)
  }
  fragment.append(promptsDiv)

  // Sampler params
  const paramsRow = createEl('div', { class: 'form-row form-row--inline' })
  const tempInput = createEl('input', { type: 'number', class: 'form-input form-input--sm', value: String(state.temperature), step: '0.1', min: '0', max: '2' })
  tempInput.addEventListener('change', () => { state.temperature = parseFloat(tempInput.value) || 0.7; updateSettings({ temperature: state.temperature }) })
  const maxInput = createEl('input', { type: 'number', class: 'form-input form-input--sm', value: String(state.maxTokens), step: '64', min: '1' })
  maxInput.addEventListener('change', () => { state.maxTokens = parseInt(maxInput.value) || 512; updateSettings({ max_tokens: state.maxTokens }) })
  paramsRow.append(
    createEl('label', { class: 'form-label form-label--sm' }, 'Temp'),
    tempInput,
    createEl('label', { class: 'form-label form-label--sm' }, 'Max tokens'),
    maxInput,
  )
  fragment.append(paramsRow)

  // Buttons
  const btnRow = createEl('div', { class: 'form-row' })
  const runBtn = createEl('button', { class: 'btn btn-primary', id: 'run-batch-btn' }, state.running ? 'Running...' : 'Run Batch')
  runBtn.disabled = state.running
  runBtn.addEventListener('click', handleRun)
  btnRow.append(runBtn)
  fragment.append(btnRow)

  // Error
  if (state.error) {
    fragment.append(createEl('p', { class: 'error-text' }, state.error))
  }

  // Results
  if (state.results) {
    fragment.append(buildResults())
  }

  el.replaceChildren(fragment)

  // Add prompt button
  const addBtn = el.querySelector('#add-prompt-btn')
  if (addBtn) addBtn.addEventListener('click', () => { state.prompts.push(''); renderForm() })
}

function buildResults() {
  const section = createEl('div', { class: 'batch-results' })

  const stats = state.results.batch_stats
  if (stats) {
    const statsDiv = createEl('div', { class: 'batch-stats' })
    statsDiv.append(
      statCard('Requests', stats.total_requests),
      statCard('Time', `${stats.elapsed_seconds.toFixed(1)}s`),
      statCard('Throughput', `${stats.throughput_tok_per_sec.toFixed(0)} tok/s`),
      statCard('Memory', `${stats.memory_peak_mb.toFixed(0)} MB`),
    )
    section.append(statsDiv)
  }

  const responses = state.results.data || []
  for (let i = 0; i < responses.length; i++) {
    const resp = responses[i]
    const card = createEl('div', { class: 'result-card' })
    const header = createEl('div', { class: 'result-card-header' })
    header.append(createEl('span', { class: 'result-card-label' }, `Response ${i + 1}`))
    if (resp.usage) {
      header.append(createEl('span', { class: 'meta-tag' }, `${resp.usage.total_tokens} tokens`))
    }

    const content = createEl('div', { class: 'result-card-content' })
    const text = resp.choices?.[0]?.message?.content || ''
    // Sanitized through DOMPurify via renderMarkdown
    content.innerHTML = renderMarkdown(text)

    card.append(header, content)
    section.append(card)
  }

  return section
}

async function handleRun() {
  const validPrompts = state.prompts.filter(p => p.trim())
  if (validPrompts.length < 2) {
    state.error = 'At least 2 prompts required'
    renderForm()
    return
  }
  if (!state.selectedModel) {
    state.error = 'No model selected'
    renderForm()
    return
  }

  state.running = true
  state.error = null
  state.results = null
  renderForm()

  try {
    const requests = validPrompts.map(prompt => ({
      model: state.selectedModel,
      messages: [{ role: 'user', content: prompt }],
      temperature: state.temperature,
      max_tokens: state.maxTokens,
    }))

    state.results = await api.request('POST', '/v1/batch/chat/completions', { requests })
  } catch (e) {
    state.error = e.message
  } finally {
    state.running = false
    renderForm()
  }
}
