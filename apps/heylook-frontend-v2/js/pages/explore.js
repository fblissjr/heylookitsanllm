// Token Explorer -- stream with logprobs, visualize token probabilities
// NOTE: No innerHTML with user content. Token text uses textContent.

import * as api from '../api.js'
import { streamChat } from '../streaming.js'
import { createEl, beforeUnloadGuard, throttleToFrame } from '../utils.js'
import { samplerParams } from '../settings.js'
import { buildSettingsPanel } from '../components/settings_panel.js'

let container = null
let state = null
let _throttledRender = null

function freshState() {
  return {
    models: [],
    selectedModel: null,
    prompt: '',
    tokens: [],       // { token, logprob, top_logprobs[] }
    thinkingTokens: [],
    renderedCount: 0,
    selectedIdx: null,
    streaming: false,
    controller: null,
  }
}

export function mount(el) {
  container = el
  state = freshState()
  buildShell()

  const modelSelect = container.querySelector('#explore-model')
  const settingsBtn = container.querySelector('#explore-settings-btn')
  const runBtn = container.querySelector('#explore-run')
  const promptInput = container.querySelector('#explore-prompt')

  modelSelect.addEventListener('change', () => { state.selectedModel = modelSelect.value })

  settingsBtn.addEventListener('click', () => {
    const sc = container.querySelector('#explore-settings-container')
    if (!sc) return
    if (sc.style.display === 'none') {
      const model = state.models.find(m => m.id === state.selectedModel)
      sc.replaceChildren(buildSettingsPanel({ capabilities: model?.capabilities || [] }))
      sc.style.display = ''
    } else {
      sc.style.display = 'none'
    }
  })

  runBtn.addEventListener('click', handleRun)

  promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRun() }
  })

  // Keyboard nav for token selection
  document.addEventListener('keydown', handleKeyNav)

  init()
  return { teardown }
}

function teardown() {
  stopStream()
  _throttledRender?.reset()
  _throttledRender = null
  document.removeEventListener('keydown', handleKeyNav)
  state = null
  container = null
}

function buildShell() {
  const page = createEl('div', { class: 'page-shell' })

  const header = createEl('div', { class: 'page-header' })
  header.append(
    createEl('h2', { class: 'page-title' }, 'Token Explorer'),
    createEl('select', { id: 'explore-model', class: 'form-select' }),
    createEl('button', { class: 'settings-toggle-btn', id: 'explore-settings-btn', type: 'button', title: 'Sampler settings' }, '\u2699'),
  )

  const settingsContainer = createEl('div', { id: 'explore-settings-container', style: 'display:none' })

  const inputRow = createEl('div', { class: 'explore-input-row' })
  inputRow.append(
    createEl('textarea', { id: 'explore-prompt', class: 'form-textarea', placeholder: 'Enter a prompt...', rows: '2' }),
    createEl('button', { id: 'explore-run', class: 'btn btn-primary' }, 'Run'),
  )

  const content = createEl('div', { id: 'explore-content', class: 'page-content' })

  page.append(header, settingsContainer, inputRow, content)
  container.append(page)
}

async function init() {
  try {
    const data = await api.listModels()
    if (!state) return
    state.models = data.data || []
    if (state.models.length > 0) state.selectedModel = state.models[0].id
    renderModelSelect()
  } catch { /* unavailable */ }
}

function renderModelSelect() {
  const select = container?.querySelector('#explore-model')
  if (!select) return
  select.replaceChildren()
  for (const m of state.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.id
    if (m.id === state.selectedModel) opt.selected = true
    select.append(opt)
  }
}

async function handleRun() {
  if (!state || state.streaming) return
  const promptInput = container?.querySelector('#explore-prompt')
  if (!promptInput) return
  const prompt = promptInput.value.trim()
  if (!prompt || !state.selectedModel) return

  state.prompt = prompt
  state.tokens = []
  state.thinkingTokens = []
  state.renderedCount = 0
  state.selectedIdx = null
  state.streaming = true
  state.controller = new AbortController()
  window.addEventListener('beforeunload', beforeUnloadGuard)

  updateRunButton(true)
  renderTokens()

  const request = {
    model: state.selectedModel,
    messages: [{ role: 'user', content: prompt }],
    logprobs: true,
    top_logprobs: 5,
    ...samplerParams(),
  }

  await streamChat(request, {
    onToken(token) {
      // Token text accumulated for display -- logprobs arrive via onLogprobs
    },
    onThinking(text) {
      // Thinking tokens don't have logprobs, just accumulate text
      if (!state) return
      state.thinkingTokens.push({ token: text, logprob: 0, top_logprobs: [] })
    },
    onLogprobs(entries) {
      if (!state) return
      for (const entry of entries) {
        state.tokens.push({
          token: entry.token,
          logprob: entry.logprob,
          top_logprobs: entry.top_logprobs || [],
        })
      }
      if (!_throttledRender) _throttledRender = throttleToFrame(renderTokens)
      _throttledRender()
    },
    onComplete() {
      if (!state) return
      state.streaming = false
      state.controller = null
      window.removeEventListener('beforeunload', beforeUnloadGuard)
      updateRunButton(false)
      renderTokens()
    },
    onError(err) {
      if (!state) return
      state.streaming = false
      state.controller = null
      window.removeEventListener('beforeunload', beforeUnloadGuard)
      updateRunButton(false)
      console.error('Explore error:', err)
    },
  }, state.controller.signal)
}

function stopStream() {
  if (state?.controller) {
    state.controller.abort()
    state.controller = null
    state.streaming = false
    window.removeEventListener('beforeunload', beforeUnloadGuard)
  }
}

function updateRunButton(running) {
  const btn = container?.querySelector('#explore-run')
  if (!btn) return
  btn.textContent = running ? 'Stop' : 'Run'
  if (running) {
    btn.onclick = stopStream
  } else {
    btn.onclick = handleRun
  }
}

// -- Rendering --

function renderTokens() {
  const el = container?.querySelector('#explore-content')
  if (!el || !state) return

  // Full rebuild needed when: first render, selection changed, or streaming just ended
  const needsFullRebuild = state.renderedCount === 0 || state.selectedIdx != null || !state.streaming

  if (needsFullRebuild) {
    renderTokensFull(el)
    return
  }

  // Incremental: append only new chips to existing flow
  let flow = el.querySelector('.token-flow:not(.token-flow--thinking)')
  if (!flow) {
    // First tokens arriving -- create the section
    const contentSection = createEl('div', { class: 'explore-tokens' })
    flow = createEl('div', { class: 'token-flow' })
    contentSection.append(flow)
    // Insert before streaming cursor if present
    const cursor = el.querySelector('.streaming-cursor')
    if (cursor) el.insertBefore(contentSection, cursor)
    else el.append(contentSection)
  }

  for (let i = state.renderedCount; i < state.tokens.length; i++) {
    flow.append(buildTokenChip(state.tokens[i], i))
  }
  state.renderedCount = state.tokens.length

  // Ensure streaming cursor exists
  if (!el.querySelector('.streaming-cursor')) {
    el.append(createEl('span', { class: 'streaming-cursor' }))
  }
}

function renderTokensFull(el) {
  if (!state) return
  const fragment = document.createDocumentFragment()

  if (state.thinkingTokens.length > 0) {
    const thinkSection = createEl('div', { class: 'explore-thinking' })
    thinkSection.append(createEl('div', { class: 'explore-section-label' }, 'Thinking'))
    const thinkFlow = createEl('div', { class: 'token-flow token-flow--thinking' })
    for (const t of state.thinkingTokens) {
      const chip = createEl('span', { class: 'token-chip token-chip--thinking' })
      chip.textContent = displayToken(t.token)
      thinkFlow.append(chip)
    }
    thinkSection.append(thinkFlow)
    fragment.append(thinkSection)
  }

  if (state.tokens.length > 0) {
    const contentSection = createEl('div', { class: 'explore-tokens' })
    if (state.thinkingTokens.length > 0) {
      contentSection.append(createEl('div', { class: 'explore-section-label' }, 'Content'))
    }
    const flow = createEl('div', { class: 'token-flow' })
    for (let i = 0; i < state.tokens.length; i++) {
      flow.append(buildTokenChip(state.tokens[i], i))
    }
    contentSection.append(flow)
    fragment.append(contentSection)
  }

  if (state.streaming) {
    fragment.append(createEl('span', { class: 'streaming-cursor' }))
  }

  if (state.selectedIdx != null && state.tokens[state.selectedIdx]) {
    fragment.append(buildDetailPanel(state.tokens[state.selectedIdx], state.selectedIdx))
  }

  state.renderedCount = state.tokens.length
  el.replaceChildren(fragment)
}

function buildTokenChip(t, i) {
  const prob = Math.exp(t.logprob)
  const chip = createEl('span', {
    class: `token-chip${i === state.selectedIdx ? ' token-chip--selected' : ''}`,
    'data-idx': String(i),
  })
  chip.style.backgroundColor = probabilityToColor(prob)
  chip.textContent = displayToken(t.token)
  chip.addEventListener('click', () => selectToken(i))
  return chip
}

function buildDetailPanel(tokenData, idx) {
  const panel = createEl('div', { class: 'token-detail' })

  const header = createEl('div', { class: 'token-detail-header' })
  const prob = Math.exp(tokenData.logprob)
  header.append(
    createEl('span', { class: 'token-detail-token' }, displayToken(tokenData.token)),
    createEl('span', { class: 'token-detail-info' }, `logprob: ${tokenData.logprob.toFixed(4)} | prob: ${(prob * 100).toFixed(1)}%`),
    createEl('span', { class: 'token-detail-pos' }, `token ${idx + 1} of ${state.tokens.length}`),
  )
  panel.append(header)

  // Top alternatives
  if (tokenData.top_logprobs.length > 0) {
    const alts = createEl('div', { class: 'token-alternatives' })
    alts.append(createEl('div', { class: 'token-alt-label' }, 'Top alternatives'))
    for (const alt of tokenData.top_logprobs) {
      const altProb = Math.exp(alt.logprob)
      const row = createEl('div', { class: 'token-alt-row' })

      const bar = createEl('div', { class: 'token-alt-bar' })
      bar.style.width = `${Math.max(2, altProb * 100)}%`
      bar.style.backgroundColor = probabilityToBarColor(altProb)

      const text = createEl('span', { class: 'token-alt-text' })
      text.textContent = displayToken(alt.token)
      const info = createEl('span', { class: 'token-alt-info' }, `${(altProb * 100).toFixed(1)}%`)

      row.append(bar, text, info)
      alts.append(row)
    }
    panel.append(alts)
  }

  return panel
}

function selectToken(idx) {
  if (!state) return
  state.selectedIdx = state.selectedIdx === idx ? null : idx
  renderTokens()
}

function handleKeyNav(e) {
  if (!state || !state.tokens.length) return
  if (e.key === 'ArrowRight') {
    e.preventDefault()
    const next = state.selectedIdx == null ? 0 : Math.min(state.selectedIdx + 1, state.tokens.length - 1)
    state.selectedIdx = next
    renderTokens()
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault()
    const prev = state.selectedIdx == null ? state.tokens.length - 1 : Math.max(state.selectedIdx - 1, 0)
    state.selectedIdx = prev
    renderTokens()
  } else if (e.key === 'Escape') {
    if (state.selectedIdx != null) { state.selectedIdx = null; renderTokens() }
  }
}

// -- Helpers --

function displayToken(token) {
  // Make whitespace visible
  return token.replace(/ /g, '\u00B7').replace(/\n/g, '\u21B5').replace(/\t/g, '\u2192')
}

function probabilityToColor(prob) {
  const p = Math.max(0, Math.min(1, prob))
  const hue = p * 120
  return `hsl(${hue}, 65%, ${25 + p * 10}%)`
}

function probabilityToBarColor(prob) {
  const p = Math.max(0, Math.min(1, prob))
  const hue = p * 120
  return `hsl(${hue}, 65%, ${35 + p * 10}%)`
}
