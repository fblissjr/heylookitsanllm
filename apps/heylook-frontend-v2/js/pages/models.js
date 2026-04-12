// Models page -- list, load/unload, scan, import

import * as api from '../api.js'
import { createEl } from '../utils.js'

let container = null
let state = null

function freshState() {
  return {
    models: [],
    scanResults: null,
    scanning: false,
    importing: false,
    loading: new Set(),
  }
}

export function mount(el) {
  container = el
  state = freshState()
  buildShell()
  loadModels()
  return { teardown }
}

function teardown() {
  state = null
  container = null
}

function buildShell() {
  const page = document.createElement('div')
  page.className = 'page-shell'

  const header = document.createElement('div')
  header.className = 'page-header'

  const title = document.createElement('h2')
  title.className = 'page-title'
  title.textContent = 'Models'

  const scanBtn = createEl('button', { class: 'btn btn-sm', id: 'scan-btn' }, 'Scan')
  scanBtn.addEventListener('click', handleScan)
  header.append(title, scanBtn)

  const content = createEl('div', { id: 'models-content', class: 'page-content' })
  page.append(header, content)
  container.append(page)
}

async function loadModels() {
  try {
    const data = await api.listAdminModels()
    if (!state) return
    state.models = data.models || []
    render()
  } catch (e) {
    const el = container?.querySelector('#models-content')
    if (el) el.replaceChildren(createEl('p', { class: 'error-text' }, `Failed to load models: ${e.message}`))
  }
}

function render() {
  const el = container?.querySelector('#models-content')
  if (!el) return

  const fragment = document.createDocumentFragment()

  // Scan results panel
  if (state.scanResults) {
    fragment.append(buildScanPanel())
  }

  // Model list
  if (state.models.length === 0) {
    const empty = createEl('p', { class: 'empty-text' }, 'No models configured. Click Scan to find models.')
    fragment.append(empty)
  } else {
    for (const model of state.models) {
      fragment.append(buildModelCard(model))
    }
  }

  // Data management section
  const dataSection = createEl('div', { class: 'perf-section', style: 'margin-top:24px' })
  dataSection.append(createEl('h3', { class: 'section-title' }, 'Data'))
  const dataInfo = createEl('p', { class: 'empty-text', style: 'margin-bottom:8px' },
    'Conversations and notebooks are stored on the server in SQLite (data/conversations.db).')
  const clearBtn = createEl('button', { class: 'btn btn-sm btn-danger' }, 'Clear all conversations & notebooks')
  clearBtn.addEventListener('click', handleClearData)
  dataSection.append(dataInfo, clearBtn)
  fragment.append(dataSection)

  el.replaceChildren(fragment)
}

function buildModelCard(model) {
  const card = createEl('div', { class: 'model-card' })
  const isLoading = state.loading.has(model.id)

  const headerRow = createEl('div', { class: 'model-card-header' })
  const name = createEl('span', { class: 'model-card-name' }, model.id)
  const badge = createEl('span', {
    class: `badge ${model.loaded ? 'badge--success' : 'badge--dim'}`,
  }, model.loaded ? 'Loaded' : 'Idle')
  headerRow.append(name, badge)

  const meta = createEl('div', { class: 'model-card-meta' })
  const provider = createEl('span', { class: 'meta-tag' }, model.provider)
  meta.append(provider)
  if (model.capabilities.length > 0) {
    for (const cap of model.capabilities) {
      meta.append(createEl('span', { class: 'meta-tag meta-tag--accent' }, cap))
    }
  }
  if (!model.enabled) {
    meta.append(createEl('span', { class: 'meta-tag meta-tag--danger' }, 'disabled'))
  }

  const actions = createEl('div', { class: 'model-card-actions' })
  if (model.loaded) {
    const unloadBtn = createEl('button', { class: 'btn btn-sm btn-danger' }, isLoading ? 'Unloading...' : 'Unload')
    unloadBtn.disabled = isLoading
    unloadBtn.addEventListener('click', () => handleToggleLoad(model.id, false))
    actions.append(unloadBtn)
  } else if (model.enabled) {
    const loadBtn = createEl('button', { class: 'btn btn-sm btn-primary' }, isLoading ? 'Loading...' : 'Load')
    loadBtn.disabled = isLoading
    loadBtn.addEventListener('click', () => handleToggleLoad(model.id, true))
    actions.append(loadBtn)
  }

  if (model.description) {
    const desc = createEl('div', { class: 'model-card-desc' }, model.description)
    card.append(headerRow, meta, desc, actions)
  } else {
    card.append(headerRow, meta, actions)
  }

  return card
}

function buildScanPanel() {
  const panel = createEl('div', { class: 'scan-panel' })

  const header = createEl('div', { class: 'scan-panel-header' })
  header.append(
    createEl('h3', {}, `Found ${state.scanResults.length} model(s)`),
  )

  const closeBtn = createEl('button', { class: 'btn btn-sm btn-ghost' }, 'Close')
  closeBtn.addEventListener('click', () => { state.scanResults = null; render() })
  header.append(closeBtn)

  panel.append(header)

  const importable = state.scanResults.filter(r => !r.already_configured)
  if (importable.length === 0) {
    panel.append(createEl('p', { class: 'empty-text' }, 'All found models are already configured.'))
    return panel
  }

  for (const result of importable) {
    const row = createEl('div', { class: 'scan-result' })
    const info = createEl('div', { class: 'scan-result-info' })
    info.append(
      createEl('span', { class: 'scan-result-name' }, result.id),
      createEl('span', { class: 'meta-tag' }, `${result.size_gb.toFixed(1)} GB`),
    )
    if (result.vision) info.append(createEl('span', { class: 'meta-tag meta-tag--accent' }, 'vision'))
    if (result.quantization) info.append(createEl('span', { class: 'meta-tag' }, result.quantization))

    const importBtn = createEl('button', { class: 'btn btn-sm btn-primary' }, 'Import')
    importBtn.addEventListener('click', () => handleImport(result))
    row.append(info, importBtn)
    panel.append(row)
  }

  return panel
}

async function handleToggleLoad(modelId, load) {
  state.loading.add(modelId)
  render()
  try {
    if (load) {
      await api.loadModel(modelId)
    } else {
      await api.unloadModel(modelId)
    }
    await loadModels()
  } catch (e) {
    console.error('Load/unload failed:', e)
  } finally {
    state.loading.delete(modelId)
    render()
  }
}

async function handleScan() {
  const btn = container?.querySelector('#scan-btn')
  if (!btn || state.scanning) return
  state.scanning = true
  btn.textContent = 'Scanning...'
  btn.disabled = true

  try {
    const data = await api.scanModels({ scan_hf_cache: true })
    state.scanResults = data.models || []
  } catch (e) {
    console.error('Scan failed:', e)
  } finally {
    state.scanning = false
    if (btn) { btn.textContent = 'Scan'; btn.disabled = false }
    render()
  }
}

async function handleClearData() {
  if (!state) return
  if (!confirm('This will permanently delete ALL conversations and notebooks. Are you sure?')) return
  try {
    const result = await api.clearAllData()
    if (!state) return
    alert(`Cleared ${result.conversations_deleted} conversations and ${result.notebooks_deleted} notebooks.`)
  } catch (e) {
    console.error('Clear data failed:', e)
  }
}

async function handleImport(result) {
  if (state.importing) return
  state.importing = true
  render()
  try {
    await api.importModels({
      models: [{ id: result.id, path: result.path, provider: result.provider }],
    })
    // Remove from scan results and refresh model list
    state.scanResults = state.scanResults.filter(r => r.id !== result.id)
    await loadModels()
  } catch (e) {
    console.error('Import failed:', e)
  } finally {
    state.importing = false
    render()
  }
}
