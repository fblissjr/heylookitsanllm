// Performance page -- system metrics + performance profile

import * as api from '../api.js'
import { createEl, statCard } from '../utils.js'

let container = null
let state = null
let pollInterval = null

function freshState() {
  return {
    metrics: null,
    profile: null,
    timeRange: '1h',
    error: null,
  }
}

export function mount(el) {
  container = el
  state = freshState()
  buildShell()
  refresh()
  schedulePoll()
  return { teardown }
}

function teardown() {
  if (pollInterval) { clearTimeout(pollInterval); pollInterval = null }
  state = null
  container = null
}

async function schedulePoll() {
  await new Promise(r => { pollInterval = setTimeout(r, 5000) })
  if (!state || !container) return
  await refreshMetrics()
  if (state && container) schedulePoll()
}

function buildShell() {
  const page = createEl('div', { class: 'page-shell' })

  const header = createEl('div', { class: 'page-header' })
  header.append(createEl('h2', { class: 'page-title' }, 'Performance'))
  const refreshBtn = createEl('button', { class: 'btn btn-sm', id: 'refresh-btn' }, 'Refresh')
  refreshBtn.addEventListener('click', refresh)
  header.append(refreshBtn)

  const content = createEl('div', { id: 'perf-content', class: 'page-content' })
  page.append(header, content)
  container.append(page)
}

async function refresh() {
  await Promise.all([refreshMetrics(), refreshProfile()])
}

async function refreshMetrics() {
  if (!state) return
  try {
    state.metrics = await api.getMetrics()
    if (state) renderMetrics()
  } catch (e) {
    if (state) { state.error = e.message; renderMetrics() }
  }
}

async function refreshProfile() {
  if (!state) return
  try {
    state.profile = await api.request('GET', `/v1/performance/profile/${state.timeRange}`)
    if (state) renderProfile()
  } catch {
    // Analytics may not be enabled (503)
    if (state) { state.profile = null; renderProfile() }
  }
}

function renderMetrics() {
  const el = container?.querySelector('#perf-content')
  if (!el) return

  // Build fresh content
  const fragment = document.createDocumentFragment()

  if (state.error && !state.metrics) {
    fragment.append(createEl('p', { class: 'error-text' }, `Failed to load: ${state.error}`))
  }

  if (state.metrics) {
    const sys = state.metrics.system
    const sysSection = createEl('div', { class: 'perf-section' })
    sysSection.append(createEl('h3', { class: 'section-title' }, 'System'))

    const sysCards = createEl('div', { class: 'stat-grid' })
    sysCards.append(
      statCard('RAM Used', `${sys.ram_used_gb.toFixed(1)} GB`),
      statCard('RAM Available', `${sys.ram_available_gb.toFixed(1)} GB`),
      statCard('RAM Total', `${sys.ram_total_gb.toFixed(1)} GB`),
      statCard('CPU', `${sys.cpu_percent.toFixed(0)}%`),
    )
    sysSection.append(sysCards)
    fragment.append(sysSection)

    // Per-model metrics
    const models = state.metrics.models
    if (models && Object.keys(models).length > 0) {
      const modSection = createEl('div', { class: 'perf-section' })
      modSection.append(createEl('h3', { class: 'section-title' }, 'Loaded Models'))

      for (const [modelId, m] of Object.entries(models)) {
        const card = createEl('div', { class: 'model-metrics-card' })
        card.append(createEl('div', { class: 'model-metrics-name' }, modelId))

        const stats = createEl('div', { class: 'stat-grid' })
        stats.append(
          statCard('Memory', `${m.memory_mb.toFixed(0)} MB`),
          statCard('Context', `${m.context_used} / ${m.context_capacity}`),
          statCard('Usage', `${m.context_percent.toFixed(0)}%`),
          statCard('Active', `${m.requests_active}`),
        )
        card.append(stats)
        modSection.append(card)
      }
      fragment.append(modSection)
    }
  }

  // Time range selector + profile section placeholder
  const profileSection = createEl('div', { class: 'perf-section', id: 'profile-section' })
  const profileHeader = createEl('div', { class: 'section-header' })
  profileHeader.append(createEl('h3', { class: 'section-title' }, 'Profile'))

  const rangeSelector = createEl('div', { class: 'range-selector' })
  for (const range of ['1h', '6h', '24h', '7d']) {
    const btn = createEl('button', {
      class: `btn btn-sm ${range === state.timeRange ? 'btn-primary' : ''}`,
    }, range)
    btn.addEventListener('click', () => {
      state.timeRange = range
      refreshProfile()
      renderMetrics()
    })
    rangeSelector.append(btn)
  }
  profileHeader.append(rangeSelector)
  profileSection.append(profileHeader)

  const profileContent = createEl('div', { id: 'profile-content' })
  profileSection.append(profileContent)
  fragment.append(profileSection)

  el.replaceChildren(fragment)
  renderProfile()
}

function renderProfile() {
  const el = container?.querySelector('#profile-content')
  if (!el) return

  if (!state.profile) {
    el.replaceChildren(createEl('p', { class: 'empty-text' }, 'Analytics not available (requires --extra analytics)'))
    return
  }

  const fragment = document.createDocumentFragment()

  // Timing breakdown
  const breakdown = state.profile.timing_breakdown
  if (breakdown?.length > 0) {
    const section = createEl('div', { class: 'profile-subsection' })
    section.append(createEl('h4', {}, 'Timing Breakdown'))
    const table = createEl('div', { class: 'breakdown-table' })
    for (const item of breakdown) {
      const row = createEl('div', { class: 'breakdown-row' })
      row.append(
        createEl('span', { class: 'breakdown-op' }, item.operation),
        createEl('span', { class: 'breakdown-val' }, `${item.avg_time_ms.toFixed(0)}ms avg`),
        createEl('span', { class: 'breakdown-count' }, `${item.count} calls`),
        createEl('span', { class: 'breakdown-pct' }, `${(item.percentage * 100).toFixed(0)}%`),
      )
      table.append(row)
    }
    section.append(table)
    fragment.append(section)
  }

  // Trends
  const trends = state.profile.trends
  if (trends?.length > 0) {
    const section = createEl('div', { class: 'profile-subsection' })
    section.append(createEl('h4', {}, 'Hourly Trends'))
    const table = createEl('div', { class: 'breakdown-table' })
    for (const t of trends.slice(-8)) {
      const hour = new Date(t.hour).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      const row = createEl('div', { class: 'breakdown-row' })
      row.append(
        createEl('span', { class: 'breakdown-op' }, hour),
        createEl('span', { class: 'breakdown-val' }, `${t.response_time_ms.toFixed(0)}ms`),
        createEl('span', { class: 'breakdown-val' }, `${t.tokens_per_second.toFixed(0)} tok/s`),
        createEl('span', { class: 'breakdown-count' }, `${t.requests} req`),
      )
      table.append(row)
    }
    section.append(table)
    fragment.append(section)
  }

  el.replaceChildren(fragment)
}
