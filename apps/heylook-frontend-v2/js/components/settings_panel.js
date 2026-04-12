// Settings panel -- collapsible sampler controls
// Reads/writes from shared settings.js. Changes take effect on next generation.

import { getSetting, updateSetting, resetSettings, PARAM_META } from '../settings.js'
import { createEl } from '../utils.js'

/**
 * Build and return a settings panel DOM element.
 * @param {object} [options]
 * @param {string[]} [options.capabilities] - Model capabilities (e.g. ['chat', 'vision', 'thinking'])
 * @param {function} [options.onChange] - Called when any setting changes
 * @returns {HTMLElement}
 */
export function buildSettingsPanel(options = {}) {
  const { capabilities = [], onChange } = options

  const panel = createEl('div', { class: 'settings-panel' })

  // Core section
  const coreSection = createEl('div', { class: 'settings-section' })
  coreSection.append(createEl('div', { class: 'settings-section-title' }, 'Sampler'))

  for (const [key, meta] of Object.entries(PARAM_META)) {
    if (meta.section !== 'core') continue
    coreSection.append(buildControl(key, meta, onChange))
  }
  panel.append(coreSection)

  // Advanced section (collapsible)
  const advDetails = createEl('details', { class: 'settings-section' })
  advDetails.append(createEl('summary', { class: 'settings-section-title settings-section-title--toggle' }, 'Advanced'))

  for (const [key, meta] of Object.entries(PARAM_META)) {
    if (meta.section !== 'advanced') continue
    advDetails.append(buildControl(key, meta, onChange))
  }

  // Thinking toggle -- only if model supports it
  if (capabilities.includes('thinking')) {
    const thinkRow = createEl('div', { class: 'settings-control' })
    const thinkLabel = createEl('label', { class: 'settings-label' }, 'Thinking Mode')
    const thinkCheck = createEl('input', { type: 'checkbox', class: 'settings-checkbox' })
    const currentVal = getSetting('enable_thinking')
    thinkCheck.checked = !!currentVal
    thinkCheck.addEventListener('change', () => {
      updateSetting('enable_thinking', thinkCheck.checked || null)
      onChange?.()
    })
    thinkRow.append(thinkLabel, thinkCheck)
    advDetails.append(thinkRow)
  }

  panel.append(advDetails)

  // Reset button
  const resetRow = createEl('div', { class: 'settings-reset-row' })
  const resetBtn = createEl('button', { class: 'btn btn-sm btn-ghost' }, 'Reset to defaults')
  resetBtn.addEventListener('click', () => {
    resetSettings()
    // Rebuild all controls with fresh values
    refreshControls(panel, onChange, capabilities)
    onChange?.()
  })
  resetRow.append(resetBtn)
  panel.append(resetRow)

  return panel
}

function buildControl(key, meta, onChange) {
  const row = createEl('div', { class: 'settings-control' })
  const labelRow = createEl('div', { class: 'settings-label-row' })

  const label = createEl('label', { class: 'settings-label' }, meta.label)
  const valueDisplay = createEl('span', { class: 'settings-value' })
  labelRow.append(label, valueDisplay)

  const currentVal = getSetting(key)
  const isInt = meta.type === 'int'

  // For seed: use a number input + clear button instead of a slider
  if (key === 'seed') {
    const inputRow = createEl('div', { class: 'settings-seed-row' })
    const input = createEl('input', {
      type: 'number', class: 'form-input form-input--sm',
      min: String(meta.min), max: String(meta.max), step: String(meta.step),
      placeholder: 'Random',
    })
    if (currentVal != null) input.value = currentVal
    valueDisplay.textContent = currentVal != null ? currentVal : 'random'

    const clearBtn = createEl('button', { class: 'btn btn-sm btn-ghost' }, 'Clear')
    clearBtn.addEventListener('click', () => {
      input.value = ''
      updateSetting('seed', null)
      valueDisplay.textContent = 'random'
      onChange?.()
    })

    input.addEventListener('change', () => {
      const v = input.value === '' ? null : parseInt(input.value)
      updateSetting('seed', v)
      valueDisplay.textContent = v != null ? v : 'random'
      onChange?.()
    })

    inputRow.append(input, clearBtn)
    row.append(labelRow, inputRow)
    return row
  }

  // Slider for everything else
  const slider = createEl('input', {
    type: 'range', class: 'settings-slider',
    min: String(meta.min), max: String(meta.max), step: String(meta.step),
  })

  // Use midpoint as visual default when value is null
  const displayVal = currentVal != null ? currentVal : defaultDisplayValue(meta)
  slider.value = displayVal
  valueDisplay.textContent = formatValue(currentVal, meta)

  slider.addEventListener('input', () => {
    const v = isInt ? parseInt(slider.value) : parseFloat(slider.value)
    updateSetting(key, v)
    valueDisplay.textContent = formatValue(v, meta)
    onChange?.()
  })

  row.append(labelRow, slider)
  return row
}

function formatValue(val, meta) {
  if (val == null) return 'default'
  if (meta.type === 'int') return String(Math.round(val))
  return val % 1 === 0 ? val.toFixed(1) : String(val)
}

function defaultDisplayValue(meta) {
  // Visual position for the slider when value is null (not sent to backend)
  return (meta.min + meta.max) / 2
}

function refreshControls(panel, onChange, capabilities) {
  const parent = panel.parentElement
  if (!parent) return
  const newPanel = buildSettingsPanel({ capabilities, onChange })
  parent.replaceChild(newPanel, panel)
}
