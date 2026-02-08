import { useState } from 'react'
import { useSettingsStore } from '../../stores/settingsStore'
import { useUIStore } from '../../stores/uiStore'
import { useModelStore } from '../../stores/modelStore'
import { Slider, Toggle } from '../primitives'
import { CloseIcon, ChevronDownIcon } from '../icons'
import type { Preset, SamplerSettings } from '../../types/settings'
import clsx from 'clsx'

export function SettingsPanel() {
  const {
    samplerSettings,
    updateSamplerSettings,
    resetSamplerToDefaults,
    getAllPresets,
    loadPreset,
    savePreset,
    deletePreset,
    activeSamplerPresetId,
  } = useSettingsStore()
  const { setActivePanel } = useUIStore()
  const { loadedModel } = useModelStore()
  const [isCreatingPreset, setIsCreatingPreset] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  const presets = getAllPresets('sampler')
  const activePreset = presets.find(p => p.id === activeSamplerPresetId)

  // Check if model supports thinking mode
  const supportsThinking = loadedModel?.capabilities?.thinking ?? false

  const handleSelectPreset = (preset: Preset) => {
    loadPreset(preset)
  }

  const handleSavePreset = () => {
    if (newPresetName.trim()) {
      savePreset('sampler', newPresetName.trim())
      setNewPresetName('')
      setIsCreatingPreset(false)
    }
  }

  const handleDeletePreset = (preset: Preset) => {
    if (!preset.isBuiltIn) {
      deletePreset(preset.id)
    }
  }

  const updateSetting = <K extends keyof SamplerSettings>(key: K, value: SamplerSettings[K]) => {
    updateSamplerSettings({ [key]: value })
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Generation Settings
          </h2>
          <button
            onClick={() => setActivePanel(null)}
            className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <CloseIcon />
          </button>
        </div>
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Control how the AI generates responses
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Presets */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Presets
          </label>
          <div className="flex flex-wrap gap-2">
            {presets.map(preset => (
              <button
                key={preset.id}
                onClick={() => handleSelectPreset(preset)}
                className={clsx(
                  'px-3 py-1.5 text-xs rounded-full transition-colors flex items-center gap-1',
                  activePreset?.id === preset.id
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                )}
              >
                {preset.name}
                {!preset.isBuiltIn && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeletePreset(preset)
                    }}
                    className="ml-1 p-0.5 rounded hover:bg-red-500/20"
                    title="Delete preset"
                  >
                    <CloseIcon className="w-3 h-3" />
                  </button>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Core Settings */}
        <div className="space-y-4">
          <Slider
            label="Temperature"
            value={samplerSettings.temperature}
            min={0}
            max={2}
            step={0.1}
            onChange={(v) => updateSetting('temperature', v)}
            description="Higher = more creative, Lower = more focused"
          />

          <Slider
            label="Max Tokens"
            value={samplerSettings.max_tokens}
            min={64}
            max={8192}
            step={64}
            onChange={(v) => updateSetting('max_tokens', v)}
            description="Maximum length of the response"
          />

          <Slider
            label="Top P"
            value={samplerSettings.top_p}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => updateSetting('top_p', v)}
            description="Nucleus sampling threshold"
          />

          <Slider
            label="Top K"
            value={samplerSettings.top_k}
            min={0}
            max={100}
            step={1}
            onChange={(v) => updateSetting('top_k', v)}
            description="Limit choices to top K tokens (0 = disabled)"
          />
        </div>

        {/* Thinking Mode Toggle */}
        {supportsThinking && (
          <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
            <Toggle
              enabled={samplerSettings.enable_thinking ?? false}
              onChange={(v) => updateSetting('enable_thinking', v)}
              label="Thinking Mode"
              description="Show model reasoning process (Qwen3)"
              variant="amber"
            />
          </div>
        )}

        {/* Advanced Settings Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 border-t border-gray-200 dark:border-gray-700"
        >
          <span>Advanced Settings</span>
          <ChevronDownIcon className={clsx('w-4 h-4 transition-transform', showAdvanced && 'rotate-180')} />
        </button>

        {/* Advanced Settings */}
        {showAdvanced && (
          <div className="space-y-4 pt-2">
            <Slider
              label="Min P"
              value={samplerSettings.min_p}
              min={0}
              max={1}
              step={0.01}
              onChange={(v) => updateSetting('min_p', v)}
              description="Minimum probability threshold"
            />

            <Slider
              label="Repetition Penalty"
              value={samplerSettings.repetition_penalty}
              min={1}
              max={2}
              step={0.05}
              onChange={(v) => updateSetting('repetition_penalty', v)}
              description="Penalize repeated tokens"
            />

            <Slider
              label="Repetition Context"
              value={samplerSettings.repetition_context_size}
              min={1}
              max={100}
              step={1}
              onChange={(v) => updateSetting('repetition_context_size', v)}
              description="Tokens to consider for repetition"
            />

            <Slider
              label="Presence Penalty"
              value={samplerSettings.presence_penalty}
              min={0}
              max={2}
              step={0.1}
              onChange={(v) => updateSetting('presence_penalty', v)}
              description="Encourage topic diversity"
            />

            <Slider
              label="Frequency Penalty"
              value={samplerSettings.frequency_penalty}
              min={0}
              max={2}
              step={0.1}
              onChange={(v) => updateSetting('frequency_penalty', v)}
              description="Reduce word repetition"
            />

            {/* Seed */}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Seed
                </label>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {samplerSettings.seed ?? 'Random'}
                </span>
              </div>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={samplerSettings.seed ?? ''}
                  onChange={(e) => updateSetting('seed', e.target.value ? parseInt(e.target.value) : undefined)}
                  placeholder="Random"
                  className="flex-1 px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                />
                <button
                  onClick={() => updateSetting('seed', undefined)}
                  className="px-3 py-1.5 text-xs rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  Clear
                </button>
              </div>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                Set for reproducible outputs
              </p>
            </div>
          </div>
        )}

        {/* Save as Preset */}
        {isCreatingPreset ? (
          <div className="flex gap-2">
            <input
              type="text"
              value={newPresetName}
              onChange={(e) => setNewPresetName(e.target.value)}
              placeholder="Preset name..."
              className="flex-1 px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSavePreset()
                if (e.key === 'Escape') setIsCreatingPreset(false)
              }}
            />
            <button
              onClick={handleSavePreset}
              disabled={!newPresetName.trim()}
              className="px-3 py-2 text-sm rounded-lg bg-primary text-white hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
            <button
              onClick={() => setIsCreatingPreset(false)}
              className="px-3 py-2 text-sm rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setIsCreatingPreset(true)}
            className="w-full px-3 py-2 text-sm rounded-lg border border-dashed border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:border-primary hover:text-primary transition-colors"
          >
            + Save as Preset
          </button>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={resetSamplerToDefaults}
          className="w-full px-4 py-2 rounded-lg text-sm font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          Reset to Defaults
        </button>
      </div>
    </div>
  )
}
