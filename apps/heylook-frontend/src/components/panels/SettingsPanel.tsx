import { useState } from 'react'
import { useSettingsStore } from '../../stores/settingsStore'
import { useUIStore } from '../../stores/uiStore'
import { useModelStore } from '../../stores/modelStore'
import { Toggle, Slider } from '../primitives'
import { CloseIcon } from '../icons'
import { SamplerControls } from '../composed/SamplerControls'
import { logger } from '../../lib/diagnostics'
import type { LogLevel } from '../../lib/diagnostics'
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
    streamTimeoutMs,
    setStreamTimeoutMs,
    logLevel,
    setLogLevel,
  } = useSettingsStore()
  const { setActivePanel } = useUIStore()
  const { loadedModel } = useModelStore()
  const [isCreatingPreset, setIsCreatingPreset] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')

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

  const handleUpdateSetting = <K extends keyof SamplerSettings>(key: K, value: SamplerSettings[K]) => {
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
              <div key={preset.id} className="flex items-center gap-0.5">
                <button
                  onClick={() => handleSelectPreset(preset)}
                  className={clsx(
                    'px-3 py-1.5 text-xs rounded-full transition-colors',
                    !preset.isBuiltIn && 'rounded-r-md',
                    activePreset?.id === preset.id
                      ? 'bg-primary text-white'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  )}
                >
                  {preset.name}
                </button>
                {!preset.isBuiltIn && (
                  <button
                    onClick={() => handleDeletePreset(preset)}
                    className={clsx(
                      'p-1 text-xs rounded-full transition-colors',
                      activePreset?.id === preset.id
                        ? 'bg-primary text-white/70 hover:text-white'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-400 hover:text-red-500 hover:bg-gray-200 dark:hover:bg-gray-700'
                    )}
                    title="Delete preset"
                    aria-label={`Delete preset ${preset.name}`}
                  >
                    <CloseIcon className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Sampler Controls (shared component) */}
        <SamplerControls
          settings={samplerSettings}
          onUpdate={handleUpdateSetting}
        />

        {/* Stream Timeout */}
        <Slider
          label="Stream Timeout (seconds)"
          value={streamTimeoutMs / 1000}
          min={10}
          max={300}
          step={5}
          onChange={(v) => setStreamTimeoutMs(v * 1000)}
          description="Seconds before a stalled generation times out"
        />

        {/* Thinking Mode Toggle */}
        {supportsThinking && (
          <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
            <Toggle
              enabled={samplerSettings.enable_thinking ?? false}
              onChange={(v) => handleUpdateSetting('enable_thinking', v)}
              label="Thinking Mode"
              description="Show model reasoning process (Qwen3)"
              variant="amber"
            />
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
      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 space-y-2">
        {/* Log Level */}
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-500 dark:text-gray-400">Diagnostic Log Level</label>
          <select
            value={logLevel}
            onChange={(e) => setLogLevel(e.target.value as LogLevel)}
            className="text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300"
          >
            <option value="error">Error</option>
            <option value="warn">Warn</option>
            <option value="info">Info</option>
            <option value="debug">Debug</option>
          </select>
        </div>

        {/* Download Diagnostic Log */}
        <button
          onClick={() => logger.downloadAsJSONL()}
          className="w-full px-4 py-2 rounded-lg text-sm font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          Download Diagnostic Log
        </button>

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
