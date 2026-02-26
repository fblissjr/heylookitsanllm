import { useState } from 'react'
import { Slider } from '../primitives'
import { ChevronDownIcon } from '../icons'
import type { SamplerSettings } from '../../types/settings'
import clsx from 'clsx'

interface SamplerControlsProps {
  settings: SamplerSettings
  onUpdate: <K extends keyof SamplerSettings>(key: K, value: SamplerSettings[K]) => void
  /** Hide advanced settings toggle -- show all controls flat */
  showAllFlat?: boolean
}

export function SamplerControls({ settings, onUpdate, showAllFlat = false }: SamplerControlsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const showAdvancedSection = showAllFlat || showAdvanced

  return (
    <div className="space-y-4">
      {/* Core Settings */}
      <Slider
        label="Temperature"
        value={settings.temperature}
        min={0}
        max={2}
        step={0.1}
        onChange={(v) => onUpdate('temperature', v)}
        description="Higher = more creative, Lower = more focused"
      />

      <Slider
        label="Max Tokens"
        value={settings.max_tokens}
        min={64}
        max={8192}
        step={64}
        onChange={(v) => onUpdate('max_tokens', v)}
        description="Maximum length of the response"
      />

      <Slider
        label="Top P"
        value={settings.top_p}
        min={0}
        max={1}
        step={0.05}
        onChange={(v) => onUpdate('top_p', v)}
        description="Nucleus sampling threshold"
      />

      <Slider
        label="Top K"
        value={settings.top_k}
        min={0}
        max={100}
        step={1}
        onChange={(v) => onUpdate('top_k', v)}
        description="Limit choices to top K tokens (0 = disabled)"
      />

      {/* Advanced Settings Toggle */}
      {!showAllFlat && (
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 border-t border-gray-200 dark:border-gray-700"
        >
          <span>Advanced Settings</span>
          <ChevronDownIcon className={clsx('w-4 h-4 transition-transform', showAdvanced && 'rotate-180')} />
        </button>
      )}

      {/* Advanced Settings */}
      {showAdvancedSection && (
        <div className="space-y-4 pt-2">
          <Slider
            label="Min P"
            value={settings.min_p}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => onUpdate('min_p', v)}
            description="Minimum probability threshold"
          />

          <Slider
            label="Repetition Penalty"
            value={settings.repetition_penalty}
            min={1}
            max={2}
            step={0.05}
            onChange={(v) => onUpdate('repetition_penalty', v)}
            description="Penalize repeated tokens"
          />

          <Slider
            label="Repetition Context"
            value={settings.repetition_context_size}
            min={1}
            max={100}
            step={1}
            onChange={(v) => onUpdate('repetition_context_size', v)}
            description="Tokens to consider for repetition"
          />

          <Slider
            label="Presence Penalty"
            value={settings.presence_penalty}
            min={0}
            max={2}
            step={0.1}
            onChange={(v) => onUpdate('presence_penalty', v)}
            description="Encourage topic diversity"
          />

          <Slider
            label="Frequency Penalty"
            value={settings.frequency_penalty}
            min={0}
            max={2}
            step={0.1}
            onChange={(v) => onUpdate('frequency_penalty', v)}
            description="Reduce word repetition"
          />

          <Slider
            label="Stream Timeout (seconds)"
            value={(settings.streamTimeoutMs ?? 30000) / 1000}
            min={10}
            max={300}
            step={5}
            onChange={(v) => onUpdate('streamTimeoutMs', v * 1000)}
            description="Seconds before a stalled generation times out"
          />

          {/* Seed */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Seed
              </label>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {settings.seed ?? 'Random'}
              </span>
            </div>
            <div className="flex gap-2">
              <input
                type="number"
                value={settings.seed ?? ''}
                onChange={(e) => onUpdate('seed', e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Random"
                className="flex-1 px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
              />
              <button
                onClick={() => onUpdate('seed', undefined)}
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
    </div>
  )
}
