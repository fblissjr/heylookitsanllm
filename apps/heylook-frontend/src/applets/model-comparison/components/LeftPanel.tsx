import { useState, useCallback } from 'react'
import { useComparisonStore } from '../stores/comparisonStore'
import { ModelMultiSelect } from './ModelMultiSelect'
import { BatchPromptInput, parsePrompts } from './BatchPromptInput'
import { RunHistory } from './RunHistory'
import { SamplerControls } from '../../../components/composed/SamplerControls'
import { Slider, Toggle } from '../../../components/primitives'
import type { SamplerSettings } from '../../../types/settings'
import { DEFAULT_SAMPLER_SETTINGS } from '../../../types/settings'

export function LeftPanel() {
  const settings = useComparisonStore((s) => s.settings)
  const updateSettings = useComparisonStore((s) => s.updateSettings)
  const startRun = useComparisonStore((s) => s.startRun)
  const activeRunId = useComparisonStore((s) => s.activeRunId)
  const runs = useComparisonStore((s) => s.runs)
  const stopRun = useComparisonStore((s) => s.stopRun)

  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([])
  const [promptText, setPromptText] = useState('')

  const activeRun = activeRunId ? runs.find((r) => r.id === activeRunId) : null
  const isRunning = activeRun?.status === 'running'

  const isBatch = settings.mode === 'batch'
  const prompts = isBatch ? parsePrompts(promptText) : [promptText.trim()]
  const validPrompts = prompts.filter((p) => p.length > 0)
  const canRun = validPrompts.length > 0 && selectedModelIds.length >= 2 && !isRunning

  const handleRun = useCallback(async () => {
    if (!canRun) return
    await startRun(validPrompts, selectedModelIds)
  }, [canRun, startRun, validPrompts, selectedModelIds])

  const handleStop = useCallback(() => {
    if (activeRunId) stopRun(activeRunId)
  }, [activeRunId, stopRun])

  // Build full SamplerSettings for the SamplerControls component
  const fullSettings: SamplerSettings = {
    ...DEFAULT_SAMPLER_SETTINGS,
    ...settings.samplerSettings,
  }

  const handleSamplerUpdate = useCallback(
    <K extends keyof SamplerSettings>(key: K, value: SamplerSettings[K]) => {
      updateSettings({
        samplerSettings: { ...settings.samplerSettings, [key]: value },
      })
    },
    [updateSettings, settings.samplerSettings]
  )

  return (
    <div className="w-80 flex-shrink-0 border-r border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Model Comparison
        </h1>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
          Run the same prompt against multiple models
        </p>
      </div>

      {/* Scrollable form body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Mode toggle */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-700 dark:text-gray-300">Batch Mode</span>
          <Toggle
            enabled={isBatch}
            onChange={(val) =>
              updateSettings({ mode: val ? 'batch' : 'single' })
            }
          />
        </div>

        {/* Prompt input */}
        {isBatch ? (
          <BatchPromptInput value={promptText} onChange={setPromptText} />
        ) : (
          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Prompt
            </label>
            <textarea
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              rows={4}
              placeholder="Enter your prompt..."
              className="w-full px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 resize-y"
            />
          </div>
        )}

        {/* Model selection */}
        <ModelMultiSelect
          selectedModelIds={selectedModelIds}
          onSelectionChange={setSelectedModelIds}
        />

        {/* Sampler controls */}
        <div className="space-y-1">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Sampler Settings
          </label>
          <SamplerControls settings={fullSettings} onUpdate={handleSamplerUpdate} />
        </div>

        {/* Logprobs toggle */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Token Probabilities
            </span>
            <Toggle
              enabled={settings.enableLogprobs}
              onChange={(val) => updateSettings({ enableLogprobs: val })}
            />
          </div>
          {settings.enableLogprobs && (
            <Slider
              label="Top Logprobs"
              value={settings.topLogprobs}
              min={1}
              max={20}
              step={1}
              onChange={(v) => updateSettings({ topLogprobs: v })}
              description="Number of alternative tokens to show"
            />
          )}
        </div>

        {/* Run / Stop button */}
        {isRunning ? (
          <button
            onClick={handleStop}
            className="w-full py-2 px-4 rounded-lg bg-red-600 hover:bg-red-700 text-white text-sm font-medium transition-colors"
          >
            Stop All
          </button>
        ) : (
          <button
            onClick={handleRun}
            disabled={!canRun}
            className="w-full py-2 px-4 rounded-lg bg-primary hover:bg-primary-hover disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
          >
            {selectedModelIds.length < 2
              ? 'Select at least 2 models'
              : `Compare ${selectedModelIds.length} Models`}
          </button>
        )}

        {/* Run history */}
        <RunHistory />
      </div>
    </div>
  )
}
