import { useState, useCallback } from 'react'
import { useModelStore } from '../../../stores/modelStore'
import { useSettingsStore } from '../../../stores/settingsStore'
import { useUIStore } from '../../../stores/uiStore'
import { useBatchStore } from '../stores/batchStore'
import { SamplerControls } from '../../../components/composed/SamplerControls'
import { BoltIcon } from '../../../components/icons'
import { PromptInput } from './PromptInput'
import type { SamplerSettings } from '../../../types/settings'
import { DEFAULT_SAMPLER_SETTINGS } from '../../../types/settings'

export function BatchCreateForm() {
  const { loadedModel, capabilities } = useModelStore()
  const globalSettings = useSettingsStore((s) => s.samplerSettings)
  const setActivePanel = useUIStore((s) => s.setActivePanel)
  const createJob = useBatchStore((s) => s.createJob)
  const setView = useBatchStore((s) => s.setView)
  const jobs = useBatchStore((s) => s.jobs)

  const [prompts, setPrompts] = useState<string[]>([])
  const [localSettings, setLocalSettings] = useState<SamplerSettings>({ ...DEFAULT_SAMPLER_SETTINGS, ...globalSettings })
  const [isSubmitting, setIsSubmitting] = useState(false)

  const batchAvailable = capabilities?.endpoints?.batch_processing?.available ?? false
  const isVLM = loadedModel?.capabilities?.vision ?? false

  const handleUpdateSetting = useCallback(<K extends keyof SamplerSettings>(key: K, value: SamplerSettings[K]) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handleSubmit = useCallback(async () => {
    if (prompts.length === 0 || !loadedModel) return
    setIsSubmitting(true)
    try {
      await createJob(prompts, loadedModel.id, localSettings)
    } finally {
      setIsSubmitting(false)
    }
  }, [prompts, loadedModel, localSettings, createJob])

  const canSubmit = prompts.length > 0 && !!loadedModel && !isSubmitting && !isVLM

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
              Batch Processing
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              Process multiple prompts at once for higher throughput
            </p>
          </div>
          {jobs.length > 0 && (
            <button
              onClick={() => setView('dashboard')}
              className="text-sm px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              View Jobs ({jobs.length})
            </button>
          )}
        </div>
      </div>

      {/* Form */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Batch capability banner */}
        {batchAvailable && (
          <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
            <BoltIcon className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-blue-700 dark:text-blue-300">
                Batch mode available
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-0.5">
                2-4x throughput vs. sequential requests. All prompts processed in a single call.
              </p>
            </div>
          </div>
        )}

        {/* VLM warning */}
        {isVLM && (
          <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
            <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
              Batch mode is text-only
            </p>
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-0.5">
              The loaded model ({loadedModel?.id}) has vision capability. Batch processing does not support VLM models. Load a text-only model to use batch mode.
            </p>
          </div>
        )}

        {/* Prompt Input */}
        <PromptInput prompts={prompts} onPromptsChange={setPrompts} />

        {/* Model label (read-only, opens global selector) */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Model
          </label>
          <button
            onClick={() => setActivePanel('models')}
            className="w-full text-left px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
          >
            {loadedModel ? (
              <span className="text-primary font-medium">{loadedModel.id}</span>
            ) : (
              <span className="text-gray-400">No model loaded -- select one...</span>
            )}
          </button>
        </div>

        {/* Sampler Controls */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Generation Parameters
          </label>
          <SamplerControls
            settings={localSettings}
            onUpdate={handleUpdateSetting}
          />
        </div>

        {/* Estimated info */}
        {prompts.length > 0 && (
          <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Prompts to process</span>
              <span className="font-medium text-gray-900 dark:text-white">{prompts.length}</span>
            </div>
            <div className="flex items-center justify-between text-sm mt-1">
              <span className="text-gray-600 dark:text-gray-400">Max tokens per prompt</span>
              <span className="font-medium text-gray-900 dark:text-white">{localSettings.max_tokens}</span>
            </div>
          </div>
        )}
      </div>

      {/* Submit Footer */}
      <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="w-full py-2.5 rounded-lg font-medium transition-all flex items-center justify-center gap-2 bg-primary hover:bg-primary-hover text-white shadow-md disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {isSubmitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <BoltIcon className="w-4 h-4" />
              <span>Run Batch ({prompts.length} prompt{prompts.length !== 1 ? 's' : ''})</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
