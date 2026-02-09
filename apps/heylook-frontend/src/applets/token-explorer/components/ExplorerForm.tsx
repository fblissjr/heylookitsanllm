import { useState, useCallback } from 'react'
import { useModelStore } from '../../../stores/modelStore'
import { useExplorerStore } from '../stores/explorerStore'
import { Slider } from '../../../components/primitives'
import { SparklesIcon, StopIcon } from '../../../components/icons'

export function ExplorerForm() {
  const models = useModelStore((s) => s.models)
  const loadedModel = useModelStore((s) => s.loadedModel)
  const activeRunId = useExplorerStore((s) => s.activeRunId)
  const runs = useExplorerStore((s) => s.runs)
  const startRun = useExplorerStore((s) => s.startRun)
  const stopRun = useExplorerStore((s) => s.stopRun)

  const activeRun = activeRunId ? runs.find((r) => r.id === activeRunId) : null
  const isStreaming = activeRun?.status === 'streaming'

  const [prompt, setPrompt] = useState('')
  const [selectedModelId, setSelectedModelId] = useState(loadedModel?.id || '')
  const [topLogprobs, setTopLogprobs] = useState(5)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)

  const canSubmit = prompt.trim().length > 0 && selectedModelId && !isStreaming

  const handleSubmit = useCallback(() => {
    if (!canSubmit) return
    startRun(prompt.trim(), selectedModelId, topLogprobs, temperature, maxTokens)
  }, [canSubmit, prompt, selectedModelId, topLogprobs, temperature, maxTokens, startRun])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canSubmit) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [canSubmit, handleSubmit],
  )

  return (
    <div className="space-y-4">
      {/* Model */}
      <div>
        <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
          Model
        </label>
        <select
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={isStreaming}
          className="w-full px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 disabled:opacity-50"
        >
          <option value="">Select model...</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}{m.id === loadedModel?.id ? ' (loaded)' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Prompt */}
      <div>
        <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
          Prompt
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter a prompt to explore token probabilities..."
          rows={3}
          disabled={isStreaming}
          className="w-full px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-600 resize-y disabled:opacity-50"
        />
      </div>

      {/* Parameters */}
      <div className="space-y-3">
        <Slider
          label="Top Logprobs"
          value={topLogprobs}
          min={0}
          max={20}
          step={1}
          onChange={setTopLogprobs}
          description="Number of alternative tokens to show (0 = none)"
        />
        <Slider
          label="Temperature"
          value={temperature}
          min={0}
          max={2}
          step={0.1}
          onChange={setTemperature}
          description="Higher = more variation"
        />
        <Slider
          label="Max Tokens"
          value={maxTokens}
          min={64}
          max={4096}
          step={64}
          onChange={setMaxTokens}
          description="Maximum generation length"
        />
      </div>

      {/* Submit / Stop */}
      {isStreaming ? (
        <button
          onClick={stopRun}
          className="w-full py-2 rounded-lg font-medium text-sm flex items-center justify-center gap-2 bg-red-500 hover:bg-red-600 text-white transition-colors"
        >
          <StopIcon className="w-4 h-4" />
          Stop
        </button>
      ) : (
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="w-full py-2 rounded-lg font-medium text-sm flex items-center justify-center gap-2 bg-primary hover:bg-primary-hover text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <SparklesIcon className="w-4 h-4" />
          Explore Tokens
        </button>
      )}
    </div>
  )
}
