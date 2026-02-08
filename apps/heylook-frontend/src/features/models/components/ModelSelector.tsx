import { useState, useCallback } from 'react'
import { useModelStore, getModelCapabilities } from '../../../stores/modelStore'
import { useUIStore } from '../../../stores/uiStore'
import { useChatStore } from '../../../applets/chat/stores/chatStore'
import { useSettingsStore } from '../../../stores/settingsStore'
import type { Model } from '../../../types/api'
import clsx from 'clsx'

export function ModelSelector() {
  const { models, loadedModel, setLoadedModel, setModelStatus, setError } = useModelStore()
  const { setActivePanel } = useUIStore()
  const { createConversation } = useChatStore()
  const { systemPrompt } = useSettingsStore()
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [contextWindow, setContextWindow] = useState(4096)
  const [isLoading, setIsLoading] = useState(false)

  const handleSelectModel = useCallback((model: Model) => {
    setSelectedModel(model)
    // Set default context window from model config if available
    const defaultContext = model.context_window || 4096
    setContextWindow(defaultContext)
  }, [])

  const handleLoadModel = useCallback(async () => {
    if (!selectedModel) return

    setIsLoading(true)
    setModelStatus('loading')
    setError(null)

    try {
      // Call the load endpoint (the backend will load the model on first use)
      // For now, we just set the model as loaded since the backend loads on-demand
      const capabilities = getModelCapabilities(selectedModel)

      setLoadedModel({
        id: selectedModel.id,
        contextWindow,
        capabilities,
      })

      // Create a new conversation with this model and current system prompt
      createConversation(selectedModel.id, systemPrompt)

      // Close the panel
      setActivePanel(null)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load model'
      setError(message)
      setModelStatus('error')
    } finally {
      setIsLoading(false)
    }
  }, [selectedModel, contextWindow, setLoadedModel, setModelStatus, setError, createConversation, setActivePanel, systemPrompt])

  const handleUnloadModel = useCallback(() => {
    setLoadedModel(null)
    setSelectedModel(null)
  }, [setLoadedModel])

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Models</h2>
          <button
            onClick={() => setActivePanel(null)}
            className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        {loadedModel && (
          <div className="mt-2 flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Current:</span>
            <span className="text-sm font-medium text-primary">{loadedModel.id}</span>
          </div>
        )}
      </div>

      {/* Model list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {models.length === 0 ? (
          <div className="text-center py-8 text-gray-400 dark:text-gray-500">
            <p>No models available</p>
            <p className="text-sm mt-1">Make sure models.toml is configured</p>
          </div>
        ) : (
          models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              isSelected={selectedModel?.id === model.id}
              isLoaded={loadedModel?.id === model.id}
              onSelect={() => handleSelectModel(model)}
            />
          ))
        )}
      </div>

      {/* Load configuration */}
      {selectedModel && !loadedModel && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-surface-darker">
          <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Load Configuration
          </h3>

          {/* Context window slider */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-600 dark:text-gray-400">
                Context Window
              </label>
              <span className="text-sm font-mono text-primary">{contextWindow.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min={512}
              max={selectedModel.context_window || 32768}
              step={512}
              value={contextWindow}
              onChange={(e) => setContextWindow(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>512</span>
              <span>{(selectedModel.context_window || 32768).toLocaleString()}</span>
            </div>
          </div>

          {/* Load button */}
          <button
            onClick={handleLoadModel}
            disabled={isLoading}
            className={clsx(
              'w-full py-2.5 rounded-lg font-medium transition-all flex items-center justify-center gap-2',
              isLoading
                ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-primary hover:bg-primary-hover text-white shadow-md'
            )}
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Loading...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                <span>Load Model</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Unload button when model is loaded */}
      {loadedModel && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={handleUnloadModel}
            className="w-full py-2 rounded-lg font-medium text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors border border-red-200 dark:border-red-800"
          >
            Unload Model
          </button>
        </div>
      )}
    </div>
  )
}

interface ModelCardProps {
  model: Model
  isSelected: boolean
  isLoaded: boolean
  onSelect: () => void
}

function ModelCard({ model, isSelected, isLoaded, onSelect }: ModelCardProps) {
  const capabilities = getModelCapabilities(model)

  return (
    <button
      onClick={onSelect}
      className={clsx(
        'w-full text-left p-3 rounded-xl border transition-all',
        isLoaded
          ? 'border-accent-green bg-accent-green/10'
          : isSelected
            ? 'border-primary bg-primary/10'
            : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 bg-white dark:bg-surface-dark'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-900 dark:text-white truncate">
              {model.id}
            </span>
            {isLoaded && (
              <span className="shrink-0 text-xs px-1.5 py-0.5 rounded bg-accent-green text-white font-medium">
                Loaded
              </span>
            )}
          </div>
          {model.owned_by && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              {model.owned_by}
            </p>
          )}
        </div>
      </div>

      {/* Capabilities */}
      <div className="flex flex-wrap gap-1.5 mt-2">
        {capabilities.chat && (
          <CapabilityBadge icon="chat" label="Chat" />
        )}
        {capabilities.vision && (
          <CapabilityBadge icon="visibility" label="Vision" color="purple" />
        )}
        {capabilities.thinking && (
          <CapabilityBadge icon="psychology" label="Thinking" color="amber" />
        )}
        {capabilities.embeddings && (
          <CapabilityBadge icon="category" label="Embeddings" color="green" />
        )}
        {capabilities.hidden_states && (
          <CapabilityBadge icon="layers" label="Hidden States" color="red" />
        )}
      </div>
    </button>
  )
}

interface CapabilityBadgeProps {
  icon: string
  label: string
  color?: 'blue' | 'purple' | 'amber' | 'green' | 'red'
}

function CapabilityBadge({ icon, label, color = 'blue' }: CapabilityBadgeProps) {
  const colorClasses = {
    blue: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
    purple: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
    amber: 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400',
    green: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
    red: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
  }

  // Using text for icons since we don't have Material Icons loaded
  const iconMap: Record<string, string> = {
    chat: 'C',
    visibility: 'V',
    psychology: 'T',
    category: 'E',
    layers: 'H',
  }

  return (
    <span className={clsx(
      'inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded',
      colorClasses[color]
    )}>
      <span className="font-bold">{iconMap[icon]}</span>
      {label}
    </span>
  )
}
