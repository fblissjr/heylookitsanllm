import { useModelStore, getModelCapabilities } from '../../../stores/modelStore'
import type { Model } from '../../../types/api'
import clsx from 'clsx'

interface ModelMultiSelectProps {
  selectedModelIds: string[]
  onSelectionChange: (ids: string[]) => void
  maxSelection?: number
}

function CapabilityBadge({ label }: { label: string }) {
  return (
    <span className="text-[10px] px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
      {label}
    </span>
  )
}

function providerLabel(model: Model): string {
  if (model.provider === 'mlx') return 'MLX'
  return model.provider || model.owned_by || '?'
}

export function ModelMultiSelect({
  selectedModelIds,
  onSelectionChange,
  maxSelection = 6,
}: ModelMultiSelectProps) {
  const models = useModelStore((s) => s.models)

  // Filter to chat-capable models (or models without capability metadata)
  const chatModels = models.filter((m) => {
    if (!m.capabilities || m.capabilities.length === 0) return true
    return m.capabilities.includes('chat')
  }).sort((a, b) => a.id.localeCompare(b.id))

  const handleToggle = (modelId: string) => {
    if (selectedModelIds.includes(modelId)) {
      onSelectionChange(selectedModelIds.filter((id) => id !== modelId))
    } else if (selectedModelIds.length < maxSelection) {
      onSelectionChange([...selectedModelIds, modelId])
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Models
        </label>
        <span className="text-xs text-gray-400">
          {selectedModelIds.length} / {maxSelection}
        </span>
      </div>

      {selectedModelIds.length > 4 && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          Many models selected -- comparison may take several minutes due to sequential model loading.
        </p>
      )}

      {chatModels.length === 0 ? (
        <p className="text-xs text-gray-400 py-4 text-center">
          No models available. Check server connection.
        </p>
      ) : (
        <div className="space-y-1 max-h-56 overflow-y-auto">
          {chatModels.map((model) => {
            const caps = getModelCapabilities(model)
            const isSelected = selectedModelIds.includes(model.id)
            const isDisabled = !isSelected && selectedModelIds.length >= maxSelection

            return (
              <button
                key={model.id}
                onClick={() => !isDisabled && handleToggle(model.id)}
                disabled={isDisabled}
                className={clsx(
                  'w-full text-left p-2 rounded-lg border transition-colors flex items-center gap-2',
                  isSelected
                    ? 'border-primary bg-primary/5'
                    : isDisabled
                      ? 'border-gray-200 dark:border-gray-800 opacity-40 cursor-not-allowed'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                )}
              >
                <div className={clsx(
                  'w-4 h-4 rounded border-2 flex-shrink-0 flex items-center justify-center',
                  isSelected
                    ? 'border-primary bg-primary'
                    : 'border-gray-300 dark:border-gray-600'
                )}>
                  {isSelected && (
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-900 dark:text-white truncate">
                    {model.id}
                  </p>
                  <div className="flex items-center gap-1 mt-0.5 flex-wrap">
                    <CapabilityBadge label={providerLabel(model)} />
                    {caps.vision && <CapabilityBadge label="vision" />}
                    {caps.thinking && <CapabilityBadge label="thinking" />}
                    {caps.hidden_states && <CapabilityBadge label="hidden" />}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
