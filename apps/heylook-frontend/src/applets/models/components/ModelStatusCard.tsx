import { useModelsStore } from '../stores/modelsStore'
import type { AdminModelConfig } from '../types'

interface Props {
  model: AdminModelConfig
}

export function ModelStatusCard({ model }: Props) {
  const loadModel = useModelsStore((s) => s.loadModel)
  const unloadModel = useModelsStore((s) => s.unloadModel)
  const actionLoading = useModelsStore((s) => s.actionLoading)
  const isLoading = actionLoading === model.id

  return (
    <div className="px-4 pb-3 space-y-2">
      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${
          !model.enabled ? 'bg-gray-500' : model.loaded ? 'bg-green-500' : 'bg-blue-500'
        }`} />
        <span className="text-sm text-gray-700 dark:text-gray-300">
          {!model.enabled ? 'Disabled' : model.loaded ? 'Loaded in memory' : 'Available (not loaded)'}
        </span>
      </div>

      {/* Capabilities */}
      {model.capabilities.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {model.capabilities.map((cap) => (
            <span
              key={cap}
              className="px-1.5 py-0.5 text-[10px] font-medium bg-primary/10 text-primary rounded"
            >
              {cap}
            </span>
          ))}
        </div>
      )}

      {/* Model path */}
      <div>
        <span className="text-[10px] text-gray-400 block">Path</span>
        <span className="text-xs text-gray-500 dark:text-gray-400 break-all">
          {String(model.config.model_path ?? 'unknown')}
        </span>
      </div>

      {/* Load/Unload buttons */}
      {model.enabled && (
        <div className="flex gap-2">
          {model.loaded ? (
            <button
              onClick={() => unloadModel(model.id)}
              disabled={isLoading}
              className="px-3 py-1 text-xs font-medium text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 rounded transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                  Unloading...
                </span>
              ) : 'Unload'}
            </button>
          ) : (
            <button
              onClick={() => loadModel(model.id)}
              disabled={isLoading}
              className="px-3 py-1 text-xs font-medium text-green-400 bg-green-500/10 hover:bg-green-500/20 rounded transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
                  Loading...
                </span>
              ) : 'Load'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
