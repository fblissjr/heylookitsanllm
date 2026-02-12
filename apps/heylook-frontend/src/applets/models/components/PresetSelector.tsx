import { useModelsStore } from '../stores/modelsStore'
import clsx from 'clsx'

export function PresetSelector({ modelId }: { modelId: string }) {
  const profiles = useModelsStore((s) => s.profiles)
  const applyProfile = useModelsStore((s) => s.applyProfile)

  if (profiles.length === 0) {
    return (
      <div className="px-4 pb-3 text-xs text-gray-400">
        Loading profiles...
      </div>
    )
  }

  return (
    <div className="px-4 pb-3">
      <p className="text-xs text-gray-400 mb-2">Apply a preset profile to this model:</p>
      <div className="flex flex-wrap gap-1.5">
        {profiles.map((p) => (
          <button
            key={p.name}
            onClick={() => applyProfile([modelId], p.name)}
            title={p.description}
            className={clsx(
              'px-2.5 py-1 text-xs font-medium rounded-full border transition-colors',
              'border-gray-300 dark:border-gray-600',
              'text-gray-700 dark:text-gray-300',
              'hover:bg-primary/10 hover:border-primary hover:text-primary'
            )}
          >
            {p.name}
          </button>
        ))}
      </div>
    </div>
  )
}
