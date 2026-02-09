import { useCallback } from 'react'
import { useComparisonStore } from '../stores/comparisonStore'
import { TrashIcon } from '../../../components/icons'
import { StatusBadge } from '../../../components/primitives/StatusBadge'
import { RunHistoryList } from '../../../components/composed/RunHistoryList'
import type { ComparisonRun } from '../types'
import clsx from 'clsx'

function RunCard({
  run,
  isActive,
  onSelect,
  onRemove,
}: {
  run: ComparisonRun
  isActive: boolean
  onSelect: (id: string) => void
  onRemove: (id: string) => void
}) {
  const modelCount = run.selectedModelIds.length
  const promptPreview = run.prompts[0].slice(0, 60)

  return (
    <button
      onClick={() => onSelect(run.id)}
      className={clsx(
        'w-full text-left p-2.5 rounded-lg border transition-colors',
        isActive
          ? 'border-primary bg-primary/5'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-gray-900 dark:text-white truncate font-medium">
            {promptPreview}{run.prompts[0].length > 60 ? '...' : ''}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <StatusBadge variant={run.status} />
            <span className="text-[10px] text-gray-400">
              {modelCount} model{modelCount !== 1 ? 's' : ''}
            </span>
            {run.mode === 'batch' && (
              <span className="text-[10px] text-gray-400">
                {run.prompts.length} prompts
              </span>
            )}
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onRemove(run.id)
          }}
          className="p-1 rounded text-gray-300 dark:text-gray-600 hover:text-red-500 dark:hover:text-red-400 shrink-0"
          title="Remove run"
        >
          <TrashIcon className="w-3 h-3" />
        </button>
      </div>
    </button>
  )
}

export function RunHistory() {
  const runs = useComparisonStore((s) => s.runs)
  const activeRunId = useComparisonStore((s) => s.activeRunId)
  const selectRun = useComparisonStore((s) => s.selectRun)
  const removeRun = useComparisonStore((s) => s.removeRun)
  const clearRuns = useComparisonStore((s) => s.clearRuns)

  const handleSelect = useCallback((id: string) => selectRun(id), [selectRun])
  const handleRemove = useCallback((id: string) => removeRun(id), [removeRun])

  const renderCard = useCallback(
    (run: ComparisonRun, isActive: boolean) => (
      <RunCard
        run={run}
        isActive={isActive}
        onSelect={handleSelect}
        onRemove={handleRemove}
      />
    ),
    [handleSelect, handleRemove]
  )

  return (
    <RunHistoryList
      runs={runs}
      activeRunId={activeRunId}
      onSelect={handleSelect}
      onRemove={handleRemove}
      onClear={clearRuns}
      renderCard={renderCard}
    />
  )
}
