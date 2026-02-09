import { useState, useCallback } from 'react'
import { useComparisonStore } from '../stores/comparisonStore'
import { ChevronDownIcon, TrashIcon } from '../../../components/icons'
import type { ComparisonRun, RunStatus } from '../types'
import clsx from 'clsx'

const statusConfig: Record<RunStatus, { label: string; color: string; bg: string }> = {
  idle: { label: 'Idle', color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-800' },
  running: { label: 'Running', color: 'text-green-600 dark:text-green-400', bg: 'bg-green-100 dark:bg-green-900/30' },
  completed: { label: 'Done', color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-100 dark:bg-blue-900/30' },
  partial: { label: 'Partial', color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/30' },
  error: { label: 'Error', color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/30' },
}

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
  const config = statusConfig[run.status]
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
            <span className={clsx('text-[10px] font-medium px-1.5 py-0.5 rounded-full', config.bg, config.color)}>
              {config.label}
            </span>
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

  const [isCollapsed, setIsCollapsed] = useState(false)

  const handleSelect = useCallback((id: string) => selectRun(id), [selectRun])
  const handleRemove = useCallback((id: string) => removeRun(id), [removeRun])

  if (runs.length === 0) return null

  return (
    <div className="mt-4 border-t border-gray-200 dark:border-gray-700 pt-3">
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="flex items-center gap-1 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
        >
          <ChevronDownIcon className={clsx('w-3 h-3 transition-transform', isCollapsed && '-rotate-90')} />
          History ({runs.length})
        </button>
        {runs.length > 1 && (
          <button
            onClick={clearRuns}
            className="flex items-center gap-1 text-[10px] text-gray-400 hover:text-red-500"
            title="Clear all runs"
          >
            <TrashIcon className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>

      {!isCollapsed && (
        <div className="space-y-1.5">
          {runs.map((run) => (
            <RunCard
              key={run.id}
              run={run}
              isActive={run.id === activeRunId}
              onSelect={handleSelect}
              onRemove={handleRemove}
            />
          ))}
        </div>
      )}
    </div>
  )
}
