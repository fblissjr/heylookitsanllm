import { useState, useCallback } from 'react'
import { useExplorerStore } from '../stores/explorerStore'
import { RunHistoryCard } from './RunHistoryCard'
import { ChevronDownIcon, TrashIcon } from '../../../components/icons'
import clsx from 'clsx'

export function RunHistory() {
  const runs = useExplorerStore((s) => s.runs)
  const activeRunId = useExplorerStore((s) => s.activeRunId)
  const selectRun = useExplorerStore((s) => s.selectRun)
  const removeRun = useExplorerStore((s) => s.removeRun)
  const clearRuns = useExplorerStore((s) => s.clearRuns)

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
            <RunHistoryCard
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
