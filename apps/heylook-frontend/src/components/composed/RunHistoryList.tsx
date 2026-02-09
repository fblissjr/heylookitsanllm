import { useState } from 'react'
import { ChevronDownIcon, TrashIcon } from '../icons'
import clsx from 'clsx'

interface RunHistoryListProps<T extends { id: string }> {
  runs: T[]
  activeRunId: string | null
  onSelect: (id: string) => void
  onRemove: (id: string) => void
  onClear: () => void
  renderCard: (run: T, isActive: boolean, onSelect: (id: string) => void, onRemove: (id: string) => void) => React.ReactNode
}

export function RunHistoryList<T extends { id: string }>({
  runs,
  activeRunId,
  onSelect,
  onRemove,
  onClear,
  renderCard,
}: RunHistoryListProps<T>) {
  const [isCollapsed, setIsCollapsed] = useState(false)

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
            onClick={onClear}
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
            <div key={run.id}>
              {renderCard(run, run.id === activeRunId, onSelect, onRemove)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
