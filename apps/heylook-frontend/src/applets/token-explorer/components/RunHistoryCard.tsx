import { TrashIcon } from '../../../components/icons'
import { StatusBadge } from '../../../components/primitives/StatusBadge'
import type { ExplorerRun } from '../types'
import clsx from 'clsx'

interface RunHistoryCardProps {
  run: ExplorerRun
  isActive: boolean
  onSelect: (id: string) => void
  onRemove: (id: string) => void
}

export function RunHistoryCard({ run, isActive, onSelect, onRemove }: RunHistoryCardProps) {
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
            {run.prompt}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <StatusBadge variant={run.status} />
            <span className="text-[10px] text-gray-400">
              {run.tokens.length} tok
            </span>
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
