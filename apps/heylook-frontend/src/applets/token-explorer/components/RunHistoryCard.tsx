import { TrashIcon } from '../../../components/icons'
import type { ExplorerRun, RunStatus } from '../types'
import clsx from 'clsx'

interface RunHistoryCardProps {
  run: ExplorerRun
  isActive: boolean
  onSelect: (id: string) => void
  onRemove: (id: string) => void
}

const statusConfig: Record<RunStatus, { label: string; color: string; bg: string }> = {
  idle: { label: 'Idle', color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-800' },
  streaming: { label: 'Streaming', color: 'text-green-600 dark:text-green-400', bg: 'bg-green-100 dark:bg-green-900/30' },
  completed: { label: 'Done', color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-100 dark:bg-blue-900/30' },
  stopped: { label: 'Stopped', color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/30' },
  error: { label: 'Error', color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/30' },
}

export function RunHistoryCard({ run, isActive, onSelect, onRemove }: RunHistoryCardProps) {
  const config = statusConfig[run.status]

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
            <span className={clsx('text-[10px] font-medium px-1.5 py-0.5 rounded-full', config.bg, config.color)}>
              {config.label}
            </span>
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
