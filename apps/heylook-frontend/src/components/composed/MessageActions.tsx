import { CopyIcon, EditIcon, TrashIcon, RefreshIcon, PlayIcon, ForwardIcon } from '../icons'
import { useUIStore } from '../../stores/uiStore'
import clsx from 'clsx'

interface MessageActionsProps {
  role: 'system' | 'user' | 'assistant'
  onCopy?: () => void
  onEdit?: () => void
  onDelete?: () => void
  onRegenerate?: () => void
  onContinue?: () => void
  onNextTurn?: () => void
  isStale?: boolean
  compact?: boolean
}

export function MessageActions({
  role,
  onCopy,
  onEdit,
  onDelete,
  onRegenerate,
  onContinue,
  onNextTurn,
  isStale,
  compact,
}: MessageActionsProps) {
  const { isMobile } = useUIStore()

  const btnClass = clsx(
    'p-1.5 rounded-lg transition-colors',
    compact ? 'p-1' : 'p-1.5',
    'text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800'
  )

  return (
    <div className={clsx(
      'flex items-center gap-1.5 transition-opacity',
      isMobile ? 'opacity-70' : 'opacity-0 group-hover:opacity-100',
      isStale && 'border-l-2 border-amber-400 pl-2'
    )}>
      {onCopy && (
        <button onClick={onCopy} className={btnClass} title="Copy">
          <CopyIcon className={compact ? 'w-3.5 h-3.5' : undefined} />
        </button>
      )}
      {onEdit && (
        <button onClick={onEdit} className={btnClass} title="Edit">
          <EditIcon className={compact ? 'w-3.5 h-3.5' : undefined} />
        </button>
      )}
      {onDelete && (
        <button
          onClick={onDelete}
          className={clsx(btnClass, 'hover:!text-red-500 hover:!bg-red-50 dark:hover:!bg-red-900/20')}
          title="Delete"
        >
          <TrashIcon className={compact ? 'w-3.5 h-3.5' : 'w-4 h-4'} />
        </button>
      )}
      {role === 'assistant' && onRegenerate && (
        <button
          onClick={onRegenerate}
          className={clsx(btnClass, 'hover:!text-primary hover:!bg-primary/10')}
          title="Regenerate"
        >
          <RefreshIcon className={compact ? 'w-3.5 h-3.5' : undefined} />
        </button>
      )}
      {role === 'assistant' && onContinue && (
        <button
          onClick={onContinue}
          className={clsx(btnClass, 'hover:!text-blue-500 hover:!bg-blue-50 dark:hover:!bg-blue-900/20')}
          title="Continue generating"
        >
          <PlayIcon className={compact ? 'w-3.5 h-3.5' : undefined} />
        </button>
      )}
      {role === 'assistant' && onNextTurn && (
        <button
          onClick={onNextTurn}
          className={clsx(btnClass, 'hover:!text-emerald-500 hover:!bg-emerald-50 dark:hover:!bg-emerald-900/20')}
          title="Generate next turn"
        >
          <ForwardIcon className={compact ? 'w-3.5 h-3.5' : undefined} />
        </button>
      )}
    </div>
  )
}
