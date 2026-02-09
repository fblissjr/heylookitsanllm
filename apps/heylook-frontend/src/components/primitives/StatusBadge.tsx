import clsx from 'clsx'

export type StatusVariant =
  | 'idle'
  | 'pending'
  | 'queued'
  | 'loading'
  | 'streaming'
  | 'running'
  | 'processing'
  | 'completed'
  | 'stopped'
  | 'partial'
  | 'failed'
  | 'error'

const variantStyles: Record<StatusVariant, { color: string; bg: string }> = {
  idle: { color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-800' },
  pending: { color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-800' },
  queued: { color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-100 dark:bg-blue-900/30' },
  loading: { color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/30' },
  streaming: { color: 'text-green-600 dark:text-green-400', bg: 'bg-green-100 dark:bg-green-900/30' },
  running: { color: 'text-green-600 dark:text-green-400', bg: 'bg-green-100 dark:bg-green-900/30' },
  processing: { color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/30' },
  completed: { color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-100 dark:bg-blue-900/30' },
  stopped: { color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/30' },
  partial: { color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/30' },
  failed: { color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/30' },
  error: { color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/30' },
}

const defaultLabels: Record<StatusVariant, string> = {
  idle: 'Idle',
  pending: 'Pending',
  queued: 'Queued',
  loading: 'Loading',
  streaming: 'Streaming',
  running: 'Running',
  processing: 'Processing',
  completed: 'Done',
  stopped: 'Stopped',
  partial: 'Partial',
  failed: 'Failed',
  error: 'Error',
}

interface StatusBadgeProps {
  variant: StatusVariant
  label?: string
  className?: string
}

export function StatusBadge({ variant, label, className }: StatusBadgeProps) {
  const style = variantStyles[variant]
  const displayLabel = label ?? defaultLabels[variant]

  return (
    <span className={clsx('text-[10px] font-medium px-1.5 py-0.5 rounded-full', style.bg, style.color, className)}>
      {displayLabel}
    </span>
  )
}
