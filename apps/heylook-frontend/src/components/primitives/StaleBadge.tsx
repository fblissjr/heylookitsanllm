import clsx from 'clsx'

interface StaleBadgeProps {
  className?: string
}

export function StaleBadge({ className }: StaleBadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium',
        'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
        className
      )}
      title="This message was generated before an upstream edit"
    >
      <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
      stale
    </span>
  )
}
