import clsx from 'clsx'

interface StreamingCursorProps {
  /** 'inline' renders as a small bar within text, 'block' as a standalone element */
  variant?: 'inline' | 'block'
  className?: string
}

/**
 * Animated cursor indicator for active streaming/generation.
 * Use 'inline' within token streams, 'block' for standalone loading indicators.
 */
export function StreamingCursor({ variant = 'inline', className }: StreamingCursorProps) {
  return (
    <span
      className={clsx(
        'inline-block bg-primary animate-pulse rounded-sm',
        variant === 'inline' ? 'w-2 h-4 mx-1 align-middle' : 'w-3 h-5',
        className
      )}
    />
  )
}
