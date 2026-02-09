import clsx from 'clsx'

interface ToggleProps {
  enabled: boolean
  onChange: (enabled: boolean) => void
  label?: string
  description?: string
  variant?: 'default' | 'amber'
  disabled?: boolean
}

export function Toggle({ enabled, onChange, label, description, variant = 'default', disabled }: ToggleProps) {
  const trackColor = enabled
    ? variant === 'amber' ? 'bg-amber-500' : 'bg-primary'
    : 'bg-gray-300 dark:bg-gray-600'

  return (
    <div className="flex items-center justify-between">
      {(label || description) && (
        <div>
          {label && (
            <label className={clsx(
              'text-sm font-medium',
              variant === 'amber'
                ? 'text-amber-800 dark:text-amber-200'
                : 'text-gray-700 dark:text-gray-300'
            )}>
              {label}
            </label>
          )}
          {description && (
            <p className={clsx(
              'text-xs',
              variant === 'amber'
                ? 'text-amber-600 dark:text-amber-400'
                : 'text-gray-400 dark:text-gray-500'
            )}>
              {description}
            </p>
          )}
        </div>
      )}
      <button
        onClick={() => !disabled && onChange(!enabled)}
        disabled={disabled}
        className={clsx(
          'relative w-11 h-6 rounded-full transition-colors',
          trackColor,
          disabled && 'opacity-50 cursor-not-allowed'
        )}
        role="switch"
        aria-checked={enabled}
      >
        <span
          className={clsx(
            'absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform',
            enabled ? 'translate-x-5.5' : 'translate-x-0.5'
          )}
        />
      </button>
    </div>
  )
}
