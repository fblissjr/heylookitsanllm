import type { ModelPerformance } from '../types'
import clsx from 'clsx'

interface MetricsBarProps {
  performance: ModelPerformance
}

function ttftColor(ttft?: number): string {
  if (ttft === undefined) return 'text-gray-400'
  if (ttft < 200) return 'text-green-600 dark:text-green-400'
  if (ttft < 500) return 'text-amber-600 dark:text-amber-400'
  return 'text-red-600 dark:text-red-400'
}

function MetricItem({
  label,
  value,
  className,
  tooltip,
}: {
  label: string
  value: string
  className?: string
  tooltip?: string
}) {
  return (
    <div className="text-center" title={tooltip}>
      <div className="text-[10px] text-gray-400 uppercase tracking-wider">{label}</div>
      <div className={clsx('text-xs font-mono font-medium', className || 'text-gray-700 dark:text-gray-300')}>
        {value}
      </div>
    </div>
  )
}

export function MetricsBar({ performance }: MetricsBarProps) {
  const { ttft, tokensPerSecond, totalDuration, completionTokens } = performance

  return (
    <div className="grid grid-cols-4 gap-2 border-t border-gray-100 dark:border-gray-800 pt-2">
      <MetricItem
        label="TTFT"
        value={ttft !== undefined ? `${ttft.toFixed(0)}ms` : '--'}
        className={ttftColor(ttft)}
        tooltip="Time to first token (includes model load time)"
      />
      <MetricItem
        label="tok/s"
        value={tokensPerSecond !== undefined ? tokensPerSecond.toFixed(1) : '--'}
      />
      <MetricItem
        label="Duration"
        value={totalDuration !== undefined ? `${(totalDuration / 1000).toFixed(1)}s` : '--'}
      />
      <MetricItem
        label="Tokens"
        value={completionTokens !== undefined ? String(completionTokens) : '--'}
      />
    </div>
  )
}
