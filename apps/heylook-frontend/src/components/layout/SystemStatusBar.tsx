import { useEffect } from 'react'
import { useSystemStore } from '../../stores/systemStore'
import { useModelStore } from '../../stores/modelStore'

interface SystemStatusBarProps {
  className?: string
}

// Color coding thresholds for context usage
function getContextColorClass(percent: number): string {
  if (percent > 90) return 'text-red-500 dark:text-red-400'
  if (percent > 70) return 'text-amber-500 dark:text-amber-400'
  return 'text-gray-500 dark:text-gray-400'
}

// Format bytes to human readable
function formatGB(gb: number): string {
  if (gb >= 10) return `${gb.toFixed(0)}G`
  return `${gb.toFixed(1)}G`
}

// Format large numbers with commas
function formatNumber(n: number): string {
  return n.toLocaleString()
}

export function SystemStatusBar({ className = '' }: SystemStatusBarProps) {
  const { metrics, isPolling, startPolling, stopPolling } = useSystemStore()
  const { loadedModel } = useModelStore()

  // Start/stop polling based on component mount
  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [startPolling, stopPolling])

  // Get metrics for the currently loaded model
  const modelMetrics = loadedModel?.id && metrics?.models
    ? metrics.models[loadedModel.id]
    : null

  const systemMetrics = metrics?.system

  // Don't render if no metrics available yet
  if (!metrics) {
    return (
      <div className={`border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-4 py-1.5 ${className}`}>
        <div className="text-xs text-gray-400 dark:text-gray-500 text-center">
          {isPolling ? 'Loading metrics...' : 'Metrics unavailable'}
        </div>
      </div>
    )
  }

  const contextPercent = modelMetrics?.context_percent ?? 0
  const contextUsed = modelMetrics?.context_used ?? 0
  const contextCapacity = modelMetrics?.context_capacity ?? 0
  const ramUsed = systemMetrics?.ram_used_gb ?? 0
  const cpuPercent = systemMetrics?.cpu_percent ?? 0

  const contextColorClass = getContextColorClass(contextPercent)

  return (
    <div className={`border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-4 py-1.5 ${className}`}>
      {/* Desktop layout - single line */}
      <div className="hidden sm:flex items-center justify-center gap-4 text-xs">
        {modelMetrics ? (
          <span className={contextColorClass}>
            Context: {contextPercent.toFixed(1)}% ({formatNumber(contextUsed)} / {formatNumber(contextCapacity)})
          </span>
        ) : (
          <span className="text-gray-400 dark:text-gray-500">No model loaded</span>
        )}
        <span className="text-gray-300 dark:text-gray-600">|</span>
        <span className="text-gray-500 dark:text-gray-400">
          RAM: {formatGB(ramUsed)}
        </span>
        <span className="text-gray-300 dark:text-gray-600">|</span>
        <span className="text-gray-500 dark:text-gray-400">
          CPU: {cpuPercent.toFixed(0)}%
        </span>
      </div>

      {/* Mobile layout - two lines */}
      <div className="sm:hidden flex flex-col items-center gap-0.5 text-xs">
        {modelMetrics ? (
          <span className={contextColorClass}>
            Context: {contextPercent.toFixed(1)}%
          </span>
        ) : (
          <span className="text-gray-400 dark:text-gray-500">No model</span>
        )}
        <div className="flex items-center gap-3 text-gray-500 dark:text-gray-400">
          <span>RAM: {formatGB(ramUsed)}</span>
          <span>CPU: {cpuPercent.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  )
}
