import { usePerformanceStore } from '../stores/performanceStore'
import { RefreshIcon } from '../../../components/icons'
import type { TimeRange } from '../types'

const TIME_RANGES: { value: TimeRange; label: string }[] = [
  { value: '1h', label: '1 Hour' },
  { value: '6h', label: '6 Hours' },
  { value: '24h', label: '24 Hours' },
  { value: '7d', label: '7 Days' },
]

function formatLastUpdated(timestamp: number | null): string {
  if (!timestamp) return 'Never'
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 5) return 'Just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ago`
}

export function LeftPanel() {
  const timeRange = usePerformanceStore((s) => s.timeRange)
  const setTimeRange = usePerformanceStore((s) => s.setTimeRange)
  const refresh = usePerformanceStore((s) => s.refresh)
  const isPolling = usePerformanceStore((s) => s.isPolling)
  const analyticsEnabled = usePerformanceStore((s) => s.analyticsEnabled)
  const lastUpdated = usePerformanceStore((s) => s.lastUpdated)
  const isLoadingProfile = usePerformanceStore((s) => s.isLoadingProfile)

  return (
    <div className="flex flex-col overflow-hidden h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Performance
        </h1>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
          System metrics and analytics
        </p>
      </div>

      {/* Controls */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Time range selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Time Range
          </label>
          <div className="space-y-1">
            {TIME_RANGES.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setTimeRange(value)}
                className={`w-full text-left px-3 py-1.5 rounded-md text-sm transition-colors ${
                  timeRange === value
                    ? 'bg-primary text-white'
                    : 'text-gray-300 hover:bg-gray-800'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Refresh button */}
        <button
          onClick={() => refresh()}
          disabled={isLoadingProfile}
          className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg border border-gray-600 hover:bg-gray-800 disabled:opacity-40 text-sm text-gray-300 transition-colors"
        >
          <RefreshIcon className={`w-4 h-4 ${isLoadingProfile ? 'animate-spin' : ''}`} />
          Refresh
        </button>

        {/* Status */}
        <div className="space-y-2 text-xs text-gray-500">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              isPolling ? 'bg-emerald-400 animate-pulse' : 'bg-gray-600'
            }`} />
            <span>{isPolling ? 'Auto-refreshing' : 'Paused'}</span>
          </div>

          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              analyticsEnabled === true
                ? 'bg-emerald-400'
                : analyticsEnabled === false
                  ? 'bg-gray-600'
                  : 'bg-amber-400'
            }`} />
            <span>
              Analytics: {
                analyticsEnabled === true
                  ? 'Enabled'
                  : analyticsEnabled === false
                    ? 'Disabled'
                    : 'Unknown'
              }
            </span>
          </div>

          <p className="text-gray-600">
            Updated: {formatLastUpdated(lastUpdated)}
          </p>
        </div>
      </div>
    </div>
  )
}
