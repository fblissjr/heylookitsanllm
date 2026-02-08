import type { PerformanceMetrics } from '../../../types/chat'
import { formatDuration, formatTPS, formatTokens, truncateString } from '../../../utils/formatters'

interface MessageMetricsFooterProps {
  performance: PerformanceMetrics
  modelId?: string
  onShowDebug: () => void
}

export function MessageMetricsFooter({
  performance,
  modelId,
  onShowDebug,
}: MessageMetricsFooterProps) {
  const tps = performance.tokensPerSecond
  const tokens = performance.completionTokens
  const ttft = performance.timeToFirstToken
  const cached = performance.cached

  // Check if we have any metrics to show
  const hasMetrics = tps !== undefined || tokens !== undefined || ttft !== undefined

  if (!hasMetrics && !modelId) {
    return null
  }

  return (
    <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 mt-2">
      {/* Desktop layout - single line */}
      <div className="hidden sm:flex items-center gap-2 flex-wrap">
        {tps !== undefined && (
          <>
            <span className="font-mono">{formatTPS(tps)}</span>
            <span className="text-gray-300 dark:text-gray-600">|</span>
          </>
        )}
        {tokens !== undefined && (
          <>
            <span className="font-mono">{formatTokens(tokens)}</span>
            <span className="text-gray-300 dark:text-gray-600">|</span>
          </>
        )}
        {ttft !== undefined && (
          <>
            <span className="font-mono">{formatDuration(ttft)} TTFT</span>
            <span className="text-gray-300 dark:text-gray-600">|</span>
          </>
        )}
        {modelId && (
          <>
            <span className="text-gray-600 dark:text-gray-300" title={modelId}>
              {truncateString(modelId, 16)}
            </span>
            {cached !== undefined && (
              <span className="text-gray-300 dark:text-gray-600">|</span>
            )}
          </>
        )}
        {cached && (
          <span className="text-green-600 dark:text-green-400">Cached</span>
        )}
        <button
          onClick={onShowDebug}
          className="ml-1 p-0.5 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          title="View detailed metrics"
          aria-label="View detailed metrics"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z" />
          </svg>
        </button>
      </div>

      {/* Mobile layout - two lines */}
      <div className="sm:hidden flex flex-col gap-0.5 w-full">
        <div className="flex items-center gap-2">
          {tps !== undefined && (
            <span className="font-mono">{formatTPS(tps)}</span>
          )}
          {tps !== undefined && tokens !== undefined && (
            <span className="text-gray-300 dark:text-gray-600">|</span>
          )}
          {tokens !== undefined && (
            <span className="font-mono">{formatTokens(tokens)}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {modelId && (
            <span className="text-gray-600 dark:text-gray-300" title={modelId}>
              {truncateString(modelId, 12)}
            </span>
          )}
          {modelId && ttft !== undefined && (
            <span className="text-gray-300 dark:text-gray-600">|</span>
          )}
          {ttft !== undefined && (
            <span className="font-mono">{formatDuration(ttft)} TTFT</span>
          )}
          <div className="flex-1" />
          <button
            onClick={onShowDebug}
            className="p-0.5 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            title="View detailed metrics"
            aria-label="View detailed metrics"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
