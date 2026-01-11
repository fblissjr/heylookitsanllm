import { useState, useRef, useEffect } from 'react'
import type { PerformanceMetrics } from '../../../types/chat'

interface PerformanceInfoProps {
  performance?: PerformanceMetrics
  rawStream?: string[]
  onShowRawStream?: () => void
}

export function PerformanceInfo({ performance, rawStream, onShowRawStream }: PerformanceInfoProps) {
  const [isOpen, setIsOpen] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  // Close popover when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  if (!performance) return null

  const formatDuration = (ms?: number) => {
    if (ms === undefined) return '-'
    if (ms < 1000) return `${Math.round(ms)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  const formatTokensPerSec = (tps?: number) => {
    if (tps === undefined) return '-'
    return `${tps.toFixed(1)} tok/s`
  }

  return (
    <div className="relative inline-block">
      <button
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        className="p-1 rounded-full text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        title="Performance metrics"
        aria-label="View performance metrics"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>

      {isOpen && (
        <div
          ref={popoverRef}
          className="absolute right-0 top-full mt-1 z-50 w-64 bg-white dark:bg-surface-dark rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-3"
        >
          <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Performance Metrics
          </h4>

          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Time to first token:</span>
              <span className="text-gray-700 dark:text-gray-200 font-mono">
                {formatDuration(performance.timeToFirstToken)}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Generation speed:</span>
              <span className="text-gray-700 dark:text-gray-200 font-mono">
                {formatTokensPerSec(performance.tokensPerSecond)}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Total duration:</span>
              <span className="text-gray-700 dark:text-gray-200 font-mono">
                {formatDuration(performance.totalDuration)}
              </span>
            </div>

            {performance.promptTokens !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Prompt tokens:</span>
                <span className="text-gray-700 dark:text-gray-200 font-mono">
                  {performance.promptTokens.toLocaleString()}
                </span>
              </div>
            )}

            {performance.completionTokens !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Completion tokens:</span>
                <span className="text-gray-700 dark:text-gray-200 font-mono">
                  {performance.completionTokens.toLocaleString()}
                </span>
              </div>
            )}

            {performance.cached !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Prefix cached:</span>
                <span className={`font-mono ${performance.cached ? 'text-green-600 dark:text-green-400' : 'text-gray-700 dark:text-gray-200'}`}>
                  {performance.cached ? 'Yes' : 'No'}
                </span>
              </div>
            )}
          </div>

          {rawStream && rawStream.length > 0 && onShowRawStream && (
            <button
              onClick={() => {
                setIsOpen(false)
                onShowRawStream()
              }}
              className="mt-3 w-full text-xs text-primary hover:text-primary-hover transition-colors text-left"
            >
              View raw stream ({rawStream.length} events)
            </button>
          )}
        </div>
      )}
    </div>
  )
}
