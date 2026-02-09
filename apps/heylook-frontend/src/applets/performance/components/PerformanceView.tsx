import { useEffect } from 'react'
import { usePerformanceStore } from '../stores/performanceStore'
import { LeftPanel } from './LeftPanel'
import { MetricsGrid } from './MetricsGrid'

export function PerformanceView() {
  const startPolling = usePerformanceStore((s) => s.startPolling)
  const stopPolling = usePerformanceStore((s) => s.stopPolling)

  useEffect(() => {
    startPolling()
    return () => {
      stopPolling()
    }
  }, [startPolling, stopPolling])

  return (
    <div className="h-full flex">
      <LeftPanel />
      <div className="flex-1 overflow-y-auto p-6">
        <MetricsGrid />
      </div>
    </div>
  )
}
