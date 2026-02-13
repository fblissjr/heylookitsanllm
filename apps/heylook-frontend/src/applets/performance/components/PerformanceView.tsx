import { useEffect } from 'react'
import { usePerformanceStore } from '../stores/performanceStore'
import { LeftPanel } from './LeftPanel'
import { MetricsGrid } from './MetricsGrid'
import { AppletLayout } from '../../../components/layout/AppletLayout'

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
    <AppletLayout leftPanel={<LeftPanel />} leftPanelWidth="w-64">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-6xl mx-auto">
          <MetricsGrid />
        </div>
      </div>
    </AppletLayout>
  )
}
