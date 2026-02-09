import { usePerformanceStore } from '../stores/performanceStore'
import { SystemMetricsCard } from './SystemMetricsCard'
import { TimingBreakdownCard } from './TimingBreakdownCard'
import { ThroughputCard } from './ThroughputCard'
import { ModelPerformanceCard } from './ModelPerformanceCard'
import { AnalyticsDisabledState } from './AnalyticsDisabledState'

export function MetricsGrid() {
  const systemMetrics = usePerformanceStore((s) => s.systemMetrics)
  const profileData = usePerformanceStore((s) => s.profileData)
  const analyticsEnabled = usePerformanceStore((s) => s.analyticsEnabled)

  return (
    <div className="space-y-4">
      {/* System metrics -- always visible */}
      {systemMetrics && <SystemMetricsCard metrics={systemMetrics} />}

      {/* Analytics cards -- only when enabled */}
      {analyticsEnabled === false ? (
        <div className="flex items-center justify-center py-12">
          <AnalyticsDisabledState />
        </div>
      ) : (
        profileData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <TimingBreakdownCard data={profileData.timing_breakdown} />
            <ThroughputCard trends={profileData.trends} />
            <div className="lg:col-span-2">
              <ModelPerformanceCard bottlenecks={profileData.bottlenecks} />
            </div>
          </div>
        )
      )}
    </div>
  )
}
