import { EmptyState } from '../../../components/primitives/EmptyState'
import { ChartBarIcon } from '../../../components/icons'

export function AnalyticsDisabledState() {
  return (
    <EmptyState
      icon={<ChartBarIcon className="w-8 h-8 text-gray-400" />}
      title="Analytics Disabled"
      description="Enable with HEYLOOK_ANALYTICS_ENABLED=true to see timing breakdowns, throughput trends, and per-model performance data."
    />
  )
}
