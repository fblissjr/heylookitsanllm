import type { PerformanceTrend } from '../types'
import { Sparkline } from './Sparkline'
import { formatTPS } from '../../../utils/formatters'

interface ThroughputCardProps {
  trends: PerformanceTrend[]
}

export function ThroughputCard({ trends }: ThroughputCardProps) {
  if (trends.length === 0) return null

  const tpsData = trends.map((t) => t.tokens_per_second)
  const totalRequests = trends.reduce((sum, t) => sum + t.requests, 0)
  const totalErrors = trends.reduce((sum, t) => sum + t.errors, 0)
  const avgTps =
    tpsData.length > 0
      ? tpsData.reduce((sum, v) => sum + v, 0) / tpsData.length
      : 0
  const errorRate = totalRequests > 0 ? (totalErrors / totalRequests) * 100 : 0

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Throughput</h3>
      <div className="mb-3">
        <Sparkline data={tpsData} width={280} height={48} color="#6366f1" />
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <p className="text-xs text-gray-500">Avg TPS</p>
          <p className="text-sm font-medium text-gray-200 tabular-nums">{formatTPS(avgTps)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Requests</p>
          <p className="text-sm font-medium text-gray-200 tabular-nums">{totalRequests.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Error Rate</p>
          <p className={`text-sm font-medium tabular-nums ${errorRate > 5 ? 'text-red-400' : 'text-gray-200'}`}>
            {errorRate.toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  )
}
