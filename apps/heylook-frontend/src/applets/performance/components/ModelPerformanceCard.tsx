import type { ModelBottleneck } from '../types'
import { formatDuration, truncateString } from '../../../utils/formatters'

interface ModelPerformanceCardProps {
  bottlenecks: ModelBottleneck[]
}

export function ModelPerformanceCard({ bottlenecks }: ModelPerformanceCardProps) {
  if (bottlenecks.length === 0) return null

  // Already sorted by avg_total_ms desc from the backend
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Model Performance</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left py-1.5 pr-2 font-medium">Model</th>
              <th className="text-right py-1.5 px-2 font-medium">Avg Response</th>
              <th className="text-right py-1.5 px-2 font-medium">Avg TTFT</th>
              <th className="text-right py-1.5 pl-2 font-medium">Requests</th>
            </tr>
          </thead>
          <tbody>
            {bottlenecks.map((b) => (
              <tr key={b.model} className="border-b border-gray-800 last:border-0">
                <td className="py-1.5 pr-2 text-gray-300" title={b.model}>
                  {truncateString(b.model, 28)}
                </td>
                <td className="py-1.5 px-2 text-right text-gray-200 tabular-nums">
                  {formatDuration(b.avg_total_ms)}
                </td>
                <td className="py-1.5 px-2 text-right text-gray-200 tabular-nums">
                  {formatDuration(b.breakdown.first_token)}
                </td>
                <td className="py-1.5 pl-2 text-right text-gray-200 tabular-nums">
                  {b.request_count.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
