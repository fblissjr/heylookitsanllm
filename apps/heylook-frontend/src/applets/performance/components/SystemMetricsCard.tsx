import type { SystemMetrics } from '../../../stores/systemStore'
import { formatGB } from '../../../utils/formatters'

interface SystemMetricsCardProps {
  metrics: SystemMetrics
}

function thresholdColor(percent: number): string {
  if (percent >= 90) return 'text-red-400'
  if (percent >= 70) return 'text-amber-400'
  return 'text-emerald-400'
}

function thresholdDotColor(percent: number): string {
  if (percent >= 90) return 'bg-red-400'
  if (percent >= 70) return 'bg-amber-400'
  return 'bg-emerald-400'
}

export function SystemMetricsCard({ metrics }: SystemMetricsCardProps) {
  const { system, models } = metrics

  const ramPercent = system.ram_total_gb > 0
    ? (system.ram_used_gb / system.ram_total_gb) * 100
    : 0

  // Aggregate context across all loaded models
  let totalContextUsed = 0
  let totalContextCapacity = 0
  for (const m of Object.values(models)) {
    totalContextUsed += m.context_used
    totalContextCapacity += m.context_capacity
  }
  const contextPercent = totalContextCapacity > 0
    ? (totalContextUsed / totalContextCapacity) * 100
    : 0

  const stats = [
    {
      label: 'RAM',
      value: `${formatGB(system.ram_used_gb)} / ${formatGB(system.ram_total_gb)}`,
      percent: ramPercent,
    },
    {
      label: 'CPU',
      value: `${system.cpu_percent.toFixed(1)}%`,
      percent: system.cpu_percent,
    },
    {
      label: 'Context',
      value: totalContextCapacity > 0
        ? `${totalContextUsed.toLocaleString()} / ${totalContextCapacity.toLocaleString()}`
        : 'No models loaded',
      percent: contextPercent,
    },
  ]

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">System Resources</h3>
      <div className="grid grid-cols-3 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="space-y-1">
            <div className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${thresholdDotColor(stat.percent)}`} />
              <span className="text-xs text-gray-400">{stat.label}</span>
            </div>
            <p className={`text-sm font-medium tabular-nums ${thresholdColor(stat.percent)}`}>
              {stat.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
