import type { TimingOperation } from '../types'
import { MiniBarChart } from './MiniBarChart'

interface TimingBreakdownCardProps {
  data: TimingOperation[]
}

const OPERATION_LABELS: Record<string, string> = {
  queue: 'Queue',
  model_load: 'Model Load',
  image_processing: 'Image Processing',
  token_generation: 'Token Generation',
  other: 'Other',
}

const OPERATION_COLORS: Record<string, string> = {
  queue: '#f59e0b',
  model_load: '#8b5cf6',
  image_processing: '#06b6d4',
  token_generation: '#6366f1',
  other: '#6b7280',
}

export function TimingBreakdownCard({ data }: TimingBreakdownCardProps) {
  if (data.length === 0) return null

  const barData = data.map((item) => ({
    label: OPERATION_LABELS[item.operation] || item.operation,
    value: item.avg_time_ms,
    color: OPERATION_COLORS[item.operation] || OPERATION_COLORS.other,
  }))

  const totalRequests = data.reduce((sum, d) => sum + d.count, 0)

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Timing Breakdown</h3>
        <span className="text-xs text-gray-500">{totalRequests.toLocaleString()} requests</span>
      </div>
      <MiniBarChart data={barData} />
    </div>
  )
}
