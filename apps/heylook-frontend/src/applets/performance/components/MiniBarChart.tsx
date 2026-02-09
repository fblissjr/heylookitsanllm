import { formatDuration } from '../../../utils/formatters'

interface BarData {
  label: string
  value: number
  color?: string
}

interface MiniBarChartProps {
  data: BarData[]
  height?: number
  formatValue?: (value: number) => string
}

const DEFAULT_COLOR = '#6366f1'

export function MiniBarChart({
  data,
  height = 24,
  formatValue = formatDuration,
}: MiniBarChartProps) {
  if (data.length === 0) return null

  const maxValue = Math.max(...data.map((d) => d.value), 1)

  return (
    <div className="space-y-2">
      {data.map((item) => {
        const widthPercent = (item.value / maxValue) * 100

        return (
          <div key={item.label} className="flex items-center gap-2">
            <span className="text-xs text-gray-400 w-28 flex-shrink-0 truncate">
              {item.label}
            </span>
            <div className="flex-1 h-full relative">
              <svg width="100%" height={height} className="block">
                <rect
                  x={0}
                  y={2}
                  width="100%"
                  height={height - 4}
                  rx={3}
                  fill="currentColor"
                  className="text-gray-800"
                />
                <rect
                  x={0}
                  y={2}
                  width={`${widthPercent}%`}
                  height={height - 4}
                  rx={3}
                  fill={item.color || DEFAULT_COLOR}
                  opacity={0.8}
                />
              </svg>
            </div>
            <span className="text-xs text-gray-300 w-16 text-right flex-shrink-0 tabular-nums">
              {formatValue(item.value)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
