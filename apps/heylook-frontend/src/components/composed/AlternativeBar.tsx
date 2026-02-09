import { probabilityToBarColor } from '../../lib/color'
import { displayToken } from '../../lib/tokens'
import type { TokenLogprob } from '../../types/api'

interface AlternativeBarProps {
  alt: TokenLogprob
  maxProb: number
  isDark: boolean
  /** Size preset: 'default' for token explorer, 'compact' for comparison cards */
  size?: 'default' | 'compact'
}

/**
 * Horizontal bar showing a single alternative token's probability.
 * Used in token detail panels and logprobs detail sections.
 */
export function AlternativeBar({ alt, maxProb, isDark, size = 'default' }: AlternativeBarProps) {
  const prob = Math.exp(alt.logprob)
  const widthPercent = maxProb > 0 ? (prob / maxProb) * 100 : 0

  const tokenWidth = size === 'compact' ? 'w-16' : 'w-20'
  const barHeight = size === 'compact' ? 'h-4' : 'h-5'
  const percentWidth = size === 'compact' ? 'w-14' : 'w-16'

  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={`${tokenWidth} font-mono truncate text-gray-700 dark:text-gray-300 text-right shrink-0`}
        title={alt.token}
      >
        {displayToken(alt.token)}
      </span>
      <div className={`flex-1 ${barHeight} bg-gray-100 dark:bg-gray-800 rounded overflow-hidden`}>
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${Math.max(2, widthPercent)}%`,
            backgroundColor: probabilityToBarColor(prob, isDark),
          }}
        />
      </div>
      <span className={`${percentWidth} text-right font-mono text-gray-500 dark:text-gray-400 shrink-0`}>
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
}
