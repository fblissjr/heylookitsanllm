import { useTheme } from '../../../contexts/ThemeContext'
import type { ExplorerToken } from '../types'
import type { TokenLogprob } from '../../../types/api'

interface TokenDetailProps {
  token: ExplorerToken
}

function probabilityToBarColor(probability: number, isDark: boolean): string {
  const p = Math.max(0, Math.min(1, probability))
  const hue = p * 120
  const saturation = 65
  const lightness = isDark ? 35 + p * 10 : 75 - p * 15
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

function AlternativeBar({ alt, maxProb, isDark }: { alt: TokenLogprob; maxProb: number; isDark: boolean }) {
  const prob = Math.exp(alt.logprob)
  const widthPercent = maxProb > 0 ? (prob / maxProb) * 100 : 0

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 font-mono truncate text-gray-700 dark:text-gray-300 text-right shrink-0" title={alt.token}>
        {alt.token === ' ' ? '\u00B7' : alt.token === '\n' ? '\u21B5' : alt.token}
      </span>
      <div className="flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${Math.max(2, widthPercent)}%`,
            backgroundColor: probabilityToBarColor(prob, isDark),
          }}
        />
      </div>
      <span className="w-16 text-right font-mono text-gray-500 dark:text-gray-400 shrink-0">
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
}

export function TokenDetail({ token }: TokenDetailProps) {
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === 'dark'

  const hasAlternatives = token.topLogprobs.length > 0
  const maxProb = hasAlternatives
    ? Math.max(...token.topLogprobs.map((a) => Math.exp(a.logprob)))
    : 0

  // Find selected token's rank among alternatives
  const rank = token.topLogprobs.findIndex((a) => a.token_id === token.tokenId)

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 p-4 space-y-3">
      {/* Selected token info */}
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs text-gray-500 dark:text-gray-400">Token #{token.index}</span>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="font-mono text-sm font-medium text-gray-900 dark:text-white">
              "{token.token}"
            </span>
            {rank >= 0 && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
                Rank #{rank + 1}
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <span className="text-xs text-gray-500 dark:text-gray-400">Probability</span>
          <div className="text-sm font-mono font-medium text-gray-900 dark:text-white">
            {(token.probability * 100).toFixed(2)}%
          </div>
          <div className="text-xs font-mono text-gray-400">
            logprob: {token.logprob.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Alternative tokens */}
      {hasAlternatives ? (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-gray-600 dark:text-gray-400">
            Top {token.topLogprobs.length} alternatives
          </p>
          {token.topLogprobs.map((alt, i) => (
            <AlternativeBar key={i} alt={alt} maxProb={maxProb} isDark={isDark} />
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-400 dark:text-gray-500">
          No alternatives available. Set Top Logprobs {'>'} 0 to see alternative tokens.
        </p>
      )}
    </div>
  )
}
