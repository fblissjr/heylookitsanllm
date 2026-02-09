import { useTheme } from '../../../contexts/ThemeContext'
import { AlternativeBar } from '../../../components/composed/AlternativeBar'
import type { ExplorerToken } from '../types'

interface TokenDetailProps {
  token: ExplorerToken
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
