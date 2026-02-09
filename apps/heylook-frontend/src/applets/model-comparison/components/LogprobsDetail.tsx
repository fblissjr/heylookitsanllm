import { useState, useCallback } from 'react'
import { useTheme } from '../../../contexts/ThemeContext'
import { probabilityToColor } from '../../../lib/color'
import { displayToken } from '../../../lib/tokens'
import { AlternativeBar } from '../../../components/composed/AlternativeBar'
import type { ComparisonToken } from '../types'
import { ChevronDownIcon } from '../../../components/icons'
import clsx from 'clsx'

interface LogprobsDetailProps {
  tokens: ComparisonToken[]
}

export function LogprobsDetail({ tokens }: LogprobsDetailProps) {
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === 'dark'

  const [isOpen, setIsOpen] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)

  const handleTokenClick = useCallback((index: number) => {
    setSelectedIndex((prev) => (prev === index ? null : index))
  }, [])

  const selectedToken = selectedIndex !== null ? tokens[selectedIndex] ?? null : null

  return (
    <div className="border-t border-gray-100 dark:border-gray-800">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 py-1.5 text-[10px] text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
      >
        <ChevronDownIcon className={clsx('w-3 h-3 transition-transform', isOpen && 'rotate-180')} />
        Token probabilities ({tokens.length})
      </button>

      {isOpen && (
        <div className="space-y-2">
          {/* Token chips */}
          <div className="flex flex-wrap gap-px">
            {tokens.map((token) => (
              <span
                key={token.index}
                role="button"
                tabIndex={0}
                onClick={() => handleTokenClick(token.index)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    handleTokenClick(token.index)
                  }
                }}
                title={`"${token.token}" | ${(token.probability * 100).toFixed(1)}%`}
                className={clsx(
                  'inline-block px-0.5 py-0.5 rounded text-[10px] font-mono cursor-pointer transition-shadow',
                  selectedIndex === token.index
                    ? 'ring-2 ring-primary shadow-md'
                    : 'hover:ring-1 hover:ring-gray-400 dark:hover:ring-gray-500'
                )}
                style={{
                  backgroundColor: probabilityToColor(token.probability, isDark),
                  color: isDark ? '#e5e7eb' : '#1f2937',
                }}
              >
                {displayToken(token.token)}
              </span>
            ))}
          </div>

          {/* Selected token detail */}
          {selectedToken && (
            <div className="border-t border-gray-100 dark:border-gray-800 pt-2 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono text-gray-900 dark:text-white">
                  "{selectedToken.token}"
                </span>
                <span className="text-gray-400">
                  {(selectedToken.probability * 100).toFixed(1)}%
                </span>
              </div>
              {selectedToken.topLogprobs.length > 0 && (() => {
                const maxProb = Math.max(
                  ...selectedToken.topLogprobs.map((a) => Math.exp(a.logprob))
                )
                return (
                  <div className="space-y-0.5">
                    {selectedToken.topLogprobs.map((alt, i) => (
                      <AlternativeBar key={i} alt={alt} maxProb={maxProb} isDark={isDark} size="compact" />
                    ))}
                  </div>
                )
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
