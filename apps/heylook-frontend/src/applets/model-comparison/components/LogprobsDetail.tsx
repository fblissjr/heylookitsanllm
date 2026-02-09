import { useState, useCallback } from 'react'
import { useTheme } from '../../../contexts/ThemeContext'
import type { ComparisonToken } from '../types'
import type { TokenLogprob } from '../../../types/api'
import { ChevronDownIcon } from '../../../components/icons'
import clsx from 'clsx'

interface LogprobsDetailProps {
  tokens: ComparisonToken[]
}

function probabilityToColor(probability: number, isDark: boolean): string {
  const p = Math.max(0, Math.min(1, probability))
  const hue = p * 120
  const saturation = 65
  const lightness = isDark ? 25 + p * 10 : 85 - p * 15
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
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
      <span className="w-16 font-mono truncate text-gray-700 dark:text-gray-300 text-right shrink-0" title={alt.token}>
        {alt.token === ' ' ? '\u00B7' : alt.token === '\n' ? '\u21B5' : alt.token}
      </span>
      <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${Math.max(2, widthPercent)}%`,
            backgroundColor: probabilityToBarColor(prob, isDark),
          }}
        />
      </div>
      <span className="w-14 text-right font-mono text-gray-500 dark:text-gray-400 shrink-0">
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
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
            {tokens.map((token) => {
              const displayText =
                token.token === '\n'
                  ? '\u21B5'
                  : token.token === '\t'
                    ? '\u2192'
                    : token.token === ' '
                      ? '\u00B7'
                      : token.token

              return (
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
                  {displayText}
                </span>
              )
            })}
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
              {selectedToken.topLogprobs.length > 0 && (
                <div className="space-y-0.5">
                  {selectedToken.topLogprobs.map((alt, i) => {
                    const maxProb = Math.max(
                      ...selectedToken.topLogprobs.map((a) => Math.exp(a.logprob))
                    )
                    return (
                      <AlternativeBar key={i} alt={alt} maxProb={maxProb} isDark={isDark} />
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
