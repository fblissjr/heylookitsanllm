import { useCallback } from 'react'
import { useTheme } from '../../../contexts/ThemeContext'
import { probabilityToColor } from '../../../lib/color'
import { displayToken } from '../../../lib/tokens'
import type { ExplorerToken } from '../types'

interface TokenChipProps {
  token: ExplorerToken
  isSelected: boolean
  onClick: (index: number) => void
}

function formatLogprob(logprob: number): string {
  return logprob.toFixed(3)
}

function formatPercent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`
}

export function TokenChip({ token, isSelected, onClick }: TokenChipProps) {
  const { resolvedTheme } = useTheme()

  const handleClick = useCallback(() => {
    onClick(token.index)
  }, [onClick, token.index])

  const bgColor = probabilityToColor(token.probability, resolvedTheme === 'dark')
  const display = displayToken(token.token)

  return (
    <span
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          handleClick()
        }
      }}
      title={`"${token.token}" | logprob: ${formatLogprob(token.logprob)} | ${formatPercent(token.probability)}`}
      className={`inline-block px-1 py-0.5 mx-px my-0.5 rounded text-xs font-mono cursor-pointer transition-shadow ${
        isSelected
          ? 'ring-2 ring-primary shadow-md'
          : 'hover:ring-1 hover:ring-gray-400 dark:hover:ring-gray-500'
      }`}
      style={{
        backgroundColor: bgColor,
        color: resolvedTheme === 'dark' ? '#e5e7eb' : '#1f2937',
      }}
    >
      {display}
    </span>
  )
}
