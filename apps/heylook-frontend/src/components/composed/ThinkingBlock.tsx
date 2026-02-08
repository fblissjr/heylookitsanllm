import { ChevronDownIcon } from '../icons'
import { formatDuration } from '../../utils/formatters'
import clsx from 'clsx'

interface ThinkingBlockProps {
  content: string
  isOpen: boolean
  onToggle: () => void
  thinkingTime?: number   // ms
  thinkingTokens?: number
}

export function ThinkingBlock({ content, isOpen, onToggle, thinkingTime, thinkingTokens }: ThinkingBlockProps) {
  const hasMetrics = thinkingTime !== undefined || thinkingTokens !== undefined
  let headerText = 'Thinking'
  if (hasMetrics) {
    const parts: string[] = []
    if (thinkingTime !== undefined) parts.push(formatDuration(thinkingTime))
    if (thinkingTokens !== undefined) parts.push(`${thinkingTokens.toLocaleString()} tokens`)
    headerText = `Thought for ${parts.join(' | ')}`
  }

  return (
    <details open={isOpen} className="bg-purple-50 dark:bg-purple-900/10 rounded-xl border-l-4 border-purple-400 overflow-hidden">
      <summary
        onClick={(e) => { e.preventDefault(); onToggle() }}
        className="flex items-center justify-between px-4 py-3 cursor-pointer select-none hover:bg-purple-100 dark:hover:bg-purple-900/20 transition-colors"
      >
        <div className="flex items-center gap-2 text-purple-700 dark:text-purple-300">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span className="text-sm font-medium">{headerText}</span>
        </div>
        <ChevronDownIcon className={clsx('w-4 h-4 text-purple-400 transition-transform', isOpen && 'rotate-180')} />
      </summary>
      {isOpen && (
        <div className="px-4 pb-4 pt-1 border-t border-purple-200 dark:border-purple-800/50">
          <pre className="text-sm text-purple-800/70 dark:text-purple-300/70 font-mono whitespace-pre-wrap overflow-x-auto">
            {content}
          </pre>
        </div>
      )}
    </details>
  )
}
