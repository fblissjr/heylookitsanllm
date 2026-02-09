import { useRef, useEffect } from 'react'
import { TokenChip } from './TokenChip'
import { StreamingCursor } from '../../../components/primitives/StreamingCursor'
import type { ExplorerToken } from '../types'

interface TokenStreamProps {
  tokens: ExplorerToken[]
  selectedIndex: number | null
  isStreaming: boolean
  thinkingTokenCount?: number
  onTokenClick: (index: number) => void
}

export function TokenStream({ tokens, selectedIndex, isStreaming, thinkingTokenCount, onTokenClick }: TokenStreamProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const endRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to end during streaming
  useEffect(() => {
    if (isStreaming && endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [tokens.length, isStreaming])

  if (tokens.length === 0) {
    return null
  }

  const boundary = thinkingTokenCount ?? 0

  return (
    <div
      ref={containerRef}
      className="leading-relaxed"
    >
      {boundary > 0 && (
        <div className="text-[10px] font-semibold text-purple-500 dark:text-purple-400 uppercase tracking-wider mb-1">
          Thinking ({boundary} tokens)
        </div>
      )}
      {tokens.map((token) => (
        <span key={token.index}>
          {boundary > 0 && token.index === boundary && (
            <div className="my-2 border-t border-purple-300 dark:border-purple-700 pt-1">
              <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Response
              </span>
            </div>
          )}
          <TokenChip
            token={token}
            isSelected={selectedIndex === token.index}
            onClick={onTokenClick}
          />
        </span>
      ))}
      {isStreaming && (
        <StreamingCursor />
      )}
      <div ref={endRef} />
    </div>
  )
}
