import { useRef, useEffect } from 'react'
import { TokenChip } from './TokenChip'
import type { ExplorerToken } from '../types'

interface TokenStreamProps {
  tokens: ExplorerToken[]
  selectedIndex: number | null
  isStreaming: boolean
  onTokenClick: (index: number) => void
}

export function TokenStream({ tokens, selectedIndex, isStreaming, onTokenClick }: TokenStreamProps) {
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

  return (
    <div
      ref={containerRef}
      className="leading-relaxed"
    >
      {tokens.map((token) => (
        <TokenChip
          key={token.index}
          token={token}
          isSelected={selectedIndex === token.index}
          onClick={onTokenClick}
        />
      ))}
      {isStreaming && (
        <span className="inline-block w-2 h-4 mx-1 bg-primary animate-pulse rounded-sm align-middle" />
      )}
      <div ref={endRef} />
    </div>
  )
}
