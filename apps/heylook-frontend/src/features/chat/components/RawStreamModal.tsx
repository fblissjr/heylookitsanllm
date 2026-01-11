import { useEffect, useRef } from 'react'

interface RawStreamModalProps {
  isOpen: boolean
  onClose: () => void
  rawStream: string[]
}

export function RawStreamModal({ isOpen, onClose, rawStream }: RawStreamModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)

  // Close on escape key
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
      return () => document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  // Close when clicking backdrop
  const handleBackdropClick = (event: React.MouseEvent) => {
    if (event.target === event.currentTarget) {
      onClose()
    }
  }

  if (!isOpen) return null

  // Parse and format raw events for display
  const formatEvent = (event: string, index: number) => {
    try {
      // Extract JSON from SSE data line
      const match = event.match(/^data: (.+)$/)
      if (match) {
        const json = JSON.parse(match[1])
        return {
          index,
          raw: event,
          parsed: json,
          type: json.choices?.[0]?.delta?.thinking ? 'thinking' :
                json.choices?.[0]?.delta?.content ? 'content' :
                json.usage ? 'usage' : 'other'
        }
      }
      return { index, raw: event, parsed: null, type: 'raw' }
    } catch {
      return { index, raw: event, parsed: null, type: 'raw' }
    }
  }

  const formattedEvents = rawStream.map(formatEvent)

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'thinking': return 'text-purple-600 dark:text-purple-400'
      case 'content': return 'text-blue-600 dark:text-blue-400'
      case 'usage': return 'text-green-600 dark:text-green-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(rawStream.join('\n'))
  }

  return (
    <div
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      onClick={handleBackdropClick}
    >
      <div
        ref={modalRef}
        className="bg-white dark:bg-surface-dark rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
            Raw Stream Events ({rawStream.length})
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={copyToClipboard}
              className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            >
              Copy All
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-full text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              aria-label="Close"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          <div className="space-y-2 font-mono text-xs">
            {formattedEvents.map((event) => (
              <div
                key={event.index}
                className="bg-gray-50 dark:bg-gray-800 rounded p-2 border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-gray-400 text-[10px]">#{event.index + 1}</span>
                  <span className={`text-[10px] font-semibold uppercase ${getTypeColor(event.type)}`}>
                    {event.type}
                  </span>
                </div>
                {event.parsed ? (
                  <pre className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-all overflow-x-auto">
                    {JSON.stringify(event.parsed, null, 2)}
                  </pre>
                ) : (
                  <pre className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-all">
                    {event.raw}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span className="text-gray-500 dark:text-gray-400">Content</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-purple-500"></span>
            <span className="text-gray-500 dark:text-gray-400">Thinking</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500"></span>
            <span className="text-gray-500 dark:text-gray-400">Usage</span>
          </span>
        </div>
      </div>
    </div>
  )
}
