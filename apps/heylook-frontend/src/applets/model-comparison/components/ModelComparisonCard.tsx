import { useState, useCallback } from 'react'
import type { ModelResult, ModelResultStatus } from '../types'
import { MetricsBar } from './MetricsBar'
import { LogprobsDetail } from './LogprobsDetail'
import { ThinkingBlock } from '../../../components/composed/ThinkingBlock'
import { MessageActions } from '../../../components/composed/MessageActions'
import { StreamingCursor } from '../../../components/primitives/StreamingCursor'
import clsx from 'clsx'

interface ModelComparisonCardProps {
  result: ModelResult
  showLogprobs: boolean
  onStop: (modelId: string) => void
  onEditResult?: (modelId: string, updates: { content?: string; thinking?: string }) => void
}

const statusDot: Record<ModelResultStatus, string> = {
  pending: 'bg-gray-400',
  loading: 'bg-blue-500 animate-pulse',
  streaming: 'bg-green-500 animate-pulse',
  completed: 'bg-green-600',
  error: 'bg-red-500',
}

const statusLabel: Record<ModelResultStatus, string> = {
  pending: 'Waiting',
  loading: 'Loading model...',
  streaming: 'Streaming',
  completed: 'Done',
  error: 'Error',
}

export function ModelComparisonCard({ result, showLogprobs, onStop, onEditResult }: ModelComparisonCardProps) {
  const [thinkingOpen, setThinkingOpen] = useState(true)
  const isActive = result.status === 'loading' || result.status === 'streaming'
  const isCompleted = result.status === 'completed'

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(result.content)
  }, [result.content])

  const handleThinkingSave = useCallback((newThinking: string) => {
    onEditResult?.(result.modelId, { thinking: newThinking })
  }, [result.modelId, onEditResult])

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className={clsx('w-2 h-2 rounded-full flex-shrink-0', statusDot[result.status])} />
          <span className="text-xs font-medium text-gray-900 dark:text-white truncate">
            {result.modelId}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400">{statusLabel[result.status]}</span>
          {isActive && (
            <button
              onClick={() => onStop(result.modelId)}
              className="text-[10px] text-red-500 hover:text-red-600 font-medium"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-3 overflow-y-auto max-h-80">
        {result.status === 'error' ? (
          <p className="text-sm text-red-500">{result.error}</p>
        ) : result.status === 'pending' ? (
          <p className="text-sm text-gray-400 italic">Waiting for model...</p>
        ) : (
          <div className="space-y-2">
            {result.thinking && (
              <ThinkingBlock
                content={result.thinking}
                isOpen={thinkingOpen}
                onToggle={() => setThinkingOpen(!thinkingOpen)}
                editable={isCompleted}
                onSave={handleThinkingSave}
              />
            )}
            <div className="text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap break-words">
              {result.content}
              {result.status === 'streaming' && (
                <StreamingCursor className="ml-0.5 align-text-bottom" />
              )}
            </div>
          </div>
        )}
      </div>

      {/* Compact actions for completed results */}
      {isCompleted && (
        <div className="px-3 py-1 flex justify-end">
          <MessageActions
            role="assistant"
            onCopy={handleCopy}
            compact
          />
        </div>
      )}

      {/* Metrics */}
      {(result.status === 'completed' || result.status === 'streaming') && (
        <div className="px-3 pb-2">
          <MetricsBar performance={result.performance} />
        </div>
      )}

      {/* Logprobs detail */}
      {showLogprobs && result.tokens.length > 0 && (
        <div className="px-3 pb-3">
          <LogprobsDetail tokens={result.tokens} />
        </div>
      )}
    </div>
  )
}
