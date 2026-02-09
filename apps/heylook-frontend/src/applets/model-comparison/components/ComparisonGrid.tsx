import { useState } from 'react'
import type { ComparisonRun } from '../types'
import { useComparisonStore } from '../stores/comparisonStore'
import { ModelComparisonCard } from './ModelComparisonCard'
import clsx from 'clsx'

interface ComparisonGridProps {
  run: ComparisonRun
}

function gridCols(modelCount: number): string {
  if (modelCount <= 2) return 'grid-cols-1 lg:grid-cols-2'
  if (modelCount === 3) return 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3'
  return 'grid-cols-1 lg:grid-cols-2'
}

export function ComparisonGrid({ run }: ComparisonGridProps) {
  const stopModel = useComparisonStore((s) => s.stopModel)
  const [activePromptIndex, setActivePromptIndex] = useState(0)

  const isBatch = run.mode === 'batch'
  const promptCount = run.prompts.length
  const modelCount = run.selectedModelIds.length

  const handleStop = (modelId: string) => {
    stopModel(run.id, modelId)
  }

  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      {/* Batch prompt tabs */}
      {isBatch && promptCount > 1 && (
        <div className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700 px-4 flex gap-1 overflow-x-auto">
          {run.prompts.map((prompt, i) => (
            <button
              key={i}
              onClick={() => setActivePromptIndex(i)}
              className={clsx(
                'px-3 py-2 text-xs whitespace-nowrap border-b-2 transition-colors',
                i === activePromptIndex
                  ? 'border-primary text-primary font-medium'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
              )}
            >
              Prompt {i + 1}
              <span className="ml-1 text-gray-400 max-w-[100px] truncate inline-block align-bottom">
                {prompt.slice(0, 30)}{prompt.length > 30 ? '...' : ''}
              </span>
            </button>
          ))}
        </div>
      )}

      {/* Current prompt display */}
      <div className="flex-shrink-0 px-4 py-2 border-b border-gray-100 dark:border-gray-800">
        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
          <span className="font-medium">Prompt:</span>{' '}
          {run.prompts[activePromptIndex]}
        </p>
      </div>

      {/* Model cards grid */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className={clsx('grid gap-4', gridCols(modelCount))}>
          {run.selectedModelIds.map((modelId) => {
            const results = run.results.get(modelId)
            const result = results?.[activePromptIndex]
            if (!result) return null

            return (
              <ModelComparisonCard
                key={modelId}
                result={result}
                showLogprobs={run.enableLogprobs}
                onStop={handleStop}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}
