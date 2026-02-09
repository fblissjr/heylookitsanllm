import { useState } from 'react'
import { Modal } from '../../../components/primitives'
import { CloseIcon, ChevronDownIcon, DownloadIcon } from '../../../components/icons'
import type { BatchJob } from '../types'
import clsx from 'clsx'

interface BatchResultsModalProps {
  job: BatchJob
  onClose: () => void
}

export function BatchResultsModal({ job, onClose }: BatchResultsModalProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0)

  const handleExport = () => {
    const data = {
      id: job.id,
      model: job.model,
      createdAt: new Date(job.createdAt).toISOString(),
      completedAt: job.completedAt ? new Date(job.completedAt).toISOString() : null,
      totalTokens: job.totalTokens,
      totalDuration: job.totalDuration,
      results: job.results.map((r) => ({
        prompt: r.prompt,
        response: r.response,
        thinking: r.thinking || null,
        usage: r.usage || null,
      })),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `batch-${job.id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Modal maxWidth="lg">
      <div className="max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Batch Results
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              {job.model} -- {job.results.length} result{job.results.length !== 1 ? 's' : ''}
              {job.totalTokens ? ` -- ${job.totalTokens.toLocaleString()} tokens` : ''}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              <DownloadIcon className="w-3.5 h-3.5" />
              Export JSON
            </button>
            <button
              onClick={onClose}
              className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {job.results.map((result, i) => {
            const isExpanded = expandedIndex === i
            return (
              <div
                key={i}
                className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden"
              >
                <button
                  onClick={() => setExpandedIndex(isExpanded ? null : i)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-800/50"
                >
                  <div className="flex-1 min-w-0">
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                      Prompt {i + 1}
                    </span>
                    <p className="text-sm text-gray-900 dark:text-white truncate mt-0.5">
                      {result.prompt}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 ml-3">
                    {result.usage && (
                      <span className="text-xs text-gray-400">
                        {result.usage.total_tokens} tokens
                      </span>
                    )}
                    <ChevronDownIcon
                      className={clsx('w-4 h-4 text-gray-400 transition-transform', isExpanded && 'rotate-180')}
                    />
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-4 pb-4 border-t border-gray-200 dark:border-gray-700">
                    {result.thinking && (
                      <div className="mt-3 p-3 rounded-lg bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800">
                        <p className="text-xs font-medium text-amber-600 dark:text-amber-400 mb-1">
                          Thinking
                        </p>
                        <p className="text-sm text-amber-800 dark:text-amber-200 whitespace-pre-wrap">
                          {result.thinking}
                        </p>
                      </div>
                    )}
                    <div className="mt-3">
                      <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                        Response
                      </p>
                      <p className="text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
                        {result.response}
                      </p>
                    </div>
                    {result.usage && (
                      <div className="mt-3 flex gap-4 text-xs text-gray-400">
                        <span>Prompt: {result.usage.prompt_tokens}</span>
                        <span>Completion: {result.usage.completion_tokens}</span>
                        <span>Total: {result.usage.total_tokens}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </Modal>
  )
}
