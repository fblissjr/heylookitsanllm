import { useState, useCallback } from 'react'
import { useBatchStore } from '../stores/batchStore'
import { BatchJobCard } from './BatchJobCard'
import { BatchResultsModal } from './BatchResultsModal'
import { EmptyState } from '../../../components/primitives'
import { PlusIcon, LayersIcon } from '../../../components/icons'

export function BatchDashboard() {
  const jobs = useBatchStore((s) => s.jobs)
  const setView = useBatchStore((s) => s.setView)
  const removeJob = useBatchStore((s) => s.removeJob)
  const retryJob = useBatchStore((s) => s.retryJob)
  const clearCompleted = useBatchStore((s) => s.clearCompleted)

  const [viewingJobId, setViewingJobId] = useState<string | null>(null)
  const viewingJob = viewingJobId ? jobs.find((j) => j.id === viewingJobId) : null

  const completedCount = jobs.filter((j) => j.status === 'completed').length
  const failedCount = jobs.filter((j) => j.status === 'failed').length
  const activeCount = jobs.filter((j) => j.status === 'processing' || j.status === 'queued').length
  const successRate = completedCount + failedCount > 0
    ? Math.round((completedCount / (completedCount + failedCount)) * 100)
    : 0

  const handleView = useCallback((id: string) => setViewingJobId(id), [])
  const handleRemove = useCallback((id: string) => removeJob(id), [removeJob])
  const handleRetry = useCallback((id: string) => retryJob(id), [retryJob])

  if (jobs.length === 0) {
    return (
      <div className="h-full flex flex-col">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
            Batch Dashboard
          </h1>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 gap-4">
          <EmptyState
            icon={<LayersIcon className="w-12 h-12" />}
            title="No batch jobs yet"
            description="Create your first batch to process multiple prompts at once."
          />
          <button
            onClick={() => setView('create')}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-primary text-white hover:bg-primary-hover"
          >
            <PlusIcon className="w-4 h-4" />
            New Batch
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
            Batch Dashboard
          </h1>
          <button
            onClick={() => setView('create')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-primary text-white hover:bg-primary-hover"
          >
            <PlusIcon className="w-4 h-4" />
            New Batch
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="px-6 py-3 border-b border-gray-200 dark:border-gray-700 flex gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 dark:text-gray-400">Active</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">{activeCount}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 dark:text-gray-400">Completed</span>
          <span className="text-sm font-medium text-green-600 dark:text-green-400">{completedCount}</span>
        </div>
        {failedCount > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Failed</span>
            <span className="text-sm font-medium text-red-600 dark:text-red-400">{failedCount}</span>
          </div>
        )}
        {completedCount + failedCount > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Success Rate</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{successRate}%</span>
          </div>
        )}
        {completedCount > 0 && (
          <div className="ml-auto">
            <button
              onClick={clearCompleted}
              className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              Clear completed
            </button>
          </div>
        )}
      </div>

      {/* Job list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {jobs.map((job) => (
          <BatchJobCard
            key={job.id}
            job={job}
            onView={handleView}
            onRemove={handleRemove}
            onRetry={handleRetry}
          />
        ))}
      </div>

      {/* Results Modal */}
      {viewingJob && viewingJob.status === 'completed' && (
        <BatchResultsModal job={viewingJob} onClose={() => setViewingJobId(null)} />
      )}
    </div>
  )
}
