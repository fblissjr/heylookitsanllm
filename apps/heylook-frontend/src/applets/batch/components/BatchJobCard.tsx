import type { BatchJob, BatchJobStatus } from '../types'
import { TrashIcon, RefreshIcon } from '../../../components/icons'
import clsx from 'clsx'

interface BatchJobCardProps {
  job: BatchJob
  onView: (id: string) => void
  onRemove: (id: string) => void
  onRetry: (id: string) => void
}

const statusConfig: Record<BatchJobStatus, { label: string; color: string; bg: string }> = {
  queued: {
    label: 'Queued',
    color: 'text-blue-600 dark:text-blue-400',
    bg: 'bg-blue-100 dark:bg-blue-900/30',
  },
  processing: {
    label: 'Processing',
    color: 'text-amber-600 dark:text-amber-400',
    bg: 'bg-amber-100 dark:bg-amber-900/30',
  },
  completed: {
    label: 'Completed',
    color: 'text-green-600 dark:text-green-400',
    bg: 'bg-green-100 dark:bg-green-900/30',
  },
  failed: {
    label: 'Failed',
    color: 'text-red-600 dark:text-red-400',
    bg: 'bg-red-100 dark:bg-red-900/30',
  },
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function BatchJobCard({ job, onView, onRemove, onRetry }: BatchJobCardProps) {
  const config = statusConfig[job.status]

  return (
    <div className="p-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-surface-dark">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={clsx('text-xs font-medium px-2 py-0.5 rounded-full', config.bg, config.color)}>
              {config.label}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {formatTimestamp(job.createdAt)}
            </span>
          </div>
          <p className="text-sm font-medium text-gray-900 dark:text-white mt-1.5 truncate">
            {job.model}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {job.prompts.length} prompt{job.prompts.length !== 1 ? 's' : ''}
            {job.totalTokens ? ` -- ${job.totalTokens.toLocaleString()} tokens` : ''}
            {job.totalDuration ? ` -- ${formatDuration(job.totalDuration)}` : ''}
          </p>
          {job.error && (
            <p className="text-xs text-red-500 mt-1">
              {job.error}
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 shrink-0">
          {job.status === 'completed' && (
            <button
              onClick={() => onView(job.id)}
              className="px-2.5 py-1 text-xs font-medium rounded-lg bg-primary/10 text-primary hover:bg-primary/20"
            >
              View
            </button>
          )}
          {job.status === 'failed' && (
            <button
              onClick={() => onRetry(job.id)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-amber-500 hover:bg-amber-50 dark:hover:bg-amber-900/20"
              title="Retry"
            >
              <RefreshIcon className="w-4 h-4" />
            </button>
          )}
          {(job.status === 'completed' || job.status === 'failed' || job.status === 'queued') && (
            <button
              onClick={() => onRemove(job.id)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20"
              title="Remove"
            >
              <TrashIcon className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Processing indicator */}
      {job.status === 'processing' && (
        <div className="mt-3">
          <div className="h-1.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
            <div className="h-full rounded-full bg-amber-500 animate-pulse w-2/3" />
          </div>
        </div>
      )}
    </div>
  )
}
