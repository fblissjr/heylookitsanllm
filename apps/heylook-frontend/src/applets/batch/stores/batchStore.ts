import { create } from 'zustand'
import { batchChat } from '../../../api/endpoints'
import type { BatchJob, BatchJobStatus, BatchJobResult, BatchView } from '../types'
import type { SamplerSettings } from '../../../types/settings'

function generateId(): string {
  return `batch-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

interface BatchState {
  // Data
  jobs: BatchJob[]
  activeJobId: string | null
  view: BatchView

  // Actions
  setView: (view: BatchView) => void
  setActiveJobId: (id: string | null) => void
  createJob: (prompts: string[], model: string, params: Partial<SamplerSettings>) => Promise<string>
  updateJobStatus: (id: string, status: BatchJobStatus, error?: string) => void
  removeJob: (id: string) => void
  clearCompleted: () => void
  retryJob: (id: string) => Promise<void>
}

export const useBatchStore = create<BatchState>((set, get) => ({
  jobs: [],
  activeJobId: null,
  view: 'create',

  setView: (view) => set({ view }),

  setActiveJobId: (id) => set({ activeJobId: id }),

  createJob: async (prompts, model, params) => {
    const id = generateId()
    const job: BatchJob = {
      id,
      status: 'queued',
      prompts,
      model,
      params,
      results: [],
      createdAt: Date.now(),
    }

    set((state) => ({
      jobs: [job, ...state.jobs],
      view: 'dashboard',
    }))

    // Immediately start processing (the endpoint is synchronous)
    set((state) => ({
      jobs: state.jobs.map((j) => (j.id === id ? { ...j, status: 'processing' as const } : j)),
    }))

    try {
      const startTime = Date.now()
      const response = await batchChat(model, prompts, {
        model,
        messages: [],
        ...params,
      })

      const duration = Date.now() - startTime

      // Parse results from the batch response
      // The batch endpoint returns choices[] with one entry per prompt
      const results: BatchJobResult[] = prompts.map((prompt, i) => {
        const choice = response.choices[i]
        return {
          prompt,
          response: choice?.message?.content ?? '',
          thinking: choice?.message?.thinking,
          usage: response.usage
            ? {
                prompt_tokens: Math.round((response.usage.prompt_tokens || 0) / prompts.length),
                completion_tokens: Math.round((response.usage.completion_tokens || 0) / prompts.length),
                total_tokens: Math.round((response.usage.total_tokens || 0) / prompts.length),
              }
            : undefined,
        }
      })

      set((state) => ({
        jobs: state.jobs.map((j) =>
          j.id === id
            ? {
                ...j,
                status: 'completed' as const,
                results,
                completedAt: Date.now(),
                totalTokens: response.usage?.total_tokens,
                totalDuration: duration,
              }
            : j
        ),
      }))
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Batch processing failed'
      set((state) => ({
        jobs: state.jobs.map((j) =>
          j.id === id ? { ...j, status: 'failed' as const, error: message } : j
        ),
      }))
    }

    return id
  },

  updateJobStatus: (id, status, error) => {
    set((state) => ({
      jobs: state.jobs.map((j) => (j.id === id ? { ...j, status, error } : j)),
    }))
  },

  removeJob: (id) => {
    set((state) => ({
      jobs: state.jobs.filter((j) => j.id !== id),
      activeJobId: state.activeJobId === id ? null : state.activeJobId,
    }))
  },

  clearCompleted: () => {
    set((state) => ({
      jobs: state.jobs.filter((j) => j.status !== 'completed'),
      activeJobId:
        state.activeJobId && state.jobs.find((j) => j.id === state.activeJobId)?.status === 'completed'
          ? null
          : state.activeJobId,
    }))
  },

  retryJob: async (id) => {
    const job = get().jobs.find((j) => j.id === id)
    if (!job || job.status !== 'failed') return

    // Create a new job with the same parameters
    await get().createJob(job.prompts, job.model, job.params)

    // Remove the failed job
    get().removeJob(id)
  },
}))
