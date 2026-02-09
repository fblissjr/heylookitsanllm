import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useBatchStore } from './batchStore'
import type { ChatCompletionResponse } from '../../../types/api'

// Mock the API
const mockBatchChat = vi.fn()

vi.mock('../../../api/endpoints', () => ({
  batchChat: (...args: unknown[]) => mockBatchChat(...args),
}))

function mockBatchResponse(promptCount: number): ChatCompletionResponse {
  return {
    id: 'batch-resp-1',
    object: 'chat.completion',
    created: Date.now(),
    model: 'test-model',
    choices: Array.from({ length: promptCount }, (_, i) => ({
      index: i,
      message: {
        role: 'assistant' as const,
        content: `Response to prompt ${i + 1}`,
      },
      finish_reason: 'stop' as const,
    })),
    usage: {
      prompt_tokens: promptCount * 10,
      completion_tokens: promptCount * 20,
      total_tokens: promptCount * 30,
    },
  }
}

describe('batchStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset store to initial state
    useBatchStore.setState({
      jobs: [],
      activeJobId: null,
      view: 'create',
    })
  })

  describe('initial state', () => {
    it('starts with empty jobs', () => {
      expect(useBatchStore.getState().jobs).toEqual([])
    })

    it('starts with create view', () => {
      expect(useBatchStore.getState().view).toBe('create')
    })

    it('starts with no active job', () => {
      expect(useBatchStore.getState().activeJobId).toBeNull()
    })
  })

  describe('setView', () => {
    it('switches view to dashboard', () => {
      useBatchStore.getState().setView('dashboard')
      expect(useBatchStore.getState().view).toBe('dashboard')
    })

    it('switches view to create', () => {
      useBatchStore.getState().setView('dashboard')
      useBatchStore.getState().setView('create')
      expect(useBatchStore.getState().view).toBe('create')
    })
  })

  describe('setActiveJobId', () => {
    it('sets active job id', () => {
      useBatchStore.getState().setActiveJobId('job-1')
      expect(useBatchStore.getState().activeJobId).toBe('job-1')
    })

    it('clears active job id', () => {
      useBatchStore.getState().setActiveJobId('job-1')
      useBatchStore.getState().setActiveJobId(null)
      expect(useBatchStore.getState().activeJobId).toBeNull()
    })
  })

  describe('createJob', () => {
    it('creates a job and transitions to dashboard', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(2))

      const prompts = ['What is AI?', 'What is ML?']
      const id = await useBatchStore.getState().createJob(prompts, 'test-model', { temperature: 0.7 })

      expect(id).toBeTruthy()
      expect(id.startsWith('batch-')).toBe(true)
      expect(useBatchStore.getState().view).toBe('dashboard')
    })

    it('marks job as completed on success', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(2))

      await useBatchStore.getState().createJob(['p1', 'p2'], 'test-model', {})

      const job = useBatchStore.getState().jobs[0]
      expect(job.status).toBe('completed')
      expect(job.results).toHaveLength(2)
      expect(job.results[0].prompt).toBe('p1')
      expect(job.results[0].response).toBe('Response to prompt 1')
      expect(job.completedAt).toBeDefined()
    })

    it('stores token count from response', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(3))

      await useBatchStore.getState().createJob(['a', 'b', 'c'], 'test-model', {})

      const job = useBatchStore.getState().jobs[0]
      expect(job.totalTokens).toBe(90) // 3 * 30
    })

    it('marks job as failed on API error', async () => {
      mockBatchChat.mockRejectedValue(new Error('Server error'))

      await useBatchStore.getState().createJob(['p1'], 'test-model', {})

      const job = useBatchStore.getState().jobs[0]
      expect(job.status).toBe('failed')
      expect(job.error).toBe('Server error')
    })

    it('adds newest jobs first', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(1))

      await useBatchStore.getState().createJob(['first'], 'model', {})
      await useBatchStore.getState().createJob(['second'], 'model', {})

      const jobs = useBatchStore.getState().jobs
      expect(jobs).toHaveLength(2)
      // Second job should be first in the array (newest first)
      expect(jobs[0].prompts[0]).toBe('second')
    })
  })

  describe('removeJob', () => {
    it('removes a job by id', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(1))

      const id = await useBatchStore.getState().createJob(['p1'], 'model', {})
      expect(useBatchStore.getState().jobs).toHaveLength(1)

      useBatchStore.getState().removeJob(id)
      expect(useBatchStore.getState().jobs).toHaveLength(0)
    })

    it('clears activeJobId if removed job was active', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(1))

      const id = await useBatchStore.getState().createJob(['p1'], 'model', {})
      useBatchStore.getState().setActiveJobId(id)

      useBatchStore.getState().removeJob(id)
      expect(useBatchStore.getState().activeJobId).toBeNull()
    })

    it('does not clear activeJobId if a different job was removed', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(1))

      const id1 = await useBatchStore.getState().createJob(['p1'], 'model', {})
      const id2 = await useBatchStore.getState().createJob(['p2'], 'model', {})
      useBatchStore.getState().setActiveJobId(id1)

      useBatchStore.getState().removeJob(id2)
      expect(useBatchStore.getState().activeJobId).toBe(id1)
    })
  })

  describe('clearCompleted', () => {
    it('removes all completed jobs', async () => {
      mockBatchChat.mockResolvedValueOnce(mockBatchResponse(1))
      mockBatchChat.mockRejectedValueOnce(new Error('fail'))

      await useBatchStore.getState().createJob(['p1'], 'model', {})
      await useBatchStore.getState().createJob(['p2'], 'model', {})

      expect(useBatchStore.getState().jobs).toHaveLength(2)

      useBatchStore.getState().clearCompleted()

      const remaining = useBatchStore.getState().jobs
      expect(remaining).toHaveLength(1)
      expect(remaining[0].status).toBe('failed')
    })
  })

  describe('retryJob', () => {
    it('creates a new job with same params and removes the failed one', async () => {
      mockBatchChat.mockRejectedValueOnce(new Error('fail'))

      await useBatchStore.getState().createJob(['p1', 'p2'], 'model', { temperature: 0.5 })

      const failedJob = useBatchStore.getState().jobs[0]
      expect(failedJob.status).toBe('failed')

      // Now succeed on retry
      mockBatchChat.mockResolvedValueOnce(mockBatchResponse(2))

      await useBatchStore.getState().retryJob(failedJob.id)

      const jobs = useBatchStore.getState().jobs
      // Failed job removed, new job created
      expect(jobs).toHaveLength(1)
      expect(jobs[0].status).toBe('completed')
      expect(jobs[0].prompts).toEqual(['p1', 'p2'])
    })

    it('does nothing for non-failed jobs', async () => {
      mockBatchChat.mockResolvedValue(mockBatchResponse(1))

      const id = await useBatchStore.getState().createJob(['p1'], 'model', {})
      expect(useBatchStore.getState().jobs[0].status).toBe('completed')

      await useBatchStore.getState().retryJob(id)

      // Should still be the same job, untouched
      expect(useBatchStore.getState().jobs).toHaveLength(1)
    })
  })
})
