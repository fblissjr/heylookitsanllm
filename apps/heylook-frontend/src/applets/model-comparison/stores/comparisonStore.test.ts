import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useComparisonStore } from './comparisonStore'
import type { TokenLogprob } from '../../../types/api'
import type { StreamCallbacks, StreamCompletionData } from '../../../api/streaming'

// Mock the streaming API
const mockStreamChat = vi.fn()

vi.mock('../../../api/streaming', () => ({
  streamChat: (...args: unknown[]) => mockStreamChat(...args),
}))

function makeLogprob(token: string, logprob: number, tokenId = 0): TokenLogprob {
  return {
    token,
    token_id: tokenId,
    logprob,
    bytes: [],
    top_logprobs: [
      { token, token_id: tokenId, logprob, bytes: [] },
      { token: 'alt', token_id: 1, logprob: logprob - 1, bytes: [] },
    ],
  }
}

/** Simulate a completed stream by invoking callbacks */
function simulateStream(
  callbacks: StreamCallbacks,
  tokens: string[] = ['Hello', ' world'],
  completionData?: StreamCompletionData
) {
  for (const t of tokens) {
    callbacks.onToken(t)
  }
  callbacks.onComplete(completionData)
}

describe('comparisonStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useComparisonStore.setState({
      runs: [],
      activeRunId: null,
      settings: {
        samplerSettings: { temperature: 0.7, max_tokens: 2048, top_p: 0.9, top_k: 0 },
        enableLogprobs: false,
        topLogprobs: 5,
        mode: 'single',
      },
    })
  })

  describe('initial state', () => {
    it('starts with empty runs', () => {
      expect(useComparisonStore.getState().runs).toEqual([])
    })

    it('starts with no active run', () => {
      expect(useComparisonStore.getState().activeRunId).toBeNull()
    })

    it('has default settings', () => {
      const { settings } = useComparisonStore.getState()
      expect(settings.enableLogprobs).toBe(false)
      expect(settings.topLogprobs).toBe(5)
      expect(settings.mode).toBe('single')
      expect(settings.samplerSettings.temperature).toBe(0.7)
    })
  })

  describe('startRun', () => {
    it('creates a run with N pending model results', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      await useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])

      const { runs, activeRunId } = useComparisonStore.getState()
      expect(runs).toHaveLength(1)

      const run = runs[0]
      expect(run.prompts).toEqual(['Hello'])
      expect(run.selectedModelIds).toEqual(['model-a', 'model-b'])
      expect(run.mode).toBe('single')
      expect(Object.keys(run.results)).toHaveLength(2)
      expect(activeRunId).toBe(run.id)
    })

    it('fires N streamChat calls for N models (single prompt)', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      await useComparisonStore.getState().startRun(
        ['test prompt'],
        ['model-a', 'model-b', 'model-c']
      )

      expect(mockStreamChat).toHaveBeenCalledTimes(3)
    })

    it('fires N x P streamChat calls for batch mode', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      await useComparisonStore.getState().startRun(
        ['prompt 1', 'prompt 2'],
        ['model-a', 'model-b']
      )

      // 2 models x 2 prompts = 4 calls
      expect(mockStreamChat).toHaveBeenCalledTimes(4)

      const run = useComparisonStore.getState().runs[0]
      expect(run.mode).toBe('batch')
      // Each model has 2 results (one per prompt)
      expect(run.results['model-a']).toHaveLength(2)
      expect(run.results['model-b']).toHaveLength(2)
    })

    it('passes correct request params to streamChat', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      useComparisonStore.getState().updateSettings({
        samplerSettings: { temperature: 0.5, max_tokens: 1024 },
      })

      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      const [request] = mockStreamChat.mock.calls[0]
      expect(request.model).toBe('model-a')
      expect(request.temperature).toBe(0.5)
      expect(request.max_tokens).toBe(1024)
      expect(request.messages).toEqual([{ role: 'user', content: 'Hello' }])
    })

    it('includes logprobs params when enabled', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      useComparisonStore.getState().updateSettings({
        enableLogprobs: true,
        topLogprobs: 10,
      })

      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      const [request] = mockStreamChat.mock.calls[0]
      expect(request.logprobs).toBe(true)
      expect(request.top_logprobs).toBe(10)
    })

    it('omits logprobs params when disabled', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      const [request] = mockStreamChat.mock.calls[0]
      expect(request.logprobs).toBeUndefined()
      expect(request.top_logprobs).toBeUndefined()
    })

    it('newest run is first', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      await useComparisonStore.getState().startRun(['First'], ['model-a'])
      await useComparisonStore.getState().startRun(['Second'], ['model-a'])

      const runs = useComparisonStore.getState().runs
      expect(runs).toHaveLength(2)
      expect(runs[0].prompts[0]).toBe('Second')
      expect(runs[1].prompts[0]).toBe('First')
    })
  })

  describe('streaming callbacks', () => {
    it('onToken transitions loading -> streaming and appends content', async () => {
      let capturedCallbacks: StreamCallbacks | null = null

      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          capturedCallbacks = callbacks
          return new Promise(() => {}) // never resolves (we control flow)
        }
      )

      // Don't await -- it won't resolve
      useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      // Give the async flow a tick to set up
      await new Promise((r) => setTimeout(r, 10))

      expect(capturedCallbacks).not.toBeNull()

      // Simulate first token
      capturedCallbacks!.onToken('Hi')

      const result = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result.status).toBe('streaming')
      expect(result.content).toBe('Hi')

      // Simulate second token
      capturedCallbacks!.onToken(' there')
      const result2 = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result2.content).toBe('Hi there')
    })

    it('onThinking appends to thinking field', async () => {
      let capturedCallbacks: StreamCallbacks | null = null

      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          capturedCallbacks = callbacks
          return new Promise(() => {})
        }
      )

      useComparisonStore.getState().startRun(['Hello'], ['model-a'])
      await new Promise((r) => setTimeout(r, 10))

      capturedCallbacks!.onThinking!('Let me think...')
      capturedCallbacks!.onThinking!(' More thinking.')

      const result = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result.thinking).toBe('Let me think... More thinking.')
      expect(result.status).toBe('streaming')
    })

    it('onComplete sets completed status with performance', async () => {
      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          simulateStream(callbacks, ['Hi'], {
            usage: {
              prompt_tokens: 5,
              completion_tokens: 1,
              total_tokens: 6,
            },
          })
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      const result = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result.status).toBe('completed')
      expect(result.content).toBe('Hi')
      expect(result.performance.promptTokens).toBe(5)
      expect(result.performance.completionTokens).toBe(1)
    })

    it('onError sets error status', async () => {
      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          callbacks.onError(new Error('Model not found'))
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      const result = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result.status).toBe('error')
      expect(result.error).toBe('Model not found')
    })

    it('onLogprobs appends tokens when enabled', async () => {
      useComparisonStore.getState().updateSettings({ enableLogprobs: true })

      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          callbacks.onToken('Hi')
          callbacks.onLogprobs!([makeLogprob('Hi', -0.5, 42)])
          callbacks.onComplete()
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])

      const result = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result.tokens).toHaveLength(1)
      expect(result.tokens[0].token).toBe('Hi')
      expect(result.tokens[0].tokenId).toBe(42)
      expect(result.tokens[0].probability).toBeCloseTo(Math.exp(-0.5))
      expect(result.tokens[0].topLogprobs).toHaveLength(2)
    })
  })

  describe('run status derivation', () => {
    it('sets completed when all models complete', async () => {
      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          simulateStream(callbacks, ['OK'])
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])

      const run = useComparisonStore.getState().runs[0]
      expect(run.status).toBe('completed')
      expect(run.completedAt).toBeDefined()
    })

    it('sets partial when some models fail', async () => {
      let callCount = 0
      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          callCount++
          if (callCount === 1) {
            simulateStream(callbacks, ['OK'])
          } else {
            callbacks.onError(new Error('Failed'))
          }
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])

      const run = useComparisonStore.getState().runs[0]
      expect(run.status).toBe('partial')
    })

    it('sets error when all models fail', async () => {
      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          callbacks.onError(new Error('Failed'))
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])

      const run = useComparisonStore.getState().runs[0]
      expect(run.status).toBe('error')
    })
  })

  describe('stopRun', () => {
    it('aborts all active controllers for a run', async () => {
      const abortSpy = vi.fn()

      mockStreamChat.mockImplementation(
        (_req: unknown, _callbacks: StreamCallbacks, signal?: AbortSignal) => {
          if (signal) {
            signal.addEventListener('abort', abortSpy)
          }
          return new Promise(() => {}) // never resolves
        }
      )

      // Don't await
      useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])
      await new Promise((r) => setTimeout(r, 10))

      useComparisonStore.getState().stopRun(
        useComparisonStore.getState().runs[0].id
      )

      expect(abortSpy).toHaveBeenCalledTimes(2)
    })

    it('does nothing for non-existent run', () => {
      // Should not throw
      useComparisonStore.getState().stopRun('nonexistent')
    })
  })

  describe('stopModel', () => {
    it('aborts only the specified model', async () => {
      const abortSpyA = vi.fn()
      const abortSpyB = vi.fn()
      let callIndex = 0

      mockStreamChat.mockImplementation(
        (_req: unknown, _callbacks: StreamCallbacks, signal?: AbortSignal) => {
          const spy = callIndex === 0 ? abortSpyA : abortSpyB
          callIndex++
          if (signal) {
            signal.addEventListener('abort', spy)
          }
          return new Promise(() => {})
        }
      )

      useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])
      await new Promise((r) => setTimeout(r, 10))

      const runId = useComparisonStore.getState().runs[0].id
      useComparisonStore.getState().stopModel(runId, 'model-a')

      expect(abortSpyA).toHaveBeenCalledTimes(1)
      expect(abortSpyB).not.toHaveBeenCalled()
    })
  })

  describe('selectRun', () => {
    it('sets active run id', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())
      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])
      const runId = useComparisonStore.getState().runs[0].id

      useComparisonStore.getState().selectRun(runId)
      expect(useComparisonStore.getState().activeRunId).toBe(runId)
    })

    it('can clear active run', () => {
      useComparisonStore.getState().selectRun(null)
      expect(useComparisonStore.getState().activeRunId).toBeNull()
    })
  })

  describe('removeRun', () => {
    it('removes a run by id', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())
      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])
      const runId = useComparisonStore.getState().runs[0].id

      useComparisonStore.getState().removeRun(runId)
      expect(useComparisonStore.getState().runs).toHaveLength(0)
    })

    it('clears activeRunId if removed run was active', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())
      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])
      const runId = useComparisonStore.getState().activeRunId!

      useComparisonStore.getState().removeRun(runId)
      expect(useComparisonStore.getState().activeRunId).toBeNull()
    })

    it('does not clear activeRunId if different run removed', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())
      await useComparisonStore.getState().startRun(['Run1'], ['model-a'])
      const run1Id = useComparisonStore.getState().runs[0].id

      await useComparisonStore.getState().startRun(['Run2'], ['model-a'])
      const activeId = useComparisonStore.getState().activeRunId

      useComparisonStore.getState().removeRun(run1Id)
      expect(useComparisonStore.getState().activeRunId).toBe(activeId)
    })
  })

  describe('clearRuns', () => {
    it('removes all runs and resets state', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())
      await useComparisonStore.getState().startRun(['Run1'], ['model-a'])
      await useComparisonStore.getState().startRun(['Run2'], ['model-a'])

      useComparisonStore.getState().clearRuns()

      const state = useComparisonStore.getState()
      expect(state.runs).toHaveLength(0)
      expect(state.activeRunId).toBeNull()
    })
  })

  describe('updateSettings', () => {
    it('merges partial settings', () => {
      useComparisonStore.getState().updateSettings({
        enableLogprobs: true,
        topLogprobs: 10,
      })

      const { settings } = useComparisonStore.getState()
      expect(settings.enableLogprobs).toBe(true)
      expect(settings.topLogprobs).toBe(10)
      // Other settings preserved
      expect(settings.mode).toBe('single')
    })

    it('updates sampler settings', () => {
      useComparisonStore.getState().updateSettings({
        samplerSettings: { temperature: 1.5 },
      })

      expect(useComparisonStore.getState().settings.samplerSettings.temperature).toBe(1.5)
    })

    it('switches mode', () => {
      useComparisonStore.getState().updateSettings({ mode: 'batch' })
      expect(useComparisonStore.getState().settings.mode).toBe('batch')
    })
  })

  describe('multi-model content isolation', () => {
    it('updates only the targeted model result', async () => {
      let callbackMap: Record<string, StreamCallbacks> = {}

      mockStreamChat.mockImplementation(
        (req: { model: string }, callbacks: StreamCallbacks) => {
          callbackMap[req.model] = callbacks
          return new Promise(() => {})
        }
      )

      useComparisonStore.getState().startRun(['Hello'], ['model-a', 'model-b'])
      await new Promise((r) => setTimeout(r, 10))

      // Simulate tokens for model-a only
      callbackMap['model-a'].onToken('Alpha')
      callbackMap['model-b'].onToken('Beta')

      const run = useComparisonStore.getState().runs[0]
      expect(run.results['model-a'][0].content).toBe('Alpha')
      expect(run.results['model-b'][0].content).toBe('Beta')
    })
  })

  describe('batch mode results', () => {
    it('stores results per prompt per model', async () => {
      let callIndex = 0
      const tokens = ['A1', 'A2', 'B1', 'B2']

      mockStreamChat.mockImplementation(
        (_req: unknown, callbacks: StreamCallbacks) => {
          simulateStream(callbacks, [tokens[callIndex++]])
          return Promise.resolve()
        }
      )

      await useComparisonStore.getState().startRun(
        ['prompt-1', 'prompt-2'],
        ['model-a', 'model-b']
      )

      const run = useComparisonStore.getState().runs[0]
      // model-a has 2 results, model-b has 2 results
      expect(run.results['model-a']).toHaveLength(2)
      expect(run.results['model-b']).toHaveLength(2)

      // All results should be completed
      for (const modelResults of Object.values(run.results)) {
        for (const result of modelResults) {
          expect(result.status).toBe('completed')
        }
      }
    })
  })

  describe('updateModelResult', () => {
    it('merges partial updates', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())
      await useComparisonStore.getState().startRun(['Hello'], ['model-a'])
      const runId = useComparisonStore.getState().runs[0].id

      useComparisonStore.getState().updateModelResult(runId, 'model-a', 0, {
        status: 'streaming',
      })

      const result = useComparisonStore.getState().runs[0].results['model-a'][0]
      expect(result.status).toBe('streaming')
      expect(result.modelId).toBe('model-a') // preserved
    })
  })
})
