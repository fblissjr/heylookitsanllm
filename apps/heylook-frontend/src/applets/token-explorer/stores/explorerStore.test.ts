import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useExplorerStore } from './explorerStore'
import type { TokenLogprob } from '../../../types/api'

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

describe('explorerStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useExplorerStore.setState({
      runs: [],
      activeRunId: null,
      selectedTokenIndex: null,
    })
  })

  describe('initial state', () => {
    it('starts with empty runs', () => {
      expect(useExplorerStore.getState().runs).toEqual([])
    })

    it('starts with no active run', () => {
      expect(useExplorerStore.getState().activeRunId).toBeNull()
    })

    it('starts with no selected token', () => {
      expect(useExplorerStore.getState().selectedTokenIndex).toBeNull()
    })
  })

  describe('startRun', () => {
    it('creates a new run with streaming status', () => {
      mockStreamChat.mockImplementation(() => {})

      useExplorerStore.getState().startRun('Hello', 'test-model', 5, 0.7, 256)

      const { runs, activeRunId } = useExplorerStore.getState()
      expect(runs).toHaveLength(1)
      expect(runs[0].prompt).toBe('Hello')
      expect(runs[0].model).toBe('test-model')
      expect(runs[0].topLogprobs).toBe(5)
      expect(runs[0].temperature).toBe(0.7)
      expect(runs[0].maxTokens).toBe(256)
      expect(runs[0].status).toBe('streaming')
      expect(runs[0].tokens).toEqual([])
      expect(activeRunId).toBe(runs[0].id)
    })

    it('calls streamChat with logprobs enabled', () => {
      mockStreamChat.mockImplementation(() => {})

      useExplorerStore.getState().startRun('Hello', 'test-model', 5, 0.7, 256)

      expect(mockStreamChat).toHaveBeenCalledTimes(1)
      const [request] = mockStreamChat.mock.calls[0]
      expect(request.logprobs).toBe(true)
      expect(request.top_logprobs).toBe(5)
      expect(request.temperature).toBe(0.7)
      expect(request.max_tokens).toBe(256)
    })

    it('adds newest runs first', () => {
      mockStreamChat.mockImplementation(() => {})

      useExplorerStore.getState().startRun('First', 'model', 5, 0.7, 256)
      useExplorerStore.getState().startRun('Second', 'model', 5, 0.7, 256)

      const runs = useExplorerStore.getState().runs
      expect(runs).toHaveLength(2)
      expect(runs[0].prompt).toBe('Second')
    })

    it('clears selected token on new run', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.setState({ selectedTokenIndex: 3 })

      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)

      expect(useExplorerStore.getState().selectedTokenIndex).toBeNull()
    })
  })

  describe('appendToken', () => {
    it('appends a token to the correct run', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().runs[0].id

      const lp = makeLogprob('Hello', -0.5, 42)
      useExplorerStore.getState().appendToken(runId, lp)

      const token = useExplorerStore.getState().runs[0].tokens[0]
      expect(token.index).toBe(0)
      expect(token.token).toBe('Hello')
      expect(token.tokenId).toBe(42)
      expect(token.logprob).toBe(-0.5)
      expect(token.probability).toBeCloseTo(Math.exp(-0.5))
      expect(token.topLogprobs).toHaveLength(2)
    })

    it('assigns sequential indices', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().runs[0].id

      useExplorerStore.getState().appendToken(runId, makeLogprob('A', -0.1))
      useExplorerStore.getState().appendToken(runId, makeLogprob('B', -0.2))
      useExplorerStore.getState().appendToken(runId, makeLogprob('C', -0.3))

      const tokens = useExplorerStore.getState().runs[0].tokens
      expect(tokens.map((t) => t.index)).toEqual([0, 1, 2])
      expect(tokens.map((t) => t.token)).toEqual(['A', 'B', 'C'])
    })

    it('does not modify other runs', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Run1', 'model', 5, 0.7, 256)
      const run1Id = useExplorerStore.getState().runs[0].id

      useExplorerStore.getState().startRun('Run2', 'model', 5, 0.7, 256)

      useExplorerStore.getState().appendToken(run1Id, makeLogprob('X', -0.1))

      const runs = useExplorerStore.getState().runs
      const run1 = runs.find((r) => r.id === run1Id)!
      const run2 = runs.find((r) => r.id !== run1Id)!
      expect(run1.tokens).toHaveLength(1)
      expect(run2.tokens).toHaveLength(0)
    })
  })

  describe('completeRun', () => {
    it('transitions run to completed', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().runs[0].id

      useExplorerStore.getState().completeRun(runId)

      const run = useExplorerStore.getState().runs[0]
      expect(run.status).toBe('completed')
      expect(run.completedAt).toBeDefined()
      expect(run.totalDuration).toBeDefined()
    })
  })

  describe('failRun', () => {
    it('transitions run to error with message', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().runs[0].id

      useExplorerStore.getState().failRun(runId, 'Connection lost')

      const run = useExplorerStore.getState().runs[0]
      expect(run.status).toBe('error')
      expect(run.error).toBe('Connection lost')
    })
  })

  describe('stopRun', () => {
    it('transitions active streaming run to stopped', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)

      useExplorerStore.getState().stopRun()

      const run = useExplorerStore.getState().runs[0]
      expect(run.status).toBe('stopped')
      expect(run.completedAt).toBeDefined()
    })

    it('does nothing if no active run', () => {
      useExplorerStore.getState().stopRun()
      expect(useExplorerStore.getState().runs).toEqual([])
    })
  })

  describe('selectRun', () => {
    it('sets active run id and clears token selection', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().runs[0].id

      useExplorerStore.setState({ selectedTokenIndex: 5 })
      useExplorerStore.getState().selectRun(runId)

      expect(useExplorerStore.getState().activeRunId).toBe(runId)
      expect(useExplorerStore.getState().selectedTokenIndex).toBeNull()
    })
  })

  describe('selectToken', () => {
    it('sets selected token index', () => {
      useExplorerStore.getState().selectToken(3)
      expect(useExplorerStore.getState().selectedTokenIndex).toBe(3)
    })

    it('clears selection with null', () => {
      useExplorerStore.getState().selectToken(3)
      useExplorerStore.getState().selectToken(null)
      expect(useExplorerStore.getState().selectedTokenIndex).toBeNull()
    })
  })

  describe('removeRun', () => {
    it('removes a run by id', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().runs[0].id

      useExplorerStore.getState().removeRun(runId)
      expect(useExplorerStore.getState().runs).toHaveLength(0)
    })

    it('clears activeRunId if removed run was active', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Hello', 'model', 5, 0.7, 256)
      const runId = useExplorerStore.getState().activeRunId!

      useExplorerStore.getState().removeRun(runId)
      expect(useExplorerStore.getState().activeRunId).toBeNull()
      expect(useExplorerStore.getState().selectedTokenIndex).toBeNull()
    })

    it('does not clear activeRunId if different run removed', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Run1', 'model', 5, 0.7, 256)
      const run1Id = useExplorerStore.getState().runs[0].id

      useExplorerStore.getState().startRun('Run2', 'model', 5, 0.7, 256)
      const run2Id = useExplorerStore.getState().activeRunId!

      useExplorerStore.getState().removeRun(run1Id)
      expect(useExplorerStore.getState().activeRunId).toBe(run2Id)
    })
  })

  describe('clearRuns', () => {
    it('removes all runs and resets state', () => {
      mockStreamChat.mockImplementation(() => {})
      useExplorerStore.getState().startRun('Run1', 'model', 5, 0.7, 256)
      useExplorerStore.getState().startRun('Run2', 'model', 5, 0.7, 256)

      useExplorerStore.getState().clearRuns()

      const state = useExplorerStore.getState()
      expect(state.runs).toHaveLength(0)
      expect(state.activeRunId).toBeNull()
      expect(state.selectedTokenIndex).toBeNull()
    })
  })
})
