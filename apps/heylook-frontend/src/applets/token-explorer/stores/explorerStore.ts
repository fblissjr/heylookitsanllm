import { create } from 'zustand'
import { streamChat } from '../../../api/streaming'
import type { TokenLogprob } from '../../../types/api'
import type { ExplorerRun, RunStatus } from '../types'
import { generateId } from '../../../lib/id'
import { tokenFromLogprob } from '../../../lib/tokens'

let abortController: AbortController | null = null

interface ExplorerState {
  runs: ExplorerRun[]
  activeRunId: string | null
  selectedTokenIndex: number | null

  startRun: (prompt: string, model: string, topLogprobs: number, temperature: number, maxTokens: number) => void
  appendToken: (runId: string, logprob: TokenLogprob) => void
  completeRun: (runId: string) => void
  failRun: (runId: string, error: string) => void
  stopRun: () => void
  selectRun: (id: string) => void
  selectToken: (index: number | null) => void
  removeRun: (id: string) => void
  clearRuns: () => void
}

export const useExplorerStore = create<ExplorerState>((set, get) => ({
  runs: [],
  activeRunId: null,
  selectedTokenIndex: null,

  startRun: (prompt, model, topLogprobs, temperature, maxTokens) => {
    // Stop any active run first
    get().stopRun()

    const id = generateId('run')
    const run: ExplorerRun = {
      id,
      prompt,
      model,
      topLogprobs,
      temperature,
      maxTokens,
      status: 'streaming',
      tokens: [],
      createdAt: Date.now(),
    }

    set((state) => ({
      runs: [run, ...state.runs],
      activeRunId: id,
      selectedTokenIndex: null,
    }))

    abortController = new AbortController()

    streamChat(
      {
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature,
        max_tokens: maxTokens,
        logprobs: true,
        top_logprobs: topLogprobs,
      },
      {
        onToken: () => {
          // Tokens are handled via onLogprobs; content is ignored
        },
        onLogprobs: (logprobs) => {
          for (const lp of logprobs) {
            get().appendToken(id, lp)
          }
        },
        onComplete: () => {
          get().completeRun(id)
        },
        onError: (error) => {
          get().failRun(id, error.message)
        },
      },
      abortController.signal,
    )
  },

  appendToken: (runId, logprob) => {
    set((state) => ({
      runs: state.runs.map((r) => {
        if (r.id !== runId) return r
        const token = tokenFromLogprob(logprob, r.tokens.length)
        return { ...r, tokens: [...r.tokens, token] }
      }),
    }))
  },

  completeRun: (runId) => {
    set((state) => ({
      runs: state.runs.map((r) =>
        r.id === runId
          ? {
              ...r,
              status: 'completed' as RunStatus,
              completedAt: Date.now(),
              totalDuration: Date.now() - r.createdAt,
            }
          : r
      ),
    }))
    abortController = null
  },

  failRun: (runId, error) => {
    set((state) => ({
      runs: state.runs.map((r) =>
        r.id === runId ? { ...r, status: 'error' as RunStatus, error } : r
      ),
    }))
    abortController = null
  },

  stopRun: () => {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    const { activeRunId } = get()
    if (activeRunId) {
      set((state) => ({
        runs: state.runs.map((r) =>
          r.id === activeRunId && r.status === 'streaming'
            ? {
                ...r,
                status: 'stopped' as RunStatus,
                completedAt: Date.now(),
                totalDuration: Date.now() - r.createdAt,
              }
            : r
        ),
      }))
    }
  },

  selectRun: (id) => {
    set({ activeRunId: id, selectedTokenIndex: null })
  },

  selectToken: (index) => {
    set({ selectedTokenIndex: index })
  },

  removeRun: (id) => {
    set((state) => ({
      runs: state.runs.filter((r) => r.id !== id),
      activeRunId: state.activeRunId === id ? null : state.activeRunId,
      selectedTokenIndex: state.activeRunId === id ? null : state.selectedTokenIndex,
    }))
  },

  clearRuns: () => {
    get().stopRun()
    set({ runs: [], activeRunId: null, selectedTokenIndex: null })
  },
}))
