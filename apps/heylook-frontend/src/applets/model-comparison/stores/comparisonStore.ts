import { create } from 'zustand'
import { streamChat } from '../../../api/streaming'
import type { StreamCompletionData } from '../../../api/streaming'
import type { TokenLogprob } from '../../../types/api'
import { DEFAULT_SAMPLER_SETTINGS } from '../../../types/settings'
import { generateId } from '../../../lib/id'
import { tokenFromLogprob } from '../../../lib/tokens'
import type {
  ComparisonRun,
  ComparisonSettings,
  ModelResult,
  ModelPerformance,
  RunStatus,
  ComparisonPersistence,
} from '../types'
import { sessionPersistence } from './persistence'

// Module-level abort controllers: keyed by `${runId}-${modelId}-${promptIndex}`
const abortControllers = new Map<string, AbortController>()

function abortKey(runId: string, modelId: string, promptIndex: number): string {
  return `${runId}-${modelId}-${promptIndex}`
}

function emptyModelResult(modelId: string): ModelResult {
  return {
    modelId,
    status: 'pending',
    content: '',
    tokens: [],
    performance: {},
  }
}

function deriveRunStatus(results: Record<string, ModelResult[]>): RunStatus {
  const allResults = Object.values(results).flat()
  if (allResults.length === 0) return 'idle'

  const hasRunning = allResults.some(
    (r) => r.status === 'pending' || r.status === 'loading' || r.status === 'streaming'
  )
  if (hasRunning) return 'running'

  const hasError = allResults.some((r) => r.status === 'error')
  const hasCompleted = allResults.some((r) => r.status === 'completed')

  if (hasError && hasCompleted) return 'partial'
  if (hasError && !hasCompleted) return 'error'
  return 'completed'
}

/** Update a single ModelResult within a run's results Record. */
function updateRunResult(
  run: ComparisonRun,
  modelId: string,
  promptIndex: number,
  updater: (result: ModelResult) => ModelResult
): ComparisonRun {
  const existing = run.results[modelId]
  if (!existing || !existing[promptIndex]) return run

  const newArr = [...existing]
  newArr[promptIndex] = updater(newArr[promptIndex])
  return { ...run, results: { ...run.results, [modelId]: newArr } }
}

interface ComparisonState {
  runs: ComparisonRun[]
  activeRunId: string | null
  settings: ComparisonSettings
  persistence: ComparisonPersistence

  // Actions
  startRun: (prompts: string[], modelIds: string[]) => Promise<void>
  stopRun: (runId: string) => void
  stopModel: (runId: string, modelId: string) => void
  selectRun: (id: string | null) => void
  removeRun: (id: string) => void
  clearRuns: () => void
  updateSettings: (settings: Partial<ComparisonSettings>) => void

  editResult: (
    runId: string,
    modelId: string,
    promptIndex: number,
    updates: { content?: string; thinking?: string }
  ) => void

  // Internal: called by streaming callbacks
  updateModelResult: (
    runId: string,
    modelId: string,
    promptIndex: number,
    updates: Partial<ModelResult>
  ) => void
  appendContent: (
    runId: string,
    modelId: string,
    promptIndex: number,
    content: string,
    isThinking: boolean
  ) => void
  appendTokens: (
    runId: string,
    modelId: string,
    promptIndex: number,
    logprobs: TokenLogprob[]
  ) => void
}

export const useComparisonStore = create<ComparisonState>((set, get) => ({
  runs: [],
  activeRunId: null,
  settings: {
    samplerSettings: {
      temperature: DEFAULT_SAMPLER_SETTINGS.temperature,
      max_tokens: DEFAULT_SAMPLER_SETTINGS.max_tokens,
      top_p: DEFAULT_SAMPLER_SETTINGS.top_p,
      top_k: DEFAULT_SAMPLER_SETTINGS.top_k,
    },
    enableLogprobs: false,
    topLogprobs: 5,
    mode: 'single',
  },
  persistence: sessionPersistence,

  startRun: async (prompts, modelIds) => {
    const runId = generateId('comp')
    const { settings } = get()

    // Build results record: each model gets one ModelResult per prompt
    const results: Record<string, ModelResult[]> = {}
    for (const modelId of modelIds) {
      results[modelId] = prompts.map(() => emptyModelResult(modelId))
    }

    const run: ComparisonRun = {
      id: runId,
      mode: prompts.length > 1 ? 'batch' : 'single',
      prompts,
      selectedModelIds: modelIds,
      params: settings.samplerSettings,
      enableLogprobs: settings.enableLogprobs,
      topLogprobs: settings.topLogprobs,
      results,
      status: 'running',
      createdAt: Date.now(),
    }

    set((state) => ({
      runs: [run, ...state.runs],
      activeRunId: runId,
    }))

    // Fire all model x prompt requests concurrently
    const promises: Promise<void>[] = []
    for (const modelId of modelIds) {
      for (let pi = 0; pi < prompts.length; pi++) {
        promises.push(
          streamModelPrompt(runId, modelId, pi, prompts[pi], settings, get)
        )
      }
    }

    await Promise.allSettled(promises)

    // Derive final run status
    const finalRun = get().runs.find((r) => r.id === runId)
    if (finalRun) {
      const finalStatus = deriveRunStatus(finalRun.results)
      set((state) => ({
        runs: state.runs.map((r) =>
          r.id === runId
            ? { ...r, status: finalStatus, completedAt: Date.now() }
            : r
        ),
      }))
    }
  },

  stopRun: (runId) => {
    const run = get().runs.find((r) => r.id === runId)
    if (!run) return

    for (const modelId of run.selectedModelIds) {
      for (let pi = 0; pi < run.prompts.length; pi++) {
        const key = abortKey(runId, modelId, pi)
        const controller = abortControllers.get(key)
        if (controller) {
          controller.abort()
          abortControllers.delete(key)
        }
      }
    }
  },

  stopModel: (runId, modelId) => {
    const run = get().runs.find((r) => r.id === runId)
    if (!run) return

    for (let pi = 0; pi < run.prompts.length; pi++) {
      const key = abortKey(runId, modelId, pi)
      const controller = abortControllers.get(key)
      if (controller) {
        controller.abort()
        abortControllers.delete(key)
      }
    }
  },

  selectRun: (id) => set({ activeRunId: id }),

  removeRun: (id) => {
    // Abort if running
    get().stopRun(id)
    set((state) => ({
      runs: state.runs.filter((r) => r.id !== id),
      activeRunId: state.activeRunId === id ? null : state.activeRunId,
    }))
  },

  clearRuns: () => {
    // Abort all running
    for (const run of get().runs) {
      if (run.status === 'running') {
        get().stopRun(run.id)
      }
    }
    set({ runs: [], activeRunId: null })
  },

  updateSettings: (partial) => {
    set((state) => ({
      settings: { ...state.settings, ...partial },
    }))
  },

  editResult: (runId, modelId, promptIndex, updates) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        return updateRunResult(run, modelId, promptIndex, (r) => ({
          ...r,
          ...(updates.content !== undefined ? { content: updates.content } : {}),
          ...(updates.thinking !== undefined ? { thinking: updates.thinking } : {}),
        }))
      }),
    }))
  },

  updateModelResult: (runId, modelId, promptIndex, updates) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        return updateRunResult(run, modelId, promptIndex, (r) => ({ ...r, ...updates }))
      }),
    }))
  },

  appendContent: (runId, modelId, promptIndex, content, isThinking) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        return updateRunResult(run, modelId, promptIndex, (r) =>
          isThinking
            ? { ...r, thinking: (r.thinking || '') + content }
            : { ...r, content: r.content + content }
        )
      }),
    }))
  },

  appendTokens: (runId, modelId, promptIndex, logprobs) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        return updateRunResult(run, modelId, promptIndex, (r) => {
          const newTokens = logprobs.map((lp, i) =>
            tokenFromLogprob(lp, r.tokens.length + i)
          )
          return { ...r, tokens: [...r.tokens, ...newTokens] }
        })
      }),
    }))
  },
}))

async function streamModelPrompt(
  runId: string,
  modelId: string,
  promptIndex: number,
  prompt: string,
  settings: ComparisonSettings,
  get: () => ComparisonState
): Promise<void> {
  const key = abortKey(runId, modelId, promptIndex)
  const controller = new AbortController()
  abortControllers.set(key, controller)

  const startTime = Date.now()
  let firstTokenTime: number | undefined

  // Transition to loading
  get().updateModelResult(runId, modelId, promptIndex, { status: 'loading' })

  // Helper to look up current result for this model/prompt
  const getCurrentResult = () =>
    get().runs.find((r) => r.id === runId)?.results[modelId]?.[promptIndex]

  // Helper to transition to streaming on first token
  const transitionToStreaming = () => {
    const result = getCurrentResult()
    if (result?.status === 'loading') {
      firstTokenTime = Date.now()
      get().updateModelResult(runId, modelId, promptIndex, {
        status: 'streaming',
      })
    }
  }

  try {
    await streamChat(
      {
        model: modelId,
        messages: [{ role: 'user', content: prompt }],
        ...settings.samplerSettings,
        ...(settings.enableLogprobs
          ? { logprobs: true, top_logprobs: settings.topLogprobs }
          : {}),
      },
      {
        onToken: (token) => {
          if (!getCurrentResult()) return
          transitionToStreaming()
          get().appendContent(runId, modelId, promptIndex, token, false)
        },

        onThinking: (thinking) => {
          if (!getCurrentResult()) return
          transitionToStreaming()
          get().appendContent(runId, modelId, promptIndex, thinking, true)
        },

        onLogprobs: (logprobs) => {
          if (settings.enableLogprobs) {
            get().appendTokens(runId, modelId, promptIndex, logprobs)
          }
        },

        onComplete: (data?: StreamCompletionData) => {
          const endTime = Date.now()
          const ttft = firstTokenTime ? firstTokenTime - startTime : undefined
          const totalDuration = endTime - startTime
          const tokenCount = data?.usage?.completion_tokens
          const tokensPerSecond =
            tokenCount && totalDuration
              ? (tokenCount / totalDuration) * 1000
              : undefined

          const performance: ModelPerformance = {
            ttft,
            tokensPerSecond,
            totalDuration,
            promptTokens: data?.usage?.prompt_tokens,
            completionTokens: tokenCount,
            thinkingTokens: data?.usage?.thinking_tokens,
          }

          get().updateModelResult(runId, modelId, promptIndex, {
            status: 'completed',
            performance,
          })
        },

        onError: (error) => {
          get().updateModelResult(runId, modelId, promptIndex, {
            status: 'error',
            error: error.message,
          })
        },
      },
      controller.signal
    )
  } catch (error) {
    // Catch errors not handled by streaming callbacks (e.g. pre-fetch failures)
    const message = error instanceof Error ? error.message : 'Unknown error'
    get().updateModelResult(runId, modelId, promptIndex, {
      status: 'error',
      error: message,
    })
  } finally {
    abortControllers.delete(key)
  }
}
