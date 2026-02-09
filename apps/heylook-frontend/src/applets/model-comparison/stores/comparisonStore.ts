import { create } from 'zustand'
import { streamChat } from '../../../api/streaming'
import type { StreamCompletionData } from '../../../api/streaming'
import type { TokenLogprob } from '../../../types/api'
import { DEFAULT_SAMPLER_SETTINGS } from '../../../types/settings'
import type {
  ComparisonRun,
  ComparisonToken,
  ComparisonSettings,
  ModelResult,
  ModelPerformance,
  RunStatus,
  ComparisonPersistence,
} from '../types'
import { sessionPersistence } from './persistence'

// Module-level abort controllers: keyed by `${runId}-${modelId}-${promptIndex}`
const abortControllers = new Map<string, AbortController>()

function generateId(): string {
  return `comp-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
}

function abortKey(runId: string, modelId: string, promptIndex: number): string {
  return `${runId}-${modelId}-${promptIndex}`
}

function tokenFromLogprob(logprob: TokenLogprob, index: number): ComparisonToken {
  return {
    index,
    token: logprob.token,
    tokenId: logprob.token_id,
    logprob: logprob.logprob,
    probability: Math.exp(logprob.logprob),
    topLogprobs: logprob.top_logprobs ?? [],
  }
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

function deriveRunStatus(results: Map<string, ModelResult[]>): RunStatus {
  const allResults: ModelResult[] = []
  for (const arr of results.values()) {
    allResults.push(...arr)
  }
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
    const runId = generateId()
    const { settings } = get()

    // Build results map: each model gets one ModelResult per prompt
    const results = new Map<string, ModelResult[]>()
    for (const modelId of modelIds) {
      results.set(
        modelId,
        prompts.map(() => emptyModelResult(modelId))
      )
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

  updateModelResult: (runId, modelId, promptIndex, updates) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        const existing = run.results.get(modelId)
        if (!existing || !existing[promptIndex]) return run

        const newArr = [...existing]
        newArr[promptIndex] = { ...newArr[promptIndex], ...updates }
        const newResults = new Map(run.results)
        newResults.set(modelId, newArr)
        return { ...run, results: newResults }
      }),
    }))
  },

  appendContent: (runId, modelId, promptIndex, content, isThinking) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        const existing = run.results.get(modelId)
        if (!existing || !existing[promptIndex]) return run

        const result = existing[promptIndex]
        const newArr = [...existing]
        if (isThinking) {
          newArr[promptIndex] = { ...result, thinking: (result.thinking || '') + content }
        } else {
          newArr[promptIndex] = { ...result, content: result.content + content }
        }
        const newResults = new Map(run.results)
        newResults.set(modelId, newArr)
        return { ...run, results: newResults }
      }),
    }))
  },

  appendTokens: (runId, modelId, promptIndex, logprobs) => {
    set((state) => ({
      runs: state.runs.map((run) => {
        if (run.id !== runId) return run
        const existing = run.results.get(modelId)
        if (!existing || !existing[promptIndex]) return run

        const result = existing[promptIndex]
        const newTokens = logprobs.map((lp, i) =>
          tokenFromLogprob(lp, result.tokens.length + i)
        )
        const newArr = [...existing]
        newArr[promptIndex] = { ...result, tokens: [...result.tokens, ...newTokens] }
        const newResults = new Map(run.results)
        newResults.set(modelId, newArr)
        return { ...run, results: newResults }
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
          const currentResult = get().runs
            .find((r) => r.id === runId)
            ?.results.get(modelId)?.[promptIndex]
          if (!currentResult) return

          // First token: transition to streaming
          if (currentResult.status === 'loading') {
            firstTokenTime = Date.now()
            get().updateModelResult(runId, modelId, promptIndex, {
              status: 'streaming',
            })
          }

          get().appendContent(runId, modelId, promptIndex, token, false)
        },

        onThinking: (thinking) => {
          const currentResult = get().runs
            .find((r) => r.id === runId)
            ?.results.get(modelId)?.[promptIndex]
          if (!currentResult) return

          // Also transition to streaming on first thinking token
          if (currentResult.status === 'loading') {
            firstTokenTime = Date.now()
            get().updateModelResult(runId, modelId, promptIndex, {
              status: 'streaming',
            })
          }

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
