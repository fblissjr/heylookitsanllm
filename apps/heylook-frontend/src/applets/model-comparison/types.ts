import type { TokenLogprob } from '../../types/api'
import type { SamplerSettings } from '../../types/settings'

export interface ComparisonToken {
  index: number
  token: string
  tokenId: number
  logprob: number
  probability: number // Math.exp(logprob)
  topLogprobs: TokenLogprob[]
}

export type RunStatus = 'idle' | 'running' | 'completed' | 'partial' | 'error'
export type ModelResultStatus = 'pending' | 'loading' | 'streaming' | 'completed' | 'error'

export interface ModelPerformance {
  ttft?: number // ms, time to first token (includes model load)
  tokensPerSecond?: number
  totalDuration?: number // ms
  promptTokens?: number
  completionTokens?: number
  thinkingTokens?: number
}

export interface ModelResult {
  modelId: string
  status: ModelResultStatus
  content: string
  thinking?: string
  tokens: ComparisonToken[] // only populated if logprobs enabled
  performance: ModelPerformance
  error?: string
}

export interface ComparisonRun {
  id: string
  mode: 'single' | 'batch'
  prompts: string[]
  selectedModelIds: string[]
  params: Partial<SamplerSettings>
  enableLogprobs: boolean
  topLogprobs: number
  // Map<modelId, results-per-prompt>
  // Single mode: results[modelId] has one entry
  // Batch mode: results[modelId] has N entries (one per prompt)
  results: Map<string, ModelResult[]>
  status: RunStatus
  createdAt: number
  completedAt?: number
}

export interface ComparisonSettings {
  samplerSettings: Partial<SamplerSettings>
  enableLogprobs: boolean
  topLogprobs: number
  mode: 'single' | 'batch'
}

// DuckDB persistence interface (stub for now, wire later)
export interface ComparisonPersistence {
  saveRun(run: ComparisonRun): Promise<void>
  loadRuns(): Promise<ComparisonRun[]>
  deleteRun(id: string): Promise<void>
}
