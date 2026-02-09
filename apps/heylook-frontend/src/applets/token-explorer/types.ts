import type { TokenLogprob } from '../../types/api'

export interface ExplorerToken {
  index: number
  token: string
  tokenId: number
  logprob: number
  probability: number // Math.exp(logprob), precomputed
  topLogprobs: TokenLogprob[]
}

export type RunStatus = 'idle' | 'streaming' | 'completed' | 'stopped' | 'error'

export interface ExplorerRun {
  id: string
  prompt: string
  model: string
  topLogprobs: number
  temperature: number
  maxTokens: number
  status: RunStatus
  tokens: ExplorerToken[]
  error?: string
  createdAt: number
  completedAt?: number
  totalDuration?: number
}
