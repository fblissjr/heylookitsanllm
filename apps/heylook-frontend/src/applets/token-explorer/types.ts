import type { LogprobToken } from '../../lib/tokens'

export type { LogprobToken as ExplorerToken }

export type RunStatus = 'idle' | 'streaming' | 'completed' | 'stopped' | 'error'

export interface ExplorerRun {
  id: string
  prompt: string
  model: string
  topLogprobs: number
  temperature: number
  maxTokens: number
  status: RunStatus
  tokens: LogprobToken[]
  error?: string
  createdAt: number
  completedAt?: number
  totalDuration?: number
}
