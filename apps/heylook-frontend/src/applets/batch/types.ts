import type { SamplerSettings } from '../../types/settings'

export type BatchJobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface BatchJobResult {
  prompt: string
  response: string
  thinking?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export interface BatchJob {
  id: string
  status: BatchJobStatus
  prompts: string[]
  model: string
  params: Partial<SamplerSettings>
  results: BatchJobResult[]
  error?: string
  createdAt: number
  completedAt?: number
  totalTokens?: number
  totalDuration?: number
}

export type BatchView = 'create' | 'dashboard'
