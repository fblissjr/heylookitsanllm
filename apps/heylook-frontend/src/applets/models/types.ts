/** Types for the Models management applet. */

export interface AdminModelConfig {
  id: string
  provider: string
  description?: string
  tags: string[]
  enabled: boolean
  capabilities: string[]
  config: Record<string, unknown>
  loaded: boolean
}

export interface ScannedModel {
  id: string
  path: string
  provider: 'mlx' | 'gguf'
  size_gb: number
  vision: boolean
  quantization?: string
  already_configured: boolean
  tags: string[]
  description: string
}

export interface ProfileInfo {
  name: string
  description: string
}

export interface ScanRequest {
  paths: string[]
  scan_hf_cache: boolean
}

export interface ImportRequest {
  models: Record<string, unknown>[]
  profile?: string
}

export interface ModelUpdatePayload {
  description?: string
  tags?: string[]
  enabled?: boolean
  capabilities?: string[]
  config?: Record<string, unknown>
}

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

export type ModelFilter = {
  provider: string[]
  status: string[]  // 'loaded' | 'available' | 'disabled'
  capability: string[]
  tag: string[]
}

export type SortField = 'name' | 'provider' | 'status'
export type SortDirection = 'asc' | 'desc'
export interface SortConfig { field: SortField; direction: SortDirection }
