import { create } from 'zustand'

// System resource metrics (RAM, CPU)
export interface SystemResourceMetrics {
  ram_used_gb: number
  ram_available_gb: number
  ram_total_gb: number
  cpu_percent: number
}

// Per-model metrics
export interface ModelMetrics {
  context_used: number
  context_capacity: number
  context_percent: number
  memory_mb: number
  requests_active: number
}

// Full system metrics response from API
export interface SystemMetrics {
  timestamp: string
  system: SystemResourceMetrics
  models: Record<string, ModelMetrics>
}

interface SystemState {
  // Data
  metrics: SystemMetrics | null
  isPolling: boolean
  error: string | null
  lastUpdated: number | null

  // Actions
  startPolling: () => void
  stopPolling: () => void
  fetchMetrics: () => Promise<void>

  // Getters
  getModelMetrics: (modelId: string) => ModelMetrics | null
}

// Module-level polling state (outside store to avoid issues with setInterval)
let pollingInterval: ReturnType<typeof setInterval> | null = null
const POLL_INTERVAL_MS = 5000 // 5 seconds

export const useSystemStore = create<SystemState>((set, get) => ({
  metrics: null,
  isPolling: false,
  error: null,
  lastUpdated: null,

  fetchMetrics: async () => {
    try {
      const response = await fetch('/v1/system/metrics')
      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.status}`)
      }
      const data: SystemMetrics = await response.json()
      set({
        metrics: data,
        error: null,
        lastUpdated: Date.now(),
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error fetching metrics'
      set({ error: message })
      // Don't clear existing metrics on error - keep stale data visible
    }
  },

  startPolling: () => {
    const { isPolling, fetchMetrics } = get()
    if (isPolling) return // Already polling

    set({ isPolling: true })

    // Fetch immediately on start
    fetchMetrics()

    // Set up interval polling
    pollingInterval = setInterval(() => {
      fetchMetrics()
    }, POLL_INTERVAL_MS)
  },

  stopPolling: () => {
    if (pollingInterval) {
      clearInterval(pollingInterval)
      pollingInterval = null
    }
    set({ isPolling: false })
  },

  getModelMetrics: (modelId: string) => {
    const { metrics } = get()
    if (!metrics || !metrics.models) return null
    return metrics.models[modelId] || null
  },
}))
