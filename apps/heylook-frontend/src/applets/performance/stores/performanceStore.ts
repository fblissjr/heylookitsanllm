import { create } from 'zustand'
import type { SystemMetrics } from '../../../stores/systemStore'
import type { PerformanceProfile, TimeRange } from '../types'

interface PerformanceState {
  // Data
  systemMetrics: SystemMetrics | null
  profileData: PerformanceProfile | null
  timeRange: TimeRange

  // UI state
  isPolling: boolean
  isLoadingProfile: boolean
  analyticsEnabled: boolean | null // null=unknown, false=disabled, true=enabled
  error: string | null
  lastUpdated: number | null

  // Actions
  setTimeRange: (range: TimeRange) => void
  fetchSystemMetrics: () => Promise<void>
  fetchPerformanceProfile: () => Promise<void>
  startPolling: () => void
  stopPolling: () => void
  refresh: () => Promise<void>
}

// Module-level polling state (same pattern as systemStore.ts)
let metricsInterval: ReturnType<typeof setInterval> | null = null
let profileInterval: ReturnType<typeof setInterval> | null = null
const METRICS_POLL_MS = 5000  // 5s for real-time metrics
const PROFILE_POLL_MS = 30000 // 30s for analytics data

// HMR cleanup to prevent interval leaks during development
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    if (metricsInterval) {
      clearInterval(metricsInterval)
      metricsInterval = null
    }
    if (profileInterval) {
      clearInterval(profileInterval)
      profileInterval = null
    }
  })
}

export const usePerformanceStore = create<PerformanceState>((set, get) => ({
  systemMetrics: null,
  profileData: null,
  timeRange: '1h',

  isPolling: false,
  isLoadingProfile: false,
  analyticsEnabled: null,
  error: null,
  lastUpdated: null,

  fetchSystemMetrics: async () => {
    try {
      const response = await fetch('/v1/system/metrics')
      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.status}`)
      }
      const data: SystemMetrics = await response.json()
      set({
        systemMetrics: data,
        error: null,
        lastUpdated: Date.now(),
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error fetching metrics'
      set({ error: message })
      // Keep stale data visible on error
    }
  },

  fetchPerformanceProfile: async () => {
    const { timeRange, analyticsEnabled } = get()
    // Don't fetch if we already know analytics is disabled
    if (analyticsEnabled === false) return

    set({ isLoadingProfile: true })
    try {
      const response = await fetch(`/v1/performance/profile/${timeRange}`)
      if (response.status === 503) {
        set({
          analyticsEnabled: false,
          isLoadingProfile: false,
          profileData: null,
        })
        // Stop profile polling since analytics is disabled
        if (profileInterval) {
          clearInterval(profileInterval)
          profileInterval = null
        }
        return
      }
      if (!response.ok) {
        throw new Error(`Failed to fetch profile: ${response.status}`)
      }
      const data: PerformanceProfile = await response.json()
      set({
        profileData: data,
        analyticsEnabled: true,
        isLoadingProfile: false,
        error: null,
        lastUpdated: Date.now(),
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error fetching profile'
      set({ error: message, isLoadingProfile: false })
      // Keep stale data visible on error
    }
  },

  setTimeRange: (range) => {
    set({ timeRange: range })
    // Refetch profile with new range
    get().fetchPerformanceProfile()
  },

  startPolling: () => {
    // Clear any existing intervals first
    if (metricsInterval) {
      clearInterval(metricsInterval)
      metricsInterval = null
    }
    if (profileInterval) {
      clearInterval(profileInterval)
      profileInterval = null
    }

    const { fetchSystemMetrics, fetchPerformanceProfile } = get()
    set({ isPolling: true })

    // Fetch immediately on start
    fetchSystemMetrics()
    fetchPerformanceProfile()

    // Set up interval polling
    metricsInterval = setInterval(() => {
      fetchSystemMetrics()
    }, METRICS_POLL_MS)

    profileInterval = setInterval(() => {
      fetchPerformanceProfile()
    }, PROFILE_POLL_MS)
  },

  stopPolling: () => {
    if (metricsInterval) {
      clearInterval(metricsInterval)
      metricsInterval = null
    }
    if (profileInterval) {
      clearInterval(profileInterval)
      profileInterval = null
    }
    set({ isPolling: false })
  },

  refresh: async () => {
    await Promise.all([
      get().fetchSystemMetrics(),
      get().fetchPerformanceProfile(),
    ])
  },
}))
