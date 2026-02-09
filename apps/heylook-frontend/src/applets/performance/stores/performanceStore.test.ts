import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { usePerformanceStore } from './performanceStore'
import type { SystemMetrics } from '../../../stores/systemStore'
import type { PerformanceProfile } from '../types'

const mockSystemMetrics: SystemMetrics = {
  timestamp: '2026-01-01T00:00:00Z',
  system: {
    ram_used_gb: 12.5,
    ram_available_gb: 19.5,
    ram_total_gb: 32,
    cpu_percent: 45.2,
  },
  models: {
    'test-model': {
      context_used: 512,
      context_capacity: 4096,
      context_percent: 12.5,
      memory_mb: 3200,
      requests_active: 1,
    },
  },
}

const mockProfile: PerformanceProfile = {
  time_range: '1h',
  timing_breakdown: [
    { operation: 'token_generation', avg_time_ms: 150, count: 42, percentage: 0 },
    { operation: 'queue', avg_time_ms: 12, count: 42, percentage: 0 },
  ],
  resource_timeline: [
    {
      timestamp: '2026-01-01T00:00:00',
      memory_gb: 12.5,
      gpu_percent: 80,
      tokens_per_second: 25.3,
      requests: 10,
    },
  ],
  bottlenecks: [
    {
      model: 'test-model',
      avg_total_ms: 250,
      breakdown: {
        queue: 10,
        model_load: 0,
        image_processing: 0,
        token_generation: 200,
        first_token: 40,
      },
      request_count: 42,
    },
  ],
  trends: [
    {
      hour: '2026-01-01T00:00:00',
      response_time_ms: 250,
      tokens_per_second: 25.3,
      requests: 10,
      errors: 0,
      response_time_change: 0,
      tps_change: 0,
    },
  ],
}

function resetStore() {
  usePerformanceStore.setState({
    systemMetrics: null,
    profileData: null,
    timeRange: '1h',
    isPolling: false,
    isLoadingProfile: false,
    analyticsEnabled: null,
    error: null,
    lastUpdated: null,
  })
}

describe('performanceStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    resetStore()
  })

  afterEach(() => {
    usePerformanceStore.getState().stopPolling()
    vi.useRealTimers()
  })

  describe('initial state', () => {
    it('starts with null metrics and profile', () => {
      const state = usePerformanceStore.getState()
      expect(state.systemMetrics).toBeNull()
      expect(state.profileData).toBeNull()
    })

    it('starts with 1h time range', () => {
      expect(usePerformanceStore.getState().timeRange).toBe('1h')
    })

    it('starts with unknown analytics status', () => {
      expect(usePerformanceStore.getState().analyticsEnabled).toBeNull()
    })

    it('starts not polling', () => {
      expect(usePerformanceStore.getState().isPolling).toBe(false)
    })
  })

  describe('fetchSystemMetrics', () => {
    it('sets metrics data on success', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      await usePerformanceStore.getState().fetchSystemMetrics()

      const state = usePerformanceStore.getState()
      expect(state.systemMetrics).toEqual(mockSystemMetrics)
      expect(state.error).toBeNull()
      expect(state.lastUpdated).not.toBeNull()
    })

    it('sets error on fetch failure', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(null, { status: 500 })
      )

      await usePerformanceStore.getState().fetchSystemMetrics()

      expect(usePerformanceStore.getState().error).toBe('Failed to fetch metrics: 500')
    })

    it('keeps stale metrics on error', async () => {
      // Set initial metrics
      usePerformanceStore.setState({ systemMetrics: mockSystemMetrics })

      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(null, { status: 500 })
      )

      await usePerformanceStore.getState().fetchSystemMetrics()

      // Metrics should still be present
      expect(usePerformanceStore.getState().systemMetrics).toEqual(mockSystemMetrics)
    })

    it('handles network errors', async () => {
      vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('Network error'))

      await usePerformanceStore.getState().fetchSystemMetrics()

      expect(usePerformanceStore.getState().error).toBe('Network error')
    })
  })

  describe('fetchPerformanceProfile', () => {
    it('sets profile data on success', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify(mockProfile), { status: 200 })
      )

      await usePerformanceStore.getState().fetchPerformanceProfile()

      const state = usePerformanceStore.getState()
      expect(state.profileData).toEqual(mockProfile)
      expect(state.analyticsEnabled).toBe(true)
      expect(state.isLoadingProfile).toBe(false)
    })

    it('detects analytics disabled on 503', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Analytics not enabled' }), { status: 503 })
      )

      await usePerformanceStore.getState().fetchPerformanceProfile()

      const state = usePerformanceStore.getState()
      expect(state.analyticsEnabled).toBe(false)
      expect(state.profileData).toBeNull()
      expect(state.isLoadingProfile).toBe(false)
    })

    it('skips fetch when analyticsEnabled is false', async () => {
      usePerformanceStore.setState({ analyticsEnabled: false })
      const fetchSpy = vi.spyOn(globalThis, 'fetch')

      await usePerformanceStore.getState().fetchPerformanceProfile()

      expect(fetchSpy).not.toHaveBeenCalled()
    })

    it('uses current timeRange in URL', async () => {
      usePerformanceStore.setState({ timeRange: '24h' })
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify(mockProfile), { status: 200 })
      )

      await usePerformanceStore.getState().fetchPerformanceProfile()

      expect(fetchSpy).toHaveBeenCalledWith('/v1/performance/profile/24h')
    })

    it('keeps stale profile on error', async () => {
      usePerformanceStore.setState({ profileData: mockProfile })

      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(null, { status: 500 })
      )

      await usePerformanceStore.getState().fetchPerformanceProfile()

      expect(usePerformanceStore.getState().profileData).toEqual(mockProfile)
    })

    it('sets isLoadingProfile during fetch', async () => {
      let resolvePromise: (value: Response) => void
      const promise = new Promise<Response>((resolve) => {
        resolvePromise = resolve
      })
      vi.spyOn(globalThis, 'fetch').mockReturnValueOnce(promise)

      const fetchPromise = usePerformanceStore.getState().fetchPerformanceProfile()
      expect(usePerformanceStore.getState().isLoadingProfile).toBe(true)

      resolvePromise!(new Response(JSON.stringify(mockProfile), { status: 200 }))
      await fetchPromise

      expect(usePerformanceStore.getState().isLoadingProfile).toBe(false)
    })
  })

  describe('setTimeRange', () => {
    it('updates timeRange', () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockProfile), { status: 200 })
      )

      usePerformanceStore.getState().setTimeRange('7d')

      expect(usePerformanceStore.getState().timeRange).toBe('7d')
    })

    it('triggers profile refetch', () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockProfile), { status: 200 })
      )

      usePerformanceStore.getState().setTimeRange('6h')

      expect(fetchSpy).toHaveBeenCalledWith('/v1/performance/profile/6h')
    })
  })

  describe('startPolling / stopPolling', () => {
    it('sets isPolling to true', () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()

      expect(usePerformanceStore.getState().isPolling).toBe(true)
    })

    it('fetches immediately on start', () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()

      // Should have been called for both metrics and profile
      expect(fetchSpy).toHaveBeenCalledWith('/v1/system/metrics')
      expect(fetchSpy).toHaveBeenCalledWith('/v1/performance/profile/1h')
    })

    it('stopPolling sets isPolling to false', () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()
      usePerformanceStore.getState().stopPolling()

      expect(usePerformanceStore.getState().isPolling).toBe(false)
    })

    it('polls metrics at 5s intervals', () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()
      fetchSpy.mockClear()

      vi.advanceTimersByTime(5000)
      expect(fetchSpy).toHaveBeenCalledWith('/v1/system/metrics')
    })

    it('polls profile at 30s intervals', () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockProfile), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()
      fetchSpy.mockClear()

      vi.advanceTimersByTime(30000)
      expect(fetchSpy).toHaveBeenCalledWith('/v1/performance/profile/1h')
    })

    it('clears intervals on stop', () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()
      usePerformanceStore.getState().stopPolling()
      fetchSpy.mockClear()

      vi.advanceTimersByTime(35000)
      expect(fetchSpy).not.toHaveBeenCalled()
    })

    it('prevents duplicate intervals on double start', () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockSystemMetrics), { status: 200 })
      )

      usePerformanceStore.getState().startPolling()
      usePerformanceStore.getState().startPolling()
      fetchSpy.mockClear()

      vi.advanceTimersByTime(5000)
      // Should only fire once per interval, not twice
      const metricsCallCount = fetchSpy.mock.calls.filter(
        (c) => c[0] === '/v1/system/metrics'
      ).length
      expect(metricsCallCount).toBe(1)
    })
  })

  describe('refresh', () => {
    it('fetches both metrics and profile', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch')
        .mockResolvedValueOnce(new Response(JSON.stringify(mockSystemMetrics), { status: 200 }))
        .mockResolvedValueOnce(new Response(JSON.stringify(mockProfile), { status: 200 }))

      await usePerformanceStore.getState().refresh()

      expect(fetchSpy).toHaveBeenCalledWith('/v1/system/metrics')
      expect(fetchSpy).toHaveBeenCalledWith('/v1/performance/profile/1h')
    })
  })
})
