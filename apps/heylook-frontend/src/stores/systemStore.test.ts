import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { useSystemStore, SystemMetrics } from './systemStore'

// Mock fetch globally
const mockFetch = vi.fn()
global.fetch = mockFetch

describe('systemStore', () => {
  beforeEach(() => {
    // Reset store to initial state
    useSystemStore.setState({
      metrics: null,
      isPolling: false,
      error: null,
      lastUpdated: null,
    })
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    // Stop any polling that might be running
    useSystemStore.getState().stopPolling()
    vi.useRealTimers()
  })

  const mockMetricsResponse: SystemMetrics = {
    timestamp: '2024-01-15T10:30:00Z',
    system: {
      ram_used_gb: 8.2,
      ram_available_gb: 23.8,
      ram_total_gb: 32.0,
      cpu_percent: 12.5,
    },
    models: {
      'Qwen3-4B-mlx': {
        context_used: 1024,
        context_capacity: 32768,
        context_percent: 3.1,
        memory_mb: 2400,
        requests_active: 0,
      },
    },
  }

  describe('initial state', () => {
    it('has null metrics initially', () => {
      const { metrics } = useSystemStore.getState()
      expect(metrics).toBeNull()
    })

    it('is not polling initially', () => {
      const { isPolling } = useSystemStore.getState()
      expect(isPolling).toBe(false)
    })

    it('has no error initially', () => {
      const { error } = useSystemStore.getState()
      expect(error).toBeNull()
    })

    it('has null lastUpdated initially', () => {
      const { lastUpdated } = useSystemStore.getState()
      expect(lastUpdated).toBeNull()
    })
  })

  describe('fetchMetrics', () => {
    it('fetches and stores metrics on success', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      await useSystemStore.getState().fetchMetrics()

      const { metrics, error, lastUpdated } = useSystemStore.getState()
      expect(metrics).toEqual(mockMetricsResponse)
      expect(error).toBeNull()
      expect(lastUpdated).not.toBeNull()
    })

    it('sets error on fetch failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      })

      await useSystemStore.getState().fetchMetrics()

      const { error } = useSystemStore.getState()
      expect(error).toContain('500')
    })

    it('keeps existing metrics on error', async () => {
      // First set some metrics
      useSystemStore.setState({ metrics: mockMetricsResponse })

      // Then simulate an error
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
      })

      await useSystemStore.getState().fetchMetrics()

      const { metrics } = useSystemStore.getState()
      expect(metrics).toEqual(mockMetricsResponse)
    })

    it('handles network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      await useSystemStore.getState().fetchMetrics()

      const { error } = useSystemStore.getState()
      expect(error).toBe('Network error')
    })
  })

  describe('startPolling', () => {
    it('sets isPolling to true', () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      useSystemStore.getState().startPolling()

      const { isPolling } = useSystemStore.getState()
      expect(isPolling).toBe(true)
    })

    it('fetches metrics immediately on start', () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      useSystemStore.getState().startPolling()

      expect(mockFetch).toHaveBeenCalledWith('/v1/system/metrics')
    })

    it('restarts polling if called again (handles React StrictMode)', () => {
      // The implementation clears and restarts the interval on each call
      // to handle React StrictMode which may call startPolling twice
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      useSystemStore.getState().startPolling()
      expect(mockFetch).toHaveBeenCalledTimes(1)

      mockFetch.mockClear()
      useSystemStore.getState().startPolling()

      // Second call should also fetch (restarts polling)
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('polls at regular intervals', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      useSystemStore.getState().startPolling()

      // Initial fetch
      expect(mockFetch).toHaveBeenCalledTimes(1)

      // Advance 5 seconds
      await vi.advanceTimersByTimeAsync(5000)
      expect(mockFetch).toHaveBeenCalledTimes(2)

      // Advance another 5 seconds
      await vi.advanceTimersByTimeAsync(5000)
      expect(mockFetch).toHaveBeenCalledTimes(3)
    })
  })

  describe('stopPolling', () => {
    it('sets isPolling to false', () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      useSystemStore.getState().startPolling()
      useSystemStore.getState().stopPolling()

      const { isPolling } = useSystemStore.getState()
      expect(isPolling).toBe(false)
    })

    it('stops interval polling', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMetricsResponse),
      })

      useSystemStore.getState().startPolling()
      expect(mockFetch).toHaveBeenCalledTimes(1)

      useSystemStore.getState().stopPolling()

      // Advance time - should not fetch again
      await vi.advanceTimersByTimeAsync(10000)
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('is safe to call when not polling', () => {
      // Should not throw
      expect(() => useSystemStore.getState().stopPolling()).not.toThrow()
    })
  })

  describe('getModelMetrics', () => {
    it('returns null when no metrics exist', () => {
      const result = useSystemStore.getState().getModelMetrics('test-model')
      expect(result).toBeNull()
    })

    it('returns null for non-existent model', () => {
      useSystemStore.setState({ metrics: mockMetricsResponse })

      const result = useSystemStore.getState().getModelMetrics('non-existent')
      expect(result).toBeNull()
    })

    it('returns metrics for existing model', () => {
      useSystemStore.setState({ metrics: mockMetricsResponse })

      const result = useSystemStore.getState().getModelMetrics('Qwen3-4B-mlx')
      expect(result).toEqual({
        context_used: 1024,
        context_capacity: 32768,
        context_percent: 3.1,
        memory_mb: 2400,
        requests_active: 0,
      })
    })
  })
})
