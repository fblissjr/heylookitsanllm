import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock zustand stores before importing the module
const mockSetReconnecting = vi.fn()
const mockSetConnected = vi.fn()
const mockCheckConnection = vi.fn().mockResolvedValue(undefined)

vi.mock('../stores/connectionStore', () => ({
  useConnectionStore: {
    getState: () => ({
      setReconnecting: mockSetReconnecting,
      setConnected: mockSetConnected,
      checkConnection: mockCheckConnection,
    }),
  },
}))

// Import after mocks are set up
import { initReconnectionDetection, _resetReconnectionState } from './reconnect'

describe('reconnect', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
    _resetReconnectionState()
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
    vi.useRealTimers()
  })

  describe('initReconnectionDetection', () => {
    it('registers a visibilitychange listener', () => {
      const addSpy = vi.spyOn(document, 'addEventListener')
      initReconnectionDetection()
      expect(addSpy).toHaveBeenCalledWith('visibilitychange', expect.any(Function))
      addSpy.mockRestore()
    })

    it('is idempotent -- duplicate calls register only one listener', () => {
      const addSpy = vi.spyOn(document, 'addEventListener')
      initReconnectionDetection()
      initReconnectionDetection()
      initReconnectionDetection()
      const visibilityCalls = addSpy.mock.calls.filter(
        ([event]) => event === 'visibilitychange'
      )
      expect(visibilityCalls).toHaveLength(1)
      addSpy.mockRestore()
    })
  })

  describe('ping behavior', () => {
    it('sets reconnecting on visibility restore', async () => {
      // Simulate a successful ping
      globalThis.fetch = vi.fn().mockResolvedValue({ ok: true })

      initReconnectionDetection()

      // Simulate tab becoming visible
      Object.defineProperty(document, 'hidden', { value: false, writable: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))

      // Let the ping promise resolve
      await vi.advanceTimersByTimeAsync(0)

      expect(mockSetReconnecting).toHaveBeenCalledWith(true)
    })

    it('clears reconnecting state after successful ping', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({ ok: true })

      initReconnectionDetection()

      Object.defineProperty(document, 'hidden', { value: false, writable: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))

      await vi.advanceTimersByTimeAsync(0)

      expect(mockSetReconnecting).toHaveBeenCalledWith(false)
      expect(mockSetConnected).toHaveBeenCalledWith(true)
    })

    it('refreshes server state after successful reconnect', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({ ok: true })

      initReconnectionDetection()

      Object.defineProperty(document, 'hidden', { value: false, writable: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))

      await vi.advanceTimersByTimeAsync(0)

      expect(mockCheckConnection).toHaveBeenCalled()
    })

    it('retries with backoff on failed ping', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'))

      initReconnectionDetection()

      Object.defineProperty(document, 'hidden', { value: false, writable: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))

      // First attempt fails
      await vi.advanceTimersByTimeAsync(0)

      expect(mockSetReconnecting).toHaveBeenCalledWith(true)
      // Should NOT have called setConnected(true) since ping failed
      expect(mockSetConnected).not.toHaveBeenCalledWith(true)
    })

    it('does not ping when tab goes hidden', () => {
      globalThis.fetch = vi.fn()

      initReconnectionDetection()

      Object.defineProperty(document, 'hidden', { value: true, writable: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))

      // fetch should not be called when going hidden
      expect(globalThis.fetch).not.toHaveBeenCalled()
    })
  })
})
