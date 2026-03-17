import { create } from 'zustand'
import { withDiagnostics } from './diagnosticMiddleware'

interface ConnectionState {
  /**
   * null = initial check in progress
   * true = connected
   * false = connection failed (initial or after retries exhausted)
   */
  isConnected: boolean | null
  /** True while actively trying to re-establish connection after tab restore */
  isReconnecting: boolean

  /** Run the initial connection check (fetch models + capabilities). */
  checkConnection: () => Promise<void>
  setReconnecting: (value: boolean) => void
  setConnected: (value: boolean | null) => void
}

export const useConnectionStore = create<ConnectionState>()(withDiagnostics('connection', (set) => ({
  isConnected: null,
  isReconnecting: false,

  checkConnection: async () => {
    // Lazy import to avoid circular dependency issues in tests
    const { useModelStore } = await import('./modelStore')
    try {
      await useModelStore.getState().initialize()
      set({ isConnected: true })
    } catch {
      set({ isConnected: false })
    }
  },

  setReconnecting: (value) => set({ isReconnecting: value }),
  setConnected: (value) => set({ isConnected: value }),
})))
