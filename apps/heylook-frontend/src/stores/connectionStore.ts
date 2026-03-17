import { create } from 'zustand'

interface ConnectionState {
  /** True while actively trying to re-establish connection after tab restore */
  isReconnecting: boolean
  /** True when connection is confirmed alive */
  isConnected: boolean

  setReconnecting: (value: boolean) => void
  setConnected: (value: boolean) => void
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  isReconnecting: false,
  isConnected: true,

  setReconnecting: (value) => set({ isReconnecting: value }),
  setConnected: (value) => set({ isConnected: value }),
}))
