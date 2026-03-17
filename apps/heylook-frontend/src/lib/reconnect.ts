import { useConnectionStore } from '../stores/connectionStore'
import { useModelStore } from '../stores/modelStore'

const PING_TIMEOUT_MS = 3000
const MAX_BACKOFF_MS = 10000
const INITIAL_BACKOFF_MS = 1000

let retryTimer: ReturnType<typeof setTimeout> | null = null
let currentBackoff = INITIAL_BACKOFF_MS

/**
 * Ping the backend to check if the connection is alive.
 * Uses /v1/models since it's lightweight and already exists.
 */
async function ping(): Promise<boolean> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), PING_TIMEOUT_MS)

  try {
    const response = await fetch('/v1/models', { signal: controller.signal })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

function cancelRetry() {
  if (retryTimer) {
    clearTimeout(retryTimer)
    retryTimer = null
  }
}

async function attemptReconnect() {
  const { setReconnecting, setConnected } = useConnectionStore.getState()

  setReconnecting(true)
  const alive = await ping()

  if (alive) {
    setReconnecting(false)
    setConnected(true)
    currentBackoff = INITIAL_BACKOFF_MS

    // Refresh the model list since state may have changed while tab was frozen
    useModelStore.getState().fetchModels().catch(() => {})
    return
  }

  // Schedule retry with exponential backoff
  retryTimer = setTimeout(attemptReconnect, currentBackoff)
  currentBackoff = Math.min(currentBackoff * 2, MAX_BACKOFF_MS)
}

/**
 * Initialize reconnection detection.
 * Listens for visibilitychange (hidden -> visible) and pings the backend.
 * Call once at app startup.
 */
export function initReconnectionDetection() {
  if (typeof document === 'undefined') return

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      // Tab going to background -- cancel any in-progress retry
      cancelRetry()
      return
    }

    // Tab restored -- check if backend is still reachable
    attemptReconnect()
  })
}
