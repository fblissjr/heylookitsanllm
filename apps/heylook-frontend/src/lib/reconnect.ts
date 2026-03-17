import { useConnectionStore } from '../stores/connectionStore'

const PING_TIMEOUT_MS = 3000
const MAX_BACKOFF_MS = 10000
const INITIAL_BACKOFF_MS = 1000

let retryTimer: ReturnType<typeof setTimeout> | null = null
let currentBackoff = INITIAL_BACKOFF_MS
let initialized = false
let visibilityHandler: (() => void) | null = null

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

    // Refresh model list + capabilities via the shared initialization path
    useConnectionStore.getState().checkConnection().catch(() => {})

    // Clean up dead streams: if the tab was frozen mid-generation, the
    // ReadableStream is dead. Stop generation so the UI doesn't show a
    // spinner forever. Fire-and-forget to stay off the critical path.
    import('../applets/chat/stores/chatStore').then(({ useChatStore }) => {
      const chatState = useChatStore.getState()
      if (chatState.streaming.isStreaming) {
        chatState.stopGeneration()
      }
    }).catch(() => {
      // chatStore not loaded yet -- nothing to clean up
    })

    return
  }

  // Schedule retry with exponential backoff
  retryTimer = setTimeout(attemptReconnect, currentBackoff)
  currentBackoff = Math.min(currentBackoff * 2, MAX_BACKOFF_MS)
}

/**
 * Initialize reconnection detection.
 * Listens for visibilitychange (hidden -> visible) and pings the backend.
 * Call once at app startup. Idempotent -- duplicate calls are ignored.
 */
export function initReconnectionDetection() {
  if (initialized || typeof document === 'undefined') return
  initialized = true

  visibilityHandler = () => {
    if (document.hidden) {
      // Tab going to background -- cancel any in-progress retry
      cancelRetry()
      return
    }

    // Tab restored -- check if backend is still reachable
    attemptReconnect()
  }
  document.addEventListener('visibilitychange', visibilityHandler)
}

/** Reset module state. Exposed for tests and HMR. */
export function _resetReconnectionState() {
  cancelRetry()
  if (visibilityHandler) {
    document.removeEventListener('visibilitychange', visibilityHandler)
    visibilityHandler = null
  }
  initialized = false
}

if (import.meta.hot) {
  import.meta.hot.dispose(_resetReconnectionState)
}
