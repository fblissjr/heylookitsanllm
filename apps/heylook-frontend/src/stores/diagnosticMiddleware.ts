// Zustand middleware that logs state transitions to DiagnosticLogger

import type { StateCreator, StoreMutatorIdentifier } from 'zustand'
import { logger } from '../lib/diagnostics'

type DiagnosticMiddleware = <
  T,
  Mps extends [StoreMutatorIdentifier, unknown][] = [],
  Mcs extends [StoreMutatorIdentifier, unknown][] = [],
>(
  storeName: string,
  f: StateCreator<T, Mps, Mcs>,
) => StateCreator<T, Mps, Mcs>

/**
 * Wraps a Zustand store creator to log state transitions via DiagnosticLogger.
 *
 * Uses referential equality (===) only for O(1) diff detection on arrays/objects.
 * Only logs keys whose top-level value changed.
 *
 * Usage:
 *   create<MyState>()(withDiagnostics('myStore', (set, get) => ({ ... })))
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const withDiagnostics: DiagnosticMiddleware = (storeName, f) => (set, get, api) => {
  const wrappedSet = ((...args: any[]) => {
    const prev = get()
    ;(set as any)(...args)
    const next = get()

    // Diff top-level keys using referential equality
    const changed: Record<string, unknown> = {}
    let hasChanges = false
    for (const key of Object.keys(next as object)) {
      if ((prev as any)[key] !== (next as any)[key]) {
        const val = (next as any)[key]
        // Don't log functions
        if (typeof val !== 'function') {
          changed[key] = val
          hasChanges = true
        }
      }
    }

    if (hasChanges) {
      logger.debug('store_transition', 'store', {
        store: storeName,
        changed,
      })
    }
  }) as typeof set

  return f(wrappedSet, get, api)
}
