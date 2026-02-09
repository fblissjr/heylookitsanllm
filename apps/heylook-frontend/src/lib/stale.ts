// Stale message detection
// Pure functions -- no stored flags, computed from timestamps.

import type { Message } from '../types/chat'

/**
 * Returns true if any upstream message (lower index) has been edited
 * after the given message was created. A message is "stale" when the
 * context it was generated from has since changed.
 */
export function isMessageStale(messages: Message[], index: number): boolean {
  const target = messages[index]
  if (!target) return false

  for (let i = 0; i < index; i++) {
    const upstream = messages[i]
    if (upstream.editedAt && upstream.editedAt > target.timestamp) {
      return true
    }
  }
  return false
}

/**
 * Returns the set of indices for all stale messages in the array.
 */
export function getStaleIndices(messages: Message[]): Set<number> {
  const stale = new Set<number>()
  for (let i = 1; i < messages.length; i++) {
    if (isMessageStale(messages, i)) {
      stale.add(i)
    }
  }
  return stale
}
