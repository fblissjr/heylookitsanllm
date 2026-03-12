/**
 * Generate a short random ID, optionally prefixed.
 * Uses crypto.randomUUID when available, falls back to Math.random.
 */
export function generateId(prefix?: string): string {
  const raw =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

  return prefix ? `${prefix}-${raw}` : raw
}
