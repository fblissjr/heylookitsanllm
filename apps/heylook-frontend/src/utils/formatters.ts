// Shared formatting utilities

/**
 * Format milliseconds to human-readable duration
 * @example formatDuration(500) => "500ms"
 * @example formatDuration(2500) => "2.50s"
 */
export function formatDuration(ms?: number): string {
  if (ms === undefined) return '-'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

/**
 * Format tokens per second
 * @example formatTPS(25.5) => "25.5 tok/s"
 */
export function formatTPS(tps?: number): string {
  if (tps === undefined) return '-'
  return `${tps.toFixed(1)} tok/s`
}

/**
 * Format token count with commas
 * @example formatTokens(1234) => "1,234 tokens"
 */
export function formatTokens(count?: number): string {
  if (count === undefined) return '-'
  return `${count.toLocaleString()} tokens`
}

/**
 * Format gigabytes for display
 * @example formatGB(8.2) => "8.2G"
 * @example formatGB(16) => "16G"
 */
export function formatGB(gb: number): string {
  if (gb >= 10) return `${gb.toFixed(0)}G`
  return `${gb.toFixed(1)}G`
}

/**
 * Format large numbers with commas
 * @example formatNumber(1234567) => "1,234,567"
 */
export function formatNumber(n: number): string {
  return n.toLocaleString()
}

/**
 * Truncate string with ellipsis
 * @example truncateString("hello world", 8) => "hello..."
 */
export function truncateString(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str
  return str.slice(0, maxLen - 1) + '...'
}
