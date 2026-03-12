/**
 * Shared probability-to-color mapping for token visualization.
 *
 * Used by TokenChip, TokenDetail, and LogprobsDetail across applets.
 */

/**
 * Map a probability (0-1) to a background color for token chips.
 * 0 = red, 0.5 = yellow, 1 = green.
 */
export function probabilityToColor(probability: number, isDark: boolean): string {
  const p = Math.max(0, Math.min(1, probability))
  const hue = p * 120 // 0=red -> 60=yellow -> 120=green
  const saturation = 65
  const lightness = isDark ? 25 + p * 10 : 85 - p * 15
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

/**
 * Map a probability (0-1) to a bar color for alternative token bars.
 * Slightly brighter than chip colors to contrast against bar backgrounds.
 */
export function probabilityToBarColor(probability: number, isDark: boolean): string {
  const p = Math.max(0, Math.min(1, probability))
  const hue = p * 120
  const saturation = 65
  const lightness = isDark ? 35 + p * 10 : 75 - p * 15
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}
