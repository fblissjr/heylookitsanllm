import type { TokenLogprob } from '../types/api'

/**
 * Shared token type for logprob visualization across applets.
 *
 * Replaces ExplorerToken and ComparisonToken which were identical.
 */
export interface LogprobToken {
  index: number
  token: string
  tokenId: number
  logprob: number
  probability: number // Math.exp(logprob), precomputed
  topLogprobs: TokenLogprob[]
}

/**
 * Convert a raw TokenLogprob from the API into a LogprobToken.
 */
export function tokenFromLogprob(logprob: TokenLogprob, index: number): LogprobToken {
  return {
    index,
    token: logprob.token,
    tokenId: logprob.token_id,
    logprob: logprob.logprob,
    probability: Math.exp(logprob.logprob),
    topLogprobs: logprob.top_logprobs ?? [],
  }
}

/**
 * Render whitespace tokens as visible Unicode symbols.
 */
export function displayToken(token: string): string {
  if (token === '\n') return '\u21B5' // return symbol
  if (token === '\t') return '\u2192' // right arrow
  if (token === ' ') return '\u00B7'  // middle dot
  return token
}
