import { describe, it, expect } from 'vitest'
import { tokenFromLogprob, displayToken } from './tokens'
import type { TokenLogprob } from '../types/api'

describe('tokenFromLogprob', () => {
  it('converts a TokenLogprob to a LogprobToken', () => {
    const logprob: TokenLogprob = {
      token: 'hello',
      token_id: 42,
      logprob: -0.5,
      bytes: [],
      top_logprobs: [],
    }
    const result = tokenFromLogprob(logprob, 3)

    expect(result.index).toBe(3)
    expect(result.token).toBe('hello')
    expect(result.tokenId).toBe(42)
    expect(result.logprob).toBe(-0.5)
    expect(result.probability).toBeCloseTo(Math.exp(-0.5))
    expect(result.topLogprobs).toEqual([])
  })

  it('handles undefined top_logprobs', () => {
    const logprob = {
      token: 'x',
      token_id: 1,
      logprob: 0,
      bytes: [],
      top_logprobs: undefined,
    } as unknown as TokenLogprob
    const result = tokenFromLogprob(logprob, 0)
    expect(result.topLogprobs).toEqual([])
  })

  it('passes through top_logprobs when present', () => {
    const alts: TokenLogprob[] = [
      { token: 'a', token_id: 1, logprob: -0.1, bytes: [], top_logprobs: [] },
    ]
    const logprob: TokenLogprob = {
      token: 'b',
      token_id: 2,
      logprob: -0.3,
      bytes: [],
      top_logprobs: alts,
    }
    const result = tokenFromLogprob(logprob, 0)
    expect(result.topLogprobs).toEqual(alts)
  })
})

describe('displayToken', () => {
  it('replaces newline with return symbol', () => {
    expect(displayToken('\n')).toBe('\u21B5')
  })

  it('replaces tab with right arrow', () => {
    expect(displayToken('\t')).toBe('\u2192')
  })

  it('replaces space with middle dot', () => {
    expect(displayToken(' ')).toBe('\u00B7')
  })

  it('passes through normal text unchanged', () => {
    expect(displayToken('hello')).toBe('hello')
  })

  it('passes through empty string', () => {
    expect(displayToken('')).toBe('')
  })
})
