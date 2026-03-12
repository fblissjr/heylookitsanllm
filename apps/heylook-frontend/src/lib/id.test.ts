import { describe, it, expect, vi, afterEach } from 'vitest'
import { generateId } from './id'

describe('generateId', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns a string', () => {
    expect(typeof generateId()).toBe('string')
  })

  it('generates unique IDs', () => {
    const ids = new Set(Array.from({ length: 100 }, () => generateId()))
    expect(ids.size).toBe(100)
  })

  it('includes prefix when provided', () => {
    const id = generateId('run')
    expect(id).toMatch(/^run-/)
  })

  describe('with crypto.randomUUID available', () => {
    it('returns a UUID format', () => {
      const id = generateId()
      expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
    })

    it('prepends prefix to UUID', () => {
      const id = generateId('test')
      expect(id).toMatch(/^test-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
    })
  })

  describe('fallback when crypto.randomUUID is unavailable', () => {
    it('returns timestamp-random format', () => {
      const original = crypto.randomUUID
      // @ts-expect-error -- removing randomUUID to test fallback
      crypto.randomUUID = undefined
      try {
        const id = generateId()
        expect(id).toMatch(/^\d+-[a-z0-9]+$/)
      } finally {
        crypto.randomUUID = original
      }
    })

    it('prepends prefix to fallback format', () => {
      const original = crypto.randomUUID
      // @ts-expect-error -- removing randomUUID to test fallback
      crypto.randomUUID = undefined
      try {
        const id = generateId('msg')
        expect(id).toMatch(/^msg-\d+-[a-z0-9]+$/)
      } finally {
        crypto.randomUUID = original
      }
    })
  })
})
