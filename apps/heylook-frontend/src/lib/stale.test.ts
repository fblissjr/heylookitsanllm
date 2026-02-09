import { describe, it, expect } from 'vitest'
import { isMessageStale, getStaleIndices } from './stale'
import type { Message } from '../types/chat'

function makeMessage(overrides: Partial<Message> & { role: Message['role'] }): Message {
  const { id = 'msg-1', content = '', timestamp = 1000, ...rest } = overrides
  return { id, content, timestamp, ...rest }
}

describe('isMessageStale', () => {
  it('returns false when no upstream edits', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hi', timestamp: 1000 }),
      makeMessage({ id: '2', role: 'assistant', content: 'hey', timestamp: 2000 }),
    ]
    expect(isMessageStale(messages, 1)).toBe(false)
  })

  it('returns true when upstream message edited after target was created', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'edited', timestamp: 1000, editedAt: 3000 }),
      makeMessage({ id: '2', role: 'assistant', content: 'hey', timestamp: 2000 }),
    ]
    expect(isMessageStale(messages, 1)).toBe(true)
  })

  it('returns false when upstream edit happened before target was created', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'edited', timestamp: 1000, editedAt: 1500 }),
      makeMessage({ id: '2', role: 'assistant', content: 'hey', timestamp: 2000 }),
    ]
    expect(isMessageStale(messages, 1)).toBe(false)
  })

  it('returns false for index 0', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hi', timestamp: 1000 }),
    ]
    expect(isMessageStale(messages, 0)).toBe(false)
  })

  it('returns false for out of bounds index', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hi', timestamp: 1000 }),
    ]
    expect(isMessageStale(messages, 5)).toBe(false)
  })

  it('cascades through multiple messages', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'first', timestamp: 1000, editedAt: 5000 }),
      makeMessage({ id: '2', role: 'assistant', content: 'reply 1', timestamp: 2000 }),
      makeMessage({ id: '3', role: 'user', content: 'second', timestamp: 3000 }),
      makeMessage({ id: '4', role: 'assistant', content: 'reply 2', timestamp: 4000 }),
    ]
    // All after index 0 are stale because msg 0 was edited at 5000
    expect(isMessageStale(messages, 1)).toBe(true)
    expect(isMessageStale(messages, 2)).toBe(true)
    expect(isMessageStale(messages, 3)).toBe(true)
  })
})

describe('getStaleIndices', () => {
  it('returns empty set when no edits', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', timestamp: 1000 }),
      makeMessage({ id: '2', role: 'assistant', timestamp: 2000 }),
    ]
    expect(getStaleIndices(messages).size).toBe(0)
  })

  it('returns all downstream indices after an edit', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', timestamp: 1000, editedAt: 5000 }),
      makeMessage({ id: '2', role: 'assistant', timestamp: 2000 }),
      makeMessage({ id: '3', role: 'user', timestamp: 3000 }),
    ]
    const stale = getStaleIndices(messages)
    expect(stale.has(1)).toBe(true)
    expect(stale.has(2)).toBe(true)
    expect(stale.has(0)).toBe(false)
  })

  it('returns empty set for single message', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', timestamp: 1000 }),
    ]
    expect(getStaleIndices(messages).size).toBe(0)
  })
})
