import { describe, it, expect } from 'vitest'
import { buildAPIMessages } from './messages'
import type { Message } from '../types/chat'

function makeMessage(overrides: Partial<Message> & { role: Message['role'] }): Message {
  const { id = 'msg-1', content = '', timestamp = Date.now(), ...rest } = overrides
  return { id, content, timestamp, ...rest }
}

describe('buildAPIMessages', () => {
  it('converts simple text messages', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hello' }),
      makeMessage({ id: '2', role: 'assistant', content: 'hi' }),
    ]
    const result = buildAPIMessages(messages)
    expect(result).toEqual([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
    ])
  })

  it('includes thinking on assistant messages', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'solve 2+2' }),
      makeMessage({ id: '2', role: 'assistant', content: '4', thinking: 'simple addition' }),
    ]
    const result = buildAPIMessages(messages)
    expect(result[1]).toEqual({
      role: 'assistant',
      content: '4',
      thinking: 'simple addition',
    })
  })

  it('does not include thinking if empty', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'assistant', content: 'hi', thinking: '' }),
    ]
    const result = buildAPIMessages(messages)
    expect(result[0]).toEqual({ role: 'assistant', content: 'hi' })
    expect(result[0]).not.toHaveProperty('thinking')
  })

  it('does not include thinking on user messages', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hello', thinking: 'hmm' } as any),
    ]
    const result = buildAPIMessages(messages)
    expect(result[0]).not.toHaveProperty('thinking')
  })

  it('handles images', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'what is this?', images: ['data:image/png;base64,abc'] }),
    ]
    const result = buildAPIMessages(messages)
    expect(result[0].content).toEqual([
      { type: 'text', text: 'what is this?' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,abc' } },
    ])
  })

  it('excludes a message by id', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hello' }),
      makeMessage({ id: '2', role: 'assistant', content: '' }),
    ]
    const result = buildAPIMessages(messages, { excludeId: '2' })
    expect(result).toHaveLength(1)
    expect(result[0].role).toBe('user')
  })

  it('prepends system prompt', () => {
    const messages: Message[] = [
      makeMessage({ id: '1', role: 'user', content: 'hello' }),
    ]
    const result = buildAPIMessages(messages, { systemPrompt: 'You are helpful.' })
    expect(result).toHaveLength(2)
    expect(result[0]).toEqual({ role: 'system', content: 'You are helpful.' })
  })

  it('handles empty messages array', () => {
    const result = buildAPIMessages([])
    expect(result).toEqual([])
  })

  it('handles systemPrompt with empty messages', () => {
    const result = buildAPIMessages([], { systemPrompt: 'Be kind' })
    expect(result).toEqual([{ role: 'system', content: 'Be kind' }])
  })
})
