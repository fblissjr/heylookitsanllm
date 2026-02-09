// Shared message building utilities
// Extracted from chatStore to be reusable across applets

import type { Message } from '../types/chat'
import type { APIMessage } from '../types/api'

/**
 * Build API-compatible messages from internal Message objects.
 *
 * Key behavior:
 * - Includes `thinking` field on assistant messages so edited thinking
 *   round-trips to the backend.
 * - Handles image content transformation (base64 data URLs -> image_url parts).
 * - Optionally excludes a specific message (e.g. an empty placeholder).
 * - Optionally prepends a system prompt.
 */
export function buildAPIMessages(
  messages: Message[],
  options?: { excludeId?: string; systemPrompt?: string }
): APIMessage[] {
  const { excludeId, systemPrompt } = options ?? {}

  const apiMessages: APIMessage[] = messages
    .filter(m => m.id !== excludeId)
    .map(m => {
      const base: APIMessage = {
        role: m.role,
        content: m.images && m.images.length > 0
          ? [
              { type: 'text' as const, text: m.content },
              ...m.images.map(img => ({ type: 'image_url' as const, image_url: { url: img } })),
            ]
          : m.content,
      }

      // Include thinking for assistant messages so edits round-trip
      if (m.role === 'assistant' && m.thinking) {
        base.thinking = m.thinking
      }

      return base
    })

  if (systemPrompt) {
    apiMessages.unshift({ role: 'system', content: systemPrompt })
  }

  return apiMessages
}
