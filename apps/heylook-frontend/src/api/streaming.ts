// SSE Streaming API Client

import type { StreamChunk, Usage, ChatCompletionRequest } from '../types/api'

export interface StreamCallbacks {
  onToken: (token: string) => void
  onThinking?: (thinking: string) => void
  onComplete: (usage?: Usage) => void
  onError: (error: Error) => void
}

export async function streamChat(
  request: ChatCompletionRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const { onToken, onThinking, onComplete, onError } = callbacks

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...request,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    if (!response.body) {
      throw new Error('No response body for streaming')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        // Process any remaining buffer
        if (buffer.trim()) {
          processLines(buffer.split('\n'), onToken, onThinking, onComplete)
        }
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete line in buffer

      processLines(lines, onToken, onThinking, onComplete)
    }

    // If we didn't receive a [DONE] signal, still call onComplete
    onComplete()
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        // User cancelled, not an error
        onComplete()
        return
      }
      onError(error)
    } else {
      onError(new Error('Unknown streaming error'))
    }
  }
}

function processLines(
  lines: string[],
  onToken: (token: string) => void,
  onThinking: ((thinking: string) => void) | undefined,
  onComplete: (usage?: Usage) => void
): void {
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.startsWith('data: ')) continue

    const data = trimmed.slice(6) // Remove 'data: ' prefix

    if (data === '[DONE]') {
      return
    }

    try {
      const chunk: StreamChunk = JSON.parse(data)
      const delta = chunk.choices?.[0]?.delta

      if (delta?.content) {
        onToken(delta.content)
      }

      if (delta?.thinking && onThinking) {
        onThinking(delta.thinking)
      }

      // Check for usage in final chunk
      if (chunk.usage) {
        onComplete(chunk.usage)
        return
      }
    } catch {
      // Skip invalid JSON chunks
    }
  }
}

// Non-streaming version for comparison
export async function chatCompletion(
  request: ChatCompletionRequest
): Promise<{ content: string; thinking?: string; usage?: Usage }> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...request,
      stream: false,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
  }

  const data = await response.json()
  const message = data.choices?.[0]?.message

  return {
    content: message?.content || '',
    thinking: message?.thinking,
    usage: data.usage,
  }
}
