// SSE Streaming API Client

import type {
  StreamChunk,
  Usage,
  EnhancedUsage,
  GenerationTiming,
  GenerationConfig,
  ChatCompletionRequest
} from '../types/api'

// Enhanced completion data from final SSE chunk
export interface StreamCompletionData {
  usage?: EnhancedUsage
  timing?: GenerationTiming
  generationConfig?: GenerationConfig
  stopReason?: string
}

export interface StreamCallbacks {
  onToken: (token: string, rawEvent?: string) => void
  onThinking?: (thinking: string, rawEvent?: string) => void
  onComplete: (data?: StreamCompletionData) => void
  onError: (error: Error) => void
  onRawEvent?: (event: string) => void  // For debugging raw SSE data
}

export async function streamChat(
  request: ChatCompletionRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const { onToken, onThinking, onComplete, onError } = callbacks

  // Track whether onComplete was already called with data to prevent double calls
  let completedWithData = false
  const wrappedOnComplete = (data?: StreamCompletionData) => {
    if (data) {
      completedWithData = true
    }
    onComplete(data)
  }

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

    const { onRawEvent } = callbacks

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        // Process any remaining buffer
        if (buffer.trim()) {
          processLines(buffer.split('\n'), onToken, onThinking, wrappedOnComplete, onRawEvent)
        }
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete line in buffer

      processLines(lines, onToken, onThinking, wrappedOnComplete, onRawEvent)
    }

    // Only call onComplete if we didn't already complete with usage data
    if (!completedWithData) {
      onComplete()
    }
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        // User cancelled, not an error
        if (!completedWithData) {
          onComplete()
        }
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
  onToken: (token: string, rawEvent?: string) => void,
  onThinking: ((thinking: string, rawEvent?: string) => void) | undefined,
  onComplete: (data?: StreamCompletionData) => void,
  onRawEvent?: (event: string) => void
): void {
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.startsWith('data: ')) continue

    const data = trimmed.slice(6) // Remove 'data: ' prefix

    // Always capture raw events for debugging
    if (onRawEvent && data !== '[DONE]') {
      onRawEvent(trimmed)
    }

    if (data === '[DONE]') {
      return
    }

    try {
      const chunk: StreamChunk = JSON.parse(data)
      const delta = chunk.choices?.[0]?.delta

      if (delta?.content) {
        onToken(delta.content, trimmed)
      }

      if (delta?.thinking && onThinking) {
        onThinking(delta.thinking, trimmed)
      }

      // Check for usage in final chunk - extract all enhanced fields
      if (chunk.usage) {
        const completionData: StreamCompletionData = {
          usage: chunk.usage,
          timing: chunk.timing,
          generationConfig: chunk.generation_config,
          stopReason: chunk.stop_reason,
        }
        onComplete(completionData)
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
