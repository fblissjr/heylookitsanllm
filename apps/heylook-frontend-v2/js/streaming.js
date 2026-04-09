// SSE streaming client -- ported from streaming.ts

function requestId() {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

/**
 * Stream a chat completion via SSE.
 *
 * @param {object} request - Chat completion request body
 * @param {object} callbacks - { onToken, onThinking, onComplete, onError }
 * @param {AbortSignal} [signal] - Abort signal
 * @param {number} [timeoutMs] - Timeout in milliseconds
 */
export async function streamChat(request, callbacks, signal, timeoutMs) {
  const { onToken, onThinking, onComplete, onError } = callbacks

  let completedWithData = false
  const signals = []
  if (signal) signals.push(signal)
  if (timeoutMs) signals.push(AbortSignal.timeout(timeoutMs))
  const combinedSignal = signals.length > 0 ? AbortSignal.any(signals) : undefined

  let reader

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId(),
      },
      body: JSON.stringify({
        ...request,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal: combinedSignal,
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || `HTTP ${response.status}`)
    }

    if (!response.body) throw new Error('No response body for streaming')

    reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        if (buffer.trim()) processLines(buffer.split('\n'), callbacks, () => { completedWithData = true })
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      processLines(lines, callbacks, () => { completedWithData = true })
    }

    if (!completedWithData) onComplete()
  } catch (error) {
    // Release HTTP connection immediately
    await reader?.cancel().catch(() => {})

    if (error.name === 'AbortError') {
      if (!completedWithData) onComplete()
      return
    }
    if (error.name === 'TimeoutError') {
      onError(new Error('Generation timed out'))
      return
    }
    onError(error)
  }
}

function processLines(lines, callbacks, markCompleted) {
  const { onToken, onThinking, onComplete } = callbacks

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.startsWith('data: ')) continue

    const data = trimmed.slice(6)
    if (data === '[DONE]') break

    try {
      const chunk = JSON.parse(data)
      const delta = chunk.choices?.[0]?.delta

      if (delta?.content) onToken(delta.content)
      if (delta?.thinking && onThinking) onThinking(delta.thinking)

      if (chunk.usage) {
        markCompleted()
        onComplete({
          usage: chunk.usage,
          timing: chunk.timing,
          generationConfig: chunk.generation_config,
          stopReason: chunk.stop_reason,
        })
        break
      }
    } catch {
      // Skip invalid JSON
    }
  }
}
