// SSE streaming for /v1/chat/completions via fetch + ReadableStream.
//
// Contract gotchas (verified against backend; do not "simplify" away):
// - AbortError is NORMAL completion: onComplete fires with partial content.
// - reader.cancel() must run on abort/error, or the browser keeps the HTTP
//   connection alive and the NEXT request fails with "Failed to fetch".
// - SSE comment lines (": keepalive", sent every 5s during long prefill)
//   must be ignored, not parsed as data.
// - usage/timing arrive in a final chunk only because we always send
//   stream_options.include_usage: true. Stream ends with `data: [DONE]`.

import { requestId, httpError } from './api.js';

// 503 model_overloaded + Retry-After is a transport-level contract emitted
// uniformly by the backend, so the bounded retry lives HERE -- every
// streaming page gets it without page-level retry state.
const MAX_BUSY_RETRIES = 3;

function sleep(ms, signal) {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); resolve(); }, { once: true });
  });
}

export async function streamChat(body, {
  signal,
  onToken,      // (delta, fullContent)
  onThinking,   // (delta, fullThinking)
  onLogprobs,   // (logprobsContentArray) -- explore page only
  onRetryWait,  // (seconds, attempt) -- server busy, retrying automatically
  onComplete,   // ({ content, thinking, usage, timing, stopReason, aborted })
  onError,      // (err) -- err.status/.code/.retryAfter set for HTTP errors
} = {}) {
  let reader = null;
  let content = '';
  let thinking = '';
  let usage = null;
  let timing = null;
  let stopReason = null;

  const finish = (aborted) =>
    onComplete?.({ content, thinking, usage, timing, stopReason, aborted });

  try {
    let res;
    for (let attempt = 1; ; attempt++) {
      res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId() },
        body: JSON.stringify({
          ...body,
          stream: true,
          stream_options: { include_usage: true },
        }),
        signal,
      });
      if (res.ok) break;

      const err = await httpError(res);
      if (err.status === 503 && err.code === 'model_overloaded' && attempt <= MAX_BUSY_RETRIES) {
        const wait = err.retryAfter ?? 2;
        onRetryWait?.(wait, attempt);
        await sleep(wait * 1000, signal);
        if (signal?.aborted) return finish(true);
        continue;
      }
      throw err;
    }

    reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buf.indexOf('\n\n')) !== -1) {
        const event = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue; // drops ": keepalive" comments
          const data = line.slice(5).trim();
          if (!data || data === '[DONE]') continue;
          let chunk;
          try { chunk = JSON.parse(data); } catch { continue; }

          // Mid-stream generation failure: the backend emits an error payload
          // instead of content. Throwing lands in the outer catch, which
          // cancels the reader and routes to onError.
          if (chunk.error) {
            const streamErr = new Error(chunk.error.message || 'Generation failed');
            streamErr.code = chunk.error.code ?? null;
            throw streamErr;
          }

          const choice = chunk.choices?.[0];
          const delta = choice?.delta;
          if (delta?.content) {
            content += delta.content;
            onToken?.(delta.content, content);
          }
          if (delta?.thinking) {
            thinking += delta.thinking;
            onThinking?.(delta.thinking, thinking);
          }
          if (choice?.logprobs?.content?.length) onLogprobs?.(choice.logprobs.content);
          if (choice?.finish_reason) stopReason = choice.finish_reason;
          if (chunk.usage) {
            usage = chunk.usage;
            timing = chunk.timing ?? null;
            stopReason = chunk.stop_reason ?? stopReason;
          }
        }
      }
    }

    finish(false);
  } catch (err) {
    try { await reader?.cancel(); } catch { /* already closed */ }
    if (err.name === 'AbortError') finish(true);
    else onError?.(err);
  }
}
