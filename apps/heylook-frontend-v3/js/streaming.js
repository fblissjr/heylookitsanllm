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

import { requestId } from './api.js';

export async function streamChat(body, {
  signal,
  onToken,      // (delta, fullContent)
  onThinking,   // (delta, fullThinking)
  onLogprobs,   // (logprobsContentArray) -- explore page only
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
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId() },
      body: JSON.stringify({
        ...body,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal,
    });

    if (!res.ok) {
      let detail = res.statusText;
      let code = null;
      try {
        const data = await res.json();
        detail = data.error?.message || data.detail || data.error?.code || detail;
        code = data.error?.code ?? null;
      } catch { /* non-JSON error body */ }
      const err = new Error(detail);
      err.status = res.status;
      err.code = code;
      err.retryAfter = Number(res.headers.get('Retry-After')) || null;
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
