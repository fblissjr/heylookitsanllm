// SSE streaming for /v1/chat/completions (streamChat) and the
// conversation-scoped generate endpoint (streamGenerate) via fetch +
// ReadableStream.
//
// Contract gotchas (verified against backend; do not "simplify" away):
// - AbortError is NORMAL completion: onComplete fires with partial content.
// - reader.cancel() must run on abort/error, or the browser keeps the HTTP
//   connection alive and the NEXT request fails with "Failed to fetch".
// - SSE comment lines (": keepalive", sent every 5s during long prefill)
//   must be ignored, not parsed as data.
// - streamChat: usage/timing arrive in a final chunk only because we always
//   send stream_options.include_usage: true. Stream ends with `data: [DONE]`.
// - streamGenerate: the Messages event grammar (spec §4) -- typed `event:`
//   lines; `heylook_saved` is ALWAYS the last event and carries the
//   authoritative stored rows. An in-band `error` event may precede it.

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

// Conversation-scoped generation (POST /v1/conversations/{id}/generate --
// the server-side saga, spec §4). The server persists everything; the
// client renders deltas and, at the end, ASSIGNS state from the saved rows.
//
// Callbacks:
//   onToken(delta, full)     -- text deltas
//   onThinking(delta, full)  -- thinking deltas
//   onRetryWait(sec, n)      -- 503 model_overloaded auto-retry
//   onSaved(payload)         -- the heylook_saved event: {messages (stored
//                               rows), end_reason, dropped_media, timing}.
//                               May be ABSENT if the connection died first
//                               (the server still persisted -- reconcile).
//   onComplete({content, thinking, usage, aborted, saved})
//   onError(err)             -- HTTP errors + in-band typed error events
export async function streamGenerate(convId, body, {
  signal,
  onToken,
  onThinking,
  onRetryWait,
  onSaved,
  onComplete,
  onError,
} = {}) {
  let reader = null;
  let content = '';
  let thinking = '';
  let usage = null;
  let saved = null;

  const finish = (aborted) => onComplete?.({ content, thinking, usage, aborted, saved });

  try {
    let res;
    for (let attempt = 1; ; attempt++) {
      res = await fetch(`/v1/conversations/${convId}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId() },
        body: JSON.stringify(body),
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
        const eventText = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        let eventType = null;
        let data = null;
        for (const line of eventText.split('\n')) {
          if (line.startsWith('event:')) eventType = line.slice(6).trim();
          else if (line.startsWith('data:')) {
            try { data = JSON.parse(line.slice(5).trim()); } catch { data = null; }
          }
          // ": keepalive" comment lines fall through both branches
        }
        if (!eventType || !data) continue;

        if (eventType === 'content_block_delta') {
          const d = data.delta ?? {};
          if (d.type === 'thinking_delta' && d.text) {
            thinking += d.text;
            onThinking?.(d.text, thinking);
          } else if (d.type === 'text_delta' && d.text) {
            content += d.text;
            onToken?.(d.text, content);
          }
        } else if (eventType === 'message_delta') {
          // usage: {input_tokens, output_tokens, ...} -- keep the wire names
          usage = data.usage ?? usage;
        } else if (eventType === 'heylook_saved') {
          saved = data;
          onSaved?.(data);
        } else if (eventType === 'error') {
          const err = new Error(data.error?.message || 'Generation failed');
          err.code = data.error?.type ?? null;
          // An error event is not the end of the stream: a partial may
          // still persist and heylook_saved may still follow. Surface the
          // error but keep reading to the server's last word -- inBand
          // tells the caller that onComplete is still coming.
          err.inBand = true;
          onError?.(err);
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

// The Stop button's server-side spelling: abort the active generation for
// this conversation. The partial persists server-side and the ongoing SSE
// stream still ends with its heylook_saved event -- so the caller keeps
// reading rather than aborting the fetch. Returns the HTTP status (null on
// network failure): 404 means the server has NOTHING active -- the caller
// is in a client-side-only phase (the 503 retry sleep, or pre-claim) and
// must abort locally instead, or the retry launches a generation the user
// explicitly stopped (review finding 2026-08-13).
export function stopGenerate(convId) {
  return fetch(`/v1/conversations/${convId}/generate`, {
    method: 'DELETE',
    headers: { 'X-Request-ID': requestId() },
  }).then((res) => res.status).catch(() => null);
}
