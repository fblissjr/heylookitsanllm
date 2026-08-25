// SSE streaming for the Messages wire: /v1/messages (streamMessages) and the
// conversation-scoped generate endpoint (streamGenerate), via fetch +
// ReadableStream. (streamChat -- the OpenAI /v1/chat/completions wire -- was
// removed with Phase 3b: no v3 page speaks it anymore; the backend endpoint
// stays for external consumers.)
//
// Contract gotchas (verified against backend; do not "simplify" away):
// - AbortError is NORMAL completion: onComplete fires with partial content.
// - reader.cancel() must run on abort/error, or the browser keeps the HTTP
//   connection alive and the NEXT request fails with "Failed to fetch".
// - SSE comment lines (": keepalive", sent every 5s during long prefill)
//   must be ignored, not parsed as data.
// - Both endpoints speak the Messages event grammar (spec §4): typed
//   `event:` lines. Extensions are namespaced: `heylook_logprobs` rides
//   per-token on /v1/messages; `heylook_saved` is ALWAYS the last event on
//   /generate and carries the authoritative stored rows; timing/KV telemetry
//   rides message_stop's `performance` object. An in-band `error` event ENDS
//   generation on /v1/messages but may PRECEDE heylook_saved on /generate --
//   which is why error handling lives in each wrapper, not the shared core.

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

// The shared typed-SSE core: POST `url`, retry on 503 model_overloaded,
// parse `event:`/`data:` blocks, hand each (eventType, data) to onEvent.
// Returns { aborted } -- AbortError and an abort during the retry sleep both
// resolve here rather than throwing, because an abort is NORMAL completion
// for every consumer. Anything onEvent throws propagates to the caller's
// catch (that is how streamMessages turns an in-band error event into
// onError while streamGenerate deliberately keeps reading past one).
async function streamTypedSSE(url, body, { signal, onRetryWait, onEvent }) {
  let reader = null;
  try {
    let res;
    for (let attempt = 1; ; attempt++) {
      res = await fetch(url, {
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
        if (signal?.aborted) return { aborted: true };
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

      let sep, sepLen;
      while (true) {
        const crlfSep = buf.indexOf('\r\n\r\n');
        const lfSep = buf.indexOf('\n\n');
        if (crlfSep === -1 && lfSep === -1) break;
        if (crlfSep !== -1 && (lfSep === -1 || crlfSep < lfSep)) {
          sep = crlfSep;
          sepLen = 4;
        } else {
          sep = lfSep;
          sepLen = 2;
        }
        const eventText = buf.slice(0, sep);
        buf = buf.slice(sep + sepLen);
        let eventType = null;
        let data = null;
        for (const line of eventText.split(/\r?\n/)) {
          if (line.startsWith('event:')) eventType = line.slice(6).trim();
          else if (line.startsWith('data:')) {
            const raw = line.slice(5).trim();
            if (raw) {
              try { data = JSON.parse(raw); } catch { data = null; }
            }
          }
          // ": keepalive" comment lines fall through both branches
        }
        if (eventType && data !== null) onEvent(eventType, data);
      }
    }
    return { aborted: false };
  } catch (err) {
    try { await reader?.cancel(); } catch { /* already closed */ }
    if (err.name === 'AbortError') return { aborted: true };
    throw err;
  }
}

// POST /v1/messages, streaming (Phase 3b: the wire notebook and explore
// speak; chat has its conversation-scoped sibling below).
//
// Callbacks:
//   onToken(delta, full)     -- text deltas
//   onThinking(delta, full)  -- thinking deltas
//   onLogprobs(tokens)       -- heylook_logprobs extension: an array of
//                               {token, logprob, top_logprobs} entries, the
//                               same shape the OpenAI wire carried
//   onRetryWait(sec, n)      -- 503 model_overloaded auto-retry
//   onComplete({content, thinking, usage, performance, aborted})
//                            -- performance is message_stop's object incl.
//                               the heylook timing fields
//   onError(err)             -- HTTP errors + the in-band error event (which
//                               ENDS a /v1/messages generation, unlike
//                               /generate's, so it routes here not onComplete)
export async function streamMessages(body, {
  signal,
  onToken,
  onThinking,
  onLogprobs,
  onRetryWait,
  onComplete,
  onError,
} = {}) {
  let content = '';
  let thinking = '';
  let usage = null;
  let performance = null;

  try {
    const { aborted } = await streamTypedSSE('/v1/messages', { ...body, stream: true }, {
      signal,
      onRetryWait,
      onEvent: (eventType, data) => {
        if (eventType === 'content_block_delta') {
          const d = data.delta ?? {};
          if (d.type === 'thinking_delta' && d.text) {
            thinking += d.text;
            onThinking?.(d.text, thinking);
          } else if (d.type === 'text_delta' && d.text) {
            content += d.text;
            onToken?.(d.text, content);
          }
        } else if (eventType === 'heylook_logprobs') {
          if (data.tokens?.length) onLogprobs?.(data.tokens);
        } else if (eventType === 'message_delta') {
          usage = data.usage ?? usage;
        } else if (eventType === 'message_stop') {
          performance = data.performance ?? null;
        } else if (eventType === 'error') {
          const err = new Error(data.error?.message || 'Generation failed');
          err.code = data.error?.type ?? null;
          throw err; // ends the stream -> outer catch -> onError
        }
      },
    });
    onComplete?.({ content, thinking, usage, performance, aborted });
  } catch (err) {
    onError?.(err);
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
  let content = '';
  let thinking = '';
  let usage = null;
  let saved = null;

  try {
    const { aborted } = await streamTypedSSE(`/v1/conversations/${convId}/generate`, body, {
      signal,
      onRetryWait,
      onEvent: (eventType, data) => {
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
          // tells the caller that onComplete is still coming. (Deliberately
          // NOT thrown, unlike streamMessages' error handling.)
          err.inBand = true;
          onError?.(err);
        }
      },
    });
    onComplete?.({ content, thinking, usage, aborted, saved });
  } catch (err) {
    onError?.(err);
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
