# Observability Guide

Last updated: 2026-04-18

`heylookitsanllm` writes four disk-backed JSONL streams under `internal/log/`
(gitignored). They exist so you can (a) prove the server is leak-free over long
windows, (b) see what workloads the server actually serves, and (c) tune
presets and unload policy based on real usage, not guesses.

This guide covers what's logged, how to configure it, the content guarantees,
and concrete recipes for monitoring and optimization.

## Contents

- [Content invariant](#content-invariant) — what is and isn't recorded
- [The four streams](#the-four-streams)
- [Environment variables and config](#environment-variables-and-config)
- [Response-time telemetry (headers and SSE)](#response-time-telemetry-headers-and-sse)
- [Tutorial: run for monitoring](#tutorial-run-for-monitoring)
- [Guide: find usage patterns](#guide-find-usage-patterns)
- [Guide: optimize from the data](#guide-optimize-from-the-data)
- [Known limitations](#known-limitations)

---

## Content invariant

**Every stream records numeric and metadata fields only. Never prompt text,
response text, token ID sequences, or message content.**

| Allowed | Disallowed |
| --- | --- |
| Token *counts* (prompt_tokens, completion_tokens) | Token *IDs* (sequences reconstruct text) |
| Sampler knobs (temperature, top_p, etc.) | `messages`, `prompt`, `response`, `content` fields |
| Byte counts, memory counters | Conversation IDs tied to content |
| Model IDs, architectures, quantization | System prompts |
| Cache hit/miss counts | Tool-call arguments |
| Timing breakdowns | Error messages that may contain user input |

The rule is enforced in `src/heylook_llm/memory.py`: snapshot and event-builder
functions take no request object, no message handles. `sampler_summary_from_request`
is the canonical extractor for sampler knobs and drops everything else.

The test suite (`tests/unit/test_memory_manager.py`) walks every logged
snapshot recursively and asserts that no key named `prompt`, `messages`,
`message`, `completion`, `response`, or `text` appears anywhere.

---

## The four streams

All live under `internal/log/`, one JSON line per record, `orjson`-serialized.

### `baseline.jsonl` — one-shot startup record

Written once per server start. Self-describes the hardware and the
observability config so every other log file is interpretable in isolation.

```json
{
  "ts": 1776533062.1,
  "event": "startup",
  "baseline_interval_seconds": 3600,
  "request_log_enabled": true,
  "model_event_log_enabled": true,
  "max_loaded_models": 2,
  "device_info": {
    "max_recommended_working_set_size": 173173080064,
    "memory_size": 206158430208,
    "architecture": "applegpu_g14d",
    "max_buffer_length": 129879801856,
    "resource_limit": 499000
  }
}
```

Always written regardless of log toggles — tiny, one-shot, zero runtime cost.

### `memory_baseline.jsonl` — periodic resource snapshot

Written every `baseline_log_interval_seconds` (default 3600). The answer to
"is the server leaking?" — monotonic growth in `rss_bytes` or `mlx_active_bytes`
over 72h means yes.

```json
{
  "ts": 1776533125.6,
  "rss_bytes": 44040192,
  "available_ram_bytes": 133122375680,
  "cpu_percent": 99.8,
  "mlx_active_bytes": 8,
  "mlx_peak_bytes": 8,
  "mlx_cache_bytes": 0,
  "loaded_models": [
    {"model_id": "qwen3-4b-4bit", "weights_bytes": 2400000000,
     "architecture": "qwen3", "quantization": "4bit",
     "param_count": 4000000000, "context_length": 32768}
  ],
  "prompt_cache": {"total_bytes": 1048576},
  "vision_cache": {"entries": 3, "hits": 12, "misses": 5, "hit_rate": 0.71},
  "idle_seconds": 42.5,
  "mode": "background",
  "inflight_requests": 0
}
```

`cpu_percent` is the average CPU utilization *since the previous snapshot*, so
with an hourly interval the value reflects the full hour — a smoothed trend
signal, not an instantaneous sample.

### `request_events.jsonl` — one line per completed request

Written on every request completion (success or error). Superset of the
in-memory `perf_collector` ring buffer that feeds the Performance page.

```json
{
  "timestamp": 1776533200.4,
  "model": "qwen3-4b-4bit",
  "success": true,
  "total_ms": 1842.5,
  "queue_ms": 12.1,
  "model_load_ms": 0.0,
  "image_processing_ms": 0.0,
  "token_generation_ms": 1830.0,
  "first_token_ms": 125.4,
  "prompt_tokens": 1024,
  "completion_tokens": 320,
  "tokens_per_second": 175.0,
  "had_images": false,
  "was_streaming": true,
  "sampler_summary": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512},
  "peak_memory_gb": 6.82,
  "kv_cache_bytes": 524288,
  "cached_tokens": 400,
  "thinking_tokens": 0,
  "content_tokens": 320,
  "content_duration_ms": 1700,
  "stop_reason": "stop",
  "provider_type": "mlx",
  "image_count": 0,
  "cache_hit_rate": 0.3906
}
```

`stop_reason` is one of `"stop"` / `"length"` / `"error"` / `"abort"`. Toggle
with `HEYLOOK_REQUEST_LOG_ENABLED=0` to keep the ring buffer but skip disk
append (useful for tests or when you don't care about long-window analysis).

### `model_events.jsonl` — one line per model load/unload

```json
{"ts": 1776533060.1, "event": "load", "model_id": "qwen3-4b-4bit",
 "path": "/Users/.../qwen3-4b-4bit", "weights_bytes": 2400000000,
 "architecture": "qwen3", "quantization": "4bit", "param_count": 4000000000,
 "context_length": 32768, "load_duration_ms": 3521.0}

{"ts": 1776540123.8, "event": "unload", "model_id": "qwen3-4b-4bit",
 "reason": "lru_evict"}
```

`reason` is one of `"lru_evict"`, `"manual"`, `"shutdown"`.
Toggle with `HEYLOOK_MODEL_EVENT_LOG_ENABLED=0`.

---

## Environment variables and config

All three log streams are controlled from two sources: `AppConfig` fields in
`models.toml`, and matching environment variables that override them at
startup. Env vars win when both are set.

| Variable | Config field | Default | Effect |
| --- | --- | --- | --- |
| `HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS` | `baseline_log_interval_seconds` | `3600` | Seconds between `memory_baseline.jsonl` entries. `0` disables entirely. |
| `HEYLOOK_REQUEST_LOG_ENABLED` | `request_log_enabled` | `true` | Per-request disk append. Accepts `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off`. |
| `HEYLOOK_MODEL_EVENT_LOG_ENABLED` | `model_event_log_enabled` | `true` | Model load/unload disk append. Same value formats. |

Example — quick dev loop with 60-second baselines and no request-event spam:

```bash
HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS=60 HEYLOOK_REQUEST_LOG_ENABLED=0 \
  heylookllm --log-level INFO
```

In `models.toml` (top level, next to `max_loaded_models`):

```toml
baseline_log_interval_seconds = 1800    # 30 min
request_log_enabled = true
model_event_log_enabled = true
```

---

## Response-time telemetry (headers and SSE)

Per-request memory telemetry is also available at request time, not just on
disk:

- **Non-streaming** responses carry two headers:
  - `x-heylook-peak-memory-gb` — peak MLX memory during this request
  - `x-heylook-kv-bytes` — bytes held in the KV prompt cache at start of this
    request
- **Streaming** responses emit the same values inside the final usage chunk's
  `timing` object. Client must pass `stream_options.include_usage=true`:

```json
{"timing": {"total_duration_ms": 1842, "peak_memory_gb": 6.82, "kv_cache_bytes": 524288}}
```

SSE response headers can't carry these values because they're sent before
generation runs. The frontend-v2 chat status bar renders the triple
"`N tokens · P.PP GB peak · K KV`" automatically.

---

## Tutorial: run for monitoring

### Day 0 — baseline

Install or update to the latest:

```bash
uv sync --dev      # installs pytest-asyncio for the full test suite
```

Start the server with defaults:

```bash
heylookllm --log-level INFO
```

Confirm the startup record:

```bash
tail -1 internal/log/baseline.jsonl | python -m json.tool
```

You should see your GPU's `device_info` and the current log toggles.

Send a request and check the headers:

```bash
curl -i http://localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "hello"}]
}' 2>&1 | grep -i 'x-heylook'
```

Expected:

```
x-heylook-peak-memory-gb: 4.213
x-heylook-kv-bytes: 131072
```

### Day 1–7 — collect

Leave the server running normally. After an hour you should have at least one
`memory_baseline.jsonl` entry and as many `request_events.jsonl` lines as
requests you sent.

```bash
wc -l internal/log/*.jsonl
```

### Day 3 — first leak check

After 72 hours of uptime:

```bash
jq -r '[.ts, .rss_bytes, .mlx_active_bytes] | @tsv' \
  internal/log/memory_baseline.jsonl | \
  awk '{print strftime("%Y-%m-%d %H:%M", $1), $2/1e9"GB rss", $3/1e9"GB mlx"}'
```

Read the output: `rss` should oscillate, not climb monotonically. `mlx_active`
should drop to near-baseline when idle. Flat or sawtooth is healthy; a steady
rise is a leak. Correlate spikes against `model_events.jsonl` to see if a
specific load caused them.

---

## Guide: find usage patterns

Point `jq` at `request_events.jsonl` to answer actual questions.

### Top models by request count

```bash
jq -r '.model' internal/log/request_events.jsonl | sort | uniq -c | sort -rn
```

### Time-of-day density

```bash
jq -r '.timestamp | strftime("%H")' internal/log/request_events.jsonl | \
  sort | uniq -c
```

Look for long idle windows — those are candidates for aggressive model
unloading via the forthcoming idle-daemon + foreground/background toggle.

### Sampler settings actually used per model

```bash
jq -r '[.model, (.sampler_summary.temperature // "default"),
        (.sampler_summary.max_tokens // "default")] | @tsv' \
  internal/log/request_events.jsonl | sort | uniq -c | sort -rn
```

If every request to model `X` uses the same temperature, that's a signal the
model default could be tightened. If requests cluster into two distinct
configurations, that's a signal for two presets.

### Cache hit rate per model

```bash
jq -r '[.model, (.prompt_tokens // 0), (.cached_tokens // 0)] | @tsv' \
  internal/log/request_events.jsonl | \
  awk '{agg_total[$1] += $2; agg_cached[$1] += $3}
       END {for (m in agg_total) printf "%s\t%.1f%%\n", m, 100*agg_cached[m]/agg_total[m]}'
```

Low cache hit rate on a conversational model → either prompts never repeat
(normal for one-shot use), or the radix cache is getting evicted too
aggressively. Cross-reference against `memory_baseline.jsonl`'s
`prompt_cache.total_bytes` over time.

### Peak memory distribution

```bash
jq -r '.peak_memory_gb | select(. != null)' \
  internal/log/request_events.jsonl | \
  sort -n | awk '
    {vals[NR]=$1}
    END {
      print "p50:", vals[int(NR*0.5)]
      print "p95:", vals[int(NR*0.95)]
      print "max:", vals[NR]
    }'
```

The p95 is your working-set baseline. The max is the worst case; if the max
approaches `max_recommended_working_set_size` from `baseline.jsonl`, you're
near the envelope.

### Thinking vs content tokens

```bash
jq -r '[.model, .thinking_tokens, .content_tokens] | @tsv' \
  internal/log/request_events.jsonl | \
  awk '{t[$1]+=$2; c[$1]+=$3} END {for (m in t) printf "%s\tthink=%d content=%d ratio=%.2f\n", m, t[m], c[m], (c[m]?t[m]/c[m]:0)}'
```

Thinking-heavy models with low content output are candidates for a
`chat-deterministic` preset that disables thinking for speed.

### Model load frequency

```bash
jq -r 'select(.event=="load") | .model_id' internal/log/model_events.jsonl | \
  sort | uniq -c | sort -rn
```

If a model is loading more than once an hour, it's being evicted too often.
Either pin it (`pin_model` API) or raise `max_loaded_models`.

---

## Guide: optimize from the data

### Tighten sampler defaults per model

Pull the distribution from the "sampler settings actually used" query. If
model `X` always gets `temperature=0.3`, set it as the model's default in
`models.toml`:

```toml
[[models]]
id = "qwen3-4b-4bit"
provider = "mlx"
[models.config]
temperature = 0.3    # match observed usage
```

Requests that omit `temperature` now get the right default; requests that pass
a different value still win.

### Right-size `max_loaded_models`

- Sum the p95 `peak_memory_gb` of each hot model.
- Compare against `device_info.max_recommended_working_set_size`.
- If you have headroom, raise `max_loaded_models` so hot models stay resident.
- If you're near the envelope, lower it — one swapped load is cheaper than an
  OOM under pressure.

### Catch memory leaks early

Put this in a cron or a `/loop` to run weekly:

```bash
tail -n 168 internal/log/memory_baseline.jsonl | \
  jq -r '.mlx_active_bytes' | \
  awk 'NR==1 {first=$1} END {
    if ($1 > first * 1.5) print "LEAK: active_mlx grew", ($1-first)/1e9, "GB over last week"
    else print "OK: active_mlx flat"
  }'
```

### Plan preset taxonomy from real workloads

Run the "usage-pattern" queries above after a week of collection, then pick
preset categories that match real clusters rather than assumed ones. The
current presets in `src/heylook_llm/data/profiles/` are a starting point; use
the sampler-distribution and cache-hit-rate queries to validate whether the
defaults match how the server is actually used.

### Detect runaway requests

```bash
jq -r 'select(.peak_memory_gb > 30) | [.model, .timestamp, .peak_memory_gb] | @tsv' \
  internal/log/request_events.jsonl | \
  awk '{print strftime("%Y-%m-%d %H:%M", $2), $1, $3"GB"}'
```

If runaways cluster around specific sampler settings or prompt lengths,
that's a tuning opportunity — or a case for a foreground-mode hard memory
cap (planned).

### Schedule aggressive unloads during idle windows

Find your longest idle window from the time-of-day query. Until the idle
daemon lands, you can manually unload via the admin API at the boundary:

```bash
# At 02:00 every night
curl -X POST http://localhost:8080/v1/admin/models/unload \
  -H 'content-type: application/json' -d '{"model_id": "big-vlm"}'
```

---

## Known limitations

These are accepted trade-offs in this slice; candidates for future work.

- **Synchronous disk I/O in the request-completion path.** Every request
  triggers an `open(path, 'ab')` + `orjson.dumps` + write + close. Typically
  sub-millisecond on local storage, but a slow or full disk would block the
  event loop. For single-user M2 Ultra workloads this is fine; moving to a
  background writer thread is tracked as follow-up work.
- **No log rotation.** `request_events.jsonl` grows ~500 bytes per request
  indefinitely. At 10 req/s continuous that's ~13 GB/month. Rotate manually
  (`mv request_events.jsonl request_events.YYYY-MM.jsonl`) or run a cron
  job. Size-based rotation will be added when the file actually hurts.
- **CPU percent is an hour-averaged value** at the default interval. That's
  a trend signal, not a real-time monitor. Use the Performance page (in-memory
  ring buffer, 60-second resolution) for live CPU view.
- **Model metadata is best-effort.** `param_count`, `quantization`, and
  `context_length` are probed via attribute introspection; models that don't
  expose those attrs get sensible defaults (`0`, `"none"`, `0`). The weights
  byte count comes from filesystem size and is always accurate.
- **Error events before `perf_ctx` is built** still emit, but with a
  minimal event shape (no timing breakdown, just the model ID and the error).

---

## Related

- `src/heylook_llm/memory.py` — implementation
- `tests/unit/test_memory_manager.py` — test suite including the content-invariant walk
- [CLAUDE.md](../CLAUDE.md) — project conventions and the content invariant rule
