# Developer Notes: MLX Provider

> **Status**: CURRENT
> **Last Updated**: 2026-07-06

This document provides a deep-dive technical reference for the `MLXProvider` in `heylookitsanllm`. This provider is responsible for running models using Apple's MLX framework and is based on the `mlx-lm` and `mlx-vlm` libraries.

## What changed 2026-07-05/06

Two process-abort bugs found and fixed during the frontend v3 build's
backend co-evolution, plus a library-drift audit. See section 4 below for
detail, and the full postmortems:

- [postmortems/mlx_thread_teardown_abort.md](./postmortems/mlx_thread_teardown_abort.md)
  -- SIGTRAP process abort on generation-thread teardown (v1.31.2).
- [postmortems/radix_thread_affinity.md](./postmortems/radix_thread_affinity.md)
  -- "no Stream(gpu, N)" crash on radix cache reuse, plus the v1.32.0
  radix-eligibility gate (v1.31.1 / v1.32.0).

## 1. Core Architecture: Unified Generation

The `MLXProvider` uses two strategies that **share a single generation loop** (`generation_core.run_generation()` calling `mlx_lm.generate.stream_generate`). Vision requests use a pre-filled cache pattern: the full VLM model runs a forward pass to populate a KV cache, then the language model generates tokens through the same code path as text-only. Models are loaded via separate libraries (`mlx-lm` for text, `mlx-vlm` for VLM) due to incompatible model structures.

### 1.1. Dynamic Model Loading

Which library loads a model is decided by `effective_loader` (v1.34.43),
resolved in `providers/common/loader_routing.py` from the config's `modalities`
+ `loader` fields (NOT the raw `vision` bool -- that is now a derived mirror; see
config.md). `is_vlm = (effective_loader == "mlx-vlm")`.

*   **`effective_loader == "mlx-lm"`:** Loaded with `mlx_lm.utils.load`. This is text-only models, and any model explicitly routed with `loader = "mlx-lm"` (e.g. a dual-capable VLM you want to run as text).
*   **`effective_loader == "mlx-vlm"`:** Loaded with `mlx_vlm.utils.load` (correctly initializes `vision_tower` and multimodal components). Under `loader = "auto"`, a model reaches here only if it declares vision AND mlx-vlm registers its `model_type`; a vision model mlx-vlm cannot load degrades to `mlx-lm` instead of crashing.

**Hard-Learned Lesson:** Loading a VLM with `lm_load` causes `TypeError`/`AttributeError` deep inside the model's `__init__` due to text-only structural assumptions.

### 1.2. Strategy Architecture (v1.16.0)

As of v1.16.0, the provider uses two strategies instead of three. Previously, `TextOnlyStrategy` and `VLMTextOnlyStrategy` were ~85% identical (~400 lines of duplication), causing every fix to be applied in 2+ places. They were unified into `UnifiedTextStrategy`.

```
MLXProvider.create_chat_completion()
  |-- has_images --> VLMVisionStrategy.generate()
  |                    |-- mlx_vlm.utils.prepare_inputs (tokenize + pixel values)
  |                    |-- VLM forward pass -> fills KV cache
  |                    |-- Sample first token from logits
  |                    |-- run_generation(pre_filled_cache=...) [lm_stream_generate]
  |
  |-- no images  --> UnifiedTextStrategy.generate()   [lm_stream_generate via generation_core]
                       |-- is_vlm=true:  VLM chat template + LanguageModelLogitsWrapper
                       |-- is_vlm=false: tokenizer.apply_chat_template + raw model
```

**`UnifiedTextStrategy`** dispatches on `is_vlm` for two things only:
1. **Chat template**: `tokenizer.apply_chat_template(enable_thinking=...)` for text-only, `vlm_apply_chat_template(processor, config, ...)` for VLM
2. **Model object**: Raw model for text-only, `LanguageModelLogitsWrapper(model.language_model)` for VLM

Everything else -- message prep, cache config, prompt cache lookup, generation loop, acceptance tracking, KV snapshot storage -- lives in `generation_core`, shared across both paths.

**`VLMVisionStrategy`** (v1.18.0) uses the pre-filled cache pattern inspired by vllm-mlx:
1. Prepare inputs via `mlx_vlm.utils.prepare_inputs` (handles image grid dimensions per model, e.g. Qwen `image_grid_thw`)
2. Create KV cache for the language model via `make_prompt_cache(LanguageModelLogitsWrapper)`
3. Run full VLM forward pass (`model(input_ids, cache=request_cache, pixel_values=...)`), filling the cache
4. Sample first token from output logits
5. Continue generation via `run_generation(pre_filled_cache=request_cache)` -- the same code path as text-only

This gives vision requests the full sampler suite (top_k, min_p, presence_penalty, logit_bias, XTC), abort support, speculative decoding acceptance tracking, and the DraftTuner. Image loading is parallelized via `BatchVisionProcessor`.

### 1.3. Generation Core (v1.17.0)

`generation_core.py` provides two entry points:

- **`generate_text()`** -- high-level: builds sampler/processors from `effective_request`, then calls `run_generation()`. Used by `UnifiedTextStrategy`.
- **`run_generation()`** -- low-level: takes pre-built sampler/processors. Used by `generate_text()` internally, and directly by `VLMVisionStrategy` with `pre_filled_cache` parameter.

**DraftTuner**: Module-level singleton that dynamically adjusts `num_draft_tokens` based on per-model rolling acceptance rates. Consulted in `run_generation()` before `lm_stream_generate`, updated in the `finally` block. Conservative policy: increase at >80% acceptance, decrease at <50%, over a 50-sample window (min 1, max 8).

### 1.4. Key Files

| File | Purpose |
|------|---------|
| `providers/mlx_provider.py` | `MLXProvider`, `UnifiedTextStrategy`, `VLMVisionStrategy`, routing |
| `providers/common/generation_core.py` | `generate_text`, `run_generation`, `DraftTuner` |
| `providers/common/vlm_inputs.py` | Standalone VLM input preparation (image extraction, parallel loading) |
| `providers/common/model_wrappers.py` | `LanguageModelLogitsWrapper` -- adapts VLM language model for mlx-lm |
| `providers/common/prompt_cache.py` | Radix-tree prompt cache manager |
| `providers/common/samplers.py` | Sampler/processor construction |

### 1.5. LanguageModelLogitsWrapper

Lives in `providers/common/model_wrappers.py` (moved from `mlx_provider.py` in v1.16.0 to break circular imports).

The wrapper is a thin `nn.Module` that adapts a VLM's language model for `lm_stream_generate`:
- `__call__` extracts `.logits` from `LanguageModelOutput`
- `__getattr__` forwards everything else (`.layers`, `.head`, etc.)
- Caches `sanitized_config`, `model_type`, `args` for mlx-lm compatibility
- Must remain transparent to `mx.fast.*` operations, `nn.QuantizedLinear`, and future `mx.compile` wrapping

### 1.6. Generation Core

Lives in `providers/common/generation_core.py` (new in v1.16.0).

`run_generation()` is the single place `lm_stream_generate` is called for text generation:
- Builds cache config from `effective_request`
- Looks up radix-tree prompt cache
- Runs generation under `wired_limit` scope (pins weights in physical memory)
- Tracks speculative decoding acceptance (Python int counters -- zero GPU sync)
- Stores KV snapshot in radix tree on completion
- Handles abort events (per-token check)
- Cleans leading whitespace on first token

MLX alignment constraints:
- All counters are Python ints, not `mx.array` -- no `.item()` calls in the hot loop
- `wired_limit` wraps entire generation, not per-call
- Logging happens in `finally` block (outside the hot loop)
- Ready for future `mx.compile` integration (wraps `_step` inside `lm_stream_generate`)

## 2. Advanced Features: Leveraging `mlx-lm`

### 2.1. Sampler and Caching Systems

The sampling and KV caching logic are ported directly from `mlx-lm`.
*   **Strength:** Full suite of samplers (top-p, top-k, min-p, repetition penalty, presence penalty, logit_bias, XTC) and memory-efficient cache types (`RotatingKVCache`, `QuantizedKVCache`).
*   **Coverage:** As of v1.18.0, **all paths** (text-only, VLM text, and VLM vision) route through `generation_core.run_generation()` and use the full sampler suite. Vision requests get the same sampling quality as text-only requests to the same model.

### 2.2. Speculative Decoding

*   **Configuration**: Set `draft_model_path` in `models.toml`. The `num_draft_tokens` parameter defaults to 3 (configurable per model or per request).
*   **Acceptance tracking**: `generation_core.run_generation()` counts draft token acceptances and logs the rate after each generation. Python int counters only -- no GPU synchronization overhead.
*   **Scope**: Supported for all text generation via `UnifiedTextStrategy`. Vision path (`VLMVisionStrategy`) uses `run_generation()` with `pre_filled_cache`, which supports DraftTuner acceptance tracking but does not pass a `draft_model` (the pre-filled cache is incompatible with speculative prefill). Speculative decoding for the decode phase of vision requests is a potential future improvement.
*   **Memory pressure**: The radix cache accepts a `memory_pressure_fn` callback. When GPU memory exceeds 85% of recommended working set, eviction triggers even if node count is below `max_nodes`.

## 3. Known Issues and Trade-offs

*   **First-token asymmetry for vision**: The first token in a vision request is sampled directly from VLM forward pass logits (outside `run_generation()`). Subsequent tokens go through the full generation loop. This means the first token lacks DraftTuner integration and has slightly different logprobs handling (logprobs are computed inline, not via the GenerationResponse path).

*   **No radix cache for vision**: Vision requests skip the radix-tree prompt cache because the pre-filled KV cache includes vision embeddings that can't be represented as token sequences. Each vision request does a full VLM forward pass.

*   **Dependency on upstream APIs**: Tightly coupled to `mlx-lm` and `mlx-vlm` function signatures. `LanguageModelLogitsWrapper` is the primary defense against `LanguageModelOutput` structure changes.

*   **No batched image processing**: VLM vision requests process images sequentially via `BatchVisionProcessor` (parallel image loading, but sequential vision tower passes). Batched vision tower passes are out of scope.

*   **Generation failures do not surface uniformly**: see 4.5 below. Streaming and non-streaming chat/messages endpoints turn a failed generation into a real error. Batch and RLM do not yet -- they still concatenate the error text into their result as if it were model output.

## 4. Concurrency, Caching, and Dependency Notes (2026-07-05/06)

### 4.1. Generation concurrency: FIFO gate + pinned executor pool

One GPU, one loaded model (by default), one shared KV cache: only one
generation can run at a time. `GenerationGate`
(`providers/common/generation_gate.py`) enforces this as a strict-FIFO
admission queue -- no preemption, arrival order, bounded depth
(`max_queue_depth`, default 8; `check_capacity()` returns a 503 once
`1 + max_queue_depth` requests are already in the system).
`gate.acquire()` is called *inside* the provider's generation generator,
on whichever thread is driving it, the first time `next()` runs.

Each streaming generation runs start-to-finish on one dedicated thread
(`streaming_utils.async_generator_with_abort`) -- required because MLX's
per-generation stream and `wired_limit` context are entered on the first
`next()` and synchronized on the last, so hopping threads mid-generation
is unsafe. As of v1.31.2, that thread is **leased from a
`_PinnedExecutorPool`**, not created and destroyed per request: threads
that ran MLX work must never be torn down while the process lives (MLX's
thread-local `CompilerCache` can crash the whole process on teardown --
see
[postmortems/mlx_thread_teardown_abort.md](./postmortems/mlx_thread_teardown_abort.md)).
The pool grows to admitted concurrency (bounded by `1 + max_queue_depth`)
and never shrinks. A single shared thread for all requests was considered
and rejected -- it deadlocks against the FIFO gate above (a queued
request's `acquire()` call would seize the only thread the active
request's remaining `next()` calls need); see that postmortem for the
full mechanism.

### 4.2. Radix cache eligibility gate

The radix prompt cache (`providers/common/prompt_cache.py`) reuses KV
state across requests via longest-prefix matching. Two invariants, both
added in the 2026-07-05/06 window, that this document previously did not
mention:

- **Snapshots must be materialized before publishing.** `snapshot_kv`
  (`cache_helpers.py`) `mx.eval`s captured per-layer KV state on the
  generating thread before it's inserted into the shared radix tree.
  Unmaterialized (lazy) state is scheduled on a thread-local GPU stream
  that dies with its thread; a later request restoring that state on a
  different thread would crash. See
  [postmortems/radix_thread_affinity.md](./postmortems/radix_thread_affinity.md).
- **Radix is bypassed entirely for non-standard caches.**
  `process_prompt_with_cache` sets
  `prompt_cache._radix_eligible = (cache_type == "standard" and no
  max_kv_size)`; `store_generation_cache` mirrors the same gate. Prefix
  restoration slices `keys[..., :N, :]`, which is correct for plain
  `KVCache` but wrong for `QuantizedKVCache` (packed-tuple state) and
  impossible for `RotatingKVCache`. Models configured with
  `cache_type = "quantized"` or `"rotating"`, or with `max_kv_size` set,
  get no prefix reuse -- a full re-prefill every request -- but no
  silent-wrong-output risk either. This does **not** cover the
  hybrid-architecture (`ArraysCache`) case from earlier work.

### 4.3. transformers 5.x compatibility patches

`mlx_provider.py` (lines 32-77) applies two monkeypatches at import time
so VLM processor loading degrades gracefully without `torchvision`
(absent on MLX-only setups, but assumed present by `transformers`):

1. `AutoVideoProcessor.from_pretrained` soft-fails to `None` instead of
   raising `ImportError`.
2. `ProcessorMixin.check_argument_for_proper_class` accepts `None` for a
   `video_processor` sub-processor instead of rejecting it.

Two other patches that used to live here were removed 2026-07-06 as
verified dead against installed `transformers 5.5.4`: one guarded a
`VIDEO_PROCESSOR_MAPPING_NAMES` access that is now backend-gated and
raises before the patch could apply; the other patched
`transformers.utils.auto_docstring`, which now binds the decorator
function directly rather than a submodule, so the patch was a silent
no-op. A dead VLM strict-load `TypeError` fallback (`_load_vlm_with_weight_fix`)
was removed at the same time -- `mlx-vlm`'s `load()` has accepted `strict`
for a long time, making the fallback unreachable.

### 4.4. Library-drift audit (2026-07-06)

Every load-bearing assumption this codebase makes about `mlx-lm` /
`mlx-vlm` internals was checked against the **installed** versions: `mlx
0.31.2`, `mlx-lm 0.31.3`, `mlx-vlm 0.6.3`, `transformers 5.5.4`.
Conclusion: no broken sites. Specifically verified still correct:
`KVCache.state` laziness (the property this document's cache-snapshot
discussion above depends on), `stream_generate` kwargs, `GenerationResponse`
fields, mlx-vlm's `apply_chat_template`/`prepare_inputs`/`LanguageModelOutput`
signatures, `mlx_lm.utils._get_classes`, and `ArraysCache`'s trim
limitation (the hybrid-model risk noted in 4.2). One deprecation was fixed as a result:
`mx.metal.device_info()` migrated to `mx.device_info()` at all three call
sites (`memory.py` startup record, `prompt_cache.py`'s memory-pressure
check -- which would have broken outright once the alias is removed --
and `/v1/capabilities`).

### 4.5. Error surfacing contract

Generation failures are yielded as `MLXErrorChunk` (module-level in
`mlx_provider.py`, `is_error = True`) instead of raised directly, so the
generator's `finally` blocks (gate release, active-generation count,
cache cleanup) still run. Four consumer sites check `is_error` and turn
it into a real error instead of delivering `.text` as assistant content:

- `api.py` streaming (OpenAI): `data: {"error": {message, type:
  "server_error", code: "generation_failed"}}` then `data: [DONE]`.
- `api.py` non-streaming: raises `HTTPException(status_code=500,
  detail=chunk.text)`.
- `messages_api.py` streaming (Anthropic Messages): `event: error`.
- `messages_api.py` non-streaming: HTTP 500.

Before v1.31.1, none of these checks existed -- a failed generation
(including the radix thread-affinity crash in 4.2) streamed its error
text as if it were a normal assistant response, and clients rendered
and sometimes persisted it as one.

**This contract is not yet uniform.** `batch_processor.py` and `rlm.py`
both consume the same provider generator directly
(`for chunk in generator: full_text += chunk.text` /
`text_parts.append(chunk.text)`) without checking `is_error` -- a failed
generation inside a batch job or an RLM iteration still gets concatenated
into the result as if it succeeded. RLM specifically feeds that text back
into its own REPL loop as a sub-answer. Tracked as a deferred fix
(typed `GenerationFailed` exception raised from the provider generator,
translated to an error only at the two streaming/HTTP writers) in
`docs/project/TODO.md`.

## 5. Token-Level Data and Logprobs (IMPLEMENTED 2025-12-15)

The MLX provider now exposes token-level data from mlx-lm's `GenerationResponse` objects, enabling OpenAI-compatible logprobs functionality.

### Implementation

When `logprobs=true` is set in a request:
- The provider yields complete `GenerationResponse` objects instead of just text
- Each response includes `token` (int) and `logprobs` (mx.array) fields
- The API layer processes these using collectors from `src/heylook_llm/logprobs.py`

### Path-Specific Support

**Text-Only Path (MLX-LM)**: Full logprobs support
- Uses `mlx_lm.generate.stream_generate`
- Returns `GenerationResponse` with token IDs and full-vocabulary log-softmax
- Supports both streaming and non-streaming logprobs

**Vision Path (VLMVisionStrategy, v1.18.0)**: Partial logprobs support
- First token: logprobs computed inline from VLM forward pass output (full vocabulary log-softmax)
- Subsequent tokens: full `GenerationResponse` logprobs via `run_generation()` (same as text path)
- First token yields a `_VisionTokenResponse` with logprobs, not a standard `GenerationResponse`

### Key Features

1. **Full vocabulary access** - logprobs array contains complete log-softmax over entire vocabulary
2. **No additional overhead** - logprobs already computed by mlx-lm during generation
3. **Tokenizer access** - provider.processor.tokenizer used for decoding tokens
4. **Streaming-aware** - works with both streaming and non-streaming responses

For logprobs implementation details, see [logprobs.md](../../internal/backend/logprobs.md) (if it exists) or consult `src/heylook_llm/logprobs.py` directly.