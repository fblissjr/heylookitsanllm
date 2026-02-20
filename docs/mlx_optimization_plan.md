# mlx engine optimization plan

last updated: 2026-02-20

## context

The heylookitsanllm backend was built with server-grade patterns (batching, queuing, multi-tenant support). For a single user on Apple Silicon, these add latency and complexity without benefit. An audit identified 6 optimization areas. After auditing the codebase, 3 are already partially implemented (KV quantization, speculative decoding, strategy-based provider unification), 1 was dead code (`mlx_optimizations.py`), and 2 are real opportunities (radix-tree caching, `mx.compile` on forward pass).

This plan phases the work from lowest-risk cleanup through high-impact optimizations.

---

## phase overview

| Phase | Goal | Risk | Impact | Status |
|-------|------|------|--------|--------|
| **1** | Simplify and harden foundation | Low | Remove dead code, add abort mechanism | DONE (v1.13.0) |
| **2** | Radix-tree prefix caching | Medium | Near-instant TTFT on conversation turns | Planned |
| **3** | Static graph compilation | Medium | Eliminate JIT recompilations during decode | Planned |
| **4** | Speculative decoding hardening | Medium | 2x+ TPS with compatible draft models | Planned |
| **5** | Provider unification | High | Single generation loop for all modalities | Planned |

---

## phase 1: simplify and harden foundation (DONE)

### 1a: dead code removal

**Deleted `src/heylook_llm/optimizations/mlx_optimizations.py`** (283 lines).

Full audit confirmed every function used hallucinated APIs (`mx.set_memory_limit_policy`, `model.allocate_kv_cache`, `mx.fast.matmul`, etc.). Zero imports anywhere in production code. The `optimizations/__init__.py` only imports `fast_json`, `fast_image`, and `status` -- unaffected.

### 1b: queue removal

**Deleted `src/heylook_llm/queue_manager.py`** (~387 lines) and all queue branches from `router.py`, `api.py`, `messages_api.py`.

Queue manager was disabled by default (`queue_config.enabled: false`). The `should_use_queue()` branch evaluated on every request for no reason. The queue's `process_with_queue()` collected full results via `list(generator)`, making it fundamentally incompatible with streaming. Its `asyncio.get_event_loop()` from a daemon thread was also unreliable.

### 1c: abort/cancellation mechanism

**Created `src/heylook_llm/providers/abort.py`** with `AbortEvent` class.

`AbortEvent` wraps `threading.Event` for thread-safe cooperative abort:
- `set()` -- signal abort
- `clear()` -- reset for new generation
- `is_set()` -- check if aborted

**Integration points:**

1. **Preemption at lock acquisition** (`mlx_provider.py`):
   ```
   lock_acquired = _generation_lock.acquire(blocking=False)
   if not lock_acquired:
       _abort_event.set()           # abort current generation
       _generation_lock.acquire()   # wait for it to finish
   _abort_event.clear()             # clean slate for new generation
   ```

2. **Per-token check in all three strategies** (`TextOnlyStrategy`, `VLMTextOnlyStrategy`, `VLMVisionStrategy`):
   ```
   for response in lm_stream_generate(...):
       if abort_event and abort_event.is_set():
           break                    # exits cleanly, finally blocks run
       yield response
   ```

3. **Disconnect detection in SSE streaming** (`api.py`, `messages_api.py`):
   ```
   # Poll disconnect every 100ms while awaiting next token
   while not chunk_future.done():
       if await request.is_disconnected():
           provider._abort_event.set()
           return
       await asyncio.sleep(0.1)
   ```

**Thread safety:** `threading.Event` is thread-safe. The clear/set/check sequence is safe because `clear()` always happens after lock acquisition.

**Cache correctness:** On abort, `prompt_cache.tokens` reflects only the tokens actually computed (truncated is fine -- next request finds shorter prefix and recomputes).

**Known limitation:** Abort happens between tokens, not mid-token. If a single token takes 500ms, we cannot abort until it completes. Long-term fix: contribute `abort_event` parameter upstream to `mlx_lm.generate.stream_generate`.

---

## phase 2: radix-tree prefix caching

### goal

Replace linear prefix matching in `prompt_cache.py` with a radix tree. TTFT drops from seconds to milliseconds for iterative conversations (editing, regeneration, follow-ups).

### how it works

Current implementation (`providers/common/prompt_cache.py`) does linear prefix matching: compare the new token sequence against the cached sequence element-by-element. This is O(n) where n is prompt length. Worse, it only supports one cached prefix per model.

A radix tree stores token subsequences as tree paths. Each node represents a block of tokens (e.g., 16 tokens) and stores the corresponding KV cache tensors. Lookup is O(k) where k is the number of shared nodes, not the total prompt length.

### design

```
RadixTreeNode:
    token_block: tuple[int, ...]   # 16-token block of token IDs
    kv_tensors: mx.array            # KV cache for this block
    children: dict[int, RadixTreeNode]  # first token of next block -> child

RadixTreeCache:
    root: RadixTreeNode
    max_nodes: int                  # memory limit

    def longest_prefix_match(tokens: list[int]) -> (matched_length, kv_cache)
    def insert(tokens: list[int], kv_cache: mx.array) -> None
    def evict_lru() -> None
```

### key properties

- **Block size of 16 tokens**: Balances granularity vs. overhead. Typical conversation turn shares hundreds of tokens with the previous prefix.
- **Delta-only prefill**: After matching the longest prefix, only the new tokens (from the match point onward) need prefill. For a 2000-token conversation where the user adds 50 tokens, prefill is 50 tokens instead of 2000.
- **Memory-bounded**: `max_nodes` prevents unbounded growth. LRU eviction removes least-recently-accessed branches.
- **Multiple prefixes**: Unlike current single-prefix cache, radix tree supports multiple conversation branches (e.g., regenerating different responses).

### key files

- `providers/common/prompt_cache.py` -- replace `PromptCache` internals
- `providers/common/cache_helpers.py` -- new file for radix tree data structure
- `providers/mlx_provider.py` -- update strategy cache integration

### verification

- Unit tests: prefix matching correctness, eviction behavior, block boundary edge cases
- Benchmark: TTFT comparison (before/after) on multi-turn conversation with 2000+ token context
- Memory test: verify bounded growth under sustained usage

---

## phase 3: static graph compilation

### goal

Wrap the decode step in `mx.compile` to eliminate JIT recompilations during the decode loop. Target: zero recompilations after first token.

### background

MLX compiles operations lazily into Metal compute graphs. Each unique combination of tensor shapes triggers a new compilation. During autoregressive decoding, shapes change on every step (KV cache grows by 1 each token), causing repeated recompilations.

### design

1. **Pre-allocate KV caches** at max context length using `mx.zeros`:
   ```python
   k_cache = mx.zeros((max_ctx, n_heads, head_dim))
   v_cache = mx.zeros((max_ctx, n_heads, head_dim))
   ```

2. **In-place slice updates** instead of concatenation:
   ```python
   k_cache[pos] = new_k  # fixed shape operation
   v_cache[pos] = new_v
   ```

3. **Position as `mx.array` scalar** (not Python int) to prevent graph recompilation:
   ```python
   current_pos = mx.array(0)  # stays as mx.array throughout
   ```

4. **Compile the decode step**:
   ```python
   @mx.compile
   def decode_step(tokens, pos, k_cache, v_cache):
       logits = model(tokens, cache=(k_cache, v_cache), pos=pos)
       return logits, k_cache, v_cache
   ```

### key challenges

- `mx.compile` requires all inputs and outputs to be `mx.array` -- no Python int positions
- Pre-allocated KV caches must be masked/sliced correctly during attention
- Need to verify compiled graph produces identical output to non-compiled path
- Prompt cache integration: the pre-allocated cache must be compatible with the existing `process_prompt_with_cache` flow

### key files

- `providers/mlx_provider.py` -- modify strategies to use compiled decode
- New `providers/common/compiled_decode.py` -- compiled decode step implementation
- Unit tests comparing compiled vs. non-compiled output

### verification

- Correctness: output comparison between compiled and non-compiled decode for 100+ tokens
- Performance: measure recompilation count (via MLX debug logging) before and after
- Memory: verify pre-allocated caches don't exceed available unified memory

---

## phase 4: speculative decoding hardening

### goal

Draft model support exists but is lightly integrated. Make it reliable for production use with automatic configuration and monitoring.

### current state

- `TextOnlyStrategy` and `VLMTextOnlyStrategy` already pass `draft_model` to `lm_stream_generate`
- Draft model loading exists in `load_model()` via `config.get('draft_model_path')`
- No automatic draft model selection, no acceptance rate monitoring, no tuning

### planned improvements

1. **Automatic draft model selection**: Given a target model (e.g., Qwen3-8B), automatically select a compatible draft model (e.g., Qwen3-0.6B) from the model registry. Compatibility means same vocabulary and architecture family.

2. **Speculation depth tuning**: Default k=4-5 tokens per speculation step. Monitor acceptance rate and adjust dynamically:
   - acceptance rate > 80%: increase k
   - acceptance rate < 50%: decrease k or disable speculation

3. **Acceptance rate monitoring**: Track per-model speculation stats:
   ```python
   class SpeculationStats:
       total_draft_tokens: int
       accepted_tokens: int
       acceptance_rate: float  # accepted / total
       avg_speedup: float      # effective TPS / base TPS
   ```

4. **Default-on for compatible pairs**: When a draft model is available and compatible, enable speculative decoding by default. User can opt out via config.

### key files

- `providers/mlx_provider.py` -- speculation stats, dynamic k adjustment
- `model_service.py` -- automatic draft model discovery
- `config.py` -- draft model configuration options
- `models.toml` -- per-model draft_model_path entries

### verification

- Correctness: speculative output must be identical to non-speculative (by construction -- rejected tokens are re-sampled)
- Performance: measure TPS with and without speculation across model sizes
- Acceptance rate logging: dashboard-visible stats for tuning

---

## phase 5: provider unification

### goal

Merge the VLM vision path into a single unified provider. Vision becomes a preprocessor rather than a separate generation path.

### current state

Three generation strategies exist:
- `TextOnlyStrategy` -- uses `mlx_lm.stream_generate` directly
- `VLMTextOnlyStrategy` -- wraps VLM's `language_model` component with `LanguageModelLogitsWrapper`, uses `mlx_lm.stream_generate`
- `VLMVisionStrategy` -- uses `mlx_vlm`'s custom generation with `create_vlm_generator_with_sampling`

The split means:
- Optimizations (compilation, caching) must be implemented 3 times
- Speculative decoding only works for text paths
- Testing surface is 3x larger

### design

Vision as a preprocessor:
```
image -> vision_tower -> image_embeddings
text -> tokenizer -> text_embeddings
[image_embeddings, text_embeddings] -> unified_LLM_forward -> tokens
```

The key insight: VLM models already have a language model component that works the same as text-only models. The only difference is how the input embeddings are constructed. By extracting vision processing into a separate preprocessor step, the generation loop becomes identical for all modalities.

### unified generation flow

```python
class UnifiedStrategy:
    def generate(self, request, ...):
        # Step 1: Build input embeddings
        if has_images:
            embeddings = vision_preprocessor.encode(images, text, processor)
        else:
            embeddings = text_embeddings(text, tokenizer)

        # Step 2: Generate (same code path for all modalities)
        for token in lm_stream_generate(
            model=language_model,
            inputs_embeds=embeddings,  # key: pass embeddings, not tokens
            sampler=sampler,
            ...
        ):
            yield token
```

### prerequisites

- Phase 3 (compiled decode) should be done first -- unification means one compilation path instead of three
- Phase 4 (speculative decoding) should be done first -- draft model compatibility is simpler with unified path
- `mlx_lm.stream_generate` must support `inputs_embeds` parameter (check upstream API)

### key files

- `providers/mlx_provider.py` -- merge three strategies into one
- `providers/common/vlm_generation.py` -- refactor into vision preprocessor
- `providers/common/vision_preprocessor.py` -- new file for image -> embeddings conversion

### verification

- Correctness: output comparison for text-only, VLM text-only, and VLM vision between unified and legacy paths
- Performance: verify no regression from the indirection
- Test reduction: merge 3 sets of strategy tests into 1

### risk

This is the highest-risk phase because it touches the core generation loop. Mitigation:
- Keep legacy strategies as fallback behind a config flag
- Extensive A/B testing before removing legacy code
- Only proceed after Phases 3 and 4 prove the unified approach works

---

## dependency graph

```
Phase 1 (DONE)
    |
    +-- Phase 2 (radix-tree caching) -- independent
    |
    +-- Phase 3 (compiled decode)
    |       |
    |       +-- Phase 4 (speculative decoding hardening)
    |               |
    |               +-- Phase 5 (provider unification)
    |
    Phase 2 can proceed in parallel with Phase 3
```

Phases 2 and 3 are independent and can be worked on in parallel. Phase 4 depends on Phase 3 (compiled decode). Phase 5 depends on both 3 and 4.

---

## expected impact summary

| Metric | Current | After Phase 2 | After Phase 3 | After Phase 4 | After Phase 5 |
|--------|---------|---------------|---------------|---------------|---------------|
| TTFT (multi-turn) | 1-3s | ~50ms | ~50ms | ~50ms | ~50ms |
| TPS (decode) | baseline | baseline | +10-20% | +100-200% | +100-200% |
| Code complexity | 3 strategies | 3 strategies | 3 strategies | 3 strategies | 1 strategy |
| JIT recompilations | per-token | per-token | 0 after first | 0 after first | 0 after first |
