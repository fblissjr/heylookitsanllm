# Performance & Optimization Guide

This document covers the end-to-end performance architecture and optimization strategies implemented across `heylookitsanllm`, spanning Apple Silicon hardware utilization, in-process MLX caching, GGUF/`llama-server` flags, and frontend rendering pipelines.

---

## 1. Hardware & Engine Concurrency on Apple Silicon

Apple Silicon architecture features a high-bandwidth unified memory bus shared between CPU cores, the Metal GPU, and the Neural Engine. Achieving peak tokens-per-second requires coordinating memory access and execution threads carefully.

```mermaid
flowchart TD
    subgraph AppleSiliconHW ["Apple Silicon Unified Memory Subsystem"]
        RAM["Unified Memory (machine-dependent)"]
        MetalCeiling["Metal Recommended Working Set (below physical; raise via iogpu.wired_limit_mb)"]
        GPU["Metal GPU Core Compute (Shaders)"]
        CPU["High-Efficiency & High-Performance CPU Cores"]
    end

    RAM --> MetalCeiling
    MetalCeiling --> GPU
    RAM --> CPU

    subgraph SWGate ["Software Invariants & Concurrency Guards"]
        Gate["Process-Global FIFO GenerationGate (one generation at a time)"]
        Pool["_PinnedExecutorPool (Non-terminating generation threads)"]
        Wired["mx.set_wired_limit (Lock weights in physical RAM)"]
    end

    Gate --> GPU
    Pool --> CPU
    Wired --> RAM
```

### 1.1. Single-Tenant Serialized Inference: Process FIFO Gate
- **Why one at a time**: the reason in [`generation_gate.py`](../../src/heylook_llm/providers/common/generation_gate.py) is **one GPU**, not bandwidth economics -- a single GPU with one loaded model and a shared KV cache admits one generation, so concurrent requests should queue and each complete rather than the newest aborting the in-flight one. [`mlx_provider.py`](../../src/heylook_llm/providers/mlx_provider.py) adds the cross-model half: serialize across all loaded MLX models, or two providers run concurrent generations on the shared Metal command queue. **No throughput mechanism is claimed** -- §3.5 below explains why an unmeasured causal story is exactly what this repo stopped writing down.
- **Scope**: the gate has two consumers, the MLX and llama-server providers.
- **Admission Queue**: admits waiting requests in arrival order up to the configured queue depth ([`config.py`](../../src/heylook_llm/config.py)); a request arriving when it is saturated gets HTTP 503 via `ModelBusyError`.
- **Queue Transparency**: Heylook acquires the gate *before* forwarding requests to `llama-server`. This avoids queueing requests inside `llama-server`'s internal HTTP queue where they could silently exceed socket read timeouts.

### 1.2. Thread Lifecycle: `_PinnedExecutorPool`
Earlier versions built one executor per request and shut it down at stream end. MLX keeps thread-local Metal state, and a pthread's TLS cleanup runs **after** its Python thread state is gone -- so deallocating those objects happens **without the GIL**, which is a `Py_FatalError` and a `SIGTRAP` process abort rather than an exception anything can catch.
- [`streaming_utils.py`](../../src/heylook_llm/streaming_utils.py) introduces **`_PinnedExecutorPool`**: worker threads allocated for MLX generation remain permanently alive for the lifetime of the process.

### 1.3. Memory Locking: `wired_limit`
Under memory pressure, macOS may compress inactive memory pages or swap them to disk.
- At startup, the server configures `mx.set_wired_limit()` to tell the macOS Mach kernel to lock active model weights into physical RAM, preventing paging pauses during generation.
- Each generation runs inside mlx-vlm's `wired_limit()` for stream synchronization, and `vlm_engine` calls `mx.reset_peak_memory()` at the request's start (the vision strategy does it before encoding images) so `mx.get_peak_memory()` is scoped per request.
- **`iogpu.wired_limit_mb` is the one lever that raises the Metal working set.** `scripts/gpu_wired_limit.sh` reads it (`status`, no root), sets it for the current boot, or installs it to persist -- the last two need root. Raising it is what flips large gguf models to the wider micro-batch on its own, and what the gguf provider's error text points at when a decode dies of "Compute error." The ceiling is **hard for MLX and advisory for llama.cpp**.

---

## 2. In-Process MLX Optimizations

### 2.1. The MLX Prefix Cache and B=1 Runner
For single-row MLX requests using supported standard, sliding-window, or hybrid recurrent caches (including `mtp` drafters), [`b1_runner.py`](../../src/heylook_llm/providers/common/b1_runner.py) and [`prefix_cache.py`](../../src/heylook_llm/providers/common/prefix_cache.py) replace per-request `BatchGenerator` / `APCCoordinator` batching (`v2.0.258`), while keeping `BatchGenerator` + upstream APC as the automatic fallback for `HEYLOOK_B1_RUNNER=0`, KV quantization (`kv_bits`), non-MTP drafters, pipeline-parallel workers (`_heylook_apc_sync_wrapped`), and unsupported cache classes:
- **Heylook B=1 runner & CoW prefix cache (`b1_runner.py`, `prefix_cache.py`)**: `PrefixCacheStore` stores zero-copy CoW MLX slice views (`KVCache`, `RotatingKVCache`, `BufferedRotatingKVCache`, `ArraysCache`, and `CacheList`) scoped by the same media-only `semantic_hash` salt and LRU / metal-headroom budget (`APC_CHECKPOINT_ENTRIES`, `APC_MEMORY_SHARE`), with `vlm_engine.refresh_snapshots`, `apc_is_empty`, and `apc_memory_pressure` covering both stores. Non-recurrent requests defer prefill snapshot storage to post-generation (`finally:`) so `KVCache` keeps `refcount == 1` during decode and shares one backing buffer across `[prompt]` (`cache_trim_tail = cur_off - n`) and `[prompt + gen]` (`trim_tail = 0`), falling back to `snapshot_rotating_fallbacks` only on rotating layers that will overflow `buffer_size` during long generations; hybrid recurrent `ArraysCache` requests still split the tail into `N - 1` + `1` before decode. Oversized chunk-prefill rotating slice views detach post-generation via `compact_pending()` on `_generation_stream()`.
- **Rolling-window rollback slack (`_ROTATING_SLACK_TOKENS` / `min_match_len`)**: because Gemma 4's chat template appends a short open-turn suffix (such as `<|thought>\n<thought|>`) on the active generation prompt and omits it when re-rendering closed historical assistant turns, `PrefixCacheStore` retains up to `_ROTATING_SLACK_TOKENS` pre-window tokens in rotating snapshots (`min_match_len`) and trims back down to `max_size` on restore in `_restore_layer`, allowing multi-turn follow-ups past `sliding_window` to hit near the prior turn end instead of falling back to an interior checkpoint.
- **Drafted Gemma 4 rotating cache rollback**: on Gemma 4 MTP requests, `b1_runner` converts plain `RotatingKVCache` layers (`keep == 0`) to `BufferedRotatingKVCache`, `gemma4_vision.gemma4_make_masks` keeps single-row unpadded decode and native `window_size=` SDPA active on chronological `BufferedRotatingKVCache` buffers, and `gemma4_runtime._bind_speculative_verify_hidden` routes `BufferedRotatingKVCache` (`keep == 0`) through `SpeculativeCacheTransaction`'s O(1) `trim()` rollback rather than wrapping it in `_RotatingCacheTransaction`.

Cross-request reuse on the `BatchGenerator` fallback path is mlx-vlm's automatic prefix cache (APC), one in-memory store per loaded model, driven by [`vlm_engine`](../../src/heylook_llm/providers/common/vlm_engine.py) (the runtime visibility plan's W10). How it reuses depends on the model's cache layers:
- **Checkpoint models** (hybrid recurrent such as qwen3_5, sliding-window such as gemma-4 and gpt-oss) restore whole-cache checkpoints taken **during prefill**, at a fixed token interval near the end of the prompt plus the prompt's end. A follow-up restores the last checkpoint before the point where it diverges -- which is why these models reuse past their sliding window and why a hybrid's recurrent state is never sliced. The interval and the number of near-end checkpoints one request takes are `vlm_engine.APC_CHECKPOINT_INTERVAL_TOKENS` and `APC_CHECKPOINT_CAPTURES`; at mlx-vlm's defaults only the prompt end and one far boundary survive, and a follow-up finds nothing (measured in the W10 spike).
  - **System-prompt and coarse interior checkpoints**: a request checkpoints where a leading system prompt ends (`vlm_inputs.system_prefix_tokens`, computed once per system prompt, `depth`/thinking setting, and loaded model from the active template; `VLMVisionStrategy` passes `depth=` alongside text) so a new conversation sharing that system prompt restores it instead of re-prefilling. For `user_only` requests and long user-side preambles that lack a system turn boundary, `capture_lengths` also adds coarse interior checkpoints (`APC_INTERIOR_STRIDE_TOKENS` / `APC_INTERIOR_MAX_CAPTURES`, power-of-2 stride doubling) so the second request already restores a prefix rather than waiting for an observed divergence snapshot on the third. A directional micro-delta guard drops an observed `stable` snapshot only when a previous planned or resident snapshot already lies within `< eff_interval` before it, preventing near-duplicate captures without hiding an earlier divergence. Because a conversation's later turns restore from the previous turn's snapshot, `vlm_engine.refresh_snapshots` keeps the system-prompt snapshot at the recent end whenever a prompt still begins with it. `vlm_engine.install_capture_policy` installs this rule on each generator; otherwise it is mlx-vlm's own rule.
  - **The store keeps many conversations' checkpoints** (`APC_CHECKPOINT_ENTRIES`), not one request's, so switching to another chat and back restores the first. mlx-vlm spends one number on both the checkpoints per request and the store size; heylook separates them, and the store's byte budget (below) also bounds it on long prompts. Each retained snapshot costs every request a little CPU in mlx-vlm's byte accounting, which is why the count stays modest ([measured]: internal/claude/improve scoreboard pairs ab2-ttft and ab3-ret).
  - **Trimmed sliding-window snapshots & single-row batch restore (`PATCH_STORE_CHECKPOINT`, `PATCH_RESTORE_BATCH`)**: before APC deep-copies a prefill-chunk checkpoint, `PATCH_STORE_CHECKPOINT` in [`mlx_vlm_patches.py`](../../src/heylook_llm/providers/common/mlx_vlm_patches.py) runs one batched `mx.eval` over batch padding/offset arrays and passes zero-copy single-row views with rotating layers trimmed to `max_size` (preserving `keep` sink tokens and true sequence `offset` from `c.offset[0]`) straight to `APCManager.store_exact_cache`, dropping unused chunk overshoot past the sliding window and removing the intermediate batch extraction copy and per-layer syncs. On restore for drafted single-row checkpoint generators (which keep batch caches rather than taking `#2356`), `PATCH_RESTORE_BATCH` wraps those single-row snapshots into `BatchRotatingKVCache` / `BatchKVCache` / `ArraysCache` in-place, preserving pre-reserved `KVCache` capacity and replacing upstream's double copy and per-layer `mx.eval` barriers with one batched eval (local measurements in `internal/claude/perf/apc_fixes_2026-11-19/`).
- **Plain KV models** (qwen3, qwen3_vl) reuse hashed blocks of the prompt.

A lone restored request gets the cache a cold one would: its own warm cache rather than a batch merge of it, so on qwen3_5 it decodes like the same request run cold. A drafted request keeps batch caches on restore, as it does cold, because speculative rollback runs through the batch-cache contract. That is heylook's own upstream PR (Blaizzy/mlx-vlm#2356), carried as a wrapper in `providers/common/mlx_vlm_patches.py` that each request's generator gets (`apply_to`) until the pin includes it.

Every restore is checked the same way: `scripts/chain_probe.py` runs each hop of a multi-turn chain fresh and restored at temperature 0 and requires identical text. A divergence at an exact top-2 tie is rounding drift, not a restore bug; read the margin before concluding either way.

Rules the engine follows, each learned the hard way in the spike:
- **The cache key's salt is the request's media only**, computed the way mlx-vlm's server computes it. Left to the generator, the salt folds in the whole prompt's embeddings and no two different prompts ever share.
- **The disk tier stays off**: it would write prompt- and image-derived cache state to disk.
- **The memory budget is heylook's**, a share of the Metal working set (`vlm_engine.APC_MEMORY_SHARE`), not mlx-vlm's automatic one, which is capped at the same size for every model. A snapshot copies every layer's cache, so its size follows the model's KV layout: under that shared cap a dense model held a fraction of the snapshots a MoE model of similar size did, and going back to an earlier chat re-prefilled on the dense model only (local measurement in `internal/claude/perf/apc_budget_gemma4_2026-11-19/`). APC still checks live headroom against its reserve before every store and restore, so a larger budget costs what an idle store holds, not memory safety. `APC_MEMORY_MAX_GB` in the environment overrides it; `engine.cache.memory_budget_bytes` reports it. A cache clear or an unload drops the store.

Known gaps, each an accepted owner decision with its write-up under `internal/claude/w10/`:
- **A turn that adds a new image re-prefills.** The request's images are keyed as one hash, so a new image changes every key. A text follow-up in an image conversation (same images) reuses.
- **Plain KV vision models (qwen3_vl) do not reuse image turns**: the cache discards in-memory blocks that contain an image, and only the disk tier would serve them.

Lessons from the single-slot design this replaced (history: [sharp_edges.md](../architecture/sharp_edges.md#prompt-cache)): never hand out live cache objects, since a quarantined zombie generation keeps rebinding them; materialize snapshots on the thread that made them; and check both engines' cache classes before trusting a round trip (mlx-vlm's rotating cache kept its position in `meta_state`, which the slot once dropped).

### 2.2. Vision Feature LRU Cache
In multimodal VLM conversations, re-evaluating high-resolution images across multi-turn exchanges is computationally expensive.
- [`vision_feature_cache.py`](../../src/heylook_llm/providers/common/vision_feature_cache.py) maintains an LRU cache of encoded vision features. [The vision strategy](../../src/heylook_llm/providers/mlx_provider.py) hands every model the cache as mlx-vlm's `vision_cache` plus `_image_key` kwargs, the way mlx-vlm's own server does, and the model looks up and stores its own tower output (qwen3_5, gemma4, deepseek_v4 and others read them; a model that reads neither ignores them and simply is not cached).
- **The key is the content of the request's whole image list**: a hash of each loaded image's pixels, joined in order (`image_content_key`), not one entry per image. The same picture keys the same whether it came as a data URL or a web link, and a web link whose file changed misses.
- So a turn whose image list is unchanged skips the vision tower, but a turn that **adds** an image misses and re-encodes every image in the history. A per-image cache is part of the plan's W10.

### 2.3. Streaming Detokenizer
The engine streams through **mlx-lm's** streaming detokenizer, vendored as [`lm_detokenizer.py`](../../src/heylook_llm/providers/common/lm_detokenizer.py) (mlx-lm itself is not a dependency), not mlx-vlm's. mlx-vlm's BPE detokenizer flushes only when a token starts with a space, so an answer with no spaces -- a count, code, CJK text -- arrives in one lump at the end; mlx-lm's streams per token (checked on Qwen3.5's tokenizer, 2026-09-23; the smoke walk-away check caught it).
- `MLXProvider.load_model` primes it with `model_path` (`generation_core.detokenizer_source`), so the class comes from tokenizer.json's decoder -- SPM or BPE -- instead of the naive one, which re-decodes the whole current line on every token.
- A continuation keeps its first token's leading space: `continuation_detokenizer` seeds a class with a settable `text`, and only such a class.

### 2.4. Logits Processors Receive a Batch Axis
A logits processor is called as `(tokens, logits)` with logits shaped **`(1, vocab)`** -- the engine passes `logits[:, -1, :]`, keeping the batch axis. Index the **vocab** axis (`logits.shape[-1]`, 1-D scratch vectors that broadcast), never `zeros_like(logits).at[tokens]`: that scatters along the size-1 batch axis, and MLX does not bounds-check a Metal scatter. The result is silent memory corruption followed by a mid-generation Metal fault and a poisoned process. Unit tests using 1-D logits stay green against it -- test the shape the engine actually sends.

---

## 3. GGUF / Llama-Server Optimizations

### 3.1. Embedded Metal Shaders
`scripts/build_llama.py` compiles `llama-server` with `GGML_METAL_EMBED_LIBRARY=ON`, putting the Metal shader **source** inside the binary; the OS's Metal compiler builds it when the process starts. The point is portability, not startup latency: the binary can be moved without dragging a loose shader file along, which is what a subprocess-spawning provider wants. A side effect worth knowing is that shader code generation tracks the installed macOS rather than Xcode (the [gguf runtime audit](../testing/gguf_runtime_audit_2026-09-23.md) §1 has the rest of the build review). `BUILD_SHARED_LIBS=OFF` completes that picture with a single self-contained static binary.

`GGML_METAL_NDEBUG` is deliberately **not** set. It compiles out load-time logging only -- including the *"allocated size is greater than the recommended max working set size"* warning, which is the ceiling that actually refuses loads on a big-unified-memory Mac. No throughput to gain, real diagnostics to lose.

### 3.2. KV Cache Quantization (`-ctk` / `-ctv`)
Long-context workloads can cause KV cache memory to rival model weights in size.
- **The KV cache is `f16` by default and stays that way.** `cache_type_k` / `cache_type_v` exist for headroom emergencies, not as a tuning default.
- When headroom is genuinely the binding constraint, quantizing key and value tensors frees unified memory for a larger context. No quality figure is quoted here because none has been measured on this hardware for these models. **A quantized V-cache needs flash attention** (§3.6), so check that before assuming `-ctv` is free.

### 3.3. Automatic Micro-Batch Sizing (`-ub`)
`n_ubatch` is unset by default and resolves **at spawn**, not to a constant:
- The provider sizes the model the way the admin fit panel does (weights plus sidecars against the live Metal working set) and reads the resulting headroom.
- Clearing the thin-headroom threshold gets the wider micro-batch; otherwise the spawn inherits llama-server's own default. Both the threshold and the wide value are constants in [`ram_fit.py`](../../src/heylook_llm/ram_fit.py) and [`llama_server_provider.py`](../../src/heylook_llm/providers/llama_server_provider.py) -- read them there. A stored `n_ubatch` wins over the auto answer in both directions.
- **The spawn log names the micro-batch it resolved.** The answer moves with `iogpu.wired_limit_mb` and with what else sits in the model directory, so a spawn quietly taking the narrow default would otherwise be indistinguishable from one quietly taking the wide one. (A *stored* value short-circuits the auto path without its own line; the full argv in the spawn log is what discloses that case.)

The wider micro-batch is a prefill win on dense and MoE models at no generation cost, but it costs materially more compute buffer -- which is why it is conditional. A vision model with thin headroom *loaded* at the wide setting (with `--fit` quietly trimming its context) and then died in its first decode with a Metal OOM that llama.cpp's own pre-flight never saw.

### 3.4. Idle Hibernation (`--sleep-idle-seconds`)
When `sleep_idle_seconds` is configured, `llama-server` frees model weights and KV buffers after the idle timeout while keeping the HTTP process alive. It is **strictly cheaper than heylook's own idle-unload**, which SIGTERMs the process group and respawns it -- that is the comparison the config field makes, and it is a local subprocess comparison, not a network one.

Do not read "cheaper" as "fast". A sleeping server **reloads the model before it emits anything**, and for a large model that reload is minutes rather than seconds -- which is why the provider raises its socket read timeout to the startup timeout for a wake instead of the normal SSE read timeout. How the weights come back is a per-model choice (`load_mode` / `-lm`), not fixed to `mmap`.

### 3.5. Speculative Decoding Mechanics & Observations
Speculative decoding uses a lightweight draft model or next-token prediction head (`-md`, `--spec-type`) to propose $N$ draft tokens that the main model verifies in a single forward pass.
- **Adaptive MTP block sizing** (`mlx_vlm.speculative.mtp._effective_mtp_block_size`, MLX only -- the gguf path's equivalents are spawn flags): `DraftTuner` was removed from `generation_core.py`; upstream `_mtp_next_block_size` / `_effective_mtp_block_size` treats a user-requested block total above the assistant's configured depth as a ceiling, falling back to the configured depth until recent `accept_lens` show consistent prefix hits at that depth (bypassed when `prefer_requested_block_size` is set). Sweeps on MoE Gemma 4 keep the default block size because larger blocks cross `SwitchGLU` `gather_qmm` tiling boundaries (local measurement in `internal/claude/perf/gemma4_mtp_v20250_2026-11-16.md`).
- **Speculative decoding defaults to OFF**, per model, with one standing carve-out: the DeepSeek-V4-Flash entry keeps it on (`draft-dspark`).

**What "default OFF" means here is *unproven*, not *known harmful*** -- and the durable content of this subsystem is measurement discipline rather than any number:

- On the one case examined most carefully -- a dense gemma-4 MTP model at vendor sampling, realistic context, matched warm cache, long generation -- spec on versus off was **a wash**, indistinguishable from noise. Every larger effect seen alongside it dissolved once one more variable was controlled: an apparent tuning win was a greedy artifact, an apparent cost was a short-generation artifact, an apparent context effect was a prompt-cache ordering mistake, and a "broken drafter" was refuted by the drafter's own output.
- **The mechanism is not understood.** Draft volumes collapse as context grows and no explanation for that has been established here, so attributing the wash to memory-bus contention would be a guess rather than a finding.
- `spec_draft_n_max` and `spec_draft_p_min` **interact, and the interaction inverts** -- a one-dimensional sweep of either finds a different and wrong optimum. There is no defensible global default: `p_min` values that help one model family cost throughput on another at every value tested.
- **Never measure this at temp 0.** Greedy acceptance is exact argmax matching; temp > 0 is rejection sampling. That is a different regime, not a quieter one. Temp 0 is fine for reproducibility and never for a throughput claim.
- Match prompt length, generation length, seed, sampling, **which binary**, and prompt-cache state across arms. An unmatched cache produced a large phantom that survived repeats and looked exactly like a finding. Matching every control you thought of only rules out the confounds you imagined.
- **MLX drafts too (v2.0.187)** with a standalone mlx-vlm drafter folder (`<model>-mtp`, from `mlx_vlm.convert --mtp` or `mlx_vlm.split_mtp`), paired by discovery and loaded next to the model (`MLXProvider._load_drafter`, which binds `mlx_vlm_patches.bind_mtp_drafter` on all `kind == "mtp"` drafters plus `gemma4_runtime.bind_gemma4_drafter` on Gemma 4); mlx-vlm's `BatchGenerator` runs the rounds. Greedy decoding matches exact argmax (`T = 0`), non-greedy MTP requests across all model families use lossless Leviathan rejection sampling against the target's filtered distribution (`samplers.build._heylook_spec_probs`, `_heylook_pending_draft` captured via `bind_mtp_drafter` and `_HeylookGemma4DraftModel.draft_block`, and `PATCH_MTP_B1_WALK`, preserving exact target marginal distributions without collapsing acceptance to exact token equality), and non-greedy requests without draft distributions sample target positions sequentially; every emitted token remains faithful to the target distribution at any temperature. While upstream `SpeculativeGenerationBatch` takes no `logits_processors`, stateless generated-token penalties (`repetition_penalty`, `presence_penalty` tagged `_heylook_mtp_safe` and preserved by `generation_core.generated_only`) are permitted on `draft_kind == "mtp"` by `vlm_engine.drafting_blocker` and evaluated per verify/resampling position against the evolving prefix (`gen_history + draft_list[:pos]`) inside `PATCH_MTP_B1_WALK` via `_apply_mtp_penalties` / `_heylook_mtp_penalties` (`greedy_sampling=False`, preserving greedy semantics with `_heylook_greedy`); stateful or structured processors (grammars, JSON schemas, logit biases) and non-MTP drafters still run undrafted rather than losing them, `draft_max_context` (v2.0.246) still routes requests whose prompt exceeds that token count back to single-token target decode where multi-token target verification is slower on long KV prefixes, and a thinking budget still drafts because that loop applies it. On Gemma 4, owned single-row mask handling in [`providers/common/gemma4_vision.py`](../../src/heylook_llm/providers/common/gemma4_vision.py) (`bind_text_masks`, `gemma4_make_masks`, `mlx_sdpa_supports_window_size`) skips redundant per-step `.item()` mask syncs and 4D masks during unpadded single-row decode and verification, and replaces dense 4D sliding-window prefill masks with native `window_size=` SDPA when supported by the loaded MLX build (`mlx >= 0.32.5` / `mlx#4734`, including source overlays built via `scripts/build_mlx.py` outside `.venv`) and `not c.rotated`; owned runtime bindings in [`providers/common/gemma4_runtime.py`](../../src/heylook_llm/providers/common/gemma4_runtime.py) (`bind_gemma4_runtime` and `bind_gemma4_drafter` for both dense and MoE Gemma 4 + MTP) route eligible wide-vocab tied LM heads (`gemma4_qmv_multi`) and dense `gate_proj + up_proj` MLPs (`gemma4_qmv_fused`) through `_hoisted_8bit_qmv_kernel` `uint2`-hoisted 8-bit multi-token QMV kernels (and upstream 4-bit verification kernels), while probing `_native_8bit_qmv_beats_hoisted` once per device/shape and automatically standing down 8-bit wide-matrix hoisted kernels (`skip_8bit_qmv`) when native `mx.quantized_matmul` is faster (e.g. `mlx#4890` + `mlx#4826` on Ultra targets), pass `window_size` sliding-window bounds in `_bind_attention_subclass` / `bind_gemma4_text_layers` when supported, eliminate per-step full-buffer `BatchRotatingKVCache` transaction copies and partial-accept copy kernels past `sliding_window` via `_bind_speculative_verify_hidden` / `_FastB1RotatingTransaction` O(1) tail slicing, replace per-layer array/sync RoPE offsets with transaction-safe `_scalar_rope_offset`, cache pre-scaled MoE `Router` norm weights, extract unpadded `B=1` `shared_kv_from_cache` views without host syncs, and hoist drafter masks plus hand off `_heylook_pending_draft` across `_HeylookGemma4DraftModel.draft_block`; [`providers/common/mlx_vlm_patches.py`](../../src/heylook_llm/providers/common/mlx_vlm_patches.py) adds generator-scoped single-row MTP verification wrappers (`PATCH_MTP_B1_WALK`, `PATCH_MTP_LEFT_PADDING`); and `vlm_engine.install_media_token_ids` preserves contiguous multimodal token spans across APC boundaries (local verification records: `internal/claude/perf/gemma4_mtp_v20249_2026-11-16.md` and `internal/claude/perf/gemma4_mtp_v20250_2026-11-16.md`). A round arrives as several tokens at once; the engine still yields one chunk per token. MLX reports `accepted` and `emitted` (counted per round), not `drafted`.
- **Per-request reporting (v2.0.78).** `performance.speculative` carries drafted (gguf only), accepted and emitted, with `acceptance_rate` (accepted/drafted) and `draft_share` (accepted/emitted) as separate fields: the engines count different things, and one ratio name used to carry either. `performance.cache` says what each request reused of its prompt. Each assistant message keeps what its generation measured (`message_stats`, counts and enum tokens only) and shows it as a muted line under the message, and the perf page aggregates the same reports per model and per hour.

Conditions, history and the underlying figures live in `internal/research/` and in `GGUFModelConfig`'s own field comments. They are deliberately not reproduced in tracked docs: every number quoted for this subsystem has later needed a condition attached to stay true.

### 3.6. Flash Attention (`-fa`)
heylook passes `-fa on` unless a model's `flash_attn` says otherwise (owner call, 2026-10-24). The vision tower inherits the setting, so off is slower for vision encoding and model load: the [audit](../testing/gguf_runtime_audit_2026-09-23.md) §9 and the 2026-10-24 `scripts/perf_ab.py` run measured both. Set `flash_attn = "off"` (or `"auto"`, llama-server's device probe) per model from the Chat page's load panel or the Models page.

### 3.7. Prompt Reuse and Vision Cost on gguf
- **Prompt reuse** is llama-server's own: the slot's longest common prefix, a host-RAM prompt cache (`-cram`, auto-sized at spawn from live working-set headroom when `cache_ram_mb` is unset) that restores earlier states across interleaved conversations, context checkpoints for sliding-window and hybrid models (`--ctx-checkpoints`, with `--checkpoint-min-step` auto-tightened on those architectures so turn boundaries closer than llama-server's default spacing still capture checkpoints), and KV shifting (`--cache-reuse`, auto-enabled on text-only `full_attention` models so mid-prompt edits shift and reuse suffix chunks). It covers image requests, and it depends on the template rendering history the same way it rendered it for generation. The mechanism is in [the deep dive](./llama_server_build_and_spawn.md#46-prompt-reuse-across-requests); the measured behaviour, and the template defect that broke one model's multi-turn reuse, are in the [audit](../testing/gguf_runtime_audit_2026-09-23.md) §5 and §10.
- **An image request's first token** costs the CPU decode of the image, the vision-tower encode, then ordinary prefill of the image tokens and the text. Several images in one request are encoded one after another. Every projector caps the tokens per image and resizes a larger image down before encoding, so above the cap a bigger file costs only its upload and CPU decode; below it, sending a smaller image is the direct lever.
- **Each engine resizes differently**, even for the same model family: different caps, rounding and pad-versus-stretch. The per-engine table is the [audit](../testing/gguf_runtime_audit_2026-09-23.md) §4, recorded as a dated fact rather than something to code against; reporting it per model from the engine itself is the plan's W4.

---

## 4. Frontend Rendering Optimizations

### 4.1. Incremental Markdown Streaming (`MarkdownStream`)
Streaming long responses into the browser DOM can saturate the JavaScript main thread if the full message is repeatedly re-parsed.
- [`markdown-stream.js`](../../frontend/js/markdown-stream.js) cuts only where no markdown construct can span the seam: a line after a blank line that is unindented, **not** a list marker (it would merge with the list above into one loose list) and **not** a blockquote `>`. Fence state is tracked open/close, so a cut never lands inside a fenced block.
- Two constructs instead **disable incremental rendering for the whole message**, because they reach forward past the only thing this scanner treats as a block break: link-reference / GFM footnote definitions, and CommonMark HTML blocks of types 1-5 (type 6 ends at a blank line and is deliberately excluded). That flag **latches** -- once set, every later paint re-renders the whole document. Correctness over speed, deliberately.
- **Prefix Commit**: Content prior to the boundary is parsed once, sanitized via DOMPurify, and committed to the DOM.
- **Tail Isolation**: Only the active, uncommitted tail is updated on each paint. This bounds rendering overhead to the size of the current paragraph or code block, rather than the entire document.

```mermaid
graph LR
    subgraph NaiveRendering ["Naive Re-Rendering (superlinear in document length)"]
        N1["Incoming Token"] --> N2["Re-parse entire document via marked()"]
        N2 --> N3["Re-sanitize entire HTML via DOMPurify()"]
        N3 --> N4["innerHTML assignment & full DOM rebuild"]
    end

    subgraph IncrementalRendering ["Incremental MarkdownStream (cost bounded by the tail block)"]
        I1["Incoming Token"] --> I2{"Safe Block Boundary?"}
        I2 -- Yes --> I3["Parse new block once & append permanent DOM nodes"]
        I2 -- No --> I4["Re-render the active tail only (bounded by the largest single block)"]
    end
```

### 4.2. Layout Cache Preservation
- **Pre-Mutation Measurements**: In [`chat.js`](../../frontend/js/pages/chat.js), viewport and scroll distances are read at the top of the paint cycle *before* mutating DOM nodes. This allows the browser to return cached layout geometry without triggering forced synchronous reflows.
- **Removal of `content-visibility: auto`**: a skipped row reports its `contain-intrinsic-size` estimate rather than its real height until the engine lazily decides it is relevant, so `scrollHeight` lurches and **the engine moves `scrollTop` itself** (clamping, then scroll anchoring). Every scroll decision derives from those two values, so all of them were poisoned. It was measured **on Chrome** -- this is not a WebKit quirk. The reproduction, with its figures, is in the comment at the removal site in [`app.css`](../../frontend/css/app.css); the layout it saved was one-time and desktop-only.
- **Measure at the top of the painter, before it mutates -- both halves matter.** Before the write, the reads are cache hits (layout is still clean from the last paint) rather than a forced re-layout -- and since nothing skips off-screen rows any more, on any engine, a forced layout walks every row in the conversation. And only before the write is it *honest*: measured after, one paint appending more than the slack (a code block, a table) reads as "the reader scrolled away" and strands the view for the rest of the generation. A cached flag fed by scroll events was tried and is **wrong** -- pinning coalesces scroll events to a handful across a whole generation, so the flag goes stale exactly when the viewport changes under it, which on a phone is every keyboard open.

### 4.3. Paint Throttling
- Painters whose cost scales with the document use `ctx.throttleTime`, at the interval each page declares. Both streaming pages use it; a per-*frame* throttle would run up to 120 times a second, which is more work than anyone can read.
- `ctx.throttle` (per-frame, via `requestAnimationFrame`) exists on the page context but currently has **no consumers** -- it is the right tool only for work cheap enough that per-frame is free.
