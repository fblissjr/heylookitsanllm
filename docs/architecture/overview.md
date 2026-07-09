# Backend Architecture

Last updated: 2026-07-09

This document describes the core architecture of the `heylookitsanllm` server: how requests flow from the API layer through the model router to provider backends, and the key technical decisions behind caching and VLM handling.

## Core Mission

A single, unified, OpenAI-compatible API for local LLM inference on Apple Silicon via MLX. The server is memory-efficient and hot-swaps different models on demand, abstracting away differences in model architecture and API contracts.

**Platform:** macOS with Apple Silicon only (MLX uses Metal GPU acceleration).

## High-Level Inference Flow

When a client sends a request to `/v1/chat/completions`, the server executes the following sequence:

1. **API Layer (`api.py`)**: FastAPI receives the request. The body is parsed and validated against a Pydantic model.

2. **Model Routing (`router.py`)**: The validated request is passed to the `ModelRouter`, which manages all provider instances via an LRU cache.
   - If the requested model is already loaded ("hot"), it is used immediately.
   - If the model is not loaded and the cache is full, the least recently used model is evicted.
   - The router then loads the new model by instantiating the appropriate provider.

3. **Provider Logic (`providers/`)**: The router delegates generation to the active provider. The API layer is unaware of backend details.
   - **`MLXProvider`**: Handles text and vision-language models on Apple Silicon. Uses two strategies (`UnifiedTextStrategy`, `VLMVisionStrategy`), both routing through `generation_core.run_generation()` for the decode phase. Vision requests use a pre-filled cache pattern: the VLM runs a forward pass to fill a KV cache, then the language model generates tokens via the same code path as text-only.
   - **`MLXEmbeddingProvider`**: Handles sentence-transformer style embedding models. Produces L2-normalized embeddings via bidirectional attention, mean pooling, and dense projection. Uses dynamic backbone loading via `load_backbone()`.

4. **RLM Endpoint (`rlm.py`)**: An alternative path for long-context tasks. Instead of a single generation call, the RLM engine runs an iterative loop: it sends the context + query to the model, extracts Python code from the response, executes it in a sandboxed REPL, and feeds output back as the next user message. The loop continues until the model calls `FINAL()` or hits max iterations. Uses `provider.create_chat_completion()` directly (no HTTP round-trip) and pins the model via `router.pin_model()` for the duration.

5. **Response Generation**: The provider returns a Python generator.
   - **Streaming** (`stream=True`): Wrapped in a `StreamingResponse`, sending tokens via Server-Sent Events (SSE).
   - **Non-Streaming** (`stream=False`): API layer consumes the entire generator and returns a single `JSONResponse`.

### Visual Flow

```
+-----------+       +-------------------+       +----------------------+
|           |       |                   |       |                      |
| FastAPI   | ----> |   ModelRouter     | ----> |  BaseProvider        |
| (api.py)  |       | (LRU Cache Logic) |       |  (base.py)           |
|           |       |                   |       |                      |
+-----------+       +--------+----------+       +-------+--------------+
      ^                      |                          |
      |                      |                          | (Polymorphic call)
      | (Response)           | (Hot-swaps/Evicts)       v
      |                      |               +--------------------+
      |                      +-------------> | MLXProvider or     |
      |                                      | MLXEmbeddingProvider|
      +------------------------------------  | (Loads model into  |
                                             |  memory/GPU)       |
                                             +--------------------+
```

## Provider Pattern

The architecture uses a **Provider Pattern** to unify disparate backends behind a consistent interface.

- **`BaseProvider`**: An abstract class defining the common contract. Every provider must implement `load_model`, `create_chat_completion`, and `unload`.
- **Concrete Providers**: Each provider encapsulates all backend-specific logic.
  - `MLXProvider` -- text and vision models via mlx-lm / mlx-vlm
  - `MLXEmbeddingProvider` -- embedding-only models; `create_chat_completion` raises `NotImplementedError`
- **`ModelRouter`**: Central controller responsible for instantiating, managing, and hot-swapping providers based on `models.toml` configuration.

Provider type is `Literal["mlx", "mlx_embedding"]` in `config.py`.

## Key Technical Decisions

### LRU Caching and Memory Management

**Problem**: Loading large models into memory is slow and resource-intensive.

**Solution**: The `ModelRouter` implements an LRU cache for provider instances.

- **Hot-Swapping**: A configurable number of models (`max_loaded_models` in `models.toml`) are kept loaded in memory.
- **Eviction**: When the cache is full, the least recently used model is evicted and its `unload()` method is called to explicitly free GPU memory.
- **Provider-Level Caching**: `MLXProvider` leverages `mlx-lm`'s built-in KV cache features (`RotatingKVCache`, `QuantizedKVCache`), configurable in `models.toml`. Radix-tree prompt caching reuses KV state for repeated prefixes.

### Unified Text Strategy + Separate Vision (v1.16.0)

**Problem**: `mlx-lm` and `mlx-vlm` have different entry points and input preparation logic.

**Solution**: `MLXProvider` uses two strategies with shared infrastructure:

1. **`UnifiedTextStrategy`**: Handles all text generation (both text-only models and VLM text-only requests). Dispatches on `is_vlm` for chat template application and model wrapping. The generation loop lives in `generation_core.run_generation()` -- a single function handling cache config, radix-tree prompt cache, `lm_stream_generate`, acceptance tracking, and KV snapshot storage.
2. **`VLMVisionStrategy`**: For requests containing images. Uses `vlm_stream_generate` via the pre-filled cache pattern. Intentionally separate because vision generation has fundamentally different semantics.

The `LanguageModelLogitsWrapper` (in `providers/common/model_wrappers.py`) adapts a VLM's language component to look like a standard `mlx-lm` model, enabling all text requests to use the full `mlx-lm` sampler pipeline.

### Embedding Provider Architecture (v1.25.0)

`MLXEmbeddingProvider` uses a different model class from `MLXProvider`:

- Model class: `EmbeddingModel` in `src/heylook_llm/models/embedding_model.py`
- Backbone loaded dynamically via `load_backbone()` using `mlx_lm.utils._get_classes()` -- supports any architecture, not just Gemma3
- Bidirectional attention via additive padding mask (0.0 for real tokens, -inf for padding)
- Mean pooling + two dense projection layers + L2 normalization
- Supports quantized variants (4-bit, 8-bit) via `nn.quantize()`

See [mlx_embedding.md](./mlx_embedding.md) for the full deep-dive.

### Server-Side Store (DuckDB)

Conversations, notebooks, messages (content-block lists), and presets
persist in a DuckDB store (`db.py`). The `Store` class serializes all
writes through a `max_workers=1` executor -- stronger than a lock, and
what lets code enforce uniqueness constraints (e.g. preset name
collisions) without a DB-level constraint.

Schema changes are gated by `_SCHEMA_VERSION`: on mismatch, every table
is dropped and recreated (fresh-start, no data migration -- acceptable
for a sole-user store). Adding a new table does NOT require a version
bump if it's purely additive (`CREATE TABLE IF NOT EXISTS`) -- bumping
the version for an additive change would needlessly drop every existing
table. The `presets` table (added v1.34.22, holds UI-authored
`{name, system_prompt, params}` bundles for the v3 frontend's preset
bar) was added this way. `clear_all_data()` (backing `POST
/v1/data/clear`) deletes conversations/messages/notebooks only -- presets
are deliberately excluded, since they're configuration a user built up,
not generated data. See [api.md](./api.md#preset-storage-v1presets) for
the endpoint surface.

### Operational Subsystems

- **Analytics DB pruning**: Cleanup thread prunes oldest 25% of records and runs VACUUM when DB exceeds `max_db_size_mb`
- **Startup logging**: Server startup logs analytics DB size (with limit) and log directory size for disk usage visibility
- **Radix cache memory pressure**: Optional `memory_pressure_fn` callback triggers eviction when GPU memory exceeds 85% of recommended working set

## Related Documentation

- [config.md](./config.md) -- configuration system and models.toml structure
- [router.md](./router.md) -- LRU cache and model routing details
- [api.md](./api.md) -- endpoint architecture
- [mlx_provider.md](./mlx_provider.md) -- MLXProvider deep-dive
- [mlx_embedding.md](./mlx_embedding.md) -- MLXEmbeddingProvider deep-dive
