# Router System -- Model Caching and Routing

Last updated: 2026-07-06

This document explains the model routing system, LRU cache implementation, thread safety mechanisms, and how models are loaded and managed.

## What changed 2026-07-05/06

No changes to `ModelRouter` itself this window. Reviewed for accuracy
against two crash fixes and a defaults audit that landed in this window
(`internal/log/log_2026-07-05.md`, `log_2026-07-06.md`) -- nothing below
was found stale. The two crash fixes concern a *different* concurrency
layer than the one this document covers: `ModelRouter`'s locks (below)
serialize *loading/evicting providers*; they say nothing about
serializing *generation* on an already-loaded model. See
[mlx_provider.md](./mlx_provider.md) section 4 for that layer (the FIFO
generation gate and the pinned executor pool) and
[../bugs/mlx_thread_teardown_abort.md](../../internal/bugs/mlx_thread_teardown_abort.md) /
[../bugs/radix_thread_affinity.md](../../internal/bugs/radix_thread_affinity.md) for
the postmortems.

## Overview

The `ModelRouter` class is the central orchestrator for all model operations:
- Loads models on-demand from `models.toml`
- Caches models in memory with configurable `max_loaded_models` (LRU eviction)
- Routes requests to appropriate providers
- Ensures thread safety with fine-grained locking
- Supports hot-reloading of configuration

**File**: `src/heylook_llm/router.py`

## Provider Map

The router maintains a map from provider type string to provider class:

```python
provider_map = {
    "mlx": MLXProvider,
    "mlx_embedding": MLXEmbeddingProvider,
}
```

Provider type is `Literal["mlx", "mlx_embedding"]` in `config.py`. Both providers are imported at module level with graceful fallback if unavailable.

## Core Architecture

### ModelRouter Class

```python
class ModelRouter:
    def __init__(self, config_path: str = "models.toml"):
        self.config_path = config_path
        self.providers: OrderedDict[str, BaseProvider] = OrderedDict()
        self.max_loaded_models: int  # from models.toml, default 2

        # Thread safety
        self.cache_lock = RLock()                    # Protects cache OrderedDict
        self.loading_locks: Dict[str, Lock] = {}     # One lock per model ID
```

## Model Loading Flow

```
Client Request
  ↓
router.get_provider(model_id)
  ↓
Check if model_id in providers (cache_lock)
  ↓
Cache Hit? → Return cached provider (move to end)
Cache Miss?
  ↓
  Acquire model-specific loading lock
  ↓
  Double-check cache (another thread may have loaded it)
  ↓
  Validate model_id is in app_config
  ↓
  Evict LRU if cache full (call provider.unload(), gc.collect(), mx.clear_cache())
  ↓
  Instantiate provider: provider_class(model_id, config.model_dump())
  ↓
  provider.load_model() [expensive: 2-30s]
  ↓
  Add to cache (cache_lock)
  ↓
  Release loading lock
  ↓
  Return provider
```

### Double-Check Locking Pattern

Prevents duplicate loading when multiple threads request the same model:

```python
def get_provider(self, model_id: str) -> BaseProvider:
    # Check 1: Before acquiring lock (fast path)
    with self.cache_lock:
        if model_id in self.providers:
            self.providers.move_to_end(model_id)
            return self.providers[model_id]

    # Acquire model-specific lock
    loading_lock = self._get_or_create_loading_lock(model_id)
    with loading_lock:
        # Check 2: After acquiring lock (another thread might have loaded it)
        with self.cache_lock:
            if model_id in self.providers:
                self.providers.move_to_end(model_id)
                return self.providers[model_id]

        # Load model
        ...
```

## LRU Cache Mechanics

Python's `OrderedDict` maintains insertion order:
- New or recently accessed models are moved to the end (most recently used)
- Eviction always removes from the front (least recently used)

### Cache Size

Default is `max_loaded_models = 2` from `models.toml`. Recommendations by RAM:
- 8GB: 1
- 16GB: 2 (default)
- 32GB: 3-4
- 64GB+: 4-6

### Eviction

```python
def _evict_lru_model(self):
    lru_model_id, lru_provider = next(iter(self.providers.items()))
    del self.providers[lru_model_id]
    lru_provider.unload()
    gc.collect()
    mx.clear_cache()
```

Pinned models (see Model Pinning below) are skipped during eviction.

## Thread Safety

**Two-level locking:**

1. `cache_lock` (RLock, short-lived): Protects the `providers` OrderedDict for reads and writes
2. `loading_locks[model_id]` (Lock, long-lived during load): Prevents parallel loading of the same model

Different models can load in parallel (they each acquire their own `loading_locks` entry). The same model serializes duplicate load attempts.

**Lock hierarchy** (prevents deadlocks):
- Always acquire `cache_lock` first
- Then acquire `loading_locks[model_id]`
- Release in reverse order

## Model Pinning

The router supports pinning models to prevent LRU eviction. Used for long-running batch jobs that cannot afford model eviction mid-run.

```python
router.pin_model(model_id)    # Load and mark as unpinnable
router.unpin_model(model_id)  # Allow eviction again
```

Pinned models are skipped in `_evict_lru_model()`. Non-pinned LRU models are evicted first.

## Router Methods

### `get_provider(model_id)`

Core method. Loads model or retrieves from cache. Raises `ValueError` if model not in config. Raises `HTTPException(404)` if provider type unknown.

### `embed(model_id, texts)`

Convenience wrapper:
```python
def embed(self, model_id: str, texts: List[str]) -> List[List[float]]:
    provider = self.get_provider(model_id)
    return provider.embed(texts)
```

### `reload_config()`

Hot-reload `models.toml`. Cache stays intact -- loaded models do not reload.

### `clear_cache()`

Unload all models and clear the cache. Used during shutdown.

## Configuration Loading

`_load_config()` reads `models.toml` at startup and on `reload_config()`. Models are NOT loaded at this point -- only configuration is parsed into `AppConfig`. Models load on first request.

## Performance Characteristics

### Load Times

| Model Size | Quantization | Load Time | Memory |
|---|---|---|---|
| 1-3B | 4-bit | 2-5s | 2-3GB |
| 7-9B | 4-bit | 5-10s | 4-6GB |
| 14B | 4-bit | 10-15s | 8-10GB |
| 30B+ | 4-bit | 20-30s | 15-20GB |

After first load, cache hits are effectively instant.

### Throughput

- Single model, cached: near-zero routing overhead
- Model switch: full load time of new model
- Two models alternating: ~95% cache hit rate
- Three+ models rotating: ~50% hit rate (frequent evictions)

## Error Handling

**Model not in config:**
```python
raise ValueError(f"Model '{model_id}' not found in configuration")
```

**Unknown provider type:**
```python
raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_type}")
```

**Load failure:**
```python
try:
    provider.load_model()
except Exception as e:
    logger.error(f"Failed to load model {model_id}: {e}")
    raise
```

## Related Documentation

- [api.md](./api.md) -- API endpoints and flow
- [config.md](./config.md) -- models.toml structure
- [mlx_provider.md](./mlx_provider.md) -- MLXProvider
- [mlx_embedding.md](./mlx_embedding.md) -- MLXEmbeddingProvider
- [overview.md](./overview.md) -- backend architecture overview
