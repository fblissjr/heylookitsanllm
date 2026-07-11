# src/heylook_llm/router.py
import tomllib
import logging
import threading
import time
import gc
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from pathlib import Path

from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider
from heylook_llm.diagnostic_logger import diag_event
from heylook_llm import observability

# Try to import MLX provider
try:
    from heylook_llm.providers.mlx_provider import MLXProvider
    import mlx.core as mx
    HAS_MLX = True
except ImportError as e:
    MLXProvider = None
    HAS_MLX = False
    # Store the error for later logging
    MLX_IMPORT_ERROR = str(e)

# Try to import MLX Embedding provider
try:
    from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider
    HAS_MLX_EMBEDDING = True
except ImportError as e:
    MLXEmbeddingProvider = None
    HAS_MLX_EMBEDDING = False


class ModelRouter:
    """Manages loading, unloading, and routing to different model providers with an LRU cache."""
    def __init__(self, config_path: str, log_level: int, initial_model_id: Optional[str] = None):
        # Store config path for reload
        self.config_path = config_path

        # Load config (TOML only)
        config_data = self._load_config(config_path)
        self.app_config = AppConfig(**config_data)

        self.providers = OrderedDict()
        self.max_loaded_models = self.app_config.max_loaded_models
        logging.info(f"Router configured to keep up to {self.max_loaded_models} models in memory.")

        # Log available providers
        if HAS_MLX:
            logging.debug("MLX provider is available")
        else:
            if 'MLX_IMPORT_ERROR' in globals() and 'mlx_vlm' in MLX_IMPORT_ERROR:
                logging.warning("MLX provider not available: mlx-vlm not installed. Run: uv sync --extra mlx")
            else:
                logging.debug("MLX provider not available. Install with: uv sync --extra mlx")

        self.log_level = log_level

        # Fine-grained locking: separate locks for cache access and model loading
        self.cache_lock = threading.RLock()  # For quick cache operations
        self.loading_locks: Dict[str, threading.Lock] = {}  # Per-model loading locks
        self.loading_locks_lock = threading.Lock()  # Protect loading_locks dict

        # Model pinning: prevents eviction during long-running batch jobs
        self._pinned: set[str] = set()

        # Capacity reservations for in-flight loads. The capacity check and
        # the multi-hundred-ms load can't share one lock hold, so without a
        # reservation two concurrent different-model loads both pass the
        # check and hold two full models in memory (check-then-act TOCTOU,
        # OOM-class on boxes sized for max_loaded_models). A side-set (like
        # _pinned) rather than a sentinel inside self.providers, so
        # self.providers always means "real, loaded providers" and reader
        # APIs need no filtering discipline.
        self._loading: set[str] = set()
        # Ceiling on how long a loader waits for another thread's in-flight
        # load to free capacity. Must exceed the slowest legitimate load
        # (100GB+ giants take minutes); its job is to turn a WEDGED load
        # into a loud error instead of silently blocking admission of every
        # other model forever (each blocked get_provider also pins an
        # asyncio default-executor thread).
        self._reservation_wait_timeout: float = 600.0

        # Idle-unload tracking (C2). time.time() of the last cache hit or load
        # per model_id. Consulted by ``unload_idle_models`` against each model's
        # effective threshold. Kept separate from self.providers' OrderedDict
        # position so unload decisions read from an explicit signal, not LRU
        # ordering.
        self._last_used_ts: Dict[str, float] = {}

        # Observability (S1.2). Set by api.py lifespan after construction.
        self.memory_manager: Optional[Any] = None

        initial_model_to_load = initial_model_id or self.app_config.default_model or None
        enabled_models = self.app_config.get_enabled_models()
        if not enabled_models:
            logging.error("No enabled models found in models.toml. Server cannot serve requests.")
            return

        # Validate the requested initial model
        if initial_model_to_load:
            model_config = self.app_config.get_model_config(initial_model_to_load)
            if not model_config:
                logging.warning(f"Initial model '{initial_model_to_load}' not found or disabled.")
                initial_model_to_load = None
            elif model_config.provider == "mlx" and not HAS_MLX:
                logging.warning(f"Initial model '{initial_model_to_load}' requires MLX provider which is not installed.")
                initial_model_to_load = None

        if not initial_model_to_load:
            logging.info("No default model configured. Models will be loaded on first request.")

        if initial_model_to_load:
            try:
                logging.info(f"Pre-warming initial model: {initial_model_to_load}")
                self.get_provider(initial_model_to_load)
                logging.info(f"Successfully pre-warmed model: {initial_model_to_load}")
            except Exception as e:
                logging.error(f"Failed to pre-warm initial model '{initial_model_to_load}': {e}")
                logging.warning(f"Continuing without pre-warming. Model '{initial_model_to_load}' will be loaded on first request.")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from TOML file."""
        config_file = Path(config_path)

        # If user specified exact file with extension
        if config_file.suffix == '.toml':
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_file, 'rb') as f:
                return tomllib.load(f)

        # If no extension, add .toml
        toml_path = config_file.with_suffix('.toml')
        if toml_path.exists():
            with open(toml_path, 'rb') as f:
                return tomllib.load(f)

        raise FileNotFoundError(
            f"Config file not found: {toml_path}. "
            f"Run 'heylookllm import' to create configuration."
        )

    def _get_or_create_loading_lock(self, model_id: str) -> threading.Lock:
        """Get or create a loading lock for a specific model."""
        with self.loading_locks_lock:
            if model_id not in self.loading_locks:
                self.loading_locks[model_id] = threading.Lock()
            return self.loading_locks[model_id]

    def _check_cache(self, model_id: str) -> Optional[BaseProvider]:
        """Check if model is in cache. Uses fine-grained locking."""
        with self.cache_lock:
            if model_id in self.providers:
                if self.log_level <= logging.DEBUG:
                    logging.debug(f"Cache hit for model: {model_id}. Reusing existing provider.")
                self.providers.move_to_end(model_id)
                self._last_used_ts[model_id] = time.time()
                return self.providers[model_id]
            return None

    def _teardown_provider(self, provider: BaseProvider) -> None:
        """Run a provider's unload + GC + Metal-cache clear sequence.

        Must be called without ``cache_lock`` held -- unloading MLX weights
        can take hundreds of ms and other threads need to continue cache
        reads. Shared between LRU eviction and idle unload so both paths
        stay in lockstep if teardown gains new steps (e.g. cache persistence
        in S3.1).
        """
        is_mlx_model = getattr(provider, "provider", "") == "mlx"
        try:
            provider.unload()
        except Exception:
            logging.error("Provider unload failed", exc_info=True)
        del provider
        gc.collect()
        if HAS_MLX and is_mlx_model:
            mx.clear_cache()

    def _evict_lru_model(self):
        """Evict least recently used non-pinned model. Must be called with cache_lock held."""
        evict_id = None
        for model_id in self.providers:
            if model_id not in self._pinned:
                evict_id = model_id
                break

        if evict_id is None:
            raise RuntimeError(
                f"All {len(self.providers)} loaded models are pinned. "
                f"Cannot evict to make room. Pinned: {self._pinned}"
            )

        lru_provider = self.providers.pop(evict_id)
        self._last_used_ts.pop(evict_id, None)
        logging.info(f"Cache full. Evicting model: {evict_id}")
        diag_event("model_evict", model=evict_id)
        observability.record_event("model_unload", tier="events", min_level="minimal",
                                   fields={"model": evict_id, "reason": "lru_evict"})
        from heylook_llm.memory import safe_mm_call
        safe_mm_call(self.memory_manager, "register_model_unload", evict_id, reason="lru_evict")

        self.cache_lock.release()
        try:
            self._teardown_provider(lru_provider)
        finally:
            self.cache_lock.acquire()

    def get_current_model_id(self) -> Optional[str]:
        """Get the most recently used model ID from cache, or None if no models loaded."""
        if self.providers:
            # OrderedDict keeps insertion order; last item is most recently used
            return next(reversed(self.providers))
        return None

    def get_loaded_models(self) -> Dict[str, BaseProvider]:
        """
        Get all currently loaded models.

        Returns:
            Dict mapping model_id to BaseProvider instance.
        """
        with self.cache_lock:
            # Return a copy to prevent external modification
            return dict(self.providers)

    def get_provider(self, model_id: str) -> BaseProvider:
        # Fallback logic when no model specified:
        # 1. Use currently loaded model (most recently used)
        # 2. Use default_model from config
        # 3. Raise error with available models
        if not model_id:
            model_id = self.get_current_model_id()
            if model_id:
                logging.debug(f"No model specified, using loaded model: {model_id}")
            elif self.app_config.default_model:
                model_id = self.app_config.default_model
                logging.debug(f"No model specified, using default: {model_id}")
            else:
                available = [m.id for m in self.app_config.get_enabled_models()]
                raise ValueError(f"No model specified and no default configured. Available: {available}")

        # Fast path: check cache first
        provider = self._check_cache(model_id)
        if provider:
            return provider

        # Get model-specific loading lock
        loading_lock = self._get_or_create_loading_lock(model_id)

        # Acquire loading lock for this specific model
        with loading_lock:
            # Double-check cache after acquiring lock (another thread might have loaded it)
            provider = self._check_cache(model_id)
            if provider:
                return provider

            # Model needs to be loaded
            load_start_time = time.time()

            # Get model config
            model_config = self.app_config.get_model_config(model_id)
            if not model_config:
                available = [m.id for m in self.app_config.get_enabled_models()]
                raise ValueError(f"Model '{model_id}' not found or disabled. Available: {available}")

            provider_map = {}
            if MLXProvider:
                provider_map["mlx"] = MLXProvider
            if MLXEmbeddingProvider:
                provider_map["mlx_embedding"] = MLXEmbeddingProvider

            provider_class = provider_map.get(model_config.provider)
            if not provider_class:
                if model_config.provider == "mlx" and not HAS_MLX:
                    raise ValueError(f"MLX provider requested but not installed. Run: uv sync --extra mlx")
                else:
                    raise ValueError(f"Unknown provider: {model_config.provider}")

            logging.info(f"Loading model '{model_id}' with provider '{model_config.provider}'...")

            # Show loading progress
            model_path = model_config.config.model_path if hasattr(model_config.config, 'model_path') else 'unknown'
            logging.info(f"Model path: {model_path}")

            try:
                # Reserve capacity BEFORE loading (evicting if needed) so the
                # capacity check and the load are one atomic commitment.
                # Without a reservation, two concurrent different-model loads
                # both pass the check and hold two full models in memory
                # (check-then-act TOCTOU). If capacity is held by other
                # threads' in-flight loads, wait (bounded) for one to publish.
                reservation_wait_start = time.time()
                while True:
                    with self.cache_lock:
                        if len(self.providers) + len(self._loading) < self.max_loaded_models:
                            self._loading.add(model_id)
                            break
                        if any(mid not in self._pinned for mid in self.providers):
                            self._evict_lru_model()
                            continue
                        if not self._loading:
                            raise RuntimeError(
                                f"All {len(self.providers)} loaded models are pinned. "
                                f"Cannot evict to make room. Pinned: {self._pinned}"
                            )
                        inflight = sorted(self._loading)
                    if time.time() - reservation_wait_start > self._reservation_wait_timeout:
                        raise RuntimeError(
                            f"Timed out after {self._reservation_wait_timeout:.0f}s waiting "
                            f"for model-load capacity to free (in-flight loads: {inflight}). "
                            f"A load may be wedged."
                        )
                    time.sleep(0.05)

                # Create provider instance
                new_provider = provider_class(
                    model_config.id,
                    model_config.config.model_dump(),
                    self.log_level <= logging.DEBUG
                )

                logging.info(f"Initializing {model_config.provider.upper()} provider...")

                # Load model (this is the expensive operation)
                new_provider.load_model()

                # Prime JIT caches before publishing the provider so concurrent
                # cache hits don't race a half-warmed model. `warmup()`'s contract
                # (BaseProvider docstring) requires it to swallow exceptions; no
                # wrapper needed here.
                new_provider.warmup()

                # Publish: the reservation becomes the real provider.
                with self.cache_lock:
                    self._loading.discard(model_id)
                    self.providers[model_id] = new_provider
                    self._last_used_ts[model_id] = time.time()

                load_time = time.time() - load_start_time
                logging.info(f"Successfully loaded model: {model_id} in {load_time:.2f}s")
                diag_event("model_load", model=model_id, provider=model_config.provider,
                           load_time_s=round(load_time, 2))
                observability.record_event("model_load", tier="events", min_level="minimal",
                                           fields={"model": model_id, "provider": model_config.provider,
                                                   "load_time_s": round(load_time, 2)})

                if self.memory_manager is not None:
                    from heylook_llm.memory import capture_model_metadata, safe_mm_call
                    try:
                        metadata = capture_model_metadata(
                            model_id,
                            new_provider,
                            getattr(model_config.config, "model_path", ""),
                        )
                    except Exception:
                        logging.debug("capture_model_metadata failed", exc_info=True)
                        metadata = None
                    if metadata is not None:
                        safe_mm_call(self.memory_manager, "register_model_load", metadata, load_time * 1000.0)

                # Log cache state after loading
                if self.log_level <= logging.DEBUG:
                    with self.cache_lock:
                        logging.debug(f"Router cache state after loading: {list(self.providers.keys())}")

                return new_provider

            except Exception as e:
                # Release the reservation so the failed load doesn't hold
                # capacity forever.
                with self.cache_lock:
                    self._loading.discard(model_id)
                load_time = time.time() - load_start_time
                logging.error(f"Failed to load model '{model_id}' after {load_time:.2f}s: {e}")
                raise e

    def list_available_models(self) -> list[str]:
        return [m.id for m in self.app_config.get_enabled_models()]

    def clear_cache(self):
        """Clear all loaded models from cache."""
        with self.cache_lock:
            # Unload all models
            for model_id in list(self.providers.keys()):
                provider = self.providers[model_id]
                try:
                    provider.unload()
                    logging.info(f"Unloaded model: {model_id}")
                except Exception as e:
                    logging.error(f"Error unloading model {model_id}: {e}")

            # Clear the cache (OrderedDict maintains order automatically)
            self.providers.clear()
            self._last_used_ts.clear()
            logging.info("Model cache cleared")
    
    def reload_config(self):
        """Reload model configuration from file."""
        try:
            # Reload the configuration from stored path
            config_data = self._load_config(self.config_path)
            self.app_config = AppConfig(**config_data)
            self.max_loaded_models = self.app_config.max_loaded_models
            logging.info(f"Model configuration reloaded from {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            raise

    def unload_model(self, model_id: str, force: bool = False) -> bool:
        """Explicitly unload a specific model from cache.

        Returns True if the model was loaded and is now unloaded, False if it wasn't loaded.
        Raises RuntimeError if model is pinned and force=False.
        """
        with self.cache_lock:
            if model_id in self._pinned and not force:
                raise RuntimeError(
                    f"Model '{model_id}' is pinned (batch job in progress). "
                    f"Use force=True to override."
                )
            if model_id not in self.providers:
                return False

            provider = self.providers.pop(model_id)

        # Unload outside the cache lock to avoid holding it during slow ops
        is_mlx = hasattr(provider, 'provider') and provider.provider == 'mlx'
        try:
            provider.unload()
            del provider
            gc.collect()
            if HAS_MLX and is_mlx:
                mx.clear_cache()
            logging.info(f"Explicitly unloaded model: {model_id}")
        except Exception as e:
            logging.error(f"Error unloading model {model_id}: {e}")

        return True

    def get_model_status(self, model_id: str) -> dict:
        """Get load status and basic metrics for a model."""
        with self.cache_lock:
            loaded = model_id in self.providers

        status = {"loaded": loaded}

        if loaded:
            provider = self.providers.get(model_id)
            if provider:
                # Try to get memory info
                if hasattr(provider, 'get_memory_usage'):
                    try:
                        status["memory_mb"] = provider.get_memory_usage()
                    except Exception:
                        pass

        return status

    def _effective_idle_threshold(self, model_id: str) -> int:
        """Per-model override beats global default. ``0`` at either level means
        "disabled"; per-model non-zero override wins over a ``0`` global.
        Returns ``0`` when no idle-unload should happen for this model.
        """
        model_config = self.app_config.get_model_config(model_id)
        per_model = None
        if model_config is not None:
            per_model = getattr(model_config.config, "unload_after_idle_seconds", None)
        if per_model is not None:
            return int(per_model)
        return int(getattr(self.app_config, "idle_unload_seconds", 0))

    def unload_idle_models(self, now_ts: Optional[float] = None) -> List[str]:
        """Unload non-pinned models whose idle window has elapsed.

        Driven by ``MemoryManager.tick()`` on the 60s resource-snapshot loop.
        Pinned models are exempt. Models with an effective threshold of ``0``
        (explicit per-model disable, or global disable with no per-model
        override) are never touched.

        ``now_ts`` defaults to ``time.time()``; tests inject a fake clock.
        Returns the list of ``model_id`` values that were unloaded.
        """
        if now_ts is None:
            now_ts = time.time()

        with self.cache_lock:
            candidates = []
            for model_id in list(self.providers.keys()):
                if model_id in self._pinned:
                    continue
                threshold = self._effective_idle_threshold(model_id)
                if threshold <= 0:
                    continue
                last_used = self._last_used_ts.get(model_id, now_ts)
                if now_ts - last_used > threshold:
                    candidates.append(model_id)

        unloaded = []
        for model_id in candidates:
            if self._unload_idle(model_id):
                unloaded.append(model_id)
        return unloaded

    def _unload_idle(self, model_id: str) -> bool:
        """Pop + tear down a single idle model. Unload runs outside the
        cache lock; weight release can take hundreds of ms and would stall
        concurrent cache reads otherwise.

        The busy check and the pop happen under ONE cache_lock hold: a
        request WAITING at the FIFO generation gate is neither 'active' nor
        recently-used (last_used was stamped at its cache hit, and gate
        waits can outlast the idle threshold) -- unloading then would tear
        the weights down under a request that's about to run. Any cache hit
        after this pop simply reloads.
        """
        with self.cache_lock:
            provider = self.providers.get(model_id)
            if provider is None:
                return False
            stats = provider.generation_queue_stats()
            if stats and (stats.get("active", 0) > 0 or stats.get("waiting", 0) > 0):
                logging.info(
                    f"Skipping idle unload of {model_id}: generation queue busy "
                    f"(active={stats.get('active', 0)}, waiting={stats.get('waiting', 0)})"
                )
                return False
            self.providers.pop(model_id, None)
            self._last_used_ts.pop(model_id, None)

        logging.info(f"Idle timeout. Unloading model: {model_id}")
        diag_event("model_idle_unload", model=model_id)
        observability.record_event("model_unload", tier="events", min_level="minimal",
                                   fields={"model": model_id, "reason": "idle_timeout"})
        from heylook_llm.memory import safe_mm_call
        safe_mm_call(self.memory_manager, "register_model_unload", model_id, reason="idle_timeout")
        self._teardown_provider(provider)
        return True

    def pin_model(self, model_id: str) -> None:
        """Pin a model to prevent LRU eviction during long-running batch jobs.

        The model must already be loaded. Pinned models are skipped during
        eviction in _evict_lru_model(), so a batch job won't lose its model
        when another request triggers a load.
        """
        with self.cache_lock:
            if model_id not in self.providers:
                raise ValueError(f"Cannot pin model '{model_id}': not currently loaded")
            self._pinned.add(model_id)
            logging.info(f"Pinned model: {model_id} (pinned: {self._pinned})")

    def unpin_model(self, model_id: str) -> None:
        """Remove pin from a model, allowing normal LRU eviction again."""
        with self.cache_lock:
            self._pinned.discard(model_id)
            logging.info(f"Unpinned model: {model_id} (pinned: {self._pinned})")

    def reload_single_model(self, model_id: str) -> None:
        """Reload config for one model without clearing entire cache.

        If the model is currently loaded, it gets unloaded and the new config
        is stored. It does NOT auto-reload -- the caller should use get_provider()
        when they want it loaded again.
        """
        # Reload config from file first
        self.reload_config()

        # Unload if currently loaded (will be re-loaded with new config on next request)
        self.unload_model(model_id)
