# src/heylook_llm/router.py
import yaml
import logging
import threading
import time
import gc
from typing import Optional, Dict
from collections import OrderedDict

from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider

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

# Try to import llama.cpp provider
try:
    from heylook_llm.providers.llama_cpp_provider import LlamaCppProvider
    HAS_LLAMA_CPP = True
except ImportError:
    LlamaCppProvider = None
    HAS_LLAMA_CPP = False


class ModelRouter:
    """Manages loading, unloading, and routing to different model providers with an LRU cache."""
    def __init__(self, config_path: str, log_level: int, initial_model_id: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.app_config = AppConfig(**yaml.safe_load(f))

        self.providers = OrderedDict()
        self.max_loaded_models = self.app_config.max_loaded_models
        logging.info(f"Router configured to keep up to {self.max_loaded_models} models in memory.")

        # Log available providers
        if HAS_MLX:
            logging.debug("MLX provider is available")
        else:
            if 'MLX_IMPORT_ERROR' in globals() and 'mlx_vlm' in MLX_IMPORT_ERROR:
                logging.warning("MLX provider not available: mlx-vlm not installed. This should have been installed with uv pip install heylookitsanllm[mlx]")
            else:
                logging.debug("MLX provider not available. Install with: uv pip install heylookitsanllm[mlx]")

        if HAS_LLAMA_CPP:
            logging.debug("Llama.cpp provider is available")
        else:
            logging.debug("Llama.cpp provider not available. Install with: uv pip install heylookitsanllm[llama-cpp]")

        self.log_level = log_level

        # Fine-grained locking: separate locks for cache access and model loading
        self.cache_lock = threading.RLock()  # For quick cache operations
        self.loading_locks: Dict[str, threading.Lock] = {}  # Per-model loading locks
        self.loading_locks_lock = threading.Lock()  # Protect loading_locks dict

        initial_model_to_load = initial_model_id or self.app_config.default_model
        enabled_models = self.app_config.get_enabled_models()
        if not enabled_models:
            logging.error("No enabled models found in models.yaml. Server cannot serve requests.")
            return

        # Check if the initial model is available and its provider is installed
        if initial_model_to_load:
            model_config = self.app_config.get_model_config(initial_model_to_load)
            if not model_config:
                logging.warning(f"Initial model '{initial_model_to_load}' not found or disabled.")
                initial_model_to_load = None
            elif model_config.provider == "mlx" and not HAS_MLX:
                logging.warning(f"Initial model '{initial_model_to_load}' requires MLX provider which is not installed.")
                initial_model_to_load = None
            elif model_config.provider == "llama_cpp" and not HAS_LLAMA_CPP:
                logging.warning(f"Initial model '{initial_model_to_load}' requires llama.cpp provider which is not installed.")
                initial_model_to_load = None

        # If no valid initial model, find first compatible one
        if not initial_model_to_load:
            for model in enabled_models:
                if model.provider == "mlx" and HAS_MLX:
                    initial_model_to_load = model.id
                    logging.info(f"Selected MLX model '{initial_model_to_load}' as initial model.")
                    break
                elif model.provider == "llama_cpp" and HAS_LLAMA_CPP:
                    initial_model_to_load = model.id
                    logging.info(f"Selected llama.cpp model '{initial_model_to_load}' as initial model.")
                    break

            if not initial_model_to_load:
                logging.warning("No compatible models found for installed providers. Models will be loaded on first request.")

        if initial_model_to_load:
            try:
                logging.info(f"Pre-warming initial model: {initial_model_to_load}")
                self.get_provider(initial_model_to_load)
                logging.info(f"Successfully pre-warmed model: {initial_model_to_load}")
            except Exception as e:
                logging.error(f"Failed to pre-warm initial model '{initial_model_to_load}': {e}")
                logging.warning(f"Continuing without pre-warming. Model '{initial_model_to_load}' will be loaded on first request.")

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
                return self.providers[model_id]
            return None

    def _evict_lru_model(self):
        """Evict least recently used model. Must be called with cache_lock held."""
        lru_id, lru_provider = self.providers.popitem(last=False)
        logging.info(f"Cache full. Evicting model: {lru_id}")

        # Check if it's an MLX model before unloading
        is_mlx_model = False
        if hasattr(lru_provider, 'provider'):
            is_mlx_model = lru_provider.provider == 'mlx'

        # Release cache lock during unloading
        self.cache_lock.release()
        try:
            lru_provider.unload()
            del lru_provider

            # Force garbage collection to free memory immediately
            gc.collect()

            # Clear MLX cache if available
            if HAS_MLX and is_mlx_model:
                mx.clear_cache()

        finally:
            self.cache_lock.acquire()

    def get_provider(self, model_id: str) -> BaseProvider:
        if not model_id:
            raise ValueError("model_id cannot be empty.")

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
            if LlamaCppProvider:
                provider_map["llama_cpp"] = LlamaCppProvider

            provider_class = provider_map.get(model_config.provider)
            if not provider_class:
                if model_config.provider == "mlx" and not HAS_MLX:
                    raise ValueError(f"MLX provider requested but not installed. Install with: pip install heylookitsanllm[mlx]")
                elif model_config.provider == "llama_cpp" and not HAS_LLAMA_CPP:
                    raise ValueError(f"Llama.cpp provider requested but not installed. Install with: pip install heylookitsanllm[llama-cpp]")
                else:
                    raise ValueError(f"Unknown provider: {model_config.provider}")

            logging.info(f"Loading model '{model_id}' with provider '{model_config.provider}'...")

            # Show loading progress
            model_path = model_config.config.model_path if hasattr(model_config.config, 'model_path') else 'unknown'
            logging.info(f"Model path: {model_path}")

            try:
                # Create provider instance
                new_provider = provider_class(
                    model_config.id,
                    model_config.config.model_dump(),
                    self.log_level <= logging.DEBUG
                )

                logging.info(f"Initializing {model_config.provider.upper()} provider...")

                # Load model (this is the expensive operation)
                new_provider.load_model()

                # Add to cache with cache lock
                with self.cache_lock:
                    # Check if we need to evict
                    if len(self.providers) >= self.max_loaded_models:
                        self._evict_lru_model()

                    # Add new provider to cache
                    self.providers[model_id] = new_provider

                load_time = time.time() - load_start_time
                logging.info(f"Successfully loaded model: {model_id} in {load_time:.2f}s")

                # Log cache state after loading
                if self.log_level <= logging.DEBUG:
                    with self.cache_lock:
                        logging.debug(f"Router cache state after loading: {list(self.providers.keys())}")

                return new_provider

            except Exception as e:
                load_time = time.time() - load_start_time
                logging.error(f"Failed to load model '{model_id}' after {load_time:.2f}s: {e}")
                raise e

    def list_available_models(self) -> list[str]:
        return [m.id for m in self.app_config.get_enabled_models()]
