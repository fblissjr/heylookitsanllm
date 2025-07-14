# src/heylook_llm/router.py
import yaml
import logging
import threading
import time
from typing import Optional
from collections import OrderedDict

from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider
from heylook_llm.providers.mlx_provider import MLXProvider
from heylook_llm.providers.llama_cpp_provider import LlamaCppProvider

class ModelRouter:
    """Manages loading, unloading, and routing to different model providers with an LRU cache."""
    def __init__(self, config_path: str, log_level: int, initial_model_id: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.app_config = AppConfig(**yaml.safe_load(f))

        self.providers = OrderedDict()
        self.max_loaded_models = self.app_config.max_loaded_models
        logging.info(f"Router configured to keep up to {self.max_loaded_models} models in memory.")

        self.log_level = log_level
        self.loading_lock = threading.Lock()

        initial_model_to_load = initial_model_id or self.app_config.default_model
        enabled_models = self.app_config.get_enabled_models()
        if not enabled_models:
            logging.error("No enabled models found in models.yaml. Server cannot serve requests.")
            return

        if initial_model_to_load and not self.app_config.get_model_config(initial_model_to_load):
            logging.warning(f"Initial model '{initial_model_to_load}' not found or disabled. Loading first available model.")
            initial_model_to_load = enabled_models[0].id

        if initial_model_to_load:
            try:
                logging.info(f"Pre-warming initial model: {initial_model_to_load}")
                self.get_provider(initial_model_to_load)
                logging.info(f"Successfully pre-warmed model: {initial_model_to_load}")
            except Exception as e:
                logging.error(f"Failed to pre-warm initial model '{initial_model_to_load}': {e}")
                logging.warning(f"Continuing without pre-warming. Model '{initial_model_to_load}' will be loaded on first request.")
                # Don't crash - just continue without pre-warming

    def get_provider(self, model_id: str) -> BaseProvider:
        if not model_id:
            raise ValueError("model_id cannot be empty.")

        with self.loading_lock:
            if self.log_level <= logging.DEBUG:
                logging.debug(f"Router cache state before access: {list(self.providers.keys())}")

            if model_id in self.providers:
                logging.debug(f"Cache hit for model: {model_id}. Reusing existing provider.")
                self.providers.move_to_end(model_id)
                return self.providers[model_id]

            # Model loading progress logging
            load_start_time = time.time()

            if len(self.providers) >= self.max_loaded_models:
                lru_id, lru_provider = self.providers.popitem(last=False)
                logging.info(f"Cache full. Evicting model: {lru_id} to make space for {model_id}.")
                lru_provider.unload()
                del lru_provider

            model_config = self.app_config.get_model_config(model_id)
            if not model_config:
                available = [m.id for m in self.app_config.get_enabled_models()]
                raise ValueError(f"Model '{model_id}' not found or disabled. Available: {available}")

            provider_map = {"mlx": MLXProvider, "llama_cpp": LlamaCppProvider}
            provider_class = provider_map.get(model_config.provider)
            if not provider_class:
                raise ValueError(f"Unknown provider: {model_config.provider}")

            logging.info(f"Loading model '{model_id}' with provider '{model_config.provider}'...")

            # Enhanced loading progress
            model_path = model_config.config.model_path if hasattr(model_config.config, 'model_path') else 'unknown'
            logging.info(f"Model path: {model_path}")

            try:
                new_provider = provider_class(
                    model_config.id,
                    model_config.config.model_dump(),
                    self.log_level <= logging.DEBUG
                )

                logging.info(f"Initializing {model_config.provider.upper()} provider...")
                new_provider.load_model()

                load_time = time.time() - load_start_time
                self.providers[model_id] = new_provider

                logging.info(f"Successfully loaded model: {model_id} in {load_time:.2f}s")

                # Log cache state after loading
                if self.log_level <= logging.DEBUG:
                    logging.debug(f"Router cache state after loading: {list(self.providers.keys())}")

                return new_provider

            except Exception as e:
                load_time = time.time() - load_start_time
                logging.error(f"Failed to load model '{model_id}' after {load_time:.2f}s: {e}")
                raise e

    def list_available_models(self) -> list[str]:
        return [m.id for m in self.app_config.get_enabled_models()]
