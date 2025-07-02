# src/edge_llm/router.py

import yaml
import logging
import threading
from edge_llm.config import AppConfig
from edge_llm.providers.base import BaseProvider
from edge_llm.providers.mlx_provider import MLXProvider
from edge_llm.providers.llama_cpp_provider import LlamaCppProvider

class ModelRouter:
    """Manages loading, unloading, and routing to different model providers."""
    def __init__(self, config_path: str, log_level: int):
        with open(config_path, 'r') as f: self.app_config = AppConfig(**yaml.safe_load(f))
        self.providers = {}
        self.current_provider_id = None
        self.log_level = log_level
        self.loading_lock = threading.Lock()
        if self.app_config.models:
            self.get_provider(self.app_config.models[0].id)

    def get_provider(self, model_id: str) -> BaseProvider:
        with self.loading_lock:
            if self.current_provider_id == model_id:
                return self.providers.get(model_id)

            if self.current_provider_id:
                self.providers.pop(self.current_provider_id, None)

            model_config_dict = next((m.model_dump() for m in self.app_config.models if m.id == model_id), None)
            if not model_config_dict:
                raise ValueError(f"Model '{model_id}' not found in models.yaml.")

            provider_map = {
                "mlx": MLXProvider,
                "llama_cpp": LlamaCppProvider
            }

            provider_class = provider_map.get(model_config_dict['provider'])
            if not provider_class:
                raise ValueError(f"Unknown provider: {model_config_dict['provider']}")

            self.providers[model_id] = provider_class(
                model_config_dict['id'],
                model_config_dict['config'],
                self.log_level <= logging.DEBUG
            )

            self.current_provider_id = model_id
            return self.providers[model_id]
