# router.py
import yaml
import logging
import threading
from edge_llm.config import AppConfig
from edge_llm.providers.base import BaseProvider

from edge_llm.providers import mlx_unified
from edge_llm.providers.llama_cpp_provider import LlamaCppProvider

def _load_provider(cfg):
    backend = mlx_unified.load(cfg)    # returns LmBackend or VlmBackend
    return backend.load(cfg["model_path"], adapter=cfg.get("adapter_path"), **cfg)

class ModelRouter:
    """Manages loading, unloading, and routing to different model providers."""
    def __init__(self, config_path: str, log_level: int):
        with open(config_path, 'r') as f: self.app_config = AppConfig(**yaml.safe_load(f))
        self.providers = {}; self.current_provider_id = None
        self.log_level = log_level; self.loading_lock = threading.Lock()
        if self.app_config.models: self.get_provider(self.app_config.models[0].id)

    def get_provider(self, model_id: str) -> BaseProvider:
        with self.loading_lock:
            if self.current_provider_id == model_id: return self.providers[model_id]
            if self.current_provider_id: self.providers.pop(self.current_provider_id, None)

            model_config = next((m.model_dump() for m in self.app_config.models if m.id == model_id), None)
            if not model_config: raise ValueError(f"Model '{model_id}' not found.")
            provider_map = {"mlx": MLXProvider, "llama_cpp": LlamaCppProvider}
            provider_class = provider_map.get(model_config['provider'])
            if not provider_class: raise ValueError(f"Unknown provider: {model_config['provider']}")

            self.providers[model_id] = provider_class(model_config['id'], model_config['config'], self.log_level <= logging.DEBUG)
            self.current_provider_id = model_id
            return self.providers[model_id]
