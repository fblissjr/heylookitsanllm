# src/router.py
import yaml, logging, threading
from .config import AppConfig
from .providers.base import BaseProvider
from .providers.mlx_provider import MLXProvider
from .providers.llama_cpp_provider import LlamaCppProvider

class ModelRouter:
    """Manages loading, unloading, and routing to different model providers."""
    def __init__(self, config_path: str, log_level: int):
        with open(config_path, 'r') as f:
            self.app_config = AppConfig(**yaml.safe_load(f))

        self.providers: Dict[str, BaseProvider] = {}
        self.current_provider_id: Optional[str] = None
        self.log_level = log_level
        self.loading_lock = threading.Lock()

        # Pre-warm the first model in the list if available
        if self.app_config.models:
            self.get_provider(self.app_config.models[0].id)

    def list_models(self):
        return [model.id for model in self.app_config.models]

    def get_provider(self, model_id: str) -> BaseProvider:
        with self.loading_lock: # Why: Prevents race conditions if multiple requests try to switch models at once.
            if self.current_provider_id == model_id:
                return self.providers[model_id]

            # Why: Unload the old provider to free up memory before loading the new one.
            if self.current_provider_id:
                # Use .pop() for safer removal in case the key somehow doesn't exist.
                self.providers.pop(self.current_provider_id, None)

            model_config_data = next((m.model_dump() for m in self.app_config.models if m.id == model_id), None)
            if not model_config_data:
                raise ValueError(f"Model '{model_id}' not found in models.yaml.")

            provider_map = {"mlx": MLXProvider, "llama_cpp": LlamaCppProvider}
            provider_class = provider_map.get(model_config_data['provider'])
            if not provider_class:
                raise ValueError(f"Unknown provider: {model_config_data['provider']}")

            # Why: This is the critical fix. We must pass all required arguments
            # (model_id, config dict, and verbose flag) to the provider's constructor.
            self.providers[model_id] = provider_class(
                model_id=model_config_data['id'],
                config=model_config_data['config'],
                verbose=(self.log_level <= logging.DEBUG)
            )

            self.current_provider_id = model_id
            return self.providers[model_id]
