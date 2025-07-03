# src/edge_llm/router.py
import yaml
import logging
import threading
from typing import Optional

from edge_llm.config import AppConfig
from edge_llm.providers.base import BaseProvider
from edge_llm.providers.mlx_provider import MLXProvider
from edge_llm.providers.llama_cpp_provider import LlamaCppProvider

class ModelRouter:
    """Manages loading, unloading, and routing to different model providers."""
    def __init__(self, config_path: str, log_level: int, initial_model_id: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.app_config = AppConfig(**yaml.safe_load(f))

        self.providers = {}
        self.current_provider_id = None
        self.log_level = log_level
        self.loading_lock = threading.Lock()

        # Validate initial model exists in config
        if initial_model_id:
            if not self.app_config.get_model_config(initial_model_id):
                logging.error(f"Initial model '{initial_model_id}' not found in config. Available models: {[m.id for m in self.app_config.models]}")
                initial_model_id = None

        # If no initial model specified or invalid, use the first enabled model
        if not initial_model_id:
            enabled_models = self.app_config.get_enabled_models()
            if enabled_models:
                initial_model_id = enabled_models[0].id
                logging.info(f"No initial model specified, using first enabled model: {initial_model_id}")

        # Try to pre-warm the initial model
        if initial_model_id:
            try:
                logging.info(f"Pre-warming initial model: {initial_model_id}")
                self.get_provider(initial_model_id)
                logging.info(f"Successfully loaded initial model: {initial_model_id}")
            except Exception as e:
                logging.error(f"Failed to pre-warm initial model '{initial_model_id}': {e}")
                logging.info("Server will start with no models loaded")

    def get_provider(self, model_id: str) -> BaseProvider:
        """Get or create a provider for the specified model."""

        # Handle None or empty model_id
        if not model_id or model_id.strip() == "":
            available_models = self.list_available_models()
            if available_models:
                model_id = available_models[0]
                logging.info(f"No model specified, using default: {model_id}")
            else:
                raise ValueError("No model specified and no models available")

        with self.loading_lock:
            # Return current provider if it's the same model
            if self.current_provider_id == model_id and model_id in self.providers:
                logging.debug(f"Reusing existing provider for model: {model_id}")
                return self.providers[model_id]

            # Unload current provider to free memory
            if self.current_provider_id and self.current_provider_id in self.providers:
                old_provider = self.providers.pop(self.current_provider_id, None)
                if old_provider:
                    logging.info(f"Unloading model: {self.current_provider_id}")
                    try:
                        del old_provider
                    except Exception as e:
                        logging.warning(f"Error during model cleanup: {e}")

            # Get model config
            model_config = self.app_config.get_model_config(model_id)
            if not model_config:
                available_models = [m.id for m in self.app_config.get_enabled_models()]
                error_msg = f"Model '{model_id}' not found or disabled. Available models: {available_models}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Check if model is enabled
            if not model_config.enabled:
                available_models = [m.id for m in self.app_config.get_enabled_models()]
                error_msg = f"Model '{model_id}' is disabled. Available models: {available_models}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Create provider
            provider_map = {
                "mlx": MLXProvider,
                "llama_cpp": LlamaCppProvider
            }

            provider_class = provider_map.get(model_config.provider)
            if not provider_class:
                raise ValueError(f"Unknown provider: {model_config.provider}")

            logging.info(f"Loading model '{model_id}' with provider '{model_config.provider}'")

            try:
                # Create the provider
                new_provider = provider_class(
                    model_config.id,
                    model_config.config.model_dump(),
                    self.log_level <= logging.DEBUG
                )

                # Only update state if provider creation was successful
                self.providers[model_id] = new_provider
                self.current_provider_id = model_id
                logging.info(f"Successfully loaded model: {model_id}")
                return new_provider

            except Exception as e:
                # Clean up on failure
                if model_id in self.providers:
                    del self.providers[model_id]
                if self.current_provider_id == model_id:
                    self.current_provider_id = None

                error_msg = f"Failed to load model '{model_id}': {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)

    def list_available_models(self) -> list[str]:
        """Return list of available model IDs."""
        return [m.id for m in self.app_config.get_enabled_models()]

    def get_current_model_id(self) -> Optional[str]:
        """Return the currently loaded model ID."""
        return self.current_provider_id
