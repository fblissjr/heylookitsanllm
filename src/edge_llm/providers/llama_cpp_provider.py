# src/edge_llm/providers/llama_cpp_provider.py
import gc
import logging
from typing import Generator, Dict, Any

from .base import BaseProvider

class LlamaCppProvider(BaseProvider):
    """Provider for running GGUF models via llama-cpp-python."""

    def load_model(self, config: dict, verbose: bool):
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Llava15ChatHandler
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Install it with: "
                "pip install 'llama-cpp-python[server]'"
            ) from e

        logging.info(f"Loading GGUF model: {config['model_path']}")
        chat_handler = None

        # Handle custom chat templates and vision models
        if config.get('chat_format_template'):
            logging.info(f"Loading custom chat template from: {config['chat_format_template']}")
            try:
                with open(config['chat_format_template'], 'r') as f:
                    template = f.read()
                formatter = Jinja2ChatFormatter(
                    template=template,
                    eos_token=config.get('eos_token'),
                    bos_token=config.get('bos_token')
                )
                chat_handler = formatter.to_chat_handler()
            except Exception as e:
                logging.error(f"Failed to load chat template: {e}")
                raise
        elif config.get("mmproj_path"):
            # Default handler for LLaVA-style multimodal models
            chat_handler = Llava15ChatHandler(
                clip_model_path=config['mmproj_path'],
                verbose=verbose
            )

        try:
            self.model = Llama(
                model_path=config['model_path'],
                chat_format=config.get('chat_format'),
                chat_handler=chat_handler,
                n_ctx=config.get('n_ctx', 4096),
                n_gpu_layers=config.get('n_gpu_layers', -1),
                n_batch=config.get('n_batch', 512),
                n_threads=config.get('n_threads'),
                verbose=verbose
            )
            logging.info(f"Successfully loaded GGUF model: {config['model_path']}")
        except Exception as e:
            logging.error(f"Failed to load GGUF model: {e}")
            raise

    def create_chat_completion(self, request: dict) -> Generator:
        """Create chat completion using llama.cpp."""
        try:
            # llama-cpp-python is already OpenAI compatible, so we pass the request through
            # The stream=True argument ensures it returns a generator
            yield from self.model.create_chat_completion(**request, stream=True)
        except Exception as e:
            logging.error(f"Chat completion failed: {e}")
            raise

    def __del__(self):
        """Ensure the Llama.cpp model is released from memory."""
        if hasattr(self, 'model'):
            try:
                del self.model
                gc.collect()
                logging.info(f"Unloaded GGUF model: {self.model_id}")
            except Exception as e:
                logging.warning(f"Error during model cleanup: {e}")
