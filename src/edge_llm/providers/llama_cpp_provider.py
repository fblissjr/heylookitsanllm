# src/edge_llm/providers/llama_cpp_provider.py

import gc
import logging
from typing import Generator, Dict, Any

from llama_cpp import Llama
# Why: This is the critical fix. We import the specific chat handlers from their
# correct submodule within the llama-cpp-python package.
from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Llava15ChatHandler

from edge_llm.providers.base import BaseProvider

class LlamaCppProvider(BaseProvider):
    """Provider for running GGUF models via llama-cpp-python."""
    # The __init__ is inherited from BaseProvider, no need to redefine it.

    def load_model(self, config: dict, verbose: bool):
        logging.info(f"Loading GGUF model: {config['model_path']}")
        chat_handler = None

        # Why: This logic correctly prioritizes a custom Jinja template over built-in formats.
        if config.get('chat_format_template'):
            logging.info(f"Loading custom chat template from: {config['chat_format_template']}")
            with open(config['chat_format_template'], 'r') as f: template = f.read()
            formatter = Jinja2ChatFormatter(template=template, eos_token=config.get('eos_token'), bos_token=config.get('bos_token'))
            chat_handler = formatter.to_chat_handler()
        elif config.get("mmproj_path"):
            # Why: This is a default handler for LLaVA-style multimodal models.
            chat_handler = Llava15ChatHandler(clip_model_path=config['mmproj_path'], verbose=verbose)

        self.model = Llama(
            model_path=config['model_path'],
            chat_format=config.get('chat_format'),
            chat_handler=chat_handler,
            n_ctx=config.get('n_ctx', 4096),
            n_gpu_layers=config.get('n_gpu_layers', -1),
            verbose=verbose
        )

    def create_chat_completion(self, request: dict) -> Generator:
        # Why: llama-cpp-python is already OpenAI compatible, so we just pass the request through.
        # The stream=True argument ensures it returns a generator, which our API handlers expect.
        yield from self.model.create_chat_completion(**request, stream=True)

    def __del__(self):
        """Ensure the Llama.cpp model is released from memory."""
        if hasattr(self, 'model'):
            del self.model
            gc.collect()
        logging.info(f"Unloaded GGUF model: {self.model_id}")
