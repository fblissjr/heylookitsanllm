# src/heylook_llm/providers/llama_cpp_provider.py
import gc
import logging
import traceback
from typing import Generator

from ..config import ChatRequest
from .base import BaseProvider

class LlamaCppProvider(BaseProvider):
    def load_model(self, config: dict, verbose: bool):
        try:
            from llama_cpp import Llama, LlamaRAMCache
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Llava15ChatHandler
        except ImportError as e:
            raise ImportError("llama-cpp-python is not installed.") from e

        logging.info(f"Loading GGUF model: {config['model_path']}")

        chat_handler = None
        if tpl_path := config.get('chat_format_template'):
            logging.info(f"Using custom Jinja2 chat template from: {tpl_path}")
            with open(tpl_path, 'r') as f:
                template = f.read()
            formatter = Jinja2ChatFormatter(template=template)
            chat_handler = formatter.to_chat_handler()
        elif config.get('mmproj_path'):
            logging.info(f"Vision model detected. Explicitly using Llava15ChatHandler.")
            chat_handler = Llava15ChatHandler(clip_model_path=config['mmproj_path'], verbose=verbose)

        try:
            self.model = Llama(
                model_path=config['model_path'],
                chat_format=config.get('chat_format'),
                chat_handler=chat_handler,
                n_ctx=config.get('n_ctx', 4096),
                n_gpu_layers=config.get('n_gpu_layers', -1),
                verbose=verbose,
            )
            self.model.set_cache(LlamaRAMCache())
        except Exception as e:
            raise e

    def create_chat_completion(self, request: ChatRequest) -> Generator:
        try:
            # FIX: Use .model_dump() to get a dictionary from the Pydantic object
            request_dict = request.model_dump(exclude_none=True)

            params = {
                "messages": request_dict.get('messages'),
                "temperature": request_dict.get('temperature', 0.8),
                "top_p": request_dict.get('top_p', 0.95),
                "top_k": request_dict.get('top_k', 40),
                "min_p": request_dict.get('min_p', 0.05),
                "repeat_penalty": request_dict.get('repetition_penalty', 1.1),
                "max_tokens": request_dict.get('max_tokens', 512),
                "stream": True,
            }
            yield from self.model.create_chat_completion(**params)
        except Exception as e:
            logging.error(f"Llama.cpp model call failed: {e}", exc_info=True)
            yield {"choices": [{"delta": {"content": f"\n\nError: Llama.cpp generation failed: {str(e)}"}}]}

    def unload(self):
        logging.info(f"Unloading GGUF model: {self.model_id}")
        if hasattr(self, 'model'):
            del self.model
            gc.collect()
