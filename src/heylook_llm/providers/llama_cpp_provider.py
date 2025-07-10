# src/heylook_llm/providers/llama_cpp_provider.py
import gc
import logging
import traceback
import json
from typing import Generator, Dict

from llama_cpp import Llama, LlamaRAMCache
from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Llava15ChatHandler

from ..config import ChatRequest
from .base import BaseProvider

class LlamaCppProvider(BaseProvider):
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None

    def load_model(self):
        logging.info(f"Loading GGUF model: {self.config['model_path']}")

        chat_handler = None

        # First, check if we need vision support
        if self.config.get('mmproj_path'):
            logging.info(f"Vision model detected. Explicitly using Llava15ChatHandler.")
            chat_handler = Llava15ChatHandler(clip_model_path=self.config['mmproj_path'], verbose=self.verbose)

        # Load the model first
        try:
            self.model = Llama(
                model_path=self.config['model_path'],
                chat_format=self.config.get('chat_format'),
                chat_handler=chat_handler,
                n_ctx=self.config.get('n_ctx', 4096),
                n_gpu_layers=self.config.get('n_gpu_layers', -1),
                verbose=self.verbose,
            )
            self.model.set_cache(LlamaRAMCache())
        except Exception as e:
            raise e

        # Now handle custom chat template if specified (and not already using vision handler)
        if tpl_path := self.config.get('chat_format_template'):
            if not chat_handler:  # Only if we haven't already set a vision handler
                logging.info(f"Using custom Jinja2 chat template from: {tpl_path}")
                try:
                    with open(tpl_path, 'r') as f:
                        template = f.read()

                    # Get tokens from the loaded model's tokenizer
                    eos_token = self.model._model.token_get_text(self.model._model.token_eos())
                    bos_token = self.model._model.token_get_text(self.model._model.token_bos())

                    chat_handler = Jinja2ChatFormatter(
                        template=template,
                        eos_token=eos_token,
                        bos_token=bos_token
                    )

                    # Update the model with the new chat handler
                    self.model.chat_handler = chat_handler

                except FileNotFoundError:
                    logging.error(f"Custom chat template not found at: {tpl_path}", exc_info=True)
                    raise
                except Exception as e:
                    logging.error(f"Failed to load custom chat template: {e}", exc_info=True)
                    raise
            else:
                logging.warning(f"Ignoring custom chat template because vision handler is already set")

    def create_chat_completion(self, request: ChatRequest) -> Generator:

        class LlamaCppStreamChunk:
            def __init__(self, text, usage=None):
                self.text = text
                self.prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
                self.generation_tokens = usage.get("completion_tokens", 0) if usage else 0

        try:
            # Check if model is loaded
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

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

            if self.verbose:
                # Log only the parameters, not the messages (which may contain base64 images)
                safe_params = {k: v for k, v in params.items() if k != 'messages'}
                safe_params['message_count'] = len(params.get('messages', []))
                logging.debug(f"Calling Llama.cpp with params: {json.dumps(safe_params, indent=2)}")

            for chunk in self.model.create_chat_completion(**params):
                text = ''
                usage = None

                # Safely extract text from chunk
                if isinstance(chunk, dict):
                    choices = chunk.get('choices', [])
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get('delta', {})
                        if isinstance(delta, dict):
                            text = delta.get('content', '')
                    usage = chunk.get('usage')

                yield LlamaCppStreamChunk(text=text, usage=usage)

        except Exception as e:
            logging.error(f"Llama.cpp model call failed: {e}", exc_info=True)
            yield LlamaCppStreamChunk(text=f"\n\nError: Llama.cpp generation failed: {str(e)}")

    def unload(self):
        logging.info(f"Unloading GGUF model: {self.model_id}")
        if hasattr(self, 'model'):
            del self.model
            gc.collect()
