# src/heylook_llm/providers/llama_cpp_provider.py
import gc
import logging
import traceback
import json
import errno
import threading
from typing import Generator, Dict

from llama_cpp import Llama, LlamaRAMCache, llama_cpp
from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Llava15ChatHandler

from ..config import ChatRequest
from .base import BaseProvider

class LlamaCppProvider(BaseProvider):
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None
        self._model_broken = False
        self._generation_lock = threading.Lock()  # Mutex for thread-safe generation

    def load_model(self):
        logging.info(f"Loading GGUF model: {self.config['model_path']}")
        
        # Reset broken flag when loading
        self._model_broken = False

        chat_handler = None

        # First, check if we need vision support
        if self.config.get('mmproj_path'):
            logging.info(f"Vision model detected. Explicitly using Llava15ChatHandler.")
            chat_handler = Llava15ChatHandler(clip_model_path=self.config['mmproj_path'], verbose=self.verbose)

        # Load the model first
        try:
            # Enable embeddings if specified in config (default False for safety)
            enable_embeddings = self.config.get('embedding', False)
            if enable_embeddings:
                logging.info("Embeddings extraction enabled for this model")
            
            self.model = Llama(
                model_path=self.config['model_path'],
                chat_format=self.config.get('chat_format'),
                chat_handler=chat_handler,
                n_ctx=self.config.get('n_ctx', 4096),
                n_gpu_layers=self.config.get('n_gpu_layers', -1),
                n_batch=self.config.get('n_batch', 512),
                n_threads=self.config.get('n_threads', None),  # None = auto-detect
                use_mmap=self.config.get('use_mmap', True),
                use_mlock=self.config.get('use_mlock', False),
                embedding=enable_embeddings,  # Use config value
                verbose=self.verbose,
            )
            
            # Set cache if enabled (following llama-cpp-python server pattern)
            # By default, enable cache for better performance with repeated prompts
            if self.config.get('cache', True):
                cache_size = self.config.get('cache_size', 2 << 30)  # Default 2GB
                self.model.set_cache(LlamaRAMCache(capacity_bytes=cache_size))
                logging.debug(f"Set RAM cache with size {cache_size} bytes")
            
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

        # Use lock to ensure thread safety (following llama-cpp-python server pattern)
        # Don't clear cache - let llama-cpp handle prefix matching for performance
        with self._generation_lock:
            try:
                # Check if model is loaded or broken
                if self.model is None or self._model_broken:
                    if self._model_broken:
                        logging.warning("Model was marked as broken, attempting to reload...")
                        # Clean up the broken model first
                        if self.model is not None:
                            try:
                                if hasattr(self.model, '_cache'):
                                    self.model._cache = None
                                del self.model
                                self.model = None
                                gc.collect()
                            except:
                                pass
                        self._model_broken = False
                        self.load_model()
                    else:
                        raise RuntimeError("Model not loaded. Call load_model() first.")

                request_dict = request.model_dump(exclude_none=True)
                
                # Process messages to ensure correct format
                messages = request_dict.get('messages', [])
                processed_messages = []
                
                # Check if this is a vision model
                has_vision = self.config.get('mmproj_path') is not None
                
                for msg in messages:
                    msg_copy = msg.copy()
                    # If content is a list (multimodal format)
                    if isinstance(msg_copy.get('content'), list):
                        if has_vision:
                            # Keep multimodal format for vision models
                            processed_messages.append(msg_copy)
                        else:
                            # Extract only text parts for non-vision models
                            text_parts = []
                            for part in msg_copy['content']:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    text_parts.append(part.get('text', ''))
                            msg_copy['content'] = ' '.join(text_parts)
                            processed_messages.append(msg_copy)
                    else:
                        # Content is already a string, keep as is
                        processed_messages.append(msg_copy)
                
                # Build parameters for generation
                params = {
                    "messages": processed_messages,
                    "temperature": request_dict.get('temperature', 0.8),
                    "top_p": request_dict.get('top_p', 0.95),
                    "top_k": request_dict.get('top_k', 40),
                    "min_p": request_dict.get('min_p', 0.05),
                    "repeat_penalty": request_dict.get('repetition_penalty', 1.1),
                    "max_tokens": request_dict.get('max_tokens', 512),
                    "stream": True,
                }
                
                # Add stop tokens - prioritize request, fallback to config
                if 'stop' in request_dict:
                    params['stop'] = request_dict['stop']
                elif 'stop' in self.config:
                    params['stop'] = self.config['stop']

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

            except BrokenPipeError as e:
                # Handle broken pipe specifically
                logging.error(f"Llama.cpp connection lost (Broken pipe): {e}")
                # Mark model as broken to force reload on next request
                self._model_broken = True
                yield LlamaCppStreamChunk(text=f"\n\nError: Connection to llama.cpp lost. Please retry your request.")
            except OSError as e:
                # Handle other OS-level errors (including EPIPE)
                if e.errno == errno.EPIPE:  # EPIPE (32)
                    logging.error(f"Llama.cpp pipe disconnected: {e}")
                    self._model_broken = True
                    yield LlamaCppStreamChunk(text=f"\n\nError: Connection to llama.cpp lost. Please retry your request.")
                else:
                    logging.error(f"Llama.cpp OS error: {e}", exc_info=True)
                    yield LlamaCppStreamChunk(text=f"\n\nError: System error occurred: {str(e)}")
            except Exception as e:
                logging.error(f"Llama.cpp model call failed: {e}", exc_info=True)
                yield LlamaCppStreamChunk(text=f"\n\nError: Llama.cpp generation failed: {str(e)}")

    def unload(self):
        logging.info(f"Unloading GGUF model: {self.model_id}")
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Clear the cache first
                if hasattr(self.model, '_cache'):
                    self.model._cache = None
                # Delete the model
                del self.model
                self.model = None
        except Exception as e:
            logging.warning(f"Error during model unload: {e}")
        finally:
            # Always collect garbage
            gc.collect()