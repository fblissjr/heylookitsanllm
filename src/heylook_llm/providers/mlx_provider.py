# src/heylook_llm/providers/mlx_provider.py
import gc
import logging
from typing import Generator, Dict

import mlx.core as mx
from mlx_lm.utils import load as lm_load
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm.utils import load as vlm_load

from .base import BaseProvider
from .common.samplers import build as build_sampler
from ..utils import process_vlm_messages

class MLXProvider(BaseProvider):
    def load_model(self, config: dict, verbose: bool):
        self.is_vlm = config.get("vision", False)
        self.config = config
        self.draft_model = None

        model_path = config['model_path']

        # Convert relative paths to absolute paths
        if model_path.startswith('./'):
            import os
            model_path = os.path.abspath(model_path)
            logging.info(f"Converted relative path to absolute: {model_path}")

        # Verify the path exists for local models
        if not model_path.startswith(('http://', 'https://')) and '/' in model_path:
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

        load_fn = vlm_load if self.is_vlm else lm_load

        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path}")

        try:
            self.model, self.processor = load_fn(model_path)
            logging.info(f"Successfully loaded model: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise

        if draft_path := config.get('draft_model_path'):
            logging.info(f"Loading draft model: {draft_path}")
            # Apply same path processing to draft model
            if draft_path.startswith('./'):
                import os
                draft_path = os.path.abspath(draft_path)
            self.draft_model, _ = lm_load(draft_path)

    def create_chat_completion(self, request: dict) -> Generator:
        # Apply model-specific defaults from config, with request taking precedence
        effective_request = self._apply_model_defaults(request)

        tokenizer = self.processor.tokenizer if self.is_vlm else self.processor
        sampler, processors = build_sampler(tokenizer, effective_request)

        if self.is_vlm:
            # Enhanced VLM generation with better parameter handling
            yield from self._generate_vlm_enhanced(effective_request, sampler, processors)
        else:
            # Standard text-only LLM
            prompt = tokenizer.apply_chat_template(
                effective_request['messages'], tokenize=False, add_generation_prompt=True
            )

            yield from lm_stream_generate(
                model=self.model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=effective_request.get('max_tokens', 512),
                sampler=sampler,
                logits_processors=processors,
                draft_model=self.draft_model,
                num_draft_tokens=self.config.get('num_draft_tokens', 5)
            )

    def _apply_model_defaults(self, request: dict) -> dict:
        """Apply model-specific defaults from config, with request taking precedence."""
        defaults = {
            'temperature': self.config.get('temperature', 1.0),
            'top_p': self.config.get('top_p', 0.95),
            'max_tokens': self.config.get('max_tokens', 512),
            'repetition_penalty': self.config.get('repetition_penalty', 1.1),
        }
        return {**defaults, **request}

    def _generate_vlm_enhanced(self, request: dict, sampler, processors) -> Generator:
        """
        Enhanced VLM generation that tries multiple approaches.
        Falls back gracefully if advanced features aren't supported.
        """
        try:
            # Skip unified approach for now - input_embeddings not widely supported yet
            logging.info("Skipping unified VLM approach (input_embeddings not supported in current mlx-lm)")
            raise NotImplementedError("input_embeddings not supported")

        except (AttributeError, TypeError, NotImplementedError, ValueError) as e:
            logging.info(f"Unified VLM approach not available ({e}), using basic VLM generation")
            try:
                # Use basic VLM generation directly
                yield from self._generate_vlm_basic(request)
            except Exception as e2:
                logging.error(f"All VLM generation methods failed: {e2}")
                # Return error message as a simple response object
                yield self._create_error_response(f"VLM generation failed: {str(e2)}")

    def _create_error_response(self, error_message: str):
        """Create a simple error response that mimics GenerationResponse."""
        class SimpleResponse:
            def __init__(self, text):
                self.text = text
                self.token = None
                self.logprobs = None
                self.from_draft = False
                self.prompt_tokens = 0
                self.prompt_tps = 0.0
                self.generation_tokens = 1
                self.generation_tps = 0.0
                self.peak_memory = 0.0

        return SimpleResponse(error_message)

    def _generate_vlm_basic(self, request: dict) -> Generator:
        """Basic VLM generation using standard mlx_vlm.generate."""
        try:
            # Import here to avoid circular imports
            from mlx_vlm.generate import stream_generate as vlm_stream_generate

            logging.info("Using basic VLM generation")

            # Process the messages to extract images and create a prompt
            images, formatted_prompt = process_vlm_messages(
                self.processor, self.model.config, request['messages']
            )

            # mlx_vlm.generate.stream_generate expects a formatted prompt and images
            # Try different approaches based on the error message
            try:
                # Method 1: Standard approach
                for response in vlm_stream_generate(
                    model=self.model,
                    processor=self.processor,
                    prompt=formatted_prompt,
                    image=images if images else None,  # Handle empty images
                    max_tokens=request.get('max_tokens', 512),
                    temperature=request.get('temperature', 1.0),
                    top_p=request.get('top_p', 0.95),
                ):
                    yield response
            except Exception as e1:
                logging.warning(f"Standard VLM approach failed: {e1}")

                # Method 2: Try with return_tensors='pt' (PyTorch tensors)
                try:
                    logging.info("Trying VLM generation with PyTorch tensors...")

                    # Process inputs manually with PyTorch tensors
                    if images:
                        # Try processing with PyTorch format
                        inputs = self.processor(
                            text=formatted_prompt,
                            images=images,
                            return_tensors="pt"  # Try PyTorch format
                        )
                        logging.info(f"Successfully processed inputs with PyTorch tensors")
                    else:
                        inputs = self.processor(
                            text=formatted_prompt,
                            return_tensors="pt"
                        )

                    # Convert back to MLX format if needed and call VLM
                    for response in vlm_stream_generate(
                        model=self.model,
                        processor=self.processor,
                        prompt=formatted_prompt,
                        image=images if images else None,
                        max_tokens=request.get('max_tokens', 512),
                        temperature=request.get('temperature', 1.0),
                        top_p=request.get('top_p', 0.95),
                    ):
                        yield response

                except Exception as e2:
                    logging.warning(f"PyTorch tensor approach failed: {e2}")

                    # Method 3: Simple text-only fallback
                    logging.info("Falling back to text-only generation")
                    simple_response = self._create_error_response(
                        f"Vision processing failed, using text-only response: {formatted_prompt}"
                    )
                    yield simple_response

        except Exception as e:
            logging.error(f"All VLM generation methods failed: {e}")
            # Return error as a valid response rather than raising
            yield self._create_error_response(f"VLM Error: {str(e)}")

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model, self.processor
        if hasattr(self, 'draft_model'):
            del self.draft_model
        gc.collect()
        mx.clear_cache()
        logging.info(f"Unloaded MLX model: {self.model_id}")
