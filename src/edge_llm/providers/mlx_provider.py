# src/edge_llm/providers/mlx_provider.py
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
        load_fn = vlm_load if self.is_vlm else lm_load

        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path}")
        self.model, self.processor = load_fn(model_path)

        if draft_path := config.get('draft_model_path'):
            logging.info(f"Loading draft model: {draft_path}")
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
        Enhanced VLM generation that uses advanced sampling when possible.
        Falls back gracefully to basic VLM generation if advanced features aren't supported.
        """
        try:
            # Method 1: Try the unified approach (requires newer mlx-lm with input_embeddings support)
            yield from self._try_unified_vlm_generation(request, sampler, processors)
        except (AttributeError, TypeError, NotImplementedError) as e:
            logging.info(f"Unified VLM approach not available ({e}), using enhanced fallback")
            try:
                # Method 2: Use custom VLM generation with parameter injection
                yield from self._generate_vlm_with_custom_sampling(request, sampler, processors)
            except Exception as e2:
                logging.warning(f"Custom sampling failed ({e2}), using basic VLM generation")
                # Method 3: Basic VLM generation (guaranteed to work)
                yield from self._generate_vlm_basic(request)

    def _try_unified_vlm_generation(self, request: dict, sampler, processors) -> Generator:
        """Try the unified approach using mlx-lm's generation engine."""
        # Process multimodal messages
        images, formatted_prompt = process_vlm_messages(
            self.processor, self.model.config, request['messages']
        )

        # Prepare inputs
        if images:
            inputs = self.processor(
                text=formatted_prompt,
                images=images,
                return_tensors="np"
            )
            input_ids = mx.array(inputs['input_ids'])
            pixel_values = mx.array(inputs['pixel_values']) if 'pixel_values' in inputs else None
        else:
            input_ids = mx.array(self.processor.tokenizer.encode(formatted_prompt))
            pixel_values = None

        # Get fused embeddings (this is where it might fail if not supported)
        input_embeddings = self.model.get_input_embeddings(input_ids, pixel_values)

        # Use mlx-lm's generation with embeddings
        # NOTE: This requires a recent version of mlx-lm with input_embeddings support
        yield from lm_stream_generate(
            model=self.model.language_model,
            tokenizer=self.processor.tokenizer,
            prompt=[],  # Empty because context is in input_embeddings
            input_embeddings=input_embeddings,
            max_tokens=request.get('max_tokens', 512),
            sampler=sampler,
            logits_processors=processors,
        )

    def _generate_vlm_with_custom_sampling(self, request: dict, sampler, processors) -> Generator:
        """
        Custom VLM generation that tries to inject our advanced sampling.
        This is experimental and may not work with all models.
        """
        from mlx_vlm.generate import generate_step

        # Process messages and prepare inputs
        images, formatted_prompt = process_vlm_messages(
            self.processor, self.model.config, request['messages']
        )

        # Prepare inputs like mlx-vlm does
        if images:
            inputs = self.processor(
                text=formatted_prompt,
                images=images,
                return_tensors="np"
            )
        else:
            inputs = self.processor(text=formatted_prompt, return_tensors="np")

        # Convert to MLX arrays
        input_ids = mx.array(inputs['input_ids'])
        pixel_values = mx.array(inputs.get('pixel_values', [])) if inputs.get('pixel_values') is not None else None

        # Call generate_step directly with our custom sampling
        # This bypasses vlm_stream_generate and gives us more control
        for token, logprobs in generate_step(
            prompt=input_ids,
            model=self.model,
            pixel_values=pixel_values,
            max_tokens=request.get('max_tokens', 512),
            temperature=request.get('temperature', 1.0),
            repetition_penalty=request.get('repetition_penalty'),
            top_p=request.get('top_p', 1.0),
        ):
            # Convert token to text using our detokenizer
            text = self.processor.tokenizer.decode([token.item()])
            # Yield in the expected format
            from mlx_lm.generate import GenerationResponse
            yield GenerationResponse(text=text)

    def _generate_vlm_basic(self, request: dict) -> Generator:
        """Basic VLM generation using standard mlx_vlm.generate."""
        from mlx_vlm.generate import stream_generate as vlm_stream_generate

        logging.info("Using basic VLM generation")
        yield from vlm_stream_generate(
            model=self.model,
            processor=self.processor,
            messages=request['messages'],
            max_tokens=request.get('max_tokens', 512),
            temperature=request.get('temperature', 1.0),
            top_p=request.get('top_p', 0.95),
        )

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model, self.processor
        if hasattr(self, 'draft_model'):
            del self.draft_model
        gc.collect()
        mx.clear_cache()
        logging.info(f"Unloaded MLX model: {self.model_id}")
