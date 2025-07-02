# src/edge_llm/providers/mlx_provider.py

import gc
import logging
from typing import Generator, Dict, Any

import mlx.core as mx
from mlx_lm.utils import load as lm_load
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm.utils import load as vlm_load, prepare_inputs
from edge_llm.utils import process_vlm_messages

from edge_llm.providers.base import BaseProvider
from edge_llm.providers.common.samplers import build as build_sampler
from edge_llm.providers.common.cache_helpers import build_or_load_cache

class MLXProvider(BaseProvider):
    """
    A unified provider for both MLX Language Models (LLMs) and
    Vision-Language Models (VLMs).
    """
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        self.config = config
        self.verbose = verbose
        self.is_vlm = config.get("vision", False)

        self.model = None
        self.processor = None
        self.draft_model = None

        self.load_model()

    def load_model(self):
        """Loads the appropriate model and an optional draft model."""
        model_path = self.config['model_path']
        adapter_path = self.config.get('adapter_path')
        draft_path = self.config.get('draft_model_path')

        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path}")

        load_fn = vlm_load if self.is_vlm else lm_load
        self.model, self.processor = load_fn(model_path, adapter_path=adapter_path)

        if draft_path:
            logging.info(f"Loading draft model: {draft_path}")
            self.draft_model, _ = lm_load(draft_path)

        cache_target_model = self.model.language_model if self.is_vlm else self.model
        self.cache, self.quantize_fn = build_or_load_cache(cache_target_model, self.config)

        logging.info(f"Model '{self.model_id}' loaded successfully.")

    def create_chat_completion(self, request: Dict) -> Generator:
        """Single entry point for generating completions."""
        language_model = self.model.language_model if self.is_vlm else self.model
        tokenizer = self.processor.tokenizer if self.is_vlm else self.processor

        sampler, processors = build_sampler(tokenizer, request)

        generator_args = request.copy()
        generator_args.update({
            "sampler": sampler,
            "logits_processors": processors,
            "prompt_cache": self.cache,
            "draft_model": self.draft_model
        })

        if self.is_vlm:
            images, formatted_prompt = process_vlm_messages(self.processor, self.model.config, request['messages'])

            if images: # Ensure we only do VLM processing if images are present
                vlm_inputs = prepare_inputs(self.processor, prompts=formatted_prompt, images=images)

                logging.info("Fusing image and text embeddings...")
                fused_embeddings = self.model.get_input_embeddings(
                    input_ids=vlm_inputs["input_ids"],
                    pixel_values=vlm_inputs["pixel_values"]
                )
                logging.info("Fusion complete. Handing off to mlx-lm engine.")

                generator_args["input_embeddings"] = fused_embeddings
                prompt_for_engine = []
            else: # Fallback to text-only if a VLM model is used without an image
                prompt_for_engine = tokenizer.encode(formatted_prompt)

        else:
            # For text-only models
            prompt_for_engine = tokenizer.apply_chat_template(request['messages'], tokenize=True, add_generation_prompt=True)

        generator = lm_stream_generate(
            model=language_model,
            tokenizer=tokenizer,
            prompt=prompt_for_engine,
            **generator_args
        )

        for result in generator:
            self.quantize_fn()
            yield result

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            del self.processor
            if self.draft_model:
                del self.draft_model
        gc.collect()
        mx.clear_cache()
        logging.info(f"Unloaded MLX model: {self.model_id}")
