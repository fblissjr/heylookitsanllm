# src/edge_llm/providers/mlx_provider.py
import gc
import logging
from typing import Generator, Dict

import mlx.core as mx
# Why: Import the single, correct generator from mlx-lm
from mlx_lm.utils import load as lm_load
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.generate import stream_generate as vlm_stream_generate

from .base import BaseProvider
from .common.samplers import build as build_sampler

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
        tokenizer = self.processor.tokenizer if self.is_vlm else self.processor
        sampler, processors = build_sampler(tokenizer, request)

        if self.is_vlm:
            # vlm_stream_generate handles everything internally.
            yield from vlm_stream_generate(
                model=self.model,
                processor=self.processor,
                messages=request['messages'],
                max_tokens=request.get('max_tokens', 512),
                temperature=request.get('temperature', 1.0),
                top_p=request.get('top_p', 0.95),
            )
        else: # Standard text-only LLM
            prompt = tokenizer.apply_chat_template(
                request['messages'], tokenize=False, add_generation_prompt=True
            )

            # Why: This is the correct, unified way to call the generator.
            # We pass the draft_model as an argument. The `stream_generate` function
            # will automatically use it if it's not None.
            yield from lm_stream_generate(
                model=self.model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=request.get('max_tokens', 512),
                sampler=sampler,
                logits_processors=processors,
                draft_model=self.draft_model, # Pass the draft model here
                num_draft_tokens=self.config.get('num_draft_tokens', 5)
            )

    def __del__(self):
        if hasattr(self, 'model'): del self.model, self.processor
        if hasattr(self, 'draft_model'): del self.draft_model
        gc.collect(); mx.clear_cache()
        logging.info(f"Unloaded MLX model: {self.model_id}")
