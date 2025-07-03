# src/heylook_llm/providers/mlx_provider.py
import gc
import logging
import traceback
from typing import Generator, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from mlx_lm.utils import load as lm_load
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.generate import stream_generate as vlm_stream_generate
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template

from ..config import ChatRequest
from .base import BaseProvider
from .common.samplers import build as build_sampler
from ..utils import load_image

class LanguageModelWrapper(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
    def __call__(self, *args, **kwargs):
        return self.language_model(*args, **kwargs).logits
    @property
    def layers(self):
        if hasattr(self.language_model, 'model'):
            return self.language_model.model.layers
        return self.language_model.layers

class MLXProvider(BaseProvider):
    def load_model(self, config: dict, verbose: bool):
        self.is_vlm = config.get("vision", False)
        self.config = config
        self.draft_model = None
        model_path = config['model_path']
        load_fn = vlm_load if self.is_vlm else lm_load
        logging.info(f"Loading {'VLM' if self.is_vlm else 'LLM'} model from: {model_path} using {'mlx_vlm' if self.is_vlm else 'mlx_lm'} loader.")
        try:
            self.model, self.processor = load_fn(model_path)
        except Exception as e:
            raise e
        if draft_path := config.get('draft_model_path'):
            if self.is_vlm:
                logging.warning("Speculative decoding is not currently supported for VLM models.")
            else:
                self.draft_model, _ = lm_load(draft_path)

    def _prepare_vlm_inputs(self, messages: List) -> Tuple[List[Image.Image], str, bool]:
        images, text_messages, has_images = [], [], False
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'image_url':
                        images.append(load_image(part.image_url.url))
                        has_images = True
                text_messages.append({"role": msg.role, "content": "".join(text_parts)})
            elif isinstance(content, str):
                text_messages.append({"role": msg.role, "content": content})

        formatted_prompt = vlm_apply_chat_template(
            self.processor, self.model.config, text_messages, num_images=len(images)
        )
        return images, formatted_prompt, has_images

    def _apply_model_defaults(self, request: ChatRequest) -> dict:
        global_defaults = {'temperature': 0.1, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'max_tokens': 512, 'repetition_penalty': 1.0}
        merged_config = global_defaults.copy()
        merged_config.update({k: v for k, v in self.config.items() if v is not None})
        # --- FIX: Use .model_dump() to correctly get a dict from the Pydantic object ---
        merged_config.update({k: v for k, v in request.model_dump().items() if v is not None})
        return merged_config

    def create_chat_completion(self, request: ChatRequest) -> Generator:
        effective_request = self._apply_model_defaults(request)
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        try:
            if self.is_vlm:
                images, formatted_prompt, has_images = self._prepare_vlm_inputs(request.messages)
                if not has_images:
                    compatible_model = LanguageModelWrapper(self.model.language_model)
                    sampler, processors = build_sampler(tokenizer, effective_request)
                    yield from lm_stream_generate(model=compatible_model, tokenizer=tokenizer, prompt=formatted_prompt, sampler=sampler, logits_processors=processors, max_tokens=effective_request['max_tokens'])
                else:
                    yield from vlm_stream_generate(model=self.model, processor=self.processor, prompt=formatted_prompt, image=images, temperature=effective_request['temperature'], max_tokens=effective_request['max_tokens'], top_p=effective_request['top_p'])
            else:
                sampler, processors = build_sampler(tokenizer, effective_request)
                prompt = tokenizer.apply_chat_template([msg.model_dump(exclude_none=True) for msg in request.messages], tokenize=False, add_generation_prompt=True)
                yield from lm_stream_generate(model=self.model, tokenizer=tokenizer, prompt=prompt, sampler=sampler, logits_processors=processors, max_tokens=effective_request['max_tokens'], draft_model=self.draft_model)
        except Exception as e:
            logging.error(f"MLX model call failed: {e}", exc_info=True)
            yield {"text": f"Error: MLX generation failed: {str(e)}"}

    def unload(self):
        logging.info(f"Unloading MLX model: {self.model_id}")
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'processor'): del self.processor
        if hasattr(self, 'draft_model'): del self.draft_model
        gc.collect()
        mx.clear_cache()
