# providers/mlx_provider.py
import gc
import logging
import mlx.core as mx

from edge_llm.providers.base import BaseProvider
from edge_llm.utils import process_vlm_messages

# mlx-vlm imports
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" # Disable advisory warnings for mlx-vlm import of transformers
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import stream_generate as vlm_stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

class MLXProvider(BaseProvider):
    def load_model(self, config: dict, verbose: bool):
        self.config = config; self.draft_model = None
        logging.info(f"Loading MLX model from vendored code: {config['model_path']}")
        self.model, self.processor = vlm_load(config['model_path'])

        draft_path = config.get('draft_model_path')
        if draft_path:
            logging.info(f"Loading MLX draft model from vendored code: {draft_path}")
            self.draft_model, _ = vlm_load(draft_path)

    def create_chat_completion(self, request: dict) -> Generator:
        sampler = make_sampler(temp=request.get("temperature", 1.0), top_p=request.get("top_p", 0.95))
        logits_processors = make_logits_processors(repetition_penalty=request.get("repetition_penalty"))

        images, formatted_prompt = process_vlm_messages(self.processor, self.model.config, request['messages'])

        active_draft_model = self.draft_model
        num_draft_tokens = request.get('num_draft_tokens') or self.config.get('num_draft_tokens', 5)

        generator_args = {
            "prompt": formatted_prompt, "model": self.model,
            "max_tokens": request.get('max_tokens', 512),
            "sampler": sampler, "logits_processors": logits_processors,
        }

        if active_draft_model:
            logging.info(f"Using speculative decoding with {num_draft_tokens} draft tokens.")
            yield from speculative_generate_step(**generator_args, draft_model=active_draft_model, num_draft_tokens=num_draft_tokens)
        else:
            generator = generate_step(**generator_args)
            for token, logprobs in generator:
                yield token, logprobs, False

    def __del__(self):
        if hasattr(self, 'model'): del self.model, self.processor; gc.collect(); mx.clear_cache()
        logging.info(f"Unloaded MLX model: {self.model_id}")
