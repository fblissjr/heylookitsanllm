# src/providers/mlx_provider.py
import gc, logging
import mlx.core as mx
from .base import BaseProvider
from ..utils import process_vlm_messages
from mlx_lm.utils import load as mlx_load
from mlx_lm.generate import generate_step, speculative_generate_step, GenerationResponse
from mlx_lm.sample_utils import make_sampler, make_logits_processors

class MLXProvider(BaseProvider):
    def load_model(self, config: dict, verbose: bool):
        self.config = config; self.draft_model = None
        logging.info(f"Loading MLX model: {config['model_path']}")
        self.model, self.processor = mlx_load(config['model_path'])

        draft_path = config.get('draft_model_path')
        if draft_path:
            logging.info(f"Loading MLX draft model: {draft_path}")
            self.draft_model, _ = mlx_load(draft_path)

    def create_chat_completion(self, request: dict) -> Generator:
        # Why: We construct the sampler and logits processors using all available
        # parameters from the request, falling back to defaults. This gives full control.
        sampler = make_sampler(
            temp=request.get("temperature", 1.0),
            top_p=request.get("top_p", 0.95),
            top_k=request.get("top_k", 40)
        )
        logits_processors = make_logits_processors(
            repetition_penalty=request.get("repetition_penalty"),
            repetition_context_size=request.get("repetition_context_size")
        )

        images, formatted_prompt = process_vlm_messages(self.processor, self.model.config, request['messages'])

        active_draft_model = self.draft_model # For now, we don't support runtime override, but this is where it would go.
        num_draft_tokens = request.get('num_draft_tokens') or self.config.get('num_draft_tokens', 5)

        generator_args = {
            "prompt": formatted_prompt,
            "model": self.model,
            "max_tokens": request.get('max_tokens', 512),
            "sampler": sampler,
            "logits_processors": logits_processors,
        }

        # Why: This logic correctly chooses the optimized speculative generator
        # when a draft model is active.
        if active_draft_model:
            logging.info(f"Using speculative decoding with {num_draft_tokens} draft tokens.")
            yield from speculative_generate_step(
                **generator_args, draft_model=active_draft_model, num_draft_tokens=num_draft_tokens
            )
        else:
            generator = generate_step(**generator_args)
            # The standard generator yields (token, logprobs). We add a `False` for `from_draft`
            # to make the output format consistent with the speculative generator.
            for token, logprobs in generator:
                yield token, logprobs, False
