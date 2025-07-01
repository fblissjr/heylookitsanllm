# src/providers/llama_cpp_provider.py
import gc, logging
from llama_cpp import Llama, Llava15ChatHandler
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from .base import BaseProvider

class LlamaCppProvider(BaseProvider):
    """Provider for running GGUF models via llama-cpp-python."""
    def load_model(self, config: dict, verbose: bool):
        logging.info(f"Loading GGUF model: {config['model_path']}")
        chat_handler = None
        if config.get('chat_format_template'):
            with open(config['chat_format_template'], 'r') as f: template = f.read()
            formatter = Jinja2ChatFormatter(template=template, eos_token=config.get('eos_token'), bos_token=config.get('bos_token'))
            chat_handler = formatter.to_chat_handler()
        elif config.get("mmproj_path"):
            chat_handler = Llava15ChatHandler(clip_model_path=config['mmproj_path'], verbose=verbose)

        self.model = Llama(
            model_path=config['model_path'], chat_handler=chat_handler, chat_format=config.get('chat_format'),
            n_ctx=config.get('n_ctx', 4096), n_gpu_layers=config.get('n_gpu_layers', -1),
            verbose=verbose,
        )

    def create_chat_completion(self, request: dict) -> Generator:
        # Why: Pass the entire request dictionary. `llama-cpp-python`'s `create_chat_completion`
        # is fully OpenAI-compatible and will automatically handle the `temperature`,
        # `top_p`, `repetition_penalty`, etc., parameters. This is the simplest and
        # most robust way to ensure feature parity.
        yield from self.model.create_chat_completion(**request)
