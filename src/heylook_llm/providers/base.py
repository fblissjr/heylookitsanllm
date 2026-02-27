# src/heylook_llm/providers/base.py
from abc import ABC, abstractmethod
from typing import Generator, Dict, Optional
from ..config import ChatRequest, ModelMetrics

class BaseProvider(ABC):
    """Abstract base class for all model backends."""
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        self.config = config
        self.verbose = verbose

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def create_chat_completion(self, request: ChatRequest) -> Generator:
        raise NotImplementedError

    def get_metrics(self) -> Optional[ModelMetrics]:
        """
        Get current metrics for this model (context usage, memory, etc.).

        Returns:
            ModelMetrics if available, None if not supported by this provider.
        """
        return None

    def clear_cache(self) -> bool:
        """
        Clear any prompt/KV cache for this model.

        This is called when the context should be fully invalidated
        (e.g., explicit user request, major prompt structure change).

        Returns:
            True if cache was cleared, False if not supported or no cache exists.
        """
        return False

    def get_tokenizer(self):
        """Return the tokenizer, or None if unavailable."""
        processor = getattr(self, 'processor', None)
        if processor is None:
            return None
        if hasattr(processor, '_tokenizer'):
            return processor._tokenizer
        if hasattr(processor, 'tokenizer'):
            return processor.tokenizer
        return processor if hasattr(processor, 'decode') else None

    def unload(self):
        """Optional method to explicitly release resources."""
        pass

    def __del__(self):
        self.unload()
