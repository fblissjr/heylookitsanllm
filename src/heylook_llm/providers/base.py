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

    def check_capacity(self) -> None:
        """Raise if the provider is too busy to accept another request.

        Called by HTTP entry points *before* starting generation so an
        overloaded provider can reject early with backpressure (HTTP 503)
        instead of letting the queue grow without bound. Default is a no-op
        (no admission limit). Providers that serialize generation (e.g. MLX)
        override this to raise ``ModelBusyError`` when their queue is full.
        Internal orchestration (batch, RLM) intentionally skips this and queues.
        """

    def generation_queue_stats(self) -> Optional[Dict]:
        """Return a snapshot of the generation queue, or None if not serialized.

        Providers that gate generation (e.g. MLX) return a dict with
        ``active`` / ``waiting`` / ``max_waiting`` / ``capacity``. Used for 503
        backpressure headers and observability. Default None (no queue).
        """
        return None

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

    def warmup(self) -> None:
        """Prime JIT caches so the first real request is fast.

        Default is a no-op. Providers that benefit (e.g. MLX models that
        JIT-compile Metal shaders per shape bucket) should override.

        Contract: implementations MUST swallow exceptions and log rather
        than propagate. Warmup is an optimization, not a correctness
        requirement; a warmup hiccup must never prevent the router from
        returning a usable provider. Callers may rely on this and omit
        their own try/except wrapper.
        """

    def __del__(self):
        self.unload()
