# src/heylook_llm/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, Dict, Optional
from ..config import ChatRequest, ModelMetrics


@dataclass(slots=True)
class GenerationChunk:
    """The one chunk type providers yield -- heylook-owned, engine-neutral.

    Replaces the de-facto contract of duck-typing mlx-lm's GenerationResponse
    (plus three runtime-patched attrs). slots=True is deliberate: attaching
    undeclared attributes was the old extension mechanism and must now fail
    loudly -- new telemetry gets a FIELD here, consumed in
    perf_collector.ChunkTelemetry.absorb(), not an attr-patch at a call site.

    ``thinking`` carries PRE-SPLIT reasoning from engines that separate it
    before it reaches us (llama-server's reasoning_content). Consumers route
    it straight to the thinking channel; the reasoning parsers only ever see
    ``text``. MLX providers never set it (their reasoning arrives inline in
    ``text`` and is split by the parser stack).

    ``logprobs`` stays engine-native (mx.array on MLX; top-n dicts elsewhere);
    logprobs.py owns the per-engine conversion. Errors are NOT chunks --
    providers raise GenerationFailed (see below), so there is no error flag.
    """

    text: str = ""
    token: Optional[int] = None
    logprobs: Any = None
    thinking: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    cached_tokens: int = 0
    kv_cache_bytes: int = 0
    queue_wait_ms: float = 0.0
    from_draft: bool = False

    @classmethod
    def from_engine(cls, r: Any) -> "GenerationChunk":
        """Duck-convert an engine chunk (mlx-lm GenerationResponse, mlx-vlm
        diffusion chunk) -- absent attributes take the field defaults."""
        return cls(
            text=getattr(r, "text", "") or "",
            token=getattr(r, "token", None),
            logprobs=getattr(r, "logprobs", None),
            finish_reason=getattr(r, "finish_reason", None),
            prompt_tokens=getattr(r, "prompt_tokens", 0) or 0,
            generation_tokens=getattr(r, "generation_tokens", 0) or 0,
            prompt_tps=getattr(r, "prompt_tps", 0.0) or 0.0,
            generation_tps=getattr(r, "generation_tps", 0.0) or 0.0,
            peak_memory=getattr(r, "peak_memory", 0.0) or 0.0,
            from_draft=bool(getattr(r, "from_draft", False)),
        )


class GenerationFailed(RuntimeError):
    """Generation could not complete; the message is safe to show the client.

    RAISED by provider generators (mid-iteration) instead of yielding error
    text as chunks -- so every consumer, including ones written later, fails
    loudly by default rather than silently concatenating error text into
    results (the bug that had RLM reasoning over "Error: MLX generation
    failed..." as if the model said it). API routes translate: HTTP 500
    non-streaming, an SSE error payload when headers are already out.
    """


class InvalidGenerationRequest(GenerationFailed):
    """The CLIENT's request can never succeed on this model (e.g. images sent
    to a text-only model). Routes translate to HTTP 400, not 500. Subclasses
    GenerationFailed so consumers may catch the base class alone."""


class BaseProvider(ABC):
    """Abstract base class for all model backends."""

    # Registry name of this backend ("mlx", "mlx_embedding", ...). A CLASS
    # attribute so neutral code (router teardown, telemetry's dim_model)
    # never has to sniff type names -- the pre-7a state was a `provider`
    # attr nobody set, which made the router's MLX cache-clear dead code.
    provider_name: str = ""

    # Neutral capability defaults. MLX overrides these per-instance after
    # load; other providers keep the defaults unless they have the concept.
    is_vlm: bool = False
    effective_loader: Optional[str] = None

    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        self.config = config
        self.verbose = verbose

    def template_info(self):
        """Chat-template metadata (ModelTemplateInfo) for reasoning-parser
        selection, or None when the provider owns templating/splitting
        itself -- None routes the parser stack to pass-through, which is
        exactly right for engines that pre-split reasoning (chunk.thinking).
        """
        return getattr(self, "_template_info", None)

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def create_chat_completion(self, request: ChatRequest, abort_event=None) -> Generator:
        """Generate a completion. ``abort_event`` is an optional per-request
        cooperative cancel signal (set by the streaming layer on disconnect);
        implementations that don't support cancellation may ignore it."""
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
