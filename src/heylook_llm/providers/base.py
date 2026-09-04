# src/heylook_llm/providers/base.py
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
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
    # Spec-decode acceptance, CUMULATIVE running totals for the request
    # (llama-server reports them on the final timings frame; MLX stamps the
    # running counters on every chunk). ChunkTelemetry latches the max.
    draft_tokens: int = 0
    draft_accepted: int = 0

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
    # The model's context window in tokens, from the ONE resolver the admin
    # row and /v1/models also read (capabilities.model_context_length), so
    # the number a client is shown is the number the provider enforces.
    # None = unknown (no guard). Set at load by providers that have one.
    context_length: Optional[int] = None

    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        self.config = config
        self.verbose = verbose
        # How many generations are in flight on this provider RIGHT NOW.
        #
        # Lives here, not in a subclass, because the router needs one question
        # it can ask any provider before tearing a model down -- and tearing a
        # model down mid-generation is unsafe in a DIFFERENT way per backend:
        # MLX frees weights under a live Metal command buffer, llama-server
        # gets SIGTERMed out from under an open HTTP stream. The count used to
        # exist only on MLXProvider, so a guard built on it silently covered
        # half the app -- and the uncovered half (gguf) was the one with no
        # protection of its own.
        self._active_generations = 0
        self._active_lock = threading.Lock()

    @property
    def active_generations(self) -> int:
        with self._active_lock:
            return self._active_generations

    @contextmanager
    def generation_active(self):
        """Count one in-flight generation for as long as the block runs.

        Wrap the GENERATOR BODY, not the call that returns it: providers hand
        back a generator, and what must be counted is the span during which it
        can still be pulled from. A generator's finally runs on exhaustion,
        on close(), and on an abort that closes it, so the count follows the
        real lifetime.
        """
        with self._active_lock:
            self._active_generations += 1
        try:
            yield
        finally:
            with self._active_lock:
                self._active_generations -= 1

    def template_info(self):
        """Chat-template metadata (ModelTemplateInfo) for reasoning-parser
        selection, or None when the provider owns templating/splitting
        itself -- None routes the parser stack to pass-through, which is
        exactly right for engines that pre-split reasoning (chunk.thinking).
        """
        return getattr(self, "_template_info", None)

    @property
    def thinking_capable(self) -> bool:
        """Whether this model can think at all -- the served ``thinking``
        capability, answered from what the provider holds. The cascade's
        last fallback for the thinking switch (v1.79.62), so every caller
        of ``resolve_effective_sampling`` on this provider must pass THIS,
        never re-derive it. gguf reads its config's ``supports_thinking``;
        MLX overrides with its template probe."""
        return bool(self.config.get("supports_thinking"))

    def render_prompt(self, request: ChatRequest) -> str:
        """The EXACT prompt string this provider would build for ``request``
        -- special tokens, role markers, thinking blocks and any open turn
        included -- without generating. What a "show me what the model sees"
        surface renders. Templating only: no forward pass, no gate. Raises
        ``GenerationFailed`` when the model is not loaded (templating needs
        the tokenizer or the live llama-server) and ``NotImplementedError``
        for a provider with no prompt (embeddings)."""
        raise NotImplementedError(f"{self.provider_name} has no chat prompt to render")

    def effective_thinking(self, request: ChatRequest) -> bool:
        """Whether THIS request's prompt is built with thinking on.

        The provider is the only thing that knows how it built the prompt, so
        it is the only honest answer to the question the reasoning parser has
        to ask before it can be armed. Callers must not re-derive it.

        This exists because they did. The parser used to resolve the flag
        itself from the RAW request while the prompt was templated from the
        CASCADE OUTPUT -- two readings of one decision, differing by the whole
        sampler layer. So `sampler="thinking"` (or a model whose
        `default_sampler` is thinking) built a thinking prompt and armed a
        content-state parser, and on a ``prefills_thinking`` template
        (Qwen3.5 pre-fills an unclosed ``<think>``) the model's output starts
        inside the block -- the entire reasoning trace lands in content. Same
        failure as v1.34.64, reachable again through a different door.

        Deriving from the shared cascade means the answer cannot disagree with
        what the provider actually sent: gguf's payload builder and MLX's
        template application both read the same resolved value. The vendor
        layer can never contribute here (VENDOR_SAMPLING_KEYS is numeric-only),
        so the base implementation needs no per-provider override.
        """
        from ..samplers import resolve_effective_sampling

        return bool(resolve_effective_sampling(
            request, self.config, thinking_capable=self.thinking_capable,
        ).get("enable_thinking"))

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
