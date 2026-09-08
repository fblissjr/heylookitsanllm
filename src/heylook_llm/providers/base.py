# src/heylook_llm/providers/base.py
import inspect
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

    Errors are NOT chunks -- providers raise GenerationFailed (see below),
    so there is no error flag.
    """

    text: str = ""
    token: Optional[int] = None
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
    # running counters on every chunk). How a consumer folds them across
    # chunks is perf_collector.ChunkTelemetry.absorb()'s call, not this
    # field's -- see the rule stated there.
    draft_tokens: int = 0
    draft_accepted: int = 0

    @classmethod
    def from_engine(cls, r: Any) -> "GenerationChunk":
        """Duck-convert an engine chunk (mlx-lm GenerationResponse, mlx-vlm
        diffusion chunk) -- absent attributes take the field defaults."""
        return cls(
            text=getattr(r, "text", "") or "",
            token=getattr(r, "token", None),
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

    # Does render_prompt() represent MEDIA in the string it returns? gguf
    # does (llama-server rewrites an image part into a positional media
    # marker, which is part of the render); the MLX text strategy strips
    # images entirely, so its preview is the text template alone. A caller
    # showing that string to a human has to SAY which it got -- a preview
    # that quietly omits the picture reads as "no image will be sent",
    # which is the opposite of the truth. Asked of the provider rather than
    # switched on a provider NAME in the route, so a new backend answers
    # for itself instead of defaulting into a wrong claim.
    render_prompt_represents_media: bool = False

    # The chat template body this provider actually LOADED WITH, set at load
    # and never after. Both engines bind the template at load (MLX installs it
    # on the tokenizer, gguf passes a file at spawn), so editing the file on
    # disk changes nothing until a reload -- and the config-level
    # `stale_reload_fields` cannot see it, because no config field moved.
    # This is the file-backed equivalent of that signal, and it is what lets
    # the template editor say "saved, reload to apply" instead of leaving the
    # reader to wonder why nothing changed. None means "not known", which is
    # every unloaded model.
    loaded_chat_template: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """Refuse, at class creation, an `unload` that cannot take `drain`.

        The contract below says an override MUST accept it. Prose did not
        hold: a subclass declaring plain `unload(self)` raises TypeError
        inside `__del__`, where Python SWALLOWS it, prints one "Exception
        ignored while calling deallocator" line and DOES NOT UNLOAD. Two test
        doubles sat in that state for a session with every suite green,
        because nothing anywhere could go red on it. This is the derive-
        rather-than-document rule the repo applies to its constant lists,
        pointed at a signature: the wrong thing now fails to import.
        """
        super().__init_subclass__(**kwargs)
        unload = cls.__dict__.get("unload")
        if unload is None:
            return  # inherits a conforming one
        params = inspect.signature(unload).parameters
        if "drain" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            return
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__}.unload must accept `drain` "
            "(`def unload(self, *, drain: bool = True)`) -- BaseProvider.__del__ "
            "passes it, and a signature that cannot take it makes the destructor "
            "raise a TypeError Python swallows, silently never unloading."
        )

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
        sampler layer. A model whose config turned thinking on therefore built
        a thinking prompt and armed a content-state parser, and on a
        ``prefills_thinking`` template
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

    def unload(self, *, drain: bool = True):
        """Optional method to explicitly release resources.

        `drain` is the CALLER saying whether it can afford to wait for
        in-flight work. Every deliberate teardown can, and must: releasing
        weights mid-decode faults Metal. `__del__` cannot -- see below.
        Implementations with nothing to wait for accept the argument and
        ignore it.

        AN OVERRIDE MUST ACCEPT ``drain`` -- enforced by
        ``__init_subclass__``, not by this paragraph, because the failure it
        describes is invisible: a subclass declaring plain ``unload(self)``
        raises TypeError inside the destructor, where Python swallows it,
        prints one ``Exception ignored while calling deallocator`` line to
        stderr, and DOES NOT UNLOAD. Two test doubles sat in that state for a
        session with every suite green. It is now a class-creation error.
        """
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
        # A DESTRUCTOR MUST NOT WAIT. This ran the full teardown, drain loop
        # and all, and MLXProvider's loop polls for up to 30s -- so a provider
        # collected while its active counter was non-zero stalled whatever
        # thread the GC happened to fire on. It was measured doing exactly
        # that in the test suite (from a counter one test forgot to reset),
        # but the hazard is here, not there: in the server, GC fires on any
        # thread, including one delivering tokens.
        #
        # `drain=False` means ONLY "do not wait, and do not reach for the
        # engine on the way out". It CANNOT mean "keep the model loaded":
        # RETURNING EARLY FROM A DESTRUCTOR RETAINS NOTHING. `__del__` runs
        # during deallocation, so the weights are released when it returns,
        # whatever it decided -- measured 2026-09-08, against the claim this
        # comment used to make. What skipping buys is narrower and real: no
        # 30s poll, and no engine teardown call (`gc.collect()` +
        # `mx.clear_cache()`) issued from a thread we did not choose while
        # another may be mid-decode.
        #
        # So a provider collected mid-generation is a bug the destructor
        # CANNOT make safe -- it can only decline to make it worse, and say
        # so. A running generation holds a reference to its provider, so
        # reaching that branch at all means one was dropped.
        self.unload(drain=False)
