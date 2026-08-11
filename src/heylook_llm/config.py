# src/heylook_llm/config.py
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Default API port. Deliberately NOT 8080: that is llama-server's default,
# and llama.cpp-ecosystem clients (including its web UI) probe
# localhost:8080 with GET /props -- moving off it avoids the collision
# class entirely (v1.49.0). One source of truth for server.py argparse,
# service_manager install defaults, and the OpenAPI servers entry.
DEFAULT_PORT = 1263
from typing import List, Literal, Optional, Union, Dict

class ImageUrl(BaseModel):
    url: str

class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

class InputAudio(BaseModel):
    """OpenAI-wire audio payload. ``data`` is RAW base64 (no data: URI --
    llama-server rejects data URIs for audio); ``url`` is the llama-server
    extension for remote audio; exactly one of the two is required.
    ``format`` is advisory only (codecs are sniffed: WAV/MP3/FLAC)."""
    data: Optional[str] = None
    url: Optional[str] = None
    format: Optional[str] = None  # "wav" | "mp3" | ...

    @model_validator(mode='after')
    def require_data_or_url(self):
        if bool(self.data) == bool(self.url):
            raise ValueError("input_audio requires exactly one of 'data' (raw base64) or 'url'")
        return self

class AudioContentPart(BaseModel):
    """Audio input block (plan Phase 7d). Served ONLY by provider="gguf"
    (llama-server); the MLX provider rejects audio with a 400 -- its audio
    towers are skipped at load."""
    type: Literal["input_audio"]
    input_audio: InputAudio

ContentPart = Union[TextContentPart, ImageContentPart, AudioContentPart]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[ContentPart]]
    thinking: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.1, le=2.0)
    repetition_context_size: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    include_performance: bool = False
    seed: Optional[int] = None
    
    # Batch processing extensions
    processing_mode: Optional[str] = Field(default=None, description="conversation|sequential|sequential_with_context")
    return_individual: Optional[bool] = Field(default=None, description="Return individual responses vs combined")
    include_timing: Optional[bool] = Field(default=None, description="Include timing information")
    
    # Image resizing parameters (from multipart endpoint)
    resize_max: Optional[int] = Field(default=None, description="Resize images to max dimension (e.g., 512, 768, 1024)")
    resize_width: Optional[int] = Field(default=None, description="Resize images to specific width")
    resize_height: Optional[int] = Field(default=None, description="Resize images to specific height")
    image_quality: Optional[int] = Field(default=None, ge=1, le=100, description="JPEG quality for resized images")
    preserve_alpha: Optional[bool] = Field(default=None, description="Preserve alpha channel (outputs PNG)")

    # Thinking mode control (Qwen3 models)
    enable_thinking: Optional[bool] = Field(default=None, description="Enable thinking mode for Qwen3 models")

    # Visual token budget per image (model-agnostic; mapped onto the loaded
    # model's own processor knob -- gemma-4 buckets / qwen pixel budget)
    vision_tokens: Optional[int] = Field(default=None, ge=16, le=16384, description="Target visual tokens per image; snapped to what the model's processor supports")

    # Additional sampler parameters
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Reduce repetition (0-2, recommended 1.5 for Qwen3 thinking)")

    # Logprobs support (OpenAI-compatible)
    logprobs: Optional[bool] = Field(default=None, description="Return log probabilities for output tokens")
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20, description="Number of top tokens with log probabilities (0-20)")

    # Streaming options (OpenAI-compatible)
    stream_options: Optional[Dict] = Field(default=None, description="Options for streaming: {include_usage: true} to get usage stats")

    # Named sampler (resolved against the SamplerRegistry at generation
    # time). Overlays the sampler's fields on top of model-level defaults;
    # explicit request fields still win. Unknown name -> 400.
    # NOT the /v1/presets user presets (v3's saved prompt+sampler bundles).
    sampler: Optional[str] = Field(
        default=None,
        description="Named sampler (e.g. 'balanced', 'thinking', 'vlm-extract'). "
                    "Fields from the sampler overlay model defaults; explicit request "
                    "fields override both. Distinct from /v1/presets user presets."
    )

    @model_validator(mode='before')
    @classmethod
    def reject_renamed_preset_field(cls, data):
        # 2026-07-20 rename: 'preset' -> 'sampler'. ChatRequest ignores
        # unknown keys, so without this guard an old client's preset would
        # be silently dropped -- fail loudly with the migration hint instead.
        if isinstance(data, dict) and 'preset' in data:
            raise ValueError(
                "'preset' was renamed to 'sampler' (named sampler configs); "
                "/v1/presets user presets are a separate system"
            )
        return data

    @field_validator('messages', mode='before')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

class PerformanceMetrics(BaseModel):
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Dict
    performance: Optional[PerformanceMetrics] = None


class BatchChatRequest(BaseModel):
    """Request for batch chat completions."""
    requests: List[ChatRequest]
    processing_mode: str = "batch"

    # Batch-specific parameters
    completion_batch_size: Optional[int] = Field(default=32, description="Max concurrent generations")
    prefill_batch_size: Optional[int] = Field(default=8, description="Max prefill parallelism")
    prefill_step_size: Optional[int] = Field(default=2048, description="Chunk size for prefill")

    @field_validator('requests', mode='before')
    @classmethod
    def validate_requests(cls, v):
        if not v:
            raise ValueError("Requests list cannot be empty")
        if len(v) < 2:
            raise ValueError("Batch requests must contain at least 2 requests")
        return v


class BatchStats(BaseModel):
    """Statistics for batch processing."""
    total_requests: int
    elapsed_seconds: float
    throughput_req_per_sec: float
    throughput_tok_per_sec: float
    prefill_time: float
    generation_time: float
    memory_peak_mb: float


class BatchChatResponse(BaseModel):
    """Response for batch chat completions."""
    object: str = "list"
    data: List[ChatCompletionResponse]
    batch_stats: BatchStats

# ── Effect classification ───────────────────────────────────────────────────
# Every field on a provider config declares WHEN a change to it takes effect,
# as ``json_schema_extra={"effect": ...}``:
#
#   identity         Changes which model this entry IS. Not editable; that is
#                    a different entry.
#   requires_reload  Changes what the loaded process IS. Editable, but taking
#                    effect costs a teardown + respawn (gguf) or unload +
#                    reload (MLX) -- so the UI must confirm and name the cost.
#   load_time_only   Fixed for the life of the process and NOT recoverable by
#                    reloading this model (e.g. max_queue_depth is process-wide
#                    -- the first provider created wins). UI: disabled, with
#                    the reason. NB the bar is "a reload cannot fix it", NOT
#                    "it feels like plumbing": gguf's host/port/server_binary/
#                    startup_timeout_s were misfiled here on the second reading
#                    and are really requires_reload, because the router builds
#                    a NEW provider per load and they all go into the fresh
#                    spawn. The tell was that PATCHing server_binary reported
#                    "no reload required" while the subprocess kept running the
#                    old binary -- the exact regression this metadata exists to
#                    prevent.
#   applies_live     Re-read by the router while the model stays loaded; takes
#                    effect immediately, no reload, no ceremony. UI: freely
#                    editable. Deliberately distinct from load_time_only --
#                    "you cannot change this" and "change it, it just works"
#                    need opposite affordances, and one bucket cannot say both.
#   per_request      A model-level DEFAULT the loaded process can vary per
#                    request without changing what it is.
#   descriptive      Not a setting at all -- it describes the model (what we
#                    serve it as), and nothing about the PROCESS depends on it.
#                    NB "descriptive" is not "inert": `modalities` and
#                    `supports_thinking` feed capability inference, which gates
#                    v3's attach button and thinking toggle. Verified those are
#                    re-derived per read (effective_capabilities is called in
#                    the route handlers, not cached at load), so the change
#                    lands immediately and still needs no reload -- which is
#                    what keeps this class distinct from requires_reload.
#
# Field-local on purpose. Every drift this replaced (an MLX-shaped reload set
# that listed no gguf load-time field; an import allowlist that silently
# dropped five) existed because the fact lived somewhere other than the
# declaration it describes. ``tests/unit/test_config_effects.py`` fails if any
# field omits it, so a new field cannot be added without classifying it.
#
# ``arg`` alongside it is the llama-server spelling, for the gguf argv builder.
EFFECT_IDENTITY = "identity"
EFFECT_REQUIRES_RELOAD = "requires_reload"
EFFECT_LOAD_TIME_ONLY = "load_time_only"
EFFECT_APPLIES_LIVE = "applies_live"
EFFECT_PER_REQUEST = "per_request"
EFFECT_DESCRIPTIVE = "descriptive"

EFFECT_CLASSES: frozenset[str] = frozenset({
    EFFECT_IDENTITY, EFFECT_REQUIRES_RELOAD, EFFECT_LOAD_TIME_ONLY,
    EFFECT_APPLIES_LIVE, EFFECT_PER_REQUEST, EFFECT_DESCRIPTIVE,
})


def _extra(field) -> dict:
    """The json_schema_extra dict for a pydantic FieldInfo ({} when absent)."""
    extra = getattr(field, "json_schema_extra", None)
    return extra if isinstance(extra, dict) else {}


def field_effect(field) -> Optional[str]:
    """Declared effect class for one FieldInfo, or None if it declares none."""
    value = _extra(field).get("effect")
    return str(value) if value is not None else None


def fields_by_effect(cls: type) -> Dict[Optional[str], frozenset]:
    """{effect class -> field names} for a provider config class.

    A total partition of ``model_fields``. Fields with no effect -- or an
    effect that is not a KNOWN class -- land under ``None``.

    That second case is load-bearing. An earlier version bucketed by the raw
    string, so a one-character typo (``"requires-reload"`` with a hyphen) got
    its own bucket, left ``None`` empty, passed the completeness test, and
    dropped the field out of the reload set: exactly the silent "no reload
    required, keeps serving the old argv" bug this metadata replaced. An
    unrecognised effect is not a category, it is a mistake.
    """
    buckets: Dict[Optional[str], set] = {e: set() for e in EFFECT_CLASSES}
    buckets[None] = set()
    for name, field in cls.model_fields.items():
        effect = field_effect(field)
        buckets[effect if effect in EFFECT_CLASSES else None].add(name)
    return {k: frozenset(v) for k, v in buckets.items()}


def invalid_effects(cls: type) -> Dict[str, str]:
    """{field name -> the bogus effect string it declared}. Empty when clean."""
    return {
        name: str(field_effect(f))
        for name, f in cls.model_fields.items()
        if field_effect(f) is not None and field_effect(f) not in EFFECT_CLASSES
    }


def reload_required_fields(cls: type) -> frozenset:
    """Fields whose change needs a reload, DERIVED per provider config class.

    Replaces a single hand-written frozenset that was MLX-shaped and therefore
    wrong for gguf: changing ``ctx_size`` on a loaded gguf model reported "no
    reload required" and kept serving the old argv.

    Includes ``identity``: swapping the weights out from under a loaded model
    is the strongest form of "needs a reload", and the old hand-written set
    listed ``model_path`` for exactly that reason.
    """
    by = fields_by_effect(cls)
    return by.get(EFFECT_REQUIRES_RELOAD, frozenset()) | by.get(EFFECT_IDENTITY, frozenset())


def configurable_fields(cls: type) -> frozenset:
    """Everything an importer/editor may legitimately set: all but identity."""
    by = fields_by_effect(cls)
    return frozenset(
        name for effect, names in by.items()
        if effect != EFFECT_IDENTITY for name in names
    )


class MLXModelConfig(BaseModel):
    # Runtime-default fields (marked with ``is_runtime_default=True``) flow
    # from models.toml into each request's effective_request dict via
    # MLXProvider._apply_model_defaults. Adding a new one updates the
    # MLX_RUNTIME_DEFAULT_FIELDS set automatically -- no hardcoded list to
    # keep in sync.
    #
    # extra="forbid": a typo in models.toml (e.g. `temperatue`) must fail
    # loudly at load time, not silently revert to defaults.
    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(json_schema_extra={"effect": EFFECT_IDENTITY})
    draft_model_path: Optional[str] = Field(
        default=None, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # Classified requires_reload rather than per_request despite being a
    # runtime default: spec decode is set up when the draft model is loaded,
    # and the old hand-written reload set listed it. An unnecessary reload
    # prompt is a nuisance; a missed one silently serves stale behaviour.
    num_draft_tokens: Optional[int] = Field(
        default=3,
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD},
    )
    # DESCRIPTION vs ROUTING split (Phase 6 refinement 2026-07-11). ``vision``
    # historically did both jobs; it is now a derived mirror of
    # ``"vision" in modalities`` (kept for back-compat with readers of
    # config["vision"]). ``modalities`` is the author-declared capability set;
    # ``loader`` selects the mlx engine (within provider="mlx" only).
    vision: bool = Field(
        default=False, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # None = "not provided" -> derived from ``vision`` in _resolve_modalities.
    # Detected at import from the config's own blocks (vision_config/audio_config
    # + *_token_id); see model_importer.detect_modalities.
    # requires_reload here, DESCRIPTIVE on the gguf config: for MLX this feeds
    # effective_loader (mlx-vlm vs mlx-lm), so changing it changes which engine
    # holds the weights. Provider-aware classification is the point.
    modalities: Optional[List[str]] = Field(
        default=None, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # Engine routing. "auto": mlx-vlm if "vision" in modalities AND mlx-vlm
    # registers the model_type, else mlx-lm. Explicit values force the engine
    # (e.g. run a dual-capable VLM as text via "mlx-lm"). Resolution + the
    # effective loader live in the provider (is_vlm derives from it).
    loader: Literal["auto", "mlx-vlm", "mlx-lm"] = Field(
        default="auto", json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # Sampler defaults: the loaded model serves any of these per request.
    temperature: Optional[float] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    top_p: Optional[float] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    top_k: Optional[int] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    min_p: Optional[float] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    max_tokens: Optional[int] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    repetition_penalty: Optional[float] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    presence_penalty: Optional[float] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    # None = AUTO (6a derive-at-load): resolved at model load from actual
    # weight bytes vs RAM (cache_defaults.resolve_cache_config). A stored
    # value is an explicit operator override.
    cache_type: Optional[Literal["standard", "rotating", "quantized"]] = Field(
        default=None,
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD},
    )
    max_kv_size: Optional[int] = Field(
        default=None,
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD},
    )
    # MLX QuantizedKVCache supports exactly 2/4/8 bits and group sizes that
    # divide the head dim; anything else fails at first generation, so reject
    # it at config-load time instead.
    kv_bits: Optional[Literal[2, 4, 8]] = Field(
        default=None,
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD},
    )
    kv_group_size: Literal[32, 64, 128] = Field(
        default=64,
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD},
    )
    # In-flight + queued requests admitted before 503 backpressure. Consumed
    # by the generation gate (process-wide; the first provider created wins).
    # Process-wide once the gate exists (first provider created wins), so it
    # is infrastructure rather than a per-model tuning control.
    max_queue_depth: int = Field(
        default=8, ge=1,
        json_schema_extra={"effect": EFFECT_LOAD_TIME_ONLY,
                           "reason": "process-wide: the first provider created "
                                     "wins, so reloading this model cannot "
                                     "change it"})
    # Chunk size for prompt prefill. None lets mlx-lm use its default (2048).
    # Larger values reduce kernel-launch overhead on very long prompts at the
    # cost of higher peak memory during prefill.
    prefill_step_size: Optional[int] = Field(
        default=None, gt=0,
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_PER_REQUEST},
    )
    # Thinking mode (Qwen3 models with <think> blocks)
    enable_thinking: bool = Field(
        default=False, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    # Per-model default visual token budget per image (request vision_tokens
    # overrides; None = the processor's own default). Mapped per family by
    # providers/common/vision_budget.py.
    vision_tokens: Optional[int] = Field(
        default=None, ge=16, le=16384,
        json_schema_extra={"effect": EFFECT_PER_REQUEST})
    # Hidden states defaults (for /v1/hidden_states endpoint)
    # Kept requires_reload to match the old hand-written set rather than
    # quietly relaxing behaviour during a refactor.
    default_hidden_layer: int = Field(  # Z-Image uses penultimate layer
        default=-2, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    default_max_length: int = Field(
        default=512, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # NOTE: no supports_thinking here (removed v1.46.0) -- MLX thinking
    # capability is DERIVED (template probe / enable_thinking / the explicit
    # ModelConfig.capabilities override). GGUFModelConfig keeps its flag:
    # the template lives inside GGUF metadata, nothing cheap to probe.
    # Idle-unload override (C2). ``None`` = use ``AppConfig.idle_unload_seconds``
    # global default. ``0`` = never idle-unload this model. Positive = per-model
    # threshold in seconds. Pinned models are exempt regardless of this value.
    # applies_live, NOT load_time_only: the router re-reads this on each idle
    # sweep, so a change takes effect on a loaded model with no reload. The UI
    # should let it be edited freely -- the opposite of max_queue_depth above,
    # which no reload of THIS model can change.
    unload_after_idle_seconds: Optional[int] = Field(
        default=None, ge=0, json_schema_extra={"effect": EFFECT_APPLIES_LIVE})
    # Default sampler applied when a request doesn't specify ``sampler`` (C4).
    # Resolved against the SamplerRegistry at request time; an unknown name
    # falls back to "skip this layer" rather than raising -- the model config
    # is validated at server startup, so an unknown name here indicates a
    # post-startup registry rebuild drift and should log at the layer, not
    # kill inference.
    default_sampler: Optional[str] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    # Chat-template source policy (C4.5):
    # - "auto": trust HF AutoTokenizer.from_pretrained (jinja wins if present);
    #   if the tokenizer ends up template-less, the provider installs whatever
    #   template_info resolved (jinja > tokenizer_config > chat_template.json)
    # - "jinja": force-load chat_template.jinja from the model dir
    # - "tokenizer_config": force the template embedded in tokenizer_config.json
    # - "chat_template_json": force the processor-side chat_template.json
    # - absolute path: load that specific .jinja file
    # Useful when a model ships a broken jinja but a working embedded template,
    # or when the user wants to test a custom template without re-exporting.
    # Force-installed on the tokenizer at LOAD, so a change needs a reload.
    chat_template_source: Optional[str] = Field(
        default=None, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})

    @model_validator(mode="after")
    def _resolve_modalities(self):
        """Reconcile ``modalities`` <-> ``vision`` (modalities is authoritative).

        - ``modalities`` unset: DERIVE AT LOAD (6a, 2026-07-28) from the model
          dir's own config.json via the shared detector -- thin entries never
          materialize what the dir already declares. Falls back to the legacy
          ``vision`` bool when there is no config.json to read (fake paths in
          tests, HF repo ids), so old toml entries keep working.
        - ``modalities`` set: an explicit OVERRIDE -- normalize (``text``
          always first, deduped, order preserved) and sync ``vision`` to
          ``"vision" in modalities`` so a contradiction (``vision = true`` but
          modalities lacks it) resolves in favor of the declared list.
        """
        if self.modalities is None:
            detected = None
            try:
                from pathlib import Path

                from .modality_detect import detect_modalities, read_model_config_json
                model_dir = Path(self.model_path)
                cfg_json = read_model_config_json(model_dir)
                if cfg_json is not None:
                    detected = detect_modalities(model_dir, cfg_json)
            except Exception:
                detected = None  # detection must never block config load
            if detected is not None:
                self.modalities = detected
                self.vision = "vision" in detected
            else:
                self.modalities = ["text", "vision"] if self.vision else ["text"]
        else:
            # Normalize: "text" always first, order-preserving dedup.
            self.modalities = list(dict.fromkeys(["text", *self.modalities]))
            self.vision = "vision" in self.modalities
        return self

    @model_validator(mode="after")
    def _rotating_requires_max_kv_size(self):
        # Enforced here because cache_helpers.make_cache raises for this at
        # FIRST GENERATION -- a config that is guaranteed to fail must not
        # validate cleanly at load/import time.
        if self.cache_type == "rotating" and self.max_kv_size is None:
            raise ValueError("cache_type='rotating' requires max_kv_size")
        return self


# Derived at import time; callers of _apply_model_defaults read from this set
# rather than a hardcoded list. If you annotate a new field with
# is_runtime_default=True on MLXModelConfig above, it automatically flows into
# effective_request without touching mlx_provider.py.
# NB this OVERLAPS `effect` without contradicting it, and the two look like
# they should agree. They answer different questions:
#   is_runtime_default -> does this flow into effective_request per generation?
#   effect             -> when does CHANGING it in models.toml take effect?
# Five fields are is_runtime_default AND requires_reload (cache_type, kv_bits,
# kv_group_size, max_kv_size, num_draft_tokens). That is correct, not a drift:
# they ride the per-request plumbing (_build_cache_config reads them out of
# effective_request every generation), but no ChatRequest field can override
# them, and a loaded provider holds its own config copy -- so editing one in
# models.toml does nothing until the model is reloaded. Do not "reconcile"
# these two sets; reconciling them would make one of the questions
# unanswerable.
MLX_RUNTIME_DEFAULT_FIELDS: frozenset[str] = frozenset(
    # via _extra(): pydantic allows json_schema_extra to be a CALLABLE, and
    # `.get` on one raises at runtime, not just under a type checker. One
    # accessor for both this and the `effect` metadata.
    name for name, field in MLXModelConfig.model_fields.items()
    if _extra(field).get("is_runtime_default")
)

class MLXEmbeddingModelConfig(BaseModel):
    """Configuration for MLX embedding models."""
    # extra="forbid" like the other two provider configs. Without it pydantic
    # DEFAULTS to ignoring unknown keys, so `max_lenght = 4096` validated fine,
    # got written to models.toml, and was then dropped forever at load -- the
    # importer's own validator could not catch it because there was nothing to
    # catch. Silent, permanent loss of an operator's setting.
    model_config = ConfigDict(extra="forbid")
    # Local path or HF repo
    model_path: str = Field(json_schema_extra={"effect": EFFECT_IDENTITY})
    # Both are baked into the loaded backbone: max_length sizes the truncation
    # the encoder was loaded for, pooling selects the head. Changing either
    # under a live model would silently mismatch the weights in memory.
    max_length: int = Field(
        default=2048, json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    pooling: Literal["mean", "cls", "none"] = Field(
        default="mean", json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})

class GGUFModelConfig(BaseModel):
    """A GGUF model served by a llama-server SUBPROCESS (plan Phase 7).

    One entry = one servable model; MTP/draft artifacts are FIELDS here,
    never their own entries (embedded MTP -> just ``spec_type``; a sidecar
    drafter -> ``draft_model_path``; the same field the MLX config uses).
    llama-server owns tokenization, chat templating (GGUF-embedded jinja),
    and reasoning splitting -- the provider surfaces pre-split thinking via
    GenerationChunk.thinking and reports template_info() = None.
    """
    model_config = ConfigDict(extra="forbid")

    # path to the .gguf file
    model_path: str = Field(json_schema_extra={"effect": EFFECT_IDENTITY})
    # multimodal projector sidecar
    mmproj_path: Optional[str] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "--mmproj"},
    )
    # sidecar drafter (e.g. gemma mtp-*.gguf). `-md`, not the `--model-draft`
    # alias: `arg` must be the spelling the provider ACTUALLY emits, so a UI or
    # a derived emitter reproduces the real command line rather than an
    # equivalent-but-different one.
    draft_model_path: Optional[str] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-md"},
    )
    # llama-server --spec-type (e.g. "draft-mtp"). NB coupled to LoRA: a loaded
    # adapter erases spec decode's win, because the draft context never
    # receives the adapter (see CLAUDE.md's gguf gotchas). Leave it ON anyway.
    spec_type: Optional[str] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "--spec-type"},
    )
    spec_draft_n_max: Optional[int] = Field(
        default=None, ge=1, le=16,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-draft-n-max"},
    )
    ctx_size: Optional[int] = Field(
        default=None, ge=512,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "--ctx-size"},
    )
    # --spec-draft-p-min: minimum probability for a drafted token to be kept.
    # NOT a minor tuning knob -- it INTERACTS with spec_draft_n_max and the
    # interaction inverts, so a 1D n_max sweep finds a DIFFERENT and wrong
    # optimum. Tune the two together or not at all.
    #
    # HISTORY, not evidence. Both anomalies that suspended these were resolved
    # 2026-08-10 (the draft-memory warning is a harmless sizing pre-flight; the
    # 3.4x `--ctx` swing was a cold-vs-warm prompt-cache artifact), and the
    # clean re-run at VENDOR SAMPLING and a REALISTIC generation length found
    # spec on/off to be a WASH on this model (the apparent ~5% cost was
    # per-draft overhead amortised over a 48-token answer) -- so the table
    # below, which is all temp 0 and short, describes a regime nobody serves. The FIELD is still right to have (reachable, harmless, and the
    # only route to the short-prompt optimum); what it is worth is unknown.
    # EVERY NUMBER BELOW CARRIES ITS PROMPT LENGTH, because that turned out to
    # be the condition the whole result depends on -- and all of it is temp 0,
    # which is not a regime anyone serves. gemma-4 12B MTP:
    #
    #                             ~30-tok prompt   ~6k prompt
    #   spec off                  59.9             57.8
    #   shipped (n_max=3, p_min=0) 60.5  (+1.0%)   66.3  (+14.7%)
    #   n_max=2, p_min=0          63.9  (+6.7%)    63.2  (+9.3%)
    #   n_max=4, p_min=0.9        69.3  (+15.7%)   67.2  (+16.3%)
    #
    # Read it carefully. Repeated 4x per config at 6k: shipped mean 66.1,
    # tuned 66.3, overlapping ranges -- TUNING BUYS NOTHING MEASURABLE at
    # realistic context on this model. Not an unresolved measurement: the
    # configs are genuinely distinct (draft counts 32/51 vs 29/35), they just
    # arrive at the same speed. Tuning's +14.7-point edge is a SHORT-PROMPT
    # effect.
    # What moved is the SHIPPED config, not the tuned one -- the default is bad
    # on short prompts and fine on long ones. And the n_max-alone answer
    # INVERTS: n_max=2 is second-best short and the worst spec option at 6k,
    # below the default it was meant to improve on.
    # Draft volumes collapse with context (243/178/94 short vs 51/42/35 at 6k
    # for the same 200-token budget); the mechanism is not understood.
    #
    # So the FIELD is justified -- tuned wins or ties everywhere measured -- but
    # the size of the payoff was a property of a ~30-token prompt, not of the
    # model. Do not quote a single percentage without its context length.
    # No defensible global default either: p_min=0.9 helps both gemmas and is
    # -11% on Qwen3.6-27B at every value tested (5 samples). Per-model.
    spec_draft_p_min: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-draft-p-min"},
    )
    # Third member of the spec tuning family (--spec-draft-n-min): the floor
    # on how many draft tokens a speculation round keeps. Same caveat as its
    # siblings: the levers interact, tune together, per-model.
    spec_draft_n_min: Optional[int] = Field(
        default=None, ge=0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-draft-n-min"},
    )
    # Expert offload. On UNIFIED memory this does not shrink total RAM -- it
    # moves those bytes out of the Metal working set, and that math onto CPU
    # cores. The lever for a model that fits in RAM but crowds out the KV
    # cache. Consider -ctk/-ctv KV quantization first: for a headroom problem
    # it is usually the better trade and a smaller change.
    # NB `-ncmoe` past the model's layer count is a SILENT no-op, not an
    # error -- the block regexes simply never match.
    n_cpu_moe: Optional[int] = Field(
        default=None, ge=0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ncmoe",
                           "ui": "advanced"},
    )
    # All expert tensors to CPU. Equivalent to n_cpu_moe >= n_layer, kept
    # separate because it needs no layer count to express. A BARE flag.
    cpu_moe: Optional[bool] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-cmoe",
                           "shape": "flag", "ui": "advanced"},
    )
    # Raw tensor-buffer overrides: the unrestricted form of the two above. A
    # pattern that matches nothing is a silent no-op, so prefer n_cpu_moe.
    override_tensor: Optional[str] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ot",
                           "ui": "advanced"},
    )
    # Draft-side expert offload (-ncmoed/-cmoed): the drafter's half of the
    # residency budget. Exists for the target-offloaded + drafter-resident
    # split, where the pair exceeds the working set while either alone fits.
    n_cpu_moe_draft: Optional[int] = Field(
        default=None, ge=0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ncmoed",
                           "ui": "advanced"},
    )
    cpu_moe_draft: Optional[bool] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-cmoed",
                           "shape": "flag", "ui": "advanced"},
    )
    # KV cache quantization (-ctk/-ctv). For a headroom problem this is
    # usually the better first lever than expert offload: it shrinks the KV
    # bytes themselves instead of moving weight math to CPU. Values are the
    # cache types the pinned llama-server build accepts; quantized V-cache
    # needs flash attention, whose default is auto in current builds.
    cache_type_k: Optional[Literal[
        "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"
    ]] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ctk",
                           "ui": "advanced"},
    )
    cache_type_v: Optional[Literal[
        "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"
    ]] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ctv",
                           "ui": "advanced"},
    )
    # -ngl; 999 = everything on GPU
    n_gpu_layers: int = Field(
        default=999,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ngl"},
    )
    # Draft-model GPU offload (-ngld). Its own knob because the pair can exceed
    # the GPU budget when the target alone does not: on a 192 GiB M2 Ultra the
    # Metal residency recommendation is ~161 GiB, so a 144 GiB target plus a
    # 10 GiB drafter is over it while either alone is under. `0` keeps the
    # drafter off the GPU. None = inherit llama-server's own default.
    n_gpu_layers_draft: Optional[int] = Field(
        default=None, ge=0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ngld"},
    )
    # llama-server's OWN idle sleep (--sleep-idle-seconds): frees the model and
    # KV cache but KEEPS THE PROCESS, and reloads on the next task. Strictly
    # cheaper than heylook's idle-unload, which SIGTERMs and respawns -- so set
    # this BELOW the effective idle_unload_seconds and you get the cheap
    # recovery first and the expensive one only for a genuinely cold model.
    # None = disabled (llama-server's default).
    sleep_idle_seconds: Optional[int] = Field(
        default=None, ge=1,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--sleep-idle-seconds"},
    )
    # -cram: prompt-cache budget in MiB. llama-server defaults to only 8192;
    # -1 = unlimited, 0 = disable the cache entirely.
    cache_ram_mb: Optional[int] = Field(
        default=None, ge=-1,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-cram"},
    )
    # -lm: how weights are brought in. `mlock` pins them against paging, which
    # is the lever for a model near the memory ceiling; llama.cpp's Metal
    # residency set is separate and already on by default.
    load_mode: Optional[Literal["none", "mmap", "mlock", "mmap+mlock", "dio"]] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-lm"},
    )
    # else required via $HEYLOOK_LLAMA_SERVER
    server_binary: Optional[str] = Field(
        default=None,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    host: str = Field(
        default="127.0.0.1",
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # 0 = pick a free port at load
    port: int = Field(
        default=0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    startup_timeout_s: float = Field(
        default=300.0,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD})
    # Raw passthrough flags. requires_reload because they are spawn argv --
    # and note this is remote argv injection into a subprocess for anyone with
    # admin PATCH access, so a UI should not make it a casual free-text field
    # (ui:"advanced" keeps it behind v3's collapsed disclosure).
    extra_args: List[str] = Field(
        default_factory=list,
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "advanced"},
    )
    # named sampler (SamplerRegistry)
    default_sampler: Optional[str] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    # model-level default cap
    max_tokens: Optional[int] = Field(
        default=None, gt=0, json_schema_extra={"effect": EFFECT_PER_REQUEST})
    # Capability DESCRIPTION. Import fills this from the GGUF's own embedded
    # chat template (gguf_metadata.supports_thinking, same enable_thinking
    # rule the MLX path uses); it stays overridable by hand, and the explicit
    # ModelConfig.capabilities override short-circuits inference entirely.
    # None = no template to judge, e.g. an MTP/drafter head.
    supports_thinking: Optional[bool] = Field(
        default=None, json_schema_extra={"effect": EFFECT_DESCRIPTIVE})
    modalities: Optional[List[str]] = Field(
        default=None, json_schema_extra={"effect": EFFECT_DESCRIPTIVE})
    # Model-level thinking DEFAULT (the MLX config's counterpart), distinct
    # from `supports_thinking` above, which only describes CAPABILITY.
    # Required since unset started meaning OFF everywhere (v1.50.0): before
    # that, a gguf model inherited its template's own default -- thinking-ON
    # for gemma-4/Qwen3.6/DeepSeek-V4 -- and with extra="forbid" and no field
    # here there was then NO way to ask for that back. The only remaining
    # route was `default_sampler = "thinking"`, which drags a presence_penalty
    # change in with it. None = unset = off.
    enable_thinking: Optional[bool] = Field(
        default=None, json_schema_extra={"effect": EFFECT_PER_REQUEST})


# Single source of truth for which providers exist and which config class
# validates each one's `config` block. Adding a provider = add an entry here
# + widen the Literal below + register the provider class in router.py's
# provider_map (which must stay in key-sync with this dict).
PROVIDER_CONFIG_CLASSES: Dict[str, type] = {
    "mlx": MLXModelConfig,
    "mlx_embedding": MLXEmbeddingModelConfig,
    "gguf": GGUFModelConfig,
}


def _validate_effect_declarations() -> None:
    """Fail at IMPORT if any provider-config field is unclassified or misspelt.

    A test would catch this too, but only when the suite runs. A misspelt
    effect is indistinguishable from a correct one at a glance and degrades
    silently in the safe-looking direction (the field simply stops being
    reload-required), so it has to be impossible to run the server with one.

    Why raising at import is proportionate here and would NOT be elsewhere:
    the input is developer-authored STATIC data. A bad classification is a code
    bug that surfaces on the first import -- in dev, in CI, on any startup
    while someone is working -- and can never be provoked by user data in
    production. Fail-fast on static developer data is cheap and correct. The
    same guard applied to runtime user input would be hostile, because then a
    bad input takes down a running server. Read the distinction before
    "fixing" this into a warning.
    """
    problems: List[str] = []
    for provider, cls in PROVIDER_CONFIG_CLASSES.items():
        for name, bad in invalid_effects(cls).items():
            problems.append(
                f"  {provider}.{name}: effect={bad!r} is not one of "
                f"{sorted(EFFECT_CLASSES)}"
            )
        missing = sorted(fields_by_effect(cls).get(None, frozenset())
                         - set(invalid_effects(cls)))
        for name in missing:
            problems.append(f"  {provider}.{name}: no `effect` declared")
    if problems:
        raise RuntimeError(
            "Provider config fields must declare when a change takes effect "
            'via json_schema_extra={"effect": ...}:\n' + "\n".join(problems)
        )


_validate_effect_declarations()


class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "mlx_embedding", "gguf"]
    config: Union[MLXModelConfig, MLXEmbeddingModelConfig, GGUFModelConfig]
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True
    # Model capabilities for discovery (e.g., ["hidden_states", "chat", "thinking", "vision"])
    capabilities: List[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def validate_config_type(cls, data):
        if isinstance(data, dict):
            provider = data.get('provider')
            v = data.get('config')
            if isinstance(v, dict):
                config_cls = PROVIDER_CONFIG_CLASSES.get(provider) if provider else None
                if config_cls is None:
                    raise ValueError(f"Unknown provider '{provider}' for model config validation")
                data['config'] = config_cls(**v)
        return data

class ScanConfig(BaseModel):
    """Watch-folder config for periodic model discovery (C3).

    Discovery is passive: ``MemoryManager.tick()`` rescans the configured
    folders and optionally the HuggingFace cache every
    ``scan_interval_seconds``. Discovered-but-not-imported models are
    surfaced via ``GET /v1/admin/models/discovered``. There is NO
    auto-import -- the frontend Models page (C5) presents an Add button
    that hits the existing ``POST /v1/admin/models/import`` path.
    """
    folders: List[str] = Field(default_factory=list)
    watch_hf_cache: bool = False
    scan_interval_seconds: int = Field(
        900, ge=0,
        description="Seconds between rescans. 0 disables periodic rescans "
                    "(no initial scan either).",
    )


class AppConfig(BaseModel):
    models: List[ModelConfig]
    default_model: Optional[str] = None

    @field_validator('default_model', mode='before')
    @classmethod
    def _blank_default_model_is_unset(cls, v):
        """Coerce the placeholder spellings of "no default" to None.

        `model_importer`/`model_service` write the literal string ``"none"``
        when a scan finds no models, and ``""`` shows up in hand-edited
        configs. Both are TRUTHY, so without this they sail past every
        ``if default_model:`` check and get routed to as a real model id.
        """
        if isinstance(v, str) and v.strip().lower() in ("", "none"):
            return None
        return v
    scan: Optional[ScanConfig] = None
    # Default is 1 (single-model) -- Apple Silicon is memory-bandwidth-bound,
    # so a second loaded-but-idle model doesn't help throughput. Field stays
    # configurable for setups that truly need hot-swap without reload.
    max_loaded_models: int = Field(default=1, ge=1)

    # Idle unload (C2). Global default applied when a model has no per-model
    # ``unload_after_idle_seconds`` override. ``0`` disables idle unload
    # entirely (for models without their own override). Pinned models are
    # always exempt.
    idle_unload_seconds: int = Field(
        1800, ge=0,
        description="Seconds of inactivity before a non-pinned model is unloaded. "
                    "0 disables idle unload globally.",
    )

    # Observability (S1.2). Env-var overrides live in memory.py:
    # HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS, HEYLOOK_REQUEST_LOG_ENABLED,
    # HEYLOOK_MODEL_EVENT_LOG_ENABLED.
    baseline_log_interval_seconds: int = Field(
        3600, ge=0,
        description="Seconds between memory_baseline.jsonl entries. 0 disables.",
    )
    request_log_enabled: bool = Field(
        True, description="Append per-request event to request_events.jsonl."
    )
    model_event_log_enabled: bool = Field(
        True, description="Append model load/unload events to model_events.jsonl."
    )

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.id == model_id and m.enabled), None)

    def get_enabled_models(self) -> List[ModelConfig]:
        return [m for m in self.models if m.enabled]


# =============================================================================
# System Metrics Models
# =============================================================================

class SystemResourceMetrics(BaseModel):
    """System-wide resource metrics (RAM, CPU)."""
    ram_used_gb: float = Field(..., description="RAM currently used in GB")
    ram_available_gb: float = Field(..., description="RAM available in GB")
    ram_total_gb: float = Field(..., description="Total system RAM in GB")
    cpu_percent: float = Field(..., description="CPU usage percentage")


class ModelMetrics(BaseModel):
    """Per-model metrics (context usage, memory)."""
    context_used: int = Field(..., description="Tokens currently in context")
    context_capacity: int = Field(..., description="Maximum context window size")
    context_percent: float = Field(..., description="Context usage percentage")
    memory_mb: float = Field(..., description="Model memory usage in MB")
    requests_active: int = Field(default=0, description="Active requests for this model")
    requests_queued: int = Field(default=0, description="Requests waiting in the FIFO generation queue behind the active one")


class SystemMetricsResponse(BaseModel):
    """Response for GET /v1/system/metrics endpoint."""
    timestamp: str = Field(..., description="ISO timestamp of metrics collection")
    system: SystemResourceMetrics
    models: Dict[str, ModelMetrics] = Field(default_factory=dict, description="Metrics per loaded model")


# =============================================================================
# Cache Management Models
# =============================================================================

class CacheInfo(BaseModel):
    """Information about a saved prompt cache."""
    cache_id: str = Field(..., description="Unique cache identifier")
    model: str = Field(..., description="Model ID this cache belongs to")
    name: str = Field(..., description="User-friendly cache name")
    description: Optional[str] = Field(default=None, description="Optional description")
    tokens_cached: int = Field(..., description="Number of tokens in cache")
    size_mb: float = Field(..., description="Cache file size in MB")
    created_at: str = Field(..., description="ISO timestamp of creation")


class CacheListResponse(BaseModel):
    """Response for listing saved caches."""
    caches: List[CacheInfo] = Field(default_factory=list)


class CacheClearRequest(BaseModel):
    """Request to clear caches."""
    model: Optional[str] = Field(default=None, description="Model ID to clear caches for (all if omitted)")


class CacheClearResponse(BaseModel):
    """Response from cache clear operation."""
    deleted_count: int


# =============================================================================
# Enhanced Streaming Metadata
# Schema documentation for stream_options.include_usage response fields.
# These models document the API contract; actual streaming uses dicts directly.
# =============================================================================

class EnhancedUsage(BaseModel):
    """Extended usage statistics including thinking tokens."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: Optional[int] = Field(default=None, description="Tokens used in thinking blocks")
    content_tokens: Optional[int] = Field(default=None, description="Tokens in actual content")
    total_tokens: int = 0


class GenerationTiming(BaseModel):
    """Timing breakdown for generation phases."""
    thinking_duration_ms: Optional[int] = Field(default=None, description="Time spent in thinking phase")
    content_duration_ms: Optional[int] = Field(default=None, description="Time spent generating content")
    total_duration_ms: int = Field(..., description="Total generation time")
    peak_memory_gb: Optional[float] = Field(
        None,
        description="Peak MLX memory used during this request in GB. Streaming only: appears in the final usage chunk when stream_options.include_usage=true.",
    )
    kv_cache_bytes: Optional[int] = Field(
        None,
        description="Bytes held in the prompt KV cache at the start of this request. Streaming only: appears in the final usage chunk when stream_options.include_usage=true.",
    )


class GenerationConfig(BaseModel):
    """Sampler configuration used for generation."""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    enable_thinking: Optional[bool] = None
    max_tokens: Optional[int] = None


# =============================================================================
# SSE Stream Chunk Models
# These document the Server-Sent Events payload for streaming responses.
# =============================================================================

class TopLogprobEntry(BaseModel):
    """A candidate token with its log probability (used in top_logprobs arrays)."""
    token: str = Field(..., description="Token text")
    token_id: int = Field(..., description="Token vocabulary ID")
    logprob: float = Field(..., description="Log probability of this token")
    bytes: List[int] = Field(default_factory=list, description="UTF-8 byte values")

class TokenLogprobInfo(BaseModel):
    """Token with its log probability and alternative candidates."""
    token: str = Field(..., description="Token text")
    token_id: int = Field(..., description="Token vocabulary ID")
    logprob: float = Field(..., description="Log probability of this token")
    bytes: List[int] = Field(default_factory=list, description="UTF-8 byte values")
    top_logprobs: Optional[List[TopLogprobEntry]] = Field(
        None, description="Alternative tokens with their logprobs"
    )

class StreamLogprobs(BaseModel):
    """Logprobs attached to a streaming chunk."""
    content: List[TokenLogprobInfo] = Field(
        default_factory=list, description="Token-level logprob data for this chunk"
    )

class StreamDelta(BaseModel):
    """Delta content in a streaming chunk."""
    role: Optional[str] = Field(default=None, description="Role (only in first chunk)")
    content: Optional[str] = Field(default=None, description="Text content delta")
    thinking: Optional[str] = Field(default=None, description="Thinking content delta")

class StreamChoice(BaseModel):
    """Single choice in a streaming chunk."""
    index: int = 0
    delta: StreamDelta = Field(default_factory=StreamDelta)
    logprobs: Optional[StreamLogprobs] = None
    finish_reason: Optional[str] = Field(
        None, description="'stop', 'length', or null while streaming"
    )

class StreamChunk(BaseModel):
    """SSE payload for a single streaming chunk (data: {...}).

    Sent as Server-Sent Events on the /v1/chat/completions endpoint
    when stream=true. The final chunk includes usage, timing, and
    generation_config when stream_options.include_usage=true.
    """
    id: str = Field(..., description="Response identifier (chatcmpl-...)")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model ID used for generation")
    choices: List[StreamChoice]
    usage: Optional[EnhancedUsage] = Field(
        None, description="Token usage (final chunk only, requires stream_options.include_usage)"
    )
    timing: Optional[GenerationTiming] = Field(
        None, description="Generation timing breakdown (final chunk only)"
    )
    generation_config: Optional[GenerationConfig] = Field(
        None, description="Sampler settings used (final chunk only)"
    )
    stop_reason: Optional[str] = Field(
        None, description="Why generation stopped (final chunk only)"
    )


# =============================================================================
# Admin API Models (Model Management)
# =============================================================================

class ScannedModelResponse(BaseModel):
    """A model discovered during filesystem scan.

    Mirrors ``model_service.ScannedModel``. Wired as the /scan route's
    ``response_model`` on purpose: this model sat unreferenced for months
    while the dataclass grew, so the declared contract and what the route
    actually returned had no way to disagree loudly.
    """
    id: str = Field(..., description="Auto-generated model identifier")
    path: str = Field(..., description="Filesystem path to model")
    provider: Literal["mlx", "mlx_embedding", "gguf"] = Field(..., description="Detected provider type")
    size_gb: float = Field(..., description="Estimated model size in GB")
    vision: bool = Field(default=False, description="Whether model supports vision (shadow of `modalities`)")
    quantization: Optional[str] = Field(default=None, description="Quantization level (4bit, 8bit, etc)")
    already_configured: bool = Field(default=False, description="True if ID already exists in models.toml")
    tags: List[str] = Field(default_factory=list)
    description: str = ""
    modalities: List[str] = Field(
        default_factory=list,
        description="Author-declared modality set (text/vision/audio/video)",
    )
    supports_thinking: Optional[bool] = Field(
        default=None,
        description="Thinking support read from the model's own chat template; "
                    "null = no template to judge (e.g. a drafter head)",
    )
    draft_model_path: Optional[str] = Field(
        default=None, description="Paired speculative drafter sidecar, if any"
    )
    draft_spec_type: Optional[str] = Field(
        default=None,
        description="The --spec-type that drafter REQUIRES. Reported, never applied: "
                    "import leaves spec_type unset because whether speculative "
                    "decoding pays off is a per-model measurement.",
    )


class ScannedModelListResponse(BaseModel):
    """Response for a model scan."""
    models: List[ScannedModelResponse] = Field(default_factory=list)
    total: int = 0


class ModelScanRequest(BaseModel):
    """Request to scan for importable models."""
    paths: List[str] = Field(default_factory=list, description="Custom paths to scan")
    scan_hf_cache: bool = Field(default=True, description="Also scan HuggingFace cache directories")


class ModelImportRequest(BaseModel):
    """Import one or more scanned models.

    extra="forbid": the 2026-07-20 rename (profile -> default_sampler) must
    fail loudly for old clients -- with Pydantic's default extra=ignore, a
    stale {"profile": ...} body would be silently dropped and every import
    stamped with the "balanced" default.
    """
    model_config = ConfigDict(extra="forbid")

    models: List[Dict] = Field(..., description="Models to import (id, path, provider, overrides)")
    default_sampler: Optional[str] = Field(default="balanced", description="Named sampler recorded as default_sampler on all imported models")


class ModelUpdateRequest(BaseModel):
    """Partial update to model config.

    extra="forbid": a config key sent at the TOP level (`{"ctx_size": ...}`
    instead of `{"config": {"ctx_size": ...}}`) used to validate, get ignored
    by the fixed top-level key list, and return 200 with nothing changed --
    the silent-drop class this repo rejects elsewhere (import, the
    preset->sampler guard). Now it 422s naming the key.
    """
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = None
    tags: Optional[List[str]] = None
    enabled: Optional[bool] = None
    capabilities: Optional[List[str]] = None
    config: Optional[Dict] = Field(default=None, description="Provider-specific config updates")


class ModelValidateRequest(BaseModel):
    """Validate a model config without saving."""
    id: str
    provider: str
    config: Dict
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True


class AdminValidationResult(BaseModel):
    """Result of config validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class BulkDefaultSamplerRequest(BaseModel):
    """Set default_sampler (a named-sampler name) on multiple models."""
    model_ids: List[str] = Field(..., description="Model IDs to update")
    sampler: str = Field(..., description="Named sampler to record as default_sampler")


class ModelStatusResponse(BaseModel):
    """Runtime status of a model."""
    loaded: bool = Field(..., description="Whether model is currently in LRU cache")
    memory_mb: Optional[float] = Field(default=None, description="Memory usage in MB (if loaded)")
    context_used: Optional[int] = Field(default=None, description="Tokens currently in context")
    context_capacity: Optional[int] = Field(default=None, description="Maximum context window")
    requests_active: Optional[int] = Field(default=None, description="Active requests for this model")


class AdminModelResponse(BaseModel):
    """Full model config for admin API responses."""
    id: str
    provider: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True
    capabilities: List[str] = Field(default_factory=list)
    config: Dict = Field(default_factory=dict)
    loaded: bool = False


class AdminModelListResponse(BaseModel):
    """Response for listing all model configs."""
    models: List[AdminModelResponse] = Field(default_factory=list)
    total: int = 0


class SamplerInfo(BaseModel):
    """Named-sampler metadata (bundled registry entry)."""
    name: str
    description: str


class SamplerListResponse(BaseModel):
    """Response for listing available named samplers."""
    samplers: List[SamplerInfo] = Field(default_factory=list)


