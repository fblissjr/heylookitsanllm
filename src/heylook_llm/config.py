# src/heylook_llm/config.py
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Default API port. 8000 is the OpenAI-compatible-server convention
# (uvicorn/FastAPI/vLLM), chosen for familiarity over uniqueness -- it CAN
# collide with other dev servers on the same box (`--port` is the escape).
# Still deliberately NOT 8080: that is llama-server's own default, and
# llama.cpp-ecosystem clients (including its web UI) probe localhost:8080
# with GET /props. One source of truth for server.py argparse,
# service_manager install defaults, and the OpenAPI servers entry.
DEFAULT_PORT = 8000
from typing import Any, ClassVar, List, Literal, Optional, Union, Dict

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

# Thinking-depth vocabulary, declared ONCE. The union of the families that
# read it: Qwen3.8 takes xhigh|medium|low, harmony takes low|medium|high. It
# WILL grow (vendors keep inventing spellings), and three separate Literals
# would let the request schema and the model-config schema drift apart -- a
# value requests accept and PATCH 422s on.
ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


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
    seed: Optional[int] = None
    # The batch-processing knobs (processing_mode & co.) and the server-side
    # image-resize knobs (resize_max & co.) left with the OpenAI chat route
    # (v1.79.66): nothing that remains read them, and /v1/messages has no
    # resize by design -- clients downscale before sending.

    # Thinking mode control (Qwen3 models)
    enable_thinking: Optional[bool] = Field(default=None, description="Enable thinking mode for Qwen3 models")
    # Depth of thinking, when thinking is on. A chat-template variable like
    # enable_thinking, not a sampler knob -- the template turns it into a
    # system-prompt instruction (Qwen3.8) or a harmony channel setting.
    #
    # The ACCEPTED SET IS PER MODEL and this Literal is their union: Qwen3.8
    # takes xhigh/medium/low and RAISES on anything else, the harmony family
    # takes low/medium/high. Constrained rather than free-form so a typo is a
    # 422 here; an unsupported-but-spelled-correctly value still reaches the
    # template, and llama-server surfaces a raised jinja exception as a 500.
    # Absent = don't pass the kwarg at all, so the template's own default
    # applies (xhigh on Qwen3.8).
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="Thinking depth when thinking is on. Valid values are "
                    "MODEL-SPECIFIC (Qwen3.8: xhigh|medium|low; harmony "
                    "models: low|medium|high). Absent = the template's own "
                    "default.",
    )

    # Additional sampler parameters
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Reduce repetition (0-2, recommended 1.5 for Qwen3 thinking)")

    # Streaming options (OpenAI-compatible)
    stream_options: Optional[Dict] = Field(default=None, description="Options for streaming: {include_usage: true} to get usage stats")

    # Continuation ("prefill"): finish the FINAL message instead of opening a
    # new assistant turn. None = auto -- a trailing assistant message is
    # continued (the long-standing prefill convention, resolve_add_generation_
    # prompt). True = continue whatever the final message is, ANY role
    # (user-role continuation is MLX-only; llama-server prefills assistant
    # turns natively but has no user-turn spelling -- it 400s). False = never
    # continue (MLX renders the trailing turn closed and opens a fresh one;
    # gguf 400s on a trailing assistant, because llama-server ALWAYS continues
    # one and pretending otherwise would lie). The response carries ONLY the
    # continuation text, never an echo of the prefill, on every provider.
    continue_final_message: Optional[bool] = Field(
        default=None,
        description="Continue the final message (prefill) instead of opening a new "
                    "assistant turn. Default: auto (a trailing assistant message is "
                    "continued). true = continue any role's final message (user-role "
                    "is MLX-only); false = never continue."
    )

    @field_validator('messages', mode='before')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

    def is_continuation(self) -> bool:
        """Whether this request continues its final message (see
        ``continue_final_message``). The ONE resolution of flag-vs-convention;
        providers and the parser selection must all ask here, never re-derive."""
        if self.continue_final_message is not None:
            return self.continue_final_message
        return bool(self.messages) and self.messages[-1].role == 'assistant'

    def resumes_thinking(self) -> bool:
        """Whether this continuation resumes INSIDE the thinking block: the
        final message is an assistant turn carrying thinking and no content
        (whitespace-only counts as none; str or parts-list). The one shape
        both engines read the same way -- llama.cpp's reasoning_content-with-
        empty-content continuation, MLX's re-opened block -- so the provider,
        the parser selection and the prompt preview all ask HERE; three
        hand-copied spellings of this test drifted apart before it existed."""
        if not self.is_continuation() or not self.messages:
            return False
        last = self.messages[-1]
        if last.role != 'assistant' or not last.thinking:
            return False
        content = last.content
        if isinstance(content, list):
            content = "".join(getattr(p, "text", None) or "" for p in content)
        return not (content and content.strip())

# ChatRequest is the INTERNAL request every provider takes; the wire that
# produces it is /v1/messages (schema/converters.to_chat_request). The
# OpenAI-shaped response models that used to sit here (ChatCompletionResponse,
# PerformanceMetrics, the Batch* trio) left with that route in v1.79.66.

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


# ``engines`` alongside ``effect``: WHICH ENGINE a field actually reaches.
#
# The provider a field is declared on ("mlx", "gguf") is not the answer,
# because provider != engine -- provider "mlx" is TWO upstream repos on
# separate release trains (mlx-lm for text, mlx-vlm for vision), which is
# the same split `effective_loader` reports on the admin row and the same
# one `tests/helpers/engines.ARMS` names. A reader asking "does this do
# anything for my model" needs the engine, and until this tag existed the
# only answer was to read the provider source.
#
# Same rule as ``effect``: declared AT the field, derived everywhere else
# (``/v1/admin/model-options`` passes it through, docs link to that rather
# than restating it), and ``tests/unit/test_config_effects.py`` fails if a
# field omits it -- so a new field cannot be added without saying where it
# applies. A hand-maintained table of the same facts is this repo's named
# defect class; do not write one.
#
# WHAT THE TAG CANNOT SAY. It is per-ENGINE, and some fields are inert
# per-ARCHITECTURE within an engine -- the KV cache knobs are swallowed
# whole by `cache_helpers.create_kv_cache`'s `hasattr(model, "make_cache")`
# early return, silently, for every architecture defining one (qwen3_5,
# gemma3, the mamba family...). A tag listing both MLX engines is true and
# still not the whole answer there, so those fields say it in their own
# ``description``. When you add a field, ask both questions.
ENGINE_MLX_LM = "mlx-lm"
ENGINE_MLX_VLM = "mlx-vlm"
ENGINE_GGUF = "gguf"

# Order is display order, and matches tests/helpers/engines.ARMS -- pinned
# by a test rather than by this comment.
ENGINES: tuple = (ENGINE_MLX_LM, ENGINE_MLX_VLM, ENGINE_GGUF)

# Both MLX engines. The common case on MLXModelConfig: most fields reach
# generation the same way whichever library holds the weights.
ENGINES_MLX: list = [ENGINE_MLX_LM, ENGINE_MLX_VLM]


def field_engines(field) -> Optional[list]:
    """Declared engines for one FieldInfo, or None if it declares none."""
    value = _extra(field).get("engines")
    return list(value) if value is not None else None


def invalid_engines(cls: type) -> Dict[str, str]:
    """{field name -> why its ``engines`` tag is bad}. Empty when clean.

    Catches the two failure shapes a typo takes: a name outside the
    vocabulary (``"mlx"``, the PROVIDER, is the likely slip) and an empty
    list, which reads as "applies nowhere" and is never what an author
    means -- a field that applies nowhere should be deleted instead.
    """
    out: Dict[str, str] = {}
    for name, f in cls.model_fields.items():
        engines = field_engines(f)
        if engines is None:
            continue
        if not engines:
            out[name] = "empty engines list (a field applying nowhere should be removed)"
            continue
        unknown = [e for e in engines if e not in ENGINES]
        if unknown:
            out[name] = f"unknown engine(s): {', '.join(map(str, unknown))}"
    return out


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

    model_path: str = Field(
        description=(
            "Model directory (or HF repo id) holding the weights, config.json "
            "and tokenizer. Identity: changing it makes the entry a different "
            "model rather than the same one reconfigured, which is why it is "
            "not editable in place."),
        json_schema_extra={"effect": EFFECT_IDENTITY, "engines": ENGINES_MLX})
    draft_model_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a smaller model used as the drafter for speculative "
            "decoding. Set it when you have measured spec decode a win on "
            "THIS model at YOUR context length -- it is unproven in general "
            "here, and a LoRA erodes whatever win exists because the adapter "
            "reaches the target only. Unset = no speculative decoding."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX})
    # Classified requires_reload rather than per_request despite being a
    # runtime default: spec decode is set up when the draft model is loaded,
    # and the old hand-written reload set listed it. An unnecessary reload
    # prompt is a nuisance; a missed one silently serves stale behaviour.
    num_draft_tokens: Optional[int] = Field(
        default=3,
        description=(
            "How many tokens the drafter proposes per speculation round. "
            "Inert without `draft_model_path`. Higher drafts more per round "
            "and wastes more when the target rejects; tune it against your "
            "own model rather than porting a number from another one."),
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX},
    )
    # DESCRIPTION vs ROUTING split (Phase 6 refinement 2026-07-11). ``vision``
    # historically did both jobs; it is now a derived mirror of
    # ``"vision" in modalities`` (kept for back-compat with readers of
    # config["vision"]). ``modalities`` is the author-declared capability set;
    # ``loader`` selects the mlx engine (within provider="mlx" only).
    # ui:"hidden": a derived mirror is a dead knob in an editor -- config
    # re-derives it whenever modalities is set or detectable, so an edit
    # silently reverts at the next load. Edit modalities instead.
    vision: bool = Field(
        default=False,
        description=(
            "DERIVED MIRROR of `\"vision\" in modalities`, kept for readers of "
            "config[\"vision\"]. Do not edit it: config re-derives it whenever "
            "modalities is set or detectable, so an edit silently reverts at "
            "the next load. Edit `modalities` instead."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "hidden",
                           "engines": ENGINES_MLX})
    # None = "not provided" -> derived from ``vision`` in _resolve_modalities.
    # Detected at import from the config's own blocks (vision_config/audio_config
    # + *_token_id); see model_importer.detect_modalities.
    # requires_reload here, DESCRIPTIVE on the gguf config: for MLX this feeds
    # effective_loader (mlx-vlm vs mlx-lm), so changing it changes which engine
    # holds the weights. Provider-aware classification is the point.
    modalities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Author-declared capability set (e.g. [\"text\", \"vision\"]). Unset "
            "= detected at load from the model dir's own config.json. On MLX "
            "this is not merely descriptive as it is on gguf: it feeds "
            "`effective_loader`, so changing it changes WHICH ENGINE holds the "
            "weights (mlx-vlm vs mlx-lm)."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX})
    # Engine routing. "auto": mlx-vlm if "vision" in modalities AND mlx-vlm
    # registers the model_type, else mlx-lm. Explicit values force the engine
    # (e.g. run a dual-capable VLM as text via "mlx-lm"). Resolution + the
    # effective loader live in the provider (is_vlm derives from it).
    loader: Literal["auto", "mlx-vlm", "mlx-lm"] = Field(
        default="auto",
        description=(
            "Which MLX engine loads this model. \"auto\" picks mlx-vlm when "
            "\"vision\" is in modalities AND mlx-vlm registers the model_type, "
            "else mlx-lm. Set it explicitly to force one -- e.g. run a "
            "dual-capable VLM as text-only via \"mlx-lm\". This is the field "
            "that DECIDES a model's engine, so it is the one place where both "
            "MLX engines are the answer by construction."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX})
    # The model's context window, when config.json does not tell the truth.
    # None = read the file (capabilities.model_context_length: top-level
    # max_position_embeddings, then the nested text block). A YaRN-scaled
    # checkpoint often ships the ORIGINAL value with the factor in
    # rope_scaling, and the file value alone makes run_generation refuse a
    # prompt the model takes; this is the per-model answer. requires_reload:
    # the provider reads it ONCE at load into MLXProvider.context_length, the
    # number the over-length guard and the admin row both report. MLX only --
    # gguf's context is what the process was SPAWNED with (`ctx_size`).
    context_length: Optional[int] = Field(
        default=None, gt=0,
        description=(
            "The model's context window when config.json does not tell the "
            "truth -- a YaRN-scaled checkpoint often ships the ORIGINAL value "
            "with the factor in rope_scaling, and the file value alone makes "
            "generation refuse a prompt the model takes. Unset = read the "
            "file. IT ALLOCATES NOTHING: MLX has no fixed context allocation, "
            "the KV cache grows lazily in 256-token steps, so this cannot "
            "reduce load time, time-to-first-token or memory. Its only "
            "consumers are the over-length refusal and the admin row. The "
            "gguf counterpart, `ctx_size`, is genuinely an allocation."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX})
    # Sampler defaults: the loaded model serves any of these per request.
    # Each sits ABOVE the vendor layer (this model's own
    # generation_config.json) and BELOW a request field, so leaving one unset
    # is not "no opinion" -- it hands the question to the model's publisher,
    # which is normally the better answer. See samplers.sampler_defaults().
    temperature: Optional[float] = Field(
        default=None,
        description=(
            "Per-model default sampling temperature. Unset = the model's own "
            "generation_config.json value, else the global floor. A request "
            "field still wins over this."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    top_p: Optional[float] = Field(
        default=None,
        description=(
            "Per-model default nucleus-sampling cutoff. Unset = the model's "
            "own generation_config.json value, else the global floor."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    top_k: Optional[int] = Field(
        default=None,
        description=(
            "Per-model default top-k cutoff; 0 disables it. Unset = the "
            "model's own generation_config.json value. Worth setting only to "
            "overrule a publisher you disagree with -- gemma-4 asks for 64 "
            "and Qwen3.6 for 20, and those are the values that reach the "
            "sampler when this is unset."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    min_p: Optional[float] = Field(
        default=None,
        description=(
            "Per-model default min-p cutoff. NOT part of the vendor layer -- "
            "generation_config.json has no such field -- so unset means the "
            "global floor, and a publisher's DOCUMENTED min_p reaches nothing "
            "automatically. Set it by hand if you want theirs."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Per-model default cap on generated tokens. Unset = the global "
            "floor's stop. A request field still wins over this."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    repetition_penalty: Optional[float] = Field(
        default=None,
        description=(
            "Per-model repetition penalty over the recent token window. Off "
            "by default; reach for it on a model that loops. Not a vendor-"
            "layer key, so unset means off rather than the publisher's value. "
            "Counts only tokens THIS reply has generated, never the prompt "
            "(v2.0.60) -- see presence_penalty."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    presence_penalty: Optional[float] = Field(
        default=None,
        description=(
            "Per-model presence penalty. Off by default since v2.0.32, when "
            "the automatic thinking overlay that applied 1.5 to every "
            "thinking model was removed -- this field is how you ask for that "
            "behaviour back on one model. Counts only tokens THIS reply has "
            "generated (v2.0.60). Before that it depended on the path: a text "
            "request penalised its whole prompt -- system prompt, earlier "
            "turns and their end-of-turn tokens -- a prompt-cache hit only the "
            "uncached suffix, an image request only the reply. gguf differs by "
            "engine design: llama.cpp also counts the tail of the prompt."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    # None = AUTO (6a derive-at-load): resolved at model load from actual
    # weight bytes vs RAM (cache_defaults.resolve_cache_config). A stored
    # value is an explicit operator override.
    cache_type: Optional[Literal["standard", "rotating", "quantized"]] = Field(
        default=None,
        description=(
            "KV cache implementation. Unset = AUTO, resolved at load from "
            "weight bytes against this machine's RAM (quantized once the "
            "weights alone claim over ~35% of it). \"quantized\" trades a "
            "little quality for KV bytes; \"rotating\" bounds the cache by "
            "DROPPING context past `max_kv_size` and requires it. "
            "IGNORED ENTIRELY for architectures that define their own "
            "make_cache -- qwen3_5, gemma3, the mamba family and others -- "
            "because create_kv_cache returns the model's cache before reading "
            "this field. No error and no warning; the setting simply does "
            "nothing. Check your model before tuning it."),
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX},
    )
    max_kv_size: Optional[int] = Field(
        default=None,
        description=(
            "Cap on KV cache length, which creates a RotatingKVCache that "
            "silently DROPS context past the cap. Deliberately never "
            "defaulted: truncation is a correctness trade, not a tuning knob. "
            "NOT A PREALLOCATION and not a load-time lever -- RotatingKVCache "
            "grows lazily in 256-token steps like every other MLX cache, and "
            "the cache is constructed per generation, not at load, so this "
            "cannot speed up loading or time-to-first-token. Same make_cache "
            "blind spot as `cache_type`: inert on architectures defining one."),
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX},
    )
    # MLX QuantizedKVCache supports exactly 2/4/8 bits and group sizes that
    # divide the head dim; anything else fails at first generation, so reject
    # it at config-load time instead.
    kv_bits: Optional[Literal[2, 4, 8]] = Field(
        default=None,
        description=(
            "Bit width for a quantized KV cache. Only 2/4/8 exist in MLX; "
            "anything else fails at first generation, so it is refused here "
            "instead. Applies when `cache_type` resolves to \"quantized\" -- "
            "and shares that field's make_cache blind spot."),
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX},
    )
    kv_group_size: Literal[32, 64, 128] = Field(
        default=64,
        description=(
            "Quantization group size for a quantized KV cache; must divide "
            "the head dim. Leave it at 64 unless you have a reason. Shares "
            "`cache_type`'s make_cache blind spot."),
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX},
    )
    # In-flight + queued requests admitted before 503 backpressure. Consumed
    # by the generation gate (process-wide; the first provider created wins).
    # Process-wide once the gate exists (first provider created wins), so it
    # is infrastructure rather than a per-model tuning control.
    max_queue_depth: int = Field(
        default=8, ge=1,
        description=(
            "In-flight plus queued generations admitted before the server "
            "answers 503. Raise it to absorb bursts, lower it to fail fast "
            "instead of queueing. SETTABLE ONLY ON AN MLX ENTRY BUT NOT MLX-"
            "ONLY IN EFFECT: the generation gate is a process-global "
            "singleton that EVERY provider queues in, gguf included, so this "
            "value governs the whole server. The gguf provider looks for the "
            "same key on its own config, where no such field exists, and so "
            "always contributes the default. First provider created wins, "
            "which is why no reload of this model can change it."),
        json_schema_extra={"effect": EFFECT_LOAD_TIME_ONLY,
                           "engines": list(ENGINES),
                           "reason": "process-wide: the first provider created "
                                     "wins, so reloading this model cannot "
                                     "change it"})
    # Chunk size for prompt prefill. None lets mlx-lm use its default (2048).
    # Larger values reduce kernel-launch overhead on very long prompts at the
    # cost of higher peak memory during prefill.
    prefill_step_size: Optional[int] = Field(
        default=None, gt=0,
        description=(
            "How many prompt tokens one prefill chunk processes. Unset = "
            "mlx-lm's default of 2048. THE lever on a prefill-bound workload "
            "-- a long fixed system prompt with a short answer, a prompt "
            "encoder, a classifier -- where raising it cuts kernel-launch "
            "overhead at the cost of higher peak memory during prefill. "
            "Per-request, so it costs no reload to try. Lowering it is a "
            "memory-pressure lever, not a speed one. Applies to IMAGE requests "
            "too since v2.0.55 (the vision prefill was one un-chunked forward "
            "before, so this field did nothing there) -- except on a family "
            "whose own policy refuses to split a prompt carrying images "
            "(gemma-4: its vision blocks attend bidirectionally). gguf's nearest "
            "equivalents are `n_ubatch`/`n_batch`, which are spawn flags."),
        json_schema_extra={"is_runtime_default": True,
                           "effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX},
    )
    # Model-level thinking DEFAULT (Qwen3 <think> blocks, gemma-4 thought
    # channels). None = unset (v1.79.62): the cascade then falls back to the
    # model's thinking CAPABILITY (on for a model whose template reads
    # enable_thinking, off otherwise) -- see samplers.resolve_effective_sampling.
    # A bool here pins it either way; the gguf config carries the same field.
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Per-model thinking default. Unset = follow the model's thinking "
            "CAPABILITY (on where the chat template reads enable_thinking, "
            "off otherwise); a bool pins it either way. A request field still "
            "wins. Reaches the model as a chat-template variable, not a "
            "sampler setting."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
    # Model-level default for the request field of the same name. Per-request
    # because it is a template variable resolved at prompt-build time.
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description=(
            "Per-model thinking DEPTH default. A CHAT-TEMPLATE VARIABLE, not "
            "a sampler knob, and sent whenever set rather than gated on "
            "enable_thinking -- gpt-oss/harmony reads it unconditionally and "
            "has no enable_thinking at all. The accepted set is PER MODEL "
            "(Qwen3.8 takes xhigh|medium|low and raises otherwise; harmony "
            "takes low|medium|high), so the type here is their union and a "
            "wrong-for-this-model value reaches the template. Unset = send "
            "nothing, leaving the template's own default."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST,
                           "engines": ENGINES_MLX})
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
        default=None, ge=0,
        description=(
            "Seconds idle before this model is unloaded. Unset = the global "
            "`idle_unload_seconds`; 0 = never idle-unload this one. Pinned "
            "models are exempt regardless. applies_live, not load_time_only: "
            "the router re-reads it each idle sweep, so a change takes effect "
            "on an already-loaded model with no reload."),
        json_schema_extra={"effect": EFFECT_APPLIES_LIVE,
                           "engines": ENGINES_MLX})
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
        default=None,
        description=(
            "Which chat template to install: \"auto\" (trust the tokenizer, "
            "filling a missing one from chat_template.jinja > "
            "tokenizer_config.json > chat_template.json), \"jinja\", "
            "\"tokenizer_config\", \"chat_template_json\", or an absolute path "
            "to a .jinja file. Use it when a model ships a broken template "
            "but a working alternative, or to test one without re-exporting. "
            "MLX ONLY and deliberately NOT the same mechanism as gguf's "
            "`chat_template_path`/`use_sidecar_chat_template` -- different "
            "vocabulary, different resolution order, different name on "
            "purpose. For an operator-owned edit prefer the "
            "`/v1/admin/models/{id}/chat-template` route, which writes one "
            "file and no config at all."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "engines": ENGINES_MLX})

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
        # Assignments below add their names to __pydantic_fields_set__, which
        # would make DERIVED values report as explicitly stored -- the admin
        # API serializes config with exclude_unset precisely to distinguish
        # set from default, and this validator must not erase that line.
        provided = set(self.__pydantic_fields_set__)
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
        for name in ("modalities", "vision"):
            if name not in provided:
                self.__pydantic_fields_set__.discard(name)
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

class GGUFModelConfig(BaseModel):
    """A GGUF model served by a llama-server SUBPROCESS (plan Phase 7).

    One entry = one servable model; MTP/draft artifacts are FIELDS here,
    never their own entries (embedded MTP -> just ``spec_type``; a sidecar
    drafter -> ``draft_model_path``; the same field the MLX config uses).
    llama-server owns tokenization, chat templating, and reasoning splitting
    -- the provider surfaces pre-split thinking via GenerationChunk.thinking
    and reports template_info() = None.

    Chat templating resolves as a THREE-WAY LADDER (v1.79.43):
    ``chat_template_path`` (an explicit file) beats a ``chat_template.jinja``
    discovered beside the .gguf, which beats the jinja EMBEDDED IN THE GGUF.
    The embedded one means whoever quantized the file chose the prompt format,
    and publishers ship materially different templates for the same weights --
    which is why a readable sidecar wins by default, and why
    ``use_sidecar_chat_template = false`` exists for the case where the
    embedded template is the one you want. NB picking a quant can still
    silently pick a format: now it is the quant's snapshot DIRECTORY that
    does it, if one ships a sidecar. Every spawn logs which rung won.

    Measured on Qwen3.8-27B (live, both templates, 2026-08-17): ggml-org
    embeds Qwen's official template byte-identically (8952 bytes); unsloth
    embeds a patched one (9993) adding a `developer` role and MERGING up to
    two leading system messages. The observable difference is narrow but real
    -- two leading system messages render under unsloth and are a 500 under
    official ("System message must be at the beginning"), because a raised
    jinja exception surfaces as a 500 from llama-server. Both templates still
    reject a system message that appears MID-conversation, so the permissive
    one is not permissive in general; do not assume a shape works without
    trying it.
    """
    model_config = ConfigDict(extra="forbid")

    # path to the .gguf file
    model_path: str = Field(
        description=(
            "Path to the .gguf weights file. Identity: changing it makes the "
            "entry a different model rather than the same one reconfigured."),
        json_schema_extra={"effect": EFFECT_IDENTITY, "engines": [ENGINE_GGUF]})
    # multimodal projector sidecar
    mmproj_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the multimodal projector sidecar that gives this model "
            "image (and audio) input. Required for vision on gguf -- without "
            "it llama-server spawns with no --mmproj and the model is text-"
            "only however its weights are described. Discovery pairs one "
            "automatically; an explicit entry must carry it by hand, which is "
            "how a vision model has lost it before."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "--mmproj",
                           "engines": [ENGINE_GGUF]},
    )
    # Override the GGUF-embedded chat template with a jinja file on disk.
    # requires_reload, not per_request: llama-server takes the template at
    # SPAWN (--chat-template-file), so there is no per-request or per-preset
    # form of this -- changing it costs a respawn, same as mmproj_path. The
    # per-request lever that DOES exist is chat_template_kwargs (the provider
    # already sends enable_thinking through it).
    #
    # `--chat-template-file`, never the `--chat-template` sibling: that one
    # takes template TEXT, and only "commonly used" builtin names unless
    # --jinja is set. A path keeps the template reviewable/diffable on disk
    # instead of inlined into models.toml.
    chat_template_path: Optional[str] = Field(
        default=None,
        description=(
            "An explicit jinja file to use instead of the template embedded "
            "in the GGUF. Top rung of the template ladder, beaten only by the "
            "operator override file. Reach for it when the publisher's "
            "embedded template is wrong for your use -- publishers ship "
            "materially different templates for identical weights, and which "
            "one you got came with the quant. llama-server takes the template "
            "at SPAWN, so there is no per-request form; the per-request lever "
            "is `chat_template_kwargs`. gguf only -- MLX's counterpart is "
            "`chat_template_source`, a different mechanism under a different "
            "name on purpose."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--chat-template-file", "engines": [ENGINE_GGUF]},
    )
    # Sidecar discovery (v1.79.43, owner ask). When `chat_template_path` is
    # unset and the model file's OWN directory contains `chat_template.jinja`,
    # that file is passed as --chat-template-file instead of falling through
    # to the template embedded in the GGUF.
    #
    # Why on by default: the embedded template is whatever the QUANTIZER baked
    # in, and this repo already documents publishers shipping materially
    # different templates for identical weights. A `chat_template.jinja` sitting
    # beside the weights is the publisher's (or the operator's) LATER, more
    # legible answer -- it is the file you can read, diff and edit, which the
    # embedded one is not.
    #
    # Why a field and not just "delete the file": a downloaded snapshot dir is
    # not somewhere to have to vandalize to get the documented default back,
    # and the embedded template is a legitimate choice -- see the class docstring
    # on the two Qwen3.8 publishers. Set this False to keep the embedded one
    # while leaving the sidecar on disk.
    #
    # Deliberately NOT named `chat_template_source`: that is the MLX-side field
    # with a different vocabulary and a different resolution order, and it does
    # not reach this provider. Two mechanisms, two names.
    use_sidecar_chat_template: bool = Field(
        default=True,
        description=(
            "When no `chat_template_path` is set and a `chat_template.jinja` "
            "sits beside the .gguf, use that file rather than the template "
            "embedded in the GGUF. On by default because the embedded one is "
            "whatever the quantizer baked in, while a sidecar is the file you "
            "can read, diff and edit. Set it False to keep the embedded "
            "template WITHOUT deleting a file out of a downloaded snapshot "
            "dir. It does not govern the operator override "
            "(chat_template.heylook.jinja), which is a separate file and a "
            "separate decision."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "engines": [ENGINE_GGUF]})
    # sidecar drafter (e.g. gemma mtp-*.gguf). `-md`, not the `--model-draft`
    # alias: `arg` must be the spelling the provider ACTUALLY emits, so a UI or
    # a derived emitter reproduces the real command line rather than an
    # equivalent-but-different one.
    draft_model_path: Optional[str] = Field(
        default=None,
        description=(
            "Sidecar drafter (e.g. a gemma `mtp-*.gguf`). THIS FIELD IS WHAT "
            "TURNS SPECULATIVE DECODING ON, not `spec_type`: the provider "
            "emits -md on this field alone, and llama.cpp infers the draft "
            "type from the drafter's own header when --spec-type is absent. "
            "Discovery pairs a sidecar automatically and leaves `spec_type` "
            "unset on purpose, so a model can be running spec decode with no "
            "models.toml entry at all. To keep it OFF, the drafter must not "
            "be paired. Unproven as a win here in general -- check your own "
            "model at your own context."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-md",
                           "engines": [ENGINE_GGUF]},
    )
    # llama-server --spec-type (e.g. "draft-mtp"). NB coupled to LoRA: a loaded
    # adapter erases spec decode's win, because the draft context never
    # receives the adapter (see CLAUDE.md's gguf gotchas). Leave it ON anyway.
    spec_type: Optional[str] = Field(
        default=None,
        description=(
            "PINS the speculative draft type (e.g. \"draft-mtp\"). NOT the "
            "on/off switch -- `draft_model_path` is, and an unset spec_type "
            "does NOT mean spec decode is off. Strictly required only for a "
            "SHARDED drafter, where llama.cpp's header read sees the first "
            "split alone; sharding of the TARGET is irrelevant. Inert without "
            "a drafter. Coupled to LoRA: an adapter reaches the target "
            "context only, so the drafter proposes the base distribution."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-type", "engines": [ENGINE_GGUF]},
    )
    spec_draft_n_max: Optional[int] = Field(
        default=None, ge=1, le=16,
        description=(
            "Ceiling on draft tokens per speculation round. Inert without a "
            "drafter. TUNE IT TOGETHER WITH `spec_draft_p_min` -- they "
            "interact and the interaction inverts, so a one-dimensional sweep "
            "finds a different and wrong optimum. Per-model: no defensible "
            "global value exists."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-draft-n-max", "engines": [ENGINE_GGUF]},
    )
    ctx_size: Optional[int] = Field(
        default=None, ge=512,
        description=(
            "Context slot llama-server allocates at spawn. Unset = -c 0, i.e. "
            "the model's training context, with --fit shrinking unset args to "
            "device memory. UNLIKE MLX'S `context_length` THIS IS A REAL "
            "ALLOCATION: lowering it genuinely reclaims memory and can make a "
            "model load that otherwise would not, because llama-server sizes "
            "the KV slot up front. MLX has no equivalent -- its cache grows "
            "lazily, so there is nothing there to size. The admin row reports "
            "`context_length` (the GGUF header ceiling) beside "
            "`context_running` (what the process actually got)."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "--ctx-size",
                           "engines": [ENGINE_GGUF]},
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
        description=(
            "Minimum probability for a drafted token to be kept. Inert "
            "without a drafter. Not a minor knob: it INTERACTS with "
            "`spec_draft_n_max`, so tune the two together or not at all. "
            "Strictly per-model -- the same value that helps both gemmas "
            "costs Qwen3.6-27B at every setting tried."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-draft-p-min", "engines": [ENGINE_GGUF]},
    )
    # Third member of the spec tuning family (--spec-draft-n-min): the floor
    # on how many draft tokens a speculation round keeps. Same caveat as its
    # siblings: the levers interact, tune together, per-model.
    spec_draft_n_min: Optional[int] = Field(
        default=None, ge=0,
        description=(
            "Floor on how many draft tokens a speculation round keeps. Inert "
            "without a drafter, and the same caveat as its two siblings: the "
            "levers interact, tune together, per-model."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--spec-draft-n-min", "engines": [ENGINE_GGUF]},
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
        description=(
            "Move this many layers' expert tensors to the CPU. On UNIFIED "
            "memory it does not shrink total RAM -- it moves those bytes out "
            "of the Metal working set and that math onto CPU cores. For a "
            "model that fits in RAM but crowds out the KV cache. Try "
            "`cache_type_k`/`cache_type_v` first: for a headroom problem that "
            "is usually the better trade and a smaller change. A value past "
            "the model's layer count is a SILENT no-op. MoE models only."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ncmoe",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    # All expert tensors to CPU. Equivalent to n_cpu_moe >= n_layer, kept
    # separate because it needs no layer count to express. A BARE flag.
    cpu_moe: Optional[bool] = Field(
        default=None,
        description=(
            "Move ALL expert tensors to the CPU. Equivalent to `n_cpu_moe` at "
            "or above the layer count, kept separate because it needs no "
            "layer count to express. Same trade as `n_cpu_moe`. MoE only."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-cmoe",
                           "shape": "flag", "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    # Raw tensor-buffer overrides: the unrestricted form of the two above. A
    # pattern that matches nothing is a silent no-op, so prefer n_cpu_moe.
    override_tensor: Optional[str] = Field(
        default=None,
        description=(
            "Raw tensor-buffer override pattern: the unrestricted form of "
            "`n_cpu_moe`/`cpu_moe`. Prefer those -- a pattern matching nothing "
            "is a silent no-op, so a typo here costs you the placement you "
            "thought you had with no error anywhere."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ot",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    # Draft-side expert offload (-ncmoed/-cmoed): the drafter's half of the
    # residency budget. Exists for the target-offloaded + drafter-resident
    # split, where the pair exceeds the working set while either alone fits.
    n_cpu_moe_draft: Optional[int] = Field(
        default=None, ge=0,
        description=(
            "`n_cpu_moe` for the DRAFTER. Its own knob because the pair can "
            "exceed the working set while either alone fits -- the "
            "target-offloaded, drafter-resident split. Inert without a "
            "drafter, and without an MoE drafter."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ncmoed",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    cpu_moe_draft: Optional[bool] = Field(
        default=None,
        description=(
            "`cpu_moe` for the DRAFTER: all of its expert tensors to the CPU. "
            "Inert without an MoE drafter."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-cmoed",
                           "shape": "flag", "ui": "advanced", "engines": [ENGINE_GGUF]},
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
        description=(
            "Quantization type for the K half of the KV cache. Unset = f16, "
            "which is the default and stays so: these exist for headroom "
            "EMERGENCIES, not as a tuning default. For a headroom problem "
            "this is usually the better first lever than expert offload -- it "
            "shrinks the KV bytes themselves rather than moving weight math "
            "to the CPU. MLX's counterpart is `cache_type`/`kv_bits`."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ctk",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    cache_type_v: Optional[Literal[
        "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"
    ]] = Field(
        default=None,
        description=(
            "Quantization type for the V half of the KV cache. Same posture "
            "as `cache_type_k` -- f16 by default and left there unless you "
            "have a headroom problem. A quantized V-cache needs flash "
            "attention, whose default is auto in current builds."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ctv",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    # -ngl; 999 = everything on GPU
    n_gpu_layers: int = Field(
        default=999,
        description=(
            "How many layers to offload to the GPU; 999 means all of them, "
            "which is what you want on Apple Silicon. Lower it only to keep "
            "weight bytes out of the Metal working set deliberately."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ngl",
                           "engines": [ENGINE_GGUF]},
    )
    # -ub: the PHYSICAL prompt-processing batch -- how many prompt tokens one
    # Metal dispatch chews through. llama-server's own default is 512, sized
    # for machines where the compute buffer competes with the weights.
    # None = AUTO, resolved at spawn by the provider (`_auto_ubatch`): 2048
    # when the model's Metal working-set headroom (ram_fit, weights + sidecars
    # against the live ceiling) clears THIN_HEADROOM_GB, else llama-server's
    # own default. 2048 is a prefill-throughput win on every model measured
    # here (small on dense, large on MoE), generation speed unchanged, and the
    # price is a larger compute buffer -- ~7 GiB more on DeepSeek V4 Flash,
    # which is exactly what pushed its Vision variant into a decode-time
    # Metal OOM at the OS-default wired limit. llama's `--fit` cannot shrink
    # an explicit value and did not catch that case, so the guard is ours,
    # reads the same ceiling `--fit` does, and flips to 2048 by itself once
    # iogpu.wired_limit_mb is raised. A stored value always wins, both ways.
    # Numbers and conditions: internal/research (2026-09-07).
    n_ubatch: Optional[int] = Field(
        default=None, ge=32,
        description=(
            "PHYSICAL prompt-processing batch: how many prompt tokens one "
            "Metal dispatch chews through. The gguf lever on a prefill-bound "
            "workload, nearest in spirit to MLX's `prefill_step_size` -- but "
            "a SPAWN flag, so it costs a reload where the MLX one does not. "
            "Unset = AUTO: the provider spawns -ub 2048 when the model's "
            "working-set headroom clears the thin threshold, else inherits "
            "llama-server's 512, logging which. A stored value always wins, "
            "both ways. 2048 is a prefill win at no generation cost, and "
            "costs a larger compute buffer -- which is what pushed one vision "
            "model into a decode-time Metal OOM that llama's own pre-flight "
            "never saw."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ub",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    # -b: the LOGICAL batch, the most tokens one llama_decode call takes.
    # llama-server's default (2048) already equals the auto micro-batch and
    # a value above it buys nothing with -np 1, so None inherits it. It is a
    # field at all because llama.cpp silently CLAMPS n_ubatch to n_batch --
    # the validator below turns that clamp into a load-time refusal.
    n_batch: Optional[int] = Field(
        default=None, ge=32,
        description=(
            "LOGICAL batch: the most tokens one llama_decode call takes. "
            "Unset inherits llama-server's 2048, which already equals the "
            "auto micro-batch, and a higher value buys nothing at -np 1. It "
            "is a field at all because llama.cpp silently CLAMPS n_ubatch to "
            "it -- the validator turns that clamp into a load-time refusal "
            "instead of a setting that reads as applied and is not."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-b",
                           "ui": "advanced", "engines": [ENGINE_GGUF]},
    )
    # Draft-model GPU offload (-ngld). Its own knob because the pair can exceed
    # the GPU budget when the target alone does not: on a 192 GiB M2 Ultra the
    # Metal residency recommendation is ~161 GiB, so a 144 GiB target plus a
    # 10 GiB drafter is over it while either alone is under. `0` keeps the
    # drafter off the GPU. None = inherit llama-server's own default.
    n_gpu_layers_draft: Optional[int] = Field(
        default=None, ge=0,
        description=(
            "GPU layers for the DRAFTER. Its own knob because the pair can "
            "exceed the GPU budget when the target alone does not. 0 keeps "
            "the drafter off the GPU entirely; unset inherits llama-server's "
            "default. Inert without a drafter."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-ngld",
                           "engines": [ENGINE_GGUF]},
    )
    # llama-server's OWN idle sleep (--sleep-idle-seconds): frees the model and
    # KV cache but KEEPS THE PROCESS, and reloads on the next task. Strictly
    # cheaper than heylook's idle-unload, which SIGTERMs and respawns -- so set
    # this BELOW the effective idle_unload_seconds and you get the cheap
    # recovery first and the expensive one only for a genuinely cold model.
    # None = disabled (llama-server's default).
    sleep_idle_seconds: Optional[int] = Field(
        default=None, ge=1,
        description=(
            "llama-server's OWN idle sleep: frees the model and KV cache but "
            "KEEPS THE PROCESS, reloading on the next request. Strictly "
            "cheaper than heylook's idle-unload, which SIGTERMs and respawns "
            "-- so set this BELOW the effective `unload_after_idle_seconds` "
            "and you get the cheap recovery first and the expensive one only "
            "for a genuinely cold model. Unset = disabled. No MLX equivalent: "
            "there is no subprocess to keep."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD,
                           "arg": "--sleep-idle-seconds", "engines": [ENGINE_GGUF]},
    )
    # -cram: prompt-cache budget in MiB. llama-server defaults to only 8192;
    # -1 = unlimited, 0 = disable the cache entirely.
    cache_ram_mb: Optional[int] = Field(
        default=None, ge=-1,
        description=(
            "llama-server's prompt-cache budget in MiB. Its default is only "
            "8192, so raise this when you re-send a long shared prefix and "
            "want it kept; -1 = unlimited, 0 = disable the cache. This is the "
            "gguf lever with no MLX counterpart you can set -- MLX's prompt "
            "cache is a single slot sized by the model, not a budget."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-cram",
                           "engines": [ENGINE_GGUF]},
    )
    # -lm: how weights are brought in. `mlock` pins them against paging, which
    # is the lever for a model near the memory ceiling; llama.cpp's Metal
    # residency set is separate and already on by default.
    load_mode: Optional[Literal["none", "mmap", "mlock", "mmap+mlock", "dio"]] = Field(
        default=None,
        description=(
            "How weights are brought in. `mlock` pins them against paging, "
            "the lever for a model near the memory ceiling; llama.cpp's Metal "
            "residency set is separate and already on. Unset = llama-server's "
            "own default."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "arg": "-lm",
                           "engines": [ENGINE_GGUF]},
    )
    # The four below are ui:"hidden": settable (requires_reload is their real
    # effect class -- a fresh spawn reads them), but no editor should offer
    # them. port 0 = pick-a-free-port is the correct behavior and nothing
    # good comes of a UI breaking it; a per-model path-to-an-executable
    # picker in a web form is a foot-cannon; host/startup_timeout_s are
    # plumbing. Declared HERE so every consumer (v3 editor, its E2E check,
    # any future UI) reads one source instead of each hand-copying the list.
    # else required via $HEYLOOK_LLAMA_SERVER
    server_binary: Optional[str] = Field(
        default=None,
        description=(
            "Override the canonical llama-server build for this model. An "
            "escape hatch for experiments, not a normal setting: it WARNS AT "
            "EVERY SPAWN naming the canonical build it shadows, because a "
            "stale binary quietly shadowing a fresh one is the failure this "
            "warning exists to prevent. Unset = the build written by "
            "scripts/build_llama.py, which is the one source."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "hidden",
                           "engines": [ENGINE_GGUF]})
    host: str = Field(
        default="127.0.0.1",
        description=(
            "Interface the llama-server subprocess binds. Plumbing -- leave "
            "it on loopback; this subprocess is heylook's, not a service."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "hidden",
                           "engines": [ENGINE_GGUF]})
    # 0 = pick a free port at load
    port: int = Field(
        default=0,
        description=(
            "Port for the llama-server subprocess. 0 means pick a free one, "
            "which is the correct behaviour -- pinning it only creates a "
            "collision you have to debug."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "hidden",
                           "engines": [ENGINE_GGUF]})
    startup_timeout_s: float = Field(
        default=300.0,
        description=(
            "How long to wait for the subprocess to report ready before the "
            "load fails. Raise it for a very large model on cold storage."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "hidden",
                           "engines": [ENGINE_GGUF]})
    # Raw passthrough flags. requires_reload because they are spawn argv --
    # and note this is remote argv injection into a subprocess for anyone with
    # admin PATCH access, so a UI should not make it a casual free-text field
    # (ui:"advanced" keeps it behind v3's collapsed disclosure).
    extra_args: List[str] = Field(
        default_factory=list,
        description=(
            "Raw flags appended to the llama-server command line, for options "
            "heylook has no field for. This is argv injection into a "
            "subprocess for anyone with admin PATCH access, so it is not a "
            "casual free-text field. The three llama.cpp flags that write to "
            "disk on their own are REFUSED here -- one of them writes PROMPT "
            "TEXT, at observability_level=off, with nothing announcing it."),
        json_schema_extra={"effect": EFFECT_REQUIRES_RELOAD, "ui": "advanced",
                           "engines": [ENGINE_GGUF]},
    )
    # model-level default cap
    max_tokens: Optional[int] = Field(
        default=None, gt=0,
        description=(
            "Per-model default cap on generated tokens. Unset = the global "
            "floor's stop, which matters more here than on MLX: "
            "llama-server's own n_predict default is UNLIMITED, so something "
            "must always cap it."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST, "engines": [ENGINE_GGUF]})
    # Capability DESCRIPTION. Import fills this from the GGUF's own embedded
    # chat template (gguf_metadata.supports_thinking, same enable_thinking
    # rule the MLX path uses); it stays overridable by hand, and the explicit
    # ModelConfig.capabilities override short-circuits inference entirely.
    # None = no template to judge, e.g. an MTP/drafter head.
    supports_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Whether this model's template does thinking. DESCRIPTIVE: it "
            "changes what the server advertises, not what the process does. "
            "Import fills it from the GGUF's embedded template; unset = no "
            "template to judge, e.g. an MTP/drafter head. gguf carries this "
            "flag where MLX does not because the template lives inside GGUF "
            "metadata, with nothing cheap to probe -- on MLX the same fact is "
            "derived from the template file."),
        json_schema_extra={"effect": EFFECT_DESCRIPTIVE, "engines": [ENGINE_GGUF]})
    modalities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Declared capability set (e.g. [\"text\", \"vision\"]). DESCRIPTIVE "
            "here, unlike its MLX namesake: gguf is one engine, so this "
            "changes what is advertised and routes nothing. What actually "
            "gives a gguf model vision is `mmproj_path`."),
        json_schema_extra={"effect": EFFECT_DESCRIPTIVE, "engines": [ENGINE_GGUF]})
    # Model-level thinking DEFAULT (the MLX config's counterpart), distinct
    # from `supports_thinking` above, which only describes CAPABILITY.
    # Required since unset started meaning OFF everywhere (v1.50.0): before
    # that, a gguf model inherited its template's own default -- thinking-ON
    # for gemma-4/Qwen3.6/DeepSeek-V4 -- and with extra="forbid" and no field
    # here there was then NO way to ask for that back. The only remaining
    # route was a named sampler, which dragged a presence_penalty change in
    # with it -- and named samplers are gone (v2.0.30). None = unset = off.
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Per-model thinking default, the counterpart to "
            "`supports_thinking` above, which only describes CAPABILITY. "
            "Unset = off. Reaches llama-server as a chat_template_kwargs "
            "entry. It exists because unset started meaning OFF everywhere in "
            "v1.50.0 -- before that a gguf model inherited its template's own "
            "default, and with extra=\"forbid\" there was then no way to ask "
            "for that back."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST, "engines": [ENGINE_GGUF]})
    # Per-model repetition control, the MLX config's counterpart. Added in
    # v2.0.32 with the removal of the automatic thinking overlay: that overlay
    # applied presence_penalty 1.5 to every thinking model on both engines,
    # and MLX could already be tuned per model while gguf could not. Removing
    # a global default without leaving a per-model lever would have been a
    # capability loss rather than a simplification.
    presence_penalty: Optional[float] = Field(
        default=None, ge=0.0, le=2.0,
        description=(
            "Per-model presence penalty, the MLX config's counterpart. Added "
            "in v2.0.32 with the removal of the automatic thinking overlay "
            "that applied 1.5 to every thinking model on both engines: MLX "
            "could already be tuned per model, gguf could not, and removing "
            "the global without leaving this lever would have been a "
            "capability loss rather than a simplification. llama.cpp "
            "penalises over a recent-token window that INCLUDES the tail of "
            "the prompt (its server feeds prompt tokens to the sampler), so "
            "the same value is not equivalent to MLX, which counts only the "
            "reply."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST, "engines": [ENGINE_GGUF]})
    # Model-level default thinking DEPTH, mirroring the MLX config's field of
    # the same name. Reaches llama-server as a chat_template_kwargs entry, so
    # the accepted set is whatever THIS model's embedded template accepts --
    # see ChatRequest.reasoning_effort for why the Literal is a union.
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description=(
            "Per-model thinking DEPTH default, mirroring the MLX field of the "
            "same name. Reaches llama-server as a chat_template_kwargs entry, "
            "so the accepted set is whatever THIS model's embedded template "
            "accepts -- and a value it rejects becomes a raised jinja "
            "exception, which llama-server returns as a 500. Unset = send "
            "nothing, leaving the template's own default."),
        json_schema_extra={"effect": EFFECT_PER_REQUEST, "engines": [ENGINE_GGUF]})

    # llama-server's default logical batch, the ceiling n_ubatch is clamped to
    # when n_batch is not set. Named so the validator's message can say it.
    LLAMA_DEFAULT_N_BATCH: ClassVar[int] = 2048

    # llama.cpp's three options that write to disk. heylook passes none of
    # them and strips the one that is env-reachable (LLAMA_ARG_LOG_FILE), so
    # `extra_args` is the last route by which a gguf model can put a file
    # somewhere heylook did not choose -- and for --log-prompts-dir that file
    # is PROMPT TEXT, at observability_level="off", with nothing announcing
    # it. Refuse them here rather than at spawn: this catches an import, an
    # admin PATCH and a hand-edited models.toml, and the author sees the
    # message where the value is.
    DISK_WRITING_FLAGS: ClassVar[tuple] = (
        "--log-file", "--log-prompts-dir", "--slot-save-path")

    @model_validator(mode="after")
    def _extra_args_writes_no_files(self):
        for token in self.extra_args:
            # `--log-file=X` and `--log-file X` are both accepted by llama.cpp,
            # so match the flag NAME, not the whole token.
            flag = token.split("=", 1)[0]
            if flag in self.DISK_WRITING_FLAGS:
                raise ValueError(
                    f"extra_args carries {flag}, which makes llama-server write "
                    f"to disk on its own. heylook owns this subprocess's output: "
                    f"it goes nowhere at observability_level='off' (the default) "
                    f"and to logs/llama_server_<id>.log above it. Raise "
                    f"observability_level instead of passing {flag}."
                )
        return self

    @model_validator(mode="after")
    def _ubatch_within_batch(self):
        # llama_context takes min(n_batch, n_ubatch) WITHOUT a word, so an
        # n_ubatch above the logical batch is a setting that reads as applied
        # and is not. Refuse it here, where the models.toml author sees it.
        ceiling = self.n_batch if self.n_batch is not None else self.LLAMA_DEFAULT_N_BATCH
        if self.n_ubatch is not None and self.n_ubatch > ceiling:
            raise ValueError(
                f"n_ubatch={self.n_ubatch} exceeds n_batch={ceiling}"
                f"{' (llama-server default)' if self.n_batch is None else ''}; "
                f"llama.cpp would silently clamp it. Raise n_batch or lower n_ubatch."
            )
        return self


# Single source of truth for which providers exist and which config class
# validates each one's `config` block. Adding a provider = add an entry here
# + widen the Literal below + register the provider class in router.py's
# provider_map (which must stay in key-sync with this dict).
PROVIDER_CONFIG_CLASSES: Dict[str, type] = {
    "mlx": MLXModelConfig,
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


def _validate_documentation_declarations() -> None:
    """Fail at IMPORT if a provider-config field is undocumented.

    Two facts, one rule, and the same fail-fast argument as
    ``_validate_effect_declarations`` above: developer-authored static data,
    a bad value is a code bug that cannot be provoked by user input, and the
    degradation is silent in the safe-looking direction.

    ``description`` -- what the field does and why you would reach for it.
    It is not decoration: ``/v1/admin/model-options`` is the ONLY surface
    that publishes it, so a field without one is a knob nobody outside this
    file can use correctly. Sixty of them shipped that way, which is how a
    consumer came to recommend two cache knobs that are inert on the model
    they were recommended for. (NOT ``/openapi.json``: no route binds these
    classes as a typed body, so they never appear in the served spec --
    their ``model_json_schema()`` carries both facts, which is the trap, not
    the carrying path.)

    ``engines`` -- which of mlx-lm / mlx-vlm / gguf the field actually
    reaches. The class a field is declared on does NOT answer this: provider
    "mlx" is two upstream repos on separate release trains, and at least one
    field (``max_queue_depth``) governs every engine from a single provider's
    config.

    Both are declared AT the field and derived everywhere else. Writing
    either fact into a doc table instead is this repo's named defect class.
    """
    problems: List[str] = []
    for provider, cls in PROVIDER_CONFIG_CLASSES.items():
        for name, bad in invalid_engines(cls).items():
            problems.append(f"  {provider}.{name}: {bad}")
        for name, field in cls.model_fields.items():
            if field_engines(field) is None:
                problems.append(
                    f"  {provider}.{name}: no `engines` declared -- name which "
                    f"of {list(ENGINES)} this field actually reaches"
                )
            if not (field.description or "").strip():
                problems.append(
                    f"  {provider}.{name}: no `description` -- "
                    f"/v1/admin/model-options is the only surface that "
                    f"publishes it"
                )
    if problems:
        raise RuntimeError(
            "Provider config fields must document themselves at the "
            "declaration:\n" + "\n".join(sorted(problems))
        )


_validate_documentation_declarations()


class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "gguf"]
    config: Union[MLXModelConfig, GGUFModelConfig]
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True
    # Model capabilities for discovery (e.g., ["chat", "thinking", "vision"])
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
    """Watch-folder config: what the model store contributes to the registry.

    Since v1.69.0 these folders are the REGISTRY, not just a notification
    feed: ``model_registry`` folds everything found here into the served set
    at router load, so a model under a scan folder is servable with no
    models.toml entry. Nothing is written -- models.toml stays override-only,
    and an entry there always wins (see model_registry for the merge rule).

    ``scan_interval_seconds = 0`` disables scanning entirely: no periodic
    rescan (``MemoryManager.tick()``) AND no load-time discovery. Setting it
    to 0 must not leave load-time discovery running, or the documented off
    switch would instead silently serve every model under the folders.

    ``MemoryManager.tick()`` additionally maintains the passive
    discovered-but-not-configured cache behind
    ``GET /v1/admin/models/discovered`` -- still useful for "what is here
    that I have not customized", now that being discovered is enough to be
    served.
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
# Admin API Models (Model Management)
# =============================================================================

class ScanConfigRequest(BaseModel):
    """PUT body for ``[scan]``. Every field optional -- absent = leave alone.

    Not a settings-table concern: these folders decide what the server
    SERVES (model_registry), so they live in models.toml beside the models,
    not in the DuckDB `settings` table that holds operational preferences.
    """
    model_config = ConfigDict(extra="forbid")

    folders: Optional[List[str]] = Field(
        default=None,
        description="Directories to discover models from. Tilde paths are "
                    "expanded by the scanner. Order preserved, duplicates "
                    "dropped.",
    )
    watch_hf_cache: Optional[bool] = Field(default=None)
    scan_interval_seconds: Optional[int] = Field(
        default=None, ge=0,
        description="0 disables scanning entirely -- no periodic rescan AND "
                    "no load-time discovery.",
    )


class ScanConfigResponse(BaseModel):
    """``[scan]`` as saved, plus what it currently adds up to."""
    folders: List[str] = Field(default_factory=list)
    watch_hf_cache: bool = False
    scan_interval_seconds: int = 900
    models_served: int = Field(
        default=0,
        description="Models the router serves after this change (models.toml "
                    "entries plus discovered) -- the observable consequence "
                    "of editing the folder list.",
    )
    warning: Optional[str] = Field(
        default=None, description="Saved, but the router reload failed.")


class ScannedModelResponse(BaseModel):
    """A model discovered during filesystem scan.

    Mirrors ``model_service.ScannedModel``. Wired as the /scan route's
    ``response_model`` on purpose: this model sat unreferenced for months
    while the dataclass grew, so the declared contract and what the route
    actually returned had no way to disagree loudly.
    """
    id: str = Field(..., description="Auto-generated model identifier")
    path: str = Field(..., description="Filesystem path to model")
    provider: Literal["mlx", "gguf"] = Field(..., description="Detected provider type")
    size_gb: float = Field(..., description="Estimated model size in GB")
    vision: bool = Field(default=False, description="Whether model supports vision (shadow of `modalities`)")
    quantization: Optional[str] = Field(default=None, description="Quantization level (4bit, 8bit, etc)")
    already_configured: bool = Field(default=False, description="True if ID already exists in models.toml")
    served: bool = Field(
        default=False,
        description="The router already serves this file (via [scan] "
                    "discovery). Distinct from already_configured, which "
                    "means it has a models.toml entry -- since v1.69.0 a "
                    "model can be served with no entry, so importing it "
                    "would change nothing.",
    )
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

    extra="forbid": a stale {"profile": ...} or {"default_sampler": ...} body
    from an old client must fail loudly rather than be silently dropped --
    both named the bundled-sampler system removed in v2.0.30.
    """
    model_config = ConfigDict(extra="forbid")

    models: List[Dict] = Field(..., description="Models to import (id, path, provider, overrides)")


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
    # requires_reload keys whose saved value differs from what the LOADED
    # process was built with (router.stale_reload_fields -- server-derived,
    # so a UI's "reload to apply" marker survives remounts and other tabs).
    # Always [] for unloaded models.
    source: Literal["config", "discovered"] = Field(
        default="config",
        description="Where this model comes from. 'config' = a models.toml "
                    "[[models]] entry. 'discovered' = found under "
                    "[scan].folders with no entry -- it is served all the "
                    "same, but it has no stored config, and the first edit "
                    "writes an entry for it.",
    )
    # Every sampler key's value for a request that says nothing, from the
    # cascade itself (samplers.sampler_defaults). ONE flat bag since v2.0.33 --
    # it was {"off":..., "on":...} while the anti-loop overlay moved a value
    # off the thinking switch. The settings panel labels blank fields with
    # these instead of the word "auto".
    sampler_defaults: Dict[str, Any] = Field(default_factory=dict)
    stale_reload_fields: List[str] = Field(default_factory=list)
    effective_loader: Optional[Literal["mlx-lm", "mlx-vlm"]] = Field(
        default=None,
        description="Which MLX library actually decodes this model -- mlx-lm "
                    "(text) or mlx-vlm (vision). Null for every non-mlx "
                    "provider: gguf is one engine, already named by "
                    "`provider`. DERIVED from the config (loader + "
                    "modalities + the model dir's model_type), so it is "
                    "answered for UNLOADED models too -- provider `mlx` is "
                    "two separate upstream repos, and this is the only field "
                    "that says which one a row means.",
    )
    context_length: Optional[int] = Field(
        default=None,
        description="The model's context window in tokens, from the one "
                    "resolver every surface reads (gguf: `<arch>.context_length` "
                    "in the GGUF header, the training context llama-server "
                    "sizes from when `ctx_size` is unset; mlx: config.json's "
                    "max_position_embeddings, v1.79.65). The ceiling a "
                    "context-size control offers, and what an over-length "
                    "prompt is refused against. Null when the files do not "
                    "say and for providers with no chat context. DERIVED, so "
                    "it is answered for unloaded models.",
    )
    context_running: Optional[int] = Field(
        default=None,
        description="The context the RESIDENT llama-server process was "
                    "actually sized to (its slot `n_ctx`, read from /props at "
                    "ready). `config.ctx_size` is what was asked; absent "
                    "means llama-server chose from the model and memory, and "
                    "this is what it chose. Null for unloaded models and for "
                    "every non-gguf provider.",
    )
    thinking_default: bool = Field(
        default=False,
        description="What thinking resolves to for this model when a request "
                    "says nothing about it: the sampling cascade's own answer "
                    "for an empty request (`config.enable_thinking`, else "
                    "whether the model can think at all). DERIVED -- answered for unloaded "
                    "models -- and the value a UI's 'model default' choice "
                    "actually means. False for a model without the thinking "
                    "capability.",
    )


class AdminModelListResponse(BaseModel):
    """Response for listing all model configs."""
    models: List[AdminModelResponse] = Field(default_factory=list)
    total: int = 0


class ChatTemplateResponse(BaseModel):
    """What chat template a model resolves to, and whether an edit reaches it.

    Answers for models that are NOT resident -- the prompt format a model will
    load with is exactly the thing worth seeing before loading it -- so every
    field here comes from files, never from a running process. The one
    exception is ``stale``, which needs a loaded model to mean anything and is
    ``null`` without one.
    """
    model_id: str
    provider: str
    template: Optional[str] = Field(
        default=None, description="The resolved template body, or null if the model has none.")
    origin: str = Field(
        description="Which rung of the ladder produced it -- the same phrase the load log uses.")
    override_present: bool = False
    override_path: Optional[str] = Field(
        default=None, description="Where the override lives, whether or not it exists yet.")
    writable: bool = Field(
        default=False, description="Whether the model folder accepts a write.")
    inert_reason: Optional[str] = Field(
        default=None,
        description=("Set when an override exists but something outranks it. An "
                     "editor whose writes go nowhere is worse than no editor, so "
                     "this is a field rather than a note."))
    override_template: Optional[str] = Field(
        default=None,
        description=("The override file's OWN body, whether or not it won. "
                     "`template` is what the model RENDERS with; these differ "
                     "exactly when an override exists but lost the ladder, and "
                     "an editor must show this one -- otherwise a rejected "
                     "template is unreadable from the surface that wrote it."))
    stale: Optional[bool] = Field(
        default=None,
        description=("True when the LOADED model renders with something other than "
                     "what is on disk now (edited since load -- reload to apply). "
                     "null when the model is not loaded, which is NOT the same as false."))
    refused_shapes: List[str] = Field(
        default_factory=list,
        description=("Conversation shapes the template just written REFUSES to render. "
                     "Empty on every read path -- only a write validates, and an empty "
                     "list there means it rendered them all. A refused shape is legal "
                     "and does not block the write, but it is the thing worth seeing: "
                     "a template that raises on a system message saves cleanly and then "
                     "fails at generation for every conversation that has a system "
                     "prompt, which is the default."))
    notes: List[str] = Field(default_factory=list)


class ChatTemplateUpdateRequest(BaseModel):
    """Write a model's chat-template override."""
    template: str = Field(
        description="The jinja body. Validated before anything touches disk.")


class FitRequest(BaseModel):
    """Evaluate whether a model (with candidate config edits) fits memory.

    ``config_overrides`` are applied over the STORED config before sizing --
    the editor can ask about an unsaved candidate. ``null`` on a key means
    reset-to-default (drop the stored value), matching the PATCH contract.
    Keys that don't affect sizing are ignored, not rejected: this endpoint
    answers "does it fit", not "is it valid" (that's /validate).
    """
    config_overrides: Dict = Field(default_factory=dict)
    headroom_gb: float = Field(default=8.0, ge=0, le=256,
                               description="GiB to leave for KV cache + compute buffers")


class FitLineResponse(BaseModel):
    """One ceiling's row of the fit table."""
    ceiling: str   # reclaimable_ram | metal_working_set | metal_max_buffer
    verdict: str   # pass | warn | fail
    need_gb: float
    have_gb: float
    note: str = ""


class FitResponse(BaseModel):
    """Server-computed fit verdict (heylook_llm.ram_fit) -- the UI renders
    this and NEVER computes fit client-side. ``hard_working_set`` is the
    provider-derived engine asymmetry: over the Metal working set is FAIL
    for MLX (refuses above the recommendation) but WARN for gguf
    (llama.cpp loads past it and degrades into paging)."""
    weights_gb: float
    headroom_gb: float
    reclaimable_gb: float
    working_set_gb: Optional[float] = None   # None off Metal
    max_buffer_gb: Optional[float] = None
    sysctl_wired_mb: Optional[int] = None
    # Set ONLY while iogpu.wired_limit_mb is at its OS default (0) AND the
    # working set is exceeded; once raised, the ceiling is a deliberate
    # choice and the hint would be noise. The UI shows the sysctl line iff
    # this is non-null.
    sysctl_suggest_mb: Optional[int] = None
    kv_headroom_gb: Optional[float] = None   # working set minus weights
    # kv_headroom_gb under ram_fit.THIN_HEADROOM_GB: the gguf provider spawns
    # with llama-server's own micro-batch rather than the larger auto value,
    # and a decode-time Metal OOM at full context is a live possibility. The
    # UI says so; sysctl_suggest_mb carries the remedy while it applies.
    headroom_thin: bool = False
    hard_working_set: bool
    verdict: str                             # pass | warn | fail (worst line)
    lines: List[FitLineResponse] = Field(default_factory=list)
    sizing_notes: List[str] = Field(default_factory=list)
    # All numbers are measured today (file sizes, device properties, vm_stat).
    # Flips when a component becomes an approximation (e.g. offload deltas);
    # the UI must render estimates in a different visual register.
    estimated: bool = False
