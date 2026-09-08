# src/heylook_llm/schema/messages.py
#
# Message types and the core MessageCreateRequest model.
# Inspired by Anthropic Messages API with extensions for heylookitsanllm
# features (thinking, hidden states, batch).

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from heylook_llm.config import ReasoningEffort
from heylook_llm.schema.content_blocks import InputContentBlock, TextBlock


class Message(BaseModel):
    """A single message in a conversation.

    Content can be a plain string (convenience) or a list of typed content
    blocks (for multimodal messages with images).
    """
    role: Literal["user", "assistant"]
    content: Union[str, List[InputContentBlock]]

    def text_content(self) -> str:
        """Extract plain text from content, regardless of format."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            block.text for block in self.content
            if isinstance(block, TextBlock)
        )


class StreamOptions(BaseModel):
    """Options that control streaming behavior."""
    include_usage: bool = Field(
        default=False, description="Include token usage statistics in the final stream event"
    )


class MessageCreateRequest(BaseModel):
    """Request body for POST /v1/messages.

    Differences from the current ChatRequest (OpenAI format):
    - system is a top-level parameter, not in the messages array
    - content uses typed blocks instead of OpenAI's content_parts
    - thinking is a top-level bool instead of enable_thinking
    - no batch-processing or image-resize params: Messages clients resize
      before sending (the server-side resize left with the OpenAI chat
      route in v1.79.66)
    """
    model: Optional[str] = Field(
        default=None,
        description="Model ID. If omitted, uses loaded model or default_model from config.",
    )
    messages: List[Message]
    system: Optional[str] = Field(
        default=None, description="System prompt. Kept out of messages array for clarity."
    )
    # Tri-state like every other knob (deliberately unlike Anthropic's
    # required max_tokens): absent = the server-side sampler cascade's
    # default. A hard 1024 default here overrode the cascade for every
    # client that simply omitted the field -- the exact knob-loss Phase 3b
    # migration guards against.
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.1, le=2.0)
    repetition_context_size: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    seed: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None

    # Thinking mode (Qwen3 models)
    thinking: Optional[bool] = Field(
        default=None, description="Enable thinking mode for models that support it (e.g. Qwen3)"
    )
    # Same vocabulary as ChatRequest.reasoning_effort -- shared alias, so the
    # two APIs cannot drift into accepting different value sets. Phase 3b is
    # migrating v3 onto this API, so a knob missing here is a control the next
    # surface to migrate silently loses.
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="Thinking depth when thinking is on. Values are "
                    "MODEL-SPECIFIC (Qwen3.8: xhigh|medium|low; harmony: "
                    "low|medium|high). Absent = the template's own default.",
    )

    # heylook extensions (Phase 3b namespace) -- same semantics and bounds as
    # the internal ChatRequest, so no sampler knob exists that this wire cannot
    # reach. `vision_tokens` is the per-image visual token budget.
    vision_tokens: Optional[int] = Field(
        default=None, ge=16, le=16384,
        description="Target visual tokens per image; snapped to what the "
                    "model's processor supports",
    )

    # NO `include_performance` here, deliberately (removed v1.79.49). This wire
    # returns telemetry UNCONDITIONALLY in both modes -- streaming emits
    # `message_stop.performance`, non-streaming carries a `performance` object
    # -- so the flag controlled nothing on the surface that declared it. Gating
    # only the non-streaming half would have split the two modes against each
    # other, and gating both would break v3's status lines, which read
    # message_stop.performance on every generation. (The OpenAI chat route,
    # which honoured such a flag, was removed in v1.79.66.)

    # Metadata passthrough
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Arbitrary metadata passed through to the response"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_retired_request_fields(cls, data):
        """Refuse fields that were REMOVED, rather than dropping them silently.

        This has to live HERE, on the wire model, not on the internal
        ChatRequest. `/v1/messages` is the only inference route left (the
        OpenAI-compatible one went in v1.79.66) and it binds this model;
        nothing binds ChatRequest as a request body any more, and converters
        build it from explicit kwargs. A guard over there cannot see a client.
        Pydantic's default `extra` policy is *ignore*, so without this a client
        asking for a removed feature gets a normal 200 and no hint -- exactly
        the "answered as though it had not asked" outcome the guard exists to
        prevent. Found by review 2026-09-06, after v1.79.74 put it on the
        wrong model and a test that built ChatRequest directly went green.
        """
        if not isinstance(data, dict):
            return data
        gone = [k for k in ("logprobs", "top_logprobs") if k in data]
        if gone:
            raise ValueError(
                f"{', '.join(gone)} is no longer supported: logprobs were removed "
                "with the token explorer (the only surface that read them) and the "
                "heylook_logprobs SSE extension is gone with them"
            )
        # Only a TRUTHY value is refused, which is where this differs from the
        # `logprobs` guard above: any logprobs value asked for a capability that
        # is gone, but `show_special_tokens: false` asked for exactly what the
        # server now always does. Refusing it would 422 the one client shape
        # that needed no change at all, with a message telling it the behaviour
        # it requested is the behaviour in force. Loud beats silent only when
        # the client would otherwise be answered as though it had not asked.
        if data.get("show_special_tokens"):
            raise ValueError(
                "show_special_tokens=true is no longer supported: it was a "
                "per-BROWSER display pref that decided what the conversation "
                "store PERSISTED, so the same conversation continued from two "
                "devices accumulated rows of two kinds with nothing recording "
                "which was which. Declared specials are now always stripped, so "
                "sending false (or omitting it) is what this server does. See "
                "docs/project/TODO.md for the design a correct version would take"
            )
        if "preset" in data or "sampler" in data:
            raise ValueError(
                "named sampler bundles were removed in v2.0.30 -- send the "
                "sampler fields themselves (temperature, top_p, ...). "
                "/v1/presets user presets are a separate, still-supported "
                "system, and the client expands one into those same fields"
            )
        return data

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v
