# src/heylook_llm/schema/messages.py
#
# Message types and the core MessageCreateRequest model.
# Inspired by Anthropic Messages API with extensions for heylookitsanllm
# features (thinking).

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from heylook_llm.config import ThinkingDepth
from heylook_llm.stop_sequences import MAX_STOP_SEQUENCE_CHARS, MAX_STOP_SEQUENCES
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


class ThinkingConfig(BaseModel):
    """Anthropic's ``thinking`` object, with two heylook relaxations: both
    fields are optional. An absent ``type`` keeps the model's own thinking
    default (as an absent bool does), so a budget can be set without also
    deciding the switch; an absent ``budget_tokens`` means no cap."""
    type: Optional[Literal["enabled", "disabled"]] = None
    budget_tokens: Optional[int] = Field(default=None, ge=1)


class MessageCreateRequest(BaseModel):
    """Request body for POST /v1/messages.

    Differences from the current ChatRequest (OpenAI format):
    - system is a top-level parameter, not in the messages array
    - content uses typed blocks instead of OpenAI's content_parts
    - thinking may be a bool as well as Anthropic's object form
    - no batch-processing or image-resize params: Messages clients resize
      before sending (the server-side resize left with the OpenAI chat
      route in v1.79.66)
    """
    model: Optional[str] = Field(
        default=None,
        description="Model ID. Required in practice: a request naming no model is a 400 listing the ids.",
    )
    messages: List[Message]
    system: Optional[str] = Field(
        default=None, description="System prompt. Kept out of messages array for clarity."
    )
    stop_sequences: Optional[List[Annotated[str, StringConstraints(
        min_length=1, max_length=MAX_STOP_SEQUENCE_CHARS)]]] = Field(
        default=None, max_length=MAX_STOP_SEQUENCES,
        description=(
            "Strings that end the reply when it produces one: the reply is cut "
            "before the match, stop_reason is stop_sequence and stop_sequence "
            "names it. Matched on the reply text only, never on thinking, and "
            "the same way on every engine (heylook_llm/stop_sequences.py)."))
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
    # No stream_options (removed 2026-09-25): include_usage was copied onto
    # the provider request and read by nothing, and a stream always carries
    # usage in message_delta. A client that still sends it gets exactly
    # that, so it is ignored rather than refused.

    # Thinking: heylook's bool, or Anthropic's object form, which also carries
    # the hard budget (plan W7). Both mean the same switch; the converter
    # flattens them onto ChatRequest.enable_thinking / thinking_budget_tokens.
    thinking: Optional[Union[bool, ThinkingConfig]] = Field(
        default=None,
        description="Thinking on or off: a bool, or Anthropic's "
                    '{"type": "enabled"|"disabled", "budget_tokens": N}. '
                    "budget_tokens is a hard cap the engine enforces (models "
                    "with the thinking_budget capability; a 400 elsewhere). "
                    "Absent = the model's default.")
    # Same type as ChatRequest.reasoning_effort -- shared alias, so the two
    # schemas cannot drift into accepting different values.
    reasoning_effort: Optional[ThinkingDepth] = Field(
        default=None,
        description="Thinking depth, in the model's own template spelling: "
                    "one of `engine.thinking.depth.values` (or an alias) on "
                    "/v1/models. Sent as the template's own depth variable. "
                    "A value the model does not offer is a 400. Absent = the "
                    "template's own default.",
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
        # WRONG SPELLINGS of fields this wire really has. Not removals -- the
        # capability is right here under another name -- which makes the silent
        # drop worse than the removals above: the client asked for a control
        # that exists, got a normal 200, and quietly received the cascade
        # default instead. Found 2026-09-20 by a client author who had to read
        # the live schema to discover `enable_thinking` is not a wire field;
        # their note was that this is the one failure a test cannot easily
        # catch, because the request SUCCEEDS.
        #
        # Each of these is a habit with an obvious origin, which is why they
        # are worth naming rather than left to `extra="forbid"`: `max_new_tokens`
        # is transformers' spelling and is what Qwen's own reference runners
        # use, `enable_thinking` is heylook's INTERNAL name (heylook.toml, the
        # provider configs, chat_template_kwargs), and `system_prompt` is what
        # the preset store calls it. A blanket forbid would also 422 an
        # Anthropic SDK sending fields we simply do not implement, which is a
        # different and less helpful failure.
        renamed = {
            "enable_thinking": "thinking",
            "max_new_tokens": "max_tokens",
            "system_prompt": "system",
        }
        misspelt = [(k, v) for k, v in renamed.items() if k in data]
        if misspelt:
            raise ValueError(
                "; ".join(
                    f"`{k}` is not a field on this API -- send `{v}` instead "
                    f"(`{k}` is the internal/config spelling, and pydantic would "
                    f"otherwise DROP it silently and answer with the default)"
                    for k, v in misspelt
                )
            )
        # `chat_template_kwargs` is llama-server's (OpenAI-extension) spelling
        # of the template variables, and heylook's own gguf provider sends it
        # -- which is how it leaked into client code and probes (2026-09-23:
        # a heylook harness sent it for weeks, got 200s, and never once turned
        # thinking off). This wire takes the two variables it supports as
        # fields of their own.
        if "chat_template_kwargs" in data:
            raise ValueError(
                "`chat_template_kwargs` is not a field on this API -- send "
                "`thinking` (bool) and/or `reasoning_effort` as top-level fields "
                "instead (pydantic would otherwise DROP it silently and answer "
                "with the default)")
        # Not built yet (owner call 2026-09-25: response_format next, tools
        # when a client needs them). Both engines could serve them, which is
        # why they are refused loudly rather than dropped: a client asking
        # for a schema-shaped reply must not get free text with a 200.
        unbuilt = [k for k in ("tools", "tool_choice", "response_format") if k in data]
        if unbuilt:
            raise ValueError(
                f"{', '.join(unbuilt)} is not supported on this server yet: tool "
                "use and structured output are not built. Send the request without "
                "it (pydantic would otherwise DROP it silently and answer with "
                "free text)")
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
