# src/heylook_llm/schema/streaming.py
#
# Structured streaming event types for SSE responses.
#
# The current OpenAI format sends a single event type ("chat.completion.chunk")
# with the delta buried in choices[0].delta. The new format uses distinct event
# types so the frontend can switch on event.type without inspecting the delta
# object.
#
# SSE wire format:
#   event: message_start
#   data: {"type": "message_start", "message": {...}}
#
#   event: content_block_start
#   data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "text": ""}}
#
#   event: content_block_delta
#   data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "...", "text": "..."}}
#
#   event: content_block_stop
#   data: {"type": "content_block_stop", "index": 0}
#
#   event: message_delta
#   data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}
#
#   event: message_stop
#   data: {"type": "message_stop", "performance": {...}}

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

from heylook_llm.schema.responses import PerformanceInfo, Usage


# ---------------------------------------------------------------------------
# Delta types (the "delta" field inside content_block_delta events)
# ---------------------------------------------------------------------------

class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    # BOTH spellings ride the wire (messages_api._block_delta): `thinking` is
    # Anthropic's field name, `text` is heylook's original and is what v3's
    # streaming.js reads. These models are DOCUMENTARY -- nothing builds or
    # validates an event through them -- which is exactly why they drifted
    # for one commit after the emitter changed.
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str = ""
    text: str = ""


class LogprobsDelta(BaseModel):
    """Incremental logprob data for one token position."""
    type: Literal["logprobs_delta"] = "logprobs_delta"
    token: str
    token_id: Optional[int] = None
    logprob: float
    top_logprobs: Optional[list] = None


ContentDelta = Union[TextDelta, ThinkingDelta, LogprobsDelta]


# ---------------------------------------------------------------------------
# Stream event types
# ---------------------------------------------------------------------------

class ContentBlockInfo(BaseModel):
    """Minimal info about a content block when it starts.

    The block opens with its content field present and empty, as Anthropic's
    does, so a client accumulating into `content_block` has something to
    accumulate into. A thinking block carries both spellings.
    """
    type: str  # "text", "thinking", "logprobs"
    text: str = ""
    thinking: str = ""


class MessageStartEvent(BaseModel):
    """First event in a stream. Contains the message shell with empty content."""
    type: Literal["message_start"] = "message_start"
    message: Dict = Field(
        ...,
        description=(
            "Partial MessageResponse with id, model, role, empty content[], "
            "and input token usage"
        ),
    )


class ContentBlockStartEvent(BaseModel):
    """Signals the beginning of a new content block."""
    type: Literal["content_block_start"] = "content_block_start"
    index: int = Field(..., description="Block index in the content array")
    content_block: ContentBlockInfo


class ContentBlockDeltaEvent(BaseModel):
    """Incremental update within a content block (token-by-token)."""
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int = Field(..., description="Block index this delta belongs to")
    delta: ContentDelta


class ContentBlockStopEvent(BaseModel):
    """Signals a content block is complete."""
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """Final metadata: stop reason and usage stats."""
    type: Literal["message_delta"] = "message_delta"
    # The vocabulary is StopReason (responses.py) -- "error" is NOT a member
    # and never was reachable; it was declared on an untraced claim and removed
    # in v1.79.40, but this description kept it one file over. These models are
    # documentary (nothing builds or validates an event through them, and
    # MessageDeltaEvent is not in components.schemas), so no client branched on
    # it -- a code reader was the one being misled, which is the readership
    # this class exists for.
    delta: Dict = Field(
        ..., description='{"stop_reason": "end_turn"|"max_tokens"|"stop_sequence"}'
    )
    usage: Optional[Usage] = None


class MessageStopEvent(BaseModel):
    """Terminal event. Carries optional performance data."""
    type: Literal["message_stop"] = "message_stop"
    performance: Optional[PerformanceInfo] = None


# Union of all stream event types for type narrowing
StreamEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
]
