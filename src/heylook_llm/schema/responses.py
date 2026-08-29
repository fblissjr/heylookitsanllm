# src/heylook_llm/schema/responses.py
#
# Response models for the new Messages API.
# The response uses typed content blocks instead of a flat choices array.

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from heylook_llm.schema.content_blocks import OutputContentBlock

# The Messages-wire stop vocabulary, defined ONCE. Anthropic's set minus the
# values heylook cannot produce (no tools, no server-side pause), plus
# `error`, which has no Anthropic counterpart -- a failure is an `error`
# EVENT there, and api.py's non-streaming failure path needs a value.
StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "error"]


class Usage(BaseModel):
    """Token usage statistics.

    Extends OpenAI's usage with thinking-specific token counts.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: Optional[int] = Field(
        default=None, description="Tokens used in thinking blocks (Qwen3)"
    )
    content_tokens: Optional[int] = Field(
        default=None, description="Tokens in non-thinking content"
    )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PerformanceInfo(BaseModel):
    """Generation performance metrics.

    Returned when include_performance=true in the request.
    """
    prompt_tps: float = Field(..., description="Prompt processing tokens per second")
    generation_tps: float = Field(..., description="Generation tokens per second")
    peak_memory_gb: Optional[float] = Field(default=None, description="Peak memory usage in GB")
    thinking_duration_ms: Optional[int] = Field(
        None, description="Time spent in thinking phase"
    )
    content_duration_ms: Optional[int] = Field(
        None, description="Time spent generating content"
    )
    total_duration_ms: Optional[int] = Field(
        None, description="Total generation time"
    )


class MessageResponse(BaseModel):
    """Response from POST /v1/messages (non-streaming).

    Content is a list of typed blocks. A simple text response is
    [TextBlock(text="...")]. A thinking model might return
    [ThinkingBlock(text="..."), TextBlock(text="...")].
    Logprobs, if requested, appear as a LogprobsBlock at the end.
    """
    id: str = Field(..., description="Unique message ID, prefixed 'msg_'")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[OutputContentBlock]
    # Anthropic's vocabulary, not OpenAI's. This read "stop"|"length"|"error"
    # until v1.79.39 -- the provider's OpenAI finish_reason passed through
    # unchanged onto an otherwise Anthropic-shaped response, so a client
    # written against the Messages API saw a value its spec does not define.
    # `error` has no Anthropic counterpart (a failure is an `error` EVENT
    # there) and is kept as a heylook extension: api.py sets it on the
    # non-streaming failure path. Map through STOP_REASON_FROM_FINISH_REASON
    # (converters.py) rather than assigning a provider value here.
    stop_reason: StopReason = "end_turn"
    usage: Usage
    performance: Optional[PerformanceInfo] = None
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Echoed from request metadata"
    )
