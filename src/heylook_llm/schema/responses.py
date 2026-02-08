# src/heylook_llm/schema/responses.py
#
# Response models for the new Messages API.
# The response uses typed content blocks instead of a flat choices array.

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from heylook_llm.schema.content_blocks import OutputContentBlock


class Usage(BaseModel):
    """Token usage statistics.

    Extends OpenAI's usage with thinking-specific token counts.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: Optional[int] = Field(
        None, description="Tokens used in thinking blocks (Qwen3)"
    )
    content_tokens: Optional[int] = Field(
        None, description="Tokens in non-thinking content"
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
    peak_memory_gb: Optional[float] = Field(None, description="Peak memory usage in GB")
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
    stop_reason: Literal["stop", "length", "error"] = "stop"
    usage: Usage
    performance: Optional[PerformanceInfo] = None
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Echoed from request metadata"
    )
