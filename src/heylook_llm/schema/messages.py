# src/heylook_llm/schema/messages.py
#
# Message types and the core MessageCreateRequest model.
# Inspired by Anthropic Messages API with extensions for heylookitsanllm
# features (thinking, logprobs, hidden states, batch).

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

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
        False, description="Include token usage statistics in the final stream event"
    )


class MessageCreateRequest(BaseModel):
    """Request body for POST /v1/messages.

    Differences from the current ChatRequest (OpenAI format):
    - system is a top-level parameter, not in the messages array
    - content uses typed blocks instead of OpenAI's content_parts
    - thinking is a top-level bool instead of enable_thinking
    - no processing_mode/return_individual (those move to BatchRequest)
    - no image resize params (handled by /v1/messages/multipart)
    """
    model: Optional[str] = Field(
        None,
        description="Model ID. If omitted, uses loaded model or default_model from config.",
    )
    messages: List[Message]
    system: Optional[str] = Field(
        None, description="System prompt. Kept out of messages array for clarity."
    )
    max_tokens: int = Field(1024, gt=0)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.1, le=2.0)
    repetition_context_size: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(None, ge=0.0, le=2.0)
    seed: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None

    # Thinking mode (Qwen3 models)
    thinking: Optional[bool] = Field(
        None, description="Enable thinking mode for models that support it (e.g. Qwen3)"
    )

    # Logprobs
    logprobs: Optional[bool] = Field(
        None, description="Return log probabilities for output tokens"
    )
    top_logprobs: Optional[int] = Field(
        None, ge=0, le=20,
        description="Number of top token alternatives with log probabilities (0-20)",
    )

    # Performance
    include_performance: bool = Field(
        False, description="Include performance metrics (tps, memory) in response"
    )

    # Metadata passthrough
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Arbitrary metadata passed through to the response"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v
