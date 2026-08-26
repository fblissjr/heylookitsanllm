# src/heylook_llm/schema/messages.py
#
# Message types and the core MessageCreateRequest model.
# Inspired by Anthropic Messages API with extensions for heylookitsanllm
# features (thinking, logprobs, hidden states, batch).

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

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
        False, description="Include token usage statistics in the final stream event"
    )


class MessageCreateRequest(BaseModel):
    """Request body for POST /v1/messages.

    Differences from the current ChatRequest (OpenAI format):
    - system is a top-level parameter, not in the messages array
    - content uses typed blocks instead of OpenAI's content_parts
    - thinking is a top-level bool instead of enable_thinking
    - no processing_mode/return_individual (those move to BatchRequest)
    - no image resize params (Messages clients resize before sending; the
      OpenAI wire keeps them for clients that want the server to do it)
    """
    model: Optional[str] = Field(
        None,
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

    # Logprobs
    logprobs: Optional[bool] = Field(
        default=None, description="Return log probabilities for output tokens"
    )
    top_logprobs: Optional[int] = Field(
        default=None, ge=0, le=20,
        description="Number of top token alternatives with log probabilities (0-20)",
    )

    # heylook extensions (Phase 3b namespace) -- same semantics and bounds as
    # ChatRequest, so a consumer migrating from /v1/chat/completions loses no
    # knob. `sampler` is the SamplerRegistry bundle name (never a /v1/presets
    # id); `vision_tokens` is the per-image visual token budget.
    sampler: Optional[str] = Field(
        default=None,
        description="Named sampler bundle (e.g. 'balanced', 'thinking'). "
                    "Resolved against the server's SamplerRegistry.",
    )
    vision_tokens: Optional[int] = Field(
        default=None, ge=16, le=16384,
        description="Target visual tokens per image; snapped to what the "
                    "model's processor supports",
    )

    # Display honesty (DESIGN.md §6): the server strips the specials a model
    # DECLARES (`special: true` in its tokenizer files) out of the text it
    # returns, as a guard against fast-detokenizer leaks. That guard also
    # deletes a special the model wrote deliberately -- and those say where in
    # the turn the model is, which is the thing an interpretability surface
    # exists to show. Opt IN to keep them: false -- the default, and what every
    # existing consumer sends by omitting the field -- strips, as before.
    # Deliberately NOT tri-state: nothing distinguishes absent from false, and a
    # third state no branch reads is one every future call site has to re-derive
    # as meaningless (`include_performance` below is the in-file precedent). Affects the text returned (and, on the
    # conversation surface, the text PERSISTED), never what is sent to the
    # model and never generation itself.
    show_special_tokens: bool = Field(
        default=False,
        description="Return the model's declared special tokens instead of "
                    "stripping them (e.g. <|im_end|>, <bos>). Display-only: "
                    "changes the text you get back, never the generation. "
                    "Text you send is never altered either way -- but text you "
                    "send BACK carries whatever you kept, so a client that "
                    "replays a reply verbatim is replaying control-token "
                    "strings (the conversation surface strips those on replay; "
                    "see conversation_generate_api._strip_history_specials).",
    )

    # Performance
    include_performance: bool = Field(
        default=False, description="Include performance metrics (tps, memory) in response"
    )

    # Metadata passthrough
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Arbitrary metadata passed through to the response"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v
