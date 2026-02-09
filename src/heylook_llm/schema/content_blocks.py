# src/heylook_llm/schema/content_blocks.py
#
# Content block types for the new Messages API.
# Input blocks (user->model) and output blocks (model->user) are distinct unions
# because certain block types only appear in one direction.

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input content blocks (appear in user messages)
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    """Plain text content block. Used in both input and output."""
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image content block for vision models.

    Supports base64-encoded data or a URL reference. Mirrors the existing
    multipart image handling but in a structured block format.
    """
    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"] = Field(
        ..., description="How the image data is provided"
    )
    media_type: Optional[str] = Field(
        None, description="MIME type, e.g. 'image/jpeg'. Required for base64."
    )
    data: Optional[str] = Field(
        None, description="Base64-encoded image data (when source_type='base64')"
    )
    url: Optional[str] = Field(
        None, description="Image URL (when source_type='url')"
    )


# Union of all block types that can appear in a user message
InputContentBlock = Union[TextBlock, ImageBlock]


# ---------------------------------------------------------------------------
# Output content blocks (appear in assistant responses)
# ---------------------------------------------------------------------------

class ThinkingBlock(BaseModel):
    """Model reasoning/thinking content (Qwen3 <think> blocks).

    Separated from the main text so frontends can display thinking in a
    collapsible section or hide it entirely.
    """
    type: Literal["thinking"] = "thinking"
    text: str


class TokenLogprob(BaseModel):
    """Log probability information for a single token."""
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class TopLogprob(BaseModel):
    """A candidate token with its log probability."""
    token: str
    token_id: Optional[int] = None
    logprob: float
    bytes: Optional[List[int]] = None


class TokenLogprobEntry(BaseModel):
    """Full logprob entry for one generated token position."""
    token: str
    token_id: Optional[int] = None
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[TopLogprob] = Field(default_factory=list)


class LogprobsBlock(BaseModel):
    """Token-level log probability data for a generation.

    Returned when `logprobs: true` is set in the request. Contains per-token
    probability information for the generated sequence.
    """
    type: Literal["logprobs"] = "logprobs"
    tokens: List[TokenLogprobEntry] = Field(default_factory=list)


class HiddenStatesBlock(BaseModel):
    """Hidden states extraction results.

    Returned by /v1/hidden_states endpoints. Contains the raw activation
    vectors from a specified model layer, with token boundary information
    for mapping back to input tokens.
    """
    type: Literal["hidden_states"] = "hidden_states"
    layer: int = Field(..., description="Layer index the states were extracted from")
    shape: List[int] = Field(..., description="Tensor shape [seq_len, hidden_dim]")
    token_boundaries: Optional[List[Dict]] = Field(
        None, description="Token-to-position mapping for the hidden states"
    )
    # Actual tensor data is too large for JSON; this block carries metadata.
    # The raw data is returned as a separate binary payload or base64 field.
    data_encoding: Literal["base64", "external"] = Field(
        "external", description="How the hidden state data is provided"
    )
    data: Optional[str] = Field(
        None, description="Base64-encoded hidden states (when data_encoding='base64')"
    )


# Union of all block types that can appear in an assistant response
OutputContentBlock = Union[TextBlock, ThinkingBlock, LogprobsBlock, HiddenStatesBlock]
