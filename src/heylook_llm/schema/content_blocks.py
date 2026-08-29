# src/heylook_llm/schema/content_blocks.py
#
# Content block types for the new Messages API.
# Input blocks (user->model) and output blocks (model->user) are distinct unions
# because certain block types only appear in one direction.

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


def _flatten_source(values: Any) -> Any:
    """Accept Anthropic's nested ``source`` object on a media block.

    Anthropic spells an image as ``{"type":"image","source":{"type":"base64",
    "media_type":...,"data":...}}`` (or ``{"type":"url","url":...}``); this
    schema stores those fields flat. Both spellings are accepted -- the flat
    one because every existing heylook client sends it, the nested one
    because this endpoint advertises itself as Messages-shaped and an
    Anthropic SDK, or anyone reading Anthropic's docs, sends that. It used to
    be a 422, which is a poor answer to a correctly-formed Messages request.

    Normalizing HERE rather than at the two consumers keeps the rest of the
    codebase (converters.py, media handling) on one shape.
    """
    if not isinstance(values, dict):
        return values
    # Only interpret `source` when the block is NOT already in the flat
    # spelling. Without this gate a flat block carrying an unrelated
    # dict-valued `source` key -- provenance metadata, say, previously
    # ignored as an extra field -- would have it deleted and its `type`
    # read as the discriminator. That was harmless only by accident (the
    # flat fields were present, so every setdefault below no-opped); one
    # missing flat field turned it into `source_type=None` and a 422
    # pointing at the wrong field.
    if "source_type" in values:
        return values
    source = values.get("source")
    if not isinstance(source, dict):
        return values
    values = {k: v for k, v in values.items() if k != "source"}
    # Anthropic's discriminator is `source.type`; ours is `source_type`.
    values.setdefault("source_type", source.get("type"))
    for key in ("media_type", "data", "url"):
        if source.get(key) is not None:
            values.setdefault(key, source[key])
    return values


class MediaSource(BaseModel):
    """Anthropic's nested media source object.

    Declared as a real field rather than left to the ``mode="before"``
    validator alone: a validator contributes NOTHING to the generated JSON
    Schema, so /openapi.json advertised only the flat fields while the docs
    told integrators to prefer the nested form. A client generated from the
    schema, or any schema-validating proxy, would have rejected the exact
    spelling the docs recommend.
    """
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


def _require_source_type(self):
    """`source_type` is optional in the SCHEMA and mandatory in FACT.

    Optional so a nested-`source` payload validates against /openapi.json;
    mandatory here because everything downstream reads the flat fields.
    """
    if self.source_type is None:
        raise ValueError(
            f"{type(self).__name__} requires `source_type`, or a `source` "
            "object carrying `type`"
        )
    return self


# ---------------------------------------------------------------------------
# Input content blocks (appear in user messages)
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    """Plain text content block. Used in both input and output."""
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image content block for vision models.

    ACCEPTS TWO SPELLINGS. Anthropic's nested form is preferred and is what
    an Anthropic SDK sends::

        {"type": "image",
         "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
        {"type": "image", "source": {"type": "url", "url": "https://..."}}

    heylook's original flat form -- ``source_type``/``media_type``/``data``
    directly on the block, the fields documented below -- is still accepted
    and is what the properties in this schema describe. A nested ``source``
    is flattened onto them before validation, so the two are interchangeable
    on the wire and identical everywhere downstream.

    ``data`` is raw base64 with no ``data:`` URI prefix.
    """
    type: Literal["image"] = "image"
    source_type: Optional[Literal["base64", "url"]] = Field(
        default=None,
        description="How the image data is provided. Required UNLESS the "
                    "Anthropic-style nested `source` object is given, from "
                    "which it is derived.",
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

    source: Optional[MediaSource] = Field(
        default=None,
        description="Anthropic's nested source object. Flattened onto the "
                    "fields above before validation; never echoed back.",
    )

    _flatten = model_validator(mode="before")(_flatten_source)
    _require = model_validator(mode="after")(_require_source_type)


class AudioBlock(BaseModel):
    """Audio content block (plan Phase 7d). Mirrors ImageBlock's shape.

    Accepts both spellings exactly as ImageBlock does -- Anthropic's nested
    ``source`` object, or the flat fields documented below.

    Bridged to the OpenAI-wire ``input_audio`` part by converters.py; only
    the gguf/llama-server provider consumes it (MLX rejects audio).
    """
    type: Literal["audio"] = "audio"
    source_type: Optional[Literal["base64", "url"]] = Field(
        default=None,
        description="How the audio data is provided. Required UNLESS the "
                    "Anthropic-style nested `source` object is given, from "
                    "which it is derived.",
    )
    media_type: Optional[str] = Field(
        default=None, description="MIME type, e.g. 'audio/wav'. Advisory -- codecs are sniffed."
    )
    data: Optional[str] = Field(
        default=None, description="Base64-encoded audio data (when source_type='base64')"
    )
    url: Optional[str] = Field(
        default=None, description="Audio URL (when source_type='url')"
    )

    source: Optional[MediaSource] = Field(
        default=None,
        description="Anthropic's nested source object. Flattened onto the "
                    "fields above before validation; never echoed back.",
    )

    _flatten = model_validator(mode="before")(_flatten_source)
    _require = model_validator(mode="after")(_require_source_type)


# Union of all block types that can appear in a user message
InputContentBlock = Union[TextBlock, ImageBlock, AudioBlock]


# ---------------------------------------------------------------------------
# Output content blocks (appear in assistant responses)
# ---------------------------------------------------------------------------

class ThinkingBlock(BaseModel):
    """Model reasoning/thinking content (Qwen3 <think> blocks).

    Separated from the main text so frontends can display thinking in a
    collapsible section or hide it entirely.

    CARRIES BOTH SPELLINGS. Anthropic's thinking block names the field
    ``thinking``; this one shipped as ``text``, so an Anthropic-shaped reader
    found the block and no content in it. Both are populated from whichever
    the constructor is given, because dropping ``text`` would break v3's
    existing readers -- ``thinking`` is the conformant name and the one new
    clients should read.
    """
    type: Literal["thinking"] = "thinking"
    text: str = ""
    thinking: str = ""

    @model_validator(mode="after")
    def _mirror(self) -> "ThinkingBlock":
        if self.thinking and not self.text:
            self.text = self.thinking
        elif self.text and not self.thinking:
            self.thinking = self.text
        elif not self.thinking and not self.text:
            # Both defaults exist only so the constructor can be given
            # either spelling. Neither being set means a reasoning block
            # carrying nothing, which renders as an empty thinking pane --
            # `text` was required before the dual spelling and this keeps
            # that guarantee.
            raise ValueError("ThinkingBlock requires 'thinking' or 'text'")
        return self


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
