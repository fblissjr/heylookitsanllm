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
    # PER FIELD, not per block, and "absent" means NULL-OR-MISSING everywhere.
    #
    # Two mistakes have been made here in a row, both the same shape: testing
    # whether a KEY EXISTS where the question is whether it carries a VALUE.
    # `source_type` is Optional in the published schema, so a client generated
    # from /openapi.json -- or any pydantic client doing model_dump() without
    # exclude_none -- serializes every unset flat field as an explicit null
    # beside the nested object:
    #   {"type":"image","source_type":null,"data":null,
    #    "source":{"type":"base64",...,"data":"..."}}
    # v1.79.40's gate (`"source_type" in values`) short-circuited on that and
    # 422'd the spelling the docs recommend. v1.79.41 fixed the gate and left
    # the SAME mistake in the whole-block early return: any set flat field
    # suppressed filling of all the others, so `source_type` alone (a client
    # that sets the discriminator and leaves the payload to `source`) resolved
    # the type and then dropped the image. Only a per-field rule is what the
    # contract in frontend_v3_spec.md §4 actually promises.
    source = values.get("source")
    if not isinstance(source, dict):
        return values
    declared = values.get("source_type")
    nested_type = source.get("type")
    # The one case where the nested object is ignored wholesale: the two
    # spellings DISAGREE about what kind of source this is. Mixing a base64
    # payload into a block that declares itself a url (or the reverse) would
    # build a block the caller never described. The flat spelling wins and the
    # nested one is left for field validation to accept or reject.
    if declared is not None and declared != nested_type:
        return values
    values = {k: v for k, v in values.items() if k != "source"}
    # Anthropic's discriminator is `source.type`; ours is `source_type`.
    if declared is None:
        values["source_type"] = nested_type
    for key in ("media_type", "data", "url"):
        if source.get(key) is not None and values.get(key) is None:
            values[key] = source[key]
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

    The PAYLOAD is checked here too, and that is the load-bearing half.
    Validating only the discriminator let a block declare itself base64 and
    carry no `data` (or url with no `url`) -- it validated clean, and then
    converters.py, which requires `source_type == "base64" and block.data`
    else `block.url`, hit its `continue` and dropped the block. The request
    answered 200, the text parts survived, and the model never saw the image:
    a confident answer about a picture that was never sent. A 422 naming the
    missing field is strictly better than a silent drop on a vision request,
    which is the whole reason the nested spelling was made to work at all.
    """
    if self.source_type is None:
        raise ValueError(
            f"{type(self).__name__} requires `source_type`, or a `source` "
            "object carrying `type`"
        )
    payload = "data" if self.source_type == "base64" else "url"
    if getattr(self, payload, None) is None:
        raise ValueError(
            f"{type(self).__name__} declares source_type="
            f"{self.source_type!r} but carries no `{payload}` -- it would be "
            "dropped silently instead of reaching the model"
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


class ThinkingBlock(BaseModel):
    """Model reasoning/thinking content (Qwen3 <think> blocks).

    Separated from the main text so frontends can display thinking in a
    collapsible section or hide it entirely.

    Defined AHEAD of the input union because it is both an input and an
    output block (see InputContentBlock below).

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


# Union of all block types that can appear in a user message. ThinkingBlock
# is in it (v1.79.63): an Anthropic-shaped client replays the assistant's
# thinking blocks in history -- Anthropic's own tool loop requires it -- and
# this server 422'd the whole request on the first one. They convert to
# ChatMessage.thinking, which the providers render the way each template
# takes it (reasoning_content / <think> reconstruction).
InputContentBlock = Union[TextBlock, ImageBlock, AudioBlock, ThinkingBlock]


# ---------------------------------------------------------------------------
# Output content blocks (appear in assistant responses)
# ---------------------------------------------------------------------------

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
