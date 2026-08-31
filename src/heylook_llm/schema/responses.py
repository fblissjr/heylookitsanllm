# src/heylook_llm/schema/responses.py
#
# Response models for the new Messages API.
# The response uses typed content blocks instead of a flat choices array.

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from heylook_llm.schema.content_blocks import OutputContentBlock

# The Messages-wire stop vocabulary, defined ONCE: Anthropic's set minus the
# values heylook cannot produce (no tools, no server-side pause).
#
# `error` is deliberately NOT here. v1.79.39 added it on the stated grounds
# that "api.py sets it on the non-streaming failure path", which was not
# true and was repeated into three documents before anyone traced it:
# MessageResponse has exactly one construction site (converters.to_stop_reason
# feeds it) and that function cannot return "error"; api.py's `stop_reason=
# "error"` is a kwarg to _maybe_log_request_event -- a JSONL telemetry field,
# not a response. A non-streaming failure RAISES HTTPException, so no
# MessageResponse is built at all. An unreachable enum member is worse than
# no member: an integrator writes and tests a branch the server cannot enter.
StopReason = Literal["end_turn", "max_tokens", "stop_sequence"]


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

    MESSAGES WIRE ONLY. The OpenAI wire returns a different model --
    ``config.PerformanceMetrics``, gated on ``include_performance`` -- and the
    previous version of this docstring described THAT model's behaviour here,
    which is the mechanism-that-does-not-exist defect the schema-parity test
    was repaired for in the same release. Referenced only by MessageResponse,
    MessageStopEvent and converters.from_openai_response_dict.

    SYMMETRY, as of v1.79.54-.55. The passage that stood here described the
    state this class was repaired OUT of -- "streaming does not send the two
    rates, which this class still declares required" -- eight lines above the
    field definitions that contradict both halves. That is the same
    declaration-only defect this model sits at the centre of, so: current
    truth, and check the fields below before trusting this paragraph.
    - non-streaming (``converters.from_openai_response_dict``) fills all nine,
      and the object is absent entirely when the run produced no tokens;
    - streaming (``messages_api.message_stop_event``, whose emitted key set is
      filtered to this model's declared fields) fills the same nine, minus
      whatever telemetry the run did not produce. The thinking/content
      durations remain streaming-only because nothing on the non-streaming
      path measures them -- the one deliberate asymmetry left.
    Every field is optional, so both payloads are subsets of one model.

    CAVEAT, a real difference and not this model's doing: the OTHER route on
    this grammar, ``POST /v1/conversations/{id}/generate``, calls
    ``message_stop_event()`` with NO timing, so its performance object carries
    the durations alone. That route's chunk telemetry rides
    ``heylook_saved.timing`` instead, which is what v3's chat page reads.
    """
    # OPTIONAL, not required (v1.79.54). They were declared required while the
    # STREAMING payload never sent them, so a client generated from
    # /openapi.json got two required fields that mode could not satisfy. Both
    # paths emit them now, but a run that produced no tokens has no rate to
    # report, and a required field the server sometimes cannot fill is a lie
    # in the other direction.
    prompt_tps: Optional[float] = Field(default=None, description="Prompt processing tokens per second")
    generation_tps: Optional[float] = Field(default=None, description="Generation tokens per second")
    peak_memory_gb: Optional[float] = Field(default=None, description="Peak memory usage in GB")
    thinking_duration_ms: Optional[int] = Field(
        default=None, description="Time spent in thinking phase"
    )
    content_duration_ms: Optional[int] = Field(
        default=None, description="Time spent generating content"
    )
    total_duration_ms: Optional[int] = Field(
        default=None, description="Total generation time"
    )
    # DECLARED as of v1.79.54. The streaming payload has always merged these
    # three in, and the model did not declare them -- so a client generated
    # from the schema had no field for them and dropped them off every
    # message_stop, silently. Declaring them is also what lets the
    # NON-streaming builder carry them, which closes the other half: they were
    # absent there by omission, not by design.
    kv_cache_bytes: Optional[int] = Field(
        default=None, description="KV cache size in bytes at the end of the run"
    )
    queue_wait_ms: Optional[float] = Field(
        default=None, description="Time spent waiting in the FIFO generation queue"
    )
    draft_acceptance: Optional[float] = Field(
        default=None, description="Speculative-decoding acceptance rate, when a drafter ran"
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
    # Map through STOP_REASON_FROM_FINISH_REASON (converters.py) rather than
    # assigning a provider value here.
    stop_reason: StopReason = "end_turn"
    usage: Usage
    performance: Optional[PerformanceInfo] = None
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Echoed from request metadata"
    )
