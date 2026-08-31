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

    ONE CONTRACT, as of v1.79.58:

        Every field, when present, is a real measurement of exactly the thing
        its name says. Absent means this mode or engine could not measure it.

    That replaced a per-field asymmetry table, and the table is the thing worth
    remembering. Four field names here did not denote a single measurement --
    ``total_duration_ms`` had two origins, ``generation_tps`` was a real
    measurement in one mode and a wall-clock fallback in the other,
    ``prompt_tps`` shipped an unmeasured 0.0 as if measured, ``queue_wait_ms``
    hid a measured zero as absence. Every one was documented rather than
    fixed, and documentation cannot help a client holding a number it cannot
    identify. Renaming and filtering at the single builder removes the
    question instead of answering it.

    ``perf_collector.build_performance`` is the ONE place this is spelled, for
    both modes and both routes on the Messages grammar. Do not restate its
    per-field reasoning here: a second copy in a docstring is how the passage
    that stood here came to describe a contract two releases dead, twice over
    (once by going stale, once by a repair that replaced it with a flat claim
    of symmetry that was equally false).

    One structural fact that belongs with the type: a single emit site is what
    makes DECLARED-BUT-NEVER-EMITTED checkable. v1.79.55 filtered emitted down
    to declared and was blind to the reverse -- which is exactly what
    ``peak_memory_gb`` was until .50 and the two rates were until .54. With one
    builder the two sets can be compared in both directions, and
    ``tests/unit/test_message_stop_payload.py`` does.
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
    # RETIRED total_duration_ms (v1.79.58). "Total" is an origin-relative
    # name and it had two origins: request arrival non-streaming, stream start
    # streaming -- so the same work reported tens of seconds in one mode and a
    # few in the other, and nothing on the wire said which you held. Replaced
    # by two names that each denote one span rather than aliased to one of
    # them: two spellings for one value is the defect class v1.79.48 cited
    # when it MOVED the load route instead of aliasing it.
    # (The OpenAI wire's own `config.GenerationTiming.total_duration_ms` is a
    # different model on a different wire and is untouched.)
    request_duration_ms: Optional[int] = Field(
        default=None,
        description=(
            "Wall time from request arrival to completion, INCLUDING FIFO "
            "queue wait and any model load. User-perceived latency."
        ),
    )
    generation_duration_ms: Optional[int] = Field(
        default=None,
        description=(
            "Wall time spent generating, EXCLUDING queue wait and model load. "
            "The denominator to use for throughput."
        ),
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
