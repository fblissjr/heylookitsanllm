# src/heylook_llm/messages_api.py
#
# /v1/messages endpoint -- Anthropic Messages-inspired API.
#
# Uses the existing provider infrastructure via converters:
#   MessageCreateRequest -> to_chat_request() -> ChatRequest -> provider
#   provider response -> from_openai_response_dict() -> MessageResponse
#
# Streaming uses StreamingEventTranslator to emit structured SSE events
# instead of the flat OpenAI chat.completion.chunk format.

import asyncio
import logging
import time
import uuid
from contextlib import closing
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from heylook_llm.providers.abort import AbortEvent
from heylook_llm.providers.base import GenerationFailed, InvalidGenerationRequest
from heylook_llm.busy_response import model_busy_response
from heylook_llm.providers.common.generation_gate import ModelBusyError
from heylook_llm.optimizations import fast_json as json
from heylook_llm.request_registry import resolve_request_id, track_request, tracked_stream
from heylook_llm.router import ModelNotFound
from heylook_llm.schema.converters import (
    from_openai_response_dict,
    to_chat_request,
    to_stop_reason,
)
from heylook_llm.schema.messages import MessageCreateRequest
from heylook_llm.schema.responses import MessageResponse
from heylook_llm.perf_collector import (
    ChunkTelemetry,
    RequestEvent,
    get_perf_collector,
    build_performance,
    headline_tps,
    usage_counts,
)
from heylook_llm.schema.content_blocks import ImageBlock
from heylook_llm.reasoning_parser import (
    merge_presplit_thinking,
    parse_reasoning,
    PassThroughParser,
    parser_factory_for,
)
from heylook_llm.thinking_parser import HybridThinkingParser

messages_router = APIRouter(
    prefix="/v1",
    tags=["Messages API"],
)


# ---------------------------------------------------------------------------
# StreamingEventTranslator
# ---------------------------------------------------------------------------

class StreamingEventTranslator:
    """State machine that translates provider token chunks into structured SSE events.

    Tracks the current content block (thinking vs text) and emits
    content_block_start / content_block_delta / content_block_stop events
    as needed.

    Event sequence:
      message_start -> [content_block_start -> content_block_delta* -> content_block_stop]*
                    -> message_delta -> message_stop
    """

    def __init__(self, message_id: str, model: str, thinking_parser=None):
        self.message_id = message_id
        self.model = model
        self.block_index = -1
        self.current_block_type: str | None = None
        # Format-aware parser injected per model (select_reasoning_parser);
        # the <think>-marker parser stays the default for direct constructions.
        self.thinking_parser = thinking_parser if thinking_parser is not None else HybridThinkingParser()

        # Counters
        self.thinking_tokens = 0
        self.content_tokens = 0
        self.prompt_tokens = 0  # the WHOLE prompt (ChunkTelemetry's)
        self.completion_tokens = 0
        self.cache = None  # the request's CacheReport, copied from telemetry
        self.stop_reason: str = "end_turn"

        # Timing
        self.start_time = time.time()
        self.thinking_start: float | None = None
        self.thinking_end: float | None = None
        self.content_start: float | None = None

    # -- SSE helpers --------------------------------------------------------

    @staticmethod
    def _sse(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # -- Public API ---------------------------------------------------------

    def message_start_event(self) -> str:
        """Emit the initial message_start SSE event."""
        msg_shell = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "usage": {"input_tokens": self.prompt_tokens, "output_tokens": 0},
        }
        return self._sse("message_start", {"type": "message_start", "message": msg_shell})

    def process_chunk(self, text: str, token_id: int | None = None) -> list[str]:
        """Process a generation chunk and return SSE event strings."""
        events = []
        deltas = self.thinking_parser.process_chunk(text, token_id=token_id)

        for delta_type, delta_text in deltas:
            events.extend(self._emit_delta(delta_type, delta_text))

        return events

    def process_presplit_thinking(self, text: str) -> list[str]:
        """Emit pre-split reasoning (GenerationChunk.thinking) straight to the
        thinking block, bypassing the text parser -- the engine already did
        the split, re-parsing another engine's output is how markers leak."""
        return self._emit_delta("thinking", text)

    def _emit_delta(self, delta_type: str, delta_text: str) -> list[str]:
        """Block bookkeeping shared by parser output and pre-split thinking."""
        if not delta_text:
            return []
        events = []
        block_type = "thinking" if delta_type == "thinking" else "text"

        # Start a new block if type changed
        if block_type != self.current_block_type:
            # Close previous block
            if self.current_block_type is not None:
                events.append(self._block_stop())
            # Open new block
            events.append(self._block_start(block_type))

        # Emit delta
        events.append(self._block_delta(block_type, delta_text))

        # Track timing + counts
        if block_type == "thinking":
            if self.thinking_start is None:
                self.thinking_start = time.time()
            self.thinking_tokens += 1
        else:
            if self.thinking_start is not None and self.thinking_end is None:
                self.thinking_end = time.time()
            if self.content_start is None:
                self.content_start = time.time()
            self.content_tokens += 1

        return events

    def flush(self) -> list[str]:
        """Flush remaining parser state and close any open block."""
        events = []
        for delta_type, text in self.thinking_parser.flush():
            if text:
                block_type = "thinking" if delta_type == "thinking" else "text"
                if block_type != self.current_block_type:
                    if self.current_block_type is not None:
                        events.append(self._block_stop())
                    events.append(self._block_start(block_type))
                events.append(self._block_delta(block_type, text))

        # Close last block
        if self.current_block_type is not None:
            events.append(self._block_stop())

        return events

    def message_delta_event(self) -> str:
        """Emit message_delta with stop reason and usage."""
        output_tokens = self.completion_tokens or (self.thinking_tokens + self.content_tokens)
        input_tokens, cache_read = usage_counts(self.prompt_tokens, self.cache)
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if cache_read is not None:
            usage["cache_read_input_tokens"] = cache_read
        if self.thinking_tokens:
            usage["thinking_tokens"] = self.thinking_tokens
            usage["content_tokens"] = self.content_tokens
        return self._sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason},
            "usage": usage,
        })

    def message_stop_event(
        self,
        telemetry: "ChunkTelemetry | None" = None,
        *,
        request_start_time: float | None = None,
    ) -> str:
        """Emit terminal ``message_stop`` with the performance object.

        Takes the TELEMETRY, not a pre-built ``timing`` dict. That is the
        change: a caller-supplied dict was where the drift entered, because
        each of the two builders on this wire spelled the same fields
        differently and no test could see it (any test supplies its own dict,
        so it asserts "given declared keys, the output is declared" and stays
        green through the bug). The spelling now lives once, in
        ``perf_collector.build_performance``.

        ``request_start_time`` is optional because not every caller has one --
        ``/v1/conversations/{id}/generate`` does, ``/v1/messages`` does, a bare
        translator constructed in a test does not. When it is absent the
        request span is simply not reported, which under this wire's contract
        means exactly what it says: not measurable here.
        """
        end_time = time.time()
        thinking_ms = content_ms = None
        if self.thinking_start is not None:
            t_end = self.thinking_end or end_time
            thinking_ms = int((t_end - self.thinking_start) * 1000)
        if self.content_start is not None:
            content_ms = int((end_time - self.content_start) * 1000)

        perf = build_performance(
            telemetry if telemetry is not None else ChunkTelemetry(),
            # The translator's own clock starts when the stream does, which is
            # AFTER get_provider -- so it is the generation span, never the
            # request span. Naming it that way is the fix: it used to be
            # reported as `total_duration_ms`, the same key the non-streaming
            # builder filled from request arrival.
            generation_duration_ms=int((end_time - self.start_time) * 1000),
            request_duration_ms=(int((end_time - request_start_time) * 1000)
                                 if request_start_time is not None else None),
            thinking_duration_ms=thinking_ms,
            content_duration_ms=content_ms,
        )
        return self._sse("message_stop", {"type": "message_stop", "performance": perf})

    # -- Private helpers ----------------------------------------------------

    def _block_start(self, block_type: str) -> str:
        self.block_index += 1
        self.current_block_type = block_type
        # Anthropic opens a block with its content field present and empty
        # ({"type":"text","text":""}), so a client that accumulates into
        # content_block has something to accumulate into.
        content_block = {"type": block_type}
        if block_type == "thinking":
            content_block["thinking"] = ""
            content_block["text"] = ""
        else:
            content_block["text"] = ""
        return self._sse("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": content_block,
        })

    def _block_stop(self) -> str:
        return self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self.block_index,
        })

    def _block_delta(self, block_type: str, text: str) -> str:
        if block_type == "thinking":
            # Anthropic's thinking_delta carries `thinking`, not `text`. Both
            # go out: `thinking` is the conformant field a Messages client
            # reads, `text` is what v3's streaming.js reads today. Dropping
            # `text` here would blank the reasoning pane on notebook and
            # explore, so it stays until those readers move.
            delta = {"type": "thinking_delta", "thinking": text, "text": text}
        else:
            delta = {"type": "text_delta", "text": text}
        return self._sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": delta,
        })


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def _refuse_unoffered_depth(router, chat_request) -> None:
    """A thinking depth the model's template does not offer is a 400 BEFORE
    any stream opens and before any load (plan W2). The provider checks
    again inside the cascade; this is the half that keeps it out of a 200.
    Detection reads files only, so it answers for unloaded models."""
    value = chat_request.reasoning_effort
    model_id = chat_request.model
    if value is None or not model_id:
        return
    model_config = router.app_config.get_model_config(model_id)
    if model_config is None:
        return  # routing reports the unknown model with its own 400
    from heylook_llm.providers.contract import thinking_controls
    from heylook_llm.thinking_controls import check_depth

    why = check_depth(value, thinking_controls(model_config))
    if why:
        raise HTTPException(status_code=400, detail=why)


@messages_router.post(
    "/messages",
    summary="Create a Message",
    description="""
Create a message using the Messages API format.

Accepts typed content blocks (text, image, and audio on the gguf arm --
MLX answers 400 for audio) and returns structured output blocks (text,
thinking). System prompt is a top-level parameter,
not embedded in the messages array.

Supports streaming via `stream: true`, which returns Server-Sent Events
with distinct event types (message_start, content_block_start,
content_block_delta, content_block_stop, message_delta, message_stop).
    """,
    response_model=MessageResponse,
    # No tags= here: the router already carries ["Messages API"], and a second
    # copy on the route emitted it twice in /openapi.json.
)
async def create_message(request: Request, msg_request: MessageCreateRequest):
    router = request.app.state.router_instance
    # Honour the client's X-Request-ID. This endpoint used to always generate
    # its own, which meant the id a client sends -- and which the docs tell it
    # to send -- named nothing the server could find. DELETE /v1/requests/{id}
    # cancels by exactly this value, so a rewritten id would be uncancellable.
    request_id = resolve_request_id(
        request.headers.get("x-request-id"), prefix="msg")
    request_start_time = time.time()

    # Convert to internal ChatRequest
    chat_request = to_chat_request(msg_request)
    _refuse_unoffered_depth(router, chat_request)

    provider_get_ms = 0.0
    provider = None
    # Per-request cooperative abort signal: shared with the streaming layer so a
    # disconnect cancels only THIS request, not a concurrent one.
    abort_event = AbortEvent()
    try:
        # Get provider and create generator (CPU-bound, run in thread)
        provider_get_start = time.time()
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)
        provider_get_ms = (time.time() - provider_get_start) * 1000
        # Backpressure: reject early (503) if the generation queue is full,
        # before committing to a streaming response.
        provider.check_capacity()
        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request, abort_event)

    except RuntimeError as e:
        if isinstance(e, ModelBusyError):
            # One speller for all three endpoints -- see busy_response.py.
            return model_busy_response(e, provider)
        raise HTTPException(status_code=500, detail=str(e))
    except ModelNotFound as e:
        # Model ROUTING failed: unknown/disabled id, or no model named --
        # the client's pick, so 400 not 500.
        # Deliberately NOT a bare `except ValueError`: get_provider re-raises
        # load failures too, and those keep their 500 and their traceback.
        logging.warning(f"[MESSAGES] Model not resolved: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"[MESSAGES] Provider error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Detect images in messages for perf tracking
    had_images = any(
        isinstance(block, ImageBlock)
        for msg in msg_request.messages
        if isinstance(msg.content, list)
        for block in msg.content
    )

    perf_ctx = {
        "request_start_time": request_start_time,
        "provider_get_ms": provider_get_ms,
        "had_images": had_images,
    }

    # Resolved ONCE, here, where the converted ChatRequest still exists: the
    # handlers below only carry the MessageCreateRequest (whose raw `thinking`
    # field is missing the whole sampler layer), so they are handed a parser
    # FACTORY rather than the facts to build one from.
    make_parser = parser_factory_for(provider, chat_request)

    if msg_request.stream:
        # Tracking lives INSIDE the async generator, not here: the response
        # body outlives this function, so a `with` block around the return
        # would unregister before a single token was produced.
        return StreamingResponse(
            tracked_stream(
                _stream_messages(generator, msg_request, request_id, http_request=request, provider=provider, perf_ctx=perf_ctx, abort_event=abort_event, make_parser=make_parser),
                request_id, abort_event),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )
    else:
        # The case this exists for. Nothing is written to the connection until
        # the generation finishes, so an abandoned client is undetectable and
        # the run continues; registering the abort signal under the client's
        # own id is what makes it nameable from another connection.
        with track_request(request_id, abort_event):
            result = await _non_stream_messages(
                generator, msg_request, request_id, request_start_time, perf_ctx=perf_ctx,
                provider=provider, make_parser=make_parser,
                abort_event=abort_event,
            )
        # Echo the id the server actually tracked. This is the one path DELETE
        # /v1/requests/{id} exists for, and it was the one giving the client no
        # way to learn its header had been rejected (a space, a slash, >128
        # chars -> a server-generated id). Without it, a later DELETE 404s with
        # nothing to explain why -- the "believing it stopped something"
        # failure the endpoint's own description says it prevents.
        # model_dump_json, not JSONResponse(model_dump()): the latter
        # re-serializes the whole tree (repo convention).
        return Response(content=result.model_dump_json(),
                        media_type="application/json",
                        headers={"X-Request-ID": request_id})


# ---------------------------------------------------------------------------
# Non-streaming handler
# ---------------------------------------------------------------------------

async def _non_stream_messages(
    generator,
    msg_request: MessageCreateRequest,
    request_id: str,
    request_start_time: float,
    perf_ctx: dict | None = None,
    provider=None,
    # Zero-arg parser factory from `parser_factory_for` (the route builds it
    # where the ChatRequest exists). The default is the neutral one -- no
    # provider facts means nothing to split on -- not a second way to select.
    make_parser=PassThroughParser,
    abort_event=None,
) -> MessageResponse:
    """Consume the provider generator and build a MessageResponse."""
    full_text = ""
    token_count = 0
    pre_thinking_parts: list = []  # chunk.thinking -- engine pre-split reasoning
    telemetry = ChunkTelemetry()  # per-chunk counters/rates tagged by the engine (mlx-lm or llama-server)

    # Thinking/content spans, the way the streaming translator keeps them:
    # the first thinking output opens the thinking span, the first content
    # output closes it and opens the content span. This mode parses the text
    # once at the end, so a SECOND parser instance (same selection, so the
    # same split) watches the chunks as they land purely for the clock. It
    # sees engine pre-split reasoning (`chunk.thinking`) as thinking too.
    # Before this the non-streaming `performance` carried both durations as
    # null while the stream filled them -- the schema promises the two modes
    # the same telemetry.
    # The same watcher counts thinking and content deltas the way the
    # streaming translator counts them (one per emitted segment), so
    # `usage.thinking_tokens` / `content_tokens` mean the same thing in both
    # modes. Non-streaming used to leave them null.
    timing_parser = make_parser()
    thinking_start = thinking_end = content_start = None
    seen = {"thinking": 0, "text": 0}

    def _saw(kind: str, at: float) -> None:
        nonlocal thinking_start, thinking_end, content_start
        seen["thinking" if kind == "thinking" else "text"] += 1
        if kind == "thinking":
            if thinking_start is None:
                thinking_start = at
        else:
            if thinking_start is not None and thinking_end is None:
                thinking_end = at
            if content_start is None:
                content_start = at

    def consume():
        nonlocal full_text, token_count
        # closing() runs the provider generator's finally now (releases the
        # generation gate) even if consumption raised -- not at GC.
        with closing(generator):
            for chunk in generator:
                at = time.time()
                chunk_thinking = getattr(chunk, 'thinking', None)
                if chunk_thinking:
                    pre_thinking_parts.append(chunk_thinking)
                    _saw("thinking", at)
                full_text += chunk.text
                token_count += 1
                telemetry.absorb(chunk)
                for kind, text in timing_parser.process_chunk(chunk.text):
                    if text:
                        _saw(kind, at)
            at = time.time()
            for kind, text in timing_parser.flush():
                if text:
                    _saw(kind, at)

    generation_start = time.time()
    try:
        await asyncio.to_thread(consume)
    except InvalidGenerationRequest as e:
        raise HTTPException(status_code=400, detail=str(e))
    except GenerationFailed as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Parse thinking with the model's format-aware parser (harmony channels,
    # gemma channels, or <think> markers -- one selection for every route)
    content_text, thinking = parse_reasoning(full_text, make_parser())

    thinking = merge_presplit_thinking(pre_thinking_parts, thinking)

    # Build an OpenAI-shaped dict so we can reuse from_openai_response_dict.
    # mlx-lm's own reason: "length" (budget exhausted) must not read as "stop"
    # -- from_openai_response_dict maps it to the Messages stop_reason.
    finish_reason = telemetry.finish_reason or "stop"
    # A CANCELLED run must not claim the model finished its turn. Without this
    # the generator simply ends early and `stop_reason` falls through to
    # `end_turn`, which positively asserts completion -- the precise defect
    # v1.79.40 fixed on the conversation-generate route, reappearing here the
    # moment this path became cancellable (v1.79.44). Same value for the same
    # reason: Anthropic has no cancellation stop reason, because cancellation
    # there is a dropped connection rather than an end state, so `max_tokens`
    # is the closest spec-defined "stopped early, and not by the model's own
    # choice". Set through `finish_reason` rather than by assigning
    # `stop_reason` directly, so it still goes through the ONE mapper
    # `TestStopReasonHasOneMapper` exists to keep as the only writer.
    # Guarded on the DEFAULT, exactly like the streaming half below. Overwriting
    # unconditionally would clobber a genuine `length` or stop-sequence when a
    # DELETE lands between the last token and the response build -- and the two
    # halves of one rule quietly disagreeing is worse than either policy.
    if (abort_event is not None and abort_event.is_set()
            and finish_reason == "stop"):
        finish_reason = "length"
    message: dict = {"role": "assistant", "content": content_text}
    if thinking is not None:
        message["thinking"] = thinking

    choice: dict = {"message": message, "index": 0, "finish_reason": finish_reason}
    input_tokens, cache_read = usage_counts(telemetry.prompt_tokens, telemetry.cache)
    openai_dict = {
        "model": msg_request.model or "unknown",
        "choices": [choice],
        "usage": {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "completion_tokens": telemetry.completion_tokens or token_count,
        },
    }
    if seen["thinking"]:
        openai_dict["usage"]["thinking_tokens"] = seen["thinking"]
        openai_dict["usage"]["content_tokens"] = seen["text"]

    # Performance: the SAME builder the streaming half uses, so the two
    # payloads cannot disagree by hand (v1.79.58). `generation_start` is
    # recorded just before the consume loop below -- it is what lets this mode
    # report a generation span at all; it never could before, which is why
    # `total_duration_ms` here meant something different from the same key on
    # the stream.
    elapsed = time.time() - request_start_time
    total_tokens = (telemetry.completion_tokens or token_count)
    if elapsed > 0 and total_tokens > 0:
        end_time = time.time()
        thinking_ms = content_ms = None
        if thinking_start is not None:
            thinking_ms = int(((thinking_end or end_time) - thinking_start) * 1000)
        if content_start is not None:
            content_ms = int((end_time - content_start) * 1000)
        openai_dict["performance"] = build_performance(
            telemetry,
            request_duration_ms=int(elapsed * 1000),
            generation_duration_ms=(int((end_time - generation_start) * 1000)
                                    if generation_start is not None else None),
            thinking_duration_ms=thinking_ms,
            content_duration_ms=content_ms,
        )

    response = from_openai_response_dict(
        openai_dict,
        metadata=msg_request.metadata,
    )

    logging.info(
        f"[MESSAGES] {request_id[:12]} completed | "
        f"tokens={total_tokens} | {elapsed:.2f}s"
    )

    # Record perf event
    if perf_ctx:
        now = time.time()
        tps = headline_tps(telemetry.generation_tps, total_tokens, elapsed, telemetry.queue_wait_ms)
        p_get_ms = perf_ctx["provider_get_ms"]
        had_imgs = perf_ctx.get("had_images", False)
        get_perf_collector().record_request(RequestEvent(
            timestamp=now,
            model=msg_request.model or "unknown",
            success=True,
            total_ms=elapsed * 1000,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            token_generation_ms=elapsed * 1000 - p_get_ms,
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=total_tokens,
            tokens_per_second=tps,
            had_images=had_imgs,
            was_streaming=False,
            queue_wait_ms=round(telemetry.queue_wait_ms, 1),
            prompt_tps=telemetry.prompt_tps,
            **RequestEvent.report_fields(telemetry),
        ))

    return response


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------

async def _stream_messages(
    generator,
    msg_request: MessageCreateRequest,
    request_id: str,
    http_request=None,
    provider=None,
    perf_ctx: dict | None = None,
    abort_event=None,
    # Zero-arg parser factory from `parser_factory_for` (the route builds it
    # where the ChatRequest exists). The default is the neutral one -- no
    # provider facts means nothing to split on -- not a second way to select.
    make_parser=PassThroughParser,
) -> AsyncGenerator[str, None]:
    """Async SSE generator using StreamingEventTranslator."""
    message_id = f"msg_{uuid.uuid4().hex[:16]}"
    model = msg_request.model or "unknown"
    translator = StreamingEventTranslator(
        message_id, model,
        thinking_parser=make_parser(),
    )

    # Resolve abort event from provider (if MLX provider with abort support)
    # abort_event is the per-request signal passed in by the route.

    from heylook_llm.streaming_utils import async_generator_with_abort, control_frame

    # message_start
    yield translator.message_start_event()

    telemetry = ChunkTelemetry()  # per-chunk counters/rates tagged by the engine (mlx-lm or llama-server)
    try:
        async for chunk in async_generator_with_abort(generator, http_request, abort_event, log_prefix=f"[MESSAGES {request_id[:12]}] "):
            # Marker guard FIRST: keepalive is the grammar's own `ping`,
            # progress the namespaced heylook_progress.
            frame = control_frame(chunk)
            if frame:
                yield frame
                continue
            # Capture provider metadata. The provider speaks OpenAI's
            # finish_reason vocabulary; renaming it at this boundary is what
            # keeps "length" off an Anthropic-shaped message_delta.
            chunk_finish = getattr(chunk, "finish_reason", None)
            if chunk_finish:
                translator.stop_reason = to_stop_reason(chunk_finish)
            telemetry.absorb(chunk)
            # The translator owns the token counts it reports in its own
            # message_delta/usage events.
            translator.prompt_tokens = telemetry.prompt_tokens
            translator.completion_tokens = telemetry.completion_tokens
            translator.cache = telemetry.cache

            # Pre-split reasoning (chunk.thinking) goes straight to the
            # thinking block; the parser only ever sees chunk.text.
            chunk_thinking = getattr(chunk, "thinking", None)
            if chunk_thinking:
                for event_str in translator.process_presplit_thinking(chunk_thinking):
                    yield event_str

            if not chunk.text:
                continue

            token_id = getattr(chunk, "token", None)

            for event_str in translator.process_chunk(chunk.text, token_id=token_id):
                yield event_str

    except InvalidGenerationRequest as e:
        # Provider request-validation guards fire at first next(), after
        # headers flushed -- type the in-band event as the CLIENT error it is
        # (Anthropic's invalid_request_error), not an api_error.
        yield translator._sse("error", {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": str(e)},
        })
        return

    except GenerationFailed as e:
        # Mid-stream failure: headers already sent -- Anthropic-style error
        # event, never content.
        yield translator._sse("error", {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })
        return

    # Same rule as the non-streaming path above and as
    # conversation_generate_api: a cancelled stream must not report the
    # model's own end. An aborted run stops between tokens, so the last chunk
    # carries no finish_reason and `stop_reason` would stay at its `end_turn`
    # default -- a consumer keying on the shared Messages grammar could not
    # tell a cancelled turn from a completed one. Only overridden when the
    # provider said nothing: a real `length`/`stop_sequence` from the engine
    # is a more specific truth and keeps priority.
    if (abort_event is not None and abort_event.is_set()
            and translator.stop_reason == "end_turn"):
        translator.stop_reason = "max_tokens"

    # Flush parser
    for event_str in translator.flush():
        yield event_str

    # message_delta + message_stop (timing/KV telemetry rides message_stop's
    # performance object -- the extension the v3 status lines read)
    yield translator.message_delta_event()
    yield translator.message_stop_event(
        telemetry, request_start_time=perf_ctx["request_start_time"] if perf_ctx else None
    )

    logging.info(f"[MESSAGES] {request_id[:12]} stream complete")

    # Record perf event
    if perf_ctx:
        now = time.time()
        total_ms = (now - perf_ctx["request_start_time"]) * 1000
        gen_tokens = translator.completion_tokens or (translator.thinking_tokens + translator.content_tokens)
        gen_time_s = now - translator.start_time
        tps = headline_tps(telemetry.generation_tps, gen_tokens, gen_time_s, telemetry.queue_wait_ms)
        p_get_ms = perf_ctx["provider_get_ms"]
        had_imgs = perf_ctx.get("had_images", False)

        get_perf_collector().record_request(RequestEvent(
            timestamp=now,
            model=model,
            success=True,
            total_ms=total_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            token_generation_ms=gen_time_s * 1000,
            prompt_tokens=translator.prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=tps,
            had_images=had_imgs,
            was_streaming=True,
            queue_wait_ms=round(telemetry.queue_wait_ms, 1),
            prompt_tps=telemetry.prompt_tps,
            **RequestEvent.report_fields(telemetry),
        ))
