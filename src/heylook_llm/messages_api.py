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
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from heylook_llm.optimizations import fast_json as json
from heylook_llm.schema.converters import from_openai_response_dict, to_chat_request
from heylook_llm.schema.messages import MessageCreateRequest
from heylook_llm.schema.responses import MessageResponse, PerformanceInfo, Usage
from heylook_llm.perf_collector import RequestEvent, get_perf_collector
from heylook_llm.schema.content_blocks import ImageBlock
from heylook_llm.thinking_parser import HybridThinkingParser, parse_thinking_content

messages_router = APIRouter(prefix="/v1", tags=["Messages API"])


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

    def __init__(self, message_id: str, model: str):
        self.message_id = message_id
        self.model = model
        self.block_index = -1
        self.current_block_type: str | None = None
        self.thinking_parser = HybridThinkingParser()

        # Counters
        self.thinking_tokens = 0
        self.content_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.stop_reason = "stop"

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
            if not delta_text:
                continue

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
        usage = {
            "input_tokens": self.prompt_tokens,
            "output_tokens": output_tokens,
        }
        if self.thinking_tokens:
            usage["thinking_tokens"] = self.thinking_tokens
            usage["content_tokens"] = self.content_tokens
        return self._sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason},
            "usage": usage,
        })

    def message_stop_event(self) -> str:
        """Emit terminal message_stop with performance data."""
        end_time = time.time()
        total_ms = int((end_time - self.start_time) * 1000)

        perf: dict = {"total_duration_ms": total_ms}
        if self.thinking_start is not None:
            t_end = self.thinking_end or end_time
            perf["thinking_duration_ms"] = int((t_end - self.thinking_start) * 1000)
        if self.content_start is not None:
            perf["content_duration_ms"] = int((end_time - self.content_start) * 1000)

        return self._sse("message_stop", {"type": "message_stop", "performance": perf})

    # -- Private helpers ----------------------------------------------------

    def _block_start(self, block_type: str) -> str:
        self.block_index += 1
        self.current_block_type = block_type
        return self._sse("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": block_type},
        })

    def _block_stop(self) -> str:
        return self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self.block_index,
        })

    def _block_delta(self, block_type: str, text: str) -> str:
        if block_type == "thinking":
            delta = {"type": "thinking_delta", "text": text}
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

@messages_router.post(
    "/messages",
    summary="Create a Message",
    description="""
Create a message using the Messages API format.

Accepts typed content blocks (text, image) and returns structured output
blocks (text, thinking, logprobs). System prompt is a top-level parameter,
not embedded in the messages array.

Supports streaming via `stream: true`, which returns Server-Sent Events
with distinct event types (message_start, content_block_start,
content_block_delta, content_block_stop, message_delta, message_stop).
    """,
    response_model=MessageResponse,
    tags=["Messages API"],
)
async def create_message(request: Request, msg_request: MessageCreateRequest):
    router = request.app.state.router_instance
    request_id = f"msg-{uuid.uuid4()}"
    request_start_time = time.time()

    # Convert to internal ChatRequest
    chat_request = to_chat_request(msg_request)

    provider_get_ms = 0.0
    try:
        # Get provider and create generator (CPU-bound, run in thread)
        provider_get_start = time.time()
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)
        provider_get_ms = (time.time() - provider_get_start) * 1000
        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request)

    except RuntimeError as e:
        if "MODEL_BUSY" in str(e):
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Server is processing another request. Retry shortly.",
                        "type": "server_error",
                        "code": "model_overloaded",
                    }
                },
                headers={"Retry-After": "1"},
            )
        raise HTTPException(status_code=500, detail=str(e))
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

    if msg_request.stream:
        return StreamingResponse(
            _stream_messages(generator, msg_request, request_id, http_request=request, provider=provider, perf_ctx=perf_ctx),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_messages(
            generator, msg_request, request_id, request_start_time, perf_ctx=perf_ctx
        )


# ---------------------------------------------------------------------------
# Non-streaming handler
# ---------------------------------------------------------------------------

async def _non_stream_messages(
    generator,
    msg_request: MessageCreateRequest,
    request_id: str,
    request_start_time: float,
    perf_ctx: dict | None = None,
) -> MessageResponse:
    """Consume the provider generator and build a MessageResponse."""
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    token_count = 0

    def consume():
        nonlocal full_text, prompt_tokens, completion_tokens, token_count
        for chunk in generator:
            full_text += chunk.text
            token_count += 1
            prompt_tokens = getattr(chunk, "prompt_tokens", prompt_tokens)
            completion_tokens = getattr(chunk, "generation_tokens", completion_tokens)

    await asyncio.to_thread(consume)

    # Parse thinking
    content_text, thinking = parse_thinking_content(full_text)

    # Build an OpenAI-shaped dict so we can reuse from_openai_response_dict
    finish_reason = "stop"
    message: dict = {"role": "assistant", "content": content_text}
    if thinking is not None:
        message["thinking"] = thinking

    openai_dict = {
        "model": msg_request.model or "unknown",
        "choices": [{"message": message, "index": 0, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens or token_count,
        },
    }

    # Performance metrics
    elapsed = time.time() - request_start_time
    total_tokens = (completion_tokens or token_count)
    if elapsed > 0 and total_tokens > 0:
        openai_dict["performance"] = {
            "prompt_tps": (prompt_tokens / elapsed) if prompt_tokens else 0,
            "generation_tps": total_tokens / elapsed,
            "total_duration_ms": int(elapsed * 1000),
        }

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
        tps = total_tokens / elapsed if elapsed > 0 and total_tokens > 0 else 0.0
        p_get_ms = perf_ctx["provider_get_ms"]
        had_imgs = perf_ctx.get("had_images", False)
        get_perf_collector().record_request(RequestEvent(
            timestamp=now,
            model=msg_request.model or "unknown",
            success=True,
            total_ms=elapsed * 1000,
            queue_ms=p_get_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            image_processing_ms=0.0,
            token_generation_ms=elapsed * 1000 - p_get_ms,
            first_token_ms=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=total_tokens,
            tokens_per_second=tps,
            had_images=had_imgs,
            was_streaming=False,
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
) -> AsyncGenerator[str, None]:
    """Async SSE generator using StreamingEventTranslator."""
    message_id = f"msg_{uuid.uuid4().hex[:16]}"
    model = msg_request.model or "unknown"
    translator = StreamingEventTranslator(message_id, model)

    # Resolve abort event from provider (if MLX provider with abort support)
    abort_event = getattr(provider, '_abort_event', None) if provider else None

    from heylook_llm.streaming_utils import async_generator_with_abort

    # message_start
    yield translator.message_start_event()

    async for chunk in async_generator_with_abort(generator, http_request, abort_event, log_prefix=f"[MESSAGES {request_id[:12]}] "):
        # Capture provider metadata
        chunk_finish = getattr(chunk, "finish_reason", None)
        if chunk_finish:
            translator.stop_reason = chunk_finish
        translator.prompt_tokens = getattr(chunk, "prompt_tokens", translator.prompt_tokens)
        translator.completion_tokens = getattr(chunk, "generation_tokens", translator.completion_tokens)

        if not chunk.text:
            continue

        token_id = getattr(chunk, "token", None)

        for event_str in translator.process_chunk(chunk.text, token_id=token_id):
            yield event_str

    # Flush parser
    for event_str in translator.flush():
        yield event_str

    # message_delta + message_stop
    yield translator.message_delta_event()
    yield translator.message_stop_event()

    logging.info(f"[MESSAGES] {request_id[:12]} stream complete")

    # Record perf event
    if perf_ctx:
        now = time.time()
        total_ms = (now - perf_ctx["request_start_time"]) * 1000
        gen_tokens = translator.completion_tokens or (translator.thinking_tokens + translator.content_tokens)
        gen_time_s = now - translator.start_time
        tps = gen_tokens / gen_time_s if gen_time_s > 0 and gen_tokens > 0 else 0.0
        p_get_ms = perf_ctx["provider_get_ms"]
        had_imgs = perf_ctx.get("had_images", False)

        # Real TTFT: use the translator's tracked timing of first output token
        first_output = translator.thinking_start or translator.content_start
        ttft_ms = (first_output - translator.start_time) * 1000 if first_output else 0.0

        get_perf_collector().record_request(RequestEvent(
            timestamp=now,
            model=model,
            success=True,
            total_ms=total_ms,
            queue_ms=p_get_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            image_processing_ms=0.0,
            token_generation_ms=gen_time_s * 1000,
            first_token_ms=ttft_ms,
            prompt_tokens=translator.prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=tps,
            had_images=had_imgs,
            was_streaming=True,
        ))
