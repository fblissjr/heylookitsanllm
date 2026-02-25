# src/heylook_llm/api.py
import asyncio
import logging
import time
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi

from heylook_llm.optimizations import fast_json as json
from heylook_llm.router import ModelRouter
from heylook_llm.config import (
    ChatRequest, ChatCompletionResponse,
    BatchChatRequest, BatchChatResponse, BatchStats, SystemMetricsResponse,
    CacheInfo, CacheListResponse, CacheClearRequest, CacheClearResponse
)
from heylook_llm.system_metrics import SystemMetricsCollector
from heylook_llm.perf_collector import RequestEvent, ResourceSnapshot, get_perf_collector
from heylook_llm.utils import log_request_start, log_request_stage, log_request_complete, log_full_request_details, log_request_summary, log_response_summary



async def _resource_snapshot_loop(app: FastAPI) -> None:
    """Background task: record a ResourceSnapshot every 60 seconds."""
    collector = get_perf_collector()
    while True:
        await asyncio.sleep(60)
        try:
            router: ModelRouter = app.state.router_instance
            metrics_collector = _get_metrics_collector(router)
            metrics = metrics_collector.collect(force_refresh=True)

            # Compute rolling TPS from events in the last 60s
            now = time.time()
            recent = [e for e in collector._events if e.timestamp >= now - 60]
            avg_tps = (sum(e.tokens_per_second for e in recent) / len(recent)) if recent else 0.0

            collector.record_resource_snapshot(ResourceSnapshot(
                timestamp=now,
                memory_gb=metrics.system.ram_used_gb,
                gpu_percent=0.0,
                tokens_per_second=avg_tps,
                requests=len(recent),
            ))
        except Exception:
            logging.debug("Resource snapshot collection failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The router is now initialized in server.py and passed in app.state
    task = asyncio.create_task(_resource_snapshot_loop(app))
    yield
    task.cancel()
    logging.info("Server shut down.")

app = FastAPI(
    title="HeylookLLM - High-Performance Local LLM Server",
    version="1.20.0",
    description="A high-performance API server for local LLM inference with OpenAI-compatible endpoints",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "OpenAI API",
            "description": "OpenAI-compatible endpoints for maximum compatibility with existing tools and libraries"
        },
        {
            "name": "Admin",
            "description": "Model management endpoints for CRUD, scanning, importing, and monitoring models"
        },
        {
            "name": "Monitoring",
            "description": "Performance monitoring and server status endpoints"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include STT router
from heylook_llm.stt_api import stt_router
app.include_router(stt_router)

# Import and include Messages API router
from heylook_llm.messages_api import messages_router
app.include_router(messages_router)

# Import and include Admin API routers (order matters: fixed paths before catch-all)
from heylook_llm.admin_api import scan_import_router, admin_router, admin_ops_router
app.include_router(scan_import_router)
app.include_router(admin_router)
app.include_router(admin_ops_router)

@app.get("/v1/models",
    summary="List Available Models",
    description="""
List all language models currently available on this server.

**Use this endpoint to:**
- Discover which models are loaded and ready for inference
- Verify a specific model is available before making requests
- Get model IDs for use in completion requests

**Returns:**
- Model IDs (e.g., "qwen2.5-coder-1.5b-instruct-4bit")
- OpenAI-compatible model objects
- Only shows models marked as `enabled: true` in models.toml
    """,
    response_description="List of available models in OpenAI-compatible format",
    tags=["OpenAI API"]
)
async def list_models(request: Request):
    """Get the list of available models in OpenAI format with capabilities."""
    router = request.app.state.router_instance
    models_data = []

    for model_id in router.list_available_models():
        model_entry = {
            "id": model_id,
            "object": "model",
            "owned_by": "user",
        }

        # Add capabilities and provider if available from config
        model_config = router.app_config.get_model_config(model_id)
        if model_config:
            model_entry["provider"] = model_config.provider

            # Use explicit capabilities if set, otherwise auto-detect
            if model_config.capabilities:
                model_entry["capabilities"] = model_config.capabilities
            else:
                # Auto-detect capabilities from model config
                capabilities = _infer_model_capabilities(model_config)
                if capabilities:
                    model_entry["capabilities"] = capabilities

        models_data.append(model_entry)

    return {"object": "list", "data": models_data}


def _infer_model_capabilities(model_config) -> list[str]:
    """Infer model capabilities from config when not explicitly set."""
    capabilities = []
    provider = model_config.provider
    config = model_config.config

    # STT models have transcription capability, not chat
    if provider == "mlx_stt":
        return ["transcription"]

    # Chat models (MLX)
    if provider == "mlx":
        capabilities.append("chat")

        # Check for vision capability
        if hasattr(config, "vision") and config.vision:
            capabilities.append("vision")

        # Check for thinking capability
        if hasattr(config, "enable_thinking") and config.enable_thinking:
            capabilities.append("thinking")
        elif hasattr(config, "supports_thinking") and config.supports_thinking:
            capabilities.append("thinking")

        # MLX models support hidden states extraction
        if provider == "mlx":
            capabilities.append("hidden_states")

    return capabilities


# Initialize metrics collector as None - will be created on first request
# Thread-safe lazy initialization with double-checked locking
import threading
_metrics_collector: SystemMetricsCollector | None = None
_metrics_collector_lock = threading.Lock()


def _get_metrics_collector(router: ModelRouter) -> SystemMetricsCollector:
    """Get or create the system metrics collector (thread-safe)."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_collector_lock:
            # Double-check after acquiring lock
            if _metrics_collector is None:
                _metrics_collector = SystemMetricsCollector(router, cache_ttl_seconds=30.0)
    return _metrics_collector


@app.get("/v1/system/metrics",
    summary="Get System Metrics",
    description="""
Get current system resource and model metrics for monitoring dashboards.

**Returns:**
- System metrics: RAM usage, CPU percentage
- Per-model metrics: Context usage, memory, active requests
- Cached for 30 seconds to minimize polling overhead

**Use Cases:**
- Build monitoring dashboards
- Track context window usage during conversations
- Monitor system resource consumption
- Alert on high memory/context usage

**Polling:**
- Recommended poll interval: 5-10 seconds
- Backend caches metrics for 30 seconds
    """,
    response_model=SystemMetricsResponse,
    response_description="Current system and model metrics",
    tags=["Monitoring"]
)
async def get_system_metrics(request: Request, force_refresh: bool = False):
    """
    Get current system and model metrics.

    Args:
        force_refresh: If true, bypass cache and collect fresh metrics
    """
    router = request.app.state.router_instance
    collector = _get_metrics_collector(router)
    return collector.collect(force_refresh=force_refresh)


@app.get("/v1/performance/profile/{time_range}",
    summary="Performance Profile",
    description="Aggregated performance profiling data for the Performance applet. "
                "Valid time_range values: 1h, 6h, 24h, 7d.",
    tags=["Monitoring"],
)
async def get_performance_profile(time_range: str):
    """Return aggregated performance profile from in-memory ring buffer."""
    valid_ranges = {"1h", "6h", "24h", "7d"}
    if time_range not in valid_ranges:
        raise HTTPException(status_code=400, detail=f"Invalid time_range. Must be one of: {', '.join(sorted(valid_ranges))}")
    return get_perf_collector().build_profile(time_range)


def _apply_image_resize(chat_request: ChatRequest) -> None:
    """Apply resize parameters to images in chat request messages, in-place."""
    if not any([chat_request.resize_max, chat_request.resize_width, chat_request.resize_height]):
        return
    from heylook_llm.utils_resize import process_image_url_with_resize
    for msg in chat_request.messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if hasattr(part, 'type') and part.type == 'image_url' and hasattr(part, 'image_url'):
                    resized = process_image_url_with_resize(
                        part.image_url.url,
                        resize_max=chat_request.resize_max,
                        resize_width=chat_request.resize_width,
                        resize_height=chat_request.resize_height,
                        image_quality=chat_request.image_quality or 85,
                        preserve_alpha=chat_request.preserve_alpha or False,
                    )
                    if resized != part.image_url.url:
                        part.image_url.url = resized


@app.post("/v1/chat/completions",
    summary="Create Chat Completion",
    description="""
Generate text completions from chat messages using the specified model.

**Key Features:**
- Automatic model loading with LRU cache (max 2 models)
- Vision model support with base64 images
- Streaming responses (Server-Sent Events)
- Batch processing for multiple prompts
- Reproducible generation with seed parameter
- Metal-optimized inference for MLX models

**Special Parameters:**
- `processing_mode`: Control batch behavior ("sequential", "parallel", "conversation")
- `return_individual`: Get separate responses for batch requests
- `include_timing`: Add performance metrics to response
- `stream`: Enable token-by-token streaming

**Performance Notes:**
- First request to a model includes loading time (~2-30s depending on size)
- Subsequent requests use cached model for instant inference
- Vision models process images in parallel for efficiency
    """,
    response_model=ChatCompletionResponse,
    response_description="Chat completion with generated text and token usage",
    responses={
        200: {
            "description": "Non-streaming: JSON response. Streaming (stream=true): Server-Sent Events where each `data:` line contains a StreamChunk JSON object, ending with `data: [DONE]`.",
            "content": {
                "text/event-stream": {
                    "schema": {"$ref": "#/components/schemas/StreamChunk"},
                },
            },
        },
    },
    tags=["OpenAI API"]
)
async def create_chat_completion(request: Request, chat_request: ChatRequest):
    router = request.app.state.router_instance
    request_id = f"req-{uuid.uuid4()}"

    request_start_time = time.time()

    try:
        # Add start time for processing time calculation
        chat_request._start_time = request_start_time
    except Exception as e:
        logging.warning(f"Request validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    # Check if this is a batch processing request
    if chat_request.processing_mode and chat_request.processing_mode != "conversation":
        # Use batch processor for non-conversation modes
        from heylook_llm.batch_processor import BatchProcessor, ProcessingMode

        logging.info(f"[API] Processing batch request with mode: {chat_request.processing_mode}")
        logging.info(f"[API] Number of messages in request: {len(chat_request.messages)}")

        _apply_image_resize(chat_request)

        for idx, msg in enumerate(chat_request.messages):
            if isinstance(msg.content, list):
                parts_info = []
                for part in msg.content:
                    if hasattr(part, 'type'):
                        if part.type == 'text':
                            parts_info.append(f"text: '{part.text[:30]}...'")
                        elif part.type == 'image_url':
                            parts_info.append("image")
                logging.info(f"[API] Message {idx}: {msg.role} - [{', '.join(parts_info)}]")
            else:
                logging.info(f"[API] Message {idx}: {msg.role} - {str(msg.content)[:50]}...")

        batch_processor = BatchProcessor(router)

        # Convert to batch request format
        from heylook_llm.batch_processor import BatchChatRequest
        batch_request = BatchChatRequest(
            model=chat_request.model,
            messages=chat_request.messages,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            top_k=chat_request.top_k,
            min_p=chat_request.min_p,
            repetition_penalty=chat_request.repetition_penalty,
            repetition_context_size=chat_request.repetition_context_size,
            max_tokens=chat_request.max_tokens,
            stream=False,  # Batch doesn't support streaming
            seed=chat_request.seed,
            processing_mode=ProcessingMode(chat_request.processing_mode),
            return_individual=chat_request.return_individual if chat_request.return_individual is not None else True,
            include_timing=chat_request.include_timing if chat_request.include_timing is not None else False
        )

        # Process batch and return
        batch_response = await batch_processor.process_batch_request(batch_request)
        return batch_response.model_dump()

    # Standard processing for conversation mode or no processing_mode specified
    image_resize_start = time.time()
    _apply_image_resize(chat_request)
    image_resize_ms = (time.time() - image_resize_start) * 1000

    # Start real-time logging
    log_request_start(request_id, chat_request.model)

    # Analyze request for image metadata
    request_dict = chat_request.model_dump()
    from heylook_llm.utils import _analyze_images_in_request
    image_stats = _analyze_images_in_request(request_dict)

    # Log enhanced request summary
    log_request_summary(
        request_id,
        chat_request.model,
        has_images=image_stats['count'] > 0,
        image_count=image_stats['count'],
        total_image_size=image_stats['total_size']
    )

    # Log detailed image processing info if images are present
    if image_stats['count'] > 0:
        logging.info(f"[IMAGE PROCESSING] Request {request_id[:8]} contains {image_stats['count']} base64 images | "
                   f"Total size: {image_stats['total_size']} | Avg: {image_stats['avg_size']} | "
                   f"Processing via: {'BATCH' if chat_request.processing_mode else 'STANDARD'} API")
        if image_stats['sizes']:
            logging.info(f"[IMAGE PROCESSING] Individual sizes: {', '.join(image_stats['sizes'])}")

    # Log full request details if DEBUG level
    if router.log_level <= logging.DEBUG:
        log_full_request_details(request_id, chat_request)

    provider_get_ms = 0.0
    try:
        log_request_stage(request_id, "routing")

        # Run CPU-bound operations in thread pool (timed for perf collection)
        provider_get_start = time.time()
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)
        provider_get_ms = (time.time() - provider_get_start) * 1000

        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        log_request_stage(request_id, "generating")
        # Run model generation in thread pool
        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request)

    except RuntimeError as e:
        # Check if this is a MODEL_BUSY error
        if "MODEL_BUSY" in str(e):
            logging.warning(f"Model busy for request {request_id[:8]}: {e}")
            log_request_complete(request_id, success=False, error_msg="Model busy")
            _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0)

            # Return 503 Service Unavailable with retry headers
            # This tells OpenAI client to retry automatically
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "The server is currently processing another request. Please retry in a moment.",
                        "type": "server_error",
                        "code": "model_overloaded"
                    }
                },
                headers={
                    "Retry-After": "1",  # Suggest retry after 1 second
                    "X-RateLimit-Limit": "1",  # We can handle 1 concurrent request
                    "X-RateLimit-Remaining": "0",  # No capacity right now
                    "X-RateLimit-Reset": str(int(time.time() + 1))  # Reset in 1 second
                }
            )
        else:
            # Other runtime errors
            logging.error(f"Runtime error: {e}", exc_info=True)
            log_request_complete(request_id, success=False, error_msg=str(e))
            _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0)
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logging.error(f"Failed to get provider or create generator: {e}", exc_info=True)
        log_request_complete(request_id, success=False, error_msg=str(e))
        _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0)
        raise HTTPException(status_code=500, detail=str(e))

    # Build perf context for handlers
    perf_ctx = {
        "request_start_time": request_start_time,
        "provider_get_ms": provider_get_ms,
        "image_resize_ms": image_resize_ms,
        "had_images": image_stats['count'] > 0,
    }

    if chat_request.stream:
        return StreamingResponse(
            stream_response_generator_async(generator, chat_request, router, request_id, http_request=request, provider=provider, perf_ctx=perf_ctx),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_response(generator, chat_request, router, request_id, request_start_time, perf_ctx=perf_ctx)

def _record_error_event(model: str, request_start_time: float, provider_get_ms: float, image_resize_ms: float, had_images: bool) -> None:
    """Record a failed request event to perf collector."""
    now = time.time()
    total_ms = (now - request_start_time) * 1000
    get_perf_collector().record_request(RequestEvent(
        timestamp=now,
        model=model,
        success=False,
        total_ms=total_ms,
        queue_ms=provider_get_ms,
        model_load_ms=provider_get_ms if provider_get_ms >= 100 else 0.0,
        image_processing_ms=image_resize_ms if had_images else 0.0,
        token_generation_ms=0.0,
        first_token_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        tokens_per_second=0.0,
        had_images=had_images,
        was_streaming=False,
    ))


async def stream_response_generator_async(generator, chat_request: ChatRequest, router, request_id, http_request: Request = None, provider=None, perf_ctx: dict | None = None):
    """Async streaming response generator that runs generation in thread pool.

    Supports Qwen3 thinking tokens: streams delta.thinking during <think> blocks,
    then delta.content for the response after </think>.

    Supports logprobs: includes token log probabilities in each chunk when requested.

    Uses HybridThinkingParser for token-level detection when token IDs are available.

    Enhanced metadata (when stream_options.include_usage=true):
    - thinking_tokens/content_tokens: Separate token counts
    - timing: thinking_duration_ms, content_duration_ms, total_duration_ms
    - generation_config: Sampler settings used
    - stop_reason: Why generation stopped
    """
    from heylook_llm.thinking_parser import HybridThinkingParser

    model_id = chat_request.model
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0

    # Enhanced timing tracking
    generation_start_time = time.time()
    first_output_time = None  # Wall clock of first yielded token (TTFT)
    thinking_start_time = None
    thinking_end_time = None
    content_start_time = None
    thinking_tokens = 0
    content_tokens = 0
    stop_reason = "stop"  # Default; updated from MLX finish_reason if available

    # Check if usage stats should be included in final chunk
    include_usage = (
        chat_request.stream_options
        and chat_request.stream_options.get('include_usage', False)
    )

    # Hybrid thinking parser: uses token IDs when available, falls back to text parsing
    thinking_parser = HybridThinkingParser()

    # Initialize logprobs collector if requested
    logprobs_collector = None
    if chat_request.logprobs:
        try:
            from heylook_llm.logprobs import StreamingLogprobsCollector
            # Get tokenizer from provider for decoding tokens
            provider = router.get_provider(chat_request.model)
            tokenizer = None
            if hasattr(provider, 'processor') and provider.processor:
                tokenizer = getattr(provider.processor, 'tokenizer', provider.processor)
            if tokenizer:
                top_logprobs = chat_request.top_logprobs or 5
                logprobs_collector = StreamingLogprobsCollector(tokenizer, top_logprobs=top_logprobs)
        except Exception as e:
            logging.warning(f"Failed to initialize streaming logprobs collector: {e}")

    log_request_stage(request_id, "streaming")

    def make_delta(delta_type: str, text: str, logprobs_delta=None) -> str:
        """Create SSE delta message for thinking or content with optional logprobs."""
        choice = {"delta": {delta_type: text}, "index": 0}
        if logprobs_delta:
            choice["logprobs"] = logprobs_delta
        response = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [choice]
        }
        return f"data: {json.dumps(response)}\n\n"

    # Resolve abort event from provider (if MLX provider with abort support)
    abort_event = getattr(provider, '_abort_event', None) if provider else None

    from heylook_llm.streaming_utils import async_generator_with_abort

    async for chunk in async_generator_with_abort(generator, http_request, abort_event, log_prefix=f"[API {request_id[:8]}] "):
        # Track finish_reason from MLX even for empty chunks (values: "length", "stop", or None)
        # The final chunk may have empty text but still carry the finish_reason
        chunk_finish_reason = getattr(chunk, 'finish_reason', None)
        if chunk_finish_reason:
            # Map MLX finish reasons to OpenAI-compatible values
            # OpenAI uses: "stop" (natural end), "length" (hit max_tokens), "content_filter"
            # MLX uses: "stop" (EOS token), "length" (hit max_tokens)
            if chunk_finish_reason == "length":
                stop_reason = "length"  # OpenAI standard for max_tokens
            elif chunk_finish_reason == "stop":
                stop_reason = "stop"  # OpenAI standard for natural completion
            else:
                stop_reason = chunk_finish_reason  # Pass through any other values

        # Also capture token counts from chunks (including final empty chunk)
        prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
        completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)

        if not chunk.text:
            continue

        token_count += 1

        # Get token ID for token-level parsing and logprobs
        token_id = getattr(chunk, 'token', None)

        # Collect logprobs if requested and available
        logprobs_delta = None
        if logprobs_collector:
            chunk_logprobs = getattr(chunk, 'logprobs', None)
            if token_id is not None and chunk_logprobs is not None:
                logprobs_delta = logprobs_collector.add_token_and_get_delta(token_id, chunk_logprobs)

        # Update token count periodically
        if token_count % 10 == 0:  # Update every 10 tokens for streaming
            from heylook_llm.utils import log_token_update
            log_token_update(request_id, token_count)

        # Process through thinking parser (uses token ID for Qwen3 thinking blocks)
        deltas = thinking_parser.process_chunk(chunk.text, token_id=token_id)
        for delta_type, text in deltas:
            if text:
                # Track timing and token counts by type
                if delta_type == "thinking":
                    if thinking_start_time is None:
                        thinking_start_time = time.time()
                    thinking_tokens += 1
                else:  # content
                    if thinking_start_time is not None and thinking_end_time is None:
                        thinking_end_time = time.time()
                    if content_start_time is None:
                        content_start_time = time.time()
                    content_tokens += 1

                if first_output_time is None:
                    first_output_time = time.time()
                yield make_delta(delta_type, text, logprobs_delta)
                logprobs_delta = None  # Only include logprobs in first delta for this token

    # Flush any remaining buffer
    for delta_type, text in thinking_parser.flush():
        if text:
            # Track final tokens during flush
            if delta_type == "thinking":
                thinking_tokens += 1
            else:
                content_tokens += 1
            yield make_delta(delta_type, text)

    # Calculate final timing
    generation_end_time = time.time()
    total_duration_ms = int((generation_end_time - generation_start_time) * 1000)

    thinking_duration_ms = None
    if thinking_start_time and thinking_end_time:
        thinking_duration_ms = int((thinking_end_time - thinking_start_time) * 1000)
    elif thinking_start_time and thinking_tokens > 0:
        # Thinking never ended (no content), calculate from now
        thinking_duration_ms = int((generation_end_time - thinking_start_time) * 1000)

    content_duration_ms = None
    if content_start_time:
        content_duration_ms = int((generation_end_time - content_start_time) * 1000)

    # Emit usage stats in final chunk if requested (OpenAI stream_options.include_usage)
    if include_usage:
        # Use tracked counts, fallback to token_count for completion_tokens
        final_prompt_tokens = prompt_tokens or 0
        final_completion_tokens = completion_tokens or token_count

        # Build enhanced usage object
        usage_data = {
            "prompt_tokens": final_prompt_tokens,
            "completion_tokens": final_completion_tokens,
            "total_tokens": final_prompt_tokens + final_completion_tokens
        }

        # Add thinking-specific fields if there were thinking tokens
        if thinking_tokens > 0:
            usage_data["thinking_tokens"] = thinking_tokens
            usage_data["content_tokens"] = content_tokens

        # Build timing object
        timing_data = {
            "total_duration_ms": total_duration_ms
        }
        if thinking_duration_ms is not None:
            timing_data["thinking_duration_ms"] = thinking_duration_ms
        if content_duration_ms is not None:
            timing_data["content_duration_ms"] = content_duration_ms

        # Build generation config from request (only include set values)
        generation_config = {
            k: v for k, v in {
                "temperature": chat_request.temperature,
                "top_p": chat_request.top_p,
                "top_k": chat_request.top_k,
                "min_p": chat_request.min_p,
                "max_tokens": chat_request.max_tokens,
                "enable_thinking": chat_request.enable_thinking,
            }.items() if v is not None
        }

        usage_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [{"delta": {}, "index": 0, "finish_reason": stop_reason}],
            "usage": usage_data,
            "timing": timing_data,
            "stop_reason": stop_reason
        }

        # Only include generation_config if non-empty
        if generation_config:
            usage_chunk["generation_config"] = generation_config

        yield f"data: {json.dumps(usage_chunk)}\n\n"

    # Log completion
    log_request_complete(request_id, success=True)

    # Record perf event
    if perf_ctx:
        now = time.time()
        req_total_ms = (now - perf_ctx["request_start_time"]) * 1000
        gen_tokens = completion_tokens or token_count
        gen_time_s = (now - generation_start_time)
        tps = gen_tokens / gen_time_s if gen_time_s > 0 and gen_tokens > 0 else 0.0
        p_get_ms = perf_ctx["provider_get_ms"]

        # Real TTFT: wall clock from generation start to first yielded token
        ttft_ms = (first_output_time - generation_start_time) * 1000 if first_output_time else 0.0

        get_perf_collector().record_request(RequestEvent(
            timestamp=now,
            model=model_id or "unknown",
            success=True,
            total_ms=req_total_ms,
            queue_ms=p_get_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            image_processing_ms=perf_ctx["image_resize_ms"] if perf_ctx["had_images"] else 0.0,
            token_generation_ms=total_duration_ms,
            first_token_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=tps,
            had_images=perf_ctx["had_images"],
            was_streaming=True,
        ))

    yield "data: [DONE]\n\n"

async def non_stream_response(generator, chat_request: ChatRequest, router, request_id, request_start_time, perf_ctx: dict | None = None):
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    token_count = 0
    logprobs_collector = None

    log_request_stage(request_id, "processing_response")

    # Initialize logprobs collector if requested
    if chat_request.logprobs:
        try:
            from heylook_llm.logprobs import LogprobsCollector
            # Get tokenizer from provider for decoding tokens
            provider = router.get_provider(chat_request.model)
            tokenizer = None
            if hasattr(provider, 'processor') and provider.processor:
                tokenizer = getattr(provider.processor, 'tokenizer', provider.processor)
            if tokenizer:
                top_logprobs = chat_request.top_logprobs or 5
                logprobs_collector = LogprobsCollector(tokenizer, top_logprobs=top_logprobs)
        except Exception as e:
            logging.warning(f"Failed to initialize logprobs collector: {e}")

    # Process generation in thread pool to avoid blocking event loop
    def consume_generator():
        nonlocal full_text, prompt_tokens, completion_tokens, token_count
        for chunk in generator:
            full_text += chunk.text
            token_count += 1

            # Collect logprobs if requested and available
            if logprobs_collector:
                token_id = getattr(chunk, 'token', None)
                chunk_logprobs = getattr(chunk, 'logprobs', None)
                if token_id is not None and chunk_logprobs is not None:
                    logprobs_collector.add_token(token_id, chunk_logprobs)

            # Update token count periodically for long responses
            if token_count % 25 == 0:
                from heylook_llm.utils import log_token_update
                log_token_update(request_id, token_count)

            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)

    await asyncio.to_thread(consume_generator)

    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens or token_count,  # Fallback to our count
        "total_tokens": (prompt_tokens or 0) + (completion_tokens or token_count)
    }

    # Parse thinking content from Qwen3 models
    from heylook_llm.thinking_parser import parse_thinking_content
    content, thinking = parse_thinking_content(full_text)

    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking

    # Build choice with optional logprobs
    choice = {"message": message, "index": 0, "finish_reason": "stop"}
    if logprobs_collector and logprobs_collector.content:
        choice["logprobs"] = logprobs_collector.to_dict()

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=chat_request.model or "unknown",
        choices=[choice],
        usage=usage_dict
    )

    # Calculate processing time
    processing_time = time.time() - request_start_time

    # Log response summary
    log_response_summary(
        request_id,
        len(full_text),
        token_count=completion_tokens or token_count,
        processing_time=processing_time
    )

    # Log full response details if DEBUG level
    if router.log_level <= logging.DEBUG:
        log_full_request_details(request_id, chat_request, full_text)
        logging.debug(f"Full non-stream response: {response.model_dump_json(indent=2)}")

    # Log successful completion
    log_request_complete(request_id, success=True)

    # Record perf event
    if perf_ctx:
        now = time.time()
        gen_tokens = completion_tokens or token_count
        tps = gen_tokens / processing_time if processing_time > 0 and gen_tokens > 0 else 0.0
        p_get_ms = perf_ctx["provider_get_ms"]
        get_perf_collector().record_request(RequestEvent(
            timestamp=now,
            model=chat_request.model or "unknown",
            success=True,
            total_ms=processing_time * 1000,
            queue_ms=p_get_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            image_processing_ms=perf_ctx["image_resize_ms"] if perf_ctx["had_images"] else 0.0,
            token_generation_ms=processing_time * 1000 - p_get_ms,
            first_token_ms=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=tps,
            had_images=perf_ctx["had_images"],
            was_streaming=False,
        ))

    return response


@app.post("/v1/batch/chat/completions",
    summary="Batch Chat Completions",
    description="""
Process multiple chat completion requests in a single batch for improved throughput.

**Performance Benefits:**
- 2-4x throughput improvement vs sequential processing
- Efficient handling of variable-length prompts via left-padding
- Optimized Metal memory management

**Requirements:**
- All requests must use the same text-only model
- Streaming is not supported (batch processing is inherently blocking)
- Minimum 2 requests per batch (recommended 3+ for best performance)

**Batch Parameters:**
- `completion_batch_size`: Max concurrent generations (default: 32)
- `prefill_batch_size`: Max prefill parallelism (default: 8)
- `prefill_step_size`: Chunk size for memory efficiency (default: 2048)

**Performance Notes:**
- Best performance with similar-length prompts (reduces padding waste)
- Larger batch sizes provide better throughput but higher latency
- Monitor batch_stats in response for throughput metrics
    """,
    response_model=BatchChatResponse,
    response_description="Batch completion results with statistics",
    tags=["OpenAI API"]
)
async def create_batch_chat_completion(request: Request, batch_request: BatchChatRequest):
    router = request.app.state.router_instance
    request_id = f"batch-req-{uuid.uuid4()}"

    start_time = time.time()

    logging.info(f"[BATCH API] Processing batch of {len(batch_request.requests)} requests")

    try:
        # Validate all requests use same model
        models = {req.model for req in batch_request.requests}
        if len(models) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"All requests must use the same model. Found: {list(models)}"
            )

        model_id = models.pop()

        # Check for streaming requests
        if any(req.stream for req in batch_request.requests):
            raise HTTPException(
                status_code=400,
                detail="Batch processing does not support streaming requests"
            )

        # Get provider (loads model if needed)
        provider = await asyncio.to_thread(router.get_provider, model_id)

        # Check if provider supports batching
        if not hasattr(provider, 'create_batch_chat_completion'):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' does not support batch processing"
            )

        logging.info(f"[BATCH API] Using provider: {provider.__class__.__name__}")

        # Process batch
        prefill_start = time.time()
        completions = await asyncio.to_thread(
            provider.create_batch_chat_completion,
            batch_request.requests
        )
        prefill_time = time.time() - prefill_start

        elapsed = time.time() - start_time

        # Build response objects
        responses = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, (req, completion) in enumerate(zip(batch_request.requests, completions)):
            response = ChatCompletionResponse(
                id=f"chatcmpl-batch-{uuid.uuid4()}",
                object="chat.completion",
                created=int(time.time()),
                model=model_id,
                choices=[{
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": completion['text']
                    },
                    "finish_reason": completion.get('finish_reason', 'stop')
                }],
                usage={
                    "prompt_tokens": completion.get('prompt_tokens', 0),
                    "completion_tokens": completion.get('completion_tokens', 0),
                    "total_tokens": completion.get('total_tokens', 0)
                }
            )
            responses.append(response)

            total_prompt_tokens += completion.get('prompt_tokens', 0)
            total_completion_tokens += completion.get('completion_tokens', 0)

        # Calculate statistics
        total_tokens = total_prompt_tokens + total_completion_tokens
        batch_stats = BatchStats(
            total_requests=len(batch_request.requests),
            elapsed_seconds=elapsed,
            throughput_req_per_sec=len(batch_request.requests) / elapsed,
            throughput_tok_per_sec=total_tokens / elapsed if total_tokens > 0 else 0,
            prefill_time=prefill_time,
            generation_time=elapsed - prefill_time,
            memory_peak_mb=0  # Placeholder - provider should provide this
        )

        logging.info(
            f"[BATCH API] Completed batch: {batch_stats.total_requests} requests in {batch_stats.elapsed_seconds:.2f}s "
            f"({batch_stats.throughput_req_per_sec:.1f} req/s, {batch_stats.throughput_tok_per_sec:.1f} tok/s)"
        )

        return BatchChatResponse(
            data=responses,
            batch_stats=batch_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[BATCH API] Error processing batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


@app.post("/v1/embeddings",
    summary="Create Embeddings",
    description="""
Generate embeddings for text using the specified model.

**Key Features:**
- Extract actual model embeddings (not hallucinated numbers)
- Support for both text-only and vision models
- Multiple pooling strategies (mean, cls, last, max)
- Optional dimension truncation
- Batch processing support

**Use Cases:**
- Text similarity search
- Semantic clustering
- Cross-modal alignment
- Prompt interpolation
- Document retrieval

**Request Body:**
- `input` (string | array[string]): Text(s) to embed
- `model` (string): Model ID to use
- `dimensions` (integer, optional): Truncate to N dimensions
- `encoding_format` (string, optional): "float" or "base64"
- `user` (string, optional): User identifier
    """,
    response_description="Embeddings in OpenAI-compatible format",
    tags=["OpenAI API"],
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "examples": {
                        "single": {
                            "summary": "Single embedding",
                            "value": {
                                "object": "list",
                                "data": [{
                                    "object": "embedding",
                                    "embedding": [0.0234, -0.1567, 0.8901],
                                    "index": 0
                                }],
                                "model": "dolphin-mistral",
                                "usage": {"prompt_tokens": 10, "total_tokens": 10}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_embeddings_endpoint(
    request: Request,
    embedding_request: dict = Body(...)
):
    """Create embeddings for the given input text(s)."""
    from heylook_llm.embeddings import EmbeddingRequest, create_embeddings

    try:
        # Parse request
        req = EmbeddingRequest(**embedding_request)

        # Get router
        router = request.app.state.router_instance

        # Create embeddings
        response = await create_embeddings(req, router)

        return response.model_dump()

    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/hidden_states",
    summary="Extract Hidden States",
    description="""
Extract raw hidden states from a specific layer of an LLM model.

**Key Differences from /v1/embeddings:**
- Returns full sequence [seq_len, hidden_dim], not pooled
- Extracts from specific layer (default: -2, second-to-last)
- Filters out padding tokens via attention mask
- Designed for use as text encoder backend for image generation

**Use Cases:**
- Text encoder for DiT-based image generation (Z-Image, etc.)
- Model interpretability and analysis
- Cross-modal alignment with per-token embeddings

**Request Body:**
- `input` (string | array[string]): Text(s) to encode (with chat template applied)
- `model` (string): Model ID to use
- `layer` (integer, optional): Layer to extract from (default: -2)
- `max_length` (integer, optional): Max sequence length (default: 512)
- `return_attention_mask` (boolean, optional): Include attention mask
- `encoding_format` (string, optional): "float" (default) or "base64"

**Note:** Only supported for MLX models.
    """,
    response_description="Hidden states with shape metadata",
    tags=["OpenAI API"],
    responses={
        200: {
            "description": "Hidden states extracted successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "float_format": {
                            "summary": "Float format response",
                            "value": {
                                "hidden_states": [[0.123, -0.456], [0.789, 0.012]],
                                "shape": [2, 2560],
                                "model": "Qwen3-4B-mxfp4-mlx",
                                "layer": -2,
                                "dtype": "bfloat16"
                            }
                        },
                        "base64_format": {
                            "summary": "Base64 format response",
                            "value": {
                                "hidden_states": "SGVsbG8gV29ybGQ=",
                                "shape": [21, 2560],
                                "model": "Qwen3-4B-mxfp4-mlx",
                                "layer": -2,
                                "dtype": "bfloat16",
                                "encoding_format": "base64"
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Model doesn't support hidden state extraction",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Hidden state extraction is not supported for this model."
                    }
                }
            }
        }
    }
)
async def extract_hidden_states_endpoint(
    request: Request,
    hidden_states_request: dict = Body(...)
):
    """Extract hidden states from the specified layer of an LLM."""
    from heylook_llm.hidden_states import HiddenStatesRequest, create_hidden_states

    try:
        # Parse request
        req = HiddenStatesRequest(**hidden_states_request)

        # Get router
        router = request.app.state.router_instance

        # Extract hidden states
        response = await create_hidden_states(req, router)

        return response.model_dump(exclude_none=True)

    except NotImplementedError as e:
        # Model doesn't support hidden state extraction
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        # Invalid request parameters
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error extracting hidden states: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/hidden_states/structured",
    summary="Extract Structured Hidden States",
    description="""
Extract hidden states with server-side chat template application and token boundary tracking.

**Key Differences from /v1/hidden_states:**
- Accepts chat components separately (user_prompt, system_prompt, etc.)
- Server applies Qwen3 chat template internally
- Returns token boundary information for each section
- Supports pre-filled thinking/assistant content

**Use Cases:**
- Z-Image embeddings with precise template control
- Token attribution research
- Ablation studies on prompt sections
- Debugging chat template formatting

**Request Body:**
- `model` (string): Model ID to use
- `user_prompt` (string): User message content (required)
- `system_prompt` (string, optional): System prompt content
- `thinking_content` (string, optional): Pre-filled thinking block
- `assistant_content` (string, optional): Pre-filled assistant response
- `enable_thinking` (boolean, optional): Control thinking mode (default: true)
- `layer` (integer, optional): Layer to extract from (default: -2)
- `max_length` (integer, optional): Max sequence length (default: 512)
- `encoding_format` (string, optional): "float" (default) or "base64"
- `return_token_boundaries` (boolean, optional): Return token indices per section
- `return_formatted_prompt` (boolean, optional): Return formatted prompt string

**Note:** Only supported for MLX models with Qwen3-style chat templates.
    """,
    response_description="Hidden states with token boundaries",
    tags=["OpenAI API"],
    responses={
        200: {
            "description": "Structured hidden states extracted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "hidden_states": "SGVsbG8gV29ybGQ=",
                        "shape": [120, 2560],
                        "model": "Qwen3-4B",
                        "layer": -2,
                        "dtype": "bfloat16",
                        "encoding_format": "base64",
                        "token_boundaries": {
                            "system": {"start": 0, "end": 35},
                            "user": {"start": 35, "end": 80}
                        },
                        "token_counts": {
                            "system": 35,
                            "user": 45,
                            "total": 120
                        }
                    }
                }
            }
        },
        422: {
            "description": "Model doesn't support structured hidden state extraction",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Structured hidden states only supported for MLX models."
                    }
                }
            }
        }
    }
)
async def extract_structured_hidden_states(
    request: Request,
    structured_request: dict = Body(...)
):
    """Extract structured hidden states with server-side chat template and token boundaries."""
    from heylook_llm.hidden_states import (
        StructuredHiddenStatesRequest,
        create_structured_hidden_states,
    )

    try:
        req = StructuredHiddenStatesRequest(**structured_request)
        router = request.app.state.router_instance
        response = await create_structured_hidden_states(req, router)
        return response.model_dump(exclude_none=True)

    except NotImplementedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error extracting structured hidden states: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/capabilities",
    summary="Get Server Capabilities",
    description="""
Get detailed information about server capabilities and optimization options.

**Returns:**
- Available performance optimizations
- Supported features and endpoints
- Optimal usage recommendations
- API extensions

**Use this endpoint to:**
- Discover fast endpoints (like multipart upload)
- Check which optimizations are active
- Get recommendations for best performance
- Understand server limits and capabilities

**Client Integration:**
Clients should query this endpoint on startup to discover:
1. Whether to use `/v1/chat/completions/multipart` for images
2. Recommended batch sizes
3. Optimal request patterns
4. Available performance features
    """,
    response_description="Server capabilities and optimization details",
    tags=["Monitoring"]
)
async def get_capabilities(request: Request):
    """Get server capabilities and optimization options."""
    from heylook_llm.optimizations.status import get_optimization_summary

    # Get optimization status
    optimizations = get_optimization_summary()

    # Check Metal availability
    try:
        import mlx.core as mx
        has_metal = mx.metal.is_available()
        if has_metal:
            device_info = mx.metal.device_info()
            metal_info = {
                "available": True,
                "device_name": device_info.get("name", "Unknown"),
                "max_recommended_working_set_size": device_info.get("max_recommended_working_set_size", 0)
            }
        else:
            metal_info = {"available": False}
    except:
        metal_info = {"available": False}

    capabilities = {
        "server_version": "1.0.1",
        "optimizations": optimizations,
        "metal": metal_info,
        "endpoints": {
            "fast_vision": {
                "available": True,
                "endpoint": "/v1/chat/completions/multipart",
                "description": "Raw image upload endpoint - 57ms faster per image",
                "benefits": {
                    "time_saved_per_image_ms": 57,
                    "bandwidth_reduction_percent": 33,
                    "supports_parallel_processing": True
                },
                "usage": {
                    "method": "POST",
                    "content_type": "multipart/form-data",
                    "fields": {
                        "model": "Model ID (required)",
                        "messages": "JSON string of messages with __RAW_IMAGE__ placeholders",
                        "images": "Image files (multiple allowed)",
                        "resize_max": "Max dimension to resize to (optional)",
                        "resize_width": "Specific width to resize to (optional)",
                        "resize_height": "Specific height to resize to (optional)",
                        "image_quality": "JPEG quality 1-100 (default: 85)",
                        "preserve_alpha": "Keep transparency, output PNG (default: false)"
                    }
                },
                "image_processing": {
                    "auto_resize": "Resize images to reduce tokens",
                    "preserve_originals": "Default behavior keeps original dimensions",
                    "format_conversion": "Automatically handles JPEG/PNG conversion",
                    "alpha_support": "Preserves transparency when requested"
                }
            },
            "standard_vision": {
                "endpoint": "/v1/chat/completions",
                "description": "Standard endpoint with base64 images",
                "supports_base64": True
            },
            "batch_processing": {
                "available": True,
                "processing_modes": ["sequential", "parallel", "conversation"],
                "description": "Process multiple prompts in one request"
            }
        },
        "features": {
            "streaming": True,
            "model_caching": {
                "enabled": True,
                "cache_size": 2,
                "eviction_policy": "LRU"
            },
            "vision_models": True,
            "concurrent_requests": True,
            "max_image_size": "No hard limit, auto-resized as needed",
            "supported_image_formats": ["JPEG", "PNG", "WEBP", "BMP", "GIF"]
        },
        "recommendations": {
            "vision_models": {
                "use_multipart": optimizations["image"]["turbojpeg_available"] or optimizations["image"]["xxhash_available"],
                "reason": "Multipart endpoint is faster and uses less bandwidth"
            },
            "batch_size": {
                "optimal": 4,
                "max": 8,
                "note": "Depends on model size and available memory"
            },
            "image_format": {
                "preferred": "JPEG",
                "quality": 85,
                "note": "JPEG with quality 85 provides best size/quality tradeoff"
            },
            "request_pattern": {
                "use_streaming": "For responses > 100 tokens",
                "reuse_connection": "Keep-alive recommended",
                "concurrent_requests": "Safe with different models"
            }
        },
        "limits": {
            "max_tokens": 4096,
            "max_images_per_request": 10,
            "max_request_size_mb": 100,
            "timeout_seconds": 300
        }
    }

    return capabilities


# =============================================================================
# Cache Management Endpoints
# =============================================================================

# Cache manager for persistent cache storage
from heylook_llm.providers.common.prompt_cache import get_global_cache_manager


@app.get("/v1/cache/list",
    summary="List Saved Prompt Caches",
    description="""
List all prompt caches currently in memory.

**Returns:**
- List of cache entries with model ID and token counts
- Cache statistics for each loaded model

**Note:** Currently shows in-memory caches only. Persistent storage coming soon.
    """,
    response_model=CacheListResponse,
    response_description="List of cached prompts",
    tags=["Monitoring"]
)
async def list_caches(request: Request, model: str = None):
    """List all prompt caches, optionally filtered by model."""
    cache_manager = get_global_cache_manager()
    cache_info = cache_manager.get_cache_info()

    caches = []
    for model_id, info in cache_info.items():
        if model and model_id != model:
            continue

        caches.append(CacheInfo(
            cache_id=f"mem-{model_id}",  # In-memory cache ID
            model=model_id,
            name=f"Active cache for {model_id}",
            description="In-memory prompt cache",
            tokens_cached=info.get("tokens_cached", 0),
            size_mb=0.0,  # Unknown for in-memory
            created_at=datetime.utcnow().isoformat()
        ))

    return CacheListResponse(caches=caches)


@app.post("/v1/cache/clear",
    summary="Clear Prompt Caches",
    description="""
Clear prompt caches for a specific model or all models.

**Use Cases:**
- Free memory by clearing unused caches
- Reset cache state when switching contexts
- Troubleshooting cache-related issues

**Note:** This clears in-memory caches. The next request will rebuild the cache.
    """,
    response_model=CacheClearResponse,
    response_description="Number of caches cleared",
    tags=["Monitoring"]
)
async def clear_caches(request: Request, body: CacheClearRequest = Body(default=CacheClearRequest())):
    """Clear prompt caches for a model or all models."""
    cache_manager = get_global_cache_manager()
    cache_info = cache_manager.get_cache_info()

    if body.model:
        # Clear specific model cache
        if body.model in cache_info:
            cache_manager.invalidate_cache(body.model)
            return CacheClearResponse(deleted_count=1)
        return CacheClearResponse(deleted_count=0)
    else:
        # Clear all caches
        count = len(cache_info)
        cache_manager.clear_all()
        return CacheClearResponse(deleted_count=count)


# Import and register multipart endpoint
from heylook_llm.api_multipart import create_chat_multipart
app.post("/v1/chat/completions/multipart",
    summary="Create Chat Completion with Raw Images (Fast)",
    description="""
High-performance vision endpoint that accepts raw image uploads instead of base64.

**🚀 Performance Benefits:**
- ⚡ 57ms faster per image (no base64 encoding/decoding)
- 📉 33% bandwidth reduction
- 🔄 Parallel image processing
- 💾 Smart image caching with xxHash

**How to Use:**
1. Send images as multipart form files
2. Include messages as JSON string with `__RAW_IMAGE__` placeholders
3. Images are injected into messages in order

**Example Request:**
```python
files = [
    ('images', ('img1.jpg', image1_bytes, 'image/jpeg')),
    ('images', ('img2.jpg', image2_bytes, 'image/jpeg'))
]
data = {
    'model': 'llava-1.5-7b-hf-4bit',
    'messages': json.dumps([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these images"},
            {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}},
            {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}}
        ]
    }])
}
response = requests.post(url + '/multipart', files=files, data=data)
```

**Perfect for:**
- ComfyUI integration
- Batch image processing
- Real-time vision applications
- Network-constrained environments
    """,
    tags=["OpenAI API"],
    responses={
        200: {
            "description": "Successful completion",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1677652288,
                        "model": "llava-1.5-7b-hf-4bit",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The first image shows a cat, while the second shows a dog."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 156,
                            "completion_tokens": 23,
                            "total_tokens": 179
                        }
                    }
                }
            }
        }
    }
)(create_chat_multipart)

def _get_api_endpoints():
    """Dynamically discover all /v1/ endpoints from registered routes."""
    endpoints = {}
    for route in app.routes:
        if hasattr(route, 'path') and route.path.startswith('/v1/'):
            # Get methods (GET, POST, etc.)
            methods = getattr(route, 'methods', {'GET'})
            method = next(iter(methods)) if methods else 'GET'
            # Create endpoint name from path
            name = route.path.replace('/v1/', '').replace('/', '_')
            endpoints[name] = {"method": method, "path": route.path}
    return endpoints


@app.get("/",
    summary="Server Information",
    description="Get server information and available endpoints",
    tags=["Monitoring"]
)
async def root():
    """Root endpoint showing server info and available APIs"""
    return {
        "name": "HeylookLLM",
        "version": "1.0.1",
        "description": "High-performance local LLM server with OpenAI-compatible API",
        "documentation": {
            "interactive": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": _get_api_endpoints(),
        "features": {
            "model_providers": ["MLX (Apple Silicon)"],
            "vision_models": True,
            "streaming": True,
            "batch_processing": True,
            "model_caching": "LRU (max 2 models)"
        },
        "quick_start": {
            "1": "GET /v1/models - List available models",
            "2": "POST /v1/chat/completions - Chat with a model",
            "3": "GET /docs - Interactive API documentation"
        }
    }


def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="""
# HeylookLLM API Documentation 🚀

A high-performance API server for local LLM inference with OpenAI-compatible endpoints.

**Platform Support**: macOS (Apple Silicon)
- MLX backend for text and vision inference
- MLX STT backend for speech-to-text

## 🎯 Key Features

### API Compatibility
- **OpenAI API**: Full compatibility with OpenAI clients and libraries

### Model Support
- **MLX Models**: Optimized for Apple Silicon with Metal acceleration
- **Vision Models**: Process images with vision-language models
- **Speech-to-Text**: Parakeet MLX models

### Performance Features
- **Smart Model Caching**: LRU cache keeps 2 models in memory
- **Fast Vision Endpoint**: `/v1/chat/completions/multipart` - 57ms faster per image
- **Async Processing**: Non-blocking request handling
- **GPU Acceleration**: Metal (Apple Silicon)

## Quick Start

### 1. Check Available Models
```bash
curl http://localhost:8080/v1/models
```

### 2. Generate Text
```bash
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen2.5-coder-1.5b-instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Process Images (Fast)
```python
import requests

files = [('images', ('image.jpg', image_bytes, 'image/jpeg'))]
data = {
    'model': 'llava-1.5-7b-hf-4bit',
    'messages': json.dumps([{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}}
        ]
    }])
}
response = requests.post('http://localhost:8080/v1/chat/completions/multipart',
                        files=files, data=data)
```

## 📚 Client Libraries

### OpenAI Python SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen2.5-coder-1.5b-instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Configuration

Models are configured in `models.toml`. The server automatically loads models on demand and manages memory with LRU eviction.

## 📈 Performance Optimization

Install with performance extras for maximum speed:
```bash
pip install heylookllm[performance]
```

This enables:
- uvloop for faster async
- orjson for 10x faster JSON
- TurboJPEG for fast image processing
- xxHash for ultra-fast caching
        """,
        routes=app.routes,
    )

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8080",
            "description": "Default server"
        }
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "GitHub Repository",
        "url": "https://github.com/fblissjr/heylookitsanllm"
    }

    # Enhanced component schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # Add streaming chunk schemas (SSE payload types not auto-discovered by FastAPI)
    from heylook_llm.config import (
        StreamChunk as _StreamChunk,
        StreamChoice as _StreamChoice,
        StreamDelta as _StreamDelta,
        StreamLogprobs as _StreamLogprobs,
        TokenLogprobInfo as _TokenLogprobInfo,
        TopLogprobEntry as _TopLogprobEntry,
        EnhancedUsage as _EnhancedUsage,
        GenerationTiming as _GenerationTiming,
        GenerationConfig as _GenerationConfig,
    )
    for _model in [
        _StreamChunk, _StreamChoice, _StreamDelta, _StreamLogprobs,
        _TokenLogprobInfo, _TopLogprobEntry, _EnhancedUsage, _GenerationTiming, _GenerationConfig,
    ]:
        _schema = _model.model_json_schema(ref_template="#/components/schemas/{model}")
        _name = _model.__name__
        # Move $defs to top-level schemas
        if "$defs" in _schema:
            for _def_name, _def_schema in _schema["$defs"].items():
                openapi_schema["components"]["schemas"][_def_name] = _def_schema
            del _schema["$defs"]
        openapi_schema["components"]["schemas"][_name] = _schema

    # Add example schemas
    openapi_schema["components"]["examples"] = {
        "simple_text_request": {
            "summary": "Simple text completion",
            "value": {
                "model": "qwen2.5-coder-1.5b-instruct-4bit",
                "messages": [
                    {"role": "user", "content": "Write a hello world in Python"}
                ],
                "max_tokens": 256
            }
        },
        "vision_request": {
            "summary": "Vision model request",
            "value": {
                "model": "llava-1.5-7b-hf-4bit",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        ]
                    }
                ],
                "max_tokens": 512
            }
        },
        "streaming_request": {
            "summary": "Streaming response",
            "value": {
                "model": "qwen2.5-coder-1.5b-instruct-4bit",
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stream": True,
                "max_tokens": 1024
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override the default OpenAPI function
app.openapi = custom_openapi
