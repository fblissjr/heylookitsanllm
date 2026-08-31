# src/heylook_llm/api.py
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager, closing
from fastapi.openapi.utils import get_openapi

from heylook_llm import __version__
from heylook_llm.optimizations import fast_json as json
from heylook_llm.router import ModelNotFound, ModelRouter
from heylook_llm.providers.abort import AbortEvent
from heylook_llm.providers.base import GenerationFailed, InvalidGenerationRequest
from heylook_llm.busy_response import model_busy_response
from heylook_llm.providers.common.generation_gate import ModelBusyError
from heylook_llm.config import (
    DEFAULT_PORT,
    PROVIDER_CONFIG_CLASSES,
    ChatRequest, ChatCompletionResponse, PerformanceMetrics,
    BatchChatRequest, BatchChatResponse, BatchStats, SystemMetricsResponse,
    CacheInfo, CacheListResponse, CacheClearRequest, CacheClearResponse,
)
from heylook_llm.system_metrics import SystemMetricsCollector
from heylook_llm.perf_collector import (
    ChunkTelemetry,
    RequestEvent,
    ResourceSnapshot,
    get_perf_collector,
    headline_tps,
    net_ttft_ms,
)
from heylook_llm.utils import log_request_start, log_request_stage, log_request_complete, log_full_request_details, log_request_summary, log_response_summary
from heylook_llm.diagnostic_logger import diag_event, exception_detail
from heylook_llm import observability
from heylook_llm.samplers import SamplerNotFound
from heylook_llm.capabilities import effective_capabilities
from heylook_llm.request_registry import resolve_request_id, track_request, tracked_stream
from heylook_llm.reasoning_parser import (
    merge_presplit_thinking,
    parse_reasoning,
    select_reasoning_parser,
)


def _init_logprobs_collector(chat_request, provider, request_id, streaming=True):
    """Initialize logprobs collector if requested. Returns collector or None."""
    if not chat_request.logprobs:
        return None
    try:
        if streaming:
            from heylook_llm.logprobs import StreamingLogprobsCollector as CollectorClass
        else:
            from heylook_llm.logprobs import LogprobsCollector as CollectorClass
        tokenizer = provider.get_tokenizer() if provider else None
        if tokenizer:
            top_logprobs = chat_request.top_logprobs or 5
            collector = CollectorClass(tokenizer, top_logprobs=top_logprobs)
            diag_event("logprobs_init", request_id=request_id, level="debug",
                       top_logprobs=top_logprobs, streaming=streaming)
            return collector
        else:
            logging.warning("Logprobs requested but tokenizer not available from provider")
            diag_event("logprobs_init_failed", request_id=request_id, level="warn",
                       reason="tokenizer_not_available", streaming=streaming)
    except Exception as e:
        logging.warning(f"Failed to initialize logprobs collector: {e}")
        diag_event("logprobs_init_failed", request_id=request_id, level="warn",
                   reason=str(e), streaming=streaming)
    return None


async def _resource_snapshot_loop(app: FastAPI) -> None:
    """Background task: record a ResourceSnapshot every 60 seconds."""
    collector = get_perf_collector()
    while True:
        await asyncio.sleep(60)
        try:
            router: ModelRouter = app.state.router_instance
            metrics_collector = _get_metrics_collector(router)
            metrics = metrics_collector.collect(force_refresh=True)

            # Compute rolling TPS from events in the last 60s. Success-only:
            # failed/503 events carry 0.0 tok/s and would drag the average
            # toward zero (same defect class as the trends aggregation).
            now = time.time()
            recent = [e for e in collector._events if e.timestamp >= now - 60]
            recent_ok = [e for e in recent if e.success]
            avg_tps = (sum(e.tokens_per_second for e in recent_ok) / len(recent_ok)) if recent_ok else 0.0

            collector.record_resource_snapshot(ResourceSnapshot(
                timestamp=now,
                memory_gb=metrics.system.ram_used_gb,
                gpu_percent=0.0,
                tokens_per_second=avg_tps,
                requests=len(recent),
            ))
        except Exception:
            logging.debug("Resource snapshot collection failed", exc_info=True)

        from heylook_llm.memory import safe_mm_call
        safe_mm_call(getattr(app.state, "memory_manager", None), "maybe_log_baseline")
        safe_mm_call(getattr(app.state, "memory_manager", None), "tick")

        # Throttled (~hourly) rotation of the JSONL telemetry streams (size + age).
        from heylook_llm import observability
        observability.maybe_rotate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The router is now initialized in server.py and passed in app.state.
    # Attach the MemoryManager here so it shares the router's lifetime.
    router: ModelRouter = app.state.router_instance
    from heylook_llm.memory import MemoryManager
    memory_manager = MemoryManager(router=router, app_config=router.app_config)
    app.state.memory_manager = memory_manager
    router.memory_manager = memory_manager

    task = asyncio.create_task(_resource_snapshot_loop(app))

    # Initialize conversation database
    from heylook_llm.db import get_connection
    app.state.db = await get_connection()

    # Wire the observability spine from the settings layer (DB > default --
    # operational settings have no env-var override layer, by design)
    # and disclose what's being written (open-source: user must see it's local).
    from heylook_llm.config_api import apply_runtime_settings, observability_log_dir
    _obs = await apply_runtime_settings(app.state.db)
    logging.info(
        "Observability: level=%s · %s (JSONL) · %dd retention · nothing transmitted "
        "· configure/disable: /v1/admin/config",
        _obs.observability_level, observability_log_dir(), _obs.observability_retention_days,
    )
    # AFTER the level is resolved from settings: log_startup_info honors the
    # kill switch, so calling it earlier would read the pre-configure default
    # and skip the record even when telemetry is enabled.
    memory_manager.log_startup_info()

    yield

    # Reap child processes FIRST and in its own try: the gguf provider's
    # "loaded model" is a running llama-server subprocess in its own process
    # group, which the terminal's Ctrl-C never reaches. Anything still loaded
    # when we exit becomes a multi-GB orphan (PPID 1). Ordered ahead of the DB
    # close, and guarded, so no later teardown failure can strand it.
    try:
        router.unload_all()
    except Exception:
        logging.error("Error unloading models at shutdown", exc_info=True)

    # Close database connection
    try:
        if hasattr(app.state, "db") and app.state.db:
            await app.state.db.close()
    except Exception:
        logging.error("Error closing the database at shutdown", exc_info=True)

    task.cancel()
    logging.info("Server shut down.")

app = FastAPI(
    title="HeylookLLM - High-Performance Local LLM Server",
    # DERIVED, never hand-written: this sat at 1.20.0 for ~60 releases and is
    # the first thing an integrating client reads off /openapi.json.
    version=__version__,
    # The real description is the narrative in custom_openapi() below, which
    # replaces this wholesale. Left as one line so the two cannot drift:
    # editing THIS string changes nothing a client ever reads.
    description="See the generated description in custom_openapi().",
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
            "name": "Models",
            "description": "Model lifecycle an INFERENCE client may drive: load (and optionally warm) a model before generating, so a cold multi-GB load happens in a request the client can show progress against rather than inside an opaque generate call. Gated like inference, not like admin -- a generate request already loads and can evict"
        },
        {
            "name": "Messages API",
            "description": "Anthropic Messages-style endpoint: top-level system prompt, typed content blocks, block-structured SSE. The wire this project's own frontend speaks."
        },
        {
            "name": "Requests",
            "description": "Cancel an in-flight generation by the X-Request-ID the client sent -- the only way to stop a NON-streaming run, which writes nothing until it finishes and so never notices an abandoned client"
        },
        {
            "name": "RLM",
            "description": "Recursive Language Model inference -- iterative code-driven exploration of long contexts"
        },
        {
            "name": "Admin",
            "description": "Model management endpoints for CRUD, scanning, importing, and monitoring models"
        },
        {
            "name": "Config",
            "description": "Operational settings (observability level/retention, ...) -- runtime CRUD, resolved DB > default. There is deliberately no env-var override layer: an env var silently beating a value set here would be invisible in the UI that set it"
        },
        {
            "name": "Telemetry",
            "description": "Frontend client telemetry ingestion (v3 -> observability events stream)"
        },
        {
            "name": "Monitoring",
            "description": "Performance monitoring and server status endpoints"
        },
        {
            "name": "Conversations",
            "description": "Conversation storage and message management"
        },
        {
            "name": "Notebooks",
            "description": "Notebook storage for text documents with LLM generation"
        },
        {
            "name": "Presets",
            "description": "Saved presets: named system prompt + sampler parameter bundles"
        },
        {
            "name": "JSpace",
            "description": "Jacobian-lens interpretability: read the model's verbalizable workspace"
        }
    ]
)

# MODEL_BUSY -> 503, for every route that does not swallow it (v1.79.57).
#
# THE POINT IS THE DEFAULT. `busy_response.py` has been the one speller since
# v1.79.53, but a helper cannot make anyone call it: v1.79.48 added a fourth
# route that never did, and its `except Exception` turned backpressure into a
# 500 for five releases. Six MORE routes were doing the same when this handler
# was written -- answering 500, 400, and in one case 200 with the busy
# sentence stringified into a data field.
#
# A helper you must remember to call has a census. A handler registered here
# has a POPULATION: every route that lets the exception out. That inverts the
# failure mode -- a new route now has to actively swallow to get this wrong,
# instead of having to actively remember to get it right. The remaining hole
# is exactly that swallow, and it is what tests/unit/test_model_busy_reaches_
# the_handler.py asserts against.
@app.exception_handler(ModelBusyError)
async def _model_busy_handler(request: Request, exc: ModelBusyError):
    return model_busy_response(exc)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Import and include Messages API router
from heylook_llm.messages_api import messages_router
app.include_router(messages_router)

# Model load for inference clients (NOT admin: a generate request already
# loads and can evict, so the admin token was gating nothing)
from heylook_llm.model_ops_api import model_ops_router
app.include_router(model_ops_router)

# Request cancellation (top-level resource: the conversation-scoped
# DELETE /v1/conversations/{id}/generate cannot name a plain /v1/messages call)
from heylook_llm.requests_api import requests_router
app.include_router(requests_router)

# Import and include RLM router
from heylook_llm.rlm import rlm_router
app.include_router(rlm_router)

# Import and include Admin API routers (order matters: fixed paths before catch-all)
from heylook_llm.admin_api import scan_import_router, admin_router, admin_ops_router
app.include_router(scan_import_router)
app.include_router(admin_router)
app.include_router(admin_ops_router)

# Import and include Conversation API router
from heylook_llm.conversation_api import conversation_router
app.include_router(conversation_router)

# Conversation-scoped generation (the server-side saga; same tag)
from heylook_llm.conversation_generate_api import generate_router
app.include_router(generate_router)

# Import and include Notebook API router
from heylook_llm.notebook_api import notebook_router
app.include_router(notebook_router)

# Import and include Preset API router
from heylook_llm.preset_api import preset_router
app.include_router(preset_router)

# Operational settings admin (App-DB settings table; DB > default, no env layer)
from heylook_llm.config_api import config_router
app.include_router(config_router)

# Frontend telemetry ingestion (v3 client events -> observability events stream)
from heylook_llm.telemetry_api import telemetry_router
app.include_router(telemetry_router)

# Import and include J-space (Jacobian lens) interpretability router
from heylook_llm.jspace_api import jspace_router
app.include_router(jspace_router)

# Data management
from heylook_llm.auth import require_admin_token, require_api_key

@app.post("/v1/data/clear",
    summary="Clear All Data",
    description="Delete all conversations, messages, and notebooks from the database.",
    tags=["Conversations"],
    dependencies=[Depends(require_admin_token)],
)
async def clear_all_data(request: Request):
    from heylook_llm.db import get_db as _get_db, clear_all_data as _clear
    conn = _get_db(request)
    result = await _clear(conn)
    return result

# Serve the v3 frontend static files at /v3 (the only frontend since
# v1.77.0 -- v2 was deleted at cutover, owner call 2026-08-18)
import pathlib as _pathlib
_v3_frontend_dir = _pathlib.Path(__file__).resolve().parent.parent.parent / "apps" / "heylook-frontend-v3"
if _v3_frontend_dir.is_dir():
    import gzip as _gzip
    import hashlib as _hashlib
    from email.utils import formatdate as _formatdate
    from mimetypes import guess_type

    from starlette.responses import FileResponse, Response

    # v3 has no build step and no content hashes in its URLs, so a cached
    # module can only ever be invalidated by revalidation. Without an explicit
    # Cache-Control a browser applies HEURISTIC freshness (~10% of the file's
    # age at cache time) and skips the request entirely -- which silently
    # mixes module versions: frequently-edited files (chat.js) refetch while
    # rarely-edited ones (preset-bar.js) serve stale for hours, and the new
    # caller calls into the old module ("X is not a function"). no-cache keeps
    # every asset revalidating, which is the property worth having.
    #
    # What it used to COST is the part that was wrong. Starlette's FileResponse
    # sets an etag but has no conditional-request branch (only StaticFiles
    # does), so every revalidation was answered with the whole file -- 427KB
    # across 22 assets, on every load. The note here used to call that "free
    # for a localhost frontend", which is true right up until the client is a
    # phone: iOS Safari discards backgrounded tabs and reloads the document, so
    # it was a half-megabyte transfer per wake-from-eviction. Answering
    # If-None-Match makes an unchanged asset cost a header round trip and no
    # body, with the same no-stale-module guarantee.
    _V3_NO_CACHE = {"Cache-Control": "no-cache"}
    # Text assets only, and only above the size where a round trip dominates.
    _V3_GZIP_TYPES = (".js", ".mjs", ".css", ".html", ".json", ".svg")
    _V3_GZIP_MIN = 1024

    def _v3_etag(stat_result) -> str:
        """Byte-identical to starlette's FileResponse etag (responses.py)."""
        base = f"{stat_result.st_mtime}-{stat_result.st_size}"
        return f'"{_hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()}"'

    # Compressed bytes keyed on the same (mtime, size) the etag derives from.
    # The tree is static between edits, so this is a hit after the first
    # request; without it every cache-missing load re-compressed ~20 assets at
    # level 6 on the event loop -- the same loop delivering SSE tokens.
    # Bounded by the number of gzip-eligible files in the tree (one entry each,
    # older generations of the same path evicted on write), not by traffic.
    _v3_gzip_cache: dict = {}

    def _v3_file_response(path, request: Request):
        stat_result = path.stat()
        base_etag = _v3_etag(stat_result)
        wants_gzip = (path.suffix in _V3_GZIP_TYPES
                      and stat_result.st_size >= _V3_GZIP_MIN
                      and "gzip" in request.headers.get("accept-encoding", ""))
        # RFC 9110: distinct entity-tags per content-coding. One etag across
        # both representations lets a shared cache answer an identity request
        # with a gzip body (or 304 an identity copy against a gzip validator).
        etag = f'{base_etag[:-1]}-gzip"' if wants_gzip else base_etag
        # Vary on EVERY exit, not just the compressed one -- a 304 or an
        # identity 200 stored without it is the same cache-poisoning bug.
        headers = {**_V3_NO_CACHE, "etag": etag, "vary": "accept-encoding"}

        if etag in [t.strip().removeprefix("W/")
                    for t in request.headers.get("if-none-match", "").split(",")]:
            return Response(status_code=304, headers=headers)

        if wants_gzip:
            key = (str(path), stat_result.st_mtime, stat_result.st_size)
            body = _v3_gzip_cache.get(key)
            if body is None:
                body = _gzip.compress(path.read_bytes(), compresslevel=6)
                # Evict older generations of THIS FILE, not the whole tree. The
                # key carries mtime+size, so an edited file lands on a new key
                # and its predecessor would otherwise linger -- that is what
                # "one generation at a time" was reaching for. Clearing the
                # whole dict achieved the opposite: every asset of a ~20-file
                # page load evicted the previous one, so the cache held exactly
                # ONE entry, nothing was ever a hit, and the level-6 compression
                # this exists to keep off the event loop ran on every load --
                # the same loop delivering SSE tokens.
                for stale in [k for k in _v3_gzip_cache if k[0] == key[0]]:
                    del _v3_gzip_cache[stale]
                _v3_gzip_cache[key] = body
            return Response(
                content=body,
                media_type=guess_type(path.name)[0] or "application/octet-stream",
                headers={
                    **headers,
                    "last-modified": _formatdate(stat_result.st_mtime, usegmt=True),
                    "content-encoding": "gzip",
                },
            )
        return FileResponse(path, stat_result=stat_result, headers=headers)

    @app.get("/v3")
    @app.get("/v3/{rest:path}")
    async def serve_v3_frontend(request: Request, rest: str = ""):
        """Serve the v3 frontend SPA -- all routes return index.html."""
        if rest:
            resolved = (_v3_frontend_dir / rest).resolve()
            if resolved.is_relative_to(_v3_frontend_dir) and resolved.is_file():
                return _v3_file_response(resolved, request)
        return _v3_file_response(_v3_frontend_dir / "index.html", request)

@app.get("/v1/models",
    summary="List Available Models",
    description="""
List all language models currently available on this server.

**Use this endpoint to:**
- Discover which models are loaded and ready for inference
- Verify a specific model is available before making requests
- Get model IDs for use in completion requests

**Returns:**
- Model IDs. These are INSTALL-LOCAL -- the registry is override-only, so the
  roster is whatever sits under the scanned folders on this machine. Resolve
  them here at runtime rather than hardcoding one (this description named a
  concrete id for years after that model was gone).
- `provider`, `modalities`, and `capabilities` per row. Gate features on
  `capabilities` (what this server will SERVE) rather than `modalities` (what
  the checkpoint author declared) -- they differ on purpose.
- OpenAI-compatible model objects
- Only shows models marked as `enabled: true` in models.toml
    """,
    response_description="List of available models in OpenAI-compatible format",
    tags=["OpenAI API"]
)
def list_models(request: Request):
    """Get the list of available models in OpenAI format with capabilities.

    Plain ``def`` (FastAPI threadpool), NOT ``async def``: deriving
    capabilities reads each model dir's ``config.json`` -- the template probes
    have always done so, and since v1.79.43 the vision capability resolves
    through the loader router, which stats the dir too. That is the same
    per-row filesystem cost that moved the two admin read routes off the event
    loop; the reads are mtime/lru cached and cheap on a warm local disk, and
    unbounded on a slow or network-mounted one.
    """
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

            # modalities = full author-declared DESCRIPTION (text/vision/audio/
            # video); capabilities below stays gated to what the server actually
            # SERVES (image input today) -- description != served.
            modalities = getattr(model_config.config, "modalities", None)
            if modalities:
                model_entry["modalities"] = modalities

            # Explicit capabilities override, else derive -- one
            # implementation, shared with /v1/admin/models (capabilities.py).
            capabilities = effective_capabilities(model_config)
            if capabilities:
                model_entry["capabilities"] = capabilities

        models_data.append(model_entry)

    return {"object": "list", "data": models_data}


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


def validate_request_sampler(sampler: str | None) -> None:
    """Reject unknown sampler names at the route boundary.

    The deep SamplerNotFound raise happens inside the provider's
    _apply_model_defaults, which runs lazily on first generator advance --
    past the route's guarded stage -- so it escapes as a bare 500. Failing
    here turns a typo'd name into an immediate 400 and skips the model load.
    """
    if not sampler:
        return
    from heylook_llm.samplers import get_sampler_registry
    registry = get_sampler_registry()
    if sampler not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"sampler '{sampler}' not found; known: {registry.list_names()}",
        )


@app.post("/v1/chat/completions",
    summary="Create Chat Completion",
    description="""
Generate text completions from chat messages using the specified model.

**Key Features:**
- Automatic model loading with LRU eviction (size set by `max_loaded_models`,
  default 1 -- so a request for a different model may evict the resident one)
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
            "headers": {
                "x-heylook-peak-memory-gb": {
                    "description": "Peak MLX memory used during this request in GB. Non-streaming only; streaming emits the same value inside the final usage chunk's timing.peak_memory_gb.",
                    "schema": {"type": "string", "example": "4.213"},
                    "required": False,
                },
                "x-heylook-kv-bytes": {
                    "description": "Bytes held in the prompt KV cache at the start of this request. Non-streaming only; streaming emits the same value inside the final usage chunk's timing.kv_cache_bytes.",
                    "schema": {"type": "string", "example": "131072"},
                    "required": False,
                },
            },
        },
    },
    tags=["OpenAI API"],
    dependencies=[Depends(require_api_key)],
)
async def create_chat_completion(request: Request, chat_request: ChatRequest):
    router = request.app.state.router_instance
    # Client-provided request ID, else generated. Through the shared resolver
    # so this endpoint and /v1/messages agree on what a request id may contain
    # -- the id reaches logs and telemetry, and is the handle
    # DELETE /v1/requests/{id} cancels by.
    request_id = resolve_request_id(
        request.headers.get("x-request-id"), prefix="req")

    request_start_time = time.time()

    diag_event("request_start", request_id=request_id,
               model=chat_request.model, stream=chat_request.stream,
               logprobs=bool(chat_request.logprobs))

    validate_request_sampler(getattr(chat_request, "sampler", None))

    try:
        # Add start time for processing time calculation
        chat_request._start_time = request_start_time  # type: ignore[attr-defined]
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
    log_request_start(request_id, chat_request.model or "unknown")

    # Analyze request for image metadata
    request_dict = chat_request.model_dump()
    from heylook_llm.utils import _analyze_images_in_request
    image_stats = _analyze_images_in_request(request_dict)

    # Observability scaffolding -- error paths below may fire before the full
    # perf_ctx is built, so construct a minimal one now so _record_error_event
    # can still emit to memory_manager.
    memory_manager = getattr(request.app.state, "memory_manager", None)
    from heylook_llm.memory import safe_mm_call
    safe_mm_call(memory_manager, "mark_request_start")
    _error_ctx = {
        "memory_manager": memory_manager,
        "image_count": image_stats['count'],
    }

    # Log enhanced request summary
    log_request_summary(
        request_id,
        chat_request.model or "unknown",
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
    provider = None
    # Per-request cooperative abort signal (see below); created before the try
    # so it's always bound for the streaming branch.
    abort_event = AbortEvent()
    # Tracks how far setup progressed, so a caught exception records WHERE it
    # died (which the bare error string never tells you). Updated before each
    # step; read by the except handlers below.
    stage = "routing"
    try:
        log_request_stage(request_id, "routing")
        diag_event("request_routed", request_id=request_id, model=chat_request.model)

        # Run CPU-bound operations in thread pool (timed for perf collection)
        stage = "provider_get"
        provider_get_start = time.time()
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)
        provider_get_ms = (time.time() - provider_get_start) * 1000

        # Backpressure: reject early (503, handled below) if the generation
        # queue is already full, before committing to a response.
        stage = "capacity_check"
        provider.check_capacity()

        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        log_request_stage(request_id, "generating")
        diag_event("generation_start", request_id=request_id,
                   provider=provider.__class__.__name__,
                   provider_get_ms=round(provider_get_ms, 1))
        # Run model generation in thread pool, with the per-request abort signal.
        stage = "generator_create"
        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request, abort_event)

    except RuntimeError as e:
        # TYPED, not a substring (v1.79.57). Both causes -- the gate's
        # check_capacity() and the router's blocked eviction -- already raise
        # ModelBusyError; `"MODEL_BUSY" in str(e)` was four hand-copied
        # spellings of a magic string, which is this repo's named defect class
        # sitting on the one condition it most needed not to miss.
        if isinstance(e, ModelBusyError):
            logging.warning(f"Model busy for request {request_id[:8]}: {e}")
            log_request_complete(request_id, success=False, error_msg="Model busy")
            diag_event("request_error", request_id=request_id, level="warn",
                       error="model_busy", model=chat_request.model, stage=stage)
            _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0, perf_ctx=_error_ctx, chat_request=chat_request)

            # 503 + retry headers so OpenAI-style clients auto-retry. ONE
            # speller for all three endpoints (busy_response.py) -- the three
            # hand-written copies had already drifted in wording, and all three
            # replaced the raised message with a fixed sentence about the queue.
            # That is wrong for the OTHER cause of MODEL_BUSY: eviction blocked
            # because every loaded model is generating, where the raise names
            # the models and the remedy.
            #
            # `provider` may be None here. The old comment claimed it could not
            # be ("MODEL_BUSY only originates from check_capacity()/
            # create_chat_completion(), both after assignment"), and that stopped
            # being true when _evict_lru_model began raising ModelBusyError from
            # inside router.get_provider -- one line BEFORE the assignment. It
            # did not crash only because `provider = None` is pre-bound far
            # above, i.e. the invariant the comment stated was already gone.
            return model_busy_response(e, provider)
        else:
            # Other runtime errors
            logging.error(f"Runtime error: {e}", exc_info=True)
            log_request_complete(request_id, success=False, error_msg=str(e))
            diag_event("request_error", request_id=request_id, level="error",
                       model=chat_request.model, stage=stage, **exception_detail(e))
            _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0, perf_ctx=_error_ctx, chat_request=chat_request)
            raise HTTPException(status_code=500, detail=str(e))

    except SamplerNotFound as e:
        # Bad request: client named a preset the server doesn't have. 400, not 500.
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="warn",
                   error="preset_not_found", model=chat_request.model, stage=stage)
        _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0, perf_ctx=_error_ctx, chat_request=chat_request)
        raise HTTPException(status_code=400, detail=str(e))

    except ModelNotFound as e:
        # Model ROUTING failed: unknown/disabled id, or the request named no
        # model and no `default_model` is configured. The client picked the
        # model, so this is a 400. Deliberately NOT a bare `except ValueError`:
        # get_provider re-raises load failures too (mlx-lm raises plain
        # ValueError for corrupt weights / unsupported model_type), and those
        # are server faults that must keep their 500 and their traceback.
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="warn",
                   error="model_not_resolved", model=chat_request.model, stage=stage)
        _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0, perf_ctx=_error_ctx, chat_request=chat_request)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logging.error(f"Failed to get provider or create generator: {e}", exc_info=True)
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="error",
                   model=chat_request.model, stage=stage, **exception_detail(e))
        _record_error_event(chat_request.model or "unknown", request_start_time, provider_get_ms, image_resize_ms, image_stats['count'] > 0)
        raise HTTPException(status_code=500, detail=str(e))

    # Build perf context for handlers. memory_manager already extracted above
    # (near _error_ctx) and mark_request_start already called.
    perf_ctx = {
        "request_start_time": request_start_time,
        "provider_get_ms": provider_get_ms,
        "image_resize_ms": image_resize_ms,
        "had_images": image_stats['count'] > 0,
        "image_count": image_stats['count'],
        "memory_manager": memory_manager,
    }

    if chat_request.stream:
        return StreamingResponse(
            tracked_stream(
                stream_response_generator_async(generator, chat_request, router, request_id, http_request=request, provider=provider, perf_ctx=perf_ctx, abort_event=abort_event),
                request_id, abort_event),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )
    else:
        # Registered for the whole blocking consume: this is the path with no
        # bytes on the wire until it finishes, so an explicit cancel is the
        # only way to stop it.
        with track_request(request_id, abort_event):
            result = await non_stream_response(generator, chat_request, router, request_id, request_start_time, provider=provider, perf_ctx=perf_ctx, abort_event=abort_event)
        diag_event("generation_complete", request_id=request_id,
                   total_ms=round((time.time() - request_start_time) * 1000, 1))
        response_headers = {"X-Request-ID": request_id}
        peak_gb = perf_ctx.get("peak_memory_gb", 0.0)
        kv_bytes = perf_ctx.get("kv_cache_bytes", 0)
        if peak_gb > 0:
            response_headers["x-heylook-peak-memory-gb"] = f"{peak_gb:.3f}"
        if kv_bytes > 0:
            response_headers["x-heylook-kv-bytes"] = str(kv_bytes)
        if isinstance(result, dict):
            return JSONResponse(content=result, headers=response_headers)
        # Pydantic's model_dump_json is a single-pass serializer; passing the
        # result through JSONResponse(content=result.model_dump(), ...) would
        # re-serialize the whole tree via json.dumps.
        return Response(
            content=result.model_dump_json(),
            media_type="application/json",
            headers=response_headers,
        )

def _provider_type(provider) -> str | None:
    """Provider TYPE ("mlx" | "mlx_embedding" | ...), from the BaseProvider
    contract's ``provider_name`` class attribute (7a) -- no class-name
    sniffing, no mislabeling of future providers."""
    if provider is None:
        return None
    return getattr(provider, "provider_name", "") or None


def _maybe_log_request_event(
    perf_ctx,
    event,
    *,
    chat_request=None,
    peak_memory_gb: float = 0.0,
    kv_cache_bytes: int = 0,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
    content_tokens: int = 0,
    thinking_duration_ms=None,
    content_duration_ms=None,
    stop_reason: str = "stop",
    provider=None,
) -> None:
    """Append one per-request record to memory_manager's request_events.jsonl.

    Content-invariant: record includes sampler knobs + counts + timings, never
    prompt text, response text, or token ID sequences.
    """
    if not perf_ctx:
        return
    mm = perf_ctx.get("memory_manager")
    if mm is None:
        return
    from heylook_llm.memory import safe_mm_call, sampler_summary_from_request
    safe_mm_call(mm, "mark_request_end")
    provider_type = _provider_type(provider)
    try:
        from dataclasses import asdict
        record = asdict(event)
        if chat_request is not None:
            record["sampler_summary"] = sampler_summary_from_request(chat_request)
        record["peak_memory_gb"] = peak_memory_gb
        record["kv_cache_bytes"] = kv_cache_bytes
        record["cached_tokens"] = cached_tokens
        record["thinking_tokens"] = thinking_tokens
        record["content_tokens"] = content_tokens
        if thinking_duration_ms is not None:
            record["thinking_duration_ms"] = thinking_duration_ms
        if content_duration_ms is not None:
            record["content_duration_ms"] = content_duration_ms
        record["stop_reason"] = stop_reason
        record["provider_type"] = provider_type or "unknown"
        record["image_count"] = perf_ctx.get("image_count", 0)
        prompt_tok = int(record.get("prompt_tokens") or 0)
        record["cache_hit_rate"] = round(cached_tokens / prompt_tok, 4) if prompt_tok > 0 else 0.0
        mm.log_request_event(record)
    except Exception:
        logging.debug("memory_manager.log_request_event failed", exc_info=True)

    # Observability spine: mirror the numeric request metrics into the content-free
    # metrics stream (logs/metrics.jsonl) for aggregation. Registry dims via getattr
    # (null for embedding providers, per the frozen §4.3 contract). Best-effort.
    observability.record_event(
        "request_complete", tier="metrics", min_level="minimal",
        fields={
            "model": getattr(event, "model", None),
            "provider": provider_type,
            "effective_loader": getattr(provider, "effective_loader", None),
            "is_vlm": getattr(provider, "is_vlm", None),
            "success": getattr(event, "success", None),
            "prompt_tokens": getattr(event, "prompt_tokens", None),
            "completion_tokens": getattr(event, "completion_tokens", None),
            "generation_tps": getattr(event, "tokens_per_second", None),
            "ttft_ms": getattr(event, "first_token_ms", None),
            "total_ms": getattr(event, "total_ms", None),
            "queue_ms": getattr(event, "queue_ms", None),
            "peak_memory_gb": peak_memory_gb,
            "kv_cache_bytes": kv_cache_bytes,
            "cached_tokens": cached_tokens,
            "stop_reason": stop_reason,
            "image_count": perf_ctx.get("image_count", 0),
            "draft_tokens": getattr(event, "draft_tokens", 0),
            "draft_accepted": getattr(event, "draft_accepted", 0),
        },
    )


def _record_error_event(model: str, request_start_time: float, provider_get_ms: float, image_resize_ms: float, had_images: bool, perf_ctx=None, chat_request=None) -> None:
    """Record a failed request event to perf collector."""
    now = time.time()
    total_ms = (now - request_start_time) * 1000
    error_event = RequestEvent(
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
    )
    get_perf_collector().record_request(error_event)
    _maybe_log_request_event(
        perf_ctx, error_event,
        chat_request=chat_request,
        stop_reason="error",
    )


async def stream_response_generator_async(generator, chat_request: ChatRequest, router, request_id, http_request: Request | None = None, provider=None, perf_ctx: dict | None = None, abort_event=None):
    """Async streaming response generator that runs generation in thread pool.

    Reasoning-aware: a factory-selected ``ReasoningParser`` routes the model
    output stream into ``delta.content`` vs ``delta.thinking`` based on the
    format signals in the model's chat template (harmony multi-channel
    vs. ``<think>`` blocks vs. pass-through). Control tokens are stripped
    regardless of whether the tokenizer's ``skip_special_tokens`` flag
    caught them.

    Enhanced metadata (when stream_options.include_usage=true):
    - thinking_tokens/content_tokens: Separate token counts
    - timing: thinking_duration_ms, content_duration_ms, total_duration_ms
    - generation_config: Sampler settings used
    - stop_reason: Why generation stopped
    """

    model_id = chat_request.model
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    token_count = 0
    telemetry = ChunkTelemetry()  # per-chunk counters/rates tagged by the engine (mlx-lm or llama-server)

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

    # Per-request parser: buffer state must never be shared across requests
    # (interleaved streams on one model corrupt each other; an aborted stream
    # leaves stale buffer for the next). Instantiation is cheap even for
    # Mistral-sized special-token sets -- the compiled strip pattern is cached
    # and shared; only the buffers are per-instance.
    # The provider answers "was this prompt built with thinking on" -- never
    # re-derived here. Reading chat_request.enable_thinking directly skipped
    # the sampler layer the provider templates from, so a request naming the
    # `thinking` sampler armed a content-state parser against a thinking
    # prompt (see BaseProvider.effective_thinking).
    thinking_parser = select_reasoning_parser(
        provider.template_info() if provider else None,
        thinking_enabled=provider.effective_thinking(chat_request) if provider else False,
        continuing=chat_request.is_continuation(),
    )

    # Initialize logprobs collector if requested
    logprobs_collector = _init_logprobs_collector(chat_request, provider, request_id, streaming=True)

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

    def note_delta(delta_type: str) -> None:
        """Timing/counter bookkeeping shared by the pre-split thinking branch
        and the parser-output loop (the same shape messages_api factored into
        StreamingEventTranslator._emit_delta)."""
        nonlocal thinking_start_time, thinking_end_time, content_start_time
        nonlocal first_output_time, thinking_tokens, content_tokens
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

    # Per-request abort event passed in by the route (set on client disconnect
    # to cancel THIS request's generation only).

    from heylook_llm.streaming_utils import async_generator_with_abort, keepalive_sse

    # Generation failure mid-stream: HTTP status is already sent, so the
    # provider's typed exception is translated into an OpenAI-style error
    # payload -- never delivered as an assistant content delta.
    try:
        async for chunk in async_generator_with_abort(generator, http_request, abort_event, log_prefix=f"[API {request_id[:8]}] "):
            ka = keepalive_sse(chunk)  # sentinel guard FIRST (shared spelling)
            if ka:
                yield ka
                continue

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

            # Token counts + memory/queue/rate telemetry (final empty chunk
            # still carries counts and the tightest native rates).
            telemetry.absorb(chunk)

            # Pre-split reasoning (engines that separate it before it reaches
            # us set chunk.thinking): route straight to the thinking channel.
            # The text parser below only ever sees chunk.text.
            chunk_thinking = getattr(chunk, 'thinking', None)
            if chunk_thinking:
                note_delta("thinking")
                yield make_delta("thinking", chunk_thinking)

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
                    # streaming path always constructs StreamingLogprobsCollector
                    # (which owns this method); the factory's union hides it
                    logprobs_delta = logprobs_collector.add_token_and_get_delta(token_id, chunk_logprobs)  # type: ignore[attr-defined]
                elif token_count == 1:
                    # Log once on first token if logprobs data is missing
                    diag_event("logprobs_missing_data", request_id=request_id, level="debug",
                               has_token_id=token_id is not None,
                               has_chunk_logprobs=chunk_logprobs is not None)

            # Update token count periodically
            if token_count % 10 == 0:  # Update every 10 tokens for streaming
                from heylook_llm.utils import log_token_update
                log_token_update(request_id, token_count)

            # Process through thinking parser (uses token ID for Qwen3 thinking blocks)
            deltas = thinking_parser.process_chunk(chunk.text, token_id=token_id)
            for delta_type, text in deltas:
                if text:
                    note_delta(delta_type)
                    yield make_delta(delta_type, text, logprobs_delta)
                    logprobs_delta = None  # Only include logprobs in first delta for this token

    except InvalidGenerationRequest as e:
        # Provider request-validation guards (audio-on-MLX, the continuation
        # guards) only fire at first next() -- after HTTP 200 + headers have
        # flushed -- so a real 400 is impossible here. Type the in-band frame
        # as the CLIENT error it is; before this branch existed they fell
        # through to GenerationFailed and read as server faults.
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="error",
                   model=chat_request.model, stage="streaming",
                   tokens_emitted=token_count, **exception_detail(e))
        error_payload = {"error": {
            "message": str(e),
            "type": "invalid_request_error",
            "code": "invalid_request",
        }}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        return

    except GenerationFailed as e:
        # HTTP 200 + headers were already flushed when streaming began, so a
        # mid-stream failure can only be surfaced in-band. Record it here --
        # this path previously wrote NOTHING to events.jsonl, so an OOM/crash
        # during decode left only a `generation_start` with no matching
        # completion. stage="streaming" distinguishes it from setup-phase errors.
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="error",
                   model=chat_request.model, stage="streaming",
                   tokens_emitted=token_count, **exception_detail(e))
        error_payload = {"error": {
            "message": str(e),
            "type": "server_error",
            "code": "generation_failed",
        }}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        return

    except Exception as e:
        # Unexpected (non-GenerationFailed) mid-stream error. Previously this
        # propagated out of the async generator to Starlette with no diagnostic
        # record, truncating the SSE stream mid-flight. Behavior change: log it,
        # then close the stream cleanly with an in-band error payload + [DONE]
        # (same shape as the GenerationFailed path) instead of propagating a raw
        # exception into an already-started response.
        logging.error(f"Unexpected streaming error for {request_id[:8]}: {e}", exc_info=True)
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="error",
                   model=chat_request.model, stage="streaming",
                   tokens_emitted=token_count, **exception_detail(e))
        error_payload = {"error": {
            "message": "Internal error during generation.",
            "type": "server_error",
            "code": "internal_error",
        }}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        return

    except (GeneratorExit, asyncio.CancelledError):
        # Client disconnected / request cancelled mid-stream (you stopped it, or
        # the browser closed the SSE). The normal finalization below never runs,
        # so log the PARTIAL request here -- this is the "log even if it didn't
        # finish" case. Best-effort, SYNC only (no await/yield -- we're unwinding
        # and must not swallow these BaseExceptions), then re-raise so the
        # generator closes properly.
        try:
            now = time.time()
            observability.record_event(
                "request_complete", tier="metrics", min_level="minimal",
                fields={
                    "model": model_id or "unknown",
                    "provider": _provider_type(provider),
                    "effective_loader": getattr(provider, "effective_loader", None),
                    "is_vlm": getattr(provider, "is_vlm", None),
                    "success": False,
                    "completion_tokens": token_count,
                    "total_ms": round((now - generation_start_time) * 1000, 1),
                    "stop_reason": "abort",
                    "image_count": (perf_ctx or {}).get("image_count", 0),
                },
            )
            log_request_complete(request_id, success=False, error_msg="aborted (client disconnect)")
        except Exception:
            pass
        raise

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
        final_prompt_tokens = telemetry.prompt_tokens or 0
        final_completion_tokens = telemetry.completion_tokens or token_count

        # Build enhanced usage object
        usage_data: dict = {
            "prompt_tokens": final_prompt_tokens,
            "completion_tokens": final_completion_tokens,
            "total_tokens": final_prompt_tokens + final_completion_tokens
        }

        # Report cached token count (from radix tree prompt cache)
        if telemetry.cached_tokens > 0:
            usage_data["prompt_tokens_details"] = {
                "cached_tokens": telemetry.cached_tokens,
            }

        # Add thinking-specific fields if there were thinking tokens
        if thinking_tokens > 0:
            usage_data["thinking_tokens"] = thinking_tokens
            usage_data["content_tokens"] = content_tokens

        # Build timing object
        timing_data: dict[str, int | float] = {
            "total_duration_ms": total_duration_ms
        }
        if thinking_duration_ms is not None:
            timing_data["thinking_duration_ms"] = thinking_duration_ms
        if content_duration_ms is not None:
            timing_data["content_duration_ms"] = content_duration_ms
        if telemetry.peak_memory_gb > 0:
            timing_data["peak_memory_gb"] = round(telemetry.peak_memory_gb, 3)
        if telemetry.kv_cache_bytes > 0:
            timing_data["kv_cache_bytes"] = telemetry.kv_cache_bytes
        if telemetry.queue_wait_ms > 0:
            timing_data["queue_wait_ms"] = round(telemetry.queue_wait_ms, 1)
        if telemetry.draft_tokens > 0:
            timing_data["draft_tokens"] = telemetry.draft_tokens
            timing_data["draft_accepted"] = telemetry.draft_accepted
            timing_data["draft_acceptance"] = round(
                telemetry.draft_accepted / telemetry.draft_tokens, 3)

        # Build generation config from request using the shared sampler-summary
        # helper so the SSE usage chunk and the request_events.jsonl schema stay
        # in lockstep.
        from heylook_llm.memory import sampler_summary_from_request
        generation_config = sampler_summary_from_request(chat_request)

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
        gen_tokens = telemetry.completion_tokens or token_count
        gen_time_s = (now - generation_start_time)
        tps = headline_tps(telemetry.generation_tps, gen_tokens, gen_time_s, telemetry.queue_wait_ms)
        p_get_ms = perf_ctx["provider_get_ms"]

        # Real TTFT: wall clock from generation start to first yielded token,
        # net of FIFO queue wait.
        raw_ttft_ms = (first_output_time - generation_start_time) * 1000 if first_output_time else 0.0
        ttft_ms = net_ttft_ms(raw_ttft_ms, telemetry.queue_wait_ms)

        stream_event = RequestEvent(
            timestamp=now,
            model=model_id or "unknown",
            success=True,
            total_ms=req_total_ms,
            queue_ms=p_get_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            image_processing_ms=perf_ctx["image_resize_ms"] if perf_ctx["had_images"] else 0.0,
            token_generation_ms=total_duration_ms,
            first_token_ms=ttft_ms,
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=tps,
            had_images=perf_ctx["had_images"],
            was_streaming=True,
            queue_wait_ms=round(telemetry.queue_wait_ms, 1),
            prompt_tps=telemetry.prompt_tps,
            draft_tokens=telemetry.draft_tokens,
            draft_accepted=telemetry.draft_accepted,
        )
        get_perf_collector().record_request(stream_event)
        _maybe_log_request_event(
            perf_ctx, stream_event,
            chat_request=chat_request,
            peak_memory_gb=telemetry.peak_memory_gb,
            kv_cache_bytes=telemetry.kv_cache_bytes,
            cached_tokens=telemetry.cached_tokens,
            thinking_tokens=thinking_tokens,
            content_tokens=content_tokens,
            thinking_duration_ms=thinking_duration_ms,
            content_duration_ms=content_duration_ms,
            stop_reason=stop_reason,
            provider=provider,
        )

    yield "data: [DONE]\n\n"

async def non_stream_response(generator, chat_request: ChatRequest, router, request_id, request_start_time, provider=None, perf_ctx: dict | None = None, abort_event=None):
    full_text = ""
    token_count = 0
    pre_thinking_parts: list[str] = []  # chunk.thinking -- engine pre-split reasoning
    telemetry = ChunkTelemetry()  # per-chunk counters/rates tagged by the engine (mlx-lm or llama-server)
    log_request_stage(request_id, "processing_response")

    # Initialize logprobs collector if requested
    logprobs_collector = _init_logprobs_collector(chat_request, provider, request_id, streaming=False)

    # Process generation in thread pool to avoid blocking event loop
    def consume_generator():
        nonlocal full_text, token_count
        first_logprob_logged = False
        # closing() releases the generation gate now (the provider generator's
        # finally) even if consumption raises -- don't wait for GC.
        with closing(generator):
            for chunk in generator:
                chunk_thinking = getattr(chunk, 'thinking', None)
                if chunk_thinking:
                    pre_thinking_parts.append(chunk_thinking)
                full_text += chunk.text
                token_count += 1

                # Collect logprobs if requested and available
                if logprobs_collector:
                    token_id = getattr(chunk, 'token', None)
                    chunk_logprobs = getattr(chunk, 'logprobs', None)
                    if token_id is not None and chunk_logprobs is not None:
                        logprobs_collector.add_token(token_id, chunk_logprobs)
                    elif not first_logprob_logged:
                        first_logprob_logged = True
                        diag_event("logprobs_missing_data", request_id=request_id, level="debug",
                                   has_token_id=token_id is not None,
                                   has_chunk_logprobs=chunk_logprobs is not None,
                                   streaming=False)

                # Update token count periodically for long responses
                if token_count % 25 == 0:
                    from heylook_llm.utils import log_token_update
                    log_token_update(request_id, token_count)

                telemetry.absorb(chunk)

    # Typed generation failures propagate out of the consume thread; translate
    # to HTTP here (client errors 400, server failures 500) -- never content.
    try:
        await asyncio.to_thread(consume_generator)
    except InvalidGenerationRequest as e:
        # Non-streaming errors raise HTTPException past the setup-phase handlers
        # in create_chat_completion, so log the diagnostic here (symmetric with
        # the streaming path). stage="generating" -- failure was during decode.
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="warn",
                   model=chat_request.model, stage="generating",
                   tokens_emitted=token_count, **exception_detail(e))
        raise HTTPException(status_code=400, detail=str(e))
    except GenerationFailed as e:
        log_request_complete(request_id, success=False, error_msg=str(e))
        diag_event("request_error", request_id=request_id, level="error",
                   model=chat_request.model, stage="generating",
                   tokens_emitted=token_count, **exception_detail(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Surface memory telemetry to the route handler for response headers.
    if perf_ctx is not None:
        perf_ctx["peak_memory_gb"] = telemetry.peak_memory_gb
        perf_ctx["kv_cache_bytes"] = telemetry.kv_cache_bytes

    usage_dict: dict = {
        "prompt_tokens": telemetry.prompt_tokens,
        "completion_tokens": telemetry.completion_tokens or token_count,  # Fallback to our count
        "total_tokens": (telemetry.prompt_tokens or 0) + (telemetry.completion_tokens or token_count)
    }

    # Report cached token count (from radix tree prompt cache)
    if telemetry.cached_tokens > 0:
        usage_dict["prompt_tokens_details"] = {
            "cached_tokens": telemetry.cached_tokens,
        }

    # Parse reasoning content with a per-request parser (shared instances
    # race with concurrent streams; see stream_response_generator_async).
    content, thinking = parse_reasoning(
        full_text,
        select_reasoning_parser(
            provider.template_info() if provider else None,
            thinking_enabled=provider.effective_thinking(chat_request) if provider else False,
            continuing=chat_request.is_continuation(),
        ),
    )

    thinking = merge_presplit_thinking(pre_thinking_parts, thinking)

    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking

    # Build choice with optional logprobs. finish_reason is mlx-lm's own
    # (the streaming path forwards the same field): "length" here is the only
    # way a client can tell a max_tokens truncation from a natural stop.
    finish_reason = telemetry.finish_reason or "stop"
    # A CANCELLED run must not report "stop", which asserts the model finished.
    # This path became cancellable in v1.79.44 (DELETE /v1/requests/{id}) and
    # kept the old fallback, so the OpenAI wire got the cancellability without
    # the honesty -- the identical defect fixed on /v1/messages in that same
    # commit, one route over, which is this repo's most-repeated shape.
    # Guarded rather than unconditional: an engine that DID report "length" or
    # a stop sequence is a more specific truth and keeps priority.
    if abort_event is not None and abort_event.is_set() and finish_reason == "stop":
        finish_reason = "length"
    choice = {"message": message, "index": 0,
              "finish_reason": finish_reason}
    if logprobs_collector and logprobs_collector.content:
        choice["logprobs"] = logprobs_collector.to_dict()

    performance = None
    if chat_request.include_performance:
        performance = PerformanceMetrics(
            prompt_tps=telemetry.prompt_tps,
            generation_tps=telemetry.generation_tps,
            peak_memory_gb=telemetry.peak_memory_gb,
        )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=chat_request.model or "unknown",
        choices=[choice],
        usage=usage_dict,
        performance=performance
    )

    # Calculate processing time
    processing_time = time.time() - request_start_time

    # Log response summary
    log_response_summary(
        request_id,
        len(full_text),
        token_count=telemetry.completion_tokens or token_count,
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
        gen_tokens = telemetry.completion_tokens or token_count
        tps = headline_tps(telemetry.generation_tps, gen_tokens, processing_time, telemetry.queue_wait_ms)
        p_get_ms = perf_ctx["provider_get_ms"]
        non_stream_event = RequestEvent(
            timestamp=now,
            model=chat_request.model or "unknown",
            success=True,
            total_ms=processing_time * 1000,
            queue_ms=p_get_ms,
            model_load_ms=p_get_ms if p_get_ms >= 100 else 0.0,
            image_processing_ms=perf_ctx["image_resize_ms"] if perf_ctx["had_images"] else 0.0,
            token_generation_ms=processing_time * 1000 - p_get_ms,
            first_token_ms=0.0,
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=tps,
            had_images=perf_ctx["had_images"],
            was_streaming=False,
            queue_wait_ms=round(telemetry.queue_wait_ms, 1),
            prompt_tps=telemetry.prompt_tps,
            draft_tokens=telemetry.draft_tokens,
            draft_accepted=telemetry.draft_accepted,
        )
        get_perf_collector().record_request(non_stream_event)
        _maybe_log_request_event(
            perf_ctx, non_stream_event,
            chat_request=chat_request,
            peak_memory_gb=telemetry.peak_memory_gb,
            kv_cache_bytes=telemetry.kv_cache_bytes,
            cached_tokens=telemetry.cached_tokens,
            provider=provider,
        )

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
    tags=["OpenAI API"],
    dependencies=[Depends(require_api_key)],
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
                model=model_id or "unknown",
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
    },
    dependencies=[Depends(require_api_key)],
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

    except ModelBusyError:
        raise  # -> app-level 503; a 500 here would call backpressure a failure
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
    },
    dependencies=[Depends(require_api_key)],
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
    except ModelBusyError:
        raise  # -> app-level 503; a 500 here would call backpressure a failure
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
    },
    dependencies=[Depends(require_api_key)],
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
    except ModelBusyError:
        raise  # -> app-level 503; a 500 here would call backpressure a failure
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
- Check which optimizations are active
- Get recommendations for best performance
- Understand server limits and capabilities

**Client Integration:**
Clients should query this endpoint on startup to discover:
1. Recommended batch sizes
2. Optimal request patterns
3. Available performance features
    """,
    response_description="Server capabilities and optimization details",
    tags=["Monitoring"]
)
async def get_capabilities(request: Request):
    """Get server capabilities and optimization options."""
    from heylook_llm import __version__ as server_version
    from heylook_llm.optimizations.status import get_optimization_summary
    from heylook_llm.samplers import get_sampler_registry

    # Get optimization status
    optimizations = get_optimization_summary()

    # Check Metal availability
    try:
        import mlx.core as mx
        has_metal = mx.metal.is_available()
        if has_metal:
            device_info = mx.device_info()
            metal_info = {
                "available": True,
                "device_name": device_info.get("name", "Unknown"),
                "max_recommended_working_set_size": device_info.get("max_recommended_working_set_size", 0)
            }
        else:
            metal_info = {"available": False}
    except Exception:
        metal_info = {"available": False}

    capabilities = {
        "server_version": server_version,
        "optimizations": optimizations,
        "metal": metal_info,
        # Named-sampler discovery (2026-07-20): the bundled SamplerRegistry
        # names, resolvable per-request via ChatRequest.sampler or per-model
        # via models.toml default_sampler. Distinct from /v1/presets (saved
        # user prompt+sampler bundles in the app DB).
        "samplers": {
            "available": get_sampler_registry().list_info(),
            "request_field": "sampler",
            "model_default_field": "default_sampler",
            "distinct_from": "/v1/presets (saved user prompt+sampler bundles)",
        },
        "endpoints": {
            # Both inference wires take images. Naming only the OpenAI one
            # here told a client discovering the server that vision was
            # chat-completions-only, which is the opposite of where the
            # project points integrators.
            "messages": {
                "endpoint": "/v1/messages",
                "description": "Anthropic Messages-conformant wire: top-level system, "
                               "typed content blocks with nested `source`, Anthropic "
                               "stop_reason vocabulary. No server-side resize -- "
                               "clients downscale before sending.",
                "supports_base64": True,
                "server_side_image_resize": False,
            },
            "standard_vision": {
                "endpoint": "/v1/chat/completions",
                "description": "OpenAI-compatible wire with base64 images; "
                               "can downscale server-side via resize_max and friends",
                "supports_base64": True,
                "server_side_image_resize": True,
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
                "eviction_policy": "LRU",
            },
            "vision_models": True,
            "concurrent_requests": True,
            "max_image_size": "No hard limit, auto-resized as needed",
            "supported_image_formats": ["JPEG", "PNG", "WEBP", "BMP", "GIF"]
        },
        "recommendations": {
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
async def list_caches(request: Request, model: str | None = None):
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
            created_at=datetime.now(timezone.utc).isoformat()
        ))

    return CacheListResponse(caches=caches)


@app.post("/v1/cache/clear",
    summary="Clear Prompt Caches",
    dependencies=[Depends(require_admin_token)],
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


def _get_api_endpoints():
    """Every `/v1` endpoint, for the `GET /` discovery payload.

    From the OpenAPI schema, NOT by walking ``app.routes``. This walked the
    routes and reported 12 of 48: a router mounted with ``include_router``
    appears there as an ``_IncludedRouter`` with no ``.path``, so everything
    behind a router was invisible -- ``/v1/messages``, every conversation,
    preset and notebook route, all 19 admin routes, and the cancel endpoint.
    v1.79.45 fixed exactly this in ``server.py``'s startup banner and left
    THIS copy, the one on the surface a client reaches first, still wrong. Two
    walks of the same thing, one fixed, is the shape this repo keeps paying
    for; both now read the schema, which is the surface that cannot drift from
    what is actually served.
    """
    endpoints = {}
    for path, operations in app.openapi().get("paths", {}).items():
        if not path.startswith("/v1/"):
            continue
        # A path can carry several methods; the schema keys them explicitly
        # rather than the arbitrary set-ordering the route walk picked from.
        for method in operations:
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue  # skip `parameters`, `summary` and friends
            name = path.replace("/v1/", "").replace("/", "_").strip("_")
            endpoints.setdefault(name, {"method": method.upper(), "path": path})
    return endpoints


@app.get("/",
    summary="Server Information",
    description="Get server information and available endpoints",
    tags=["Monitoring"]
)
async def root():
    """Root endpoint showing server info and available APIs"""
    from heylook_llm import __version__
    return {
        "name": "HeylookLLM",
        "version": __version__,
        "description": "High-performance local LLM server with OpenAI-compatible API",
        "documentation": {
            "interactive": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": _get_api_endpoints(),
        # Derived, not hand-listed: PROVIDER_CONFIG_CLASSES is the single
        # source of truth for the provider set, and this key claimed MLX was
        # the only one for the three releases since the gguf provider landed.
        # Same staleness class as the OpenAPI header and the /v1/capabilities
        # endpoints map, on the surface a client reaches FIRST.
        "features": {
            "model_providers": sorted(PROVIDER_CONFIG_CLASSES),
            "vision_models": True,
            "audio_input": "gguf models only",
            "streaming": True,
            "batch_processing": True,
            "model_caching": "LRU (size set by max_loaded_models, default 1)"
        },
        "quick_start": {
            "1": "GET /v1/models - what this server is serving right now",
            "2": "POST /v1/messages - generate (Anthropic Messages-shaped)",
            "3": "POST /v1/chat/completions - generate (OpenAI-compatible)",
            "4": "GET /docs - interactive API documentation"
        }
    }


def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=f"""
# HeylookLLM API

Local multimodal LLM inference on Apple Silicon. Two inference wires, one
model registry, per-model engine choice.

Server version **{__version__}**. Default base URL `http://localhost:{DEFAULT_PORT}`.

> **Integrating an external app?** Read
> [docs/api_integration.md](https://github.com/fblissjr/heylookitsanllm/blob/main/docs/api_integration.md)
> first. This schema is generated from the code and is authoritative for
> field names, types and bounds; that document covers which endpoint to pick
> and what will bite you, which a schema cannot express.

## Providers

- **mlx** -- text and vision, via mlx-lm / mlx-vlm, Metal-accelerated.
- **gguf** -- one `llama-server` subprocess per loaded model. Adds audio
  input; MLX rejects audio (400), because audio towers are stripped at load.
- **mlx_embedding** -- embeddings.

## The two inference wires

- **`POST /v1/messages`** (Messages API) -- Anthropic-style: top-level
  `system`, typed content blocks, block-structured SSE ending at
  `message_stop` with no `[DONE]`. This is the wire the bundled `/v3`
  frontend speaks and the direction the project is heading.
  Media blocks use Anthropic's nested `source`
  (`{{"type":"image","source":{{"type":"base64","media_type":...,"data":...}}}}`);
  heylook's older flat spelling is still accepted. It has no server-side
  resize -- clients downscale before sending.
- **`POST /v1/chat/completions`** (OpenAI API) -- OpenAI-compatible, kept
  for external consumers and existing SDK clients. Unlike Messages it can
  downscale images for you via `resize_max` / `resize_width` /
  `resize_height` / `image_quality` / `preserve_alpha`.

Both accept the same sampler knobs, and every knob is optional: **absent
means the server-side sampler cascade decides**. Sending a client-side
default for `max_tokens` silently overrides the model's configured floor.

## Quick start

```bash
curl http://localhost:{DEFAULT_PORT}/v1/models        # what is served right now
curl http://localhost:{DEFAULT_PORT}/v1/capabilities  # version, sampler roster
```

Model ids are **not stable across installs** -- the registry is
override-only, so any model under a scanned folder is served with derived
defaults. Resolve ids from `/v1/models` at runtime and gate features on each
row's `capabilities` (what this server will actually serve) rather than its
`modalities` (what the checkpoint author declared).

```bash
curl http://localhost:{DEFAULT_PORT}/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "<id from /v1/models>",
    "system": "You are concise.",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 256
  }}'
```

Vision, on the OpenAI wire, with server-side downscaling:

```bash
curl http://localhost:{DEFAULT_PORT}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "<a model whose capabilities include \\"vision\\">",
    "messages": [{{"role": "user", "content": [
      {{"type": "text", "text": "What is in this image?"}},
      {{"type": "image_url", "image_url": {{"url": "data:image/jpeg;base64,..."}}}}
    ]}}],
    "resize_max": 1024
  }}'
```

## Client libraries

The OpenAI SDKs work unmodified against `/v1/chat/completions`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:{DEFAULT_PORT}/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="<id from /v1/models>",
    messages=[{{"role": "user", "content": "Hello!"}}],
)
```

## Errors

- **400** -- unknown or disabled `model`, or none given with no server
  default. The reason and the available ids are in `detail`. Pick another
  model.
- **500** -- the model exists but failed to load. That model is broken.
- **422** -- the request body failed validation. The offending field and
  reason are in `detail`.
- **503** -- generation queue full: `{{"error":{{"code":"model_overloaded"}}}}`
  plus `Retry-After`. Back off and retry; the server serialises generation.
- **In-band SSE `error`** -- a failure after the response headers flushed,
  so the status is already 200. Treat `invalid_request_error` as a 400 and
  `api_error` as a 500, and never render its message as model output.

## Operational notes

- **Startup loads nothing**, and the load runs BEFORE the response begins --
  nothing is written to the connection while it does, on either wire. A cold
  model therefore looks like a hang. `POST /v1/models/{{id}}/load` pays it up
  front in a call you can show progress against; it is the same
  `get_provider` the generate call makes, so it adds no work. Add
  `?warm=true` for a readiness call that also pays the first forward pass --
  it takes the generation gate, so keep it out of a per-request pre-flight.
- One model resident by default, LRU eviction; batch work by model.
- Auth is opt-in and off by default: `HEYLOOK_API_KEY`
  (`Authorization: Bearer`, loopback-exempt unless
  `HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true`) gates inference,
  `HEYLOOK_ADMIN_TOKEN` (`X-Heylook-Admin-Token`) gates admin.
- Send `X-Request-ID`. It is echoed back on both wires, it correlates the
  server-side logs, and it is the handle `DELETE /v1/requests/{{id}}` cancels
  by -- the only way to stop a NON-streaming run, which writes nothing until
  it finishes and so never notices an abandoned client.
- Models are configured in `models.toml`, but entries are overrides only --
  a new download needs no edit.
        """,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{DEFAULT_PORT}",
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

    # Add example schemas.
    #
    # Model ids here are PLACEHOLDERS on purpose. A real id is install-local
    # (the registry serves whatever is under the scanned folders), and the
    # concrete ids that used to sit here outlived the models by years --
    # a copy-pasteable example that 400s is worse than an obvious blank.
    _MODEL_PLACEHOLDER = "<id from GET /v1/models>"
    openapi_schema["components"]["examples"] = {
        "simple_text_request": {
            "summary": "Simple text completion (OpenAI wire)",
            "value": {
                "model": _MODEL_PLACEHOLDER,
                "messages": [
                    {"role": "user", "content": "Write a hello world in Python"}
                ],
                "max_tokens": 256
            }
        },
        "vision_request": {
            "summary": "Vision request with server-side downscaling (OpenAI wire)",
            "value": {
                "model": "<a model whose capabilities include \"vision\">",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        ]
                    }
                ],
                "resize_max": 1024,
                "max_tokens": 512
            }
        },
        "streaming_request": {
            "summary": "Streaming response (OpenAI wire)",
            "value": {
                "model": _MODEL_PLACEHOLDER,
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stream": True,
                "max_tokens": 1024
            }
        },
        "messages_text_request": {
            "summary": "Messages wire: top-level system, string content",
            "value": {
                "model": _MODEL_PLACEHOLDER,
                "system": "You are concise.",
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stream": True,
                "max_tokens": 1024
            }
        },
        "messages_vision_request": {
            "summary": "Messages wire: image block (Anthropic's nested source)",
            "value": {
                "model": "<a model whose capabilities include \"vision\">",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "<raw base64, no data: prefix>"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 512
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override the default OpenAPI function
app.openapi = custom_openapi
