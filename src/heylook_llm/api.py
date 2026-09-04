# src/heylook_llm/api.py
import asyncio
import logging
import time
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi

from heylook_llm import __version__
from heylook_llm.router import ModelRouter
from heylook_llm.busy_response import model_busy_response
from heylook_llm.providers.common.generation_gate import ModelBusyError
from heylook_llm.config import (
    DEFAULT_PORT,
    PROVIDER_CONFIG_CLASSES,
    SystemMetricsResponse,
    CacheInfo,
    CacheListResponse,
    CacheClearRequest,
    CacheClearResponse,
)
from heylook_llm.system_metrics import SystemMetricsCollector
from heylook_llm.perf_collector import ResourceSnapshot, get_perf_collector
from heylook_llm.diagnostic_logger import diag_event
from heylook_llm import observability
from heylook_llm.capabilities import derived_model_facts


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
            "name": "Models",
            "description": "Model lifecycle an INFERENCE client may drive: load (and optionally warm) a model before generating, so a cold multi-GB load happens in a request the client can show progress against rather than inside an opaque generate call. Gated like inference, not like admin -- a generate request already loads and can evict"
        },
        {
            "name": "Messages API",
            "description": "THE inference wire (v1.79.66: the OpenAI-compatible chat route was removed). Anthropic Messages-conformant: top-level system prompt, typed content blocks, block-structured SSE, plus documented heylook extensions. The wire this project's own frontend speaks."
        },
        {
            "name": "Embeddings",
            "description": "Text embeddings from an mlx_embedding model, in the OpenAI list shape"
        },
        {
            "name": "Hidden States",
            "description": "Hidden-state extraction for analysis and interpretability work"
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
- The OpenAI list shape (`object: "list"`, `data: [...]`), which is what the bundled frontend and external clients read
- Only shows models marked as `enabled: true` in models.toml
    """,
    response_description="List of available models (OpenAI list shape, heylook fields per row)",
    tags=["Models"]
)
def list_models(request: Request):
    """Get the list of available models (OpenAI list shape) with capabilities.

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

            # Capabilities, the thinking default and the context window come
            # from the ONE derivation the admin row also reads
            # (capabilities.derived_model_facts), so a page reading either
            # list gates on the same capabilities, labels its "model default"
            # thinking choice with the value generation will use, and can
            # size a prompt against the ceiling the provider enforces instead
            # of learning it from a 400. `context_length` is null when the
            # files do not say.
            facts = derived_model_facts(model_config)
            if facts.capabilities:
                model_entry["capabilities"] = facts.capabilities
            model_entry["thinking_default"] = facts.thinking_default
            model_entry["context_length"] = facts.context_length

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


@app.post("/v1/embeddings",
    summary="Create Embeddings",
    description="""
Generate embeddings for text using the specified model.

**Key Features:**
- Extract actual model embeddings (not hallucinated numbers)
- Support for both text-only and vision models
- Multiple pooling strategies (mean, cls, last, max)
- Optional dimension truncation

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
    response_description="Embeddings in the OpenAI list shape",
    tags=["Embeddings"],
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
    tags=["Hidden States"],
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
    tags=["Hidden States"],
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
            # ONE inference wire (v1.79.66). Images ride it as content blocks;
            # there is no server-side resize anywhere any more.
            "messages": {
                "endpoint": "/v1/messages",
                "description": "Anthropic Messages-conformant wire: top-level system, "
                               "typed content blocks with nested `source`, Anthropic "
                               "stop_reason vocabulary. No server-side resize -- "
                               "clients downscale before sending.",
                "supports_base64": True,
                "server_side_image_resize": False,
            },
        },
        "features": {
            "streaming": True,
            "model_caching": {
                "enabled": True,
                "eviction_policy": "LRU",
            },
            "vision_models": True,
            "concurrent_requests": True,
            "max_image_size": "No hard limit; clients downscale before sending",
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
        "description": "Local LLM server for Apple Silicon: MLX and gguf models behind one Anthropic Messages-conformant API",
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
            "model_caching": "LRU (size set by max_loaded_models, default 1)"
        },
        "quick_start": {
            "1": "GET /v1/models - what this server is serving right now",
            "2": "POST /v1/messages - generate (Anthropic Messages-conformant)",
            "3": "GET /docs - interactive API documentation"
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

## The inference wire

**`POST /v1/messages`** (Messages API) -- Anthropic Messages-conformant:
top-level `system`, typed content blocks, block-structured SSE ending at
`message_stop` with no `[DONE]`, plus documented heylook extensions
(`heylook_logprobs`, `heylook_progress`, `X-Request-ID` cancellation). This
is the wire the bundled `/v3` frontend speaks. Media blocks use Anthropic's
nested `source`
(`{{"type":"image","source":{{"type":"base64","media_type":...,"data":...}}}}`);
heylook's older flat spelling is still accepted. There is no server-side
resize -- clients downscale before sending. The OpenAI-compatible
`/v1/chat/completions` route was removed in v1.79.66: nothing the project
cares about spoke it, and one generation now has one grammar.

Every sampler knob is optional: **absent means the server-side sampler
cascade decides**. Sending a client-side default for `max_tokens` silently
overrides the model's configured floor.

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

Vision: an image block with Anthropic's nested `source` (downscale it
yourself first; the server does not resize):

```bash
curl http://localhost:{DEFAULT_PORT}/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "<a model whose capabilities include \\"vision\\">",
    "messages": [{{"role": "user", "content": [
      {{"type": "text", "text": "What is in this image?"}},
      {{"type": "image", "source": {{"type": "base64", "media_type": "image/jpeg", "data": "..."}}}}
    ]}}],
    "max_tokens": 512
  }}'
```

## Client libraries

The Anthropic SDKs reach this server with `base_url` set to its origin (the
SDK appends `/v1/messages` itself); the deliberate differences from
Anthropic's own service are listed in `docs/api_integration.md`.

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:{DEFAULT_PORT}", api_key="not-needed")
response = client.messages.create(
    model="<id from /v1/models>",
    max_tokens=256,
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

    # (The OpenAI SSE chunk schemas that used to be appended here left with
    # that route in v1.79.66; the Messages events are declared in schema/.)

    # Add example schemas.
    #
    # Model ids here are PLACEHOLDERS on purpose. A real id is install-local
    # (the registry serves whatever is under the scanned folders), and the
    # concrete ids that used to sit here outlived the models by years --
    # a copy-pasteable example that 400s is worse than an obvious blank.
    _MODEL_PLACEHOLDER = "<id from GET /v1/models>"
    openapi_schema["components"]["examples"] = {
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
