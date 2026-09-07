# src/heylook_llm/api.py
"""App assembly (v1.79.67): the FastAPI app, its lifespan, the MODEL_BUSY
handler, CORS, router mounting and the static frontend mount.
Every route lives in a ``*_api.py`` router except ``/v1/data/clear`` below,
``rlm.py``'s own router, and the asset routes ``frontend_static.py`` registers;
the OpenAPI narrative is
``openapi_doc.py``; the route guards the inference routes share are
``request_guards.py``. This module used to carry the OpenAI-compatible chat
route and half a dozen others (2,600 lines at v1.79.65); a new endpoint is a
router module plus one ``include_router`` line here and its tag below.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from heylook_llm import __version__
from heylook_llm.busy_response import model_busy_response
from heylook_llm.monitoring_api import get_metrics_collector
from heylook_llm.openapi_doc import build_openapi
from heylook_llm.perf_collector import ResourceSnapshot, get_perf_collector
from heylook_llm.providers.common.generation_gate import ModelBusyError
from heylook_llm.router import ModelRouter


async def _resource_snapshot_loop(app: FastAPI) -> None:
    """Background task: record a ResourceSnapshot every 60 seconds."""
    collector = get_perf_collector()
    while True:
        await asyncio.sleep(60)
        try:
            router: ModelRouter = app.state.router_instance
            metrics_collector = get_metrics_collector(router)
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
from heylook_llm.model_ops_api import model_ops_router, models_router
app.include_router(models_router)
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

# Monitoring, embeddings and hidden states (in api.py itself until v1.79.67)
from heylook_llm.monitoring_api import monitoring_router
app.include_router(monitoring_router)
from heylook_llm.embeddings_api import embeddings_router
app.include_router(embeddings_router)
from heylook_llm.hidden_states_api import hidden_states_router
app.include_router(hidden_states_router)

# Data management.
#
# THE ONE ROUTE DECLARED INLINE HERE, deliberately -- the module docstring's
# rule has this as its named exception. It fits no existing router and making
# one for a single handler was judged more ceremony than it buys: it is not
# conversation CRUD (it deletes notebooks too), not admin model management,
# not config. That mismatch is the actual reason it is homeless, and moving it
# into conversation_api.py would not fix it -- it would just hide a
# notebook-deleting route under a conversations prefix.
#
# Two things follow, both worth knowing before "tidying" this away:
#   - test_startup_banner.py PINS the placement ("mounted on the app itself,
#     not via a router"), so moving it is a test change, not a pure refactor.
#   - presets and settings deliberately SURVIVE it (db.clear_all_data) -- they
#     are configuration, not data, the same split the schema-bump drop list
#     makes. The frontend's Danger zone and the E2E harness both call it.
from heylook_llm.auth import require_admin_token

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

# The frontend at `/`. Registration order is NOT load-bearing: there is no
# catch-all, and the two path params are scoped under /js/ and /css/, so no
# asset route can shadow the API. Mechanics + invariants: frontend_static.py.
from heylook_llm.frontend_static import mount_frontend
mount_frontend(app)


# The generated document lives in openapi_doc.py; FastAPI caches it on the
# app, so the first /openapi.json (or the startup banner) pays the build once.
def _openapi():
    return build_openapi(app)


app.openapi = _openapi
