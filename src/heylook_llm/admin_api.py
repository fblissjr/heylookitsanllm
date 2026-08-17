# src/heylook_llm/admin_api.py
"""Admin API endpoints for model management.

All endpoints under /v1/admin/models/ -- separated from the OpenAI-compatible
/v1/models endpoint to avoid breaking existing integrations.
"""

import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request

from heylook_llm.auth import require_admin_token
from heylook_llm.capabilities import effective_capabilities
from heylook_llm.config import (
    PROVIDER_CONFIG_CLASSES,
    AdminModelListResponse,
    AdminModelResponse,
    AdminValidationResult,
    BulkDefaultSamplerRequest,
    configurable_fields,
    FitRequest,
    FitResponse,
    ModelImportRequest,
    ModelScanRequest,
    ModelStatusResponse,
    ModelUpdateRequest,
    ModelValidateRequest,
    SamplerInfo,
    SamplerListResponse,
    ScannedModelListResponse,
)
from heylook_llm.model_service import ModelService

logger = logging.getLogger(__name__)

admin_router = APIRouter(
    prefix="/v1/admin/models",
    tags=["Admin"],
    dependencies=[Depends(require_admin_token)],
)


def _get_service(request: Request) -> ModelService:
    """Get the ModelService instance from app state."""
    if not hasattr(request.app.state, 'model_service'):
        router = request.app.state.router_instance
        request.app.state.model_service = ModelService(router.config_path)
    return request.app.state.model_service


def _get_loaded_model_ids(request: Request) -> set[str]:
    """Get the set of currently loaded model IDs."""
    router = request.app.state.router_instance
    return set(router.get_loaded_models().keys())


def _served_configs(request: Request) -> list:
    """Every model the ROUTER can serve -- models.toml plus discovery.

    The router's ``app_config`` is the merged snapshot built at startup and
    refreshed on reload, so it already contains discovered models. Reading it
    (rather than rescanning per request) is what keeps this surface honest:
    everything listed here resolves in ``router.get_provider``, so the v3
    Load button cannot 404 on a row the page just drew. It also keeps the
    recursive filesystem walk off the event loop -- these routes are async and
    never await, so a seconds-scale scan here freezes in-flight SSE streams.
    """
    return list(request.app.state.router_instance.app_config.models)


def _resolve_config(request: Request, model_id: str):
    """A model's config whether it is written down or only discovered."""
    written = _get_service(request).get_config(model_id)
    if written is not None:
        return written
    return next((m for m in _served_configs(request) if m.id == model_id), None)


def _safe_reload_config(request: Request) -> str | None:
    """Reload router config, returning a warning string on failure instead of raising."""
    try:
        request.app.state.router_instance.reload_config()
        return None
    except Exception as e:
        logger.error(f"Config reload failed after update: {e}")
        return f"Config saved but runtime reload failed: {e}. Changes will apply on next restart."


def _model_config_to_response(mc, loaded_ids: set[str], router=None) -> AdminModelResponse:
    """Convert a ModelConfig to an AdminModelResponse.

    ``capabilities`` is DERIVED through the same shared helper /v1/models
    uses. Reading ``mc.capabilities`` directly meant reporting the stored
    override, which is empty on every entry that never hand-wrote one -- so
    the Models page listed no capabilities at all while the chat page, one
    endpoint over, gated its whole UI on them.

    ``config`` is the STORED keys only (exclude_unset), not the resolved
    model. Absent IS how a default is spelled in models.toml, and the config
    editor renders exactly that distinction: a stored key is a set value, a
    missing one shows the schema default as a placeholder, and the row's
    non-default summary chip only exists because this response can tell the
    two apart. A resolved dump made every default look deliberately chosen
    (chip on every row, ``n_gpu_layers 999`` rendered as an explicit choice).
    """
    loaded = mc.id in loaded_ids
    return AdminModelResponse(
        id=mc.id,
        provider=mc.provider,
        description=mc.description,
        tags=mc.tags,
        enabled=mc.enabled,
        capabilities=effective_capabilities(mc),
        config=(mc.config.model_dump(exclude_unset=True)
                if hasattr(mc.config, 'model_dump') else dict(mc.config)),
        loaded=loaded,
        stale_reload_fields=(
            router.stale_reload_fields(mc.id) if router is not None and loaded else []
        ),
    )


# =============================================================================
# Route registration order within admin_router matters!
#
# FastAPI matches routes in registration order. Because {model_id:path} uses
# Starlette's :path converter (greedy, matches slashes), a request like
# GET /v1/admin/models/my-model/status would be swallowed by a catch-all
# GET /{model_id:path} if the catch-all is registered first.
#
# Order: fixed paths -> sub-resource paths -> catch-all paths
# =============================================================================

# --- List (fixed path, no conflict) ---

@admin_router.get(
    "",
    summary="List All Model Configs",
    description="List all model configurations including disabled models, with full config details.",
    response_model=AdminModelListResponse,
)
async def list_model_configs(request: Request):
    router = request.app.state.router_instance
    loaded_ids = _get_loaded_model_ids(request)
    configs = _served_configs(request)
    models = [_model_config_to_response(c, loaded_ids, router) for c in configs]
    return AdminModelListResponse(models=models, total=len(models))


# --- Create (fixed path, no conflict) ---

@admin_router.post(
    "",
    summary="Add Model Config",
    description="Add a new model configuration to models.toml.",
    response_model=AdminModelResponse,
    status_code=201,
)
async def add_model_config(request: Request, body: dict):
    service = _get_service(request)
    router = request.app.state.router_instance
    try:
        config = service.add_config(body)
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        response = _model_config_to_response(config, loaded_ids, router)
        result = response.model_dump()
        if warning:
            result["warning"] = warning
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Sub-resource routes (must register BEFORE catch-all) ---

@admin_router.get(
    "/{model_id:path}/status",
    summary="Get Model Status",
    description="Get runtime status of a model (loaded state, memory, metrics).",
    response_model=ModelStatusResponse,
)
async def get_model_status(model_id: str, request: Request):
    router = request.app.state.router_instance
    status = router.get_model_status(model_id)
    return ModelStatusResponse(**status)


@admin_router.post(
    "/{model_id:path}/fit",
    summary="Evaluate Memory Fit",
    description=(
        "Will this model (with optional candidate config edits) fit memory? "
        "Server-computed via heylook_llm.ram_fit -- clients must render this, "
        "never re-derive it. hard_working_set carries the engine asymmetry: "
        "over the Metal working set is FAIL for MLX, WARN for gguf."
    ),
    response_model=FitResponse,
)
def evaluate_model_fit(model_id: str, request: Request, body: FitRequest):
    # Sync on purpose: FastAPI runs it on a worker thread, keeping the file
    # stats + vm_stat/sysctl subprocess calls off the event loop. The only
    # MLX touched is mx.device_info() (a device query, no stream work) and
    # ram_fit caches it after the first call.
    from dataclasses import asdict

    from heylook_llm.ram_fit import fit_for_config

    mc = _resolve_config(request, model_id)
    if mc is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # exclude_none: absent and None both mean "use the provider default";
    # sizing only reads the path fields anyway.
    config = mc.config.model_dump(exclude_none=True)
    for key, value in body.config_overrides.items():
        if value is None:
            # null = reset-to-default, same spelling as the PATCH contract
            config.pop(key, None)
        else:
            config[key] = value

    # Provider-derived, not layout-guessed: gguf is the one engine that
    # treats the working set as advisory.
    hard_working_set = mc.provider != "gguf"
    report = fit_for_config(config, body.headroom_gb, hard_working_set=hard_working_set)
    if report.weights_gb == 0.0 and "model_path does not exist" in report.sizing_notes:
        raise HTTPException(
            status_code=422,
            detail={"field": "model_path", "error": "model_path does not exist"},
        )
    return FitResponse(**asdict(report))


@admin_router.post(
    "/{model_id:path}/toggle",
    summary="Toggle Model Enabled",
    description="Toggle a model's enabled/disabled state.",
    response_model=AdminModelResponse,
)
async def toggle_model(model_id: str, request: Request):
    service = _get_service(request)
    router = request.app.state.router_instance
    try:
        config = service.toggle_enabled(model_id)
        _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        return _model_config_to_response(config, loaded_ids, router)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@admin_router.post(
    "/{model_id:path}/load",
    summary="Load Model",
    description="Explicitly load a model into the LRU cache.",
)
async def load_model(model_id: str, request: Request, warm: bool = False):
    """Load a model; with ``warm=true`` also run a 1-token generation.

    ``warm`` exists so spawn harnesses (scripts/dev_server.sh,
    tests/e2e/lib/server.mjs) have ONE canonical readiness call instead of
    each inventing poll-the-model-list + hand-rolled warm requests: load
    puts weights in the LRU; the warm generation additionally pays the
    first-forward-pass cost (Metal kernel JIT) through the normal
    generation path (FIFO gate, sampler cascade). Returns 200 with
    ``warmed: false`` + ``warm_error`` if the warm generation fails --
    the model is loaded and the server usable either way.
    """
    router = request.app.state.router_instance
    return await _load_and_warm(router, model_id, warm)


async def _load_and_warm(router, model_id: str, warm: bool) -> dict:
    """The one load(+warm) body -- shared by /load and /reload so the warm
    contract cannot fork between them."""
    try:
        import asyncio
        provider = await asyncio.to_thread(router.get_provider, model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    result: dict = {"status": "loaded", "model_id": model_id}
    if warm:
        from heylook_llm.config import ChatMessage, ChatRequest

        warm_request = ChatRequest(
            model=model_id,
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=1,
            stream=False,
        )

        def _consume() -> None:
            gen = provider.create_chat_completion(warm_request)
            try:
                for _ in gen:
                    pass
            finally:
                gen.close()

        start = time.time()
        try:
            await asyncio.to_thread(_consume)
            result["warmed"] = True
            result["warm_ms"] = int((time.time() - start) * 1000)
        except Exception as e:
            result["warmed"] = False
            result["warm_error"] = str(e)[:500]
    return result


@admin_router.post(
    "/{model_id:path}/reload",
    summary="Reload Model",
    description=(
        "Unload then load(+warm) as ONE server-owned operation -- the "
        "browser-driven unload-then-load pair could strand a model unloaded "
        "if the client died between the calls. Same response shape as /load."
    ),
)
async def reload_model(model_id: str, request: Request, warm: bool = False):
    import asyncio

    router = request.app.state.router_instance
    # Re-read models.toml first: the v3 editor flow has already
    # reload_config'd after its PATCH, but a hand-edit of the file has not --
    # without this, "reload" would rebuild the provider from stale config.
    try:
        await asyncio.to_thread(router.reload_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config reload failed: {e}")
    # A load already in flight would be silently JOINED by _load_and_warm
    # (built from the pre-save config snapshot) while this route reports a
    # reload that never happened -- refuse honestly instead. Best-effort:
    # a load STARTING between this check and the load below still joins;
    # closing that fully would mean holding the per-model load lock across
    # the async boundary, which is not worth the machinery for an admin op.
    if router.is_loading(model_id):
        raise HTTPException(
            status_code=409,
            detail=f"A load of '{model_id}' is already in flight; "
                   f"reload after it completes.",
        )
    try:
        # to_thread: the unload drain can wait up to 30s on active
        # generations, and running it on the event loop would freeze the
        # very SSE streams it is waiting to drain (guaranteed timeout +
        # force-unload under an active Metal command buffer).
        # A reload of an unloaded model is just a load -- unload_model
        # returning False (not loaded) is not an error here.
        await asyncio.to_thread(router.unload_model, model_id)
    except RuntimeError as e:
        # Pinned (RLM job / j-space analysis in progress): the caller can
        # act on this -- it is a conflict, not a server fault.
        raise HTTPException(status_code=409, detail=str(e))
    return await _load_and_warm(router, model_id, warm)


@admin_router.post(
    "/{model_id:path}/unload",
    summary="Unload Model",
    description="Explicitly unload a model from the LRU cache.",
)
async def unload_model(model_id: str, request: Request):
    import asyncio

    router = request.app.state.router_instance
    # Same two mechanisms as /reload (ride-along fixes, same review): the
    # unload drain must not run ON the event loop (it waits on generations
    # whose SSE delivery the loop drives), and a pinned model is a 409 the
    # caller can act on, not a raw 500.
    try:
        unloaded = await asyncio.to_thread(router.unload_model, model_id)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if unloaded:
        return {"status": "unloaded", "model_id": model_id}
    return {"status": "not_loaded", "model_id": model_id}


# --- Catch-all routes (LAST -- {model_id:path} is greedy) ---

@admin_router.get(
    "/{model_id:path}",
    summary="Get Model Config",
    description="Get full configuration for a single model.",
    response_model=AdminModelResponse,
)
async def get_model_config(model_id: str, request: Request):
    router = request.app.state.router_instance
    config = _resolve_config(request, model_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    loaded_ids = _get_loaded_model_ids(request)
    return _model_config_to_response(config, loaded_ids, router)


@admin_router.patch(
    "/{model_id:path}",
    summary="Update Model Config",
    description="Update specific fields of a model's configuration. Returns which fields require reload.",
)
async def update_model_config(model_id: str, request: Request, updates: ModelUpdateRequest):
    service = _get_service(request)
    router = request.app.state.router_instance
    try:
        update_dict = updates.model_dump(exclude_none=True)
        config, reload_fields = service.update_config(model_id, update_dict)
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        response = _model_config_to_response(config, loaded_ids, router)
        result = {
            "model": response.model_dump(),
            "reload_required_fields": reload_fields,
        }
        if warning:
            result["warning"] = warning
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        # A value pydantic accepts but TOML cannot serialize reaches the writer
        # and raises AFTER validation passed. That is a bad request, not a
        # server fault -- as a 500 it tells the caller nothing about which
        # value was rejected. (Nulls, the common case, are handled upstream as
        # "unset this field" and never get here.)
        raise HTTPException(
            status_code=400,
            detail=f"Config value is not storable in models.toml: {e}",
        )


@admin_router.delete(
    "/{model_id:path}",
    summary="Remove Model Config",
    description="Remove a model from configuration. Model files stay on disk.",
)
async def remove_model_config(model_id: str, request: Request):
    import asyncio

    service = _get_service(request)
    router = request.app.state.router_instance

    # Unload if currently loaded -- off the event loop, and pinned = 409
    # BEFORE the config row is deleted (removing a model an RLM job is
    # actively running would be the worse half of the failure).
    try:
        await asyncio.to_thread(router.unload_model, model_id)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if not service.remove_config(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    warning = _safe_reload_config(request)
    result: dict = {"status": "removed", "model_id": model_id}
    if warning:
        result["warning"] = warning
    return result


# --- Scan / Import ---

# NOTE: These are POST endpoints at fixed paths -- they must be registered
# BEFORE the catch-all {model_id:path} GET/PATCH/DELETE routes. FastAPI
# resolves routes in registration order, so we use the router's add_api_route
# to control order. The final registration order is handled at the bottom.

async def _discovered_models(request: Request):
    """Return the passively-discovered models cache (C3).

    Populated by ``MemoryManager`` scanning the ``[scan]`` folders + HF cache
    at ``scan_interval_seconds``. Distinct from the active ``POST /scan`` which
    runs synchronously against user-specified paths. Endpoint is read-only;
    the frontend hits ``POST /v1/admin/models/import`` on click-to-add.
    """
    memory_manager = getattr(request.app.state, "memory_manager", None)
    if memory_manager is None:
        return {"discovered": [], "last_scan_ts": 0.0, "count": 0}
    try:
        return memory_manager.discovered_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery snapshot failed: {e}")


async def _scan_for_models(request: Request, scan_request: ModelScanRequest):
    """Scan filesystem for importable models."""
    service = _get_service(request)
    try:
        results = service.scan_paths(
            paths=scan_request.paths or [],
            scan_hf=scan_request.scan_hf_cache,
        )
        return {
            "models": [asdict(r) for r in results],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")


async def _import_models(request: Request, import_request: ModelImportRequest):
    """Import scanned models with configuration."""
    service = _get_service(request)
    router = request.app.state.router_instance
    try:
        imported = service.import_models(
            models_to_import=import_request.models,
            default_sampler=import_request.default_sampler,
        )
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        result: dict = {
            "imported": [_model_config_to_response(c, loaded_ids, router).model_dump() for c in imported],
            "total": len(imported),
        }
        if warning:
            result["warning"] = warning
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Validate ---

async def _validate_config(request: Request, validate_request: ModelValidateRequest):
    """Validate a model config without saving."""
    service = _get_service(request)
    result = service.validate_config(validate_request.model_dump())
    return AdminValidationResult(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
    )


# --- Samplers (named sampler configs; called "profiles", then briefly
# "sampler presets", until the 2026-07-20 naming unification) ---

async def _list_samplers(request: Request):
    """List available named samplers (the bundled SamplerRegistry)."""
    service = _get_service(request)
    samplers_dict = service.get_samplers()
    samplers = [SamplerInfo(name=k, description=v["description"]) for k, v in samplers_dict.items()]
    return SamplerListResponse(samplers=samplers)


async def _bulk_set_default_sampler(request: Request, body: BulkDefaultSamplerRequest):
    """Record a named sampler as default_sampler on multiple models."""
    service = _get_service(request)
    router = request.app.state.router_instance
    try:
        updated = service.bulk_set_default_sampler(body.model_ids, body.sampler)
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        result: dict = {
            "updated": [_model_config_to_response(c, loaded_ids, router).model_dump() for c in updated],
            "total": len(updated),
        }
        if warning:
            result["warning"] = warning
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Route registration order matters for {model_id:path} catch-all.
# Fixed-path POST routes must come first.
# =============================================================================

# Create a separate router for the fixed-path endpoints that need priority
scan_import_router = APIRouter(
    prefix="/v1/admin/models",
    tags=["Admin"],
    dependencies=[Depends(require_admin_token)],
)

scan_import_router.add_api_route(
    "/scan",
    _scan_for_models,
    methods=["POST"],
    summary="Scan for Models",
    description=(
        "Scan filesystem paths and HF cache for importable models. Body: "
        "{paths?: [], scan_hf_cache: bool} -- `paths` is how local model "
        "folders (a modelzoo of GGUFs, say) are found; the HF cache is a "
        "separate, additive source, not the only one."
    ),
    response_model=ScannedModelListResponse,
)

scan_import_router.add_api_route(
    "/discovered",
    _discovered_models,
    methods=["GET"],
    summary="Discovered Models (Watch Folders)",
    description=(
        "Read-only snapshot of the passive watch-folders discovery cache. "
        "Populated by MemoryManager periodically scanning the [scan].folders + "
        "HF cache. Returns {discovered, last_scan_ts, count}."
    ),
)

scan_import_router.add_api_route(
    "/import",
    _import_models,
    methods=["POST"],
    summary="Import Models",
    description="Import scanned models into configuration.",
)

scan_import_router.add_api_route(
    "/validate",
    _validate_config,
    methods=["POST"],
    summary="Validate Config",
    description="Validate a model config without saving.",
)

scan_import_router.add_api_route(
    "/samplers",
    _list_samplers,
    methods=["GET"],
    summary="List Samplers",
    description=(
        "List available named samplers (bundled SamplerRegistry -- same names "
        "ChatRequest.sampler accepts). Distinct from /v1/presets, the saved "
        "user prompt+sampler bundles. Renamed from /profiles 2026-07-20."
    ),
)

scan_import_router.add_api_route(
    "/bulk-default-sampler",
    _bulk_set_default_sampler,
    methods=["POST"],
    summary="Bulk Set Default Sampler",
    description=(
        "Record a named sampler as default_sampler on multiple models at once. "
        "Renamed from /bulk-profile 2026-07-20."
    ),
)


# =============================================================================
# Server-level admin operations (prefix: /v1/admin, NOT /v1/admin/models)
# =============================================================================

admin_ops_router = APIRouter(
    prefix="/v1/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin_token)],
)


def _flatten_optional(schema: dict) -> dict:
    """Pull the real type out of pydantic's `anyOf: [T, null]` for Optional[T].

    Every optional field arrives wrapped, and a client should not have to
    understand that encoding to render a number input.
    """
    branches = schema.get("anyOf")
    if not branches:
        return schema
    real = [b for b in branches if b.get("type") != "null"]
    return real[0] if len(real) == 1 else schema


def _field_options(cls) -> list[dict]:
    """The editable options for one provider config class.

    Built from ``model_json_schema()`` rather than by introspecting
    annotations: pydantic already emits type, bounds, enum choices and the
    default, AND merges ``json_schema_extra``, so `effect`/`arg`/`ui` ride
    along for free. Hand-rolled introspection here would be a fourth thing to
    keep in step with the field declarations.
    """
    schema = cls.model_json_schema()
    properties = schema.get("properties", {})
    out = []
    for name in sorted(configurable_fields(cls)):
        prop = dict(properties.get(name, {}))
        inner = _flatten_optional(prop)
        entry = {
            "name": name,
            "effect": prop.get("effect"),
            "type": inner.get("type"),
            "default": prop.get("default"),
            "required": name in schema.get("required", []),
        }
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum"):
            if key in inner:
                entry[key] = inner[key]
        # Pass-through hints declared on the field: `arg` (the flag actually
        # emitted -- pinned to the argv builder by a test), `ui`, `shape`
        # ("flag" = a bare flag, no value), and `reason` (why a
        # load_time_only field is not editable, which the class cannot imply).
        for key in ("arg", "ui", "shape", "reason"):
            if key in prop:
                entry[key] = prop[key]
        out.append(entry)
    return out


@admin_ops_router.get(
    "/model-options",
    summary="Model Option Schema",
    description=(
        "Every settable config option per provider, with the effect class that "
        "says WHEN a change takes effect: per_request (live), applies_live "
        "(re-read while loaded), requires_reload (needs a respawn -- confirm "
        "and name the cost), load_time_only (cannot be changed, not even by "
        "reloading), descriptive (changes what we advertise, not the process). "
        "Derived from the provider config classes, so a new field appears here "
        "without touching this route."
    ),
)
async def get_model_options():
    """The schema a UI needs to render per-model defaults generically.

    This is deliberately NOT under /v1/admin/models: that router owns
    `/{model_id:path}`, which would capture this path as a model id.

    It is also the first consumer that can distinguish all the effect classes.
    Both in-process consumers collapse them -- the reload set to a binary, the
    import allowlist to "not identity" -- so until something reads `effect`
    per field, a misclassification is invisible.
    """
    return {
        "providers": {
            name: {"fields": _field_options(cls)}
            for name, cls in PROVIDER_CONFIG_CLASSES.items()
        }
    }


@admin_ops_router.post(
    "/reload",
    summary="Reload Models",
    description=(
        "Reload model configuration and clear model cache without restarting "
        "the server. Clears loaded models, re-reads models.toml, and returns "
        "the new model list."
    ),
)
async def reload_models(request: Request):
    """Reload model configuration without restarting."""
    router = request.app.state.router_instance
    try:
        router.clear_cache()
        router.reload_config()
        return {
            "status": "success",
            "message": "Model configuration reloaded",
            "cache_cleared": True,
            "models_available": router.list_available_models(),
        }
    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
