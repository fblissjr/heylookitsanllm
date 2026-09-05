# src/heylook_llm/admin_api.py
"""Admin API endpoints for model management.

All endpoints under /v1/admin/models/ -- separated from the /v1/models list
(OpenAI list shape) that inference clients read.

``load`` is the one lifecycle op that is NOT here: it moved to
``POST /v1/models/{id}/load`` (model_ops_api.py) because the admin gate
protected nothing an inference request could not already do. ``unload`` and
``reload`` stay, because they stop a model out from under other clients.
"""

import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from heylook_llm.auth import require_admin_token
from heylook_llm.capabilities import config_dict, derived_model_facts
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
    ScanConfigRequest,
    ScanConfigResponse,
    ScannedModelListResponse,
)
from heylook_llm.model_ops_api import load_and_warm
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
    """Reload router config, returning a warning string on failure instead of raising.

    BLOCKING, and by more than a file read: reload_config re-runs discovery
    (model_registry.discover) as well as re-parsing models.toml. Every caller
    must already be off the event loop -- see the threadpool banner below.
    """
    try:
        request.app.state.router_instance.reload_config()
        return None
    except Exception as e:
        logger.error(f"Config reload failed after update: {e}")
        return f"Config saved but runtime reload failed: {e}. Changes will apply on next restart."


def _written_down_ids(request: Request) -> set[str]:
    """Ids with an actual models.toml entry (as opposed to discovered ones).

    One cheap TOML read, shared across a whole list render rather than paid
    per row.
    """
    return {c.id for c in _get_service(request).list_configs()}


def _model_config_to_response(mc, loaded_ids: set[str], router=None,
                              written_ids: set[str] | None = None) -> AdminModelResponse:
    """Convert a ModelConfig to an AdminModelResponse.

    ``source`` distinguishes a models.toml entry from a model discovery
    serves with no entry at all, and it is NOT derivable from ``config``.
    Checked live rather than assumed: a discovered model's ``config`` is not
    empty, it carries whatever the SCANNER set (model_path, mmproj_path,
    modalities, supports_thinking) because those fields really were assigned
    on the entry the merge built. So a discovered model reads on the wire
    exactly like a hand-written one with those keys stored -- which is the
    argument for this field, not against it. The UI needs the difference
    because editing a discovered model WRITES an entry for it, and because
    those values are re-derived every load rather than stored.
    ``written_ids=None`` means "caller did not compute it", which reports
    every model as configured -- the pre-v1.69.0 answer, and the safe one for
    callers that never see discovered models.

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

    ``effective_loader`` is DERIVED, and derived from the config rather than
    from a loaded provider: ``MLXProvider.effective_loader`` is null for every
    model that is not resident, which is most of them, and a field that only
    answers for resident models cannot tell a harness which engines it can
    cover. Provider ``mlx`` is two upstream repos (mlx-lm, mlx-vlm); this is the
    only field on the wire that says which one a row means.
    """
    loaded = mc.id in loaded_ids
    # Every DERIVED value on the row comes from the ONE derivation /v1/models
    # also reads (capabilities.derived_model_facts), so the two lists cannot
    # disagree. They land top-level beside `provider`, never inside `config`:
    # this repo derives the config editor's field list from the provider
    # classes, and a read-only value living in a config class would show up
    # on the models page as an editable option. The per-row filesystem cost
    # (config.json, GGUF header, template probes -- all cached after the first
    # read) is why this route runs on the threadpool.
    facts = derived_model_facts(mc, router if loaded else None)
    return AdminModelResponse(
        id=mc.id,
        provider=mc.provider,
        description=mc.description,
        tags=mc.tags,
        enabled=mc.enabled,
        capabilities=facts.capabilities,
        config=config_dict(mc.config, exclude_unset=True),
        loaded=loaded,
        source=("config" if written_ids is None or mc.id in written_ids
                else "discovered"),
        stale_reload_fields=(
            router.stale_reload_fields(mc.id) if router is not None and loaded else []
        ),
        effective_loader=facts.effective_loader,
        context_length=facts.context_length,
        context_running=facts.context_running,
        thinking_default=facts.thinking_default,
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

# =============================================================================
# Why the mutating handlers below are `def`, not `async def`
#
# Each one does blocking work: a models.toml read/modify/write, and discovery
# (model_registry.discover -- a recursive walk of the [scan] folders plus GGUF
# header reads, unbounded in principle). An `async def` that never awaits runs
# that work ON the event loop, which freezes every in-flight SSE generation
# stream for its whole duration. A plain `def` hands the handler to FastAPI's
# threadpool instead; a handler that genuinely must await (remove_model_config)
# wraps the blocking calls in asyncio.to_thread. Same reasoning, same scan, as
# MemoryManager._maybe_rescan_models (memory.py), which pushes it to an executor
# so a periodic rescan doesn't stall streams.
#
# Note a single mutation runs the walk TWICE -- once in ModelService (to
# materialize a discovered model into an entry, or for the delete guard), once
# in the reload -- so this is not a theoretical stall. Kept as two scans on
# purpose: sharing one snapshot across the write and the reload would mean
# materializing an entry from a scan the reload no longer agrees with, and the
# staleness would land exactly where entries get WRITTEN. Both are off the loop
# now, so the cost is admin-request latency, not stalled streams.
#
# Read routes are models.toml-only or read the router's already-merged snapshot,
# and never scan (see _served_configs) -- but the two that build an
# AdminModelResponse (list, get) are `def` all the same: `effective_loader` is
# derived per row from the model dir's config.json, so a list response is one
# mtime-stat per served model. Small, and still disk, still per request, still
# the event loop if it were `async`.
# =============================================================================

# --- List (fixed path, no conflict) ---

@admin_router.get(
    "",
    summary="List All Model Configs",
    description="List all model configurations including disabled models, with full config details.",
    response_model=AdminModelListResponse,
)
def list_model_configs(request: Request):
    router = request.app.state.router_instance
    loaded_ids = _get_loaded_model_ids(request)
    configs = _served_configs(request)
    written = _written_down_ids(request)
    models = [_model_config_to_response(c, loaded_ids, router, written)
              for c in configs]
    return AdminModelListResponse(models=models, total=len(models))


# --- Create (fixed path, no conflict) ---

@admin_router.post(
    "",
    summary="Add Model Config",
    description="Add a new model configuration to models.toml.",
    response_model=AdminModelResponse,
    status_code=201,
)
def add_model_config(request: Request, body: dict):
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

    from heylook_llm.ram_fit import fit_for_config, unsizeable_reason

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
    # A report that sizes to zero is a NON-ANSWER, not a pass -- 0 GiB clears
    # every ceiling, so returning it would be this gate waving through the
    # exact case it exists to refuse. Asked through the shared predicate
    # rather than re-derived: the hand-written form here required both a zero
    # AND the exact note "model_path does not exist", which `size_config_gb`
    # emits only when the path is missing entirely -- so an existing directory
    # holding no weight files answered "pass" (v1.79.56).
    if (why := unsizeable_reason(report)) is not None:
        raise HTTPException(
            status_code=422,
            detail={"field": "model_path", "error": why},
        )
    return FitResponse(**asdict(report))


@admin_router.post(
    "/{model_id:path}/toggle",
    summary="Toggle Model Enabled",
    description="Toggle a model's enabled/disabled state.",
    response_model=AdminModelResponse,
)
def toggle_model(model_id: str, request: Request):
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
    "/{model_id:path}/reload",
    summary="Reload Model",
    description=(
        "Unload then load(+warm) as ONE server-owned operation -- the "
        "browser-driven unload-then-load pair could strand a model unloaded "
        "if the client died between the calls. Same response shape as /load. "
        "`ctx_size` (gguf only, 400 otherwise) sets the model's context size "
        "for THIS load and persists it as the model's `ctx_size` config -- "
        "the same models.toml write a PATCH makes, so there is one place the "
        "value lives. `0` means Auto: drop the stored value and let "
        "llama-server size the context from the model and device memory. "
        "When the value is unchanged and the model is already resident with "
        "nothing stale, this is a plain load (no restart) -- pressing Load "
        "with the same choice must not throw away a warm process."
    ),
)
async def reload_model(
    model_id: str,
    request: Request,
    warm: bool = False,
    ctx_size: int | None = Query(
        default=None, ge=0,
        description="gguf only. Context size to load with; persisted as the "
                    "model's `ctx_size`. 0 = Auto (unset, llama-server "
                    "decides).",
    ),
):
    import asyncio

    router = request.app.state.router_instance
    if ctx_size is not None:
        # Persist FIRST, through the one config writer, so the reload below
        # builds the provider from the saved value -- and so a later PATCH,
        # the models page, and this route can never disagree about what the
        # model's context is. The provider check reads the merged view (a
        # discovered model has no models.toml entry until this write
        # materializes one); the stored value reads the service, which is the
        # file's truth rather than the router's last-loaded snapshot.
        mc = router.app_config.get_model_config(model_id)
        if mc is None:
            raise HTTPException(status_code=400,
                                detail=f"Model '{model_id}' not found or not enabled")
        if mc.provider != "gguf":
            raise HTTPException(
                status_code=400,
                detail=f"ctx_size applies to gguf models only; '{model_id}' is "
                       f"provider '{mc.provider}' (MLX has no fixed context allocation)",
            )
        service = _get_service(request)
        written = service.get_config(model_id) or mc
        stored = written.config.model_dump(exclude_unset=True).get("ctx_size")
        wanted = ctx_size or None  # 0 -> unset, the models.toml spelling of Auto
        if wanted != stored:
            try:
                await asyncio.to_thread(
                    service.update_config, model_id, {"config": {"ctx_size": wanted}})
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        elif (model_id in router.get_loaded_models()
              and not router.stale_reload_fields(model_id)):
            # Same value, resident, nothing else pending: a restart would only
            # pay the load again for an identical process.
            return await load_and_warm(router, model_id, warm)
    # Re-read models.toml first: the v3 editor flow has already
    # reload_config'd after its PATCH, but a hand-edit of the file has not --
    # without this, "reload" would rebuild the provider from stale config.
    try:
        await asyncio.to_thread(router.reload_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config reload failed: {e}")
    # A load already in flight would be silently JOINED by load_and_warm
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
        # to_thread: the unload drain can wait up to 30s, and running it on
        # the event loop would freeze the very SSE streams it is waiting to
        # drain (guaranteed timeout + force-unload under an active Metal
        # command buffer). The drain is NOT what covers an active generation
        # any more -- unload_model refuses that outright, below -- it covers
        # the cases the refusal does not: requests WAITING at the generation
        # gate (not "generating", but about to be), and force=True, which
        # skips the refusal and lands on the drain, which is when the 30s
        # matters most.
        # A reload of an unloaded model is just a load -- unload_model
        # returning False (not loaded) is not an error here.
        await asyncio.to_thread(router.unload_model, model_id)
    except RuntimeError as e:
        # TWO causes, both conflicts the caller can act on rather than server
        # faults: pinned (RLM job / j-space analysis in progress), and
        # GENERATING. The second is newer and changes what this endpoint does:
        # a reload issued during any generation now refuses rather than waiting
        # the model out -- including a detached run the requester never started
        # and cannot see, since a run outlives the response that began it. The
        # raised message names the remedy (stop it, or force).
        raise HTTPException(status_code=409, detail=str(e))
    return await load_and_warm(router, model_id, warm)


@admin_router.post(
    "/{model_id:path}/unload",
    summary="Unload Model",
    description="Explicitly unload a model from the LRU cache.",
)
async def unload_model(model_id: str, request: Request):
    import asyncio

    router = request.app.state.router_instance
    # Same two mechanisms as /reload (ride-along fixes, same review): the
    # unload drain must not run ON the event loop (it waits on gate waiters
    # whose SSE delivery the loop drives), and a refusal is a 409 the caller
    # can act on, not a raw 500. Refusals: pinned, or generating.
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
def get_model_config(model_id: str, request: Request):
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
def update_model_config(model_id: str, request: Request, updates: ModelUpdateRequest):
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
    description=(
        "Remove a model from configuration. Model files stay on disk. 409 when "
        "the entry is a DISABLED override for a file discovery still finds -- "
        "deleting it would silently re-enable the model."
    ),
)
async def remove_model_config(model_id: str, request: Request):
    import asyncio

    service = _get_service(request)
    router = request.app.state.router_instance

    # Unload if currently loaded -- off the event loop, and a refusal (pinned,
    # or generating) is a 409 BEFORE the config row is deleted: removing a
    # model an RLM job is actively running, or one mid-generation, would be
    # the worse half of the failure.
    try:
        await asyncio.to_thread(router.unload_model, model_id)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # to_thread, not a sync handler: this route has to await the unload above,
    # so the blocking tail (remove_config runs discovery for the disabled-
    # override guard, then the reload runs it again) goes to a thread by hand.
    try:
        removed = await asyncio.to_thread(service.remove_config, model_id)
    except ValueError as e:
        # The disabled-override guard. It refuses because deleting the entry
        # would silently RE-ENABLE a still-discovered model -- a conflict the
        # caller can act on, and its message is the whole point, so it must not
        # surface as a bare 500 with the text swallowed.
        raise HTTPException(status_code=409, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    warning = await asyncio.to_thread(_safe_reload_config, request)
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


def _scan_for_models(request: Request, scan_request: ModelScanRequest):
    """Scan filesystem for importable models (threadpool: the walk blocks)."""
    service = _get_service(request)
    try:
        results = service.scan_paths(
            paths=scan_request.paths or [],
            scan_hf=scan_request.scan_hf_cache,
        )
        # `already_configured` alone stopped being enough at v1.69.0: a model
        # under [scan].folders with no entry reports False, so a UI that
        # offers "Import" on that flag offers it for models the router is
        # ALREADY serving. Mark what the router actually serves, matched on
        # the resolved path (the same identity rule the registry merges on)
        # so a symlinked spelling doesn't read as a different file.
        from heylook_llm.model_registry import path_identity

        served_paths = {
            path_identity(p) for m in _served_configs(request)
            if (p := getattr(m.config, "model_path", None))
        }
        rows = []
        for r in results:
            row = asdict(r)
            row["served"] = path_identity(r.path) in served_paths
            rows.append(row)
        return {"models": rows, "total": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")


def _import_models(request: Request, import_request: ModelImportRequest):
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


def _bulk_set_default_sampler(request: Request, body: BulkDefaultSamplerRequest):
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


def _get_scan_config(request: Request):
    """The [scan] watch folders -- what the server discovers models from.

    models_served is filled here as well as on PUT: the same field reading a
    real count after a write and a hardcoded 0 on read would be a worse lie
    than omitting it.
    """
    saved = _get_service(request).get_scan_config()
    return {**saved, "models_served": len(_served_configs(request))}


def _put_scan_config(request: Request, body: ScanConfigRequest):
    """Update [scan]; a folder added here becomes servable models.

    `def`, not `async def`: this writes models.toml and then reloads the
    router, which re-runs discovery -- see the threadpool banner above.

    The reload is NOT optional here the way it is for a sampler tweak. Adding
    a folder changes WHAT EXISTS, and leaving the router on its old snapshot
    would show the caller a saved config whose models it cannot serve until
    something else happens to reload. The warning field carries a reload that
    failed, same contract as every other write route.
    """
    service = _get_service(request)
    try:
        saved = service.set_scan_config(
            folders=body.folders,
            watch_hf_cache=body.watch_hf_cache,
            scan_interval_seconds=body.scan_interval_seconds,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    warning = _safe_reload_config(request)
    served = len(_served_configs(request))
    result = {**saved, "models_served": served}
    if warning:
        result["warning"] = warning
    return result


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
    "/scan-config",
    _get_scan_config,
    methods=["GET"],
    summary="Get Watch Folders",
    description=(
        "The [scan] table from models.toml: which folders the server "
        "discovers models from. Since v1.69.0 a model found here is SERVED "
        "with no [[models]] entry -- models.toml is override-only."
    ),
    response_model=ScanConfigResponse,
)

scan_import_router.add_api_route(
    "/scan-config",
    _put_scan_config,
    methods=["PUT"],
    summary="Set Watch Folders",
    description=(
        "Update [scan] and reload the router. Absent fields are left alone. "
        "Adding a folder makes every model under it servable; "
        "scan_interval_seconds=0 turns discovery off entirely. Comments in "
        "models.toml survive the write."
    ),
    response_model=ScanConfigResponse,
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
def reload_models(request: Request):
    """Reload model configuration without restarting.

    Sync handler (threadpool): clear_cache drains and unloads every provider
    and reload_config re-runs discovery -- both would freeze in-flight streams
    from the event loop.
    """
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
