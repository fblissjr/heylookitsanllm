# src/heylook_llm/admin_api.py
"""Admin API endpoints for model management.

All endpoints under /v1/admin/models/ -- separated from the OpenAI-compatible
/v1/models endpoint to avoid breaking existing integrations.
"""

import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request

from heylook_llm.auth import require_admin_token
from heylook_llm.config import (
    AdminModelListResponse,
    AdminModelResponse,
    AdminValidationResult,
    BulkDefaultSamplerRequest,
    ModelImportRequest,
    ModelScanRequest,
    ModelStatusResponse,
    ModelUpdateRequest,
    ModelValidateRequest,
    SamplerInfo,
    SamplerListResponse,
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


def _safe_reload_config(request: Request) -> str | None:
    """Reload router config, returning a warning string on failure instead of raising."""
    try:
        request.app.state.router_instance.reload_config()
        return None
    except Exception as e:
        logger.error(f"Config reload failed after update: {e}")
        return f"Config saved but runtime reload failed: {e}. Changes will apply on next restart."


def _model_config_to_response(mc, loaded_ids: set[str]) -> AdminModelResponse:
    """Convert a ModelConfig to an AdminModelResponse."""
    return AdminModelResponse(
        id=mc.id,
        provider=mc.provider,
        description=mc.description,
        tags=mc.tags,
        enabled=mc.enabled,
        capabilities=mc.capabilities,
        config=mc.config.model_dump() if hasattr(mc.config, 'model_dump') else dict(mc.config),
        loaded=mc.id in loaded_ids,
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
    service = _get_service(request)
    loaded_ids = _get_loaded_model_ids(request)
    configs = service.list_configs()
    models = [_model_config_to_response(c, loaded_ids) for c in configs]
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
    try:
        config = service.add_config(body)
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        response = _model_config_to_response(config, loaded_ids)
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
    "/{model_id:path}/toggle",
    summary="Toggle Model Enabled",
    description="Toggle a model's enabled/disabled state.",
    response_model=AdminModelResponse,
)
async def toggle_model(model_id: str, request: Request):
    service = _get_service(request)
    try:
        config = service.toggle_enabled(model_id)
        _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        return _model_config_to_response(config, loaded_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@admin_router.post(
    "/{model_id:path}/load",
    summary="Load Model",
    description="Explicitly load a model into the LRU cache.",
)
async def load_model(model_id: str, request: Request):
    router = request.app.state.router_instance
    try:
        import asyncio
        await asyncio.to_thread(router.get_provider, model_id)
        return {"status": "loaded", "model_id": model_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@admin_router.post(
    "/{model_id:path}/unload",
    summary="Unload Model",
    description="Explicitly unload a model from the LRU cache.",
)
async def unload_model(model_id: str, request: Request):
    router = request.app.state.router_instance
    if router.unload_model(model_id):
        return {"status": "unloaded", "model_id": model_id}
    else:
        return {"status": "not_loaded", "model_id": model_id}


# --- Catch-all routes (LAST -- {model_id:path} is greedy) ---

@admin_router.get(
    "/{model_id:path}",
    summary="Get Model Config",
    description="Get full configuration for a single model.",
    response_model=AdminModelResponse,
)
async def get_model_config(model_id: str, request: Request):
    service = _get_service(request)
    config = service.get_config(model_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    loaded_ids = _get_loaded_model_ids(request)
    return _model_config_to_response(config, loaded_ids)


@admin_router.patch(
    "/{model_id:path}",
    summary="Update Model Config",
    description="Update specific fields of a model's configuration. Returns which fields require reload.",
)
async def update_model_config(model_id: str, request: Request, updates: ModelUpdateRequest):
    service = _get_service(request)
    try:
        update_dict = updates.model_dump(exclude_none=True)
        config, reload_fields = service.update_config(model_id, update_dict)
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        response = _model_config_to_response(config, loaded_ids)
        result = {
            "model": response.model_dump(),
            "reload_required_fields": reload_fields,
        }
        if warning:
            result["warning"] = warning
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@admin_router.delete(
    "/{model_id:path}",
    summary="Remove Model Config",
    description="Remove a model from configuration. Model files stay on disk.",
)
async def remove_model_config(model_id: str, request: Request):
    service = _get_service(request)
    router = request.app.state.router_instance

    # Unload if currently loaded
    router.unload_model(model_id)

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
    try:
        imported = service.import_models(
            models_to_import=import_request.models,
            default_sampler=import_request.default_sampler,
        )
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        result: dict = {
            "imported": [_model_config_to_response(c, loaded_ids).model_dump() for c in imported],
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
    try:
        updated = service.bulk_set_default_sampler(body.model_ids, body.sampler)
        warning = _safe_reload_config(request)
        loaded_ids = _get_loaded_model_ids(request)
        result: dict = {
            "updated": [_model_config_to_response(c, loaded_ids).model_dump() for c in updated],
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
    description="Scan filesystem paths and HF cache for importable models.",
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
