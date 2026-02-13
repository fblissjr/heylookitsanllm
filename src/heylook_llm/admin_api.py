# src/heylook_llm/admin_api.py
"""Admin API endpoints for model management.

All endpoints under /v1/admin/models/ -- separated from the OpenAI-compatible
/v1/models endpoint to avoid breaking existing integrations.
"""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from heylook_llm.config import (
    AdminModelListResponse,
    AdminModelResponse,
    AdminValidationResult,
    BulkProfileRequest,
    ModelImportRequest,
    ModelScanRequest,
    ModelStatusResponse,
    ModelUpdateRequest,
    ModelValidateRequest,
    ProfileInfo,
    ProfileListResponse,
)
from heylook_llm.model_service import ModelService

logger = logging.getLogger(__name__)

admin_router = APIRouter(
    prefix="/v1/admin/models",
    tags=["Admin"],
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


# --- List / Get ---

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


# --- Create ---

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


# --- Update ---

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


# --- Delete ---

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


# --- Toggle ---

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


# --- Load / Unload ---

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


# --- Status ---

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


# --- Scan / Import ---

# NOTE: These are POST endpoints at fixed paths -- they must be registered
# BEFORE the catch-all {model_id:path} GET/PATCH/DELETE routes. FastAPI
# resolves routes in registration order, so we use the router's add_api_route
# to control order. The final registration order is handled at the bottom.

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
            profile_name=import_request.profile,
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


# --- Profiles ---

async def _list_profiles(request: Request):
    """List available preset profiles."""
    service = _get_service(request)
    profiles_dict = service.get_profiles()
    profiles = [ProfileInfo(name=k, description=v["description"]) for k, v in profiles_dict.items()]
    return ProfileListResponse(profiles=profiles)


async def _bulk_apply_profile(request: Request, body: BulkProfileRequest):
    """Apply a profile to multiple models."""
    service = _get_service(request)
    try:
        updated = service.bulk_apply_profile(body.model_ids, body.profile)
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
)

scan_import_router.add_api_route(
    "/scan",
    _scan_for_models,
    methods=["POST"],
    summary="Scan for Models",
    description="Scan filesystem paths and HF cache for importable models.",
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
    "/profiles",
    _list_profiles,
    methods=["GET"],
    summary="List Profiles",
    description="List available preset profiles.",
)

scan_import_router.add_api_route(
    "/bulk-profile",
    _bulk_apply_profile,
    methods=["POST"],
    summary="Bulk Apply Profile",
    description="Apply a profile to multiple models at once.",
)
