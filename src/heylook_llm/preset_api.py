# src/heylook_llm/preset_api.py
"""User preset API endpoints.

CRUD for named system_prompt + sampler-params bundles, backed by the DuckDB
store (db.py). These are UI-authored and expanded client-side into explicit
request fields -- distinct from the bundled TOML sampler registry
(presets.py), which is server-side and request-scoped via
``ChatRequest.preset``. Wire contract: docs/frontend_v3_spec.md §4.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from heylook_llm import db
from heylook_llm.db import get_db as _get_db

logger = logging.getLogger(__name__)

preset_router = APIRouter(
    prefix="/v1/presets",
    tags=["Presets"],
)


class PresetCreate(BaseModel):
    name: str
    system_prompt: str | None = None
    params: dict = {}


class PresetUpdate(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    params: dict | None = None


@preset_router.get(
    "",
    summary="List Presets",
    description="List all saved presets (system prompt + sampler params), ordered by name.",
)
async def list_presets(request: Request):
    conn = _get_db(request)
    presets = await db.list_presets(conn)
    return {"presets": presets, "total": len(presets)}


@preset_router.post(
    "",
    summary="Create Preset",
    description="Create a named preset. Names are unique.",
    status_code=201,
)
async def create_preset(request: Request, body: PresetCreate):
    conn = _get_db(request)
    try:
        return await db.create_preset(
            conn, name=body.name, system_prompt=body.system_prompt, params=body.params
        )
    except db.PresetNameTaken as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@preset_router.put(
    "/{preset_id}",
    summary="Update Preset",
    description="Update preset fields (name, system prompt, params). Only set fields are patched.",
)
async def update_preset(preset_id: str, request: Request, body: PresetUpdate):
    conn = _get_db(request)
    kwargs = {
        k: getattr(body, k)
        for k in body.model_fields_set
        if k in {"name", "system_prompt", "params"}
    }
    try:
        preset = await db.update_preset(conn, preset_id, **kwargs)
    except db.PresetNameTaken as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:  # no/invalid fields
        raise HTTPException(status_code=400, detail=str(e))
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset


@preset_router.delete(
    "/{preset_id}",
    summary="Delete Preset",
    description="Delete a preset.",
)
async def delete_preset(preset_id: str, request: Request):
    conn = _get_db(request)
    deleted = await db.delete_preset(conn, preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"status": "deleted", "id": preset_id}
