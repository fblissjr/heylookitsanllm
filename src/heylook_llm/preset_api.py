# src/heylook_llm/preset_api.py
"""User preset API endpoints.

CRUD for named system_prompt + sampler-params bundles, backed by the DuckDB
store (db.py). These are UI-authored and expanded client-side into explicit
request fields. They are now the ONLY named-bundle system: the bundled TOML
sampler registry that used to sit beside them, server-side and request-scoped
via ``ChatRequest.preset``, was removed in v2.0.30.
Wire contract: docs/frontend_v3_spec.md §4.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

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


class Preset(BaseModel):
    """A stored preset. Mirrors db._preset_row_to_dict."""
    id: str
    name: str
    system_prompt: str | None = Field(
        default=None,
        description="Null or empty means the preset carries no prompt: applying it "
                    "leaves the document's prompt alone rather than blanking it.",
    )
    params: dict = Field(
        default_factory=dict,
        description="Sampler knobs the preset pins. The server never applies them: a "
                    "client copies each key onto the request itself (on /v1/messages, "
                    "`enable_thinking` is spelled `thinking`). Absent keys stay unset.",
    )
    created_at: str = Field(description="ISO-8601 UTC timestamp")
    updated_at: str = Field(description="ISO-8601 UTC timestamp")


class PresetList(BaseModel):
    presets: list[Preset]
    total: int


class PresetDeleted(BaseModel):
    status: Literal["deleted"]
    id: str


_Responses = dict[int | str, dict[str, Any]]
_ERR_400: _Responses = {400: {"description": "Blank name, `params` not an object, or no fields to update"}}
_ERR_404: _Responses = {404: {"description": "Preset not found"}}
_ERR_409: _Responses = {409: {"description": "Preset name already exists"}}


@preset_router.get(
    "",
    summary="List Presets",
    description="List all saved presets (system prompt + sampler params), ordered by name.",
    response_model=PresetList,
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
    response_model=Preset,
    responses={**_ERR_400, **_ERR_409},
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
    description="Update preset fields (name, system prompt, params). Only set fields are "
                "patched; a sent `params` REPLACES the stored object, it is not merged.",
    response_model=Preset,
    responses={**_ERR_400, **_ERR_404, **_ERR_409},
)
async def update_preset(preset_id: str, request: Request, body: PresetUpdate):
    conn = _get_db(request)
    # PresetUpdate's fields ARE the updatable set -- db.update_preset
    # re-filters against _UPDATABLE_PRESET_FIELDS and 400s on empty.
    kwargs = {k: getattr(body, k) for k in body.model_fields_set}
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
    response_model=PresetDeleted,
    responses=_ERR_404,
)
async def delete_preset(preset_id: str, request: Request):
    conn = _get_db(request)
    deleted = await db.delete_preset(conn, preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"status": "deleted", "id": preset_id}
