# src/heylook_llm/config_api.py
"""Operational settings admin API (/v1/admin/config).

CRUD over runtime-mutable operational settings (obs level/retention, ...),
persisted in the App DB ``settings`` table (db.py) and resolved
env > DB > default (settings.py). Distinct from the model registry
(``models.toml``, Phase 6) and from user presets (``/v1/presets``). This is the
backend for the v3 admin/settings config panel. Wire contract:
docs/frontend_v3_spec.md §4.
"""

import logging
import os

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import ValidationError

from heylook_llm import db
from heylook_llm.db import get_db as _get_db
from heylook_llm.settings import SettingsSchema, resolve_settings, setting_env_key

logger = logging.getLogger(__name__)

config_router = APIRouter(prefix="/v1/admin/config", tags=["Config"])


async def _snapshot(conn) -> dict:
    """Effective settings + what's stored + what env is forcing (precedence made visible)."""
    stored = await db.get_all_settings(conn)
    effective = resolve_settings(stored)
    env_overrides = [n for n in SettingsSchema.model_fields if setting_env_key(n) in os.environ]
    return {
        "effective": effective.model_dump(),   # env > DB > default -- what's actually in force
        "stored": stored,                       # only explicitly-set DB values
        "env_overrides": env_overrides,         # fields currently pinned by an env var
    }


@config_router.get(
    "",
    summary="Get Config",
    description="Effective operational settings (env > DB > default), the raw stored "
                "overrides, and which fields are currently forced by an env var.",
)
async def get_config(request: Request):
    conn = _get_db(request)
    return await _snapshot(conn)


@config_router.put(
    "",
    summary="Update Config",
    description="Set one or more operational settings. Body is a {key: value} map; "
                "unknown keys and invalid values are rejected (422) before anything "
                "persists. Returns the new effective config.",
)
async def update_config(request: Request, updates: dict = Body(...)):
    conn = _get_db(request)
    # Validate the whole proposed set against the schema first: extra="forbid"
    # rejects unknown keys, field types/bounds reject bad values -- nothing
    # persists unless the update is valid.
    try:
        SettingsSchema(**updates)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    for key, value in updates.items():
        try:
            await db.set_setting(conn, key, value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return await _snapshot(conn)


@config_router.delete(
    "/{key}",
    summary="Reset Config Key",
    description="Delete a stored override so the setting falls back to its default "
                "(or env, if set). 404 for an unknown setting key.",
)
async def reset_config(key: str, request: Request):
    if key not in SettingsSchema.model_fields:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")
    conn = _get_db(request)
    await db.delete_setting(conn, key)
    return await _snapshot(conn)
