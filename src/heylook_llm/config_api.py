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
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import ValidationError

from heylook_llm import db, observability
from heylook_llm.db import get_db as _get_db
from heylook_llm.settings import SettingsSchema, resolve_settings_safe

logger = logging.getLogger(__name__)

config_router = APIRouter(prefix="/v1/admin/config", tags=["Config"])


def observability_log_dir() -> Path:
    """Where the JSONL telemetry streams live (bootstrap config -- env or default)."""
    return Path(os.environ.get("HEYLOOK_LOGS_DIR", "logs"))


async def apply_observability_settings(conn) -> SettingsSchema:
    """Resolve effective settings and push obs level + log dir into the spine cache.

    Called at startup and after every settings change so the (sync, hot-path)
    ``record_event`` cache stays current without a DB hit. Never raises -- a bad
    env/DB value falls back to defaults + a warning.
    """
    stored = await db.get_all_settings(conn)
    settings, err = resolve_settings_safe(stored)
    if err:
        logger.warning("Observability settings invalid, using defaults: %s", err)
    observability.configure(
        level=settings.observability_level,
        log_dir=observability_log_dir(),
        retention_days=settings.observability_retention_days,
    )
    return settings


async def _snapshot(conn) -> dict:
    """Effective settings + what's stored + what env is forcing (precedence made visible)."""
    stored = await db.get_all_settings(conn)
    effective, err = resolve_settings_safe(stored)
    snap = {
        "effective": effective.model_dump(),   # DB > default -- what's actually in force
        "stored": stored,                       # only explicitly-set DB values
    }
    if err:
        snap["error"] = err                     # surface an invalid stored value, don't 500
    return snap


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
        validated = SettingsSchema(**updates)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    # Persist the Pydantic-COERCED value (validated.<key>), not the raw request
    # value -- so `stored` holds a typed value (e.g. 30, not "30") that matches
    # `effective` instead of relying on re-coercion on every read.
    for key in updates:
        try:
            await db.set_setting(conn, key, getattr(validated, key))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    # Refresh the in-process spine cache so an observability level/retention
    # change takes effect immediately (no restart, no per-event DB hit).
    await apply_observability_settings(conn)
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
