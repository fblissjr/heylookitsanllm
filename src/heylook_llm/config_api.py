# src/heylook_llm/config_api.py
"""Operational settings admin API (/v1/admin/config).

CRUD over runtime-mutable operational settings (obs level/retention, MLX
cache cap, ...), persisted in the App DB ``settings`` table (db.py) and
resolved DB > default (settings.py -- deliberately NO env-var override
layer for operational settings). Distinct from the model registry
(``models.toml``, Phase 6) and from user presets (``/v1/presets``). This is the
backend for the v3 admin/settings config panel. Wire contract:
docs/frontend_v3_spec.md §4.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import ValidationError

from heylook_llm.auth import require_admin_token

from heylook_llm import db, observability
from heylook_llm.db import get_db as _get_db
from heylook_llm.settings import SettingsSchema, resolve_settings_safe

logger = logging.getLogger(__name__)

# Gated like every other /v1/admin/* router: HEYLOOK_ADMIN_TOKEN, when set,
# is a no-op when unset. This was the one admin router without the dependency.
config_router = APIRouter(
    prefix="/v1/admin/config",
    tags=["Config"],
    dependencies=[Depends(require_admin_token)],
)


def observability_log_dir() -> Path:
    """Where the JSONL telemetry streams live (bootstrap config -- env or default)."""
    return Path(os.environ.get("HEYLOOK_LOGS_DIR", "logs"))


# Captured on the first cap so clearing the override can restore MLX's own
# default (mx.set_cache_limit returns the previous limit). None = never capped.
_mlx_default_cache_limit: int | None = None


def _apply_mlx_cache_limit(gb: float | None) -> None:
    """Best-effort apply of the MLX buffer-cache cap. Never raises."""
    global _mlx_default_cache_limit
    try:
        import mlx.core as mx
        if gb is None:
            if _mlx_default_cache_limit is not None:
                mx.set_cache_limit(_mlx_default_cache_limit)
                _mlx_default_cache_limit = None
            return
        prev = mx.set_cache_limit(int(gb * 1024**3))
        if _mlx_default_cache_limit is None:
            _mlx_default_cache_limit = prev
    except Exception as e:
        logger.warning("MLX cache limit not applied: %s", e)


async def apply_runtime_settings(conn) -> SettingsSchema:
    """Resolve effective settings and push them into the in-process consumers:
    the observability spine cache (level/log dir/retention) and the MLX
    buffer-cache cap.

    Called at startup and after every settings change so the (sync, hot-path)
    ``record_event`` cache stays current without a DB hit. Never raises -- a bad
    env/DB value falls back to defaults + a warning.
    """
    stored = await db.get_all_settings(conn)
    settings, err = resolve_settings_safe(stored)
    if err:
        logger.warning("Stored settings invalid, using defaults: %s", err)
    observability.configure(
        level=settings.observability_level,
        log_dir=observability_log_dir(),
        retention_days=settings.observability_retention_days,
    )
    _apply_mlx_cache_limit(settings.mlx_cache_limit_gb)
    return settings


async def _snapshot(conn) -> dict:
    """Effective settings (DB > default) + the raw stored overrides."""
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
    description="Effective operational settings (DB > default; operational settings "
                "have no env-var override layer by design) and the raw stored overrides.",
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
    # Refresh the in-process consumers so a settings change takes effect
    # immediately (no restart, no per-event DB hit).
    prev_level = observability.current_level()
    await apply_runtime_settings(conn)
    _maybe_emit_startup_record(request, prev_level)
    return await _snapshot(conn)


def _maybe_emit_startup_record(request: Request, prev_level: str) -> None:
    """Emit the one-shot startup record when telemetry flips off -> on mid-run.

    log_startup_info honors the kill switch, so a server booted at ``off``
    wrote nothing; without this, streams enabled via this API would carry no
    hardware/config header for the very sessions logging was turned on for.
    """
    if prev_level == "off" and observability.current_level() != "off":
        from heylook_llm.memory import safe_mm_call
        safe_mm_call(
            getattr(request.app.state, "memory_manager", None), "log_startup_info"
        )


@config_router.delete(
    "/{key}",
    summary="Reset Config Key",
    description="Delete a stored override so the setting falls back to its built-in "
                "default. 404 for an unknown setting key.",
)
async def reset_config(key: str, request: Request):
    if key not in SettingsSchema.model_fields:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")
    conn = _get_db(request)
    await db.delete_setting(conn, key)
    # Re-apply like PUT does -- otherwise the reset only takes effect after a
    # restart while GET already reports the default as effective.
    prev_level = observability.current_level()
    await apply_runtime_settings(conn)
    _maybe_emit_startup_record(request, prev_level)
    return await _snapshot(conn)
