# src/heylook_llm/config_api.py
"""Operational settings admin API (/v1/admin/config).

CRUD over runtime-mutable operational settings (obs level/retention, MLX
cache cap), stored in the server config file's ``[settings]`` table
(heylook.toml, owner decision 2026-09-23: server settings live in one file)
and resolved file > default (settings.py -- deliberately NO env-var override
layer). Writes go through the same comment-preserving writer as every other
config write. A server started read-only (HEYLOOK_READONLY_MODEL_CONFIG: a
dev server, E2E, a loop run) shares that file with the daily server, so its
writes are held in memory for its own lifetime and never written. Distinct
from per-model settings (each model's model.heylook.toml) and from user
presets (``/v1/presets``). Wire contract: docs/frontend_v3_spec.md §4.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import ValidationError

from heylook_llm.auth import require_admin_token

from heylook_llm import observability
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


# A read-only instance's settings writes, held for its own lifetime (option
# (a), 2026-09-24): its logging level must still be settable, and the file is
# the daily server's.
_MEMORY_ONLY: dict = {}


def _service(app):
    from heylook_llm.model_service import ModelService

    return ModelService(app.state.router_instance.config_path)


def stored_settings(app) -> dict:
    """The explicitly set settings: the config file's [settings] table, plus
    this instance's memory-only writes when it is read-only. {} when there is
    no router or no file (defaults apply)."""
    stored: dict = {}
    if getattr(app.state, "router_instance", None) is not None:
        try:
            stored = dict(_service(app)._read_toml().get("settings") or {})
        except Exception as e:  # noqa: BLE001 - settings fall back to defaults
            logger.warning("Settings not read from the config file: %s", e)
    return {**stored, **_MEMORY_ONLY}


def apply_runtime_settings(app) -> SettingsSchema:
    """Resolve effective settings and push them into the in-process consumers:
    the observability spine cache (level/log dir/retention) and the MLX
    buffer-cache cap.

    Called at startup and after every settings change so the (sync, hot-path)
    ``record_event`` cache stays current without a file read. Never raises --
    an invalid stored value falls back to defaults + a warning.
    """
    settings, err = resolve_settings_safe(stored_settings(app))
    if err:
        logger.warning("Stored settings invalid, using defaults: %s", err)
    observability.configure(
        level=settings.observability_level,
        log_dir=observability_log_dir(),
        retention_days=settings.observability_retention_days,
    )
    _apply_mlx_cache_limit(settings.mlx_cache_limit_gb)
    return settings


def _snapshot(app) -> dict:
    """Effective settings (stored > default) + the raw stored values."""
    stored = stored_settings(app)
    effective, err = resolve_settings_safe(stored)
    snap = {
        "effective": effective.model_dump(),   # stored > default -- what's in force
        "stored": stored,                       # only explicitly set values
    }
    if _MEMORY_ONLY:
        snap["memory_only"] = sorted(_MEMORY_ONLY)  # held by this read-only instance
    if err:
        snap["error"] = err                     # surface an invalid stored value, don't 500
    return snap


def _write(app, changes: dict) -> None:
    """Apply ``{key: value | None}`` (None = reset) to the stored settings."""
    from heylook_llm.model_registry import readonly

    if readonly():
        for key, value in changes.items():
            if value is None:
                _MEMORY_ONLY.pop(key, None)
            else:
                _MEMORY_ONLY[key] = value
        return
    service = _service(app)
    with service._lock:
        data = service._read_toml()
        settings = dict(data.get("settings") or {})
        for key, value in changes.items():
            if value is None:
                settings.pop(key, None)
            else:
                settings[key] = value
        if settings:
            data["settings"] = settings
        else:
            data.pop("settings", None)
        service._write_toml(data)


@config_router.get(
    "",
    summary="Get Config",
    description="Effective operational settings (heylook.toml [settings] > default; operational settings "
                "have no env-var override layer by design) and the raw stored overrides.",
)
def get_config(request: Request):
    return _snapshot(request.app)


@config_router.put(
    "",
    summary="Update Config",
    description="Set one or more operational settings. Body is a {key: value} map; "
                "unknown keys and invalid values are rejected (422) before anything "
                "persists. Returns the new effective config.",
)
def update_config(request: Request, updates: dict = Body(...)):
    # Validate the whole proposed set against the schema first: extra="forbid"
    # rejects unknown keys, field types/bounds reject bad values -- nothing
    # persists unless the update is valid. Store the COERCED value
    # (validated.<key>), so `stored` matches `effective`. A null clears.
    try:
        validated = SettingsSchema(**updates)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    try:
        _write(request.app, {key: getattr(validated, key) for key in updates})
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Refresh the in-process consumers so a settings change takes effect
    # immediately (no restart, no per-event file read).
    prev_level = observability.current_level()
    apply_runtime_settings(request.app)
    _maybe_emit_startup_record(request, prev_level)
    return _snapshot(request.app)


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
def reset_config(key: str, request: Request):
    if key not in SettingsSchema.model_fields:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")
    _write(request.app, {key: None})
    # Re-apply like PUT does -- otherwise the reset only takes effect after a
    # restart while GET already reports the default as effective.
    prev_level = observability.current_level()
    apply_runtime_settings(request.app)
    _maybe_emit_startup_record(request, prev_level)
    return _snapshot(request.app)
