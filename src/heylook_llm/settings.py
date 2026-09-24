# src/heylook_llm/settings.py
"""Operational settings: schema, defaults, and file > default resolution.

Operational settings are runtime-mutable config (obs level/retention, ...) edited
via ``/v1/admin/config`` and stored in heylook.toml's ``[settings]`` table
(config_api; a read-only instance holds them in memory).
This module is the *contract*: it declares which settings exist, their types and
defaults, validates writes, and resolves an effective value.

**Single source of truth: the config file (or the built-in default).** There is
deliberately NO env-var override layer for operational settings -- an env var
silently overriding a value you set in the admin UI is a footgun (you edit it,
nothing changes). Env vars are reserved for *bootstrap* concerns that have no UI
counterpart and thus can't conflict: ``HEYLOOK_LOGS_DIR`` (where telemetry is
written), ``HEYLOOK_DB_PATH`` (where the store lives).

NOTE: this is the config *mechanism*. Its first real consumer is the observability
spine (reads ``observability_level`` / ``observability_retention_days`` via
``resolve_settings``). New settings are added as fields here -- keys in the
[settings] table need no schema change.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SettingsSchema(BaseModel):
    """The full operational-settings surface -- types, defaults, validation.

    ``extra="forbid"`` so an unknown key from the frontend or a stale file key is
    rejected at validation (same policy as ``heylook.toml`` typos in config.py).
    Every field has a default: the DB stores only what's explicitly set, and a
    missing setting falls back here.
    """

    model_config = ConfigDict(extra="forbid")

    # File logging is OPT-IN (owner rule 2026-08-13): the default writes no
    # files under logs/ at all. Raise the level via /v1/admin/config (or the
    # admin UI) when you want telemetry/diagnostics captured.
    observability_level: Literal["off", "minimal", "standard", "debug"] = "off"
    observability_retention_days: int = Field(default=30, ge=0)
    # Cap on MLX's buffer cache (GB) WITHIN a request. The allocator keeps
    # freed buffers for reuse, but the cache is already cleared after every
    # generation (MLXProvider.create_chat_completion's finally) and on every
    # unload, so between requests the MLX process holds its weights and the
    # prefix cache's snapshots (bounded by mlx-vlm's own APC budget), which a
    # cap does not touch. A cap would only bound prefill and decode, at the
    # likely cost of re-allocation there. None = MLX's own default; no default
    # cap (unmeasured, and the idle case it would serve is already handled).
    mlx_cache_limit_gb: float | None = Field(default=None, gt=0)


def resolve_settings(stored: Mapping[str, Any]) -> SettingsSchema:
    """Resolve effective settings: stored value > built-in default.

    ``stored`` is the raw ``{key: value}`` map (the file's [settings]); unknown keys
    (a setting since removed from the schema) are dropped, not errored, so a
    stale file can't break resolution. Raises ``pydantic.ValidationError`` on an
    invalid stored value (fail loud, like the strict TOML validation).
    """
    fields = SettingsSchema.model_fields
    merged: dict[str, Any] = {k: v for k, v in stored.items() if k in fields}
    return SettingsSchema(**merged)


def resolve_settings_safe(stored: Mapping[str, Any]) -> tuple[SettingsSchema, str | None]:
    """Like ``resolve_settings`` but NEVER raises.

    On an invalid stored value returns all-defaults + a short human-readable
    error string (for logging + surfacing in the API). Use at startup and in
    read paths where a bad DB value must not crash the server.
    """
    try:
        return resolve_settings(stored), None
    except ValidationError as e:
        summary = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return SettingsSchema(), summary
