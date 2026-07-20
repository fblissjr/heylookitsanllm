# src/heylook_llm/settings.py
"""Operational settings: schema, defaults, and DB > default resolution.

Operational settings are runtime-mutable config (obs level/retention, ...) edited
via ``/v1/admin/config`` and persisted in the App DB ``settings`` table (db.py).
This module is the *contract*: it declares which settings exist, their types and
defaults, validates writes, and resolves an effective value.

**Single source of truth: the DB (or the built-in default).** There is
deliberately NO env-var override layer for operational settings -- an env var
silently overriding a value you set in the admin UI is a footgun (you edit it,
nothing changes). Env vars are reserved for *bootstrap* concerns that have no UI
counterpart and thus can't conflict: ``HEYLOOK_LOGS_DIR`` (where telemetry is
written), ``HEYLOOK_DB_PATH`` (where the store lives).

NOTE: this is the config *mechanism*. Its first real consumer is the observability
spine (reads ``observability_level`` / ``observability_retention_days`` via
``resolve_settings``). New settings are added as fields here -- key->value rows in
the DB are schema-stable, so adding one is never a DDL change.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SettingsSchema(BaseModel):
    """The full operational-settings surface -- types, defaults, validation.

    ``extra="forbid"`` so an unknown key from the frontend or a stale DB row is
    rejected at validation (same policy as ``models.toml`` typos in config.py).
    Every field has a default: the DB stores only what's explicitly set, and a
    missing setting falls back here.
    """

    model_config = ConfigDict(extra="forbid")

    observability_level: Literal["off", "minimal", "standard", "debug"] = "minimal"
    observability_retention_days: int = Field(default=30, ge=0)
    # Cap on MLX's buffer cache (GB). The allocator keeps freed buffers for
    # reuse and never returns them to the OS, so server RSS pins at the
    # prompt-spike high-water mark; a cap bounds idle RSS at the cost of
    # re-allocation on the next spike. None = MLX's own default (uncapped in
    # practice). Matters when other memory-hungry jobs share the box.
    mlx_cache_limit_gb: float | None = Field(default=None, gt=0)


def resolve_settings(db_values: Mapping[str, Any]) -> SettingsSchema:
    """Resolve effective settings: DB-stored value > built-in default.

    ``db_values`` is the raw ``{key: value}`` map from the store; unknown keys
    (a setting since removed from the schema) are dropped, not errored, so a
    stale DB can't break resolution. Raises ``pydantic.ValidationError`` on an
    invalid stored value (fail loud, like the strict TOML validation).
    """
    fields = SettingsSchema.model_fields
    merged: dict[str, Any] = {k: v for k, v in db_values.items() if k in fields}
    return SettingsSchema(**merged)


def resolve_settings_safe(db_values: Mapping[str, Any]) -> tuple[SettingsSchema, str | None]:
    """Like ``resolve_settings`` but NEVER raises.

    On an invalid stored value returns all-defaults + a short human-readable
    error string (for logging + surfacing in the API). Use at startup and in
    read paths where a bad DB value must not crash the server.
    """
    try:
        return resolve_settings(db_values), None
    except ValidationError as e:
        summary = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return SettingsSchema(), summary
