# src/heylook_llm/settings.py
"""Operational settings: schema, defaults, and env > DB > default resolution.

Operational settings are runtime-mutable config (obs level/retention, ...) edited
via ``/v1/admin/config`` and persisted in the App DB ``settings`` table (db.py).
This module is the *contract*: it declares which settings exist, their types and
defaults, validates writes, and resolves an effective value.

Precedence (highest wins): env override > DB-stored value > built-in default.
Env is the always-wins escape hatch, matching the existing ``HEYLOOK_*`` posture.

NOTE: this is the config *mechanism*. Its first real consumer is the Phase 1
observability spine (reads ``observability_level`` / ``observability_retention_days``
via ``resolve_settings``). New settings are added as fields here -- key->value
rows in the DB are schema-stable, so adding one is never a DDL change.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_ENV_PREFIX = "HEYLOOK_"


class SettingsSchema(BaseModel):
    """The full operational-settings surface -- types, defaults, validation.

    ``extra="forbid"`` so an unknown key from the frontend or a stale DB row is
    rejected at validation (same policy as ``models.toml`` typos in config.py).
    Every field has a default: the DB stores only what's explicitly set, and a
    missing setting falls back here.
    """

    model_config = ConfigDict(extra="forbid")

    observability_level: Literal["off", "minimal", "standard", "debug"] = "minimal"
    observability_retention_days: int = Field(30, ge=0)


def setting_env_key(field_name: str) -> str:
    """Env var name that overrides ``field_name`` (e.g. ``HEYLOOK_OBSERVABILITY_LEVEL``)."""
    return _ENV_PREFIX + field_name.upper()


def resolve_settings(
    db_values: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> SettingsSchema:
    """Resolve effective settings: env override > DB value > default.

    ``db_values`` is the raw ``{key: value}`` map from the store; unknown keys
    (a setting since removed from the schema) are dropped, not errored, so a
    stale DB can't break resolution. Env values arrive as strings and are coerced
    by Pydantic. Raises ``pydantic.ValidationError`` on an invalid value from
    either source (fail loud, like the strict TOML validation).
    """
    if env is None:
        env = os.environ
    fields = SettingsSchema.model_fields
    merged: dict[str, Any] = {k: v for k, v in db_values.items() if k in fields}
    for name in fields:
        key = setting_env_key(name)
        if key in env:
            merged[name] = env[key]  # env wins; Pydantic coerces the string
    return SettingsSchema(**merged)


def resolve_settings_safe(
    db_values: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> tuple[SettingsSchema, str | None]:
    """Like ``resolve_settings`` but NEVER raises.

    On an invalid env/DB value returns all-defaults + a short human-readable
    error string (for logging + surfacing in the API). Use at startup and in
    read paths where a typo in a ``HEYLOOK_*`` var must not crash the server.
    """
    try:
        return resolve_settings(db_values, env), None
    except ValidationError as e:
        summary = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return SettingsSchema(), summary
