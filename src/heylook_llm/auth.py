"""Optional admin + inference authentication.

Two independent gates, both opt-in via env vars:

- ``HEYLOOK_ADMIN_TOKEN`` (header ``X-Heylook-Admin-Token``): gates the
  three admin routers plus ``/v1/data/clear`` and ``/v1/cache/clear``.
  S1.6.
- ``HEYLOOK_API_KEY`` (header ``Authorization: Bearer <value>``): gates
  inference endpoints (chat completions, messages, embeddings, RLM,
  hidden-states). C1.5. Loopback traffic is exempt by default; set
  ``HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true`` to close the carve-out.

Both dependencies are no-ops when their env var is unset or empty -- the
default single-user localhost deployment stays open. Token comparison uses
``hmac.compare_digest`` so a wrong-length guess and a close-match guess
take the same time.

Design rationale: the server's default LAN exposure (``--host 0.0.0.0`` so
the Ubuntu+4090 box can reach it) makes the home network the implicit
trust boundary. The admin-token + api-key pair lets users tighten that
boundary without breaking the default UX when they don't need to.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request


_ADMIN_TOKEN_ENV = "HEYLOOK_ADMIN_TOKEN"
_ADMIN_TOKEN_HEADER = "X-Heylook-Admin-Token"

_API_KEY_ENV = "HEYLOOK_API_KEY"
_API_KEY_ENFORCE_LOOPBACK_ENV = "HEYLOOK_API_KEY_ENFORCE_LOOPBACK"
_AUTHORIZATION_HEADER = "Authorization"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_TRUTHY = frozenset({"true", "1", "yes", "on"})


def require_admin_token(request: Request) -> None:
    """FastAPI dependency: gate admin routes behind HEYLOOK_ADMIN_TOKEN.

    Raises ``HTTPException(401)`` when the env var is set to a non-empty
    value AND the incoming request's ``X-Heylook-Admin-Token`` header
    doesn't match. No-op when the env var is unset or empty.
    """
    expected = os.environ.get(_ADMIN_TOKEN_ENV, "").strip()
    if not expected:
        return None

    provided = request.headers.get(_ADMIN_TOKEN_HEADER) or ""
    if not hmac.compare_digest(provided, expected):
        logging.warning(
            "admin-token mismatch on %s; set HEYLOOK_ADMIN_TOKEN or send "
            "the X-Heylook-Admin-Token header",
            request.url.path if hasattr(request, "url") else "<request>",
        )
        raise HTTPException(
            status_code=401,
            detail=f"Admin endpoint requires {_ADMIN_TOKEN_HEADER} header.",
        )
    return None


def _is_loopback(request: Request) -> bool:
    """Return True only when we can positively identify the request as
    loopback. An absent ``request.client`` means we cannot tell, and the
    safe default is to treat it as non-loopback (fail closed).
    """
    client = getattr(request, "client", None)
    if client is None:
        return False
    host = getattr(client, "host", None)
    return host in _LOOPBACK_HOSTS


def require_api_key(request: Request) -> None:
    """FastAPI dependency: gate inference routes behind HEYLOOK_API_KEY.

    Raises ``HTTPException(401)`` when the env var is set AND the request
    fails the check. The request passes when:

    1. The env var is unset or empty.
    2. The client is loopback AND ``HEYLOOK_API_KEY_ENFORCE_LOOPBACK`` is
       not truthy (default carve-out -- local dev tools don't need to
       carry the key).
    3. The ``Authorization: Bearer <value>`` header matches the env value
       under constant-time comparison. The ``Bearer`` scheme token is
       matched case-insensitively per RFC 6750.
    """
    expected = os.environ.get(_API_KEY_ENV, "").strip()
    if not expected:
        return None

    enforce_loopback = (
        os.environ.get(_API_KEY_ENFORCE_LOOPBACK_ENV, "").strip().lower()
        in _TRUTHY
    )
    if _is_loopback(request) and not enforce_loopback:
        return None

    provided = request.headers.get(_AUTHORIZATION_HEADER) or ""
    scheme, _, token = provided.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        logging.warning(
            "api-key mismatch on %s from %s; set HEYLOOK_API_KEY or send "
            "'Authorization: Bearer <key>'",
            request.url.path if hasattr(request, "url") else "<request>",
            getattr(getattr(request, "client", None), "host", "<unknown>"),
        )
        raise HTTPException(
            status_code=401,
            detail="Inference endpoint requires 'Authorization: Bearer <key>' header.",
        )
    return None
