"""Optional admin authentication.

``HEYLOOK_ADMIN_TOKEN`` (header ``X-Heylook-Admin-Token``) gates the admin
routers plus ``/v1/data/clear`` and ``/v1/cache/clear``. It is a no-op when
the env var is unset or empty -- the default single-user deployment stays
open. Token comparison uses ``hmac.compare_digest`` so a wrong-length guess
and a close-match guess take the same time.

There is no inference API key. ``HEYLOOK_API_KEY`` gated only some inference
routes (not the conversation, notebook, preset or generate routers), which
looked like protection without being it, and the owner does not set it; it
was removed in v2.0.127 (see docs/project/TODO.md for what a real gate would
need).

Design rationale: the server's default LAN exposure (``--host 0.0.0.0`` so
another LAN machine can reach it) makes the home network the trust
boundary.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request


_ADMIN_TOKEN_ENV = "HEYLOOK_ADMIN_TOKEN"
_ADMIN_TOKEN_HEADER = "X-Heylook-Admin-Token"


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
