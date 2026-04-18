"""Optional admin-endpoint authentication (S1.6).

Opt-in via ``HEYLOOK_ADMIN_TOKEN`` env var. When unset (or set to an empty
string), the middleware is a no-op and admin endpoints are unauthenticated --
the default for the localhost-only single-user deployment. When set to a
non-empty value, admin endpoints require an ``X-Heylook-Admin-Token`` request
header matching it; otherwise they return 401.

Applied as a FastAPI dependency on the three admin routers (admin_router,
scan_import_router, admin_ops_router) and on the two mutating loose endpoints
(`/v1/data/clear`, `/v1/cache/clear`). NOT applied to inference endpoints --
those are the traffic all the client apps rely on and a token on them would
break every request simultaneously.

Design rationale: the server's default LAN exposure (`--host 0.0.0.0` so the
Ubuntu+4090 box can reach it) makes the home network the implicit trust
boundary. That's fine for known clients on the user's own LAN; the token
adds a cheap defense-in-depth layer for any non-routine LAN exposure without
breaking the default UX.
"""

from __future__ import annotations

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

    provided = request.headers.get(_ADMIN_TOKEN_HEADER)
    if provided != expected:
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
