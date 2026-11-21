"""Browser session login and logout routes (/login, /logout).

API clients authenticate directly on every request with ``x-api-key`` or
``Authorization: Bearer`` and never touch these endpoints. Browser users enter
the server API key once on ``GET /login``; ``POST /login`` verifies the key in
constant time and sets an ``HttpOnly; SameSite=Strict`` cookie holding a
random session token (``Secure`` over HTTPS, host-only, working identically on
``http://127.0.0.1``). ``POST /logout`` revokes that session token server-side
and clears the cookie.
"""

from __future__ import annotations

import urllib.parse

import orjson
from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse, Response

from heylook_llm import observability
from heylook_llm.auth import (
    check_cookie_csrf,
    extract_session_cookies,
    format_clear_session_cookie,
    format_peer,
    format_session_cookie,
    get_session_store,
    is_https_scope,
    log_auth_failure,
    render_login_html,
    resolve_access_key,
    verify_api_key,
)

auth_router = APIRouter(tags=["Auth"])

_MAX_LOGIN_BODY_BYTES = 8192


def _prefers_json(request: Request) -> bool:
    ctype = (request.headers.get("content-type") or "").lower()
    accept = (request.headers.get("accept") or "").lower()
    return "application/json" in ctype or "application/json" in accept


@auth_router.api_route("/login", methods=["GET", "HEAD"], include_in_schema=False)
def login_page(request: Request) -> Response:
    state = resolve_access_key()
    if not state.enabled:
        return RedirectResponse(url="/", status_code=303, headers={"Cache-Control": "no-store"})
    if state.error or not state.key:
        body = render_login_html(f"Server access key is misconfigured: {state.error}")
        return Response(
            content=b"" if request.method == "HEAD" else body,
            status_code=503,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    store = get_session_store()
    if any(store.verify(tok, state.key) for tok in extract_session_cookies(request.scope)):
        return RedirectResponse(url="/", status_code=303, headers={"Cache-Control": "no-store"})

    body = render_login_html()
    return Response(
        content=b"" if request.method == "HEAD" else body,
        status_code=200,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@auth_router.post("/login", include_in_schema=False)
async def login_submit(request: Request) -> Response:
    csrf_ok, csrf_reason = check_cookie_csrf(request.scope)
    if not csrf_ok:
        log_auth_failure(request.scope, reason=f"cross_origin_login:{csrf_reason}")
        if _prefers_json(request):
            return Response(
                content=orjson.dumps({
                    "detail": "Cross-origin login request refused (Origin must match Host).",
                    "type": "error",
                    "error": {
                        "type": "forbidden",
                        "code": "csrf_rejected",
                        "message": "Cross-origin login request refused (Origin must match Host).",
                    },
                }),
                status_code=403,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=render_login_html("Cross-origin login request refused."),
            status_code=403,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    state = resolve_access_key()
    if not state.enabled:
        if _prefers_json(request):
            return Response(
                content=orjson.dumps({"status": "ok", "gate": "off"}),
                status_code=200,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        return RedirectResponse(url="/", status_code=303, headers={"Cache-Control": "no-store"})

    if state.error or not state.key:
        msg = f"Server access key is misconfigured: {state.error}"
        if _prefers_json(request):
            return Response(
                content=orjson.dumps({"detail": msg}),
                status_code=503,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=render_login_html(msg),
            status_code=503,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    # Read bounded body so unauthenticated callers cannot stream unbounded bytes.
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_LOGIN_BODY_BYTES:
            log_auth_failure(request.scope, reason="login_body_too_large")
            return Response(
                content=orjson.dumps({"detail": "Login payload too large."}),
                status_code=413,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        chunks.append(chunk)
    raw_body = b"".join(chunks)

    ctype = (request.headers.get("content-type") or "").lower()
    submitted = ""
    if "application/json" in ctype:
        try:
            parsed = orjson.loads(raw_body) if raw_body else {}
        except Exception:
            parsed = {}
        if isinstance(parsed, dict) and isinstance(parsed.get("api_key"), str):
            submitted = parsed["api_key"]
    else:
        form = urllib.parse.parse_qs(
            raw_body.decode("utf-8", errors="replace"),
            keep_blank_values=True,
        )
        values = form.get("api_key") or []
        submitted = values[0] if values else ""

    if not verify_api_key(submitted.strip(), state.key):
        log_auth_failure(request.scope, reason="invalid_login_key")
        if _prefers_json(request):
            return Response(
                content=orjson.dumps({
                    "detail": "Invalid API key.",
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                        "message": "Invalid API key.",
                    },
                }),
                status_code=401,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=render_login_html("Invalid API key."),
            status_code=401,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    session_token = get_session_store().create(state.key)
    secure = is_https_scope(request.scope)
    cookie_header = format_session_cookie(session_token, secure=secure).decode("latin-1")
    observability.record_event(
        "auth_login",
        tier="events",
        min_level="standard",
        fields={"peer": format_peer(request.scope)},
    )

    if _prefers_json(request):
        return Response(
            content=orjson.dumps({"status": "ok"}),
            status_code=200,
            media_type="application/json",
            headers={
                "Set-Cookie": cookie_header,
                "Cache-Control": "no-store",
            },
        )

    return RedirectResponse(
        url="/",
        status_code=303,
        headers={
            "Set-Cookie": cookie_header,
            "Cache-Control": "no-store",
        },
    )


@auth_router.post("/logout", include_in_schema=False)
def logout_submit(request: Request) -> Response:
    store = get_session_store()
    for tok in extract_session_cookies(request.scope):
        store.revoke(tok)

    secure = is_https_scope(request.scope)
    clear_cookie = format_clear_session_cookie(secure=secure).decode("latin-1")
    observability.record_event(
        "auth_logout",
        tier="events",
        min_level="standard",
        fields={"peer": format_peer(request.scope)},
    )

    if _prefers_json(request) or request.headers.get("x-request-id"):
        return Response(
            content=orjson.dumps({"status": "logged_out"}),
            status_code=200,
            media_type="application/json",
            headers={
                "Set-Cookie": clear_cookie,
                "Cache-Control": "no-store",
            },
        )

    return RedirectResponse(
        url="/login",
        status_code=303,
        headers={
            "Set-Cookie": clear_cookie,
            "Cache-Control": "no-store",
        },
    )
