"""Server-wide access gate, always-on browser origin/framing guards, and optional extra admin auth.

One pure ASGI middleware (``AccessGateMiddleware``), installed beside
``HostCheckMiddleware``, covers every HTTP/WebSocket route, the static
frontend, streaming responses, and any router added later:

Always-on (independent of whether an API key is configured):
  - ``Content-Security-Policy: frame-ancestors 'self'`` (plus
    ``X-Frame-Options: SAMEORIGIN``) on every HTTP response.
  - State-changing requests (``POST``, ``PUT``, ``PATCH``, ``DELETE``, and
    WebSockets) are refused with ``403`` whenever ``Sec-Fetch-Site`` is not
    ``same-origin`` or ``none`` (blocking ``cross-site`` and sibling-subdomain
    ``same-site``) or ``Origin`` does not match this server's ``Host`` (or an
    ``X-Forwarded-Host`` explicitly listed in ``own_names`` / ``allowed_hosts``
    for ``tailscale serve``). Non-browser clients omit both headers and pass.
    This protects loopback and ``tailscale serve`` endpoints from cross-tab
    browser requests without needing ``HEYLOOK_ADMIN_TOKEN`` (which the
    frontend never sends).

When a key is configured in:
  1. ``HEYLOOK_API_KEY`` (env var), or
  2. a mode-600 file at ``~/.heylook/api_key`` (or ``$HEYLOOK_API_KEY_FILE``),
the authentication gate turns on across every route (never read from or written
to ``heylook.toml``, which ``/v1/admin/config`` reads and writes). All key
comparisons run in constant time via ``hmac.compare_digest`` on UTF-8 bytes.

Accepted credentials (any one suffices):
  - ``x-api-key: <key>`` (Anthropic SDK ``api_key`` default)
  - ``Authorization: Bearer <key>`` (OpenAI-style ``/v1/models`` discovery and
    Anthropic SDK ``auth_token``)
  - ``heylook_session`` cookie (HttpOnly, SameSite=Strict, Secure over HTTPS,
    host-only, storing a random 256-bit session token whose SHA-256 digest is
    held in ``SessionStore`` -- never the API key itself). State-changing
    cookie requests additionally require at least one of ``Origin`` or
    ``Sec-Fetch-Site`` to be present.

There are NO source-IP exemptions (loopback and LAN proxies forward external
clients from ``127.0.0.1`` and LAN addresses). The ONLY unauthenticated routes
when the key is set are ``EXEMPT_ROUTES``: ``/login``, the four static assets
it loads, and ``/health`` (which returns only ``{"status": "ok"}``).

``HEYLOOK_ADMIN_TOKEN`` (header ``X-Heylook-Admin-Token``) stays as an optional
extra factor on admin routes plus ``/v1/data/clear`` and ``/v1/cache/clear``
(note: the web UI never sends ``X-Heylook-Admin-Token``).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import hmac
import html
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from typing import Any, Literal
import urllib.parse

from fastapi import HTTPException, Request

from heylook_llm import observability
from heylook_llm.host_check import _configured as _configured_hosts, own_names

logger = logging.getLogger(__name__)

API_KEY_ENV = "HEYLOOK_API_KEY"
API_KEY_FILE_ENV = "HEYLOOK_API_KEY_FILE"
DEFAULT_API_KEY_PATH = Path.home() / ".heylook" / "api_key"

SESSION_COOKIE_NAME = "heylook_session"
SESSION_TTL_SECONDS = 30 * 86400  # 30 days
MAX_ACTIVE_SESSIONS = 128

_ADMIN_TOKEN_ENV = "HEYLOOK_ADMIN_TOKEN"
_ADMIN_TOKEN_HEADER = "X-Heylook-Admin-Token"

_SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

# The ONLY routes reachable without credentials when the gate is on.
# Do not add routes here lightly: tests/contract/test_access_gate.py walks
# app.routes and asserts every other route answers 401 without credentials.
EXEMPT_ROUTES: frozenset[tuple[str, str]] = frozenset({
    ("/health", "GET"),
    ("/health", "HEAD"),
    ("/login", "GET"),
    ("/login", "HEAD"),
    ("/login", "POST"),
    ("/css/app.css", "GET"),
    ("/css/app.css", "HEAD"),
    ("/js/login.js", "GET"),
    ("/js/login.js", "HEAD"),
    ("/icon.svg", "GET"),
    ("/icon.svg", "HEAD"),
    ("/favicon.ico", "GET"),
    ("/favicon.ico", "HEAD"),
})

_LOGIN_HTML_PATH = Path(__file__).resolve().parent.parent.parent / "frontend" / "login.html"
_LOGIN_ERROR_PLACEHOLDER = "<!--LOGIN_ERROR-->"

_LOG_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _clean_log_token(value: str, max_len: int = 200) -> str:
    """Strip control characters and bound length before interpolating into logs."""
    cleaned = _LOG_CONTROL_RE.sub("?", value or "")
    return cleaned[:max_len]


def _display_path(path: Path) -> str:
    """Render paths under the user's home directory with a leading ``~`` so
    admin responses and logs never echo OS account names."""
    try:
        rel = path.expanduser().resolve().relative_to(Path.home().resolve())
        return (Path("~") / rel).as_posix()
    except Exception:
        try:
            rel = path.expanduser().relative_to(Path.home())
            return (Path("~") / rel).as_posix()
        except Exception:
            return path.name or "api_key"


@dataclass(frozen=True, slots=True)
class AccessKeyState:
    enabled: bool
    key: str | None
    source: Literal["env", "file", "none"]
    source_detail: str
    error: str | None = None


_file_cache_lock = threading.Lock()
_file_cache_stamp: tuple[str, int, int, int] | None = None
_file_cache_state: AccessKeyState | None = None


def _read_key_file(key_path: Path, *, explicit: bool) -> AccessKeyState:
    """Read a mode-600 key file, failing closed on bad permissions or empty files."""
    global _file_cache_stamp, _file_cache_state

    shown = _display_path(key_path)
    try:
        st = key_path.stat()
    except FileNotFoundError:
        if explicit:
            return AccessKeyState(
                enabled=True,
                key=None,
                source="file",
                source_detail=shown,
                error=f"Configured key file {shown} does not exist.",
            )
        return AccessKeyState(
            enabled=False,
            key=None,
            source="none",
            source_detail="unset",
            error=None,
        )
    except OSError as exc:
        return AccessKeyState(
            enabled=True,
            key=None,
            source="file",
            source_detail=shown,
            error=f"Cannot stat key file {shown}: {exc.strerror or exc}",
        )

    if not key_path.is_file():
        return AccessKeyState(
            enabled=True,
            key=None,
            source="file",
            source_detail=shown,
            error=f"Key path {shown} is not a regular file.",
        )

    mode = st.st_mode & 0o777
    if (mode & 0o077) != 0:
        return AccessKeyState(
            enabled=True,
            key=None,
            source="file",
            source_detail=shown,
            error=(
                f"Key file {shown} has mode {mode:04o} (group/other permissions set); "
                f"run: chmod 600 {shown}"
            ),
        )

    stamp = (str(key_path), st.st_mtime_ns, st.st_size, mode)
    with _file_cache_lock:
        if _file_cache_stamp == stamp and _file_cache_state is not None:
            return _file_cache_state

    try:
        raw = key_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return AccessKeyState(
            enabled=True,
            key=None,
            source="file",
            source_detail=shown,
            error=f"Cannot read key file {shown}: {exc.strerror or exc}",
        )

    if not raw:
        return AccessKeyState(
            enabled=True,
            key=None,
            source="file",
            source_detail=shown,
            error=f"Key file {shown} is empty; write a non-empty key or remove the file.",
        )

    state = AccessKeyState(
        enabled=True,
        key=raw,
        source="file",
        source_detail=shown,
        error=None,
    )
    with _file_cache_lock:
        _file_cache_stamp = stamp
        _file_cache_state = state
    return state


def resolve_access_key() -> AccessKeyState:
    """Resolve the active access key from env or mode-600 file (never heylook.toml)."""
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        return AccessKeyState(
            enabled=True,
            key=env_key,
            source="env",
            source_detail=API_KEY_ENV,
            error=None,
        )

    if API_KEY_FILE_ENV in os.environ:
        raw_path = os.environ[API_KEY_FILE_ENV].strip()
        if not raw_path:
            return AccessKeyState(
                enabled=False,
                key=None,
                source="none",
                source_detail="unset",
                error=None,
            )
        return _read_key_file(Path(raw_path).expanduser(), explicit=True)

    return _read_key_file(DEFAULT_API_KEY_PATH, explicit=False)


def verify_api_key(provided: str, expected: str) -> bool:
    """Constant-time UTF-8 key comparison (safe on arbitrary Unicode input)."""
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _sha256_bytes(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


class SessionStore:
    """In-memory store of random browser session tokens.

    Stores only ``sha256(session_token) -> (expires_at, sha256(api_key))``,
    never the raw API key or raw cookie value. Rotating the API key
    automatically invalidates sessions minted under the previous key.
    """

    def __init__(self, *, ttl_seconds: int = SESSION_TTL_SECONDS, max_sessions: int = MAX_ACTIVE_SESSIONS) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[bytes, tuple[float, bytes]] = OrderedDict()

    def _prune_unlocked(self, now: float) -> None:
        expired = [k for k, (exp, _) in self._sessions.items() if now >= exp]
        for k in expired:
            del self._sessions[k]

    def create(self, expected_key: str, *, now: float | None = None) -> str:
        ts = time.time() if now is None else now
        token = secrets.token_urlsafe(32)
        token_digest = _sha256_bytes(token)
        key_digest = _sha256_bytes(expected_key)
        with self._lock:
            self._prune_unlocked(ts)
            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[token_digest] = (ts + self._ttl_seconds, key_digest)
        return token

    def verify(self, token: str, expected_key: str, *, now: float | None = None) -> bool:
        if not token or not expected_key:
            return False
        ts = time.time() if now is None else now
        token_digest = _sha256_bytes(token)
        key_digest = _sha256_bytes(expected_key)
        with self._lock:
            entry = self._sessions.get(token_digest)
            if entry is None:
                return False
            expires_at, stored_key_digest = entry
            if ts >= expires_at or not hmac.compare_digest(stored_key_digest, key_digest):
                del self._sessions[token_digest]
                return False
            self._sessions.move_to_end(token_digest)
            return True

    def revoke(self, token: str) -> bool:
        if not token:
            return False
        token_digest = _sha256_bytes(token)
        with self._lock:
            return self._sessions.pop(token_digest, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            self._prune_unlocked(time.time())
            return len(self._sessions)


_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _session_store


def _header_first(scope: dict[str, Any], name_lower: bytes) -> str | None:
    for k, v in scope.get("headers") or ():
        if k.lower() == name_lower:
            return v.decode("latin-1")
    return None


def _headers_all(scope: dict[str, Any], name_lower: bytes) -> list[str]:
    return [v.decode("latin-1") for k, v in scope.get("headers") or () if k.lower() == name_lower]


def extract_session_cookies(scope: dict[str, Any]) -> list[str]:
    """Return every ``heylook_session`` cookie value sent on ``scope`` (HTTP/1.1 & HTTP/2)."""
    tokens: list[str] = []
    for raw_cookie in _headers_all(scope, b"cookie"):
        for chunk in raw_cookie.split(";"):
            part = chunk.strip()
            if not part or "=" not in part:
                continue
            cname, _, cval = part.partition("=")
            if cname.strip() == SESSION_COOKIE_NAME:
                val = cval.strip().strip('"')
                if val:
                    tokens.append(val)
    return tokens


_FORWARDED_PROTO_RE = re.compile(r"(?:^|;)\s*proto\s*=\s*\"?([a-zA-Z0-9+.-]+)\"?", re.I)


def is_https_transport(scope: dict[str, Any]) -> bool:
    """True when the request arrived over TLS directly or through an HTTPS reverse proxy."""
    if str(scope.get("scheme") or "").lower() == "https":
        return True
    xfp = _header_first(scope, b"x-forwarded-proto")
    if xfp:
        first_hop = xfp.split(",", 1)[0].strip().lower()
        return first_hop == "https"
    fwd = _header_first(scope, b"forwarded")
    if fwd:
        first_element = fwd.split(",", 1)[0]
        m = _FORWARDED_PROTO_RE.search(first_element)
        if m:
            return m.group(1).lower() == "https"
    return False


def is_https_scope(scope: dict[str, Any]) -> bool:
    """True when the browser session is over HTTPS (transport/proxy headers or https Origin)."""
    if is_https_transport(scope):
        return True
    origin = _header_first(scope, b"origin")
    if origin and origin.strip().lower().startswith("https://"):
        return True
    return False


def format_session_cookie(token: str, *, secure: bool) -> bytes:
    parts = [
        f"{SESSION_COOKIE_NAME}={token}",
        "Path=/",
        f"Max-Age={SESSION_TTL_SECONDS}",
        "HttpOnly",
        "SameSite=Strict",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts).encode("latin-1")


def format_clear_session_cookie(*, secure: bool) -> bytes:
    parts = [
        f"{SESSION_COOKIE_NAME}=",
        "Path=/",
        "Max-Age=0",
        "HttpOnly",
        "SameSite=Strict",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts).encode("latin-1")


def _normalized_authority(host_port: str, scheme: str) -> tuple[str, int | None] | None:
    """Parse ``host[:port]`` into ``(lowercase_host, non_default_port_or_None)``."""
    raw = host_port.strip()
    if not raw or "/" in raw or "@" in raw or "?" in raw or "#" in raw:
        return None
    try:
        parsed = urllib.parse.urlsplit(f"//{raw}")
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not hostname:
        return None
    host_clean = hostname.lower().rstrip(".")
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    return (host_clean, port)


def check_request_origin(
    scope: dict[str, Any],
    *,
    require_browser_signal: bool = False,
    own: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Cross-origin & sibling-site protection for state-changing requests.

    Enforced independently of whether an API key is configured:
      1. ``Sec-Fetch-Site``, when present, MUST be ``same-origin`` or ``none``
         (rejecting ``cross-site`` and sibling-subdomain ``same-site``).
      2. ``Origin``, when present, MUST be an ``http(s)`` origin whose
         ``host:port`` matches ``Host`` (or a recognized ``X-Forwarded-Host``
         in ``own_names`` / ``heylook.toml``'s ``allowed_hosts`` for proxies
         like ``tailscale serve`` that rewrite ``Host`` to loopback), and must
         not downgrade ``https`` to ``http`` over an HTTPS proxy.
      3. When ``require_browser_signal`` is True (cookie-authenticated state
         changes and ``POST /login``), at least one of ``Origin`` or
         ``Sec-Fetch-Site`` must be present. Non-browser clients (``curl``,
         SDKs, scripts) omit both and pass when ``require_browser_signal`` is
         False.
    """
    sfs_raw = _header_first(scope, b"sec-fetch-site")
    sfs = sfs_raw.strip().lower() if sfs_raw is not None else None
    if sfs is not None and sfs not in ("same-origin", "none"):
        return False, f"sec_fetch_site_{_clean_log_token(sfs, 32)}"

    origin_raw = _header_first(scope, b"origin")
    if origin_raw is not None:
        origin = origin_raw.strip()
        try:
            parsed = urllib.parse.urlsplit(origin)
            origin_port = parsed.port
        except ValueError:
            return False, "malformed_origin"
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https") or not parsed.hostname:
            return False, "invalid_origin"
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username:
            return False, "invalid_origin"
        if is_https_transport(scope) and scheme != "https":
            return False, "origin_scheme_downgrade"

        host_raw = _header_first(scope, b"host")
        if not host_raw:
            return False, "missing_host"
        effective_scheme = "https" if is_https_transport(scope) else scheme
        host_auth = _normalized_authority(host_raw, effective_scheme)
        origin_host = parsed.hostname.lower().rstrip(".")
        if (scheme == "http" and origin_port == 80) or (scheme == "https" and origin_port == 443):
            origin_port = None
        origin_auth = (origin_host, origin_port)
        if host_auth is not None and origin_auth == host_auth:
            return True, "ok"

        # Proxies like `tailscale serve` rewrite `Host` to 127.0.0.1:<port> and
        # pass the public hostname in `X-Forwarded-Host`. Accept it ONLY when
        # that hostname is one of this server's recognized non-wildcard names.
        xfh_raw = _header_first(scope, b"x-forwarded-host")
        if xfh_raw:
            xfh_first = xfh_raw.split(",", 1)[0].strip()
            xfh_auth = _normalized_authority(xfh_first, effective_scheme)
            if xfh_auth is not None and origin_auth == xfh_auth:
                known = ((own if own is not None else own_names()) | _configured_hosts(scope)) - {"*", "localhost"}
                if xfh_auth[0] in known:
                    return True, "ok"
                return False, f"unlisted_forwarded_host:{_clean_log_token(xfh_auth[0], 96)}"

        return False, "origin_mismatch"

    if sfs in ("same-origin", "none"):
        return True, "ok"

    if require_browser_signal:
        return False, "missing_origin_and_sec_fetch_site"

    return True, "ok"


def check_cookie_csrf(scope: dict[str, Any]) -> tuple[bool, str]:
    """Cross-site protection for cookie-authenticated state changes and /login."""
    return check_request_origin(scope, require_browser_signal=True)


def format_peer(scope: dict[str, Any]) -> str:
    """Format socket peer (plus bounded X-Forwarded-For hint when proxied) for logs."""
    client = scope.get("client")
    peer = f"{client[0]}:{client[1]}" if isinstance(client, (tuple, list)) and len(client) >= 2 else (
        str(client[0]) if isinstance(client, (tuple, list)) and client else "unknown"
    )
    xff = _header_first(scope, b"x-forwarded-for")
    if xff:
        return f"{peer} (xff={_clean_log_token(xff.split(',')[0].strip(), 64)})"
    return peer


def log_auth_failure(scope: dict[str, Any], *, reason: str) -> None:
    """Log a rejected access attempt with HTTP method, path, peer, and reason."""
    method = _clean_log_token(str(scope.get("method") or scope.get("type") or "HTTP"), 16)
    path = _clean_log_token(str(scope.get("path") or "<unknown>"), 200)
    peer = format_peer(scope)
    logger.warning("Access gate refused %s %s from %s (%s)", method, path, peer, reason)
    observability.record_event(
        "auth_failed",
        tier="events",
        min_level="minimal",
        fields={"method": method, "path": path, "peer": peer, "reason": reason},
    )


def render_login_html(error_message: str | None = None) -> bytes:
    """Render ``frontend/login.html`` with an optional escaped alert banner."""
    try:
        template = _LOGIN_HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        template = (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<title>Sign in · heylook</title></head><body>"
            "<h1>heylook</h1><!--LOGIN_ERROR-->"
            "<form method='post' action='/login'>"
            "<label for='api-key'>API key</label>"
            "<input type='password' id='api-key' name='api_key' autocomplete='current-password' required>"
            "<button type='submit'>Sign in</button></form></body></html>"
        )
    if error_message:
        escaped = html.escape(error_message)
        banner = f'<div id="login-error" class="error-note" role="alert">{escaped}</div>'
    else:
        banner = '<div id="login-error" class="error-note" role="alert" hidden></div>'
    return template.replace(_LOGIN_ERROR_PLACEHOLDER, banner, 1).encode("utf-8")


def _wants_html_login(scope: dict[str, Any], path: str, method: str) -> bool:
    """True for a browser GET/HEAD navigation to a non-API page expecting HTML."""
    if method not in ("GET", "HEAD") or path.startswith("/v1/"):
        return False
    if path in ("/openapi.json", "/manifest.json"):
        return False
    accept = (_header_first(scope, b"accept") or "").lower()
    return "text/html" in accept


async def _send_asgi_response(
    send,
    scope: dict[str, Any],
    *,
    status: int,
    content_type: bytes,
    body: bytes,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    is_head = str(scope.get("method") or "").upper() == "HEAD"
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", content_type),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": b"" if is_head else body})


_CSP_FRAME_ANCESTORS = b"frame-ancestors 'self'"
_X_FRAME_OPTIONS = b"SAMEORIGIN"


def _ensure_framing_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    present = {k.lower() for k, _ in headers}
    if b"content-security-policy" not in present:
        headers.append((b"content-security-policy", _CSP_FRAME_ANCESTORS))
    if b"x-frame-options" not in present:
        headers.append((b"x-frame-options", _X_FRAME_OPTIONS))
    return headers


def _with_security_headers(send, *, extra_response_headers: list[tuple[bytes, bytes]] | None = None):
    async def wrapped(message):
        if message["type"] == "http.response.start":
            hdrs = list(message.get("headers") or ())
            if extra_response_headers:
                existing = {k.lower() for k, _ in hdrs}
                for ek, ev in extra_response_headers:
                    if ek.lower() not in existing:
                        hdrs.append((ek, ev))
            _ensure_framing_headers(hdrs)
            message = {**message, "headers": hdrs}
        await send(message)

    return wrapped


async def _send_csrf_403(send, scope: dict[str, Any], *, reason: str) -> None:
    if reason.startswith("unlisted_forwarded_host:"):
        fwd_host = reason.split(":", 1)[1]
        msg = (
            f"Cross-origin state-changing request refused: X-Forwarded-Host '{fwd_host}' "
            "is not in heylook.toml's allowed_hosts. Add it there if you reach this server "
            "through tailscale serve or a reverse proxy that rewrites Host."
        )
    else:
        msg = (
            "Cross-origin state-changing request refused "
            "(Origin must match Host; Sec-Fetch-Site must be same-origin or none)."
        )
    csrf_body = json.dumps({
        "detail": msg,
        "type": "error",
        "error": {
            "type": "forbidden",
            "code": "csrf_rejected",
            "message": msg,
        },
    }).encode("utf-8")
    await _send_asgi_response(
        send, scope, status=403, content_type=b"application/json", body=csrf_body,
    )


class AccessGateMiddleware:
    """Single app-wide ASGI origin/framing & authentication gate."""

    def __init__(self, app) -> None:
        self.app = app
        self._own_names = own_names()

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        send_out = _with_security_headers(send) if scope_type == "http" else send
        path = scope.get("path") or ""
        method = str(scope.get("method") or ("GET" if scope_type == "websocket" else "")).upper()
        is_state_changing = scope_type == "websocket" or method not in _SAFE_METHODS

        # Always-on (independent of API key): refuse cross-origin and same-site
        # browser state-changing requests before any auth or route logic runs.
        if is_state_changing:
            origin_ok, origin_reason = check_request_origin(
                scope, require_browser_signal=False, own=self._own_names,
            )
            if not origin_ok:
                log_auth_failure(scope, reason=f"cross_origin:{origin_reason}")
                if scope_type == "websocket":
                    return await send({"type": "websocket.close", "code": 1008})
                return await _send_csrf_403(send_out, scope, reason=origin_reason)

        state = resolve_access_key()
        if not state.enabled:
            return await self.app(scope, receive, send_out)

        # /health stays up even when the key file is misconfigured, revealing nothing.
        if scope_type == "http" and (path, method) in {("/health", "GET"), ("/health", "HEAD")}:
            return await self.app(scope, receive, send_out)

        # Fail closed when a key file exists with bad permissions or is empty.
        if state.error or not state.key:
            log_auth_failure(scope, reason="key_config_error")
            if scope_type == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            payload = json.dumps({
                "detail": f"Server access key is misconfigured: {state.error}",
                "type": "error",
                "error": {
                    "type": "api_error",
                    "code": "auth_misconfigured",
                    "message": f"Server access key is misconfigured: {state.error}",
                },
            }).encode("utf-8")
            return await _send_asgi_response(
                send_out, scope, status=503, content_type=b"application/json", body=payload,
            )

        if scope_type == "http" and (path, method) in EXEMPT_ROUTES:
            return await self.app(scope, receive, send_out)

        expected = state.key
        failure_reason = "missing_credentials"

        # 1. x-api-key: <key> (Anthropic SDK default)
        api_key_hdr = _header_first(scope, b"x-api-key")
        if api_key_hdr is not None:
            candidate = api_key_hdr.strip()
            if candidate and verify_api_key(candidate, expected):
                return await self.app(scope, receive, send_out)
            if candidate:
                failure_reason = "invalid_api_key"

        # 2. Authorization: Bearer <key> (OpenAI discovery & Anthropic auth_token)
        auth_hdr = _header_first(scope, b"authorization")
        if auth_hdr is not None:
            scheme, _, bearer_tok = auth_hdr.strip().partition(" ")
            candidate = bearer_tok.strip()
            if scheme.lower() == "bearer" and candidate:
                if verify_api_key(candidate, expected):
                    return await self.app(scope, receive, send_out)
                failure_reason = "invalid_bearer_token"
            elif auth_hdr.strip():
                failure_reason = "unsupported_authorization_scheme"

        # 3. Session cookie (web UI)
        session_tokens = extract_session_cookies(scope)
        store = get_session_store()
        if session_tokens:
            if any(store.verify(tok, expected) for tok in session_tokens):
                if is_state_changing:
                    csrf_ok, csrf_reason = check_request_origin(
                        scope, require_browser_signal=True, own=self._own_names,
                    )
                    if not csrf_ok:
                        log_auth_failure(scope, reason=f"cross_origin_cookie:{csrf_reason}")
                        if scope_type == "websocket":
                            return await send({"type": "websocket.close", "code": 1008})
                        return await _send_csrf_403(send_out, scope, reason=csrf_reason)
                if scope_type == "http":
                    send_session = _with_security_headers(
                        send, extra_response_headers=[(b"x-heylook-session", b"1")],
                    )
                    return await self.app(scope, receive, send_session)
                return await self.app(scope, receive, send)
            if failure_reason == "missing_credentials":
                failure_reason = "invalid_session"

        log_auth_failure(scope, reason=failure_reason)
        if scope_type == "websocket":
            return await send({"type": "websocket.close", "code": 1008})

        extra: list[tuple[bytes, bytes]] = []
        if session_tokens:
            extra.append((b"set-cookie", format_clear_session_cookie(secure=is_https_scope(scope))))

        if _wants_html_login(scope, path, method):
            html_body = render_login_html()
            return await _send_asgi_response(
                send_out,
                scope,
                status=401,
                content_type=b"text/html; charset=utf-8",
                body=html_body,
                extra_headers=extra,
            )

        extra.append((b"www-authenticate", b"Bearer"))
        json_body = json.dumps({
            "detail": "Authentication required.",
            "type": "error",
            "error": {
                "type": "authentication_error",
                "code": "auth_required",
                "message": "Authentication required. Send x-api-key, Authorization: Bearer, or sign in at /login.",
            },
        }).encode("utf-8")
        return await _send_asgi_response(
            send_out,
            scope,
            status=401,
            content_type=b"application/json",
            body=json_body,
            extra_headers=extra,
        )


def access_config_view() -> dict[str, Any]:
    """Read-only summary of the access gate for ``/v1/admin/config``. Never includes the key."""
    state = resolve_access_key()
    out: dict[str, Any] = {
        "enabled": state.enabled,
        "source": state.source,
        "source_detail": state.source_detail,
        "admin_token_enabled": bool(os.environ.get(_ADMIN_TOKEN_ENV, "").strip()),
    }
    if state.error:
        out["error"] = state.error
    return out


def is_loopback_host(host: str) -> bool:
    """True iff ``host`` binds strictly to loopback (``localhost``, ``127.0.0.0/8``, ``::1``)."""
    cleaned = (host or "").strip().lower().strip("[]").rstrip(".")
    if cleaned == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def log_startup_access_state(host: str, port: int) -> None:
    """Validate key configuration, log gate status and source, and warn loudly if exposed open."""
    state = resolve_access_key()
    if state.error:
        raise RuntimeError(state.error)

    if state.enabled:
        logger.info("Access gate: ON (key source: %s -> %s)", state.source, state.source_detail)
    else:
        logger.info(
            "Access gate: OFF (no %s env var and no ~/.heylook/api_key file)",
            API_KEY_ENV,
        )
        if not is_loopback_host(host):
            logger.warning(
                "SECURITY WARNING: Listening on non-loopback address %s:%s with NO access key "
                "configured! Every route (inference, conversations, notebooks, presets, model "
                "load/unload, and config) is open without authentication. Run "
                "'heylookllm key create --copy' (mode-600 ~/.heylook/api_key) or set %s.",
                host,
                port,
                API_KEY_ENV,
            )


def client_auth_headers(explicit_key: str | None = None) -> dict[str, str]:
    """Headers for local CLI/probe scripts calling a heylook server."""
    key = (explicit_key or "").strip()
    if not key:
        state = resolve_access_key()
        key = state.key or ""
    headers: dict[str, str] = {}
    if key:
        headers["x-api-key"] = key
    admin_tok = os.environ.get(_ADMIN_TOKEN_ENV, "").strip()
    if admin_tok:
        headers[_ADMIN_TOKEN_HEADER] = admin_tok
    return headers


def require_admin_token(request: Request) -> None:
    """FastAPI dependency: optional second factor gating admin routes behind HEYLOOK_ADMIN_TOKEN.

    Raises ``HTTPException(401)`` when the env var is set to a non-empty
    value AND the incoming request's ``X-Heylook-Admin-Token`` header
    doesn't match. No-op when the env var is unset or empty.
    """
    expected = os.environ.get(_ADMIN_TOKEN_ENV, "").strip()
    if not expected:
        return None

    provided = request.headers.get(_ADMIN_TOKEN_HEADER) or ""
    if not verify_api_key(provided, expected):
        logger.warning(
            "admin-token mismatch on %s from %s; set HEYLOOK_ADMIN_TOKEN or send "
            "the X-Heylook-Admin-Token header",
            _clean_log_token(request.url.path, 200) if hasattr(request, "url") else "<request>",
            format_peer(request.scope) if hasattr(request, "scope") else "unknown",
        )
        raise HTTPException(
            status_code=401,
            detail=f"Admin endpoint requires {_ADMIN_TOKEN_HEADER} header.",
        )
    return None


def target_api_key_path() -> Path:
    """Path written/cleared by ``heylookllm key`` (honors ``$HEYLOOK_API_KEY_FILE`` when non-empty)."""
    raw = os.environ.get(API_KEY_FILE_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_API_KEY_PATH


def _reset_file_cache() -> None:
    global _file_cache_stamp, _file_cache_state
    with _file_cache_lock:
        _file_cache_stamp = None
        _file_cache_state = None


def _atomic_write_0600(target: Path, content: str) -> None:
    """Atomically write ``content`` to ``target`` with mode 0600, refusing symlinks."""
    shown = _display_path(target)
    if target.is_symlink():
        raise RuntimeError(f"Refusing to write through symlink {shown}.")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent == DEFAULT_API_KEY_PATH.parent:
        try:
            parent.chmod(0o700)
        except OSError:
            pass

    tmp_path = parent / f".{target.name}.tmp.{os.getpid()}.{secrets.token_hex(6)}"
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise
    os.close(fd)

    try:
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_access_key_file(
    secret: str,
    *,
    overwrite: bool = False,
    key_path: Path | None = None,
) -> Path:
    """Atomically write a mode-0600 access key file (refusing overwrite unless ``overwrite``)."""
    cleaned = (secret or "").strip()
    if not cleaned:
        raise ValueError("Refusing to write an empty access key.")

    target = key_path if key_path is not None else target_api_key_path()
    shown = _display_path(target)
    if target.is_symlink():
        raise RuntimeError(f"Refusing to write through symlink {shown}.")
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Key file already exists at {shown}. Pass --rotate to replace it "
            "(invalidates active browser sessions and requires updating remote clients)."
        )

    _atomic_write_0600(target, f"{cleaned}\n")
    _reset_file_cache()
    return target


def create_access_key(*, rotate: bool = False, key_path: Path | None = None) -> tuple[str, Path]:
    """Atomically mint and write a mode-0600 access key file (refusing overwrite without ``rotate``)."""
    new_key = f"hlk_{secrets.token_urlsafe(32)}"
    target = write_access_key_file(new_key, overwrite=rotate, key_path=key_path)
    return new_key, target


def clear_access_key(*, key_path: Path | None = None) -> tuple[bool, Path]:
    """Delete the configured key file if present."""
    target = key_path if key_path is not None else target_api_key_path()
    existed = target.exists() or target.is_symlink()
    target.unlink(missing_ok=True)
    _reset_file_cache()
    return existed, target


def manage_access_key(args: Any) -> int:
    """CLI handler for ``heylookllm key [status|create|clear]`` (plain mode-0600 file only)."""
    action = getattr(args, "action", "status") or "status"
    target = target_api_key_path()
    shown = _display_path(target)

    if action == "status":
        state = resolve_access_key()
        if state.error:
            print(
                f"Access gate: MISCONFIGURED (refusing requests with 503)\n  {state.error}",
                file=sys.stderr,
            )
            return 1
        if state.enabled:
            print(f"Access gate: ON (source: {state.source} -> {state.source_detail})")
        else:
            print("Access gate: OFF (unset; run 'heylookllm key create --copy' to enable)")
        if os.environ.get(_ADMIN_TOKEN_ENV, "").strip():
            print("Admin second factor: ON (HEYLOOK_ADMIN_TOKEN; note: web UI never sends this header)")
        return 0

    if action == "create":
        rotate = bool(getattr(args, "rotate", False))
        copy_to_clipboard = bool(getattr(args, "copy", False))
        try:
            new_key, target = create_access_key(rotate=rotate)
        except (FileExistsError, RuntimeError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        shown = _display_path(target)
        verb = "Rotated" if rotate else "Created"
        if copy_to_clipboard:
            try:
                subprocess.run(["pbcopy"], input=new_key.encode("utf-8"), check=True)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                print(
                    f"{verb} access key at {shown} (mode 0600), but clipboard copy failed ({exc}).",
                    file=sys.stderr,
                )
                return 1
            print(f"{verb} access key at {shown} (mode 0600) and copied to clipboard.")
        else:
            print(f"{verb} access key at {shown} (mode 0600):")
            print(new_key)

        if os.environ.get(API_KEY_ENV, "").strip():
            print(
                f"Note: {API_KEY_ENV} is set in this shell and overrides {shown} when set in the server environment.",
                file=sys.stderr,
            )
        return 0

    if action == "clear":
        try:
            existed, target = clear_access_key()
        except OSError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        shown = _display_path(target)
        print(f"Removed {shown}." if existed else f"No key file at {shown}.")
        if os.environ.get(API_KEY_ENV, "").strip():
            print(
                f"Warning: {API_KEY_ENV} is still set in the environment; unset it to turn the gate off.",
                file=sys.stderr,
            )
        return 0

    print(f"Unknown key action: {action}", file=sys.stderr)
    return 2
