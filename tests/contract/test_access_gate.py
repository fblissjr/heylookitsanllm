# tests/contract/test_access_gate.py
"""Contract tests for the single app-wide access gate (AccessGateMiddleware).

Covers:
  1. Exhaustive enumeration of ``app.routes`` (recursing through FastAPI's
     ``_IncludedRouter`` wrappers) asserting every non-exempt ``(path, method)``
     returns 401 without credentials so a forgotten router fails CI.
  2. All three credential types (``x-api-key``, ``Authorization: Bearer``, and
     the ``heylook_session`` cookie from ``POST /login``), including SSE
     streaming on ``POST /v1/messages`` and ``POST /logout`` revocation.
  3. Cross-origin cookie request refusal (403) across sibling subdomains,
     cross-port loopback, ``Origin: null``, scheme downgrade, and missing
     ``Origin`` + ``Sec-Fetch-Site``.
  4. Zero source-IP exemptions (loopback IPv4/IPv6 and LAN proxy peers all 401).
  5. Composition with ``HEYLOOK_ADMIN_TOKEN`` as an optional second factor on
     admin routes, and ``/v1/admin/config`` exposing gate state without ever
     echoing the key.
"""

from __future__ import annotations

import re

from fastapi.routing import APIRoute
import pytest
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from heylook_llm.auth import EXEMPT_ROUTES, SESSION_COOKIE_NAME, get_session_store

TEST_KEY = "test-gate-secret-key-9f8e7d6c5b4a"
ROTATED_KEY = "rotated-gate-secret-key-1a2b3c4d"

_PARAM_RE = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")


def _walk_http_routes(routes, prefix: str = ""):
    """Yield ``(route_template, method)`` across ``app.routes`` and nested routers."""
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _walk_http_routes(inner.routes, prefix=prefix)
        elif isinstance(route, (APIRoute, Route)):
            full_path = prefix + route.path
            for method in sorted(route.methods or ()):
                if method == "OPTIONS":
                    continue
                yield (full_path, method.upper())
        elif isinstance(route, Mount) and getattr(route, "routes", None):
            yield from _walk_http_routes(route.routes, prefix=prefix + route.path.rstrip("/"))


def _concrete_path(template: str) -> str:
    """Instantiate path parameters to non-exempt sample paths."""
    if template == "/js/{rest:path}":
        return "/js/app.js"
    if template == "/css/{rest:path}":
        return "/css/protected.css"
    if template == "/v1/performance/profile/{time_range}":
        return "/v1/performance/profile/1h"
    return _PARAM_RE.sub("sample-id", template)


class TestRoutePopulationGate:
    """Every served route must sit behind the single ASGI access gate."""

    def test_exempt_routes_match_strict_allowlist_and_health_reveals_nothing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)

        expected_exempt = frozenset({
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
        assert EXEMPT_ROUTES == expected_exempt

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert client.head("/health").status_code == 200

        login_get = client.get("/login")
        assert login_get.status_code == 200
        assert 'autocomplete="current-password"' in login_get.text
        assert TEST_KEY not in login_get.text

        for asset in ("/css/app.css", "/js/login.js", "/icon.svg", "/favicon.ico"):
            assert client.get(asset).status_code == 200, asset

    def test_every_non_exempt_route_answers_401_without_credentials(
        self, app, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)

        discovered = list(_walk_http_routes(app.routes))
        discovered_templates = {tpl for tpl, _ in discovered}
        # Guard the walk itself so _IncludedRouter blind spots can never return.
        must_see = {
            "/",
            "/index.html",
            "/manifest.json",
            "/js/{rest:path}",
            "/css/{rest:path}",
            "/openapi.json",
            "/docs",
            "/redoc",
            "/login",
            "/logout",
            "/health",
            "/v1/models",
            "/v1/messages",
            "/v1/embeddings",
            "/v1/conversations",
            "/v1/conversations/{conv_id}/generate",
            "/v1/notebooks",
            "/v1/presets",
            "/v1/admin/models",
            "/v1/admin/config",
            "/v1/system/metrics",
            "/v1/capabilities",
            "/v1/telemetry/events",
            "/v1/data/clear",
        }
        assert must_see <= discovered_templates, (
            f"Route walk missed expected mounts: {sorted(must_see - discovered_templates)}"
        )

        checked = 0
        failures: list[str] = []
        for template, method in discovered:
            path = _concrete_path(template)
            if (path, method) in EXEMPT_ROUTES:
                continue
            checked += 1
            res = client.request(method, path)
            if res.status_code != 401:
                failures.append(f"{method} {path} (from {template}) -> {res.status_code}")

        assert checked >= 50, f"Expected >=50 gated (route, method) pairs, checked {checked}"
        assert not failures, "Routes escaped the access gate:\n  " + "\n  ".join(failures)

    def test_browser_html_navigation_returns_401_with_login_page(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        res = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
        assert res.status_code == 401
        assert "text/html" in res.headers["content-type"]
        assert 'id="login-form"' in res.text

        api_res = client.get("/v1/models", headers={"Accept": "text/html"})
        assert api_res.status_code == 401
        assert api_res.headers["content-type"] == "application/json"
        assert api_res.json()["error"]["code"] == "auth_required"


class TestNoSourceIpExemptions:
    """Loopback and LAN proxies hide real client addresses; none may be exempt."""

    @pytest.mark.parametrize(
        "peer_ip, extra_headers",
        [
            ("127.0.0.1", {}),
            ("::1", {}),
            ("192.0.2.251", {"X-Forwarded-For": "198.51.100.42", "X-Forwarded-Proto": "https"}),
        ],
    )
    def test_unauthenticated_peers_always_get_401(
        self,
        app,
        monkeypatch: pytest.MonkeyPatch,
        peer_ip: str,
        extra_headers: dict[str, str],
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        with TestClient(app, client=(peer_ip, 51234), raise_server_exceptions=False) as c:
            res = c.get("/v1/models", headers=extra_headers)
            assert res.status_code == 401
            assert res.json()["error"]["code"] == "auth_required"


class TestThreeCredentialTypes:
    """x-api-key, Authorization: Bearer, and session cookie must all succeed, streaming included."""

    def _mint_session_cookie(self, client: TestClient) -> str:
        login = client.post(
            "/login",
            json={"api_key": TEST_KEY},
            headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
        )
        assert login.status_code == 200, login.text
        token = login.cookies.get(SESSION_COOKIE_NAME)
        assert token
        assert token != TEST_KEY
        assert TEST_KEY not in login.headers.get("set-cookie", "")
        return token

    @pytest.mark.parametrize("cred_kind", ["x_api_key", "bearer", "cookie"])
    def test_credential_succeeds_for_discovery_non_streaming_and_sse_streaming(
        self, app, monkeypatch: pytest.MonkeyPatch, cred_kind: str
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        with TestClient(app, raise_server_exceptions=False) as c:
            headers: dict[str, str] = {}
            if cred_kind == "x_api_key":
                headers["x-api-key"] = TEST_KEY
            elif cred_kind == "bearer":
                headers["Authorization"] = f"Bearer {TEST_KEY}"
            else:
                self._mint_session_cookie(c)
                headers["Origin"] = "http://testserver"

            models_res = c.get("/v1/models", headers=headers)
            assert models_res.status_code == 200
            assert len(models_res.json()["data"]) >= 1

            index_res = c.get("/", headers=headers)
            assert index_res.status_code == 200

            non_stream = c.post(
                "/v1/messages",
                headers=headers,
                json={
                    "model": "test-mlx-model",
                    "max_tokens": 16,
                    "stream": False,
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
            )
            assert non_stream.status_code == 200
            blocks = non_stream.json()["content"]
            assert "".join(b["text"] for b in blocks if b["type"] == "text") == "Hello, world!"

            with c.stream(
                "POST",
                "/v1/messages",
                headers=headers,
                json={
                    "model": "test-mlx-model",
                    "max_tokens": 16,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Stream hello"}],
                },
            ) as stream_res:
                assert stream_res.status_code == 200
                raw_sse = "".join(stream_res.iter_text())
            assert "event: message_start" in raw_sse
            assert "Hello" in raw_sse and "world!" in raw_sse
            assert "event: message_stop" in raw_sse

    def test_cookie_flags_on_loopback_http_vs_proxied_https_and_logout_revocation(
        self, app, mock_router, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        monkeypatch.setattr(
            mock_router.app_config,
            "allowed_hosts",
            ["testserver", "heylook.lan.example"],
        )

        # 1. Plain HTTP loopback (http://127.0.0.1:42193): HttpOnly + SameSite=Strict, NO Secure.
        with TestClient(
            app, base_url="http://127.0.0.1:42193", raise_server_exceptions=False
        ) as local_c:
            res_http = local_c.post(
                "/login",
                data={"api_key": TEST_KEY},
                headers={"Origin": "http://127.0.0.1:42193"},
                follow_redirects=False,
            )
            assert res_http.status_code == 303
            sc_http = res_http.headers["set-cookie"]
            assert "HttpOnly" in sc_http
            assert "SameSite=Strict" in sc_http
            assert "Domain=" not in sc_http
            assert "Secure" not in sc_http

            token = local_c.cookies.get(SESSION_COOKIE_NAME)
            assert token and token != TEST_KEY
            authed = local_c.get("/v1/models")
            assert authed.status_code == 200
            assert authed.headers.get("x-heylook-session") == "1"

            # Logout revokes server-side token AND clears cookie.
            logout = local_c.post(
                "/logout",
                headers={"Origin": "http://127.0.0.1:42193", "Accept": "application/json"},
            )
            assert logout.status_code == 200
            assert "Max-Age=0" in logout.headers["set-cookie"]
            assert not get_session_store().verify(token, TEST_KEY)

            # Replaying the revoked token must fail immediately with 401.
            replay = local_c.get("/v1/models", headers={"Cookie": f"{SESSION_COOKIE_NAME}={token}"})
            assert replay.status_code == 401

        # 2. Proxied HTTPS (reverse proxy sets X-Forwarded-Proto: https): MUST include Secure.
        with TestClient(
            app, base_url="http://heylook.lan.example", raise_server_exceptions=False
        ) as https_c:
            res_https = https_c.post(
                "/login",
                json={"api_key": TEST_KEY},
                headers={
                    "Origin": "https://heylook.lan.example",
                    "X-Forwarded-Proto": "https",
                },
            )
            assert res_https.status_code == 200
            sc_https = res_https.headers["set-cookie"]
            assert "HttpOnly" in sc_https
            assert "SameSite=Strict" in sc_https
            assert "Secure" in sc_https
            assert "Domain=" not in sc_https

    def test_key_rotation_invalidates_existing_sessions(
        self, app, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        with TestClient(app, raise_server_exceptions=False) as c:
            login = c.post(
                "/login",
                json={"api_key": TEST_KEY},
                headers={"Origin": "http://testserver"},
            )
            assert login.status_code == 200
            assert c.get("/v1/models").status_code == 200

            monkeypatch.setenv("HEYLOOK_API_KEY", ROTATED_KEY)
            after = c.get("/v1/models")
            assert after.status_code == 401


class TestCrossOriginCookieProtection:
    """State-changing cookie requests must refuse cross-origin and sibling-subdomain callers."""

    @pytest.mark.parametrize(
        "bad_headers",
        [
            pytest.param(
                {"Origin": "https://other.lan.example", "Sec-Fetch-Site": "same-site", "X-Forwarded-Proto": "https"},
                id="sibling_subdomain_same_site_rejected",
            ),
            pytest.param(
                {"Origin": "http://heylook.lan.example:8080"},
                id="cross_port_origin_rejected",
            ),
            pytest.param(
                {"Origin": "null", "Sec-Fetch-Site": "cross-site"},
                id="null_origin_rejected",
            ),
            pytest.param(
                {"Sec-Fetch-Site": "cross-site"},
                id="sec_fetch_site_cross_site_rejected",
            ),
            pytest.param(
                {"Origin": "http://heylook.lan.example", "X-Forwarded-Proto": "https"},
                id="https_to_http_scheme_downgrade_rejected",
            ),
            pytest.param(
                {},
                id="missing_both_origin_and_sec_fetch_site_rejected",
            ),
        ],
    )
    def test_cross_origin_cookie_post_is_refused_with_403(
        self,
        app,
        mock_router,
        monkeypatch: pytest.MonkeyPatch,
        bad_headers: dict[str, str],
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        monkeypatch.setattr(
            mock_router.app_config,
            "allowed_hosts",
            ["testserver", "heylook.lan.example"],
        )
        with TestClient(
            app, base_url="https://heylook.lan.example", raise_server_exceptions=False
        ) as c:
            login = c.post(
                "/login",
                json={"api_key": TEST_KEY},
                headers={
                    "Origin": "https://heylook.lan.example",
                    "X-Forwarded-Proto": "https",
                },
            )
            assert login.status_code == 200
            assert c.cookies.get(SESSION_COOKIE_NAME)

            res = c.post(
                "/v1/messages",
                headers=bad_headers,
                json={
                    "model": "test-mlx-model",
                    "max_tokens": 8,
                    "stream": False,
                    "messages": [{"role": "user", "content": "csrf"}],
                },
            )
            assert res.status_code == 403
            assert res.json()["error"]["code"] == "csrf_rejected"

    def test_loopback_cross_port_cookie_post_is_refused_with_403(
        self, app, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        with TestClient(
            app, base_url="http://127.0.0.1:42193", raise_server_exceptions=False
        ) as c:
            login = c.post(
                "/login",
                json={"api_key": TEST_KEY},
                headers={"Origin": "http://127.0.0.1:42193"},
            )
            assert login.status_code == 200

            # Same-origin loopback POST succeeds.
            ok = c.post(
                "/v1/messages",
                headers={"Origin": "http://127.0.0.1:42193"},
                json={
                    "model": "test-mlx-model",
                    "max_tokens": 8,
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert ok.status_code == 200

            # Another local app on 127.0.0.1:3000 shares SameSite cookies on 127.0.0.1;
            # Origin vs Host port check must reject it with 403.
            bad = c.post(
                "/v1/messages",
                headers={"Origin": "http://127.0.0.1:3000"},
                json={
                    "model": "test-mlx-model",
                    "max_tokens": 8,
                    "stream": False,
                    "messages": [{"role": "user", "content": "csrf"}],
                },
            )
            assert bad.status_code == 403
            assert bad.json()["error"]["code"] == "csrf_rejected"

    def test_cross_origin_login_post_is_refused_with_403(
        self, app, mock_router, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        monkeypatch.setattr(
            mock_router.app_config,
            "allowed_hosts",
            ["testserver", "heylook.lan.example"],
        )
        with TestClient(
            app, base_url="http://heylook.lan.example", raise_server_exceptions=False
        ) as c:
            res = c.post(
                "/login",
                json={"api_key": TEST_KEY},
                headers={"Origin": "https://other.lan.example", "Sec-Fetch-Site": "same-site"},
            )
            assert res.status_code == 403


class TestAdminTokenSecondFactorAndConfigVisibility:
    """HEYLOOK_ADMIN_TOKEN stays as an optional extra factor on admin routes; config shows state only."""

    def test_admin_config_shows_gate_state_never_the_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        res = client.get("/v1/admin/config", headers={"x-api-key": TEST_KEY})
        assert res.status_code == 200
        body = res.json()
        assert body["access"] == {
            "enabled": True,
            "source": "env",
            "source_detail": "HEYLOOK_API_KEY",
            "admin_token_enabled": False,
        }
        assert TEST_KEY not in res.text

    def test_admin_token_layers_behind_access_gate(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", TEST_KEY)
        monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", "extra-admin-secret")

        # 1. Admin token alone without access key -> 401 at outer gate.
        only_admin = client.get(
            "/v1/admin/config",
            headers={"X-Heylook-Admin-Token": "extra-admin-secret"},
        )
        assert only_admin.status_code == 401
        assert only_admin.json()["error"]["code"] == "auth_required"

        # 2. Access key alone -> inference/discovery allowed, admin routes 401.
        assert client.get("/v1/models", headers={"x-api-key": TEST_KEY}).status_code == 200
        missing_second = client.get("/v1/admin/config", headers={"x-api-key": TEST_KEY})
        assert missing_second.status_code == 401
        assert "X-Heylook-Admin-Token" in missing_second.json()["detail"]

        # 3. Both factors present -> admin routes succeed.
        both = client.get(
            "/v1/admin/config",
            headers={
                "x-api-key": TEST_KEY,
                "X-Heylook-Admin-Token": "extra-admin-secret",
            },
        )
        assert both.status_code == 200
        assert both.json()["access"]["admin_token_enabled"] is True
        assert TEST_KEY not in both.text
        assert "extra-admin-secret" not in both.text
