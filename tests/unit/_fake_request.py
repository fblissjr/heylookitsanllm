"""Minimal Starlette-Request stand-in for auth-dependency tests.

Shared between test_admin_auth.py (admin-token) and test_api_key_auth.py
(inference api-key). Covers the two attributes those dependencies touch:
``request.headers.get(name)`` (case-insensitive per RFC 7230) and
``request.client.host`` (resolved peer IP, optional -- ``None`` for test
clients and some middlewares, which must fail closed).

Kept deliberately small -- it isn't a Request mock, just enough surface to
let a FastAPI dependency run without spinning up a full ASGI stack.
"""

from __future__ import annotations

from typing import Optional


class _FakeClient:
    def __init__(self, host: str):
        self.host = host


class _CaseInsensitiveHeaders:
    def __init__(self, items: dict[str, str]):
        self._items = {k.lower(): v for k, v in items.items()}

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._items.get(name.lower(), default)


class FakeRequest:
    """Stand-in for ``fastapi.Request`` scoped to auth-dependency tests."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        host: str | None = "192.168.1.50",
    ):
        self._headers = _CaseInsensitiveHeaders(headers or {})
        self.client: _FakeClient | None = (
            _FakeClient(host) if host is not None else None
        )

    @property
    def headers(self) -> _CaseInsensitiveHeaders:
        return self._headers
