"""Tests for admin-endpoint authentication (S1.6).

The admin-token middleware is opt-in via HEYLOOK_ADMIN_TOKEN env var.
When unset, admin endpoints are unauthenticated (backward-compat for
the default localhost-only single-user deployment). When set, admin
endpoints require X-Heylook-Admin-Token header matching the env value.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from _fake_request import FakeRequest as _FakeRequest


class TestRequireAdminToken:
    def test_no_env_var_allows_through(self, monkeypatch: pytest.MonkeyPatch):
        """HEYLOOK_ADMIN_TOKEN unset: middleware is a no-op."""
        monkeypatch.delenv("HEYLOOK_ADMIN_TOKEN", raising=False)
        from heylook_llm.auth import require_admin_token

        # No header, no env var -- must not raise.
        require_admin_token(_FakeRequest())

    def test_matching_header_allows_through(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", "secret-token-value")
        from heylook_llm.auth import require_admin_token

        request = _FakeRequest({"X-Heylook-Admin-Token": "secret-token-value"})
        require_admin_token(request)

    def test_missing_header_rejects(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", "secret")
        from heylook_llm.auth import require_admin_token

        with pytest.raises(HTTPException) as excinfo:
            require_admin_token(_FakeRequest())
        assert excinfo.value.status_code == 401

    def test_wrong_header_rejects(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", "secret")
        from heylook_llm.auth import require_admin_token

        with pytest.raises(HTTPException) as excinfo:
            require_admin_token(_FakeRequest({"X-Heylook-Admin-Token": "wrong"}))
        assert excinfo.value.status_code == 401

    def test_empty_env_var_is_no_op(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit empty HEYLOOK_ADMIN_TOKEN='' should be treated as unset.

        Catches the footgun where someone exports the var without assigning
        a value. Without this, every admin call would reject because ''
        never matches any header. Explicit no-op on empty is friendlier.
        """
        monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", "")
        from heylook_llm.auth import require_admin_token

        # Empty token -> treated as unset -> allow through.
        require_admin_token(_FakeRequest())

    def test_header_case_insensitive(self, monkeypatch: pytest.MonkeyPatch):
        """HTTP headers are case-insensitive per RFC 7230; Starlette's
        Request.headers.get() handles this. Confirm the middleware works
        regardless of header case."""
        monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", "secret")
        from heylook_llm.auth import require_admin_token

        # Lowercase
        require_admin_token(_FakeRequest({"x-heylook-admin-token": "secret"}))
        # Uppercase
        require_admin_token(_FakeRequest({"X-HEYLOOK-ADMIN-TOKEN": "secret"}))
