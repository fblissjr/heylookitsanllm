"""Tests for optional bearer API-key auth on inference endpoints.

Design (C1.5):
- ``HEYLOOK_API_KEY`` unset/empty: no check, inference endpoints are open
  (back-compat for the default localhost single-user setup).
- ``HEYLOOK_API_KEY`` set: ``Authorization: Bearer <value>`` required for
  non-loopback clients on gated inference endpoints. Loopback (127.0.0.1,
  ::1) is exempt by default -- dev tools on the same machine don't need
  to carry the key.
- ``HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true``: disable the loopback carve-out
  so loopback clients also need the key (for paranoid / public-exposed
  setups).

Constant-time comparison via ``hmac.compare_digest``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from _fake_request import FakeRequest as _FakeRequest


class TestRequireApiKey:
    def test_no_env_var_allows_through(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("HEYLOOK_API_KEY", raising=False)
        from heylook_llm.auth import require_api_key

        require_api_key(_FakeRequest())

    def test_empty_env_var_allows_through(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit empty HEYLOOK_API_KEY='' treated as unset -- same rule
        as HEYLOOK_ADMIN_TOKEN. Catches the export-without-value footgun."""
        monkeypatch.setenv("HEYLOOK_API_KEY", "")
        from heylook_llm.auth import require_api_key

        require_api_key(_FakeRequest())

    def test_missing_header_rejects_for_lan(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret-inference-key")
        from heylook_llm.auth import require_api_key

        with pytest.raises(HTTPException) as exc:
            require_api_key(_FakeRequest(host="192.168.1.50"))
        assert exc.value.status_code == 401

    def test_wrong_header_rejects(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        from heylook_llm.auth import require_api_key

        with pytest.raises(HTTPException) as exc:
            require_api_key(
                _FakeRequest(
                    headers={"Authorization": "Bearer wrong-value"},
                    host="192.168.1.50",
                )
            )
        assert exc.value.status_code == 401

    def test_matching_bearer_header_allows_through(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        from heylook_llm.auth import require_api_key

        require_api_key(
            _FakeRequest(
                headers={"Authorization": "Bearer secret"},
                host="192.168.1.50",
            )
        )

    def test_bearer_scheme_required(self, monkeypatch: pytest.MonkeyPatch):
        """Authorization header without 'Bearer ' prefix must be rejected
        even if the raw value matches. Prevents confusion with Basic auth or
        a raw token in the header."""
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        from heylook_llm.auth import require_api_key

        with pytest.raises(HTTPException):
            require_api_key(
                _FakeRequest(
                    headers={"Authorization": "secret"},
                    host="192.168.1.50",
                )
            )

    def test_bearer_scheme_case_insensitive(self, monkeypatch: pytest.MonkeyPatch):
        """RFC 6750 says the scheme token is case-insensitive."""
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        from heylook_llm.auth import require_api_key

        require_api_key(
            _FakeRequest(
                headers={"Authorization": "bearer secret"},
                host="192.168.1.50",
            )
        )
        require_api_key(
            _FakeRequest(
                headers={"Authorization": "BEARER secret"},
                host="192.168.1.50",
            )
        )


class TestLoopbackCarveOut:
    def test_loopback_ipv4_exempt_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        monkeypatch.delenv("HEYLOOK_API_KEY_ENFORCE_LOOPBACK", raising=False)
        from heylook_llm.auth import require_api_key

        require_api_key(_FakeRequest(host="127.0.0.1"))

    def test_loopback_ipv6_exempt_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        monkeypatch.delenv("HEYLOOK_API_KEY_ENFORCE_LOOPBACK", raising=False)
        from heylook_llm.auth import require_api_key

        require_api_key(_FakeRequest(host="::1"))

    def test_enforce_loopback_flag_closes_carve_out(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        monkeypatch.setenv("HEYLOOK_API_KEY_ENFORCE_LOOPBACK", "true")
        from heylook_llm.auth import require_api_key

        with pytest.raises(HTTPException):
            require_api_key(_FakeRequest(host="127.0.0.1"))

    def test_enforce_loopback_false_values(self, monkeypatch: pytest.MonkeyPatch):
        """Only 'true'/'1'/'yes' should close the carve-out. Anything else
        (including 'false', 'no', '0', empty) leaves loopback exempt."""
        from heylook_llm.auth import require_api_key

        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        for falsy in ("false", "no", "0", "", "nope"):
            monkeypatch.setenv("HEYLOOK_API_KEY_ENFORCE_LOOPBACK", falsy)
            require_api_key(_FakeRequest(host="127.0.0.1"))

    def test_enforce_loopback_with_matching_key_still_allows(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When ENFORCE_LOOPBACK=true and loopback client sends the right key,
        it passes."""
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        monkeypatch.setenv("HEYLOOK_API_KEY_ENFORCE_LOOPBACK", "true")
        from heylook_llm.auth import require_api_key

        require_api_key(
            _FakeRequest(
                headers={"Authorization": "Bearer secret"},
                host="127.0.0.1",
            )
        )

    def test_no_client_treated_as_non_loopback(self, monkeypatch: pytest.MonkeyPatch):
        """Requests without a client (test clients, odd middlewares) must
        default to the non-loopback path -- i.e. require the key. Never
        fail-open."""
        monkeypatch.setenv("HEYLOOK_API_KEY", "secret")
        from heylook_llm.auth import require_api_key

        req = _FakeRequest()
        req.client = None  # type: ignore[assignment]
        with pytest.raises(HTTPException):
            require_api_key(req)
