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


# One pure guard, one row per case: (env value or None for unset, header
# dicts each sent separately, expected status -- None means allowed through).
# - empty env: an exported-but-unassigned HEYLOOK_ADMIN_TOKEN='' is treated
#   as unset. Without that, every admin call would reject because '' never
#   matches any header.
# - case: HTTP headers are case-insensitive per RFC 7230; Starlette's
#   Request.headers.get() handles this, and the guard must work regardless
#   of header case (lowercase and uppercase both pass).
_ADMIN_TOKEN_ROWS = [
    pytest.param(None, [{}], None, id="no_env_var_allows_through"),
    pytest.param(
        "secret-token-value",
        [{"X-Heylook-Admin-Token": "secret-token-value"}],
        None,
        id="matching_header_allows_through",
    ),
    pytest.param("secret", [{}], 401, id="missing_header_rejects"),
    pytest.param(
        "secret", [{"X-Heylook-Admin-Token": "wrong"}], 401, id="wrong_header_rejects"
    ),
    pytest.param("", [{}], None, id="empty_env_var_is_no_op"),
    pytest.param(
        "secret",
        [{"x-heylook-admin-token": "secret"}, {"X-HEYLOOK-ADMIN-TOKEN": "secret"}],
        None,
        id="header_case_insensitive",
    ),
]


class TestRequireAdminToken:
    @pytest.mark.parametrize("env, header_sets, expected_status", _ADMIN_TOKEN_ROWS)
    def test_admin_token_table(
        self, monkeypatch: pytest.MonkeyPatch, env, header_sets, expected_status
    ):
        if env is None:
            monkeypatch.delenv("HEYLOOK_ADMIN_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HEYLOOK_ADMIN_TOKEN", env)
        from heylook_llm.auth import require_admin_token

        for headers in header_sets:
            if expected_status is None:
                require_admin_token(_FakeRequest(headers))
            else:
                with pytest.raises(HTTPException) as excinfo:
                    require_admin_token(_FakeRequest(headers))
                assert excinfo.value.status_code == expected_status
