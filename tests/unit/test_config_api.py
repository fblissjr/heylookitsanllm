# tests/unit/test_config_api.py
"""HTTP contract for /v1/admin/config (config_api.py).

Runs the router on a minimal FastAPI app with an in-memory store -- no server,
no model loads. The store is covered in test_settings_store.py and precedence
in test_settings_resolver.py; these pin the wire shapes and status-code mapping
(422 unknown key / bad value, 404 unknown reset key).
"""

from unittest.mock import call, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import db, observability
from heylook_llm.config_api import config_router


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    app.include_router(config_router)
    app.state.db = await db.get_connection(path=":memory:")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await app.state.db.close()


@pytest.mark.unit
class TestConfigEndpoints:
    @pytest.mark.asyncio
    async def test_get_returns_defaults_when_unset(self, client):
        res = await client.get("/v1/admin/config")
        assert res.status_code == 200
        body = res.json()
        assert body["effective"]["observability_level"] == "minimal"
        assert body["effective"]["observability_retention_days"] == 30
        assert body["stored"] == {}

    @pytest.mark.asyncio
    async def test_put_persists_and_returns_effective(self, client):
        res = await client.put("/v1/admin/config", json={"observability_level": "debug"})
        assert res.status_code == 200
        body = res.json()
        assert body["effective"]["observability_level"] == "debug"
        assert body["stored"] == {"observability_level": "debug"}
        # persisted across requests
        res2 = await client.get("/v1/admin/config")
        assert res2.json()["effective"]["observability_level"] == "debug"

    @pytest.mark.asyncio
    async def test_put_unknown_key_rejected(self, client):
        res = await client.put("/v1/admin/config", json={"bogus": 1})
        assert res.status_code == 422
        # nothing persisted
        assert (await client.get("/v1/admin/config")).json()["stored"] == {}

    @pytest.mark.asyncio
    async def test_put_invalid_value_rejected(self, client):
        res = await client.put("/v1/admin/config", json={"observability_retention_days": -1})
        assert res.status_code == 422
        assert (await client.get("/v1/admin/config")).json()["stored"] == {}

    @pytest.mark.asyncio
    async def test_reset_restores_default(self, client):
        await client.put("/v1/admin/config", json={"observability_level": "off"})
        res = await client.delete("/v1/admin/config/observability_level")
        assert res.status_code == 200
        assert res.json()["effective"]["observability_level"] == "minimal"
        assert res.json()["stored"] == {}

    @pytest.mark.asyncio
    async def test_reset_unknown_key_404(self, client):
        res = await client.delete("/v1/admin/config/bogus")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_put_refreshes_spine_level_immediately(self, client):
        # a level change must take effect in the in-process spine cache without
        # a restart (config_api calls apply_runtime_settings after persist)
        await client.put("/v1/admin/config", json={"observability_level": "debug"})
        assert observability.current_level() == "debug"
        await client.put("/v1/admin/config", json={"observability_level": "off"})
        assert observability.current_level() == "off"

    @pytest.mark.asyncio
    async def test_reset_reapplies_settings_immediately(self, client):
        # DELETE must re-apply like PUT does -- a reset that only takes effect
        # after restart silently diverges from what GET reports as effective
        await client.put("/v1/admin/config", json={"observability_level": "debug"})
        assert observability.current_level() == "debug"
        await client.delete("/v1/admin/config/observability_level")
        assert observability.current_level() == "minimal"


@pytest.mark.unit
class TestMlxCacheLimit:
    @pytest.mark.asyncio
    async def test_default_is_none(self, client):
        body = (await client.get("/v1/admin/config")).json()
        assert body["effective"]["mlx_cache_limit_gb"] is None

    @pytest.mark.asyncio
    async def test_nonpositive_rejected(self, client):
        for bad in (0, -2):
            res = await client.put("/v1/admin/config", json={"mlx_cache_limit_gb": bad})
            assert res.status_code == 422
        assert (await client.get("/v1/admin/config")).json()["stored"] == {}

    @pytest.mark.asyncio
    async def test_put_applies_limit_in_bytes(self, client):
        with patch("mlx.core.set_cache_limit", return_value=999) as set_limit:
            res = await client.put("/v1/admin/config", json={"mlx_cache_limit_gb": 1.5})
        assert res.status_code == 200
        assert res.json()["effective"]["mlx_cache_limit_gb"] == 1.5
        set_limit.assert_called_once_with(int(1.5 * 1024**3))

    @pytest.mark.asyncio
    async def test_reset_restores_captured_mlx_default(self, client):
        # first cap captures MLX's previous (default) limit from the return
        # value; clearing the override restores exactly that value
        with patch("mlx.core.set_cache_limit", return_value=999) as set_limit:
            await client.put("/v1/admin/config", json={"mlx_cache_limit_gb": 2})
            await client.delete("/v1/admin/config/mlx_cache_limit_gb")
        assert set_limit.call_args_list[-1] == call(999)

    @pytest.mark.asyncio
    async def test_mlx_failure_does_not_break_config_api(self, client):
        # best-effort like observability.configure: an MLX error must never
        # fail the settings write itself
        with patch("mlx.core.set_cache_limit", side_effect=RuntimeError("no metal")):
            res = await client.put("/v1/admin/config", json={"mlx_cache_limit_gb": 4})
        assert res.status_code == 200
        assert res.json()["stored"] == {"mlx_cache_limit_gb": 4.0}

