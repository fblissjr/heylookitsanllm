# tests/unit/test_config_api.py
"""HTTP contract for /v1/admin/config (config_api.py).

Runs the router on a minimal FastAPI app over a temporary heylook.toml -- no
server, no model loads. Precedence is in test_settings_resolver.py; these pin
the wire shapes, the status-code mapping (422 unknown key / bad value, 404
unknown reset key), that a write lands in the file's [settings] table, and
that a read-only instance holds its writes in memory instead.
"""

import tomllib
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import config_api, observability
from heylook_llm.config_api import config_router


@pytest.fixture
def config_file(tmp_path):
    f = tmp_path / "heylook.toml"
    f.write_text('# the owner\'s note\n[scan]\nfolders = ["/models"]\n')
    return f


@pytest_asyncio.fixture
async def client(config_file, monkeypatch):
    monkeypatch.setattr(config_api, "_MEMORY_ONLY", {})
    app = FastAPI()
    app.include_router(config_router)
    app.state.router_instance = SimpleNamespace(config_path=str(config_file))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.unit
class TestConfigEndpoints:
    @pytest.mark.asyncio
    async def test_get_returns_defaults_when_unset(self, client):
        res = await client.get("/v1/admin/config")
        assert res.status_code == 200
        body = res.json()
        assert body["effective"]["observability_level"] == "off"
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
        await client.put("/v1/admin/config", json={"observability_level": "debug"})
        res = await client.delete("/v1/admin/config/observability_level")
        assert res.status_code == 200
        assert res.json()["effective"]["observability_level"] == "off"
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
        assert observability.current_level() == "off"


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



@pytest.mark.unit
class TestTheFileHoldsTheSettings:
    @pytest.mark.asyncio
    async def test_a_write_lands_in_the_config_files_settings_table(self, client, config_file):
        await client.put("/v1/admin/config", json={"observability_level": "debug"})
        data = tomllib.loads(config_file.read_text())
        assert data["settings"] == {"observability_level": "debug"}
        assert data["scan"]["folders"] == ["/models"]              # the rest untouched
        await client.delete("/v1/admin/config/observability_level")
        assert "settings" not in tomllib.loads(config_file.read_text())

    @pytest.mark.asyncio
    async def test_a_read_only_instance_holds_its_writes_in_memory(self, client, config_file, monkeypatch):
        """A dev server or loop run shares the daily server's file, and still
        needs its own logging level (option (a), 2026-09-24)."""
        from heylook_llm.model_registry import READONLY_ENV
        monkeypatch.setenv(READONLY_ENV, "1")
        before = config_file.read_text()
        res = await client.put("/v1/admin/config", json={"observability_level": "debug"})
        assert res.status_code == 200
        assert res.json()["effective"]["observability_level"] == "debug"
        assert res.json()["memory_only"] == ["observability_level"]
        assert config_file.read_text() == before
        assert observability.current_level() == "debug"
        await client.put("/v1/admin/config", json={"observability_level": "off"})
