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
from unittest.mock import patch

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
        assert body["effective"]["mlx_cache_limit_gb"] is None
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

    # An unknown key, a bad value, or a non-positive mlx_cache_limit_gb is a
    # 422 and nothing is persisted. Each row is a list of bodies, each refused.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bodies", [
        pytest.param([{"bogus": 1}], id="put_unknown_key_rejected"),
        pytest.param([{"observability_retention_days": -1}], id="put_invalid_value_rejected"),
        pytest.param([{"mlx_cache_limit_gb": 0}, {"mlx_cache_limit_gb": -2}],
                     id="nonpositive_mlx_cache_limit_rejected"),
    ])
    async def test_invalid_put_is_422_and_persists_nothing(self, client, bodies):
        for body in bodies:
            res = await client.put("/v1/admin/config", json=body)
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

    # A level change takes effect in the in-process spine cache without a
    # restart (config_api calls apply_runtime_settings after persist). DELETE
    # must re-apply like PUT does -- a reset that only takes effect after
    # restart silently diverges from what GET reports as effective.
    # Rows: the second step, after PUT debug, and the level it must leave.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("second_step", [
        pytest.param("put_off", id="put_refreshes_spine_level_immediately"),
        pytest.param("delete", id="reset_reapplies_settings_immediately"),
    ])
    async def test_settings_reapply_immediately(self, client, second_step):
        await client.put("/v1/admin/config", json={"observability_level": "debug"})
        assert observability.current_level() == "debug"
        if second_step == "put_off":
            await client.put("/v1/admin/config", json={"observability_level": "off"})
        else:
            await client.delete("/v1/admin/config/observability_level")
        assert observability.current_level() == "off"


@pytest.mark.unit
class TestMlxCacheLimit:
    @pytest.mark.asyncio
    async def test_the_allocator_holds_the_cap_then_gets_its_own_default_back(
            self, client, monkeypatch):
        """The MLX allocator's limit is the observable: a fake allocator that
        holds a limit and, like mx.set_cache_limit, answers with the previous
        one. Two caps then a clear: the clear must give back the allocator's
        own default, not the first cap."""
        default = 7_000_000_000
        state = {"limit": default}

        def set_cache_limit(n):
            prev, state["limit"] = state["limit"], n
            return prev

        monkeypatch.setattr(config_api, "_mlx_default_cache_limit", None)
        with patch("mlx.core.set_cache_limit", side_effect=set_cache_limit):
            res = await client.put("/v1/admin/config", json={"mlx_cache_limit_gb": 1.5})
            assert res.status_code == 200
            assert res.json()["effective"]["mlx_cache_limit_gb"] == 1.5
            assert state["limit"] == 1_610_612_736  # 1.5 GiB
            await client.put("/v1/admin/config", json={"mlx_cache_limit_gb": 2})
            assert state["limit"] == 2_147_483_648  # 2 GiB
            await client.delete("/v1/admin/config/mlx_cache_limit_gb")
        assert state["limit"] == default

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
