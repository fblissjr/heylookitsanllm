# tests/unit/test_preset_api.py
"""HTTP contract for /v1/presets (preset_api.py).

Runs the router on a minimal FastAPI app with an in-memory store -- no
server, no model loads. The db layer is covered in test_preset_store.py;
these tests pin the status-code mapping (409 name collision, 400 bad
fields, 404 unknown id) and the wire shapes.
"""

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import db
from heylook_llm.preset_api import preset_router


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    app.include_router(preset_router)
    app.state.db = await db.get_connection(path=":memory:")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await app.state.db.close()


@pytest.mark.unit
class TestPresetEndpoints:
    @pytest.mark.asyncio
    async def test_create_and_list(self, client):
        res = await client.post("/v1/presets", json={
            "name": "pirate",
            "system_prompt": "Talk like a pirate.",
            "params": {"temperature": 1.2},
        })
        assert res.status_code == 201
        created = res.json()
        assert created["name"] == "pirate"
        assert created["params"] == {"temperature": 1.2}

        res = await client.get("/v1/presets")
        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert body["presets"][0]["id"] == created["id"]

    @pytest.mark.asyncio
    async def test_duplicate_name_is_409(self, client):
        await client.post("/v1/presets", json={"name": "dup"})
        res = await client.post("/v1/presets", json={"name": "dup"})
        assert res.status_code == 409

    @pytest.mark.asyncio
    async def test_blank_name_is_400(self, client):
        res = await client.post("/v1/presets", json={"name": "  "})
        assert res.status_code == 400

    @pytest.mark.asyncio
    async def test_update_partial(self, client):
        created = (await client.post("/v1/presets", json={
            "name": "v1", "system_prompt": "old", "params": {"top_k": 1},
        })).json()
        res = await client.put(f"/v1/presets/{created['id']}", json={"system_prompt": "new"})
        assert res.status_code == 200
        updated = res.json()
        assert updated["system_prompt"] == "new"
        assert updated["name"] == "v1"
        assert updated["params"] == {"top_k": 1}

    @pytest.mark.asyncio
    async def test_update_can_clear_system_prompt(self, client):
        created = (await client.post("/v1/presets", json={
            "name": "clearable", "system_prompt": "x",
        })).json()
        res = await client.put(f"/v1/presets/{created['id']}", json={"system_prompt": None})
        assert res.status_code == 200
        assert res.json()["system_prompt"] is None

    @pytest.mark.asyncio
    async def test_update_rename_collision_is_409(self, client):
        await client.post("/v1/presets", json={"name": "a"})
        b = (await client.post("/v1/presets", json={"name": "b"})).json()
        res = await client.put(f"/v1/presets/{b['id']}", json={"name": "a"})
        assert res.status_code == 409

    @pytest.mark.asyncio
    async def test_update_no_fields_is_400(self, client):
        created = (await client.post("/v1/presets", json={"name": "nofields"})).json()
        res = await client.put(f"/v1/presets/{created['id']}", json={})
        assert res.status_code == 400

    @pytest.mark.asyncio
    async def test_update_unknown_id_is_404(self, client):
        res = await client.put("/v1/presets/ghost", json={"name": "x"})
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_delete(self, client):
        created = (await client.post("/v1/presets", json={"name": "gone"})).json()
        res = await client.delete(f"/v1/presets/{created['id']}")
        assert res.status_code == 200
        res = await client.delete(f"/v1/presets/{created['id']}")
        assert res.status_code == 404
