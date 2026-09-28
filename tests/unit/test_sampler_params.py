"""A stored `params` bag is validated where it is written (v2.0.184).

Every route that writes one -- conversation, notebook and preset, create and
update -- takes `schema.sampler_params.SamplerParams`. Before, any JSON object
was stored: an unknown key sat there unused, and a wrong-typed value was
accepted and then failed every generation on that document.
"""

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import db
from heylook_llm.conversation_api import conversation_router
from heylook_llm.notebook_api import notebook_router
from heylook_llm.preset_api import preset_router


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    for router in (conversation_router, notebook_router, preset_router):
        app.include_router(router)
    app.state.db = await db.get_connection(path=":memory:")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await app.state.db.close()


async def _write(client, route: str, params: dict):
    """One params write through `route`; returns (status, body)."""
    if route == "preset-create":
        r = await client.post("/v1/presets", json={"name": repr(params), "params": params})
    elif route == "preset-update":
        pid = (await client.post("/v1/presets", json={"name": "p" + repr(params)})).json()["id"]
        r = await client.put(f"/v1/presets/{pid}", json={"params": params})
    elif route == "conversation-create":
        r = await client.post("/v1/conversations", json={"params": params})
    elif route == "conversation-update":
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        r = await client.put(f"/v1/conversations/{cid}", json={"params": params})
    elif route == "notebook-create":
        r = await client.post("/v1/notebooks", json={"title": "t", "content": "", "params": params})
    else:
        nid = (await client.post("/v1/notebooks", json={"title": "t", "content": ""})).json()["id"]
        r = await client.put(f"/v1/notebooks/{nid}", json={"params": params})
    return r.status_code, r.json()


ROUTES = ["preset-create", "preset-update", "conversation-create",
          "conversation-update", "notebook-create", "notebook-update"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES)
async def test_every_params_write_validates_the_bag(client, route):
    for bad, named in [({"temperature": 0.7, "made_up": 1}, "made_up"),
                       ({"top_p": "high"}, "top_p"),
                       ({"top_p": 1.5}, "top_p"),
                       ({"reasoning_effort": "two words"}, "reasoning_effort")]:
        status, body = await _write(client, route, bad)
        assert status == 422, f"{route} stored {bad}: {body}"
        assert named in str(body["detail"]), f"{route}: the 422 does not name {named!r}: {body}"

    status, body = await _write(client, route, {
        "temperature": 1, "enable_thinking": True, "reasoning_effort": "low",
        "thinking_budget_tokens": 3, "seed": None})
    assert status in (200, 201), body
    if "params" in body:  # a notebook PUT answers without the document
        assert body["params"] == {"temperature": 1.0, "enable_thinking": True,
                                  "reasoning_effort": "low", "thinking_budget_tokens": 3}
