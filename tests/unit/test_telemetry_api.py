# tests/unit/test_telemetry_api.py
"""HTTP contract for POST /v1/telemetry/events (telemetry_api.py).

Frontend client events are appended to the observability events stream with
source='frontend-v3', level-gated, batch- and field-size-bounded.
"""

import orjson
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import observability as obs
from heylook_llm.telemetry_api import telemetry_router


@pytest_asyncio.fixture
async def client(tmp_path):
    obs.configure(level="debug", log_dir=tmp_path)
    app = FastAPI()
    app.include_router(telemetry_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, tmp_path


def _events(tmp_path):
    p = tmp_path / "events.jsonl"
    return [orjson.loads(line) for line in p.read_bytes().splitlines() if line] if p.exists() else []


@pytest.mark.unit
class TestIngest:
    @pytest.mark.asyncio
    async def test_batch_appended_with_frontend_source(self, client):
        c, tmp = client
        res = await c.post("/v1/telemetry/events", json={"events": [
            {"type": "client_error", "level": "error", "message": "boom", "route": "/chat"},
            {"type": "stream_stall", "level": "warn", "gap_ms": 1200},
        ]})
        assert res.status_code == 200
        assert res.json()["accepted"] == 2
        recs = _events(tmp)
        assert {r["type"] for r in recs} == {"client_error", "stream_stall"}
        assert all(r["source"] == "frontend-v3" for r in recs)
        err = next(r for r in recs if r["type"] == "client_error")
        assert err["message"] == "boom"
        assert err["route"] == "/chat"

    @pytest.mark.asyncio
    async def test_events_without_type_skipped(self, client):
        c, tmp = client
        res = await c.post("/v1/telemetry/events", json={"events": [
            {"level": "info", "nope": 1},   # no type -> skipped
            {"type": "ok", "a": 1},
        ]})
        assert res.json()["accepted"] == 1
        assert [r["type"] for r in _events(tmp)] == ["ok"]

    @pytest.mark.asyncio
    async def test_non_list_body_rejected_gracefully(self, client):
        c, _ = client
        res = await c.post("/v1/telemetry/events", json={"events": "nope"})
        assert res.status_code == 200
        assert res.json()["accepted"] == 0

    @pytest.mark.asyncio
    async def test_oversized_field_truncated(self, client):
        c, tmp = client
        await c.post("/v1/telemetry/events", json={"events": [
            {"type": "e", "blob": "x" * 5000},
        ]})
        (rec,) = _events(tmp)
        assert len(rec["blob"]) == 2000  # bounded backstop

    @pytest.mark.asyncio
    async def test_off_level_suppresses(self, tmp_path):
        obs.configure(level="off", log_dir=tmp_path)
        app = FastAPI()
        app.include_router(telemetry_router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            res = await c.post("/v1/telemetry/events", json={"events": [{"type": "e"}]})
        # endpoint still returns accepted count, but nothing is written at off
        assert res.json()["accepted"] == 1
        assert not (tmp_path / "events.jsonl").exists()
