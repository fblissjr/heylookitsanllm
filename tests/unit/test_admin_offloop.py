# tests/unit/test_admin_offloop.py
"""Mutating admin routes must not run discovery ON the event loop.

Claim (what breaks if this file is deleted): a models-page edit -- a config
PATCH, an enable/disable toggle, a bulk sampler set, a delete, an explicit
scan, a reload -- freezes every in-flight SSE generation stream for as long as
the ``[scan]`` walk takes. The walk is a recursive filesystem traversal plus
GGUF header reads over whatever the operator put in ``[scan].folders``, so its
duration is unbounded in principle, and one mutation pays for it TWICE:
``ModelService`` runs discovery to materialize an entry, then the route's
``reload_config`` runs it again.

The oracle is a HEARTBEAT, not the shape of the handler: a coroutine that
never awaits and a ``def`` handler are indistinguishable from the outside
except that the first one starves other tasks. Both plausible fixes (``def``
in FastAPI's threadpool, ``async def`` + ``asyncio.to_thread``) pass this;
only blocking fails it. ``test_the_heartbeat_can_actually_starve`` is the
control -- it mounts a route that DOES block and asserts the harness sees it,
so a green run here cannot be the heartbeat quietly measuring nothing.

Same reasoning as ``MemoryManager._maybe_rescan_models`` (memory.py), which
pushes the identical scan to an executor for the identical reason.
"""

import asyncio
import time

import pytest
import tomli_w
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm.admin_api import admin_ops_router, admin_router, scan_import_router
from heylook_llm.model_service import ModelService

# One stubbed scan stalls its caller this long. Sized to dwarf heartbeat
# jitter without making the suite wait: a free loop delivers ~40 ticks per
# blocked interval, a stalled one delivers 0-2.
SCAN_BLOCK_S = 0.2
TICK_S = 0.005
MIN_TICKS = 8


class _FakeRouter:
    """Enough router for the admin routes -- no MLX, no providers.

    ``reload_config`` re-runs discovery exactly like the real one
    (ModelRouter._load_config -> _with_discovered), because that second scan is
    half of what these routes pay and a fake that skipped it would hide it.
    """

    def __init__(self, config_path):
        self.config_path = str(config_path)
        self.reloads = 0
        self.app_config = None
        self.reload_config()

    def reload_config(self):
        import tomllib

        from heylook_llm.config import AppConfig
        from heylook_llm.model_registry import discover, merge_discovered

        self.reloads += 1
        with open(self.config_path, "rb") as f:
            data = tomllib.load(f)
        self.app_config = AppConfig(**merge_discovered(data, discover(data)))

    def get_loaded_models(self):
        return {}

    def unload_model(self, model_id):
        return False

    def clear_cache(self):
        pass

    def stale_reload_fields(self, model_id):
        return []

    def list_available_models(self):
        return [m.id for m in self.app_config.models]


@pytest.fixture
def admin_app(tmp_path, monkeypatch):
    """Admin routers over a real ModelService whose scan is slow-by-stub."""
    store = tmp_path / "store"
    store.mkdir()
    blob = store / "found.gguf"
    blob.write_text("x")
    # OUTSIDE the scan folder: deleting this entry is the case that runs
    # discovery for the disabled-override guard and then succeeds.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    off = elsewhere / "off.gguf"
    off.write_text("x")

    cfg = tmp_path / "models.toml"
    cfg.write_text(tomli_w.dumps({
        "default_model": "none",
        "models": [{
            "id": "written-off",
            "provider": "gguf",
            "enabled": False,
            "config": {"model_path": str(off)},
        }],
        "scan": {"folders": [str(store)]},
    }))

    import heylook_llm.model_importer as mi

    scans: list[str] = []

    def slow_scan(self, path):
        scans.append(str(path))
        time.sleep(SCAN_BLOCK_S)
        if str(path) != str(store):
            return []
        return [{"id": "found", "provider": "gguf", "enabled": True,
                 "config": {"model_path": str(blob)}}]

    monkeypatch.setattr(mi.ModelImporter, "scan_directory", slow_scan)

    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(scan_import_router)
    app.include_router(admin_ops_router)
    app.state.model_service = ModelService(str(cfg))
    app.state.router_instance = _FakeRouter(cfg)
    app.state.scan_store = str(store)
    app.state.scan_blob = str(blob)
    scans.clear()  # the fake router's construction reload already scanned
    app.state.scans = scans
    return app


def _case(app, name):
    """(method, url, request kwargs) for one mutating admin route."""
    store, blob = app.state.scan_store, app.state.scan_blob
    return {
        "patch": ("PATCH", "/v1/admin/models/found",
                  {"json": {"config": {"chat_template_path": blob}}}),
        "toggle": ("POST", "/v1/admin/models/found/toggle", {}),
        "delete": ("DELETE", "/v1/admin/models/written-off", {}),
        "bulk_sampler": ("POST", "/v1/admin/models/bulk-default-sampler",
                         {"json": {"model_ids": ["found"],
                                   "sampler": "balanced"}}),
        "scan": ("POST", "/v1/admin/models/scan",
                 {"json": {"paths": [store], "scan_hf_cache": False}}),
        "reload": ("POST", "/v1/admin/reload", {}),
    }[name]


async def _ticks_during(client, method, url, **kwargs):
    """Run one request; count how many times an idle task got to run.

    Returns (response, ticks). A handler that blocks the loop starves the
    heartbeat no matter how long the request takes.
    """
    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(TICK_S)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(TICK_S * 2)  # heartbeat is definitely running
    baseline = ticks
    try:
        response = await client.request(method, url, **kwargs)
    finally:
        stop.set()
        await beat
    return response, ticks - baseline


@pytest.mark.unit
class TestMutatingAdminRoutesStayOffTheEventLoop:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", [
        "patch", "toggle", "delete", "bulk_sampler", "scan", "reload",
    ])
    async def test_route_does_not_freeze_the_loop(self, admin_app, case):
        method, url, kwargs = _case(admin_app, case)
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            response, ticks = await _ticks_during(client, method, url, **kwargs)

        assert response.status_code == 200, response.text
        # Without this the test goes vacuous the day discovery gets cached or
        # skipped on this path: nothing slow left to run on the loop, so the
        # heartbeat sails through and the async-def regression stops being
        # guarded rather than stopping being possible.
        assert admin_app.state.scans, f"{method} {url} never ran a scan"
        assert ticks >= MIN_TICKS, (
            f"{method} {url} let only {ticks} heartbeat ticks through -- the "
            f"discovery scan ran on the event loop, so every in-flight SSE "
            f"stream stalled with it"
        )

    @pytest.mark.asyncio
    async def test_the_heartbeat_can_actually_starve(self, admin_app):
        """Control: prove the oracle above is capable of failing.

        Without this, a heartbeat that ticks for an unrelated reason (or a
        stub that stopped being slow) would make every assertion above pass
        while measuring nothing.
        """
        @admin_app.get("/blocking-control")
        async def _blocking_control():  # the shape this test exists to reject
            time.sleep(SCAN_BLOCK_S * 2)
            return {"ok": True}

        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            response, ticks = await _ticks_during(client, "GET", "/blocking-control")

        assert response.status_code == 200
        assert ticks < MIN_TICKS, (
            f"a handler that sleeps {SCAN_BLOCK_S * 2}s on the loop let "
            f"{ticks} ticks through -- the heartbeat is not measuring "
            f"event-loop availability"
        )


@pytest.mark.unit
class TestDeleteRefusalIsAConflict:
    """The disabled-override guard's message has to REACH the caller.

    ModelService.remove_config raises ValueError to refuse deleting a disabled
    entry that discovery still finds (deleting it would silently re-enable the
    model). Uncaught, that is a 500 whose body is "Internal Server Error" --
    the explanation, which is the entire value of the guard, never arrives.
    """

    @pytest.mark.asyncio
    async def test_deleting_a_disabled_override_returns_409(self, admin_app):
        # Give the discovered model an entry, disabled -- now discovery still
        # finds the file the entry turns off.
        service = admin_app.state.model_service
        service.toggle_enabled("found")

        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            response = await client.delete("/v1/admin/models/found")

        assert response.status_code == 409, response.text
        assert "re-enable" in response.json()["detail"]
