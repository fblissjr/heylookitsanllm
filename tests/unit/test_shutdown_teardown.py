# tests/unit/test_shutdown_teardown.py
#
# The server must not outlive its children. The gguf provider spawns
# llama-server with start_new_session=True (its own process group), so the
# terminal's Ctrl-C -- SIGINT to the FOREGROUND process group -- never reaches
# it, and nothing else reaps it. Before this, every heylook exit leaked a
# multi-GB llama-server: observed 2026-07-26, two orphans holding ~22GB with
# PPID 1, long after their parent was gone.
#
# Three layers, tested here and in the provider/router suites:
#   1. lifespan shutdown -> router.unload_all()      (this file)
#   2. router.unload_all() -> every provider.unload() (test_router.py)
#   3. atexit -> kill any still-registered group      (test_llama_server_provider.py)
# Layer 1 covers graceful exit INCLUDING Ctrl-C (uvicorn traps SIGINT and runs
# the ASGI lifespan shutdown). Layer 3 is the backstop for paths that skip it.

import asyncio
import types

import pytest


class _FakeRouter:
    def __init__(self):
        self.app_config = types.SimpleNamespace(
            observability_level="off", idle_unload_seconds=0,
        )
        self.unload_all_calls = 0
        self.memory_manager = None

    def unload_all(self):
        self.unload_all_calls += 1


class _FakeDB:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.fixture
def driven_lifespan(monkeypatch):
    """Run api.lifespan against a fake app, stubbing everything it touches
    except the teardown under test."""
    from heylook_llm import api as api_mod

    fake_db = _FakeDB()

    async def _fake_get_connection(*a, **kw):
        return fake_db

    async def _fake_apply_runtime_settings(_db):
        return types.SimpleNamespace(
            observability_level="off", observability_retention_days=7,
        )

    async def _idle_loop(_app):
        await asyncio.Event().wait()

    monkeypatch.setattr("heylook_llm.db.get_connection", _fake_get_connection)
    monkeypatch.setattr(
        "heylook_llm.config_api.apply_runtime_settings", _fake_apply_runtime_settings
    )
    monkeypatch.setattr(
        "heylook_llm.config_api.observability_log_dir", lambda: "logs", raising=False
    )
    monkeypatch.setattr(api_mod, "_resource_snapshot_loop", _idle_loop)

    class _FakeMemoryManager:
        def __init__(self, **kw):
            pass

        def log_startup_info(self):
            pass

    monkeypatch.setattr("heylook_llm.memory.MemoryManager", _FakeMemoryManager)

    app = types.SimpleNamespace(state=types.SimpleNamespace())
    app.state.router_instance = _FakeRouter()
    return api_mod.lifespan, app, fake_db


def test_lifespan_shutdown_unloads_all_models(driven_lifespan):
    """Claim: leaving providers loaded at shutdown strands a llama-server
    subprocess per gguf model. Delete this and Ctrl-C leaks them again --
    the exact 2026-07-26 orphan bug.
    """
    lifespan, app, _ = driven_lifespan

    async def drive():
        async with lifespan(app):
            assert app.state.router_instance.unload_all_calls == 0  # not before

    asyncio.run(drive())
    assert app.state.router_instance.unload_all_calls == 1


def test_lifespan_shutdown_unloads_even_if_db_close_fails(driven_lifespan):
    """Claim: teardown order must not let one failure strand the subprocesses.
    The DB close used to run first with nothing after it guarded.
    """
    lifespan, app, fake_db = driven_lifespan

    async def _boom():
        raise RuntimeError("db close exploded")

    fake_db.close = _boom

    async def drive():
        async with lifespan(app):
            pass

    asyncio.run(drive())
    assert app.state.router_instance.unload_all_calls == 1
