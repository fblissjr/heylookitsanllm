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
#
# Driven as a system: the real app and its real lifespan, a real ModelRouter
# over a real config file, a model loaded through POST /v1/models/{id}/load,
# and the real DuckDB store in a temp dir. Only the engine is a stand-in
# (MockProvider, whose unload is the resource release a real provider does).

import contextlib
import logging
import textwrap

import pytest
from starlette.testclient import TestClient

from _mock_provider import MockProvider

_TOML = textwrap.dedent("""
    max_loaded_models = 1
    allowed_hosts = ["testserver"]

    [[models]]
    id = "m"
    provider = "mlx"
    enabled = true
    config = { model_path = "/fake/m" }
""").strip()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The real app wired to a real router; app state restored afterwards so
    the contract suite's session app is untouched."""
    from heylook_llm.api import app
    from heylook_llm.router import ModelRouter

    monkeypatch.setenv("HEYLOOK_DB_PATH", str(tmp_path / "store.duckdb"))
    monkeypatch.setattr("heylook_llm.router.MLXProvider", MockProvider)
    config = tmp_path / "heylook.toml"
    config.write_text(_TOML)
    router = ModelRouter(config_path=str(config), log_level=logging.ERROR,
                         initial_model_id=None)
    # The lifespan warms row facts on a background thread; over a stand-in
    # engine there is nothing to warm and a thread outliving the test is the
    # teardown-crash class tests/README warns about.
    router.warm_model_facts = lambda: None

    for name in ("router_instance", "db", "memory_manager"):
        monkeypatch.setattr(app.state, name, getattr(app.state, name, None), raising=False)
    app.state.router_instance = router
    return app, router


def _serve_and_load(app):
    client = TestClient(app, raise_server_exceptions=False)
    with client:
        r = client.post("/v1/models/m/load")
        assert r.status_code == 200, r.text


def test_lifespan_shutdown_unloads_all_models(served):
    """Claim: leaving providers loaded at shutdown strands a llama-server
    subprocess per gguf model. Delete this and Ctrl-C leaks them again --
    the exact 2026-07-26 orphan bug.
    """
    app, router = served
    _serve_and_load(app)

    assert router.get_loaded_models() == {}
    assert router.get_model_status("m")["loaded"] is False


def test_lifespan_shutdown_unloads_even_if_db_close_fails(served, monkeypatch):
    """Claim: teardown order must not let one failure strand the subprocesses.
    The DB close used to run first with nothing after it guarded.
    """
    from heylook_llm import db

    real_close = db.Store.close

    async def close_then_explode(self):
        await real_close(self)
        raise RuntimeError("db close exploded")

    monkeypatch.setattr(db.Store, "close", close_then_explode)
    app, router = served
    with contextlib.suppress(Exception):   # the outcome is the assertion below
        _serve_and_load(app)

    assert router.get_loaded_models() == {}
