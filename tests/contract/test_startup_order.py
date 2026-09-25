# tests/contract/test_startup_order.py
"""The lifespan's startup order is the behaviour: telemetry is configured
BEFORE anything it should record happens.

Twice the most expensive or most informative event of a run went unrecorded
because it ran ahead of the settings: the startup record (fixed 2026-08-19)
and the `--model-id` pre-warm, which loaded inside ModelRouter's constructor
before the memory manager or the observability level existed (moved into the
lifespan in v2.0.166). Order is the claim here, so order is what is asserted,
through the real lifespan.
"""
from unittest import mock

from starlette.testclient import TestClient


def test_settings_then_startup_record_then_prewarm(app):
    from heylook_llm import config_api
    from heylook_llm.memory import MemoryManager

    seen: list[str] = []
    real_apply = config_api.apply_runtime_settings
    router = app.state.router_instance

    def apply(a):
        seen.append("settings")
        return real_apply(a)

    def prewarm():
        # The load's telemetry needs the memory manager attached by now.
        seen.append("prewarm" if router.memory_manager is not None else "prewarm-without-telemetry")

    with mock.patch.object(config_api, "apply_runtime_settings", apply), \
            mock.patch.object(MemoryManager, "log_startup_info", lambda self: seen.append("startup-record")), \
            mock.patch.object(router, "prewarm_startup_model", prewarm):
        with TestClient(app, raise_server_exceptions=False):
            pass
    assert seen == ["settings", "startup-record", "prewarm"]
