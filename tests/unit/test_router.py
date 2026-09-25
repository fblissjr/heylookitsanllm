"""Router tests.

Fixtures use TOML because `router.py:_load_config` is TOML-only. The tests
exercise real LRU eviction, hot-swap, and max_loaded_models behavior with
a MockProvider, so they're structural tests of the router itself and worth
keeping.
"""

import logging
import os
import tempfile
import textwrap
import unittest

import pytest
from unittest.mock import patch

from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider


# The model dirs must really EXIST. _load_config audits every configured path
# and warns about one that has gone missing (a models.toml entry outliving a
# directory rename), and `test_healthy_config_logs_no_warning` asserts the
# happy path logs NOTHING at WARNING. Real empty dirs keep that assertion at
# full strength instead of narrowing it to step around a legitimate warning.
# Directories, not files: an MLX model_path names a checkpoint DIR.
_MODEL_ROOT = tempfile.mkdtemp(prefix="heylook-router-tests-")
for _name in ("model1", "model2", "model3"):
    os.makedirs(os.path.join(_MODEL_ROOT, _name), exist_ok=True)


_BASE_TOML = textwrap.dedent("""
    default_model = "{default_model}"
    max_loaded_models = {max_loaded_models}

    [[models]]
    id = "model1-mlx"
    provider = "mlx"
    config = {{ model_path = "{model_root}/model1" }}

    [[models]]
    id = "model2-llama"
    provider = "mlx"
    config = {{ model_path = "{model_root}/model2" }}

    [[models]]
    id = "model3-mlx"
    provider = "mlx"
    config = {{ model_path = "{model_root}/model3" }}
""").strip()


def _render_config(
    *,
    default_model: str = "model1-mlx",
    max_loaded_models: int = 2,
) -> str:
    return _BASE_TOML.format(
        default_model=default_model,
        max_loaded_models=max_loaded_models,
        model_root=_MODEL_ROOT,
    )


@patch('heylook_llm.router.MLXProvider', new=MockProvider)
class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.temp_config_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.toml'
        )
        self.temp_config_file.write(_render_config())
        self.temp_config_file.close()
        self.config_path = self.temp_config_file.name

    def tearDown(self):
        os.unlink(self.config_path)

    def test_initialization(self):
        """Router inits cleanly with no models and no default."""
        with open(self.config_path, "w") as f:
            f.write('default_model = ""\nmodels = []\n')
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        self.assertEqual(router.list_available_models(), [])
        self.assertEqual(len(router.providers), 0)

    def test_no_startup_preload_without_explicit_request(self):
        """Nothing loads at startup unless `--model-id` asks for it.

        Claim: startup preload is opt-in. Delete this and the server can go
        back to pinning a multi-GB model into RAM on every boot.
        """
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        self.assertEqual(len(router.providers), 0)

    def test_explicit_initial_model_is_preloaded(self):
        """`--model-id` is the one thing that still pre-warms at startup.

        Claim: opting out of the implicit preload must not remove the explicit
        one.
        """
        router = ModelRouter(
            config_path=self.config_path,
            log_level=logging.INFO,
            initial_model_id='model2-llama',
        )
        # Recorded at construction, loaded by the lifespan once telemetry is
        # wired (v2.0.166).
        self.assertNotIn('model2-llama', router.providers)
        router.prewarm_startup_model()
        self.assertIn('model2-llama', router.providers)

    def test_healthy_config_logs_no_warning(self):
        """The happy path stays quiet: every configured path exists."""
        with self.assertNoLogs(level=logging.WARNING):
            ModelRouter(
                config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
            )

    def test_a_request_naming_no_model_is_refused(self):
        """No default model (owner call 2026-09-25): a `default_model` key
        left in heylook.toml is ignored, and a model already loaded is not a
        fallback either. The refusal lists the ids to pick from."""
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        router.get_provider('model1-mlx')  # loaded: still not a fallback
        with self.assertRaises(ValueError) as ctx:
            router.get_provider('')
        self.assertIn('No model specified', str(ctx.exception))
        self.assertIn('model2-llama', str(ctx.exception))


@pytest.fixture
def make_router(tmp_path, monkeypatch):
    """A router over _BASE_TOML with MockProvider standing in for MLXProvider."""
    monkeypatch.setattr('heylook_llm.router.MLXProvider', MockProvider)

    def _make(**overrides):
        path = tmp_path / "heylook.toml"
        path.write_text(_render_config(**overrides))
        return ModelRouter(config_path=str(path), log_level=logging.DEBUG,
                           initial_model_id=None)
    return _make


@pytest.mark.unit
class TestProviderCache:
    # Repeated get_provider for a loaded id is a cache hit: the same object,
    # no growth, no reload.
    @pytest.mark.parametrize("accesses", [
        pytest.param(['model1-mlx', 'model1-mlx'], id="get_provider_loads_and_caches"),
        # Re-accessing already-loaded models (hot swap between two) is a hit.
        pytest.param(['model1-mlx', 'model2-llama', 'model1-mlx', 'model2-llama'],
                     id="hot_swapping"),
    ])
    def test_repeated_get_provider_is_a_cache_hit(self, make_router, accesses):
        router = make_router()
        first_seen = {}
        for model_id in accesses:
            provider = router.get_provider(model_id)
            assert provider.model_id == model_id
            assert first_seen.setdefault(model_id, provider) is provider
            assert len(router.providers) == len(first_seen)
        assert len(router.providers) == len(set(accesses))


@pytest.mark.unit
class TestLruEviction:
    # A load past max_loaded_models evicts the oldest provider and calls its
    # unload(); the rest stay resident and are not unloaded.
    @pytest.mark.parametrize("max_loaded, loads, resident, evicted", [
        # Third load evicts the oldest.
        pytest.param(2, ['model1-mlx', 'model2-llama', 'model3-mlx'],
                     {'model2-llama', 'model3-mlx'}, 'model1-mlx', id="lru_eviction"),
        # With max=1, a new load always evicts the previous.
        pytest.param(1, ['model1-mlx', 'model2-llama'],
                     {'model2-llama'}, 'model1-mlx', id="max_loaded_models_one"),
    ])
    def test_the_oldest_is_evicted_and_unloaded(self, make_router, max_loaded,
                                                loads, resident, evicted):
        router = make_router(max_loaded_models=max_loaded)
        loaded = {}
        for model_id in loads:
            loaded[model_id] = router.get_provider(model_id)
            assert model_id in router.providers
            assert len(router.providers) <= max_loaded
        assert set(router.providers) == resident
        loaded[evicted].unload.assert_called_once()
        for model_id in resident:
            loaded[model_id].unload.assert_not_called()


@pytest.mark.unit
class TestUnloadAll:
    # Server shutdown must reap every loaded model. The gguf provider's
    # "loaded" IS a running llama-server subprocess, so a provider left in
    # the cache at exit is a multi-GB orphan process.
    @pytest.mark.parametrize("loads, raiser", [
        pytest.param(['model1-mlx', 'model2-llama'], None,
                     id="unload_all_unloads_every_provider"),
        # Shutdown is best-effort and must not stop at the first raiser -- one
        # provider that throws would otherwise strand every subprocess behind
        # it in the iteration order.
        pytest.param(['model1-mlx', 'model2-llama'], 'model1-mlx',
                     id="unload_all_continues_after_a_failing_unload"),
        pytest.param([], None, id="unload_all_on_empty_router_is_safe"),
    ])
    def test_unload_all_reaps_every_provider(self, make_router, loads, raiser):
        router = make_router()
        providers = [router.get_provider(model_id) for model_id in loads]
        if raiser is not None:
            router.providers[raiser].unload.side_effect = RuntimeError("teardown exploded")

        router.unload_all()  # must not raise

        for provider in providers:
            provider.unload.assert_called_once()
        assert len(router.providers) == 0
        for provider in providers:   # let GC-time __del__ unload quietly
            provider.unload.side_effect = None


if __name__ == '__main__':
    unittest.main()


@pytest.mark.unit
class TestConfiguredPathAudit:
    """The startup report for entries whose configured paths no longer exist.

    Owner ask 2026-09-08: one summary rather than a warning per entry per
    field, and once per process rather than on every reload -- a warning that
    repeats unchanged is one people learn to scroll past.

    Every fixture here is SYNTHETIC. No path in this file comes from the
    machine it runs on: the missing ones are invented and the present one is
    pytest's tmp_path. That is deliberate, not incidental -- real entries carry
    absolute home paths, and a fixture is the easiest place for one to end up
    committed.
    """

    @staticmethod
    def _reset():
        from heylook_llm.router import ModelRouter
        ModelRouter._paths_audited = False

    def _audit(self, config_data, caplog):
        from heylook_llm.router import ModelRouter
        self._reset()
        with caplog.at_level(logging.WARNING):
            ModelRouter._audit_configured_paths(config_data)
        return caplog.text

    # Every dead entry is named once, with its field, in ONE block, once per
    # process; a live entry is not reported. "LIVE" stands for a real
    # directory (tmp_path); every other path is invented.
    @pytest.mark.parametrize("entries, audits, named, not_named, blocks", [
        pytest.param([("ghost", "/synthetic/gone/ghost")], 1,
                     ("ghost", "model_path"), (), 1,
                     id="a_dead_entry_is_named_once_with_its_field"),
        pytest.param([("fine", "LIVE")], 1, (), ("fine",), 0,
                     id="a_live_entry_is_not_reported"),
        # ONE block, not one per entry -- that is the whole change.
        pytest.param([("ghost-a", "/synthetic/gone/a"), ("ghost-b", "/synthetic/gone/b")], 1,
                     ("ghost-a", "ghost-b"), (), 1,
                     id="every_dead_entry_appears_in_one_summary"),
        # The second audit is a reload.
        pytest.param([("ghost", "/synthetic/gone/x")], 2, ("ghost",), (), 1,
                     id="it_reports_once_per_process_not_once_per_load"),
    ])
    def test_the_dead_entry_summary(self, caplog, tmp_path, entries, audits,
                                    named, not_named, blocks):
        from heylook_llm.router import ModelRouter
        live = tmp_path / "present"
        live.mkdir()
        data = {"models": [
            {"id": model_id, "config": {"model_path": str(live) if path == "LIVE" else path}}
            for model_id, path in entries
        ]}
        self._reset()
        with caplog.at_level(logging.WARNING):
            for _ in range(audits):
                ModelRouter._audit_configured_paths(data)
        text = caplog.text
        for fragment in named:
            assert fragment in text
        for fragment in not_named:
            assert fragment not in text
        assert text.count("no longer exist") == blocks

    # The advice line. Two entries can claim ONE resolved path (verified in
    # this repo: a plain entry beside a text-only twin). The report says so,
    # because 'delete it, discovery will re-add it' is false when a twin still
    # claims the path -- and that reasoning is what makes a naive prune
    # dangerous. A lone dead entry says removal costs nothing else.
    @pytest.mark.parametrize("entries, advice", [
        pytest.param([("twin-a", "/synthetic/gone/shared"), ("twin-b", "/synthetic/gone/shared")],
                     "another entry also claims this model_path",
                     id="a_twin_claiming_the_same_path_is_disclosed"),
        pytest.param([("solo", "/synthetic/gone/solo")],
                     "takes this id and nothing else",
                     id="a_lone_dead_entry_says_removal_costs_nothing_else"),
    ])
    def test_the_removal_advice(self, caplog, entries, advice):
        text = self._audit({"models": [
            {"id": model_id, "config": {"model_path": path}} for model_id, path in entries
        ]}, caplog)
        assert advice in text


# --------------------------------------------------------------------------
# One engine family resident at a time (v2.0.49).
#
# `max_loaded_models` is a COUNT, so at 2 it happily admits an MLX model and a
# gguf model together -- and that specific pair is the unsafe one. MLX holds
# weights in THIS process's Metal working set and treats the recommended size
# as HARD; a gguf model is a llama-server subprocess whose engine treats the
# same number as a debug warning and degrades into paging. `mx.set_wired_limit`
# is set once at startup and never shrinks when the subprocess takes residency,
# so MLX keeps believing it owns the whole budget. The mixed pair degrades one
# side gently and can kill the other outright.
#
# Two MLX models are NOT that case (one process, one wired limit, and the RAM
# gate sees both), which is why the rule is per PROVIDER rather than a blunt
# max_loaded_models = 1.
# --------------------------------------------------------------------------
_MIXED_TOML = textwrap.dedent("""
    max_loaded_models = 3

    [[models]]
    id = "m-mlx-a"
    provider = "mlx"
    enabled = true
    config = {{ model_path = "{root}/model1" }}

    [[models]]
    id = "m-mlx-b"
    provider = "mlx"
    enabled = true
    config = {{ model_path = "{root}/model2" }}

    [[models]]
    id = "m-gguf"
    provider = "gguf"
    enabled = true
    config = {{ model_path = "{root}/model3/w.gguf" }}
""").strip()


@patch('heylook_llm.router.MLXProvider', new=MockProvider)
@patch('heylook_llm.providers.llama_server_provider.LlamaServerProvider', new=MockProvider)
class TestOneEngineFamilyResident(unittest.TestCase):
    def setUp(self):
        open(os.path.join(_MODEL_ROOT, "model3", "w.gguf"), "a").close()
        self.f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.toml')
        self.f.write(_MIXED_TOML.format(root=_MODEL_ROOT))
        self.f.close()

    def tearDown(self):
        os.unlink(self.f.name)

    def _router(self):
        return ModelRouter(config_path=self.f.name, log_level=logging.DEBUG,
                           initial_model_id=None)

    def test_two_models_of_the_SAME_engine_stay_resident_together(self):
        """The rule must not collapse into max_loaded_models = 1. This is the
        case the memory accounting gets right, and the one the two Qwen-Image
        prompt encoders need -- alternating them otherwise costs a full
        evict-and-reload per switch."""
        r = self._router()
        r.get_provider('m-mlx-a')
        r.get_provider('m-mlx-b')
        assert set(r.providers) == {'m-mlx-a', 'm-mlx-b'}

    def test_a_generating_foreign_model_is_backpressure_not_a_kill(self):
        """Evicting mid-generation would destroy a running request. MODEL_BUSY
        is the existing contract for 'ask again shortly'."""
        from heylook_llm.providers.common.generation_gate import ModelBusyError
        r = self._router()
        r.get_provider('m-mlx-a')
        # active_generations is a read-only property over this backing field
        r.providers['m-mlx-a']._active_generations = 1

        with pytest.raises(ModelBusyError):
            r.get_provider('m-gguf')
        assert set(r.providers) == {'m-mlx-a'}

    def test_the_resident_kind_comes_from_the_LOADED_object_not_the_config(self):
        """A reload can change a model's provider in models.toml while it is
        resident. The rule must reason about what is IN MEMORY, so the kind is
        stamped on the instance at construction.

        Driven through a real reload: after m-mlx-a is loaded, the config is
        rewritten to call that id gguf. Loading m-mlx-b must keep both (the
        resident object is MLX); a config-derived kind would read m-mlx-a as
        gguf and evict it. Loading m-gguf still evicts it."""
        r = self._router()
        r.get_provider('m-mlx-a')
        with open(self.f.name) as fh:
            text = fh.read()
        with open(self.f.name, 'w') as fh:
            fh.write(text.replace('id = "m-mlx-a"\nprovider = "mlx"',
                                  'id = "m-mlx-a"\nprovider = "gguf"', 1))
        r.reload_config()
        assert r.app_config.get_model_config('m-mlx-a').provider == 'gguf'

        r.get_provider('m-mlx-b')
        assert set(r.providers) == {'m-mlx-a', 'm-mlx-b'}
        r.get_provider('m-gguf')
        assert set(r.providers) == {'m-gguf'}


# Loading one engine family evicts the other. The count has room (3), so only
# the engine rule can do this; symmetry matters because the hazard is the
# PAIR, not a direction.
@pytest.mark.unit
@pytest.mark.parametrize("resident, incoming", [
    pytest.param('m-mlx-a', 'm-gguf', id="loading_gguf_evicts_a_resident_mlx_model"),
    pytest.param('m-gguf', 'm-mlx-a', id="loading_mlx_evicts_a_resident_gguf_model"),
])
def test_loading_one_engine_family_evicts_the_other(tmp_path, monkeypatch, resident, incoming):
    monkeypatch.setattr('heylook_llm.router.MLXProvider', MockProvider)
    monkeypatch.setattr('heylook_llm.providers.llama_server_provider.LlamaServerProvider',
                        MockProvider)
    open(os.path.join(_MODEL_ROOT, "model3", "w.gguf"), "a").close()
    path = tmp_path / "mixed.toml"
    path.write_text(_MIXED_TOML.format(root=_MODEL_ROOT))
    r = ModelRouter(config_path=str(path), log_level=logging.DEBUG, initial_model_id=None)

    r.get_provider(resident)
    assert set(r.providers) == {resident}
    r.get_provider(incoming)
    assert set(r.providers) == {incoming}, (
        f"a {resident} model stayed resident alongside {incoming}: {sorted(r.providers)}")
