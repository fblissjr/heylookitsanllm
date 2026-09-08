# tests/unit/test_llama_server_provider.py
#
# Unit tests for the llama-server subprocess provider (plan Phase 7b) --
# everything testable WITHOUT a running llama-server: config validation,
# the sampler cascade -> request payload, and the SSE -> GenerationChunk
# adapter fed canned llama-server stream frames.
#
# Claims (what breaks if a test is deleted):
# - config tests: "gguf" regresses out of the provider registry / Literal and
#   models.toml entries stop validating.
# - payload tests: the sampler cascade (floor -> vendor -> model fields ->
#   sampler -> explicit request fields) or the llama.cpp param mapping
#   (repetition_penalty -> repeat_penalty, always-send max_tokens, thinking
#   -> chat_template_kwargs) silently drifts.
# - SSE adapter tests: llama-server frames (reasoning_content deltas, usage
#   chunk with timings, keepalive comments, [DONE]) stop mapping onto
#   GenerationChunk fields -- telemetry goes dark or thinking is dropped.

import io
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from heylook_llm.providers import llama_server_provider as llama_mod

from heylook_llm.config import ChatRequest, ModelConfig, GGUFModelConfig, PROVIDER_CONFIG_CLASSES
from heylook_llm.providers.base import GenerationChunk
from heylook_llm.samplers import GLOBAL_SAMPLER_FLOOR
from heylook_llm.providers.llama_server_provider import LlamaServerProvider


def req(**kw):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return ChatRequest.model_validate(body)


def make_provider(**config):
    config.setdefault("model_path", "/fake/model.gguf")
    return LlamaServerProvider("test-gguf", config, False)


def _weights(tmp_path):
    """A model file that really exists, for tests that reach load_model().

    load_model() stats every configured path before spawning, so the default
    `/fake/model.gguf` above is deliberately unusable there -- it is fine for
    the argv/payload tests, which never spawn.
    """
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return path


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

class TestGGUFConfig:
    def test_registry_has_gguf(self):
        assert PROVIDER_CONFIG_CLASSES["gguf"] is GGUFModelConfig

    def test_model_config_builds_gguf(self):
        mc = ModelConfig.model_validate({
            "id": "m", "provider": "gguf",
            "config": {"model_path": "/x/model.gguf", "mmproj_path": "/x/mmproj.gguf"},
        })
        assert isinstance(mc.config, GGUFModelConfig)
        assert mc.config.mmproj_path == "/x/mmproj.gguf"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            GGUFModelConfig.model_validate({"model_path": "/x.gguf", "surprise": True})

    def test_capability_inference(self):
        from heylook_llm.capabilities import infer_model_capabilities

        plain = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf", "config": {"model_path": "/x.gguf"}})
        assert infer_model_capabilities(plain) == ["chat"]

        vision = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf",
             "config": {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf"}})
        assert "vision" in infer_model_capabilities(vision)

        thinking = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf",
             "config": {"model_path": "/x.gguf", "supports_thinking": True}})
        caps = infer_model_capabilities(thinking)
        assert "thinking" in caps
        assert "hidden_states" not in caps  # MLX-only feature stays MLX-only


# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------

class TestProviderSurface:
    def test_provider_name_and_template_info(self):
        p = make_provider()
        assert LlamaServerProvider.provider_name == "gguf"
        assert p.template_info() is None  # llama-server owns templating/split

    def test_unload_without_load_is_safe(self):
        make_provider().unload()  # must not raise


# ---------------------------------------------------------------------------
# Spawn args
# ---------------------------------------------------------------------------

class TestBuildArgs:
    """The spawn command is the whole configuration surface of a llama-server;
    a field that never reaches argv is a field that silently does nothing."""

    @staticmethod
    def _args(**config):
        from pathlib import Path
        return make_provider(**config)._build_args(Path("/bin/llama-server"), 1234)

    def test_memory_and_lifecycle_flags_reach_argv(self):
        args = self._args(
            n_gpu_layers_draft=0,
            cache_ram_mb=32768,
            sleep_idle_seconds=120,
            load_mode="mmap+mlock",
        )
        pairs = list(zip(args, args[1:]))
        assert ("-ngld", "0") in pairs
        assert ("-cram", "32768") in pairs
        assert ("--sleep-idle-seconds", "120") in pairs
        assert ("-lm", "mmap+mlock") in pairs

    def test_absent_knobs_emit_nothing(self):
        # Every one of these has a llama-server default worth inheriting; a
        # provider that always passes a value silently overrides upstream.
        args = self._args()
        for flag in ("-ngld", "-cram", "--sleep-idle-seconds", "-lm", "-b", "-ub"):
            assert flag not in args

    @staticmethod
    def _spawn_args(monkeypatch, headroom, **config):
        # The seam _auto_ubatch reads; the real one sizes files against the
        # live Metal ceiling, which a unit test must not depend on.
        monkeypatch.setattr(LlamaServerProvider, "_working_set_headroom_gb",
                            lambda self: headroom)
        p = make_provider(**config)
        return p._build_args(Path("/bin/llama-server"), 1234,
                             auto_ubatch=p._auto_ubatch())

    def test_auto_ubatch_is_2048_with_headroom(self, monkeypatch):
        args = self._spawn_args(monkeypatch, 40.0)
        assert ("-ub", "2048") in list(zip(args, args[1:]))
        assert "-b" not in args  # n_batch None = llama-server's own 2048

    def test_auto_ubatch_inherits_llama_default_at_the_ceiling(self, monkeypatch):
        # DeepSeek V4 Flash Vision, 2026-09-07: 16 GiB of headroom loaded at
        # 2048 and died in the first decode with a Metal OOM. Below the
        # threshold the flag must be ABSENT, not 512 -- llama-server's own
        # default is the thing being inherited.
        args = self._spawn_args(monkeypatch, 16.0)
        assert "-ub" not in args

    def test_auto_ubatch_inherits_when_no_ceiling_is_readable(self, monkeypatch):
        assert "-ub" not in self._spawn_args(monkeypatch, None)

    def test_stored_ubatch_wins_over_auto_both_ways(self, monkeypatch):
        pairs = lambda a: list(zip(a, a[1:]))
        assert ("-ub", "512") in pairs(self._spawn_args(monkeypatch, 40.0, n_ubatch=512))
        assert ("-ub", "4096") in pairs(self._spawn_args(monkeypatch, 16.0, n_ubatch=4096))

    def test_ubatch_above_batch_is_refused_not_clamped(self):
        # llama_context takes min(n_batch, n_ubatch) silently; the config
        # must not validate a value the process would quietly ignore.
        with pytest.raises(ValueError, match="clamp"):
            GGUFModelConfig(model_path="/fake/model.gguf", n_ubatch=4096)
        with pytest.raises(ValueError, match="clamp"):
            GGUFModelConfig(model_path="/fake/model.gguf", n_ubatch=4096, n_batch=2048)
        ok = GGUFModelConfig(model_path="/fake/model.gguf", n_ubatch=4096, n_batch=4096)
        assert ok.n_ubatch == 4096

    @pytest.mark.parametrize("token", [
        "--log-prompts-dir", "--log-file", "--slot-save-path",
        "--log-prompts-dir=/tmp/p",  # `=` form: the flag NAME is what matters
    ])
    def test_extra_args_may_not_make_llama_server_write_files(self, token):
        # extra_args is appended to argv verbatim, so it was the last route by
        # which a gguf model could put a file somewhere heylook did not choose
        # -- and for --log-prompts-dir that file is PROMPT TEXT, written at
        # observability_level="off", with nothing announcing it. Stripping the
        # env var while leaving this open would have made the invariant a
        # claim rather than a fact.
        with pytest.raises(ValueError, match="write"):
            GGUFModelConfig(model_path="/fake/model.gguf", extra_args=[token, "/tmp/x"])

    def test_extra_args_still_passes_ordinary_flags(self):
        # Guard the guard: a validator that rejected extra_args outright would
        # pass every case above for free.
        cfg = GGUFModelConfig(model_path="/fake/model.gguf",
                              extra_args=["--verbose", "--slot-prompt-similarity", "0.5"])
        assert cfg.extra_args[0] == "--verbose"

    def test_chat_template_override_reaches_argv(self):
        args = self._args(chat_template_path="/tmp/qwen38-official.jinja")
        assert ("--chat-template-file", "/tmp/qwen38-official.jinja") in \
            list(zip(args, args[1:]))

    def test_missing_chat_template_fails_before_spawn(self, tmp_path, monkeypatch):
        # The failure this prevents is silent: at observability_level=off the
        # subprocess log is DEVNULL, so llama-server dying on an unreadable
        # template file would surface as a bare startup timeout. Assert we
        # never even reach Popen.
        monkeypatch.setattr(LlamaServerProvider, "_resolve_binary",
                            lambda self: tmp_path / "llama-server")
        monkeypatch.setattr(
            llama_mod.subprocess, "Popen",
            lambda *a, **k: pytest.fail("spawned despite an unreadable template"))
        provider = make_provider(model_path=str(_weights(tmp_path)),
                                 chat_template_path=str(tmp_path / "nope.jinja"))
        with pytest.raises(FileNotFoundError, match="chat_template_path"):
            provider.load_model()

    def test_present_chat_template_passes_preflight(self, tmp_path, monkeypatch):
        # Guard the guard: the check must key on the file existing, not reject
        # the field outright (which would pass the test above for free).
        tmpl = tmp_path / "ok.jinja"
        tmpl.write_text("{{ messages }}")
        monkeypatch.setattr(LlamaServerProvider, "_resolve_binary",
                            lambda self: tmp_path / "llama-server")
        spawned = []
        monkeypatch.setattr(
            llama_mod.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(
                RuntimeError("stop here -- preflight passed")))
        provider = make_provider(model_path=str(_weights(tmp_path)),
                                 chat_template_path=str(tmpl))
        with pytest.raises(RuntimeError, match="preflight passed"):
            provider.load_model()
        assert spawned, "preflight rejected a template file that exists"

    @pytest.mark.parametrize("field, filename", [
        ("model_path", "gone.gguf"),
        ("mmproj_path", "gone-mmproj.gguf"),
        ("draft_model_path", "gone-draft.gguf"),
    ])
    def test_missing_configured_file_fails_before_spawn(
            self, tmp_path, monkeypatch, field, filename):
        # A models.toml entry outliving a directory rename. llama-server exits
        # 1 immediately and its output is DEVNULL at the default observability
        # level, so the operator gets `exited with code 1 -- output not
        # captured` and nothing naming the file (2026-09-06). The error must
        # name the FIELD, since the entry can carry four paths and only one of
        # them is wrong.
        monkeypatch.setattr(LlamaServerProvider, "_resolve_binary",
                            lambda self: tmp_path / "llama-server")
        monkeypatch.setattr(
            llama_mod.subprocess, "Popen",
            lambda *a, **k: pytest.fail(f"spawned despite a missing {field}"))
        cfg = {"model_path": str(_weights(tmp_path)), field: str(tmp_path / filename)}
        provider = make_provider(**cfg)
        with pytest.raises(FileNotFoundError, match=field):
            provider.load_model()

    def test_present_configured_files_pass_preflight(self, tmp_path, monkeypatch):
        # Guard the guard, same reason as the template pair above: a check that
        # rejected these fields outright would pass every case above for free.
        monkeypatch.setattr(LlamaServerProvider, "_resolve_binary",
                            lambda self: tmp_path / "llama-server")
        spawned = []
        monkeypatch.setattr(
            llama_mod.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(
                RuntimeError("stop here -- preflight passed")))
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"GGUF")
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF")
        provider = make_provider(model_path=str(_weights(tmp_path)),
                                 mmproj_path=str(mmproj),
                                 draft_model_path=str(draft))
        with pytest.raises(RuntimeError, match="preflight passed"):
            provider.load_model()
        assert spawned, "preflight rejected files that all exist"

    def test_absent_chat_template_leaves_the_gguf_embedded_one_in_force(self):
        # The default MUST stay "whatever the quantizer baked in". Emitting a
        # flag here would override the embedded template with something the
        # user never chose -- the exact failure this field exists to make
        # explicit rather than accidental.
        assert "--chat-template-file" not in self._args()

    @pytest.mark.parametrize("field,flag,value", [
        ("n_gpu_layers_draft", "-ngld", 0),   # 0 = drafter entirely off the GPU
        ("cache_ram_mb", "-cram", 0),         # 0 = disable the prompt cache
        ("cache_ram_mb", "-cram", -1),        # -1 = unlimited
    ])
    def test_falsy_but_meaningful_values_are_not_dropped(self, field, flag, value):
        # These went through `if cfg.get(x)` once; 0 and -1 are real settings,
        # not "unset", and truthiness silently discarded them.
        args = self._args(**{field: value})
        assert (flag, str(value)) in list(zip(args, args[1:]))


# ---------------------------------------------------------------------------
# Spawn environment
# ---------------------------------------------------------------------------

class TestSpawnEnvironment:
    """heylook owns where llama-server's output goes.

    `observability_level = off` (the default) sends the subprocess's stdout to
    DEVNULL, so nothing reaches disk. `LLAMA_ARG_LOG_FILE` in the environment
    defeats that: llama-server opens its own file, and llama.cpp's logger sends
    output to a set file INSTEAD of stdout (common/log.cpp), so the variable
    both writes a file heylook did not sanction and diverts the stream heylook
    captures when the level IS raised. It is the only env-borne write of the
    three disk-writing options -- --log-prompts-dir and --slot-save-path carry
    no `.set_env` -- so removing it closes the whole surface.
    """

    def test_the_log_file_var_is_stripped_and_its_siblings_are_not(
            self, tmp_path, monkeypatch):
        # One test, two halves, because either alone passes for a wrong reason:
        # a provider that forgot `env=` entirely inherits os.environ and fails
        # the first half; a provider that wiped LLAMA_ARG_* wholesale (or passed
        # a bare env) passes the first half and fails the second. The siblings
        # are behaviour knobs someone may be setting on purpose -- they get the
        # warning, not the scrub.
        monkeypatch.setenv("LLAMA_ARG_LOG_FILE", str(tmp_path / "sneaky.log"))
        monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "4096")
        monkeypatch.setattr(LlamaServerProvider, "_resolve_binary",
                            lambda self: tmp_path / "llama-server")
        # load_model's own precondition, made this test's rather than
        # inherited: above "off" it mkdirs a CWD-relative logs/ in the repo and
        # opens an append handle that fake_popen's raise never lets it close.
        # The level is process-global and another module's fixture sets it to
        # "debug" with no teardown, so "it is off right now" is luck, not a
        # property of this file.
        monkeypatch.setattr(llama_mod.observability, "current_level", lambda: "off")

        # Popen is patched process-globally, so this ALSO intercepts ram_fit's
        # `vm_stat` probe, which runs first. Key the capture on the binary we
        # spawn rather than merging every call's kwargs into one bag -- that
        # bag was correct only because the llama-server call happened to be
        # last, and would read another process's environment the moment either
        # moved.
        seen = {}

        def fake_popen(argv, *a, **k):
            if argv and str(argv[0]).endswith("llama-server"):
                seen.update(k)
                raise RuntimeError("stop here -- spawned")
            raise FileNotFoundError("unrelated probe, not this test's subject")

        monkeypatch.setattr(llama_mod.subprocess, "Popen", fake_popen)
        provider = make_provider(model_path=str(_weights(tmp_path)))
        with pytest.raises(RuntimeError, match="spawned"):
            provider.load_model()

        child_env = seen["env"]  # absent key = the provider never passed one
        assert "LLAMA_ARG_LOG_FILE" not in child_env
        assert child_env.get("LLAMA_ARG_CACHE_RAM") == "4096"


# ---------------------------------------------------------------------------
# Sleep/wake timeout
# ---------------------------------------------------------------------------

class TestSleepWakeTimeout:
    """`--sleep-idle-seconds` frees the model but keeps the process, so the
    next request pays a full RELOAD before the first byte. On a large model
    that is minutes -- far past the 120s wedge-detection timeout."""

    def test_normal_timeout_when_sleep_is_not_configured(self):
        p = make_provider()
        p._base_url = "http://127.0.0.1:1"
        assert p._request_timeout() == llama_mod._SSE_READ_TIMEOUT_S

    def test_awake_server_keeps_the_wedge_timeout(self, monkeypatch):
        p = make_provider(sleep_idle_seconds=60, startup_timeout_s=900.0)
        p._base_url = "http://127.0.0.1:1"
        monkeypatch.setattr(p, "_is_sleeping", lambda: False)
        assert p._request_timeout() == llama_mod._SSE_READ_TIMEOUT_S

    def test_sleeping_server_gets_the_reload_budget(self, monkeypatch):
        p = make_provider(sleep_idle_seconds=60, startup_timeout_s=900.0)
        p._base_url = "http://127.0.0.1:1"
        monkeypatch.setattr(p, "_is_sleeping", lambda: True)
        assert p._request_timeout() == 900.0

    def test_unreachable_props_does_not_raise(self):
        # Best-effort probe: an older llama-server without /props is_sleeping,
        # or one mid-restart, must degrade to the normal timeout, not a 500.
        p = make_provider(sleep_idle_seconds=60)
        p._base_url = "http://127.0.0.1:1"  # nothing listening
        assert p._is_sleeping() is False
        assert p._request_timeout() == llama_mod._SSE_READ_TIMEOUT_S


# ---------------------------------------------------------------------------
# Orphan prevention: the process-exit backstop
# ---------------------------------------------------------------------------

class _FakeProc:
    """Stands in for Popen: alive until someone signals its group."""

    def __init__(self, pid=4242):
        self.pid = pid
        self._rc = None

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        self._rc = -15
        return self._rc


class TestSubprocessRegistry:
    """llama-server is spawned with start_new_session=True, so it sits in its
    OWN process group -- the terminal's Ctrl-C (SIGINT to the foreground
    group) never reaches it. Nothing else reaps it either, so every heylook
    exit used to leak a multi-GB llama-server that outlived its parent
    (observed 2026-07-26: two orphans, ~22GB, PPID 1). The registry + the
    atexit backstop are what close that hole.
    """

    def setup_method(self):
        llama_mod._ACTIVE_PROCS.clear()

    teardown_method = setup_method

    def test_spawned_process_is_registered(self, monkeypatch):
        p = make_provider()
        proc = _FakeProc()
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        p._register_proc(proc)
        assert proc in llama_mod._ACTIVE_PROCS

    def test_unload_deregisters(self, monkeypatch):
        """Claim: an unloaded model must not be killed again at exit -- its pid
        may have been recycled by then."""
        p = make_provider()
        proc = _FakeProc()
        killed = []
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        p._register_proc(proc)
        p._proc = proc
        p.unload()
        assert proc not in llama_mod._ACTIVE_PROCS
        assert killed, "unload should still signal the group"

    def test_destructor_signals_without_waiting(self, monkeypatch):
        """A destructor must not block, on THIS provider too.

        v2.0.28 gave `unload` a `drain` argument and honoured it in MLX only;
        here it was accepted and ignored, with a comment saying so. But
        `BaseProvider.__del__` is inherited, so collecting a gguf provider ran
        SIGTERM -> wait(10) -> SIGKILL -> wait(5): up to 15 seconds on
        whatever thread the GC fired on. Both halves of that release's claim
        failed for gguf.
        """
        waits = []

        class _RecordingProc(_FakeProc):
            def wait(self, timeout=None):
                waits.append(timeout)
                return super().wait(timeout)

        p = make_provider()
        proc = _RecordingProc()
        killed = []
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        p._register_proc(proc)
        p._proc = proc

        p.unload(drain=False)

        assert waits == [], f"unload(drain=False) waited on the subprocess: {waits}"
        assert killed == [(proc.pid, signal.SIGTERM)], (
            "drain=False must still SIGNAL -- not signalling strands a whole "
            "llama-server holding GPU memory for the life of the parent, which "
            "is worse than an unreaped child"
        )
        assert proc in llama_mod._ACTIVE_PROCS, (
            "an unwaited process must stay registered: without a wait we cannot "
            "escalate to SIGKILL here, so the atexit backstop has to keep that "
            "option. Safe because we did NOT wait -- the Popen is unreaped, so "
            "its poll() there still speaks for this pid."
        )

    def test_deliberate_unload_still_waits(self, monkeypatch):
        """The other half of the same contract: a caller that CAN afford to
        wait still does, or drain=False stops being a distinction."""
        waits = []

        class _RecordingProc(_FakeProc):
            def wait(self, timeout=None):
                waits.append(timeout)
                return super().wait(timeout)

        p = make_provider()
        proc = _RecordingProc()
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: None)
        p._register_proc(proc)
        p._proc = proc

        p.unload()

        assert waits, "the deliberate teardown path stopped waiting for the process to exit"
        assert proc not in llama_mod._ACTIVE_PROCS

    def test_backstop_kills_leftover_process_group(self, monkeypatch):
        """Claim: this is the last line of defense. Delete it and any exit path
        that skips the lifespan shutdown (startup crash, second Ctrl-C) leaks
        the subprocess."""
        proc = _FakeProc()
        killed = []
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        llama_mod._ACTIVE_PROCS.add(proc)

        llama_mod._kill_orphans()

        assert killed == [(4242, signal.SIGTERM)]
        assert not llama_mod._ACTIVE_PROCS

    def test_backstop_skips_already_dead_process(self, monkeypatch):
        """A pid that already exited must not be signalled -- the number may
        belong to something else by now."""
        proc = _FakeProc()
        proc._rc = 0
        killed = []
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        llama_mod._ACTIVE_PROCS.add(proc)

        llama_mod._kill_orphans()

        assert killed == []

    def test_backstop_actually_runs_on_interpreter_exit(self):
        """End-to-end: importing the provider must arm the hook.

        Claim: a registry nobody drains is dead code. Asserted by really
        exiting a Python process rather than introspecting atexit's private
        registry, so this fails if the registration is dropped OR if atexit
        never reaches it.
        """
        script = (
            "import os, sys\n"
            "from heylook_llm.providers import llama_server_provider as m\n"
            "os.killpg = lambda pgid, sig: print(f'KILLED {pgid} {sig}')\n"
            "os.getpgid = lambda pid: pid\n"
            "class P:\n"
            "    pid = 4242\n"
            "    def poll(self): return None\n"
            "    def wait(self, timeout=None): return -15\n"
            "m._ACTIVE_PROCS.add(P())\n"
            "sys.exit(0)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        assert "KILLED 4242" in out.stdout, (
            f"atexit backstop did not run. stdout={out.stdout!r} stderr={out.stderr[-500:]!r}"
        )


# ---------------------------------------------------------------------------
# Sampler cascade -> payload
# ---------------------------------------------------------------------------

class TestPayload:
    def test_floor_applied_and_max_tokens_always_sent(self):
        p = make_provider()
        payload = p._build_payload(req())
        assert payload["temperature"] == GLOBAL_SAMPLER_FLOOR["temperature"]
        assert payload["max_tokens"] == 4096  # llama default is UNLIMITED; must always send
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_request_overrides_and_param_mapping(self):
        p = make_provider()
        payload = p._build_payload(req(temperature=0.1, repetition_penalty=1.3, max_tokens=64, seed=7, presence_penalty=1.5, top_k=20))
        assert payload["temperature"] == 0.1
        assert payload["repeat_penalty"] == 1.3  # llama.cpp's name
        assert "repetition_penalty" not in payload
        assert payload["max_tokens"] == 64
        assert payload["seed"] == 7
        assert payload["presence_penalty"] == 1.5
        assert payload["top_k"] == 20

    def test_model_config_max_tokens_beats_floor(self):
        # Deleting this resurrects the dead-overlay bug (code-review
        # 2026-07-26): the floor pre-seeds max_tokens, so a guarded
        # "if not in merged" write could never fire and a model-level
        # max_tokens silently fell back to 4096.
        p = make_provider(max_tokens=8000)
        payload = p._build_payload(req())
        assert payload["max_tokens"] == 8000

    def test_request_max_tokens_beats_model_config(self):
        p = make_provider(max_tokens=8000)
        payload = p._build_payload(req(max_tokens=64))
        assert payload["max_tokens"] == 64

    def test_thinking_reaches_the_template_and_changes_no_sampler_value(self):
        """gguf mirrors MLX: thinking is a TEMPLATE kwarg, not a sampler
        change. It carried presence_penalty 1.5 until v2.0.32; that value was
        never measured and contradicted the guidance of the family it came
        from, so the switch now travels alone."""
        p = make_provider()
        payload = p._build_payload(req(enable_thinking=True))
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert payload["presence_penalty"] == GLOBAL_SAMPLER_FLOOR["presence_penalty"]

    def test_a_looping_model_can_still_be_tuned_per_model(self):
        """gguf gained the per-model field in v2.0.32, having had none: the
        removal must not cost the ABILITY, only the automatic default."""
        p = make_provider(presence_penalty=1.5)
        assert p._build_payload(req(enable_thinking=True))["presence_penalty"] == 1.5

    def test_request_thinking_off_no_penalty(self):
        p = make_provider()
        payload = p._build_payload(req(enable_thinking=False))
        assert payload["presence_penalty"] == 0.0  # floor value, no overlay
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_explicit_presence_penalty_beats_thinking_overlay(self):
        p = make_provider()
        payload = p._build_payload(req(enable_thinking=True, presence_penalty=0.3))
        assert payload["presence_penalty"] == 0.3

    def test_the_vendor_layer_overlays_the_floor(self):
        """What replaced the named-sampler layers: the model's OWN published
        settings. The floor is only reached where the model says nothing, so
        this is the layer that has to actually reach the wire."""
        p = make_provider()
        payload = p._build_payload(req())
        assert payload["temperature"] == GLOBAL_SAMPLER_FLOOR["temperature"]

        from heylook_llm.samplers import resolve_effective_sampling
        merged = resolve_effective_sampling(
            req(), {"model_path": "x"}, vendor={"temperature": 0.31, "top_k": 64})
        assert merged["temperature"] == 0.31, "vendor did not beat the floor"
        assert merged["top_k"] == 64

    def test_enable_thinking_maps_to_chat_template_kwargs(self):
        p = make_provider()
        on = p._build_payload(req(enable_thinking=True))
        off = p._build_payload(req(enable_thinking=False))
        assert on["chat_template_kwargs"] == {"enable_thinking": True}
        assert off["chat_template_kwargs"] == {"enable_thinking": False}

    def test_unset_thinking_is_sent_as_an_explicit_off(self):
        """Claim: an omitted enable_thinking means OFF on gguf, exactly as it
        already does on MLX -- and it must travel as an explicit `false`.

        Omitting the key is not "no opinion" here. llama-server runs --jinja,
        so with no chat_template_kwargs it applies the GGUF's own template
        default, which is thinking-ON for gemma-4 / Qwen3.6 / DeepSeek-V4.
        MLX resolves the same unset request to False. That made one v3
        checkbox mean opposite things per engine, and left no way at all to
        turn thinking off on a gguf model (the control only ever sends
        true/null). Asserting the sent VALUE, not just the key's presence,
        is the point: a bare `"chat_template_kwargs" in payload` check would
        pass on a payload that says true.
        """
        payload = make_provider()._build_payload(req())
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["presence_penalty"] == 0.0

    def test_multimodal_content_parts_pass_through(self):
        p = make_provider()
        content = [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        payload = p._build_payload(req(messages=[{"role": "user", "content": content}]))
        assert payload["messages"][0]["content"] == content


# ---------------------------------------------------------------------------
# SSE stream -> GenerationChunk
# ---------------------------------------------------------------------------

def _stream_bytes(*frames: str) -> io.BytesIO:
    return io.BytesIO(("\n".join(frames) + "\n").encode())


CANNED = [
    ": keepalive ping",
    'data: {"choices":[{"delta":{"role":"assistant","content":null},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"reasoning_content":"let me "},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"reasoning_content":"think"},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"content":" world"},"index":0,"finish_reason":null}]}',
    'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}',
    'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,'
    '"prompt_tokens_details":{"cached_tokens":2}},'
    '"timings":{"prompt_per_second":100.5,"predicted_per_second":42.0,'
    '"cache_n":2,"prompt_n":5,"predicted_n":3,"draft_n":12,"draft_n_accepted":5}}',
    "data: [DONE]",
    'data: {"choices":[{"delta":{"content":"MUST NOT APPEAR"},"index":0}]}',
]


class TestSSEAdapter:
    def collect(self, frames):
        p = make_provider()
        return list(p._stream_chunks(_stream_bytes(*frames), abort_event=None))

    def test_full_stream_mapping(self):
        chunks = self.collect(CANNED)
        assert all(isinstance(c, GenerationChunk) for c in chunks)

        thinking = "".join(c.thinking for c in chunks if c.thinking)
        text = "".join(c.text for c in chunks if c.text)
        assert thinking == "let me think"
        assert text == "Hello world"
        # nothing after [DONE]
        assert "MUST NOT APPEAR" not in text

        finish = [c.finish_reason for c in chunks if c.finish_reason]
        assert finish == ["stop"]

        final = chunks[-1]
        assert final.prompt_tokens == 7
        assert final.generation_tokens == 3
        assert final.cached_tokens == 2
        assert final.prompt_tps == 100.5
        assert final.generation_tps == 42.0
        # spec-decode counters (present only when MTP/draft was active)
        assert final.draft_tokens == 12
        assert final.draft_accepted == 5

    def test_abort_stops_stream(self):
        class Abort:
            def __init__(self):
                self.calls = 0

            def is_set(self):
                self.calls += 1
                return self.calls > 2

        p = make_provider()
        chunks = list(p._stream_chunks(_stream_bytes(*CANNED), abort_event=Abort()))
        assert len(chunks) < 6  # cut short, never reached the end of the stream

    def test_error_frame_raises_instead_of_ending_cleanly(self, monkeypatch):
        # llama-server reports a decode failure INSIDE the stream (the HTTP
        # status was already 200): `data: {"error": {...}}`. It has no
        # `choices`, so the adapter used to skip it and the stream ended as a
        # zero-token end_turn -- an empty reply for what was a Metal OOM.
        from heylook_llm.providers.base import GenerationFailed
        p = make_provider()
        monkeypatch.setattr(llama_mod.ram_fit, "fit_for_config",
                            lambda cfg, **kw: pytest.fail("no sizing for a non-compute error"))
        with pytest.raises(GenerationFailed, match="template"):
            list(p._stream_chunks(_stream_bytes(
                'data: {"error":{"code":500,"message":"template failed","type":"server_error"}}'),
                abort_event=None))

    def test_compute_error_names_the_metal_ceiling_and_the_sysctl(self, monkeypatch):
        # "Compute error." is all llama-server says; the ggml OOM line is in a
        # log that is DEVNULL by default. The message the reader gets must
        # carry the headroom and the remedy the fit panel would show.
        from heylook_llm.providers.base import GenerationFailed
        from types import SimpleNamespace
        monkeypatch.setattr(llama_mod.ram_fit, "fit_for_config",
                            lambda cfg, **kw: SimpleNamespace(
                                kv_headroom_gb=16.0, working_set_gb=161.3,
                                sysctl_suggest_mb=177458))
        p = make_provider()
        with pytest.raises(GenerationFailed) as ei:
            list(p._stream_chunks(_stream_bytes(
                'data: {"error":{"code":500,"message":"Compute error.","type":"server_error"}}'),
                abort_event=None))
        text = str(ei.value)
        assert "working set" in text and "16.0 GiB" in text
        assert "iogpu.wired_limit_mb=177458" in text

    def test_malformed_frame_raises_generation_failed(self):
        from heylook_llm.providers.base import GenerationFailed

        p = make_provider()
        with pytest.raises(GenerationFailed):
            list(p._stream_chunks(_stream_bytes("data: {not json"), abort_event=None))


class TestContinuationEchoStrip:
    """llama-server ECHOES a continued assistant message's prefill back as the
    leading content delta(s) (observed live on the pinned build: prefill
    "1, 2, 3," came back as the first delta '1, 2, 3, '). The strip is
    POSITIONAL: retokenization can attach whitespace to the echoed span, so a
    byte-equality check would false-negative and leak the echo through."""

    def collect(self, frames, echo_chars):
        p = make_provider()
        return list(p._stream_chunks(_stream_bytes(*frames), abort_event=None,
                                     echo_chars=echo_chars))

    def test_echo_in_one_delta_is_stripped(self):
        frames = [
            'data: {"choices":[{"delta":{"content":"1, 2, 3, "},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"4, 5"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        text = "".join(c.text for c in self.collect(frames, echo_chars=len("1, 2, 3,")))
        assert text == " 4, 5"  # the delta's surplus space is real continuation

    def test_echo_spanning_deltas_is_stripped(self):
        frames = [
            'data: {"choices":[{"delta":{"content":"1, 2"},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":", 3, 4"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        text = "".join(c.text for c in self.collect(frames, echo_chars=len("1, 2, 3")))
        assert text == ", 4"

    def test_pure_echo_deltas_are_swallowed_not_emitted_empty(self):
        frames = [
            'data: {"choices":[{"delta":{"content":"prefix"},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" tail"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        chunks = self.collect(frames, echo_chars=len("prefix"))
        assert all(c.text or c.thinking or c.finish_reason or c.prompt_tokens for c in chunks)
        assert "".join(c.text for c in chunks) == " tail"

    def test_thinking_deltas_untouched_without_a_thinking_prefill(self):
        frames = [
            'data: {"choices":[{"delta":{"reasoning_content":"hmm"},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"prefixreal"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        chunks = self.collect(frames, echo_chars=len("prefix"))
        assert "".join(c.thinking or "" for c in chunks) == "hmm"
        assert "".join(c.text for c in chunks) == "real"

    def test_thinking_prefill_echo_is_stripped_from_the_reasoning_channel(self):
        # A resumed thought: llama-server re-emits the prefilled reasoning as
        # the leading reasoning_content delta(s), then the new trace. The
        # two channels are stripped independently -- content was not
        # prefilled here and must come through whole.
        frames = [
            'data: {"choices":[{"delta":{"reasoning_content":"so f"},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"ar and on"},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"answer"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        p = make_provider()
        chunks = list(p._stream_chunks(_stream_bytes(*frames), abort_event=None,
                                       echo_chars=0, echo_thinking_chars=len("so far")))
        assert "".join(c.thinking or "" for c in chunks) == " and on"
        assert "".join(c.text for c in chunks) == "answer"
        assert all(c.text or c.thinking or c.finish_reason for c in chunks)


class TestContinuationGuards:
    """What llama-server cannot express must 400, not silently do the wrong
    thing: user-role continuation has no llama-server spelling, and a trailing
    assistant message is ALWAYS continued (so false cannot be honored)."""

    def _req(self, messages, flag):
        from heylook_llm.config import ChatRequest
        return ChatRequest(model="m", messages=messages, continue_final_message=flag)

    def _payload(self, req):
        return {"messages": [m.model_dump(exclude_none=True) for m in req.messages]}

    def test_user_role_continuation_400s(self):
        from heylook_llm.providers.base import InvalidGenerationRequest
        p = make_provider()
        req = self._req([{"role": "user", "content": "finish my sentence"}], True)
        with pytest.raises(InvalidGenerationRequest, match="assistant turns only"):
            p._continuation_echo_chars(req, self._payload(req))

    def test_false_with_trailing_assistant_400s(self):
        from heylook_llm.providers.base import InvalidGenerationRequest
        p = make_provider()
        req = self._req([{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "he"}], False)
        with pytest.raises(InvalidGenerationRequest, match="always continues"):
            p._continuation_echo_chars(req, self._payload(req))

    def test_auto_trailing_assistant_returns_prefill_length(self):
        p = make_provider()
        req = self._req([{"role": "user", "content": "count"},
                         {"role": "assistant", "content": "1, 2,"}], None)
        assert p._continuation_echo_chars(req, self._payload(req)) == (5, 0)

    def test_no_continuation_returns_zero(self):
        p = make_provider()
        req = self._req([{"role": "user", "content": "hi"}], None)
        assert p._continuation_echo_chars(req, self._payload(req)) == (0, 0)

    def test_thinking_prefill_echo_is_measured_lstripped(self):
        # llama-server echoes a prefilled reasoning_content back on the
        # reasoning channel minus its LEADING whitespace (measured live on
        # b10814, 2026-09-04): the strip is sized to the lstripped string so
        # it can only ever under-strip, never eat a real token.
        p = make_provider()
        req = self._req([{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "",
                          "thinking": "  partial thought \n"}], None)
        payload = self._payload(req)
        assert p._continuation_echo_chars(req, payload) == (0, len("partial thought \n"))

    def test_wire_message_renames_thinking_to_reasoning_content(self):
        # heylook's `thinking` is llama-server's `reasoning_content`; the
        # unrenamed key was silently ignored, so every replayed assistant
        # turn rendered an EMPTY thinking block and a mid-thought continue
        # re-thought from scratch.
        from heylook_llm.config import ChatMessage
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider
        msg = ChatMessage(role="assistant", content="", thinking="so far")
        wire = LlamaServerProvider._wire_message(msg)
        assert wire == {"role": "assistant", "content": "", "reasoning_content": "so far"}
        user = ChatMessage(role="user", content="hi", thinking="never")
        assert "reasoning_content" not in LlamaServerProvider._wire_message(user)

    def test_text_parts_prefill_is_flattened_not_refused(self):
        # /code-review 53b266c finding 1: block-form prefill (what the
        # Messages converter produces, with no opt-out field) must WORK --
        # the payload's copy is flattened to the exact measured string so
        # the positional strip stays exact.
        p = make_provider()
        req = self._req([
            {"role": "user", "content": "count"},
            {"role": "assistant",
             "content": [{"type": "text", "text": "1, 2,"},
                         {"type": "text", "text": "3,"}]},
        ], None)
        payload = self._payload(req)
        chars = p._continuation_echo_chars(req, payload)
        assert payload["messages"][-1]["content"] == "1, 2, 3,"
        assert chars == (len("1, 2, 3,"), 0)

    def test_non_text_parts_continue_unstripped_not_400(self):
        # No knowable rendered length -> continuation proceeds (llama-server
        # always continues a trailing assistant turn) with NOTHING stripped,
        # the pre-v1.61 behavior -- never a new 400 on old traffic.
        p = make_provider()
        req = self._req([
            {"role": "user", "content": "look"},
            {"role": "assistant",
             "content": [{"type": "text", "text": "as you can see"},
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
        ], None)
        payload = self._payload(req)
        assert p._continuation_echo_chars(req, payload) == (0, 0)
        assert isinstance(payload["messages"][-1]["content"], list)  # untouched


class TestBinaryResolution:
    """The canonical build is the ONE intended source; overrides are escape
    hatches and must be LOUD -- on 2026-08-13 a stale $HEYLOOK_LLAMA_SERVER
    silently shadowed a freshly built canonical binary (predating a new model
    arch), and nothing said so until the load failed."""

    def _with_canonical(self, tmp_path, monkeypatch):
        canonical = tmp_path / "canonical" / "llama-server"
        canonical.parent.mkdir(parents=True)
        canonical.write_text("#!/bin/true\n")
        monkeypatch.setattr(LlamaServerProvider, "DEFAULT_BUILD", canonical)
        return canonical

    def test_canonical_build_used_when_nothing_overrides(self, tmp_path, monkeypatch, caplog):
        canonical = self._with_canonical(tmp_path, monkeypatch)
        monkeypatch.delenv("HEYLOOK_LLAMA_SERVER", raising=False)
        p = make_provider()
        assert p._resolve_binary() == canonical

    def test_env_override_warns_about_shadowing(self, tmp_path, monkeypatch, caplog):
        import logging as _logging
        canonical = self._with_canonical(tmp_path, monkeypatch)
        override = tmp_path / "elsewhere" / "llama-server"
        override.parent.mkdir(parents=True)
        override.write_text("#!/bin/true\n")
        monkeypatch.setenv("HEYLOOK_LLAMA_SERVER", str(override))
        p = make_provider()
        with caplog.at_level(_logging.WARNING):
            resolved = p._resolve_binary()
        assert resolved == override
        warnings = [r for r in caplog.records if r.levelno >= _logging.WARNING
                    and "HEYLOOK_LLAMA_SERVER" in r.getMessage()]
        assert warnings, "an env-var override shadowing the canonical build must WARN"
        assert any(str(canonical) in r.getMessage() for r in warnings), \
            "the warning must NAME the canonical build being shadowed"

    def test_server_binary_override_warns_with_its_source(self, tmp_path, monkeypatch, caplog):
        import logging as _logging
        self._with_canonical(tmp_path, monkeypatch)
        override = tmp_path / "per-model" / "llama-server"
        override.parent.mkdir(parents=True)
        override.write_text("#!/bin/true\n")
        monkeypatch.delenv("HEYLOOK_LLAMA_SERVER", raising=False)
        p = make_provider(server_binary=str(override))
        with caplog.at_level(_logging.WARNING):
            assert p._resolve_binary() == override
        assert any("server_binary" in r.getMessage() for r in caplog.records
                   if r.levelno >= _logging.WARNING), \
            "a models.toml server_binary override must WARN naming its source"


@pytest.mark.unit
class TestReasoningEffort:
    """Thinking DEPTH is a template variable, not a sampler knob.

    Qwen3.8 added it and defaults to xhigh, so without a route the server can
    only ever run that model at maximum reasoning depth.
    """

    @staticmethod
    def _payload(config=None, **body):
        # make_provider, not __new__: __new__ skips __init__, so any attribute
        # _build_payload starts reading would fail these as AttributeError
        # rather than as a real assertion.
        return make_provider(**(config or {}))._build_payload(req(**body))

    def test_effort_rides_chat_template_kwargs(self):
        kw = self._payload(enable_thinking=True, reasoning_effort="low",
                           max_tokens=32)["chat_template_kwargs"]
        assert kw == {"enable_thinking": True, "reasoning_effort": "low"}

    def test_effort_is_sent_even_with_thinking_off(self):
        """NOT gated on enable_thinking. gpt-oss/harmony reads
        reasoning_effort unconditionally and has no enable_thinking at all, so
        gating made the knob unreachable for the one family the docs name as
        taking low|medium|high. A template that ignores the variable is
        unaffected -- jinja forwards unknown kwargs as template variables."""
        kw = self._payload(enable_thinking=False, reasoning_effort="low",
                           max_tokens=32)["chat_template_kwargs"]
        assert kw == {"enable_thinking": False, "reasoning_effort": "low"}

    def test_effort_alone_still_reaches_the_template(self):
        """The harmony shape: depth set, thinking never mentioned."""
        kw = self._payload(reasoning_effort="high", max_tokens=32)["chat_template_kwargs"]
        assert kw["reasoning_effort"] == "high"

    def test_absent_effort_leaves_the_templates_own_default(self):
        kw = self._payload(enable_thinking=True, max_tokens=32)["chat_template_kwargs"]
        assert "reasoning_effort" not in kw

    def test_model_level_default_reaches_the_payload(self):
        """The third route the CHANGELOG claims (per request / per preset /
        per model). It depends on the single line added to
        EFFECTIVE_SAMPLER_KEYS, so it can regress silently."""
        kw = self._payload({"reasoning_effort": "medium"},
                           enable_thinking=True, max_tokens=32)["chat_template_kwargs"]
        assert kw["reasoning_effort"] == "medium"

    def test_request_beats_the_model_level_default(self):
        kw = self._payload({"reasoning_effort": "medium"},
                           reasoning_effort="low", max_tokens=32)["chat_template_kwargs"]
        assert kw["reasoning_effort"] == "low"

    def test_a_typo_is_rejected_before_it_reaches_the_template(self):
        # llama-server surfaces a raised jinja exception as a 500, so a bad
        # value has to fail here where the error can name the field.
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            req(enable_thinking=True, reasoning_effort="xtreme")


# ---------------------------------------------------------------------------
# The process generation gate (v1.79.60)
# ---------------------------------------------------------------------------
#
# Claim: a request forwarded to llama-server takes the process FIFO gate first
# and holds it until the stream is exhausted, so a second request queues on
# this side instead of sitting in llama-server's own queue past the 120s read
# timeout and coming back as a 500. Field-observed 2026-09-01: three requests
# in a row behind one abandoned thinking-mode run each answered "unreachable:
# timed out", and the model reported loaded the moment the run ended.

import urllib.error

from heylook_llm.providers.base import GenerationFailed
from heylook_llm.providers.common.generation_gate import GenerationGate, ModelBusyError


class TestGenerationGate:
    def _gated(self, monkeypatch, frames=None):
        p = make_provider()
        p._base_url = "http://127.0.0.1:1"
        # A private single-flight gate, so the test neither shares nor leaks
        # state through the process-wide one.
        p._gen_gate = GenerationGate(max_waiting=0)
        monkeypatch.setattr(
            llama_mod.urllib.request, "urlopen",
            lambda *a, **k: _stream_bytes(*(frames or CANNED)),
        )
        return p

    def test_gate_is_taken_when_driven_and_released_after_the_stream(self, monkeypatch):
        p = self._gated(monkeypatch)
        gen = p.create_chat_completion(req())
        assert p._gen_gate.busy is False, "nothing is acquired until the generator is driven"
        next(gen)
        assert p._gen_gate.busy is True, "held across the stream"
        list(gen)
        assert p._gen_gate.busy is False, "released on exhaustion"

    def test_gate_is_released_when_the_forward_fails(self, monkeypatch):
        p = self._gated(monkeypatch)

        def timed_out(*a, **k):
            raise urllib.error.URLError("timed out")

        monkeypatch.setattr(llama_mod.urllib.request, "urlopen", timed_out)
        with pytest.raises(GenerationFailed):
            list(p.create_chat_completion(req()))
        assert p._gen_gate.busy is False

    def test_gate_is_released_when_the_stream_is_closed_early(self, monkeypatch):
        p = self._gated(monkeypatch)
        gen = p.create_chat_completion(req())
        next(gen)
        gen.close()
        assert p._gen_gate.busy is False

    def test_check_capacity_answers_busy_while_another_generation_holds_the_gate(self):
        p = make_provider()
        p._gen_gate = GenerationGate(max_waiting=0)
        p.check_capacity()  # idle: admitted
        p._gen_gate.acquire()
        try:
            with pytest.raises(ModelBusyError):
                p.check_capacity()
            assert p.generation_queue_stats()["active"] == 1
        finally:
            p._gen_gate.release()
        assert p.generation_queue_stats()["active"] == 0

    def test_two_providers_share_the_process_gate(self):
        # One GPU. A gate per provider would let a gguf run and an MLX run
        # overlap, which is the concurrency the gate exists to prevent.
        assert make_provider()._gen_gate is make_provider()._gen_gate



@pytest.mark.unit
class TestRunningContext:
    """``running_ctx`` is what the process GOT, read from /props at ready.

    The parse is guarded, not indexed: a build that moves the key must
    degrade to "unknown", never to a load failure, and a bool or a zero is
    not a context.
    """

    def test_reads_the_slot_ctx(self):
        props = {"default_generation_settings": {"n_ctx": 32768, "id": 0}, "total_slots": 1}
        assert LlamaServerProvider._ctx_from_props(props) == 32768

    @pytest.mark.parametrize("props", [
        {}, {"default_generation_settings": {}}, {"default_generation_settings": None},
        {"default_generation_settings": {"n_ctx": 0}},
        {"default_generation_settings": {"n_ctx": True}},
        {"default_generation_settings": {"n_ctx": "32768"}},
        [], None,
    ])
    def test_anything_else_is_none(self, props):
        assert LlamaServerProvider._ctx_from_props(props) is None

    def test_unload_forgets_it(self):
        p = make_provider()
        p.running_ctx = 4096
        p.unload()
        assert p.running_ctx is None


# ---------------------------------------------------------------------------
# Prefill progress (v1.79.65): return_progress on the payload, prompt_progress
# frames reported up the request's signal channel in the cross-engine meaning
# ---------------------------------------------------------------------------

class TestPrefillProgress:
    def test_payload_asks_for_progress(self):
        p = make_provider()
        assert p._build_payload(req())["return_progress"] is True

    def test_progress_frame_reports_work_only_and_yields_nothing(self):
        from heylook_llm.providers.abort import AbortEvent
        signals = AbortEvent()
        frames = [
            'data: {"choices":[{"delta":{},"index":0,"finish_reason":null}],'
            '"prompt_progress":{"total":100,"cache":20,"processed":60,"time_ms":5}}',
            'data: {"choices":[{"delta":{"content":"Hi"},"index":0,"finish_reason":null}]}',
        ]
        p = make_provider()
        chunks = list(p._stream_chunks(_stream_bytes(*frames), abort_event=signals))
        # llama-server's processed INCLUDES the cache hit; the signal carries
        # the work this request runs, cache subtracted from both.
        assert signals.prefill_progress() == (40, 80)
        assert [c.text for c in chunks] == ["Hi"]

    def test_a_fully_cached_prompt_reports_nothing(self):
        from heylook_llm.providers.abort import AbortEvent
        signals = AbortEvent()
        frames = [
            'data: {"choices":[{"delta":{},"index":0,"finish_reason":null}],'
            '"prompt_progress":{"total":50,"cache":50,"processed":50,"time_ms":0}}',
        ]
        list(make_provider()._stream_chunks(_stream_bytes(*frames), abort_event=signals))
        assert signals.prefill_progress() is None

    def test_no_signal_channel_is_fine(self):
        frames = [
            'data: {"choices":[{"delta":{},"index":0,"finish_reason":null}],'
            '"prompt_progress":{"total":10,"cache":0,"processed":5,"time_ms":1}}',
        ]
        assert list(make_provider()._stream_chunks(_stream_bytes(*frames), abort_event=None)) == []


class TestMediaInTheContinuedTurn:
    """Continuing an assistant turn that carries media works on some chat
    templates and cannot work on others, so the provider ASKS instead of
    assuming either way.

    Measured on b10830 with gemma-4-E4B + mmproj (2026-09-07): success tracks
    exactly one thing -- does the rendered template still contain a media
    marker for each media part sent? That model's template DROPS the marker
    when the final assistant turn is the one being continued (0 markers for 1
    image) and llama-server answers a flat 400 "Failed to tokenize prompt";
    the SAME model with the SAME image on a non-final assistant turn renders
    1 marker and answers 200, as does a user turn. DeepSeek-V4-Flash-Vision
    keeps the marker and continues such a turn in production. So the check is
    a property of the render, not a list of model names that would rot -- and
    a first cut of this guard refused the case unconditionally, which would
    have broken the model that works.
    """

    IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}

    def _payload(self, msgs):
        return {"messages": msgs}

    def _answering(self, monkeypatch, rendered):
        p = make_provider()
        p._base_url = "http://127.0.0.1:1"
        monkeypatch.setattr(
            llama_mod.urllib.request, "urlopen",
            lambda *a, **k: io.BytesIO(json.dumps({"prompt": rendered}).encode()))
        return p

    def test_a_dropped_marker_is_detected(self, monkeypatch):
        p = self._answering(monkeypatch, "<|turn>model\nThe three colours are red,")
        assert p._media_markers_dropped(self._payload(
            [{"role": "assistant", "content": [self.IMG, {"type": "text", "text": "x"}]}])) is True

    def test_a_kept_marker_is_allowed(self, monkeypatch):
        p = self._answering(monkeypatch, "<|turn>model\n<__media_abc__>The three colours are red,")
        assert p._media_markers_dropped(self._payload(
            [{"role": "assistant", "content": [self.IMG, {"type": "text", "text": "x"}]}])) is False

    def test_two_images_need_two_markers(self, monkeypatch):
        # An undercount is the failure mode, so one marker for two images must
        # read the same as none for one.
        p = self._answering(monkeypatch, "<|turn>user\n<__media_a__>only one marker")
        assert p._media_markers_dropped(self._payload(
            [{"role": "user", "content": [self.IMG, self.IMG]}])) is True

    def test_no_media_never_asks(self, monkeypatch):
        # The precheck costs an HTTP round trip; it must not fire on the
        # overwhelmingly common text-only path.
        def explode(*a, **k):
            raise AssertionError("/apply-template was called with no media in the request")
        p = make_provider()
        p._base_url = "http://127.0.0.1:1"
        monkeypatch.setattr(llama_mod.urllib.request, "urlopen", explode)
        assert p._media_markers_dropped(self._payload(
            [{"role": "assistant", "content": [{"type": "text", "text": "x"}]}])) is False

    def test_an_unreachable_template_call_does_not_refuse(self, monkeypatch):
        # Fail OPEN: refusing on a question we could not ask would be worse
        # than the tokenizer error this replaces.
        def boom(*a, **k):
            raise urllib.error.URLError("nope")
        p = make_provider()
        p._base_url = "http://127.0.0.1:1"
        monkeypatch.setattr(llama_mod.urllib.request, "urlopen", boom)
        assert p._media_markers_dropped(self._payload(
            [{"role": "user", "content": [self.IMG]}])) is False
