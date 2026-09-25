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

from heylook_llm.config import ChatRequest, GGUFModelConfig
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

# Construction through ModelConfig: test_config.py
# test_pydantic_construction_round_trip (row gguf_model_config_builds);
# extra="forbid": test_config.py test_extra_keys_are_forbidden (row
# gguf_extra_fields_forbidden); capabilities: test_audio_content.py
# TestCapability::test_inferred_capabilities (the gguf rows).


# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------

class TestProviderSurface:
    # The declared engine (provider_name), as the router reads it:
    # test_generation_chunk.py TestConcreteProviders.

    # unload is safe before any load, and forgets the running context the
    # process reported.
    @pytest.mark.parametrize("running_ctx", [None, 4096], ids=["never_loaded", "running_ctx_set"])
    def test_unload_is_safe_and_forgets_running_ctx(self, running_ctx):
        p = make_provider()
        if running_ctx is not None:
            p.running_ctx = running_ctx
        p.unload()  # must not raise
        assert p.running_ctx is None


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

    # -ub 2048 with headroom; ABSENT (not 512 -- llama-server's own default is
    # the thing being inherited) at the ceiling or when no ceiling is
    # readable; a stored n_ubatch wins over auto both ways. n_batch unset
    # never emits -b (llama-server's own 2048).
    # - at_the_ceiling: DeepSeek V4 Flash Vision, 2026-09-07, loaded at 2048
    #   on this headroom and died in the first decode with a Metal OOM.
    @pytest.mark.parametrize("headroom, config, expected_ub", [
        (40.0, {}, "2048"),
        (16.0, {}, None),
        (None, {}, None),
        (40.0, {"n_ubatch": 512}, "512"),
        (16.0, {"n_ubatch": 4096}, "4096"),
    ], ids=["2048_with_headroom", "inherits_at_the_ceiling", "inherits_when_no_ceiling_readable",
            "stored_wins_over_auto_with_headroom", "stored_wins_over_auto_at_the_ceiling"])
    def test_auto_ubatch(self, monkeypatch, headroom, config, expected_ub):
        args = self._spawn_args(monkeypatch, headroom, **config)
        if expected_ub is None:
            assert "-ub" not in args
        else:
            assert ("-ub", expected_ub) in list(zip(args, args[1:]))
        assert "-b" not in args

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

    @staticmethod
    def _stub_spawn(tmp_path, monkeypatch, popen):
        monkeypatch.setattr(LlamaServerProvider, "_resolve_binary",
                            lambda self: tmp_path / "llama-server")
        monkeypatch.setattr(llama_mod.subprocess, "Popen", popen)

    # A configured path that is missing fails before Popen, naming the FIELD
    # (the entry can carry four paths and only one of them is wrong). The
    # failure this prevents is silent: llama-server exits 1 immediately and
    # its output is kept nowhere at the default observability level, so the
    # operator got `exited with code 1 -- output not captured` and nothing
    # naming the file (2026-09-06: a models.toml entry outliving a directory
    # rename); an unreadable template surfaced as a bare startup timeout.
    @pytest.mark.parametrize("field, filename", [
        ("chat_template_path", "nope.jinja"),
        ("model_path", "gone.gguf"),
        ("mmproj_path", "gone-mmproj.gguf"),
        ("draft_model_path", "gone-draft.gguf"),
    ])
    def test_missing_configured_file_fails_before_spawn(
            self, tmp_path, monkeypatch, field, filename):
        self._stub_spawn(tmp_path, monkeypatch,
                         lambda *a, **k: pytest.fail(f"spawned despite a missing {field}"))
        cfg = {"model_path": str(_weights(tmp_path)), field: str(tmp_path / filename)}
        provider = make_provider(**cfg)
        with pytest.raises(FileNotFoundError, match=field):
            provider.load_model()

    # Guard the guard: the check keys on the file existing, not on rejecting
    # the field outright (which would pass every case above for free).
    @pytest.mark.parametrize("files", [
        {"chat_template_path": "ok.jinja"},
        {"mmproj_path": "mmproj.gguf", "draft_model_path": "draft.gguf"},
    ], ids=["chat_template", "mmproj_and_draft"])
    def test_present_configured_files_pass_preflight(self, tmp_path, monkeypatch, files):
        spawned = []
        self._stub_spawn(tmp_path, monkeypatch,
                         lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(
                             RuntimeError("stop here -- preflight passed")))
        cfg = {"model_path": str(_weights(tmp_path))}
        for field, filename in files.items():
            (tmp_path / filename).write_bytes(b"GGUF")
            cfg[field] = str(tmp_path / filename)
        provider = make_provider(**cfg)
        with pytest.raises(RuntimeError, match="preflight passed"):
            provider.load_model()
        assert spawned, f"preflight rejected files that all exist: {files}"


# ---------------------------------------------------------------------------
# Spawn environment
# ---------------------------------------------------------------------------

class TestSpawnEnvironment:
    """heylook owns where llama-server's output goes.

    `observability_level = off` (the default) keeps the subprocess's output
    nowhere: the pump drains the pipe for cache events and writes no file. `LLAMA_ARG_LOG_FILE` in the environment
    defeats that: llama-server opens its own file, and llama.cpp's logger sends
    output to a set file INSTEAD of stdout (common/log.cpp), so the variable
    both writes a file heylook did not sanction and diverts the stream heylook
    captures when the level IS raised. It is the only env-borne write of the
    three disk-writing options -- --log-prompts-dir and --slot-save-path carry
    no `.set_env` -- so removing it closes the whole surface.
    """

    @staticmethod
    def _spawn_env(tmp_path, monkeypatch) -> dict:
        """The env load_model hands llama-server's Popen."""
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

        return seen["env"]  # absent key = the provider never passed one

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
        child_env = self._spawn_env(tmp_path, monkeypatch)
        assert "LLAMA_ARG_LOG_FILE" not in child_env
        assert child_env.get("LLAMA_ARG_CACHE_RAM") == "4096"

    @pytest.mark.parametrize("inherited", [None, "600"])
    def test_metal_keep_alive_is_set_unless_inherited(
            self, tmp_path, monkeypatch, inherited):
        key = llama_mod.METAL_KEEP_ALIVE_ENV
        if inherited is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, inherited)
        child_env = self._spawn_env(tmp_path, monkeypatch)
        expected = inherited or str(LlamaServerProvider.METAL_RESIDENCY_KEEP_ALIVE_S)
        assert child_env.get(key) == expected
        # ggml-metal's counter is an atomic_int of 5 ms ticks: past this it
        # wraps negative and residency turns off entirely.
        assert 0 < LlamaServerProvider.METAL_RESIDENCY_KEEP_ALIVE_S < (2**31 - 1) // 200


# ---------------------------------------------------------------------------
# Sleep/wake timeout
# ---------------------------------------------------------------------------

class TestSleepWakeTimeout:
    """`--sleep-idle-seconds` frees the model but keeps the process, so the
    next request pays a full RELOAD before the first byte. On a large model
    that is minutes -- far past the 120s wedge-detection timeout."""

    # The reload budget (startup_timeout_s) applies iff sleep is configured
    # AND the server is asleep; otherwise the wedge timeout. `sleeping` None
    # means the real probe runs against an address nothing listens on: the
    # best-effort /props probe (an older llama-server without it, or one
    # mid-restart) must degrade to "awake" and the normal timeout, not a 500.
    @pytest.mark.parametrize("config, sleeping, expected", [
        ({}, None, llama_mod._SSE_READ_TIMEOUT_S),
        ({"sleep_idle_seconds": 60, "startup_timeout_s": 900.0}, False, llama_mod._SSE_READ_TIMEOUT_S),
        ({"sleep_idle_seconds": 60, "startup_timeout_s": 900.0}, True, 900.0),
        ({"sleep_idle_seconds": 60}, None, llama_mod._SSE_READ_TIMEOUT_S),
    ], ids=["sleep_not_configured", "awake_keeps_wedge_timeout", "sleeping_gets_reload_budget",
            "unreachable_props_does_not_raise"])
    def test_request_timeout(self, monkeypatch, config, sleeping, expected):
        p = make_provider(**config)
        p._base_url = "http://127.0.0.1:1"  # nothing listening
        if sleeping is not None:
            monkeypatch.setattr(p, "_is_sleeping", lambda: sleeping)
        assert p._request_timeout() == expected


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


class _RecordingProc(_FakeProc):
    """_FakeProc that records every wait() it is asked for.

    The teardown contract is "who waits", so the wait CALL is the observable
    both destructor tests are aimed at -- shared rather than pasted into each,
    which is the hand-copied shape this repo treats as a defect with a delay.
    """

    def __init__(self, pid=4242):
        super().__init__(pid)
        self.waits: list = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return super().wait(timeout)


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

    # A spawned process is registered; a deliberate unload signals the group
    # and deregisters it -- an unloaded model must not be killed again at
    # exit, since its pid may have been recycled by then.
    @pytest.mark.parametrize("unload", [False, True], ids=["spawn_registers", "unload_deregisters"])
    def test_registry_tracks_spawn_and_unload(self, monkeypatch, unload):
        p = make_provider()
        proc = _FakeProc()
        killed = []
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        p._register_proc(proc)
        if unload:
            p._proc = proc
            p.unload()
        assert (proc in llama_mod._ACTIVE_PROCS) is not unload
        assert bool(killed) is unload, "unload should still signal the group"

    def test_destructor_signals_without_waiting(self, monkeypatch):
        """A destructor must not block, on THIS provider too.

        v2.0.28 gave `unload` a `drain` argument and honoured it in MLX only;
        here it was accepted and ignored, with a comment saying so. But
        `BaseProvider.__del__` is inherited, so collecting a gguf provider ran
        SIGTERM -> wait(10) -> SIGKILL -> wait(5): up to 15 seconds on
        whatever thread the GC fired on. Both halves of that release's claim
        failed for gguf.
        """
        p = make_provider()
        proc = _RecordingProc()
        killed = []
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        p._register_proc(proc)
        p._proc = proc

        p.unload(drain=False)

        assert proc.waits == [], (
            f"unload(drain=False) waited on the subprocess: {proc.waits}")
        assert killed == [(proc.pid, signal.SIGTERM)], (
            "drain=False must still SIGNAL -- not signalling strands a whole "
            "llama-server holding GPU memory for the life of the parent, which "
            "is worse than an unreaped child"
        )
        assert proc in llama_mod._ACTIVE_PROCS, (
            "an unwaited process must stay registered: this call cannot know the "
            "server exited, so deregistering would leave nothing watching the "
            "pid. Safe because we did NOT wait -- the Popen is unreaped, so its "
            "poll() in _kill_orphans still speaks for this pid. That backstop is "
            "ONE more SIGTERM at exit, not an escalation -- it has no SIGKILL."
        )

    def test_deliberate_unload_still_waits(self, monkeypatch):
        """The other half of the same contract: a caller that CAN afford to
        wait still does, or drain=False stops being a distinction."""
        p = make_provider()
        proc = _RecordingProc()
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: None)
        p._register_proc(proc)
        p._proc = proc

        p.unload()

        assert proc.waits, (
            "the deliberate teardown path stopped waiting for the process to exit")
        assert proc not in llama_mod._ACTIVE_PROCS

    # _kill_orphans is the last line of defense: delete it and any exit path
    # that skips the lifespan shutdown (startup crash, second Ctrl-C) leaks
    # the subprocess. It SIGTERMs live groups only -- a pid that already
    # exited may belong to something else by now -- and empties the registry.
    @pytest.mark.parametrize("returncode, expected_kills", [
        (None, [(4242, signal.SIGTERM)]),
        (0, []),
    ], ids=["kills_leftover_group", "skips_already_dead"])
    def test_backstop_signals_live_groups_only(self, monkeypatch, returncode, expected_kills):
        proc = _FakeProc()
        proc._rc = returncode
        killed = []
        monkeypatch.setattr(llama_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(llama_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        llama_mod._ACTIVE_PROCS.add(proc)

        llama_mod._kill_orphans()

        assert killed == expected_kills
        assert not llama_mod._ACTIVE_PROCS

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
    # The cascade floor < model config < request, in llama.cpp's names.
    # - floor: max_tokens is ALWAYS sent -- llama's default is UNLIMITED.
    # - request_overrides_and_mapping: repetition_penalty travels as
    #   llama.cpp's repeat_penalty; no reasoning budget unless asked.
    # - model_max_tokens_beats_floor: the floor pre-seeds max_tokens, so a
    #   guarded "if not in merged" write could never fire and a model-level
    #   max_tokens silently fell back to the floor (code-review 2026-07-26).
    # - looping_model_tuned_per_model: gguf gained the per-model field in
    #   v2.0.32, having had none: the thinking overlay's removal must not cost
    #   the ABILITY, only the automatic default.
    @pytest.mark.parametrize("config, body, expected, absent", [
        ({}, {},
         {"temperature": GLOBAL_SAMPLER_FLOOR["temperature"],
          "max_tokens": GLOBAL_SAMPLER_FLOOR["max_tokens"],
          "stream": True, "stream_options": {"include_usage": True},
          "messages": [{"role": "user", "content": "hi"}]}, ()),
        ({}, dict(temperature=0.1, repetition_penalty=1.3, max_tokens=64, seed=7,
                  presence_penalty=1.5, top_k=20),
         {"temperature": 0.1, "repeat_penalty": 1.3, "max_tokens": 64, "seed": 7,
          "presence_penalty": 1.5, "top_k": 20},
         ("repetition_penalty", "reasoning_budget_tokens")),
        ({}, dict(thinking_budget_tokens=128), {"reasoning_budget_tokens": 128}, ()),
        ({"max_tokens": 8000}, {}, {"max_tokens": 8000}, ()),
        ({"max_tokens": 8000}, {"max_tokens": 64}, {"max_tokens": 64}, ()),
        ({"presence_penalty": 1.5}, {"enable_thinking": True}, {"presence_penalty": 1.5}, ()),
    ], ids=["floor_and_max_tokens_always_sent", "request_overrides_and_mapping",
            "reasoning_budget_only_when_asked", "model_max_tokens_beats_floor",
            "request_max_tokens_beats_model", "looping_model_tuned_per_model"])
    def test_payload_cascade(self, config, body, expected, absent):
        payload = make_provider(**config)._build_payload(req(**body))
        assert {k: payload.get(k) for k in expected} == expected
        assert not set(absent) & set(payload)

    def test_per_model_sampling_matches_mlx_and_reaches_the_payload(self):
        """Owner call 2026-09-25: a gguf model's own file tunes sampling as
        an MLX model's does. Every per-request sampler field MLX's config
        has, gguf's has; a validated value reaches llama-server, and a
        request field still beats it."""
        from heylook_llm.config import (EFFECT_PER_REQUEST, GGUFModelConfig,
                                        MLXModelConfig)
        from heylook_llm.samplers import EFFECTIVE_SAMPLER_KEYS

        def per_request(cls):
            return {name for name, f in cls.model_fields.items()
                    if name in EFFECTIVE_SAMPLER_KEYS
                    and (f.json_schema_extra or {}).get("effect") == EFFECT_PER_REQUEST}

        assert per_request(MLXModelConfig) <= per_request(GGUFModelConfig)
        cfg = GGUFModelConfig(model_path="/fake/model.gguf", temperature=0.3,
                              top_p=0.8, top_k=12, min_p=0.05,
                              repetition_penalty=1.1).model_dump()
        p = LlamaServerProvider("test-gguf", cfg, False)
        payload = p._build_payload(req())
        assert (payload["temperature"], payload["top_p"], payload["top_k"],
                payload["min_p"], payload["repeat_penalty"]) == (0.3, 0.8, 12, 0.05, 1.1)
        assert p._build_payload(req(temperature=0.9))["temperature"] == 0.9



    # The vendor layer: the model's OWN published settings (a real GGUF whose
    # header carries `general.sampling.*`) reach the REQUEST BODY through
    # `_build_payload`, and stay a layer: the request outranks it, and a key
    # the header does not publish still gets the floor rather than being
    # dropped. Each row also pins the header parse (`vendor_sampling`).
    #
    # Why through the provider: the version this replaces asserted the FLOOR
    # value against `make_provider()`, whose model_path does not exist -- so
    # `vendor_sampling` returned nothing and the assertion held whether or not
    # the provider consulted the vendor layer at all. Its second half called
    # `resolve_effective_sampling` with a literal vendor dict, exercising
    # samplers.py and never the provider; deleting the provider's `vendor=`
    # argument left the whole suite green. top_k is the one that bit: heylook
    # sends every sampler key explicitly, so llama.cpp's own read of this block
    # is overridden on every request and a missing layer silently sent the
    # floor. (The reporting side has its own guard,
    # `test_vendor_layer_reaches_the_report_on_every_engine`.)
    _FULL_HEADER = (("general.sampling.temp", "F32", 0.5),
                    ("general.sampling.top_p", "F32", 0.8),
                    ("general.sampling.top_k", "I32", 20))

    @pytest.mark.parametrize("header, body, vendor, expected", [
        (_FULL_HEADER, {},
         {"temperature": 0.5, "top_p": 0.8, "top_k": 20},
         {"temperature": 0.5, "top_p": 0.8, "top_k": 20}),
        (_FULL_HEADER, {"temperature": 0.1},
         {"temperature": 0.5, "top_p": 0.8, "top_k": 20},
         {"temperature": 0.1}),
        ((("general.sampling.temp", "F32", 0.5),), {},
         {"temperature": 0.5},
         {"temperature": 0.5, "top_p": GLOBAL_SAMPLER_FLOOR["top_p"]}),
    ], ids=["header_reaches_payload", "request_outranks_header", "floor_where_header_silent"])
    def test_the_vendor_layer_reaches_the_payload(self, tmp_path, header, body, vendor, expected):
        import helpers.gguf as g
        from heylook_llm.gguf_metadata import vendor_sampling

        f = g.write_gguf(tmp_path / "m.gguf", [("general.architecture", g.STR, "qwen3")] +
                         [(k, getattr(g, t), v) for k, t, v in header])
        assert vendor_sampling(f) == {k: pytest.approx(v) for k, v in vendor.items()}
        payload = make_provider(model_path=str(f))._build_payload(req(**body))
        assert {k: payload[k] for k in expected} == {k: pytest.approx(v) for k, v in expected.items()}

    # Thinking is a TEMPLATE kwarg, not a sampler change: enable_thinking
    # always travels in chat_template_kwargs and presence_penalty stays the
    # floor (it carried 1.5 until v2.0.32; that value was never measured and
    # contradicted the guidance of the family it came from).
    # - unset: an omitted enable_thinking means OFF on gguf, exactly as on
    #   MLX, and must travel as an explicit `false`. llama-server runs --jinja,
    #   so with no chat_template_kwargs it applies the GGUF's own template
    #   default, which is thinking-ON for gemma-4 / Qwen3.6 / DeepSeek-V4:
    #   one v3 checkbox meant opposite things per engine, with no way to turn
    #   thinking off on gguf. Asserting the sent VALUE is the point: a bare
    #   `"chat_template_kwargs" in payload` check passes on a payload that
    #   says true.
    @pytest.mark.parametrize("body, expected_kwargs, expected_presence", [
        ({"enable_thinking": True}, {"enable_thinking": True}, GLOBAL_SAMPLER_FLOOR["presence_penalty"]),
        ({"enable_thinking": False}, {"enable_thinking": False}, GLOBAL_SAMPLER_FLOOR["presence_penalty"]),
        ({}, {"enable_thinking": False}, 0.0),
    ], ids=["on", "off", "unset_is_explicit_off"])
    def test_thinking_reaches_the_template_and_changes_no_sampler_value(
            self, body, expected_kwargs, expected_presence):
        payload = make_provider()._build_payload(req(**body))
        assert payload["chat_template_kwargs"] == expected_kwargs
        assert payload["presence_penalty"] == expected_presence

    # Every content part is forwarded verbatim: llama-server decodes the media
    # itself (mmproj), so the payload builder must neither mangle nor drop a
    # part -- an audio part lost here is a silent text-only answer.
    @pytest.mark.parametrize("content", [
        pytest.param([
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ], id="multimodal_content_parts_pass_through"),
        pytest.param([
            {"type": "text", "text": "what do you hear?"},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        ], id="audio_part_forwards_verbatim"),
    ])
    def test_content_parts_pass_through(self, content):
        p = make_provider()
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
        # llama-server's prompt_tokens is the whole prompt; its cached part
        # rides beside it, already the CacheReport shape.
        assert (final.cache.prompt_tokens, final.cache.cached_tokens,
                final.cache.processed_tokens, final.cache.outcome) == (7, 2, 5, "reused")
        assert final.prompt_tps == 100.5
        assert final.generation_tps == 42.0
        # spec decode (present only when MTP/draft was active): drafted and
        # accepted are llama-server's own counts
        assert (final.spec.drafted, final.spec.accepted) == (12, 5)


    # A failure inside the stream raises GenerationFailed carrying the
    # server's message, and is never sized as a compute error. llama-server
    # reports a decode failure INSIDE the stream (the HTTP status was already
    # 200): `data: {"error": {...}}`. It has no `choices`, so the adapter used
    # to skip it and the stream ended as a zero-token end_turn -- an empty
    # reply for what was a Metal OOM.
    @pytest.mark.parametrize("frame, match", [
        ('data: {"error":{"code":500,"message":"template failed","type":"server_error"}}', "template"),
        ("data: {not json", None),
    ], ids=["error_frame", "malformed_frame"])
    def test_stream_failure_raises_generation_failed(self, monkeypatch, frame, match):
        from heylook_llm.providers.base import GenerationFailed
        p = make_provider()
        monkeypatch.setattr(llama_mod.ram_fit, "fit_for_config",
                            lambda cfg, **kw: pytest.fail("no sizing for a non-compute error"))
        with pytest.raises(GenerationFailed, match=match):
            list(p._stream_chunks(_stream_bytes(frame), abort_event=None))

    def test_compute_error_names_the_metal_ceiling_and_the_sysctl(self, monkeypatch):
        # "Compute error." is all llama-server says; the ggml OOM line is in a
        # log that is kept nowhere by default. The message the reader gets must
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

    # Exactly echo_chars leading content chars are dropped, across delta
    # boundaries, and a delta that was pure echo is swallowed rather than
    # emitted empty.
    # - one_delta: the delta's surplus space is real continuation.
    @pytest.mark.parametrize("deltas, echo, expected", [
        (["1, 2, 3, ", "4, 5"], "1, 2, 3,", " 4, 5"),
        (["1, 2", ", 3, 4"], "1, 2, 3", ", 4"),
        (["prefix", " tail"], "prefix", " tail"),
    ], ids=["echo_in_one_delta", "echo_spanning_deltas", "pure_echo_delta_swallowed"])
    def test_content_echo_is_stripped(self, deltas, echo, expected):
        frames = [
            'data: {"choices":[{"delta":{"content":' + json.dumps(d) + '},"index":0,"finish_reason":null}]}'
            for d in deltas
        ] + ["data: [DONE]"]
        chunks = self.collect(frames, echo_chars=len(echo))
        assert all(c.text or c.thinking or c.finish_reason or c.prompt_tokens for c in chunks)
        assert "".join(c.text for c in chunks) == expected

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

    def test_a_closed_thoughts_framing_newline_is_not_new_thinking(self):
        # Save & Continue with an edited thought AND a partial reply: the
        # thought is closed in the prompt, and llama-server's reasoning echo
        # carries the template's newline before </think> (frames as sent by
        # the Qwen3.8-27B gguf, 2026-09-25). Without the drop the stored
        # thought gained a trailing "\n". An OPEN thought (no content
        # prefill) resumes, and a leading newline there is real.
        thought, prefix = "The user is asking. I should answer.", "I sound like"
        frames = [
            'data: {"choices":[{"delta":{"reasoning_content":' + json.dumps(thought + "\n") + '},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":' + json.dumps(prefix) + '},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" a goat"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        p = make_provider()
        chunks = list(p._stream_chunks(_stream_bytes(*frames), abort_event=None,
                                       echo_chars=len(prefix), echo_thinking_chars=len(thought)))
        assert "".join(c.thinking or "" for c in chunks) == ""
        assert "".join(c.text for c in chunks) == " a goat"

        open_frames = [
            'data: {"choices":[{"delta":{"reasoning_content":' + json.dumps(thought + "\nmore") + '},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        chunks = list(p._stream_chunks(_stream_bytes(*open_frames), abort_event=None,
                                       echo_chars=0, echo_thinking_chars=len(thought)))
        assert "".join(c.thinking or "" for c in chunks) == "\nmore"


class TestContinuationGuards:
    """What llama-server cannot express must 400, not silently do the wrong
    thing: user-role continuation has no llama-server spelling, and a trailing
    assistant message is ALWAYS continued (so false cannot be honored)."""

    def _req(self, messages, flag):
        from heylook_llm.config import ChatRequest
        return ChatRequest(model="m", messages=messages, continue_final_message=flag)

    def _payload(self, req):
        return {"messages": [m.model_dump(exclude_none=True) for m in req.messages]}

    # What llama-server cannot express is an InvalidGenerationRequest (400).
    @pytest.mark.parametrize("messages, flag, match", [
        ([{"role": "user", "content": "finish my sentence"}], True, "assistant turns only"),
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "he"}], False,
         "always continues"),
    ], ids=["user_role_continuation", "false_with_trailing_assistant"])
    def test_inexpressible_continuation_400s(self, messages, flag, match):
        from heylook_llm.providers.base import InvalidGenerationRequest
        p = make_provider()
        req = self._req(messages, flag)
        with pytest.raises(InvalidGenerationRequest, match=match):
            p._continuation_echo_chars(req, self._payload(req))

    # (content, thinking) echo lengths: the prefill's length for a trailing
    # assistant turn (auto), (0, 0) with nothing to continue.
    @pytest.mark.parametrize("messages, expected", [
        ([{"role": "user", "content": "count"}, {"role": "assistant", "content": "1, 2,"}], (5, 0)),
        ([{"role": "user", "content": "hi"}], (0, 0)),
    ], ids=["auto_trailing_assistant", "no_continuation"])
    def test_echo_length(self, messages, expected):
        p = make_provider()
        req = self._req(messages, None)
        assert p._continuation_echo_chars(req, self._payload(req)) == expected

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

    # An override is used, and WARNs naming its source and the canonical build
    # it shadows: $HEYLOOK_LLAMA_SERVER, or a models.toml server_binary.
    @pytest.mark.parametrize("source", ["HEYLOOK_LLAMA_SERVER", "server_binary"])
    def test_an_override_is_used_and_warns(self, tmp_path, monkeypatch, caplog, source):
        import logging as _logging
        canonical = self._with_canonical(tmp_path, monkeypatch)
        override = tmp_path / "elsewhere" / "llama-server"
        override.parent.mkdir(parents=True)
        override.write_text("#!/bin/true\n")
        if source == "HEYLOOK_LLAMA_SERVER":
            monkeypatch.setenv("HEYLOOK_LLAMA_SERVER", str(override))
            p = make_provider()
        else:
            monkeypatch.delenv("HEYLOOK_LLAMA_SERVER", raising=False)
            p = make_provider(server_binary=str(override))
        with caplog.at_level(_logging.WARNING):
            assert p._resolve_binary() == override
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno >= _logging.WARNING and source in r.getMessage()]
        assert warnings, f"an override from {source} must WARN naming its source"
        assert any(str(canonical) in m for m in warnings), \
            "the warning must NAME the canonical build being shadowed"


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

    # reasoning_effort rides chat_template_kwargs regardless of thinking;
    # absent leaves the template's own default; request > model default.
    # - thinking_off: NOT gated on enable_thinking. gpt-oss/harmony reads
    #   reasoning_effort unconditionally and has no enable_thinking at all,
    #   so gating made the knob unreachable for the one family the docs name
    #   as taking low|medium|high. A template that ignores the variable is
    #   unaffected -- jinja forwards unknown kwargs as template variables.
    # - effort_alone: the harmony shape, depth set, thinking never mentioned.
    # - model_level_default: the third route the CHANGELOG claims (per request
    #   / per preset / per model). It depends on the single line added to
    #   EFFECTIVE_SAMPLER_KEYS, so it can regress silently.
    @pytest.mark.parametrize("config, body, expected", [
        (None, dict(enable_thinking=True, reasoning_effort="low"),
         {"enable_thinking": True, "reasoning_effort": "low"}),
        (None, dict(enable_thinking=False, reasoning_effort="low"),
         {"enable_thinking": False, "reasoning_effort": "low"}),
        (None, dict(reasoning_effort="high"),
         {"enable_thinking": False, "reasoning_effort": "high"}),
        (None, dict(enable_thinking=True),
         {"enable_thinking": True}),
        ({"reasoning_effort": "medium"}, dict(enable_thinking=True),
         {"enable_thinking": True, "reasoning_effort": "medium"}),
        ({"reasoning_effort": "medium"}, dict(reasoning_effort="low"),
         {"enable_thinking": False, "reasoning_effort": "low"}),
    ], ids=["rides_chat_template_kwargs", "sent_with_thinking_off", "effort_alone",
            "absent_leaves_template_default", "model_level_default", "request_beats_model_default"])
    def test_effort_reaches_the_template(self, config, body, expected):
        assert self._payload(config, max_tokens=32, **body)["chat_template_kwargs"] == expected

    def test_the_templates_own_list_decides_and_names_the_variable(self):
        """Plan W2: a value the in-force template does not offer is refused
        before it reaches llama-server (which answers a raised jinja
        exception with a 500), and an offered one is sent under the
        template's OWN variable name."""
        from heylook_llm.providers.base import InvalidGenerationRequest

        p = make_provider()
        p.loaded_chat_template = (
            "{% if reasoning_strength not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{% endif %}"
            "Strength: {{ reasoning_strength | default('high') }}"
            "{% for m in messages %}{{ m.content }}{% endfor %}")
        with pytest.raises(InvalidGenerationRequest, match="low, high"):
            p._build_payload(req(reasoning_effort="xtreme", max_tokens=32))
        kw = p._build_payload(req(reasoning_effort="low", max_tokens=32))["chat_template_kwargs"]
        assert kw["reasoning_strength"] == "low" and "reasoning_effort" not in kw


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

    # Nothing is acquired until the generator is driven; the gate is held
    # across the stream and released on every exit: exhaustion, a forward
    # that fails, and an early close.
    @pytest.mark.parametrize("exit_by", ["exhaustion", "forward_failure", "early_close"])
    def test_gate_is_released_on_every_exit(self, monkeypatch, exit_by):
        p = self._gated(monkeypatch)
        if exit_by == "forward_failure":
            def timed_out(*a, **k):
                raise urllib.error.URLError("timed out")
            monkeypatch.setattr(llama_mod.urllib.request, "urlopen", timed_out)
        gen = p.create_chat_completion(req())
        assert p._gen_gate.busy is False, "nothing is acquired until the generator is driven"
        if exit_by == "forward_failure":
            with pytest.raises(GenerationFailed):
                list(gen)
        else:
            next(gen)
            assert p._gen_gate.busy is True, "held across the stream"
            gen.close() if exit_by == "early_close" else list(gen)
        assert p._gen_gate.busy is False, f"not released on {exit_by}"

    def test_the_model_counts_as_busy_while_llama_server_is_still_prefilling(self, monkeypatch):
        # llama-server answers urlopen only once it has a first result, so the
        # prefill happens INSIDE urlopen; a router asking "is this model
        # generating?" in that window must hear yes, or loading another model
        # evicts (SIGTERMs) this one mid-request.
        p = self._gated(monkeypatch)
        seen = {}

        def prefilling(*a, **k):
            seen["active"] = p.active_generations
            return _stream_bytes(*CANNED)

        monkeypatch.setattr(llama_mod.urllib.request, "urlopen", prefilling)
        chunks = list(p.create_chat_completion(req()))
        assert seen["active"] == 1
        assert p.active_generations == 0, "released when the stream ends"
        # the gate wait rides the first chunk, as on MLX (build_performance
        # nets it out of the generation span); an idle gate measures a tiny
        # positive wait, never the unmeasured zero
        assert chunks[0].queue_wait_ms > 0

    def test_a_waiter_cancelled_while_queued_never_forwards(self, monkeypatch):
        # Reported 2026-09-13: the gate was taken without a cancel check, so a
        # request abandoned while queued still took its turn and forwarded to
        # llama-server. The MLX chat path already passed the check; gguf did not.
        from heylook_llm.providers.abort import AbortEvent

        p = self._gated(monkeypatch)
        monkeypatch.setattr(
            llama_mod.urllib.request, "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("forwarded a cancelled request")),
        )
        p._gen_gate = GenerationGate(max_waiting=1)
        p._gen_gate.acquire()  # someone else is generating
        try:
            abort = AbortEvent()
            abort.set()  # the client is already gone
            assert list(p.create_chat_completion(req(), abort_event=abort)) == []
            assert p._gen_gate.snapshot()["waiting"] == 0, "left the queue"
            assert p._gen_gate.busy is True, "did not steal or release the other run's slot"
        finally:
            p._gen_gate.release()

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

    def test_a_gguf_generation_makes_an_mlx_model_answer_busy(self, monkeypatch):
        """One GPU, one queue across engines: while a gguf stream is in
        flight, an MLX model's admission check answers busy (503), and
        admits again once the stream ends. A gate per provider would let a
        gguf run and an MLX run overlap, which is the concurrency the gate
        exists to prevent."""
        from heylook_llm import router as router_mod
        from heylook_llm.providers.common.generation_gate import reset_process_gate

        if router_mod.MLXProvider is None:
            pytest.skip("mlx not importable")
        reset_process_gate()
        try:
            gguf = make_provider(max_queue_depth=0)  # single-flight gate
            gguf._base_url = "http://127.0.0.1:1"   # "loaded"
            monkeypatch.setattr(llama_mod.urllib.request, "urlopen",
                                lambda *a, **k: _stream_bytes(*CANNED))
            mlx = router_mod.MLXProvider("m-mlx", {"model_path": "/fake/mlx"}, False)
            mlx.check_capacity()  # idle: admitted
            stream = gguf.create_chat_completion(req())
            next(stream)  # the gguf generation is running
            with pytest.raises(ModelBusyError):
                mlx.check_capacity()
            assert mlx.generation_queue_stats()["active"] == 1
            stream.close()
            mlx.check_capacity()  # released: admitted again
        finally:
            reset_process_gate()



@pytest.mark.unit
class TestRunningContext:
    """``running_ctx`` is what the process GOT, read from /props at ready.

    The parse is guarded, not indexed: a build that moves the key must
    degrade to "unknown", never to a load failure, and a bool or a zero is
    not a context.
    """

    @pytest.mark.parametrize("props, expected", [
        ({"default_generation_settings": {"n_ctx": 32768, "id": 0}, "total_slots": 1}, 32768),
        ({}, None), ({"default_generation_settings": {}}, None),
        ({"default_generation_settings": None}, None),
        ({"default_generation_settings": {"n_ctx": 0}}, None),
        ({"default_generation_settings": {"n_ctx": True}}, None),
        ({"default_generation_settings": {"n_ctx": "32768"}}, None),
        ([], None), (None, None),
    ])
    def test_ctx_from_props(self, props, expected):
        got = LlamaServerProvider._ctx_from_props(props)
        assert got == expected and type(got) is type(expected)


# ---------------------------------------------------------------------------
# Prefill progress (v1.79.65): return_progress on the payload, prompt_progress
# frames reported up the request's signal channel in the cross-engine meaning
# ---------------------------------------------------------------------------

class TestPrefillProgress:
    def test_payload_asks_for_progress(self):
        p = make_provider()
        assert p._build_payload(req())["return_progress"] is True

    # prompt_progress frames report (processed - cache, total - cache) up the
    # request's signal channel -- llama-server's processed INCLUDES the cache
    # hit; the signal carries the work this request runs -- and yield nothing.
    # A fully cached prompt reports nothing; no signal channel is fine.
    @pytest.mark.parametrize("progress, with_signals, expected_progress, expected_texts", [
        ({"total": 100, "cache": 20, "processed": 60, "time_ms": 5}, True, (40, 80), ["Hi"]),
        ({"total": 50, "cache": 50, "processed": 50, "time_ms": 0}, True, None, []),
        ({"total": 10, "cache": 0, "processed": 5, "time_ms": 1}, False, None, []),
    ], ids=["reports_work_only_and_yields_nothing", "fully_cached_reports_nothing", "no_signal_channel"])
    def test_progress_frames(self, progress, with_signals, expected_progress, expected_texts):
        from heylook_llm.providers.abort import AbortEvent
        signals = AbortEvent() if with_signals else None
        frames = ['data: {"choices":[{"delta":{},"index":0,"finish_reason":null}],'
                  '"prompt_progress":' + json.dumps(progress) + '}']
        frames += ['data: {"choices":[{"delta":{"content":' + json.dumps(t) + '},"index":0,"finish_reason":null}]}'
                   for t in expected_texts]
        chunks = list(make_provider()._stream_chunks(_stream_bytes(*frames), abort_event=signals))
        assert [c.text for c in chunks] == expected_texts
        if signals is not None:
            assert signals.prefill_progress() == expected_progress


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

    # Dropped iff the render carries fewer media markers than media parts.
    # - two_images_one_marker: an undercount is the failure mode, so one
    #   marker for two images reads the same as none for one.
    @pytest.mark.parametrize("rendered, messages, dropped", [
        ("<|turn>model\nThe three colours are red,",
         [{"role": "assistant", "content": [IMG, {"type": "text", "text": "x"}]}], True),
        ("<|turn>model\n<__media_abc__>The three colours are red,",
         [{"role": "assistant", "content": [IMG, {"type": "text", "text": "x"}]}], False),
        ("<|turn>user\n<__media_a__>only one marker",
         [{"role": "user", "content": [IMG, IMG]}], True),
    ], ids=["dropped_marker_detected", "kept_marker_allowed", "two_images_one_marker"])
    def test_media_markers_dropped(self, monkeypatch, rendered, messages, dropped):
        p = self._answering(monkeypatch, rendered)
        assert p._media_markers_dropped(self._payload(messages)) is dropped

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


@pytest.mark.unit
class TestDrafterGivesWayToFit:
    """Spec decode is on wherever a drafter ships, but a drafter that turns a
    model that fits into one that does not is dropped at spawn, never the
    model. Decided against live reclaimable RAM (ram_fit)."""

    def _sizes(self, monkeypatch, with_drafter, alone, reclaimable=160.0):
        from types import SimpleNamespace
        from heylook_llm import ram_fit

        def fake(config, **_):
            gb = with_drafter if config.get("draft_model_path") else alone
            return SimpleNamespace(weights_gb=gb, headroom_gb=8.0, reclaimable_gb=reclaimable)
        monkeypatch.setattr(ram_fit, "fit_for_config", fake)

    # Dropped only when the drafter ALONE makes the model not fit.
    @pytest.mark.parametrize("with_drafter, alone, skipped", [
        (162.5, 150.4, "short by 10.5 GiB"),
        (150.0, 140.0, None),   # both fit: keep it
        (170.0, 160.0, None),   # neither fits: not the drafter's fault
    ], ids=["short_only_because_of_drafter", "both_fit", "neither_fits"])
    def test_drafter_gives_way_only_when_it_alone_breaks_fit(self, monkeypatch, with_drafter, alone, skipped):
        p = make_provider(draft_model_path="/fake/dspark.gguf", spec_type="draft-dspark")
        self._sizes(monkeypatch, with_drafter, alone)
        p._drop_drafter_if_short()
        if skipped:
            assert "draft_model_path" not in p.config and "spec_type" not in p.config
            assert skipped in p.drafter_skipped
        else:
            assert p.config["draft_model_path"] == "/fake/dspark.gguf" and p.drafter_skipped is None


# Discovery pairs drafters the build may not run (a split-out MTP head with no
# embeddings). A load that exits with a drafter set is retried once without
# it, and the next load of the same drafter skips straight there. A load
# failure with no drafter is not retried.
@pytest.mark.unit
@pytest.mark.parametrize("drafter", ["/fake/mtp-shared.gguf", None], ids=["with_drafter", "without_drafter"])
def test_a_drafter_load_failure_costs_one_retry_not_the_model(monkeypatch, drafter):
    from heylook_llm.providers import llama_server_provider as lsp
    monkeypatch.setattr(lsp, "_UNLOADABLE_DRAFTERS", set())
    monkeypatch.setattr(LlamaServerProvider, "_resolve_binary", lambda self: Path("/fake/llama-server"))
    spawns = []

    def load_once(self):
        spawns.append(self.config.get("draft_model_path"))
        if self.config.get("draft_model_path") or drafter is None:
            raise lsp.LlamaServerLoadExit("llama-server exited with code 1 while loading")
    monkeypatch.setattr(LlamaServerProvider, "_load_once", load_once)

    if drafter is None:
        with pytest.raises(lsp.LlamaServerLoadExit):
            make_provider().load_model()
        assert spawns == [None]
        return

    p = make_provider(draft_model_path=drafter, spec_type="draft-mtp")
    p.load_model()
    assert spawns == [drafter, None]
    assert "draft_model_path" not in p.config and "spec_type" not in p.config
    assert "retried without it" in p.drafter_skipped

    spawns.clear()
    make_provider(draft_model_path=drafter).load_model()
    assert spawns == [None]


def test_flash_attn_reports_what_auto_resolved_to():
    """Unset flash_attn is llama-server's auto; the row shows what it resolved
    to, read from its log. The lines are libllama's own
    (llama-context.cpp resolve_fused_ops and the forced cases); the first one
    is the target model's context, so a drafter's later line cannot flip it."""
    from heylook_llm.providers.gguf_describe import flash_attn_setting
    from heylook_llm.providers.llama_server_provider import SpawnLog

    log = SpawnLog()
    for line in ("llama_context: flash_attn            = auto\n",
                 "resolve_fused_ops: Flash Attention enabled\n",
                 "resolve_fused_ops: Flash Attention not supported, set to disabled\n"):
        log.note_line(line)
    assert log.flash_attn == "on"
    forced = SpawnLog()
    forced.note_line("llama_init_from_model: enabling flash_attn since it is required for quantized V cache\n")
    assert forced.flash_attn == "on"

    seen = flash_attn_setting(None, "on", loaded=True)
    assert (seen.value, seen.auto, seen.provenance) == ("on", "on", "observed")
    assert flash_attn_setting(None, None, loaded=True).provenance == "unknown"
    assert flash_attn_setting(None, None, loaded=False).value == "auto"
    chosen = flash_attn_setting("off", "on", loaded=True)
    assert (chosen.value, chosen.configured, chosen.provenance) == ("off", "off", "configured")


def test_gguf_metrics_report_what_is_known_and_null_the_rest(tmp_path, monkeypatch):
    """gguf was absent from /v1/system/metrics (no get_metrics), so the perf
    page said "No models loaded" with a gguf model resident. Read through the
    route, with the model loaded by the router and faked only at the process
    edge (a live pid, /props, the chat stream). Memory is the llama-server
    process's footprint (this test's own process stands in for it); context
    used is null until a request ran, never 0; a process that exited reports
    memory unknown, not 0."""
    import logging
    import os

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from heylook_llm import monitoring_api
    from heylook_llm.providers.common.generation_gate import reset_process_gate
    from heylook_llm.router import ModelRouter

    proc = _FakeProc(pid=os.getpid())

    def urlopen(target, *a, **k):
        url = target if isinstance(target, str) else target.full_url
        if url.endswith("/props"):
            return _stream_bytes(json.dumps({"default_generation_settings": {"n_ctx": 8192}}))
        return _stream_bytes(*CANNED)  # prompt_tokens 7 + completion 3

    def load_model(self):  # the spawn, as far as the process edge goes
        self._proc, self._base_url = proc, "http://127.0.0.1:1"
        self.running_ctx = self._read_running_ctx()

    monkeypatch.setattr(llama_mod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(LlamaServerProvider, "load_model", load_model)
    monkeypatch.setattr(LlamaServerProvider, "unload", lambda self, **kw: None)
    monkeypatch.setattr(monitoring_api, "_metrics_collector", None)
    reset_process_gate()
    toml = tmp_path / "heylook.toml"
    toml.write_text('max_loaded_models = 1\n\n[[models]]\nid = "g1"\nprovider = "gguf"\n\n'
                    '[models.config]\nmodel_path = "/fake/g1.gguf"\n')
    router = ModelRouter(config_path=str(toml), log_level=logging.INFO, initial_model_id=None)
    app = FastAPI()
    app.include_router(monitoring_api.monitoring_router)
    app.state.router_instance = router
    client = TestClient(app)

    def row():
        res = client.get("/v1/system/metrics", params={"force_refresh": "true"})
        assert res.status_code == 200, res.text
        return res.json()["models"]["g1"]

    try:
        provider = router.get_provider("g1")
        m = row()
        assert m["memory_mb"] and m["memory_mb"] > 0 and "llama-server" in m["memory_source"]
        assert (m["context_capacity"], m["context_used"], m["context_percent"]) == (8192, None, None)
        list(provider.create_chat_completion(req()))
        m = row()
        assert (m["context_used"], m["context_percent"]) == (10, round(10 / 8192 * 100, 1))
        proc._rc = 0  # llama-server exited
        assert row()["memory_mb"] is None
    finally:
        reset_process_gate()
