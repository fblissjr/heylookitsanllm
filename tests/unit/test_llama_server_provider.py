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
# - payload tests: the sampler cascade (floor -> default_sampler -> named
#   sampler -> explicit request fields) or the llama.cpp param mapping
#   (repetition_penalty -> repeat_penalty, always-send max_tokens, thinking
#   -> chat_template_kwargs) silently drifts.
# - SSE adapter tests: llama-server frames (reasoning_content deltas, usage
#   chunk with timings, keepalive comments, [DONE]) stop mapping onto
#   GenerationChunk fields -- telemetry goes dark or thinking is dropped.

import io
import signal
import subprocess
import sys

import pytest

from heylook_llm.providers import llama_server_provider as llama_mod

from heylook_llm.config import ChatRequest, ModelConfig, GGUFModelConfig, PROVIDER_CONFIG_CLASSES
from heylook_llm.providers.base import GenerationChunk
from heylook_llm.providers.llama_server_provider import LlamaServerProvider


def req(**kw):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return ChatRequest.model_validate(body)


def make_provider(**config):
    config.setdefault("model_path", "/fake/model.gguf")
    return LlamaServerProvider("test-gguf", config, False)


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
        from heylook_llm.api import _infer_model_capabilities

        plain = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf", "config": {"model_path": "/x.gguf"}})
        assert _infer_model_capabilities(plain) == ["chat"]

        vision = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf",
             "config": {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf"}})
        assert "vision" in _infer_model_capabilities(vision)

        thinking = ModelConfig.model_validate(
            {"id": "m", "provider": "gguf",
             "config": {"model_path": "/x.gguf", "supports_thinking": True}})
        caps = _infer_model_capabilities(thinking)
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
        for flag in ("-ngld", "-cram", "--sleep-idle-seconds", "-lm"):
            assert flag not in args

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
        assert payload["temperature"] == 0.7
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

    def test_request_thinking_engages_antiloop_overlay(self):
        """Claim: gguf mirrors MLX -- a thinking request gets the slimmed
        'thinking' sampler's presence_penalty (loop control). Without it a
        thinking model runs floor sampling with zero repetition control."""
        p = make_provider()
        payload = p._build_payload(req(enable_thinking=True))
        assert payload["presence_penalty"] == 1.5
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}

    def test_request_thinking_off_no_penalty(self):
        p = make_provider()
        payload = p._build_payload(req(enable_thinking=False))
        assert payload["presence_penalty"] == 0.0  # floor value, no overlay
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_explicit_presence_penalty_beats_thinking_overlay(self):
        p = make_provider()
        payload = p._build_payload(req(enable_thinking=True, presence_penalty=0.3))
        assert payload["presence_penalty"] == 0.3

    def test_unknown_default_sampler_skips_not_raises(self):
        """Shared-resolver semantics: a default_sampler missing from the
        registry logs and skips (models validate at startup; a miss here is
        post-startup registry drift) -- it must not 400 every request."""
        p = make_provider(default_sampler="gone-from-registry")
        payload = p._build_payload(req())
        assert payload["temperature"] == 0.7

    def test_request_sampler_suppresses_default_sampler(self):
        """Shared-resolver semantics: naming a request sampler replaces the
        default_sampler layer; fields only the default set revert to floor."""
        p = make_provider(default_sampler="thinking")
        payload = p._build_payload(req(sampler="deterministic"))
        assert payload["presence_penalty"] == 0.0

    def test_default_sampler_overlays_floor(self):
        from heylook_llm.samplers import get_sampler_registry

        det_temp = get_sampler_registry()._presets["deterministic"]["temperature"]
        assert det_temp != 0.7  # must differ from the floor for this to prove anything
        p = make_provider(default_sampler="deterministic")
        payload = p._build_payload(req())
        assert payload["temperature"] == det_temp

    def test_named_request_sampler_beats_default_sampler(self):
        from heylook_llm.samplers import get_sampler_registry

        reg = get_sampler_registry()._presets
        assert reg["balanced"]["temperature"] != reg["deterministic"]["temperature"]
        p = make_provider(default_sampler="deterministic")
        payload = p._build_payload(req(sampler="balanced"))
        assert payload["temperature"] == reg["balanced"]["temperature"]

    def test_enable_thinking_maps_to_chat_template_kwargs(self):
        p = make_provider()
        on = p._build_payload(req(enable_thinking=True))
        off = p._build_payload(req(enable_thinking=False))
        unset = p._build_payload(req())
        assert on["chat_template_kwargs"] == {"enable_thinking": True}
        assert off["chat_template_kwargs"] == {"enable_thinking": False}
        assert "chat_template_kwargs" not in unset

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

    def test_malformed_frame_raises_generation_failed(self):
        from heylook_llm.providers.base import GenerationFailed

        p = make_provider()
        with pytest.raises(GenerationFailed):
            list(p._stream_chunks(_stream_bytes("data: {not json"), abort_event=None))
