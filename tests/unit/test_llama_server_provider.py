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
from heylook_llm.samplers import GLOBAL_SAMPLER_FLOOR
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
        for flag in ("-ngld", "-cram", "--sleep-idle-seconds", "-lm"):
            assert flag not in args

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
        provider = make_provider(chat_template_path=str(tmp_path / "nope.jinja"))
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
        provider = make_provider(chat_template_path=str(tmpl))
        with pytest.raises(RuntimeError, match="preflight passed"):
            provider.load_model()
        assert spawned, "preflight rejected a template file that exists"

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
        assert payload["temperature"] == GLOBAL_SAMPLER_FLOOR["temperature"]

    def test_request_sampler_suppresses_default_sampler(self):
        """Shared-resolver semantics: naming a request sampler replaces the
        default_sampler layer; fields only the default set revert to floor."""
        p = make_provider(default_sampler="thinking")
        payload = p._build_payload(req(sampler="deterministic"))
        assert payload["presence_penalty"] == 0.0

    def test_default_sampler_overlays_floor(self):
        from heylook_llm.samplers import get_sampler_registry

        det_temp = get_sampler_registry()._presets["deterministic"]["temperature"]
        assert det_temp != GLOBAL_SAMPLER_FLOOR["temperature"]  # must differ from the floor for this to prove anything
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
        assert payload["presence_penalty"] == 0.0  # off => no anti-loop overlay

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

    def test_thinking_deltas_are_never_stripped(self):
        frames = [
            'data: {"choices":[{"delta":{"reasoning_content":"hmm"},"index":0,"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"prefixreal"},"index":0,"finish_reason":null}]}',
            "data: [DONE]",
        ]
        chunks = self.collect(frames, echo_chars=len("prefix"))
        assert "".join(c.thinking or "" for c in chunks) == "hmm"
        assert "".join(c.text for c in chunks) == "real"


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
        assert p._continuation_echo_chars(req, self._payload(req)) == 5

    def test_no_continuation_returns_zero(self):
        p = make_provider()
        req = self._req([{"role": "user", "content": "hi"}], None)
        assert p._continuation_echo_chars(req, self._payload(req)) == 0

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
        assert chars == len("1, 2, 3,")

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
        assert p._continuation_echo_chars(req, payload) == 0
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

