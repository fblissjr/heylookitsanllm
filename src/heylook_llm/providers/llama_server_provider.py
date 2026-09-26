# src/heylook_llm/providers/llama_server_provider.py
#
# GGUF provider: one llama-server SUBPROCESS per loaded model (plan Phase 7b).
#
# Design (dossier: the gguf driving-models research doc):
# - load_model() spawns llama-server in its own process group and polls
#   /health (503 "Loading model" = warming; 200 = ready; process exit =
#   load failure). unload() SIGTERMs the group -- "loaded model" ==
#   "running subprocess", so router LRU/idle-unload just work.
# - create_chat_completion() streams the subprocess's OpenAI-compat
#   /v1/chat/completions over SSE and adapts frames to GenerationChunk.
#   llama-server pre-splits reasoning (reasoning_content deltas, --jinja
#   default-on) -> GenerationChunk.thinking; template_info() stays None so
#   heylook's parser stack is pass-through (never re-parse another engine's
#   output).
# - Sampler cascade IS MLX's: the shared samplers.resolve_effective_sampling
#   (floor -> VENDOR -> model fields -> explicit request fields).
#   The vendor layer comes from the GGUF HEADER (v2.0.22), not from a
#   generation_config.json -- a gguf dir carries no such file, but the
#   converter writes those very values into `general.sampling.*`, so the
#   layer has the same source and the same meaning as MLX's.
#   max_tokens is ALWAYS sent (llama-server's default is unlimited).
# - -np 1 by OUR choice (full context per slot, matches heylook's
#   serialized semantics) -- not a compat requirement.
# - Shares the process FIFO generation gate (v1.79.60). It used to rely on
#   llama-server queueing its own requests, and that queue is invisible to
#   heylook: a second request forwarded while the first was still generating
#   sat in it past the 120s read timeout and came back as a 500 "unreachable:
#   timed out", i.e. "this model is broken", for a backend that was merely
#   busy. With -np 1 there is exactly one slot, so heylook's own gate is the
#   correct model of it: acquire in arrival order, and answer the same
#   MODEL_BUSY -> 503 contract the MLX path does from check_capacity().
# - Pure stdlib (urllib/subprocess/socket): the provider must import and
#   run on machines with no MLX and no extra deps.

import atexit
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Generator, Optional

from .. import observability, ram_fit
from ..config import ChatRequest, GGUFModelConfig
from ..samplers import GLOBAL_SAMPLER_FLOOR, resolve_effective_sampling
from ..thinking_controls import depth_variable, detect as detect_thinking
from .common.generation_gate import GenerationCancelled, get_process_gate
# ONE filename for both engines, imported rather than re-spelled -- a second
# copy of a literal filename is a second place for the editor to write
# somewhere the loader does not look. template_info is stdlib+orjson only, so
# this keeps the gguf provider's no-MLX-import property.
from .common.template_info import HEYLOOK_OVERRIDE, HEYLOOK_TEMPLATE_FILENAME
from .llama_cache_witness import CacheWitness, fingerprint
from .base import (
    BaseProvider,
    CacheReport,
    GenerationChunk,
    GenerationFailed,
    InvalidGenerationRequest,
    SpecReport,
)

# Every live llama-server we spawned, so no exit path can leak one.
#
# We spawn with start_new_session=True (own process group, so unload can kill
# the whole tree). The cost of that isolation: the terminal's Ctrl-C sends
# SIGINT to the FOREGROUND process group only, which we are no longer in --
# so the subprocess survives its parent unless someone explicitly reaps it.
# The graceful path is lifespan shutdown -> router.unload_all() -> unload();
# this atexit hook is the backstop for exits that skip it (a crash during
# startup, a second Ctrl-C forcing uvicorn to quit). SIGKILL of the parent
# remains uncoverable -- nothing runs in that case.
_ACTIVE_PROCS: "set" = set()
# _ACTIVE_PROCS is touched from ARBITRARY THREADS and needs this lock. Its
# mutators are `_register_proc` (a spawn, on a request thread), `unload` (a
# router eviction or the idle-unload tick), `__del__` -> unload (whatever
# thread the GC fired on), and `_kill_orphans` (atexit). The sweep in
# `_register_proc` used to build its dead list by calling `poll()` INSIDE a
# comprehension over the set -- and poll() is a syscall, so the GIL is released
# mid-iteration and a concurrent discard raises "Set changed size during
# iteration" out of the spawn path, failing a model load for a reason nothing
# in the traceback connects to a different model being collected.
_ACTIVE_PROCS_LOCK = threading.Lock()


def _kill_orphans() -> None:
    """Reap any llama-server still registered at interpreter exit.

    Best-effort and silent: this runs during shutdown, where logging handlers
    may already be torn down and raising would be pointless.
    """
    while True:
        with _ACTIVE_PROCS_LOCK:
            if not _ACTIVE_PROCS:
                return
            proc = _ACTIVE_PROCS.pop()
        try:
            if proc.poll() is not None:
                continue  # already exited; its pid may be recycled by now
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            pass


atexit.register(_kill_orphans)

# Read timeout for the SSE response. llama-server emits a keepalive comment
# every 30s (sse_ping_interval default), so a healthy stream never blocks a
# read longer than that; 120s means "server wedged", not "model is slow".
_SSE_READ_TIMEOUT_S = 120.0

# The ONE env var that lets llama-server write a file on its OWN, routing
# around heylook's file-logging master switch -- removed from the child's
# environment at spawn (see load_model). It is the whole env-borne write
# surface: of llama.cpp's three options that write to disk, only --log-file
# carries a `.set_env` (common/arg.cpp); --log-prompts-dir (prompt TEXT) and
# --slot-save-path (KV cache) are CLI-only, so they cannot arrive this way and
# heylook passes neither. Every OTHER LLAMA_ARG_* is a behaviour knob someone
# may be setting deliberately -- those are warned about, never removed.
_ENV_STRIPPED_AT_SPAWN = ("LLAMA_ARG_LOG_FILE",)

# llama-server reads three env vars that do NOT carry the LLAMA_ARG_ prefix
# the warning loop keys on, so they were invisible to it. Two are worth
# naming; both are surfaced, not stripped, like every other behaviour knob.
#
# LLAMA_API_KEY is the one that BREAKS us: set, llama-server demands a bearer
# token heylook does not send, and /health is a PUBLIC endpoint (server.cpp),
# so load_model polls 200, the model reports READY, and every generation then
# 401s. `_read_running_ctx` swallows its own 401, so context_running silently
# reads None as well. Nothing else in the stack names the cause.
# MTMD_BACKEND_DEVICE moves the mmproj tower to another backend.
# HF_TOKEN is deliberately NOT here: it is near-ubiquitous, heylook passes
# local paths so llama-server never downloads, and warning on it is noise.
_ENV_SURFACED_UNPREFIXED = ("LLAMA_API_KEY", "MTMD_BACKEND_DEVICE")

# ggml-metal reads this at device init (ggml-metal-device.m). heylook sets it
# at spawn unless the environment already does; see load_model.
METAL_KEEP_ALIVE_ENV = "GGML_METAL_RESIDENCY_KEEP_ALIVE_S"

# llama.cpp applies these BEFORE both env and CLI (common/arg.cpp: "config
# file applies first, so env variables and CLI arguments override it"), and
# their keys dispatch into the same arg handlers -- so a `log-file` line in
# one is a spawn setting heylook neither passes nor can override, since our
# only counter would be passing --log-file ourselves, which redirects the
# stream we capture. Their existence is therefore REPORTED at spawn: heylook
# cannot own the log destination in their presence and should not claim to.
def _llama_system_config_paths() -> list[Path]:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    user_dir = Path(xdg) if xdg else Path.home() / ".config"
    return [Path("/etc/llama.cpp/config.ini"), user_dir / "llama.cpp" / "config.ini"]

# ggml's quantized cache types (config.cache_type_v's Literal minus the float
# ones). A quantized V cache cannot run with flash attention off: llama.cpp
# refuses the context (llama-context.cpp, "quantized V cache requires
# flash_attn to be enabled").
_FLOAT_CACHE_TYPES = frozenset({"f32", "f16", "bf16"})


def effective_flash_attn(cfg: dict) -> tuple[str, str]:
    """The -fa value a spawn passes, and why. The ONE decision: the spawn
    and the engine report both call it.

    Off unless the model's config says otherwise (owner, 2026-09-26: never on
    by default for gguf). The one exception is a quantized V cache, which
    llama.cpp cannot run without it, so the default there is on rather than a
    model that fails to load. `auto` is an explicit choice (llama-server's own
    device probe)."""
    configured = cfg.get("flash_attn")
    if configured:
        return configured, f"set to {configured} for this model"
    ctv = cfg.get("cache_type_v")
    if ctv and ctv not in _FLOAT_CACHE_TYPES:
        return "on", f"on: the {ctv} V cache requires it (llama.cpp refuses it off)"
    return "off", "off: heylook's default for gguf models; set it per model to change"


def flash_attn_from_log(line: str) -> Optional[str]:
    """"on"/"off" when a llama-server log line settles flash attention, else
    None. Under auto, libllama's device probe logs "Flash Attention enabled"
    or "... not supported, set to disabled" (llama-context.cpp
    ``resolve_fused_ops``); a quantized V cache or tensor split forces it on,
    and Grok forces it off, each with its own line."""
    if "Flash Attention enabled" in line or "enabling flash_attn since" in line:
        return "on"
    if "Flash Attention not supported" in line or "flash_attn is not compatible" in line:
        return "off"
    return None


class SpawnLog:
    """Load facts only llama-server's own log states, read off the pump.
    The first flash-attention line is the target model's context; a drafter
    or the projector may log their own after it."""

    def __init__(self) -> None:
        self.flash_attn: Optional[str] = None

    def note_line(self, line: str) -> None:
        if self.flash_attn is None:
            self.flash_attn = flash_attn_from_log(line)


def _pump_output(stream, tee, *sinks) -> None:
    """Drain llama-server's output pipe for the life of the process.

    Always drained, whatever the observability level: a pipe nobody reads
    fills and then blocks llama-server mid-write. Each line goes to the log
    file when file logging was on at spawn (``tee``), and to each sink's
    ``note_line`` (the cache witness, which keeps only the cache events it
    recognizes, and the ``SpawnLog``). Nothing here may raise out of the
    loop: a dead pump is a wedged server.
    """
    try:
        for raw in iter(stream.readline, b""):
            if tee is not None:
                try:
                    tee.write(raw)
                    tee.flush()
                except Exception:  # noqa: BLE001 - closed at unload; keep draining
                    tee = None
            line = raw.decode("utf-8", "replace")
            for sink in sinks:
                try:
                    sink.note_line(line)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001 - the stream closed under us at teardown
        pass
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


# cascade key -> llama-server request key
_PAYLOAD_KEY_MAP = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("min_p", "min_p"),
    ("repetition_penalty", "repeat_penalty"),
    # The penalty window. No floor sets it, so it goes out only when a layer
    # does; llama.cpp's window counts the prompt's tail too (see
    # GGUFModelConfig.presence_penalty), MLX's only the reply.
    ("repetition_context_size", "repeat_last_n"),
    ("presence_penalty", "presence_penalty"),
    ("seed", "seed"),
    # Per-request since llama.cpp's reasoning-budget sampler; applied only
    # where llama-server found the template's thinking end tags.
    ("thinking_budget_tokens", "reasoning_budget_tokens"),
)


class LlamaServerLoadExit(RuntimeError):
    """llama-server exited before it was ready."""


# (drafter path, binary) pairs this process has seen llama-server refuse.
_UNLOADABLE_DRAFTERS: set = set()


class LlamaServerProvider(BaseProvider):
    """Serve a GGUF model through a managed llama-server subprocess."""

    provider_name = "gguf"

    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None  # no in-process model object; MLX-only surfaces gate on this
        self.processor = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None
        # Per process: a respawn starts with empty caches, so a fresh witness.
        self._cache_witness: Optional[CacheWitness] = None
        self._spawn_log: Optional[SpawnLog] = None
        # The last request's context (prompt plus reply), None until one ran.
        self._context_used: Optional[int] = None
        self._base_url: Optional[str] = None
        # The model's own recommended decode settings, read from the GGUF
        # header once and cached. None = not read yet (the file cannot change
        # under a loaded model, and re-reading per request would stat a
        # multi-GB file on the hot path).
        self._vendor_sampling: Optional[Dict] = None
        # The context the RUNNING process was sized to, read from /props once
        # it is ready. Config carries what was ASKED (`ctx_size`, absent =
        # llama-server's own model-derived, memory-fitted default); this is
        # what it GOT, which is the number a "context size" control has to
        # show for the Auto case, or it is offering a choice with a blank
        # default. None until ready, and None if /props does not say.
        self.running_ctx: Optional[int] = None
        # The process-wide gate (one GPU), same config key as MLX.
        self._gen_gate = get_process_gate(int(self.config.get("max_queue_depth", 8)))

    def check_capacity(self) -> None:
        """Reject (ModelBusyError -> 503) when the FIFO queue is already full.

        Same contract as MLXProvider.check_capacity. Before v1.79.60 this was
        the base no-op and a busy llama-server surfaced as a 500 after the read
        timeout, which a client cannot tell from a broken model.
        """
        self._gen_gate.check_capacity()

    def generation_queue_stats(self) -> dict:
        """Snapshot of the FIFO generation queue (active/waiting/capacity)."""
        return self._gen_gate.snapshot()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # Canonical local build location, checked when nothing overrides it. A
    # llama.cpp checkout built here (fixed dir under home, outside the repo)
    # is picked up with zero config; the literal path is pinned by a test.
    DEFAULT_BUILD = Path.home() / ".heylook" / "llama.cpp" / "build" / "bin" / "llama-server"

    @classmethod
    def binary_choice(cls, cfg: Dict) -> tuple[Optional[Path], str, bool]:
        """``(path, source, is_override)``: which llama-server a spawn would
        use, with no logging and no raising. ``path`` is None only when
        nothing is configured and no canonical build exists. The one
        precedence rule, shared by the spawn and the engine contract."""
        candidate = cfg.get("server_binary")
        source = "heylook.toml server_binary"
        if not candidate:
            candidate = os.environ.get("HEYLOOK_LLAMA_SERVER")
            source = "$HEYLOOK_LLAMA_SERVER"
        if not candidate:
            return (cls.DEFAULT_BUILD if cls.DEFAULT_BUILD.is_file() else None,
                    "the canonical build", False)
        return Path(candidate).expanduser(), source, True

    @classmethod
    def canonical_build_label(cls) -> Optional[str]:
        """The canonical build's commit and build number from its manifest,
        e.g. "4e416ee7 (build 11138)", or None without a manifest."""
        manifest = cls.DEFAULT_BUILD.parents[1] / "heylook-build.json"
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            return None
        sha = str(data.get("sha") or data.get("rev") or "")[:8]
        m = re.search(r"build (\d+)", str(data.get("version") or ""))
        return f"{sha} (build {m.group(1)})" if m else sha or None

    @classmethod
    def binary_report(cls, cfg: Dict) -> tuple[Optional[str], str]:
        """``(value, reason)`` naming the binary for the engine contract:
        the canonical build by commit, an override by file name only (no
        absolute paths leave the server this way)."""
        path, source, is_override = cls.binary_choice(cfg)
        if path is None:
            return None, "no llama-server binary: run scripts/build_llama.py"
        if not is_override:
            label = cls.canonical_build_label()
            return (f"canonical build {label}" if label else "canonical build",
                    "no server_binary or $HEYLOOK_LLAMA_SERVER override")
        return path.name, f"{source}, overriding the canonical build"

    @classmethod
    def keep_alive_choice(cls, environ) -> tuple[str, str, bool]:
        """``(value, reason, inherited)`` for GGML_METAL_RESIDENCY_KEEP_ALIVE_S:
        an inherited value wins; otherwise heylook's process-lifetime value."""
        inherited = environ.get(METAL_KEEP_ALIVE_ENV)
        if inherited is not None:
            return (inherited, f"{METAL_KEEP_ALIVE_ENV} set in heylook's "
                               f"environment; it wins over heylook's value", True)
        return (str(cls.METAL_RESIDENCY_KEEP_ALIVE_S),
                "the life of the process: heylook's idle unload is what ends it",
                False)

    def _resolve_binary(self) -> Path:
        """server_binary > $HEYLOOK_LLAMA_SERVER > the canonical build.

        The canonical build (DEFAULT_BUILD -- written ONLY by
        scripts/build_llama.py, updated with one command) is the intended
        single source (owner rule 2026-08-13: one llama-server build, no
        silent shadowing). The two overrides remain as escape hatches for
        experiments -- but they are LOUD now: both rot silently otherwise
        (an exported env var keeps pointing at an old build after a newer
        one lands; exactly that shadowed a fresh canonical build on
        2026-08-13, and the load failure on a new model arch was the first
        anyone heard of it). An override WARNS at every spawn, naming its
        source and the canonical build it beats, so it can never again be
        the thing nobody remembers is set.
        """
        path, source, is_override = self.binary_choice(self.config)
        if not is_override and path is not None:
            logging.info(f"[GGUF] using the canonical build at {self.DEFAULT_BUILD}")
            return path
        if path is None:
            raise RuntimeError(
                "No llama-server binary configured. Build one with "
                "`uv run scripts/build_llama.py`, or set server_binary in "
                "heylook.toml / the $HEYLOOK_LLAMA_SERVER env var (a Homebrew "
                "or upstream release binary works too)."
            )
        if not path.is_file():
            raise RuntimeError(
                f"llama-server binary not found at '{path}'. Build it "
                f"(cmake --build ... --target llama-server) or fix "
                f"server_binary / $HEYLOOK_LLAMA_SERVER."
            )
        shadow = (f" -- OVERRIDING the canonical build at {self.DEFAULT_BUILD}"
                  if self.DEFAULT_BUILD.is_file() else "")
        logging.warning(f"[GGUF] llama-server binary from {source}: {path}{shadow}")
        return path

    @staticmethod
    def _free_port() -> int:
        # Tiny bind race window, single-user localhost: acceptable, and it
        # avoids a stdout-parsing reader thread for llama's --port 0 mode.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    # A template that handles media has to BRANCH on a content part's type,
    # and that structure is the same across families even though the tokens
    # emitted are not. This is the primary test precisely because a token list
    # is vocabulary-bound and this is not: measured 2026-08-30 across the three
    # sidecars served here, Qwen3.8 emits `<|vision_start|><|image_pad|>` and
    # Muse-Glimmer emits `<|patch|>`, sharing NO tokens -- while both spell the
    # branch `part['type'] == 'image'`. A token allowlist scores the second one
    # as media-blind, which is the exact false refusal this shape avoids.
    _MEDIA_BRANCH = re.compile(
        r"""==\s*['"](image|image_url|video|audio|input_audio)['"]""", re.I)

    # Secondary: templates that emit a media token unconditionally rather than
    # behind a type test (gemma's `<start_of_image>` sits inline). Kept as an
    # OR so a family using neither spelling is the only miss.
    _MEDIA_MARKERS = (
        "vision_start", "image_pad", "video_pad", "audio_pad", "__media__",
        "start_of_image", "start_of_audio", "image_soft_token",
        "<image>", "<img", "<audio>", "<|image", "<|audio", "<|video",
        "<|patch",
    )

    @classmethod
    def _template_handles_media(cls, text: str) -> bool:
        """Whether a jinja template does anything with non-text content parts.

        Failure direction is deliberate and asymmetric: an unrecognised
        template reads as media-blind, which costs the sidecar its promotion
        and leaves the pre-1.79.43 behaviour (the embedded template) in force.
        A false refusal is a cosmetic loss; a false acceptance loads a vision
        tower and then renders prompts that can never reference an image,
        which nothing downstream flags and which surfaces only as a model
        confidently answering about a picture it never received.
        """
        return bool(cls._MEDIA_BRANCH.search(text)) or any(
            marker in text.lower() for marker in cls._MEDIA_MARKERS)

    @staticmethod
    def _is_media_served(cfg: Dict) -> bool:
        """Whether an entry is served with a projector attached.

        `mmproj_path` is the operative half -- it is what makes llama-server
        load a vision tower -- and `modalities` is checked too so an entry
        declaring vision without a projector still gets the guard.

        Takes the CONFIG rather than reading self: the ladder below has to
        answer for models that are not resident, so the admin template view
        can show what a spawn WOULD resolve without spawning one.
        """
        return bool(cfg.get("mmproj_path")) or "vision" in (cfg.get("modalities") or [])

    @staticmethod
    def _sidecar_chat_template(model_path: str,
                               filename: str = "chat_template.jinja") -> Optional[Path]:
        """``filename`` sitting beside the .gguf, or None.

        Parameterized so the publisher's ``chat_template.jinja`` and the
        operator's own override file are found by ONE routine -- the gate
        below (weights must exist, so a bare path cannot make the process CWD
        a model directory) has to hold for both, and a second copy of it is a
        second place for it to be wrong.

        The GGUF's own embedded template is whatever the quantizer baked in,
        and the same weights reach us from different publishers with different
        templates (see GGUFModelConfig's docstring). A sidecar jinja next to
        the weights is the readable, diffable answer, so it WINS over the
        embedded one -- but never over an explicit ``chat_template_path``,
        which is someone naming a file on purpose.

        Probes only the model file's own directory: a split GGUF's shards all
        live there, so this is the same folder either way. Returns None rather
        than raising for every unreadable case -- discovery finding nothing is
        the ordinary path, and must degrade to the embedded template exactly
        as it did before this existed. That is also what keeps ``_build_args``
        usable with paths that do not exist, which its drift test relies on.
        """
        if not model_path:
            return None
        try:
            weights = Path(model_path).expanduser()
            # Gate on the MODEL FILE existing, not just on the sidecar. An empty
            # or bare-filename model_path makes `.parent` the process CWD, so a
            # stray chat_template.jinja wherever the server happened to be
            # started could become a model's prompt format -- the same
            # CWD-relative trap that made file logging opt-in. Requiring the
            # weights to be present means discovery only ever reads a real model
            # directory, and it also keeps `_build_args` honest for callers
            # passing paths that do not exist (the argv/metadata drift test),
            # which get a clean None instead of whatever sits beside them.
            if not weights.is_file():
                return None
            candidate = weights.parent / filename
            return candidate if candidate.is_file() else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _sidecar_default() -> bool:
        """The `use_sidecar_chat_template` default, READ OFF THE FIELD.

        Not hand-copied as a literal here: production passes a validated
        `model_dump()` so the key is always present, but the provider also
        accepts RAW dicts (the argv/metadata drift test builds one directly,
        and the unit tests do too), and those hit this fallback. A copied
        `True` would mean flipping the field's default left every raw-dict
        caller on the old behaviour, with the suite green while the shipped
        default and the enforced default disagreed.
        """
        from ..config import GGUFModelConfig
        return bool(GGUFModelConfig.model_fields["use_sidecar_chat_template"].default)

    def added_tokens(self) -> frozenset:
        """The GGUF's own added tokens (control + user-defined), read from the
        header: llama-server owns the tokenizer, so there is no template info
        to ask."""
        from .. import gguf_metadata
        return gguf_metadata.added_tokens(Path(str(self.config.get("model_path") or "")))

    def _resolve_chat_template(self) -> tuple[Optional[str], str]:
        """This provider's own view of the ladder -- see resolve_chat_template."""
        return self.resolve_chat_template(
            self.config, getattr(self, "model_id", None))

    @classmethod
    def resolve_chat_template(cls, cfg: Dict, model_id: Optional[str] = None,
                              *, log: bool = True) -> tuple[Optional[str], str]:
        """The template file to spawn with, and a phrase for WHERE it came from.

        Takes a CONFIG, not a live provider, so the admin template view
        answers for models that are not resident and answers with the SAME
        ladder a spawn would walk. ``log=False`` for that caller: the media
        refusal below is a SPAWN warning, and emitting it from a read-only
        route made merely opening the template panel log as though a load had
        been refused. Once per REQUEST, so on every panel open, every save
        and every revert -- which makes the real one indistinguishable
        from the noise. ("Three times per save-and-confirm cycle" is what
        this said until it was counted: `_template_view` runs once per
        route, so opening and saving is two, and only a revert makes three.) A second implementation for previewing is
        the defect this repo keeps naming: it agrees on the day it is written
        and silently diverges after.

        The caller logs the phrase: a template is the single biggest
        determinant of what the model actually sees, and switching it silently
        -- which is what dropping a file into a model directory now does -- is
        the kind of change that shows up later as "the model got worse" with
        nothing to point at.

        A sidecar does NOT win when taking it would cost the model a modality
        it is being served with. Found live 2026-08-30: all three sidecar-
        carrying gguf models here are multimodal, and one of them ships a
        sidecar with no media markers at all -- so the unguarded rule would
        have loaded that model's projector and then rendered its prompts
        through a template that can never reference an image. A silently
        text-only vision model is a worse outcome than an unfashionable
        template, so the guard resolves that collision toward the embedded one
        and says so loudly.
        """
        explicit = cfg.get("chat_template_path")
        if explicit:
            return str(Path(explicit).expanduser()), "configured"

        # The operator's OWN override, discovered beside the weights the same
        # way a publisher's sidecar is -- but deliberately NOT gated on
        # `use_sidecar_chat_template`. That flag chooses between the
        # PUBLISHER's sidecar and the quantizer's embedded template, and
        # neither of those is a file the operator wrote. Letting it suppress
        # the override would mean the template editor writes a file that
        # nothing reads, with no error anywhere -- the one failure an editor
        # must not have.
        sidecar = cls._sidecar_chat_template(
            cfg.get("model_path", ""), HEYLOOK_TEMPLATE_FILENAME)
        # The CONSTANT, not a prettier phrase for the log. The origin is
        # compared against it (the admin view decides "is this override
        # actually in force" that way), and a human-readable second spelling
        # is a comparison that silently never matches -- which is exactly what
        # "heylook override" vs HEYLOOK_OVERRIDE did until a test caught it.
        origin = HEYLOOK_OVERRIDE
        if sidecar is None:
            origin = "sidecar"
            use_sidecar = cfg.get("use_sidecar_chat_template")
            if use_sidecar is None:
                use_sidecar = cls._sidecar_default()
            if not use_sidecar:
                return None, "embedded in the GGUF"

            sidecar = cls._sidecar_chat_template(cfg.get("model_path", ""))
            if sidecar is None:
                return None, "embedded in the GGUF"

        if cls._is_media_served(cfg):
            try:
                text = sidecar.read_text(errors="replace")
            except OSError:
                # Unreadable at the moment of the check: same conservative
                # direction as an unrecognised marker -- decline the promotion
                # rather than spawn against a file we could not inspect.
                text = ""
            if not cls._template_handles_media(text):
                # model_id is optional: the provider is also constructed
                # via __new__ in the argv/metadata drift test, which never runs
                # BaseProvider's __init__, and the admin view calls this with
                # a config alone. A log line must not be the thing that raises.
                if log:
                    logging.warning(
                        f"[GGUF] {model_id or '<unconstructed>'}: "
                        f"IGNORING the {origin} chat template "
                        f"{sidecar} -- this model is served with a projector "
                        f"(mmproj/vision) and that template contains no media "
                        f"markers, so using it would load the vision tower and "
                        f"then render prompts that can never reference an image. "
                        f"Using the GGUF's embedded template instead. Set "
                        f"chat_template_path explicitly to override this refusal."
                    )
                return None, f"embedded in the GGUF ({origin} skipped: no media handling)"

        return str(sidecar), origin

    # The micro-batch AUTO resolves to when the working set allows it. Why
    # 2048 and why not always: GGUFModelConfig.n_ubatch.
    AUTO_UBATCH = 2048
    # llama-server's own -ub default, what an unset-and-not-auto spawn gets.
    # Pinned against the build tree's common/common.h by
    # test_non_causal_table_matches_the_build.
    LLAMA_DEFAULT_N_UBATCH = 512
    # llama-server's prompt-cache defaults (common/common.h: cache_ram_mib,
    # n_ctx_checkpoints, checkpoint_min_step), what a spawn gets unless
    # cache_ram_mb or extra_args say otherwise. Pinned against the build tree
    # by test_non_causal_table_matches_the_build.
    LLAMA_DEFAULT_CACHE_RAM_MIB = 8192
    LLAMA_DEFAULT_CTX_CHECKPOINTS = 32
    LLAMA_DEFAULT_CHECKPOINT_MIN_STEP = 8192

    @staticmethod
    def extra_arg_value(extra_args, names) -> Optional[str]:
        """The value extra_args gives any spelling in ``names`` (``--f N`` or
        ``--f=N``); the last one wins, as in llama.cpp's own parse."""
        args = [str(a) for a in (extra_args or [])]
        found = None
        for i, a in enumerate(args):
            flag, eq, val = a.partition("=")
            if flag in names:
                found = val if eq else (args[i + 1] if i + 1 < len(args) else None)
        return found

    # "The life of the process" as a number, because ggml-metal has no
    # "forever": it counts the keep-alive down in 5 ms ticks in an atomic_int,
    # so a value above ~10.7M s wraps negative and turns residency OFF, and
    # 0 or below silently means its default. Thirty days stays clear of the
    # wrap even if upstream shortens the tick several-fold.
    METAL_RESIDENCY_KEEP_ALIVE_S = 30 * 24 * 3600

    # Projectors whose image tokens llama.cpp decodes NON-causally
    # (mtmd_decode_use_non_causal, tools/mtmd/mtmd.cpp), mapped to the most
    # tokens one image gets at llama.cpp's defaults and whether
    # --image-max-tokens lowers that. A non-causal batch must fit one
    # micro-batch (llama_context::decode asserts n_ubatch >= n_tokens), and
    # the image is split by n_batch, not n_ubatch -- so an image with more
    # tokens than the micro-batch ABORTS the process.
    #
    # A hand copy of C++ with no API (owner call 2026-09-23: a table plus a
    # test, never a build-time parser that could block a llama.cpp update).
    # test_non_causal_table_matches_the_build reads the source of the build
    # this provider spawns and goes red when it moves.
    # - gemma4v is non-causal only above the E2B/E4B text widths; it is
    #   listed whole, which caps those two only when the micro-batch is small.
    # - gemma3 is a fixed token count and ignores the flag; it is listed so
    #   the table mirrors the source's set.
    NON_CAUSAL_IMAGE_PROJECTORS: dict[str, tuple[int, bool]] = {
        "gemma4v": (1120, True),
        "gemma4uv": (1120, True),
        "deepseek4v": (384, True),
        "gemma3": (256, False),
    }

    @classmethod
    def effective_ubatch(cls, cfg: Dict, auto_ubatch: Optional[int]) -> tuple[int, Optional[str]]:
        """``(n_ubatch, clamp_note)``: the micro-batch llama.cpp will actually
        run, and a note when a clamp changed the requested one.

        llama_context (src/llama-context.cpp, at the build commit) clamps
        twice for a causal model: n_batch to the context
        (``min(n_ctx, n_batch)``), then n_ubatch to that n_batch. The context
        is known before spawn only when ctx_size is set; unset, llama-server
        sizes from the training context, far above any batch. ONE function
        over the same inputs the argv carries: the image-cap decision and
        both halves of the report call it.
        """
        requested = cfg.get("n_ubatch") or auto_ubatch or cls.LLAMA_DEFAULT_N_UBATCH
        n_batch = cfg.get("n_batch") or GGUFModelConfig.LLAMA_DEFAULT_N_BATCH
        ctx = cfg.get("ctx_size") or 0
        batch = min(ctx, n_batch) if ctx > 0 else n_batch
        effective = min(batch, requested)
        if effective == requested:
            return effective, None
        if ctx > 0 and ctx < n_batch and effective == ctx:
            return effective, f"{requested}, clamped to ctx_size {ctx}"
        return effective, f"{requested}, clamped to n_batch {batch}"

    @classmethod
    def image_token_cap_decision(
            cls, cfg: Dict, auto_ubatch: Optional[int]
    ) -> tuple[Optional[int], Optional[str], Optional[str]]:
        """``(cap, reason, level)`` for ``--image-max-tokens``.

        ``cap`` is the micro-batch when a non-causal projector's largest image
        would not fit it, else None. ``reason`` is the sentence the spawn log
        and the engine contract both carry; ``level`` is the log level
        ("info"/"warning"), None when the projector is causal or absent.

        Never RAISES a projector's limit: llama.cpp takes a custom maximum
        above its default as given, so the flag is only passed when it
        lowers. Takes a CONFIG so the contract's static half can answer for an
        unloaded model whose micro-batch is stored.
        """
        mmproj = cfg.get("mmproj_path")
        if not mmproj:
            return None, None, None
        from .. import gguf_metadata
        proj = gguf_metadata.vision_projector_type(Path(mmproj).expanduser())
        if proj not in cls.NON_CAUSAL_IMAGE_PROJECTORS:
            return None, f"the {proj} projector decodes images causally; never capped", None
        default_max, cappable = cls.NON_CAUSAL_IMAGE_PROJECTORS[proj]
        ubatch, _ = cls.effective_ubatch(cfg, auto_ubatch)
        if default_max <= ubatch:
            return None, (f"{proj} projector decodes images non-causally; its "
                          f"{default_max}-token maximum fits the {ubatch}-token "
                          f"micro-batch"), "info"
        if any(str(a).split("=", 1)[0] == "--image-max-tokens"
               for a in cfg.get("extra_args") or []):
            return None, (f"{proj} projector decodes images non-causally; "
                          f"extra_args sets --image-max-tokens, so heylook does "
                          f"not cap it at the {ubatch}-token micro-batch. Above "
                          f"that, an image aborts llama-server."), "warning"
        if not cappable:
            return None, (f"{proj} projector decodes a fixed {default_max} "
                          f"tokens per image non-causally, more than the "
                          f"{ubatch}-token micro-batch, and --image-max-tokens "
                          f"cannot lower it. Any image will abort llama-server; "
                          f"raise n_ubatch."), "warning"
        return ubatch, (f"{proj} projector decodes images non-causally; "
                        f"--image-max-tokens {ubatch} (default {default_max} "
                        f"exceeds the {ubatch}-token micro-batch)"), "info"

    def _image_token_cap(self, auto_ubatch: Optional[int]) -> Optional[int]:
        """The decision above for this provider's config, logged at spawn
        when the projector is non-causal and recorded for describe_observed."""
        cap, reason, level = self.image_token_cap_decision(self.config, auto_ubatch)
        self._image_cap_reason = reason
        if level == "warning":
            logging.warning(f"[GGUF] {self.model_id}: {reason}")
        elif level == "info":
            logging.info(f"[GGUF] {self.model_id}: {reason}")
        return cap

    def _working_set_headroom_gb(self) -> Optional[float]:
        """Metal working set minus this model's sized weights + sidecars, via
        the SAME sizing the admin fit panel shows. None when there is no
        ceiling to read (no Metal, sizing failed) -- never raises, because a
        sizing hiccup must not refuse a spawn."""
        try:
            return ram_fit.fit_for_config(
                dict(self.config), hard_working_set=False).kv_headroom_gb
        except Exception:  # noqa: BLE001 -- best-effort by design
            logging.debug(f"[GGUF] {self.model_id}: fit sizing failed",
                          exc_info=True)
            return None

    # Why this spawn runs without the drafter its config names, or None.
    drafter_skipped: Optional[str] = None

    def _drop_drafter_if_short(self) -> None:
        """Spawn without the drafter when the model fits and the model plus
        its drafter does not.

        Spec decode is on wherever a model ships a drafter (owner decision
        2026-09-24), and a drafter can cost enough memory to make a model
        that fits stop fitting (DeepSeek-V4-Flash-Vision's is ~10 GiB). A
        model that will not load is a loss; a model without spec decode only
        runs slower, so the drafter is what gives way. Decided HERE, at
        spawn, against live reclaimable RAM (ram_fit, the pre-flight's own
        measure): the binding limit is RAM left after everything else on the
        machine, which moves, and llama.cpp's own --fit sizes against the
        Metal working set, which does not see it. Both numbers are logged, so
        "short by N GiB" can be answered by freeing memory instead.
        Best-effort: a sizing failure keeps the configured spawn.
        """
        if not self.config.get("draft_model_path"):
            return
        alone_cfg = {k: v for k, v in self.config.items()
                     if k not in ("draft_model_path", "spec_type")}
        try:
            both = ram_fit.fit_for_config(dict(self.config), hard_working_set=False)
            alone = ram_fit.fit_for_config(alone_cfg, hard_working_set=False)
        except Exception:  # noqa: BLE001 -- best-effort by design
            logging.debug(f"[GGUF] {self.model_id}: drafter fit sizing failed", exc_info=True)
            return
        need = both.weights_gb + both.headroom_gb
        if need <= both.reclaimable_gb:
            return
        if alone.weights_gb + alone.headroom_gb > alone.reclaimable_gb:
            return  # the model does not fit on its own; the drafter is not the difference
        self._without_drafter(
            f"model + drafter need {both.weights_gb:.1f} GiB + {both.headroom_gb:.0f} GiB "
            f"headroom, {both.reclaimable_gb:.1f} GiB is reclaimable now (short by "
            f"{need - both.reclaimable_gb:.1f} GiB); the model alone fits. Free that much "
            f"memory and reload to run spec decode")

    def _auto_ubatch(self) -> Optional[int]:
        """AUTO_UBATCH when the headroom clears ram_fit.THIN_HEADROOM_GB,
        else None (inherit llama-server's default). Logged at spawn either
        way, because the answer moves with the sysctl and with what else
        sits in the model dir, and a spawn that quietly ran at 512 would be
        indistinguishable from one that quietly ran at 2048."""
        if self.config.get("n_ubatch") is not None:
            self._ubatch_reason = "set in heylook.toml for this model"
            return None  # explicit wins; _build_args reads it directly
        headroom = self._working_set_headroom_gb()
        if headroom is None:
            reason = "auto: llama-server default (no Metal ceiling to size against)"
            value = None
        elif headroom >= ram_fit.THIN_HEADROOM_GB:
            reason = (f"auto: {self.AUTO_UBATCH} (working-set headroom "
                      f"{headroom:.1f} GiB)")
            value = self.AUTO_UBATCH
        else:
            reason = (f"auto: llama-server default (working-set headroom "
                      f"{headroom:.1f} GiB < {ram_fit.THIN_HEADROOM_GB:.0f}; raise "
                      f"iogpu.wired_limit_mb to lift it)")
            value = None
        self._ubatch_reason = reason
        logging.info(f"[GGUF] {self.model_id}: n_ubatch {reason}")
        return value

    def _build_args(self, binary: Path, port: int,
                    chat_template: Optional[str] = None,
                    template_resolved: bool = False,
                    auto_ubatch: Optional[int] = None,
                    image_max_tokens: Optional[int] = None) -> list:
        """Build the spawn argv.

        ``auto_ubatch`` is what ``_auto_ubatch`` resolved for a config that
        leaves n_ubatch unset; a stored n_ubatch always wins over it. It is a
        parameter so this builder stays PURE (the argv/metadata drift test
        calls it with paths that do not exist and must not size anything).
        ``image_max_tokens`` is ``_image_token_cap``'s answer, passed down for
        the same reason (it reads the mmproj header).

        ``chat_template``/``template_resolved`` let ``load_model`` resolve the
        template ONCE and hand the answer down. Without that, argv and the
        spawn log each probed the filesystem ~85 lines apart, so a sidecar
        created or removed between the two calls made the log describe a
        command line that was never issued -- and the log is the only record,
        since llama-server's own output is kept nowhere at the default
        observability level (the pump reads it only for cache events). Callers that pass nothing (the argv/metadata
        drift test) still resolve inline.
        """
        cfg = self.config
        args = [
            str(binary),
            "-m", cfg["model_path"],
            "--host", cfg.get("host", "127.0.0.1"),
            "--port", str(port),
            "-np", "1",
            "-ngl", str(cfg.get("n_gpu_layers", 999)),
            "--no-webui",
        ]
        if cfg.get("ctx_size"):
            args += ["--ctx-size", str(cfg["ctx_size"])]
        # Always explicit: llama-server's own default is auto (on where the
        # device supports it), and heylook's is off (effective_flash_attn).
        args += ["-fa", effective_flash_attn(cfg)[0]]
        # Batch sizing. `is not None` throughout: an unset field inherits
        # llama-server's own default, and for n_ubatch "unset" means the
        # auto answer load_model resolved (None = inherit, again).
        if cfg.get("n_batch") is not None:
            args += ["-b", str(cfg["n_batch"])]
        # The REQUEST goes on argv; llama.cpp applies its clamps to it.
        # effective_ubatch mirrors those clamps from the same inputs for the
        # image cap and the report, so what is reported is what runs.
        ubatch = cfg["n_ubatch"] if cfg.get("n_ubatch") is not None else auto_ubatch
        if ubatch is not None:
            args += ["-ub", str(ubatch)]
        if cfg.get("mmproj_path"):
            args += ["--mmproj", cfg["mmproj_path"]]
        if image_max_tokens is not None:
            args += ["--image-max-tokens", str(image_max_tokens)]
        # Absent -> llama-server uses the template embedded in the GGUF, which
        # is whatever the quantizer baked in. See GGUFModelConfig's docstring
        # for why that is a real choice and not a formality.
        # Absent config + no sidecar -> llama-server uses the embedded template.
        # expanduser happens inside _resolve_chat_template: `~` is an accepted
        # spelling on this config surface (server_binary expands it), and
        # llama-server gets argv directly with no shell, so an unexpanded `~`
        # would reach it literally.
        template_arg = (chat_template if template_resolved
                        else self._resolve_chat_template()[0])
        if template_arg:
            args += ["--chat-template-file", template_arg]
        if cfg.get("draft_model_path"):
            args += ["-md", cfg["draft_model_path"]]
        if cfg.get("spec_type"):
            args += ["--spec-type", cfg["spec_type"]]
        # Truthiness here, `is not None` two lines down -- deliberate, not an
        # oversight. 0 is a MEANINGFUL value for p_min/-ngld/-cram, so dropping
        # it would lose a real setting. For n_max it is INVALID (ge=1), so the
        # two tests are equivalent over every value that can reach here through
        # validation. Pinned by test_n_max_bound_and_emitter_agree: if that
        # bound is ever relaxed to allow 0, this line silently starts dropping
        # a valid setting, and that test goes red first.
        if cfg.get("spec_draft_n_max"):
            args += ["--spec-draft-n-max", str(cfg["spec_draft_n_max"])]
        # `is not None`: 0.0 is a real setting (keep every draft), and it is
        # also llama.cpp's default -- so truthiness would make "explicitly 0.0"
        # indistinguishable from unset. Tune this WITH spec_draft_n_max; the
        # two interact and the interaction inverts (see the config docstring).
        if cfg.get("spec_draft_p_min") is not None:
            args += ["--spec-draft-p-min", str(cfg["spec_draft_p_min"])]
        # `is not None`: 0 = "no floor" is an explicit choice.
        if cfg.get("spec_draft_n_min") is not None:
            args += ["--spec-draft-n-min", str(cfg["spec_draft_n_min"])]
        # Expert offload. `is not None` again: 0 means "offload no layers",
        # which is a meaningful explicit choice, not an absent one.
        if cfg.get("n_cpu_moe") is not None:
            args += ["-ncmoe", str(cfg["n_cpu_moe"])]
        if cfg.get("cpu_moe"):
            args += ["-cmoe"]  # BARE flag: llama.cpp takes no value for this
        if cfg.get("override_tensor"):
            args += ["-ot", cfg["override_tensor"]]
        # Draft-side expert offload mirrors the target pair.
        if cfg.get("n_cpu_moe_draft") is not None:
            args += ["-ncmoed", str(cfg["n_cpu_moe_draft"])]
        if cfg.get("cpu_moe_draft"):
            args += ["-cmoed"]  # BARE flag, like -cmoe
        # KV cache quantization; usually the first lever for KV headroom.
        if cfg.get("cache_type_k"):
            args += ["-ctk", cfg["cache_type_k"]]
        if cfg.get("cache_type_v"):
            args += ["-ctv", cfg["cache_type_v"]]
        # `is not None`, not truthiness: 0 is meaningful for both (keep the
        # drafter off the GPU / disable the prompt cache), and -1 means
        # "unlimited" for -cram.
        if cfg.get("n_gpu_layers_draft") is not None:
            args += ["-ngld", str(cfg["n_gpu_layers_draft"])]
        if cfg.get("cache_ram_mb") is not None:
            args += ["-cram", str(cfg["cache_ram_mb"])]
        if cfg.get("sleep_idle_seconds"):
            args += ["--sleep-idle-seconds", str(cfg["sleep_idle_seconds"])]
        if cfg.get("load_mode"):
            args += ["-lm", cfg["load_mode"]]
        # Trace verbosity: libllama's own INFO lines (common/log.cpp maps them
        # to trace) are the only place llama-server states what an auto
        # setting resolved to -- flash attention's device probe among them
        # (SpawnLog). At the default (3) they are not printed and the
        # observed value stays unknown; found by the first live load after
        # W1 shipped. Measured 2026-09-25: a few hundred extra lines at load
        # and a few dozen per request, not per token. Before extra_args, so
        # an explicit -lv there wins (llama.cpp takes the last occurrence).
        args += ["-lv", "4"]
        args += list(cfg.get("extra_args") or [])
        return args

    def _record_load_report(self, auto_ubatch: Optional[int], keep_alive: str,
                            keep_alive_reason: str) -> None:
        """What this spawn decided, recorded for describe_observed(). Built
        from the same values the argv and child env were built from."""
        from heylook_llm.config import field_effect
        from .contract import Setting

        cfg = self.config
        stored = cfg.get("n_ubatch")
        auto = auto_ubatch or self.LLAMA_DEFAULT_N_UBATCH
        effective, clamp = self.effective_ubatch(cfg, auto_ubatch)
        effect = field_effect(GGUFModelConfig.model_fields["n_ubatch"])
        binary, binary_reason = self.binary_report(cfg)
        cap = self._last_image_cap
        self._load_report = {
            "n_ubatch": Setting(
                value=effective, configured=stored, auto=auto,
                reason=((getattr(self, "_ubatch_reason", None) or "decided at spawn")
                        + (f"; in force: {clamp}" if clamp else "")),
                provenance="observed", effect=effect),
            "image_max_tokens": Setting(
                value=cap, auto=cap,
                reason=getattr(self, "_image_cap_reason", None)
                or "no projector: the model takes no images",
                provenance="observed" if cfg.get("mmproj_path") else "not_applicable"),
            "metal_keep_alive": Setting(
                value=keep_alive, auto=str(self.METAL_RESIDENCY_KEEP_ALIVE_S),
                reason=keep_alive_reason, provenance="observed"),
            "binary": Setting(value=binary, auto=binary, reason=binary_reason,
                              provenance="observed"),
        }

    def get_metrics(self):
        """What the metrics endpoint and /status report for this model,
        without a call into llama-server: nothing here may wait on a
        generation. Memory is the llama-server process's resident size (one
        syscall; its weights are mapped from the file, so physical footprint
        would leave them out); context capacity is what /props said at ready; context
        used is the last request's prompt plus reply. Unknown is None."""
        from ..config import ModelMetrics
        from ..process_memory import resident_mb

        proc = getattr(self, "_proc", None)
        memory = resident_mb(proc.pid) if proc is not None and proc.poll() is None else None
        used, capacity = getattr(self, "_context_used", None), self.running_ctx
        return ModelMetrics(
            context_used=used, context_capacity=capacity,
            context_percent=round(used / capacity * 100, 1) if used is not None and capacity else None,
            memory_mb=memory, memory_source="the llama-server process's resident memory: mapped weights plus KV cache and buffers",
            requests_active=self.active_generations,
            requests_queued=self._gen_gate.snapshot()["waiting"])

    def describe_observed(self):
        """The engine contract's observed half: what this process was spawned
        with, read back from fields load_model recorded. No call to the
        process and no lock, so a listing never waits on a generation."""
        from .contract import Fact, Observed
        from .gguf_describe import flash_attn_setting

        ctx = self.running_ctx
        return Observed(
            context_running=Fact(
                value=ctx, provenance="observed" if ctx else "unknown",
                source="llama-server's /props at ready" if ctx
                else "llama-server's /props did not report a context size"),
            loaded_template=self.loaded_chat_template,
            settings={**(getattr(self, "_load_report", None) or {}),
                      "flash_attn": flash_attn_setting(
                          self.config,
                          getattr(getattr(self, "_spawn_log", None), "flash_attn", None),
                          loaded=True)},
            speculative={"in_force": self._spec_in_force()},
        )

    def _spec_in_force(self):
        """Whether this process drafts: read from the config it spawned with
        (a drafter dropped at spawn is gone from it), and why not."""
        from .contract import Fact

        if getattr(self, "_proc", None) is None:
            return Fact(provenance="unknown", source="not loaded")
        drafter, spec = self.config.get("draft_model_path"), self.config.get("spec_type")
        if drafter:
            return Fact(value=True, provenance="observed",
                        source=f"spawned with -md {Path(drafter).name}")
        if spec:
            return Fact(value=True, provenance="observed",
                        source=f"spawned with --spec-type {spec} (the built-in head)")
        return Fact(value=False, provenance="observed",
                    source=self.drafter_skipped or "spawned without a drafter")

    def load_model(self):
        """Spawn llama-server and wait until it serves.

        A drafter never costs the model (owner decision 2026-09-24: spec
        decode is on wherever a drafter ships, so discovery pairs files the
        build may not be able to run -- Qwen3.8-Flash-Next's split-out MTP
        head has no embeddings of its own and this build will not load it as
        a draft model). When llama-server exits during load with a drafter
        set, the spawn is retried once without it; if that loads, the
        drafter is remembered as unloadable by this binary for the life of
        the process, so later loads skip straight to it.
        """
        drafter = self.config.get("draft_model_path")
        key = (drafter, str(self._resolve_binary()))
        if drafter and key in _UNLOADABLE_DRAFTERS:
            self._without_drafter(f"drafter {Path(drafter).name} failed to load in this "
                                  f"llama-server build earlier in this process")
        try:
            return self._load_once()
        except LlamaServerLoadExit as first:
            if not self.config.get("draft_model_path"):
                raise
            self._without_drafter(f"llama-server exited while loading with drafter "
                                  f"{Path(drafter).name} ({first}); retried without it")
        self._load_once()
        _UNLOADABLE_DRAFTERS.add(key)
        logging.warning(
            f"[GGUF] {self.model_id}: loaded without its drafter, so the drafter is the "
            f"cause. To stop paying the failed first spawn after a restart, put "
            f'unset = ["draft_model_path", "spec_type"] in this model\'s model.heylook.toml')

    def _without_drafter(self, reason: str) -> None:
        """Run this spawn without its drafter, saying why (``drafter_skipped``)."""
        self.drafter_skipped = reason
        logging.warning(f"[GGUF] {self.model_id}: speculative decoding off for this spawn: {reason}")
        self.config = {k: v for k, v in self.config.items()
                       if k not in ("draft_model_path", "spec_type")}

    def _load_once(self):
        binary = self._resolve_binary()
        host = self.config.get("host", "127.0.0.1")
        port = int(self.config.get("port") or 0) or self._free_port()
        # ONE resolution for both argv and the log below. argv itself is
        # built AFTER the file pre-flight: the auto micro-batch sizes the
        # model's files against the Metal ceiling, and a missing file must
        # fail with the message that names the field, not inside sizing.
        resolved_template, template_origin = self._resolve_chat_template()

        # Pre-flight EVERY configured file HERE, not in _build_args (which
        # stays pure -- it is exercised by the argv/metadata drift test with
        # paths that do not exist). At observability_level=off, which is the
        # DEFAULT, the subprocess's output is kept nowhere, so llama-server
        # exiting on a file that is not there leaves NO diagnostic anywhere:
        # a heylook.toml entry left behind by a directory rename produced
        # `exited with code 1 -- output not captured` and nothing else
        # (2026-09-06). Worse, the missing file HAD already been noticed and
        # thrown away -- _sidecar_chat_template stats the weights and returns
        # None when they are absent, so the template ladder degraded to its
        # bottom rung and the spawn log announced a template decision for a
        # model file that did not exist. Stat once, here, and say which field
        # is wrong; do not let any other probe swallow the answer first.
        for field in ("model_path", "mmproj_path", "draft_model_path"):
            value = self.config.get(field)
            if value and not Path(value).expanduser().is_file():
                raise FileNotFoundError(
                    f"[GGUF] {self.model_id}: {field} points at {value}, "
                    f"which is not a readable file. llama-server would exit "
                    f"immediately with its output discarded, so the spawn is "
                    f"refused here instead. Fix the path in heylook.toml, or "
                    f"delete the entry -- a model under [scan].folders is "
                    f"served with derived defaults and needs no entry at all."
                )

        # The template override keeps its own message: the remedy differs
        # (removing the field is a legitimate fix, and a missing file must not
        # degrade to the embedded template -- that would silently serve a
        # different prompt format than configured).
        template = self.config.get("chat_template_path")
        if template and not Path(template).expanduser().is_file():
            raise FileNotFoundError(
                f"[GGUF] {self.model_id}: chat_template_path points at "
                f"{template}, which is not a readable file. Fix the path, "
                f"or remove the field to fall back to the normal resolution "
                f"(a chat_template.jinja beside the .gguf if there is one, "
                f"else the template embedded in the GGUF) -- removing it no "
                f"longer means the embedded template unconditionally."
            )

        self._drop_drafter_if_short()
        auto_ubatch = self._auto_ubatch()
        self._last_image_cap = self._image_token_cap(auto_ubatch)
        args = self._build_args(binary, port, resolved_template, True,
                                auto_ubatch=auto_ubatch,
                                image_max_tokens=self._last_image_cap)

        # Snapshot the BODY this process spawned with, so the admin template
        # view can tell an edited-on-disk file from the one llama-server is
        # running. Read here rather than lazily: the file can change under us
        # at any time, and the whole value of the field is that it does NOT.
        # Best-effort -- a template we cannot re-read must never block a spawn.
        if resolved_template:
            try:
                self.loaded_chat_template = Path(
                    resolved_template).read_text(encoding="utf-8", errors="replace")
            except OSError:
                self.loaded_chat_template = None
        else:
            from .. import gguf_metadata
            self.loaded_chat_template = gguf_metadata.chat_template(
                Path(str(self.config.get("model_path") or "")))
        # Say which template is in force, every spawn, with its hash (plan
        # W3). A sidecar is discovered from the filesystem, so the answer can
        # change without heylook.toml changing -- dropping a chat_template.jinja
        # next to the weights is enough to alter the prompt format.
        # Unannounced, that is a behaviour change with no artifact naming it;
        # this line is the artifact, and the hash tells two spawns apart by
        # what the model actually saw, not only by which rung won.
        import hashlib
        body_hash = (hashlib.sha256(self.loaded_chat_template.encode("utf-8")).hexdigest()[:12]
                     if self.loaded_chat_template else "unreadable")
        logging.info(
            f"[GGUF] {self.model_id}: chat template {template_origin}"
            + (f" ({resolved_template})" if resolved_template else "")
            + f" sha256={body_hash}"
        )

        # Subprocess output honors the file-logging master switch: at
        # observability_level=off (the default) NOTHING is written under
        # logs/ -- this was the one writer that ignored the switch, and it
        # sprinkled logs/ dirs wherever the server happened to be started.
        if observability.current_level() == "off":
            self._log_handle = None
            # Say "was off AT SPAWN", not "observability_level=off": during
            # router pre-warm the in-process cache still holds the pre-
            # configure default, so the DB setting may already be higher --
            # claiming its value here would be false (review 2026-08-13).
            log_ref = ("output not captured (file logging was off when this "
                       "model spawned; ensure observability_level>off and "
                       "reload to capture llama-server logs)")
        else:
            log_dir = Path(os.environ.get("HEYLOOK_LOGS_DIR", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.model_id)
            log_path = log_dir / f"llama_server_{safe_id}.log"
            self._log_handle = open(log_path, "ab")
            log_ref = f"see {log_path}"

        # A flag heylook passes explicitly WINS over its env var (llama.cpp
        # warns and overrides), so the spawn argv is safe -- but a flag we do
        # NOT pass is set silently from the environment, and the running
        # process then differs from what heylook.toml and the admin API say it
        # is. Those are surfaced, never scrubbed: someone may be setting one
        # deliberately, and quietly editing the child's environment would be
        # its own invisible behaviour change. The ONE exception, and why it is
        # an exception, is at _ENV_STRIPPED_AT_SPAWN.
        child_env = os.environ.copy()
        for key in _ENV_STRIPPED_AT_SPAWN:
            stripped = child_env.pop(key, None)
            if stripped is not None:
                logging.warning(
                    f"[GGUF] {key}={stripped!r} was set in the environment and "
                    f"has been REMOVED from llama-server's environment. It "
                    f"would make the subprocess write its own log file "
                    f"regardless of observability_level, and divert the output "
                    f"heylook captures into that file. heylook owns this "
                    f"subprocess's log destination."
                )

        # Metal residency keep-alive (plan W9). llama.cpp keeps the weights'
        # residency sets requested for this long after the last GPU work, then
        # lets the pages go cold; the first request after that gap pays for it,
        # badly on a model near the working-set ceiling. It is NOT heylook's
        # idle unload. heylook's unload ends the process, so keeping the
        # weights resident for the life of the process is exactly "as long as
        # heylook keeps the model loaded", and never goes stale when
        # idle_unload_seconds changes live. An inherited
        # value wins: someone may be setting it deliberately.
        keep_alive, keep_alive_reason, inherited = self.keep_alive_choice(child_env)
        if inherited:
            logging.warning(
                f"[GGUF] {METAL_KEEP_ALIVE_ENV}={keep_alive!r} set in "
                f"the environment; heylook's own value "
                f"({self.METAL_RESIDENCY_KEEP_ALIVE_S} s) is not applied.")
        else:
            child_env[METAL_KEEP_ALIVE_ENV] = keep_alive
        self._record_load_report(auto_ubatch, keep_alive, keep_alive_reason)

        llama_env = sorted(
            k for k in child_env
            if k.startswith("LLAMA_ARG_") or k in _ENV_SURFACED_UNPREFIXED
        )
        if llama_env:
            logging.warning(
                f"[GGUF] {', '.join(llama_env)} set in the environment. Flags "
                f"heylook passes explicitly override these, but any flag it "
                f"does NOT pass is being set from the environment and will not "
                f"be visible in this model's config."
            )

        # Reported, not overridable -- see _llama_system_config_paths.
        present = [p for p in _llama_system_config_paths() if p.is_file()]
        if present:
            logging.warning(
                f"[GGUF] llama.cpp system config present "
                f"({', '.join(str(p) for p in present)}). Its keys are applied "
                f"BEFORE heylook's flags and are not visible in this model's "
                f"config; a log-file/log-prompts-dir/slot-save-path key there "
                f"writes to disk whatever observability_level says."
            )

        logging.info(f"[GGUF] Spawning llama-server for '{self.model_id}': {' '.join(args)}")
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,  # drained by _pump_output, see there
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=child_env,
            start_new_session=True,  # own process group: unload kills the whole tree
        )
        self._register_proc(self._proc)
        self._cache_witness = CacheWitness()
        self._spawn_log = SpawnLog()
        threading.Thread(
            target=_pump_output,
            args=(self._proc.stdout, self._log_handle, self._cache_witness, self._spawn_log),
            name=f"llama-log-{self.model_id}", daemon=True,
        ).start()
        self._base_url = f"http://{host}:{port}"

        timeout_s = float(self.config.get("startup_timeout_s") or 300.0)
        deadline = time.time() + timeout_s
        while True:
            rc = self._proc.poll()
            if rc is not None:
                self._cleanup_handles()
                raise LlamaServerLoadExit(
                    f"llama-server exited with code {rc} while loading "
                    f"'{self.model_id}' -- {log_ref}"
                )
            try:
                with urllib.request.urlopen(self._base_url + "/health", timeout=5) as resp:
                    if resp.status == 200:
                        break
            except urllib.error.HTTPError as e:
                if e.code != 503:  # 503 = "Loading model", keep waiting
                    logging.debug(f"[GGUF] health probe HTTP {e.code}")
            except (urllib.error.URLError, OSError):
                pass  # not listening yet
            if time.time() > deadline:
                self.unload()
                raise RuntimeError(
                    f"llama-server for '{self.model_id}' not ready after "
                    f"{timeout_s:.0f}s -- {log_ref}"
                )
            time.sleep(0.5)

        self.running_ctx = self._read_running_ctx()
        logging.info(
            f"[GGUF] llama-server ready for '{self.model_id}' at {self._base_url}"
            + (f" (context {self.running_ctx})" if self.running_ctx else "")
        )

    def _read_running_ctx(self) -> Optional[int]:
        """``n_ctx`` of the live slot, from /props. Best-effort, None if unsure.

        One read at ready time, not per request: the value is fixed for the
        life of the process (context is a spawn-time argument), and /props is
        exempt from counting as a task so the read cannot wake or busy anything.
        """
        if self._base_url is None:
            return None
        try:
            with urllib.request.urlopen(self._base_url + "/props", timeout=10) as resp:
                return self._ctx_from_props(json.loads(resp.read()))
        except Exception as e:
            logging.debug(f"[GGUF] could not read /props for '{self.model_id}': {e}")
            return None

    @staticmethod
    def _ctx_from_props(props: dict) -> Optional[int]:
        """The per-slot context out of a /props payload, or None.

        llama-server reports it as ``default_generation_settings.n_ctx``
        (the slot's ``n_ctx_slot`` in its own load log). Guarded rather than
        indexed: an older or newer build that moves the key must degrade to
        "unknown", never to a load failure.
        """
        settings = props.get("default_generation_settings") if isinstance(props, dict) else None
        n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) and n_ctx > 0:
            return n_ctx
        return None

    def _cleanup_handles(self):
        # getattr, because this runs on the DESTRUCTOR path: a provider
        # collected before __init__ finished setting `_log_handle` raised
        # AttributeError inside __del__, where Python swallows it and the
        # object is then never unloaded. Same silent-never-unload the `drain`
        # contract exists to prevent, arriving by a different route -- it
        # shows up as one "Exception ignored while calling deallocator" line
        # in the suite and nothing else.
        # The log FILE only. Never close self._proc.stdout here: the pump
        # owns it, and closing a reader another thread is blocked in
        # readline() on waits for the buffer lock until llama-server writes
        # or exits -- on the destructor path, a GC thread hung on a live child.
        if getattr(self, "_log_handle", None) is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    @staticmethod
    def _register_proc(proc) -> None:
        """Track a spawned llama-server for the exit backstop (see _ACTIVE_PROCS).

        Sweeps exited processes first. The destructor path signals without
        waiting and deliberately leaves its Popen registered, so nothing else
        ever reaps it: without this, a long-lived parent that drops gguf
        providers by collection accumulates one zombie child and one registry
        entry per occurrence, unbounded. `poll()` reaps a finished child, and
        an entry whose process has exited is exactly what `_kill_orphans`
        would skip anyway.
        """
        # Snapshot under the lock, poll OUTSIDE it (poll() is a syscall and the
        # lock guards a set, not a subprocess), then apply under it again.
        with _ACTIVE_PROCS_LOCK:
            snapshot = list(_ACTIVE_PROCS)
        dead = [p for p in snapshot if p.poll() is not None]
        with _ACTIVE_PROCS_LOCK:
            for p in dead:
                _ACTIVE_PROCS.discard(p)
            _ACTIVE_PROCS.add(proc)

    def unload(self, *, drain: bool = True):
        """Stop this model's llama-server.

        `drain` is NOT ignorable here, which is what the comment this replaces
        claimed for a release. It is true that unloading gguf never waited for
        in-flight REQUESTS -- SIGTERM goes out either way -- but it does wait
        for the PROCESS to exit: up to 10s, plus 5s more after escalating to
        SIGKILL. `__del__` passes drain=False and can afford neither, because
        it runs on whatever thread the GC fired on. The destructor contract
        shipped MLX-only in v2.0.28 and left this provider blocking for up to
        15s there; gguf is not a footnote to the MLX path.

        drain=False therefore still SIGNALS -- skipping that strands a whole
        llama-server holding GPU memory for the life of the parent, which is
        far worse than an unreaped child -- and skips both waits.
        """
        proc = getattr(self, "_proc", None)
        self._proc = None
        self._base_url = None
        self.running_ctx = None
        self._context_used = None
        if proc is None or proc.poll() is not None:
            with _ACTIVE_PROCS_LOCK:
                _ACTIVE_PROCS.discard(proc)
            self._cleanup_handles()
            return
        if not drain:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                logging.error(
                    f"[GGUF] error signalling llama-server for '{self.model_id}'",
                    exc_info=True,
                )
            self._cleanup_handles()
            # LEFT REGISTERED, deliberately: this call did not wait, so it
            # cannot know the server exited, and dropping it from the registry
            # would leave nothing at all watching the pid. It is SAFE to leave
            # precisely because we did not wait -- the Popen is unreaped, so
            # `poll()` in _kill_orphans still speaks for this pid rather than
            # for whatever may have inherited it.
            #
            # What that backstop gives is ONE MORE SIGTERM at interpreter
            # exit, and no more: `_kill_orphans` does not escalate to SIGKILL
            # (only the drain=True path below does). A server that ignores
            # SIGTERM twice is not recovered by this path, and saying so is
            # the point -- the first version of this comment claimed the
            # backstop kept an escalation it has never had.
            logging.warning(
                f"[GGUF] llama-server for '{self.model_id}' was signalled from a "
                "destructor and not waited on. A provider collected with a live "
                "subprocess means a dropped reference -- the deliberate teardown "
                "path (router eviction, lifespan shutdown) waits."
            )
            return
        # Deregister FIRST: once we've decided to stop it AND to wait, the exit
        # hook must never signal this pid again -- once reaped it may belong to
        # something else.
        with _ACTIVE_PROCS_LOCK:
            _ACTIVE_PROCS.discard(proc)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)  # llama-server handles SIGTERM gracefully
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning(f"[GGUF] llama-server for '{self.model_id}' ignored SIGTERM; killing")
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        except Exception:
            logging.error(f"[GGUF] error stopping llama-server for '{self.model_id}'", exc_info=True)
        finally:
            self._cleanup_handles()
            logging.info(f"[GGUF] llama-server for '{self.model_id}' stopped")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _wire_message(msg) -> dict:
        """One ChatMessage as llama-server's chat-completions JSON.

        ``thinking`` (heylook's field) goes out as ``reasoning_content``
        (llama-server's, v1.79.62) -- the template decides what to do with
        it: Qwen3.8 renders every assistant turn's reasoning by default
        (`preserve_thinking`), gemma-4 only the last. Before the rename the
        key rode the wire unrenamed, llama-server ignored it, and every
        replayed assistant turn rendered an EMPTY thinking block -- and a
        Save & Continue from a reply stopped mid-thought re-thought from
        scratch while the route glued the new trace onto the old one.
        Continuation is llama.cpp's own: reasoning_content with EMPTY content
        resumes INSIDE the open thinking block; reasoning plus content
        closes the block and continues the content.
        """
        out = msg.model_dump(exclude_none=True)
        thinking = out.pop("thinking", None)
        if thinking and out.get("role") == "assistant":
            out["reasoning_content"] = thinking
        return out

    def _vendor_defaults(self) -> Dict:
        """The GGUF header's recommended sampling, cached per provider.

        Read lazily rather than at load so a provider built without one (the
        unit tests, the argv drift test) still resolves a cascade.
        """
        if self._vendor_sampling is None:
            from .. import gguf_metadata
            self._vendor_sampling = gguf_metadata.vendor_sampling(
                Path(str(self.config.get("model_path") or "")))
        return self._vendor_sampling

    @property
    def thinking_controls(self) -> Optional[dict]:
        """The thinking controls of the template this process spawned with
        (plan W2), or None before a spawn or when it has no template."""
        return detect_thinking(getattr(self, "loaded_chat_template", None))

    @property
    def thinking_capable(self) -> bool:
        """The served ``thinking`` capability, from the template in force:
        it reads the thinking switch. ``supports_thinking`` (read from the
        EMBEDDED template at discovery) answers only when no template is
        known -- the same rule capabilities.gguf_thinking_capable reports."""
        controls = self.thinking_controls
        if controls is not None:
            return controls.get("switch") is not None
        return bool(self.config.get("supports_thinking"))

    def _build_payload(self, request: ChatRequest) -> dict:
        # The shared cascade (samplers.resolve_effective_sampling) -- ONE
        # implementation with MLX, not a mirror. The vendor layer comes from
        # the GGUF HEADER (v2.0.23): a gguf dir ships no generation_config.json,
        # which used to be read as "no vendor layer here" and was the wrong
        # conclusion -- the converter writes those very values into
        # `general.sampling.*`. Keys llama-server doesn't take are dropped
        # below by _PAYLOAD_KEY_MAP.
        merged = resolve_effective_sampling(
            request, self.config, vendor=self._vendor_defaults(),
            thinking_capable=self.thinking_capable,
            thinking=self.thinking_controls)

        payload = {
            "model": self.model_id,
            "messages": [self._wire_message(m) for m in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            # Prefill progress frames (`prompt_progress` on the stream), the
            # gguf half of the cross-engine progress signal -- see
            # _stream_chunks and AbortEvent.set_prefill_progress.
            "return_progress": True,
            # llama-server's n_predict default is -1 = UNLIMITED; never omit.
            "max_tokens": int(merged.get("max_tokens") or GLOBAL_SAMPLER_FLOOR["max_tokens"]),
        }
        for src, dst in _PAYLOAD_KEY_MAP:
            value = merged.get(src)
            if value is not None:
                payload[dst] = value
        template_kwargs: dict = {}
        enable_thinking = merged.get("enable_thinking")
        if enable_thinking is not None:
            # must be a JSON bool -- llama-server rejects string values
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        # Sent whenever set, NOT gated on enable_thinking. The gate this
        # replaces assumed every template reads reasoning_effort inside a
        # thinking branch; gpt-oss/harmony refutes that -- its template reads
        # reasoning_effort UNCONDITIONALLY (defaulting to "medium") and has no
        # enable_thinking at all, so gating made the knob unreachable for the
        # exact family the docs name as taking low|medium|high. A template
        # that does not read the variable simply ignores it (jinja forwards
        # unknown kwargs as template variables), so the gate cost reachability
        # and bought nothing.
        reasoning_effort = merged.get("reasoning_effort")
        if reasoning_effort:
            # Named by the in-force template's own depth variable (plan W2).
            template_kwargs[depth_variable(self.thinking_controls)] = str(reasoning_effort)
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        if request.response_schema is not None:
            # llama-server's own spelling (an empty schema = any object). Its
            # chat parser admits the reasoning block first and constrains the
            # reply after it, and returns only the JSON as content.
            payload["json_schema"] = request.response_schema
        return payload

    def _is_sleeping(self) -> bool:
        """Whether llama-server has idled its model out (``--sleep-idle-seconds``).

        GET /props is explicitly exempt from counting as a task, so asking does
        not itself wake the server or reset its idle timer. Best-effort: an
        unreachable/older server just reports False and we use the normal
        timeout.
        """
        if self._base_url is None:
            return False
        try:
            with urllib.request.urlopen(self._base_url + "/props", timeout=10) as resp:
                return bool(json.loads(resp.read()).get("is_sleeping"))
        except Exception:
            return False

    def _request_timeout(self) -> float:
        """Socket timeout for a generation request.

        Normally ``_SSE_READ_TIMEOUT_S`` -- a healthy stream never blocks a read
        longer than llama-server's 30s keepalive, so 120s means "wedged". But a
        SLEEPING server reloads the model before it emits anything, and for a
        large model that reload is minutes, not seconds. Waiting on the sleep
        path with the wedge-detection timeout would turn a working
        configuration into a timeout on the first request after an idle gap.
        """
        if self.config.get("sleep_idle_seconds") and self._is_sleeping():
            wake_timeout = float(self.config.get("startup_timeout_s") or 300.0)
            logging.info(
                f"[GGUF] '{self.model_id}' is sleeping; allowing {wake_timeout:.0f}s "
                f"for llama-server to reload it"
            )
            return max(_SSE_READ_TIMEOUT_S, wake_timeout)
        return _SSE_READ_TIMEOUT_S

    def _continuation_echo_chars(self, request: ChatRequest, payload: dict) -> tuple[int, int]:
        """``(content_chars, thinking_chars)`` of prefill llama-server will
        ECHO back, to strip from the stream's two channels.

        May NORMALIZE ``payload`` in place: an all-text parts-list prefill is
        flattened to the exact string being measured, so the strip stays
        positional-and-exact.

        llama-server natively continues a trailing assistant message (the
        rendered turn stays open -- verified on the pinned build via
        /apply-template), but its response RE-EMITS the prefilled content as
        the leading delta(s). heylook's contract on every provider is
        "response = continuation only", so the echo is stripped positionally
        (not by string match: retokenization can attach whitespace to the
        echoed span, so byte-equality would false-negative).

        The REASONING channel echoes too (measured live on b10814, 2026-09-04):
        a prefilled ``reasoning_content`` comes back as the leading
        reasoning_content delta(s), byte-exact except that llama-server drops
        its LEADING whitespace. So that strip is sized to the lstripped
        string: the minimal count, which can only ever under-strip (a stray
        leading space survives into the new trace) and never eat real tokens.

        Also enforces what llama-server cannot express:
        - user-role continuation (explicit ``continue_final_message: true``
          with a non-assistant final message) has no llama-server spelling;
        - ``continue_final_message: false`` with a trailing assistant message
          cannot be honored -- llama-server ALWAYS continues one, and
          pretending otherwise would return a continuation labelled as a
          fresh turn.
        """
        last_role = request.messages[-1].role if request.messages else None
        if request.continue_final_message is True and last_role != "assistant":
            raise InvalidGenerationRequest(
                "user-role continuation is not supported on gguf models: "
                "llama-server prefills assistant turns only. Use an MLX model "
                "for continuing a non-assistant message."
            )
        if request.continue_final_message is False and last_role == "assistant":
            raise InvalidGenerationRequest(
                "continue_final_message=false cannot be honored on gguf models: "
                "llama-server always continues a trailing assistant message. "
                "Omit the flag or drop the trailing assistant turn."
            )
        if not request.is_continuation():
            return 0, 0
        last = request.messages[-1]
        thinking_chars = len((last.thinking or "").lstrip())
        content = last.content
        if isinstance(content, str):
            return len(content), thinking_chars
        # Parts-list content (standard for many SDKs, and what the Messages
        # API converter produces for block-form prefill -- refusing it broke
        # requests that streamed fine pre-v1.61). The positional strip needs
        # the EXACT string llama-server renders as prefill, so for all-text
        # parts we flatten the PAYLOAD's copy ourselves (same ' '-join rule
        # as the MLX _prepare_messages flatten) and measure that. Non-text
        # parts in a trailing assistant message have no knowable rendered
        # length: continuation still happens (llama-server always continues
        # a trailing assistant turn) but nothing is stripped -- the pre-strip
        # v1.60 behavior, logged so it is at least visible.
        parts = content or []
        if all(getattr(p, "type", None) == "text" for p in parts):
            flattened = " ".join(getattr(p, "text", None) or "" for p in parts)
            payload["messages"][-1]["content"] = flattened
            return len(flattened), thinking_chars
        # Media in the OPEN turn works on some templates and cannot work on
        # others, so ASK rather than assume either way. Measured on b10830,
        # 2026-09-07: whether the request succeeds tracks exactly one thing --
        # does the rendered template still contain a media marker for each
        # media part sent? gemma-4-E4B's template DROPS the marker when the
        # final assistant turn is the one being continued (0 markers for 1
        # image) and the request is a flat 400 "Failed to tokenize prompt";
        # the same model with the same image on a NON-final assistant turn
        # renders 1 marker and answers 200, as does a user turn. A template
        # that keeps the marker is fine -- DeepSeek-V4-Flash-Vision continues
        # such a turn in production here.
        #
        # So the check is the invariant, not a model list: a media part with
        # no marker to bind to cannot be tokenized. Costs one /apply-template
        # on a path that is already rare (continuation + media in the open
        # turn), and turns an opaque tokenizer error into a sentence naming
        # the cause and the way out.
        if self._media_markers_dropped(payload):
            raise InvalidGenerationRequest(
                f"'{self.model_id}' cannot continue an assistant message that "
                f"carries media: this model's chat template drops the media "
                f"marker from the turn being continued, so llama-server has an "
                f"image with nowhere to put it and fails to tokenize the "
                f"prompt. Remove the attachment from this message to continue "
                f"it, or generate a fresh reply instead. (Other models keep the "
                f"marker and continue such a turn normally.)"
            )
        logging.warning(
            f"[GGUF] '{self.model_id}': continuing a trailing assistant message "
            f"with non-text parts -- the template keeps the media marker, so the "
            f"request is valid, but the prefill echo cannot be measured and is "
            f"NOT stripped from the response"
        )
        return 0, thinking_chars

    # A media part llama-server holds must have a marker in the rendered
    # prompt to bind to. Counting them is how the open-turn case above stays
    # a PROPERTY rather than a list of model names that would rot.
    _MEDIA_MARKER_RE = re.compile(r"<__media_[^>]*__>")

    def _media_markers_dropped(self, payload: dict) -> bool:
        """True when the template renders FEWER media markers than the
        request carries media parts. Best-effort: any failure to ask answers
        False, because refusing a request on a template call that did not
        happen would be worse than the tokenizer error it is trying to
        replace."""
        sent = sum(
            1
            for m in payload.get("messages", [])
            if isinstance(m.get("content"), list)
            for part in m["content"]
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_audio")
        )
        if not sent or self._base_url is None:
            return False
        body = {k: payload[k] for k in ("messages", "chat_template_kwargs") if k in payload}
        try:
            req = urllib.request.Request(
                self._base_url + "/apply-template",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                rendered = json.load(resp).get("prompt") or ""
        except Exception:
            logging.debug(f"[GGUF] '{self.model_id}': media-marker precheck skipped",
                          exc_info=True)
            return False
        return len(self._MEDIA_MARKER_RE.findall(rendered)) < sent

    # /apply-template renders media markers in place, so the preview string
    # shows where each image sits in the conversation.
    render_prompt_represents_media = True

    def render_prompt(self, request: ChatRequest) -> str:
        """The exact prompt llama-server renders for ``request`` (its own
        ``/apply-template``, same body as a generation, so the same template
        rung, the same chat_template_kwargs and the same continuation
        resolution). No slot is taken: /apply-template is a template call,
        not a task, so it neither queues behind nor blocks a generation."""
        if self._base_url is None:
            raise GenerationFailed(f"Model '{self.model_id}' is not loaded")
        payload = self._build_payload(request)
        self._continuation_echo_chars(request, payload)  # validates + normalizes
        body = {k: payload[k] for k in ("messages", "chat_template_kwargs") if k in payload}
        http_request = urllib.request.Request(
            self._base_url + "/apply-template",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=30) as resp:
                rendered = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = self._error_detail(e)
            if e.code == 400:
                raise InvalidGenerationRequest(detail)
            raise GenerationFailed(detail)
        except (urllib.error.URLError, OSError) as e:
            raise GenerationFailed(f"llama-server for '{self.model_id}' unreachable: {e}")
        prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
        if not isinstance(prompt, str):
            raise GenerationFailed("llama-server /apply-template returned no prompt")
        return prompt

    def create_chat_completion(self, request: ChatRequest, abort_event=None) -> Generator:
        if self._base_url is None:
            raise GenerationFailed(f"Model '{self.model_id}' is not loaded")
        payload = self._build_payload(request)
        echo_chars, echo_thinking_chars = self._continuation_echo_chars(request, payload)

        http_request = urllib.request.Request(
            self._base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Take the process gate BEFORE forwarding, and hold it for the whole
        # stream. llama-server has one slot (-np 1); a request forwarded while
        # that slot is busy waits in llama-server's own queue, which emits no
        # keepalive until the stream starts, so heylook's 120s read timeout
        # fires on a backend that is merely busy. Gating here means the wait
        # happens in arrival order on this side, where check_capacity() can
        # answer 503 and a cancelled waiter is released, and the read timeout
        # goes back to meaning what its comment says: wedged.
        #
        # The cancel check is what makes "a cancelled waiter is released"
        # true. Without it (v1.79.44 through v2.0.41) a request abandoned via
        # DELETE /v1/requests/{id} or a hung-up stream kept its place in the
        # queue, took its turn, and FORWARDED to llama-server -- paying the
        # prefill for a client that was gone -- before _stream_chunks saw the
        # flag on the first frame. Same shape as the MLX chat path.
        wait_started = time.perf_counter()
        try:
            self._gen_gate.acquire(
                cancel_check=abort_event.is_set if abort_event is not None else None)
        except GenerationCancelled:
            return
        queue_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        # Active from the moment this request holds the gate, so the router
        # sees the model as busy and refuses to SIGTERM llama-server out from
        # under it. That window includes the urlopen below: llama-server sends
        # its response headers only once it has a first result, so the whole
        # prefill (and any image encode) happens inside urlopen. Until
        # 2026-09-24 the count started after it returned, so loading another
        # model during a long gguf prefill could evict this one mid-request.
        # (Before that, nothing counted at all: the base returned None for
        # generation_queue_stats, which made every gguf model look idle.)
        with self.generation_active():
            try:
                try:
                    response = urllib.request.urlopen(http_request, timeout=self._request_timeout())
                except urllib.error.HTTPError as e:
                    detail = self._error_detail(e)
                    if e.code == 400:
                        raise InvalidGenerationRequest(detail)
                    raise GenerationFailed(detail)
                except (urllib.error.URLError, OSError) as e:
                    raise GenerationFailed(
                        f"llama-server for '{self.model_id}' unreachable: {e}"
                    )
            except BaseException:
                self._gen_gate.release()
                raise
            inner = self._explained(self._stream_chunks(
                response, abort_event,
                echo_chars=echo_chars, echo_thinking_chars=echo_thinking_chars),
                payload)
            try:
                # The gate wait rides the first chunk, as on MLX, so
                # build_performance nets it out of the generation span.
                tagged = False
                for chunk in inner:
                    if not tagged:
                        chunk.queue_wait_ms = queue_wait_ms
                        tagged = True
                    if chunk.prompt_tokens:
                        self._context_used = chunk.prompt_tokens + (chunk.generation_tokens or 0)
                    yield chunk
            finally:
                inner.close()
                # Closing the connection frees the llama-server slot on abort.
                try:
                    response.close()
                except Exception:
                    pass
                self._gen_gate.release()

    def _explained(self, chunks, payload: dict) -> Generator:
        """Attach the cache witness's why to the request's CacheReport, and
        remember this request once llama-server has reported on it."""
        witness = self._cache_witness
        if witness is None:
            yield from chunks
            return
        messages, head = fingerprint(payload)
        whole = None
        for chunk in chunks:
            if chunk.cache is not None:
                chunk.cache = witness.explain(messages, head, chunk.cache)
                whole = chunk.cache.prompt_tokens
            yield chunk
        if whole is not None:
            witness.record(messages, head, whole)

    @staticmethod
    def _error_detail(e: urllib.error.HTTPError) -> str:
        try:
            body = json.loads(e.read().decode(errors="replace"))
            return body.get("error", {}).get("message") or str(body)
        except Exception:
            return f"llama-server returned HTTP {e.code}"

    def _stream_chunks(self, fp, abort_event, echo_chars: int = 0,
                       echo_thinking_chars: int = 0) -> Generator:
        """Adapt llama-server SSE lines to GenerationChunk.

        Split out from create_chat_completion so it is unit-testable with a
        canned byte stream -- no HTTP, no subprocess.

        ``echo_chars`` / ``echo_thinking_chars``: leading chars to drop from
        the content / reasoning channel -- llama-server echoes the prefill
        of a continued assistant message back as the first delta(s) of each
        channel it was given (see _continuation_echo_chars). The two counts
        are independent: a channel that was not prefilled is never touched.

        When BOTH channels were prefilled, the thought is closed in the
        prompt, so whitespace-only reasoning right after the reasoning echo is
        template framing and is dropped. With no content prefill the thought
        is still open and resumes, and a resumed thought may begin with a
        newline, so that case keeps it.
        """
        closed_thought = echo_thinking_chars > 0 and echo_chars > 0
        for raw_line in fp:
            if abort_event is not None and abort_event.is_set():
                logging.info(f"[GGUF] generation aborted for '{self.model_id}'")
                return
            line = raw_line.strip()
            if not line or line.startswith(b":"):  # keepalive comment
                continue
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):]
            if data == b"[DONE]":
                return
            try:
                frame = json.loads(data)
            except (ValueError, UnicodeDecodeError) as e:
                raise GenerationFailed(
                    f"Malformed SSE frame from llama-server: {e}"
                )
            # A mid-stream failure arrives as `data: {"error": {...}}`, not
            # as an HTTP status -- the headers were 200 before decode ran.
            # Without this the frame has no `choices`, _frame_to_chunk
            # returns None, the stream ends, and the client gets a clean
            # end_turn with zero tokens (DeepSeek V4 Flash Vision, Metal OOM,
            # 2026-09-07: "Compute error." became an empty reply).
            if isinstance(frame, dict) and "error" in frame:
                raise GenerationFailed(self._describe_engine_error(frame["error"]))
            self._report_prefill_progress(frame.get("prompt_progress"), abort_event)
            chunk = self._frame_to_chunk(frame)
            if chunk is None:
                continue
            if echo_chars > 0 and chunk.text:
                cut = min(echo_chars, len(chunk.text))
                echo_chars -= cut
                chunk.text = chunk.text[cut:]
            if echo_thinking_chars > 0 and chunk.thinking:
                cut = min(echo_thinking_chars, len(chunk.thinking))
                echo_thinking_chars -= cut
                chunk.thinking = chunk.thinking[cut:] or None
            if closed_thought and echo_thinking_chars == 0 and chunk.thinking:
                # The rest of a closed thought's echo is the template's framing
                # before </think>, which llama-server's split keeps (Qwen3.8:
                # "<thought>\n"); it is not new thinking.
                chunk.thinking = chunk.thinking.lstrip() or None
                if chunk.thinking:
                    closed_thought = False
            if not chunk.text and not chunk.thinking and not chunk.finish_reason \
                    and not chunk.prompt_tokens and not chunk.generation_tokens:
                continue  # the delta was pure echo -- nothing to emit
            yield chunk

    def _describe_engine_error(self, err) -> str:
        """llama-server's error, plus what it most likely means HERE.

        Its "Compute error." is ggml's `llama_decode` returning -3, and on
        Metal the cause underneath is almost always the GPU working set
        running out (`kIOGPUCommandBufferCallbackErrorOutOfMemory` in the
        subprocess log -- which at the default observability level is kept
        nowhere, so this message is the only place the reader will ever see
        it). Sizing the model against the live ceiling says whether that
        reading is plausible and what to do; the fit panel shows the same.
        """
        msg = err.get("message") if isinstance(err, dict) else None
        msg = str(msg or err)
        if "compute error" not in msg.lower():
            return f"llama-server error for '{self.model_id}': {msg}"
        text = (f"llama-server compute error for '{self.model_id}' -- on Metal "
                f"this is almost always the GPU working set running out "
                f"(Insufficient Memory) at decode time")
        try:
            report = ram_fit.fit_for_config(dict(self.config), hard_working_set=False)
        except Exception:  # noqa: BLE001
            report = None
        if report is not None and report.kv_headroom_gb is not None:
            text += (f"; this model has {report.kv_headroom_gb:.1f} GiB of "
                     f"working-set headroom for KV + compute at "
                     f"{report.working_set_gb:.0f} GiB")
            if report.sysctl_suggest_mb:
                text += (f". Raise the ceiling: sudo sysctl "
                         f"iogpu.wired_limit_mb={report.sysctl_suggest_mb} "
                         f"(scripts/gpu_wired_limit.sh persists it)")
            text += ", or lower ctx_size / n_ubatch on this model"
        return text + "."

    @staticmethod
    def _report_prefill_progress(progress, abort_event) -> None:
        """Route a `prompt_progress` frame (``return_progress``) up the
        request's signal channel, in the cross-engine meaning: work THIS
        request runs, cached prefix excluded. llama-server's `processed`
        counts from zero INCLUDING the cache hit, so both numbers have
        `cache` subtracted -- its README's "actual timed progress". A fully
        cached prompt has no work to report and reports nothing."""
        report = getattr(abort_event, "set_prefill_progress", None)
        if report is None or not isinstance(progress, dict):
            return
        cache = int(progress.get("cache") or 0)
        total = int(progress.get("total") or 0) - cache
        if total <= 0:
            return
        report(max(0, int(progress.get("processed") or 0) - cache), total)

    @staticmethod
    def _frame_to_chunk(frame: dict) -> Optional[GenerationChunk]:
        choices = frame.get("choices") or []
        usage = frame.get("usage")
        timings = frame.get("timings")

        text = ""
        thinking = None
        finish_reason = None
        if choices:
            first = choices[0]
            delta = first.get("delta") or {}
            text = delta.get("content") or ""
            thinking = delta.get("reasoning_content") or None
            finish_reason = first.get("finish_reason")

        if not text and not thinking and not finish_reason and not usage and not timings:
            return None  # role-prelude frame

        chunk = GenerationChunk(text=text, thinking=thinking, finish_reason=finish_reason)
        cached = None
        if usage:
            # llama-server's prompt_tokens is the WHOLE prompt
            # (slot.task->n_tokens()), its cached part reported beside it --
            # already the CacheReport shape.
            chunk.prompt_tokens = usage.get("prompt_tokens") or 0
            chunk.generation_tokens = usage.get("completion_tokens") or 0
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
        if timings:
            chunk.prompt_tps = timings.get("prompt_per_second") or 0.0
            chunk.generation_tps = timings.get("predicted_per_second") or 0.0
            if cached is None:
                cached = timings.get("cache_n")
            if not chunk.prompt_tokens and "prompt_n" in timings:
                # timings alone: processed (prompt_n) plus cached is the whole
                chunk.prompt_tokens = (timings.get("prompt_n") or 0) + (cached or 0)
            if not chunk.generation_tokens:
                chunk.generation_tokens = timings.get("predicted_n") or 0
            # present only when speculative decoding was active this request
            if timings.get("draft_n"):
                chunk.spec = SpecReport(
                    accepted=timings.get("draft_n_accepted") or 0,
                    drafted=timings.get("draft_n"),
                    emitted=timings.get("predicted_n") or chunk.generation_tokens or None)
        if cached is not None and chunk.prompt_tokens:
            # Why a request missed is not reported by llama-server (only the
            # count), so reason stays unset here; plan W5's fingerprint adds
            # probable causes separately.
            chunk.cache = CacheReport(
                prompt_tokens=chunk.prompt_tokens, cached_tokens=cached,
                outcome="reused" if cached > 0 else "miss")
        return chunk
