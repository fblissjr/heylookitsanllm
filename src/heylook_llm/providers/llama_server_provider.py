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
#   (floor -> thinking anti-loop overlay -> model fields -> default_sampler
#   -> request.sampler -> explicit request fields). No vendor layer passed:
#   GGUF dirs carry no generation_config.json.
#   max_tokens is ALWAYS sent (llama-server's default is unlimited).
# - -np 1 by OUR choice (full context per slot, matches heylook's
#   serialized semantics) -- not a compat requirement.
# - No shared MLX FIFO gate: llama-server queues its own requests.
#   check_capacity() stays the base no-op.
# - Pure stdlib (urllib/subprocess/socket): the provider must import and
#   run on machines with no MLX and no extra deps.

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Generator, Optional

from ..config import ChatRequest
from ..samplers import GLOBAL_SAMPLER_FLOOR, SamplerNotFound, resolve_effective_sampling
from .base import BaseProvider, GenerationChunk, GenerationFailed, InvalidGenerationRequest

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


def _kill_orphans() -> None:
    """Reap any llama-server still registered at interpreter exit.

    Best-effort and silent: this runs during shutdown, where logging handlers
    may already be torn down and raising would be pointless.
    """
    while _ACTIVE_PROCS:
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

# cascade key -> llama-server request key
_PAYLOAD_KEY_MAP = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("min_p", "min_p"),
    ("repetition_penalty", "repeat_penalty"),
    ("presence_penalty", "presence_penalty"),
    ("seed", "seed"),
)


class LlamaServerProvider(BaseProvider):
    """Serve a GGUF model through a managed llama-server subprocess."""

    provider_name = "gguf"

    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None  # no in-process model object; MLX-only surfaces gate on this
        self.processor = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None
        self._base_url: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # Canonical local build location, checked when nothing overrides it. A
    # llama.cpp checkout built here (fixed dir under home, outside the repo)
    # is picked up with zero config; the literal path is pinned by a test.
    DEFAULT_BUILD = Path.home() / ".heylook" / "llama.cpp" / "build" / "bin" / "llama-server"

    def _resolve_binary(self) -> Path:
        """server_binary > $HEYLOOK_LLAMA_SERVER > the binary we built ourselves.

        The last fallback exists because the other two are both fragile in
        ways that bite silently. A server_binary in models.toml survives a
        reimport (the CLI merges by default now) but is per-model and gone
        under `--fresh`; and an exported env var goes stale the moment a
        newer binary is built, pointing a working server at an old build
        with nothing to say so.

        Falling back to the canonical build location is safe precisely because
        nothing else writes there -- if the file is not there, nothing has
        been built and the error below is still the right answer. This does
        soften the older "one of the two is REQUIRED" rule, deliberately: that
        rule predates there being a canonical location.
        """
        candidate = self.config.get("server_binary") or os.environ.get("HEYLOOK_LLAMA_SERVER")
        if not candidate and self.DEFAULT_BUILD.is_file():
            logging.info(
                f"[GGUF] no server_binary or $HEYLOOK_LLAMA_SERVER set; using "
                f"the local build at {self.DEFAULT_BUILD}"
            )
            return self.DEFAULT_BUILD
        if not candidate:
            raise RuntimeError(
                "No llama-server binary configured. Build one with "
                "`uv run scripts/build_llama.py`, or set server_binary in "
                "models.toml / the $HEYLOOK_LLAMA_SERVER env var (a Homebrew "
                "or upstream release binary works too)."
            )
        path = Path(candidate).expanduser()
        if not path.is_file():
            raise RuntimeError(
                f"llama-server binary not found at '{path}'. Build it "
                f"(cmake --build ... --target llama-server) or fix "
                f"server_binary / $HEYLOOK_LLAMA_SERVER."
            )
        return path

    @staticmethod
    def _free_port() -> int:
        # Tiny bind race window, single-user localhost: acceptable, and it
        # avoids a stdout-parsing reader thread for llama's --port 0 mode.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _build_args(self, binary: Path, port: int) -> list:
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
        if cfg.get("mmproj_path"):
            args += ["--mmproj", cfg["mmproj_path"]]
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
        args += list(cfg.get("extra_args") or [])
        return args

    def load_model(self):
        binary = self._resolve_binary()
        host = self.config.get("host", "127.0.0.1")
        port = int(self.config.get("port") or 0) or self._free_port()
        args = self._build_args(binary, port)

        log_dir = Path(os.environ.get("HEYLOOK_LOGS_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.model_id)
        log_path = log_dir / f"llama_server_{safe_id}.log"
        self._log_handle = open(log_path, "ab")

        # llama-server reads LLAMA_ARG_* from the environment for most flags.
        # A CLI arg WINS over its env var (llama.cpp warns and overrides), so
        # anything heylook passes is safe -- but a flag we DON'T pass is set
        # silently, and then the running process differs from what models.toml
        # and the admin API say it is. Surface it rather than stripping the
        # env: someone may be using it deliberately, and quietly changing the
        # child's environment would be its own invisible behaviour.
        llama_env = sorted(k for k in os.environ if k.startswith("LLAMA_ARG_"))
        if llama_env:
            logging.warning(
                f"[GGUF] {', '.join(llama_env)} set in the environment. Flags "
                f"heylook passes explicitly override these, but any flag it "
                f"does NOT pass is being set from the environment and will not "
                f"be visible in this model's config."
            )

        logging.info(f"[GGUF] Spawning llama-server for '{self.model_id}': {' '.join(args)}")
        self._proc = subprocess.Popen(
            args,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group: unload kills the whole tree
        )
        self._register_proc(self._proc)
        self._base_url = f"http://{host}:{port}"

        timeout_s = float(self.config.get("startup_timeout_s") or 300.0)
        deadline = time.time() + timeout_s
        while True:
            rc = self._proc.poll()
            if rc is not None:
                self._cleanup_handles()
                raise RuntimeError(
                    f"llama-server exited with code {rc} while loading "
                    f"'{self.model_id}' -- see {log_path}"
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
                    f"{timeout_s:.0f}s -- see {log_path}"
                )
            time.sleep(0.5)

        logging.info(f"[GGUF] llama-server ready for '{self.model_id}' at {self._base_url}")

    def _cleanup_handles(self):
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    @staticmethod
    def _register_proc(proc) -> None:
        """Track a spawned llama-server for the exit backstop (see _ACTIVE_PROCS)."""
        _ACTIVE_PROCS.add(proc)

    def unload(self):
        proc = getattr(self, "_proc", None)
        self._proc = None
        self._base_url = None
        # Deregister FIRST: once we've decided to stop it, the exit hook must
        # never signal this pid again -- by then it may belong to something else.
        _ACTIVE_PROCS.discard(proc)
        if proc is None or proc.poll() is not None:
            self._cleanup_handles()
            return
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

    def _build_payload(self, request: ChatRequest) -> dict:
        # The shared cascade (samplers.resolve_effective_sampling) -- ONE
        # implementation with MLX, not a mirror. No vendor layer: GGUF dirs
        # ship no generation_config.json. Keys llama-server doesn't take
        # (vision_tokens etc.) are dropped below by _PAYLOAD_KEY_MAP.
        merged = resolve_effective_sampling(request, self.config)

        payload = {
            "model": self.model_id,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            # llama-server's n_predict default is -1 = UNLIMITED; never omit.
            "max_tokens": int(merged.get("max_tokens") or GLOBAL_SAMPLER_FLOOR["max_tokens"]),
        }
        for src, dst in _PAYLOAD_KEY_MAP:
            value = merged.get(src)
            if value is not None:
                payload[dst] = value
        enable_thinking = merged.get("enable_thinking")
        if enable_thinking is not None:
            # must be a JSON bool -- llama-server rejects string values
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
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

    def _continuation_echo_chars(self, request: ChatRequest, payload: dict) -> int:
        """Chars of prefill llama-server will ECHO back, to strip from the stream.

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
            return 0
        content = request.messages[-1].content
        if isinstance(content, str):
            return len(content)
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
            return len(flattened)
        logging.warning(
            f"[GGUF] '{self.model_id}': continuing a trailing assistant message "
            f"with non-text parts -- prefill echo cannot be measured and is NOT "
            f"stripped from the response"
        )
        return 0

    def create_chat_completion(self, request: ChatRequest, abort_event=None) -> Generator:
        if self._base_url is None:
            raise GenerationFailed(f"Model '{self.model_id}' is not loaded")
        try:
            payload = self._build_payload(request)
        except SamplerNotFound as e:
            raise InvalidGenerationRequest(str(e))
        echo_chars = self._continuation_echo_chars(request, payload)

        http_request = urllib.request.Request(
            self._base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
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
        try:
            yield from self._stream_chunks(response, abort_event, echo_chars=echo_chars)
        finally:
            # Closing the connection frees the llama-server slot on abort.
            try:
                response.close()
            except Exception:
                pass

    @staticmethod
    def _error_detail(e: urllib.error.HTTPError) -> str:
        try:
            body = json.loads(e.read().decode(errors="replace"))
            return body.get("error", {}).get("message") or str(body)
        except Exception:
            return f"llama-server returned HTTP {e.code}"

    def _stream_chunks(self, fp, abort_event, echo_chars: int = 0) -> Generator:
        """Adapt llama-server SSE lines to GenerationChunk.

        Split out from create_chat_completion so it is unit-testable with a
        canned byte stream -- no HTTP, no subprocess.

        ``echo_chars``: leading CONTENT chars to drop -- llama-server echoes
        the prefill of a continued assistant message back as the first
        delta(s) (see _continuation_echo_chars). Thinking deltas are never
        stripped: the echo is content-channel only.
        """
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
            chunk = self._frame_to_chunk(frame)
            if chunk is None:
                continue
            if echo_chars > 0 and chunk.text:
                cut = min(echo_chars, len(chunk.text))
                echo_chars -= cut
                chunk.text = chunk.text[cut:]
                if not chunk.text and not chunk.thinking and not chunk.finish_reason \
                        and not chunk.prompt_tokens and not chunk.generation_tokens:
                    continue  # the delta was pure echo -- nothing to emit
            yield chunk

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
        if usage:
            chunk.prompt_tokens = usage.get("prompt_tokens") or 0
            chunk.generation_tokens = usage.get("completion_tokens") or 0
            details = usage.get("prompt_tokens_details") or {}
            chunk.cached_tokens = details.get("cached_tokens") or 0
        if timings:
            chunk.prompt_tps = timings.get("prompt_per_second") or 0.0
            chunk.generation_tps = timings.get("predicted_per_second") or 0.0
            if not chunk.cached_tokens:
                chunk.cached_tokens = timings.get("cache_n") or 0
            if not chunk.generation_tokens:
                chunk.generation_tokens = timings.get("predicted_n") or 0
            # present only when speculative decoding was active this request
            chunk.draft_tokens = timings.get("draft_n") or 0
            chunk.draft_accepted = timings.get("draft_n_accepted") or 0
        return chunk
