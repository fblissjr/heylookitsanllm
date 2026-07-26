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
# - Sampler cascade mirrors MLX: GLOBAL_SAMPLER_FLOOR -> model
#   default_sampler -> request.sampler -> explicit request fields.
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
from ..samplers import GLOBAL_SAMPLER_FLOOR, SamplerNotFound, get_sampler_registry
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

# Request sampler fields copied verbatim into the cascade when set.
_REQUEST_SAMPLER_FIELDS = (
    "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "presence_penalty", "max_tokens", "seed",
)

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

    def _resolve_binary(self) -> Path:
        candidate = self.config.get("server_binary") or os.environ.get("HEYLOOK_LLAMA_SERVER")
        if not candidate:
            raise RuntimeError(
                "No llama-server binary configured. Set server_binary in "
                "models.toml or the $HEYLOOK_LLAMA_SERVER env var."
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
        if cfg.get("spec_draft_n_max"):
            args += ["--spec-draft-n-max", str(cfg["spec_draft_n_max"])]
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
        # Cascade order mirrors MLX: floor -> model config -> model's
        # default_sampler -> request's named sampler -> explicit request
        # fields. The config overlay is UNCONDITIONAL over the floor (the
        # floor pre-seeds max_tokens, so a "not in merged" guard can never
        # fire -- the dead-overlay bug caught in the 2026-07-26 review).
        merged = dict(GLOBAL_SAMPLER_FLOOR)
        if self.config.get("max_tokens"):
            merged["max_tokens"] = self.config["max_tokens"]
        registry = get_sampler_registry()
        registry.apply_sampler(merged, self.config.get("default_sampler"))
        registry.apply_sampler(merged, request.sampler)

        for field in _REQUEST_SAMPLER_FIELDS:
            value = getattr(request, field, None)
            if value is not None:
                merged[field] = value
        # thinking: a sampler bundle may carry enable_thinking; the request wins
        if request.enable_thinking is not None:
            merged["enable_thinking"] = request.enable_thinking

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

    def create_chat_completion(self, request: ChatRequest, abort_event=None) -> Generator:
        if self._base_url is None:
            raise GenerationFailed(f"Model '{self.model_id}' is not loaded")
        try:
            payload = self._build_payload(request)
        except SamplerNotFound as e:
            raise InvalidGenerationRequest(str(e))

        http_request = urllib.request.Request(
            self._base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(http_request, timeout=_SSE_READ_TIMEOUT_S)
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
            yield from self._stream_chunks(response, abort_event)
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

    def _stream_chunks(self, fp, abort_event) -> Generator:
        """Adapt llama-server SSE lines to GenerationChunk.

        Split out from create_chat_completion so it is unit-testable with a
        canned byte stream -- no HTTP, no subprocess.
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
            if chunk is not None:
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
