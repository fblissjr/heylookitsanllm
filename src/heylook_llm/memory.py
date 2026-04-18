"""
Memory & observability telemetry for heylookitsanllm.

Three disk-backed JSONL streams plus a one-shot startup record:
- internal/log/memory_baseline.jsonl : periodic resource snapshot (configurable
  interval, default 3600s, disable with 0)
- internal/log/request_events.jsonl  : one line per completed request (sampler
  settings, timings, peak memory, cache hit rate, draft stats, stop reason)
- internal/log/model_events.jsonl    : model load/unload events with
  weights_bytes, quantization, param_count, context_length
- internal/log/baseline.jsonl        : one-shot startup record with
  mx.metal.device_info() + AppConfig subset

Content invariant (load-bearing, applies to every stream): numeric and metadata
only. Never prompt text, response text, token ID sequences, message content, or
conversation handles. Token *counts* are fine; token IDs are not. Snapshot and
event-builder functions take no request object, no conversation, no message
body -- only primitives and dicts of primitives.

All writes are best-effort: a failed append logs a warning but never raises.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import orjson
import psutil

try:
    import mlx.core as mx
    import mlx.utils as mx_utils
    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    mx_utils = None  # type: ignore[assignment]
    HAS_MLX = False

# Hoisted from inside MemoryManager._prompt_cache_stats -- no circular
# dependency risk since prompt_cache.py does not import memory.
from heylook_llm.providers.common.prompt_cache import get_global_cache_manager


DEFAULT_LOG_DIR = Path("internal/log")
BASELINE_FILE = "memory_baseline.jsonl"
REQUEST_FILE = "request_events.jsonl"
MODEL_FILE = "model_events.jsonl"
STARTUP_FILE = "baseline.jsonl"

StopReason = Literal["stop", "length", "error", "abort"]


def safe_mm_call(mm: Any, method_name: str, *args: Any, **kwargs: Any) -> None:
    """Call ``mm.method_name(*args, **kwargs)`` if ``mm`` is not None, swallow errors.

    Replaces the ``if mm is not None: try: ... except: logging.debug(...)`` pattern
    at every MemoryManager call site. Observability failures must never break
    request handling.
    """
    if mm is None:
        return
    try:
        getattr(mm, method_name)(*args, **kwargs)
    except Exception:
        logging.debug(f"memory_manager.{method_name} failed", exc_info=True)


@dataclass
class ModelMetadata:
    """Per-loaded-model metadata captured at load time."""
    model_id: str
    path: str
    weights_bytes: int
    architecture: str
    quantization: str
    param_count: int
    context_length: int


def parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_path_for_log(raw_path: str) -> str:
    """Strip the user's home directory prefix from a model path.

    Model paths in `models.toml` often contain the user's home dir. That's not
    prompt content, but it's still user-identifying noise in the observability
    streams. Replace it with ``~`` so log files stay portable and grep-friendly.
    """
    if not raw_path:
        return ""
    try:
        home = str(Path.home())
    except Exception:
        return raw_path
    if raw_path.startswith(home + os.sep) or raw_path == home:
        return "~" + raw_path[len(home):]
    return raw_path


def capture_model_metadata(model_id: str, provider: Any, model_path: str) -> ModelMetadata:
    """Best-effort metadata extraction from a freshly-loaded provider.

    All fields degrade to safe defaults if probing fails -- the metadata is for
    observability, not correctness.
    """
    weights_bytes = 0
    try:
        p = Path(model_path)
        if p.is_dir():
            for child in p.iterdir():
                if child.is_file() and child.suffix in (".safetensors", ".npz", ".gguf"):
                    weights_bytes += child.stat().st_size
        elif p.is_file():
            weights_bytes = p.stat().st_size
    except Exception:
        pass

    model_obj = getattr(provider, "model", None)

    architecture = type(model_obj).__name__ if model_obj is not None else "unknown"
    try:
        cfg = getattr(model_obj, "config", None) or getattr(model_obj, "args", None)
        if cfg is not None:
            architecture = getattr(cfg, "model_type", architecture) or architecture
    except Exception:
        pass

    quantization = "none"
    try:
        cfg = getattr(model_obj, "args", None) or getattr(model_obj, "config", None)
        q = getattr(cfg, "quantization", None) if cfg is not None else None
        if isinstance(q, dict):
            bits = q.get("bits")
            quantization = f"{bits}bit" if bits else "quantized"
        elif q:
            quantization = str(q)
    except Exception:
        pass

    param_count = 0
    if HAS_MLX and model_obj is not None and mx_utils is not None:
        try:
            params = mx_utils.tree_flatten(model_obj.parameters())
            for _, arr in params:
                size = getattr(arr, "size", None)
                if size is not None:
                    param_count += int(size)
        except Exception:
            pass

    context_length = 0
    try:
        cfg = getattr(model_obj, "args", None) or getattr(model_obj, "config", None)
        if cfg is not None:
            for attr in ("max_position_embeddings", "max_context_length", "max_seq_len"):
                val = getattr(cfg, attr, None)
                if val:
                    context_length = int(val)
                    break
    except Exception:
        pass

    return ModelMetadata(
        model_id=model_id,
        path=_normalize_path_for_log(str(model_path)),
        weights_bytes=weights_bytes,
        architecture=architecture,
        quantization=quantization,
        param_count=param_count,
        context_length=context_length,
    )


class MemoryManager:
    """
    Owns the three JSONL log streams and the model metadata cache.

    Constructed in the FastAPI lifespan; reachable via
    ``request.app.state.memory_manager``. All logging methods are thread-safe
    via an internal lock. Disk writes are best-effort -- a failed append logs a
    warning but never raises.
    """

    def __init__(
        self,
        router: Any,
        app_config: Any,
        log_dir: Path = DEFAULT_LOG_DIR,
    ):
        self.router = router
        self.app_config = app_config
        self.log_dir = log_dir

        env_interval = os.environ.get("HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS")
        if env_interval is not None:
            try:
                self.baseline_interval = max(0, int(env_interval))
            except ValueError:
                self.baseline_interval = int(getattr(app_config, "baseline_log_interval_seconds", 3600))
        else:
            self.baseline_interval = int(getattr(app_config, "baseline_log_interval_seconds", 3600))

        self.request_log_enabled = parse_bool_env(
            os.environ.get("HEYLOOK_REQUEST_LOG_ENABLED"),
            bool(getattr(app_config, "request_log_enabled", True)),
        )
        self.model_event_log_enabled = parse_bool_env(
            os.environ.get("HEYLOOK_MODEL_EVENT_LOG_ENABLED"),
            bool(getattr(app_config, "model_event_log_enabled", True)),
        )

        self._lock = threading.Lock()
        self._last_baseline_ts = 0.0
        self._last_request_ts = time.time()
        self._inflight_requests = 0
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)

        self.model_metadata: dict[str, ModelMetadata] = {}
        self.mode = "background"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logging.warning(f"MemoryManager: could not create {self.log_dir}: {exc}")

    # -- request lifecycle -------------------------------------------------

    def mark_request_start(self) -> None:
        with self._lock:
            self._inflight_requests += 1
            self._last_request_ts = time.time()

    def mark_request_end(self) -> None:
        with self._lock:
            self._inflight_requests = max(0, self._inflight_requests - 1)
            self._last_request_ts = time.time()

    # -- model lifecycle ---------------------------------------------------

    def register_model_load(
        self,
        metadata: ModelMetadata,
        load_duration_ms: float,
    ) -> None:
        with self._lock:
            self.model_metadata[metadata.model_id] = metadata

        if not self.model_event_log_enabled:
            return

        record = {
            "ts": time.time(),
            "event": "load",
            "model_id": metadata.model_id,
            "path": metadata.path,
            "weights_bytes": metadata.weights_bytes,
            "architecture": metadata.architecture,
            "quantization": metadata.quantization,
            "param_count": metadata.param_count,
            "context_length": metadata.context_length,
            "load_duration_ms": round(float(load_duration_ms), 1),
        }
        self._append_jsonl(self.log_dir / MODEL_FILE, record)

    def register_model_unload(self, model_id: str, reason: str) -> None:
        with self._lock:
            self.model_metadata.pop(model_id, None)

        if not self.model_event_log_enabled:
            return

        record = {
            "ts": time.time(),
            "event": "unload",
            "model_id": model_id,
            "reason": reason,
        }
        self._append_jsonl(self.log_dir / MODEL_FILE, record)

    # -- per-request log ---------------------------------------------------

    def log_request_event(self, event: dict) -> None:
        """Append one per-request record. Caller enforces the content invariant."""
        if not self.request_log_enabled:
            return
        self._append_jsonl(self.log_dir / REQUEST_FILE, event)

    # -- periodic maintenance ----------------------------------------------

    def tick(self) -> None:
        """Periodic maintenance called from the 60s resource-snapshot loop.

        Currently: trigger idle-unload on the router. S2.4 adds memory-pressure
        reclaim here. Best-effort -- per-call exceptions log at debug and
        never propagate (observability-path discipline).
        """
        router = getattr(self, "router", None)
        if router is None:
            return
        try:
            router.unload_idle_models()
        except Exception:
            logging.debug("MemoryManager.tick: unload_idle_models failed", exc_info=True)

    # -- periodic baseline -------------------------------------------------

    def maybe_log_baseline(self) -> bool:
        """Write a baseline record if ``baseline_interval`` has elapsed."""
        if self.baseline_interval <= 0:
            return False
        now = time.time()
        if now - self._last_baseline_ts < self.baseline_interval:
            return False
        try:
            snapshot = self.snapshot()
        except Exception:
            logging.exception("MemoryManager: snapshot failed")
            return False
        self._append_jsonl(self.log_dir / BASELINE_FILE, snapshot)
        self._last_baseline_ts = now
        return True

    def snapshot(self) -> dict:
        """Build one baseline-record dict without writing to disk."""
        now = time.time()
        vm = psutil.virtual_memory()
        rss_bytes = self._process.memory_info().rss
        cpu_percent = self._process.cpu_percent(interval=None)

        with self._lock:
            inflight = self._inflight_requests
            last_request_ts = self._last_request_ts
            mode = self.mode
            loaded = [
                {
                    "model_id": m.model_id,
                    "weights_bytes": m.weights_bytes,
                    "architecture": m.architecture,
                    "quantization": m.quantization,
                    "param_count": m.param_count,
                    "context_length": m.context_length,
                }
                for m in self.model_metadata.values()
            ]

        record: dict[str, Any] = {
            "ts": now,
            "rss_bytes": int(rss_bytes),
            "available_ram_bytes": int(vm.available),
            "cpu_percent": round(cpu_percent, 2),
            "mlx_active_bytes": 0,
            "mlx_peak_bytes": 0,
            "mlx_cache_bytes": 0,
            "loaded_models": loaded,
            "prompt_cache": self._prompt_cache_stats(),
            "vision_cache": self._vision_cache_stats(),
            "idle_seconds": round(now - last_request_ts, 2),
            "mode": mode,
            "inflight_requests": inflight,
        }

        if mx is not None:
            try:
                # Top-level accessors; mx.metal.get_* were deprecated.
                record["mlx_active_bytes"] = int(mx.get_active_memory())
                record["mlx_peak_bytes"] = int(mx.get_peak_memory())
                record["mlx_cache_bytes"] = int(mx.get_cache_memory())
            except Exception:
                pass

        return record

    def _prompt_cache_stats(self) -> dict:
        try:
            cm = get_global_cache_manager()
            return {"total_bytes": int(cm.total_cache_bytes)}
        except Exception:
            return {"total_bytes": 0}

    def _vision_cache_stats(self) -> dict:
        combined = {"entries": 0, "bytes": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
        if self.router is None:
            return combined
        try:
            providers = self.router.get_loaded_models()
        except Exception:
            return combined

        hits = 0
        misses = 0
        entries = 0
        nbytes = 0
        for provider in providers.values():
            cache = None
            strategies = getattr(provider, "_strategies", None)
            if isinstance(strategies, dict):
                vision_strategy = strategies.get("vision")
                if vision_strategy is not None:
                    cache = getattr(vision_strategy, "_vision_cache", None)
            if cache is None or not hasattr(cache, "stats"):
                continue
            try:
                s = cache.stats()
            except Exception:
                continue
            hits += int(s.get("hits", 0))
            misses += int(s.get("misses", 0))
            entries += int(s.get("entries", 0))
            nbytes += int(s.get("bytes", 0))

        total = hits + misses
        combined["entries"] = entries
        combined["bytes"] = nbytes
        combined["hits"] = hits
        combined["misses"] = misses
        combined["hit_rate"] = (hits / total) if total > 0 else 0.0
        return combined

    # -- startup self-description ------------------------------------------

    def log_startup_info(self) -> None:
        """Write one record describing hardware + config toggles. Always runs."""
        info: dict[str, Any] = {
            "ts": time.time(),
            "event": "startup",
            "baseline_interval_seconds": self.baseline_interval,
            "request_log_enabled": self.request_log_enabled,
            "model_event_log_enabled": self.model_event_log_enabled,
            "max_loaded_models": int(getattr(self.app_config, "max_loaded_models", 0) or 0),
        }
        if mx is not None:
            try:
                info["device_info"] = {k: _json_safe(v) for k, v in mx.metal.device_info().items()}
            except Exception:
                info["device_info"] = None
        self._append_jsonl(self.log_dir / STARTUP_FILE, info)

    # -- private ------------------------------------------------------------

    def _append_jsonl(self, path: Path, record: dict) -> None:
        try:
            line = orjson.dumps(record) + b"\n"
            with open(path, "ab") as f:
                f.write(line)
        except Exception as exc:
            logging.warning(f"MemoryManager: failed to append to {path}: {exc}")


def _json_safe(value: Any) -> Any:
    """Coerce a device_info value to a JSON-friendly primitive."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sampler_summary_from_request(request: Any) -> dict:
    """Extract the sampler knobs from a ChatRequest as a plain dict.

    Only the knobs -- not the messages. Passing in the messages is a content
    invariant violation; this function is the canonical way to produce the
    sampler_summary field for a request event.
    """
    fields = (
        "temperature", "top_p", "top_k", "min_p",
        "repetition_penalty", "repetition_context_size", "presence_penalty",
        "max_tokens", "enable_thinking", "seed",
        "preset",
    )
    result: dict[str, Any] = {}
    for name in fields:
        value = getattr(request, name, None)
        if value is not None:
            result[name] = value
    return result
