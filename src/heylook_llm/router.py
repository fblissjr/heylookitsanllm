# src/heylook_llm/router.py
import tomllib
import logging
import threading
import time
import gc
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from pathlib import Path

from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider
from heylook_llm.diagnostic_logger import diag_event
from heylook_llm import observability

# Try to import MLX provider
try:
    from heylook_llm.providers.mlx_provider import MLXProvider
    import mlx.core as mx
    HAS_MLX = True
except ImportError as e:
    MLXProvider = None
    HAS_MLX = False
    # Store the error for later logging
    MLX_IMPORT_ERROR = str(e)

class ModelNotFound(ValueError):
    """The requested model could not be RESOLVED to a config entry.

    Distinct from every other ValueError `get_provider` can raise: a failed
    load (mlx-lm/transformers raise bare ValueError for corrupt weights, an
    unsupported model_type, a malformed config.json) and operator/config faults
    (missing provider install, unknown provider) are SERVER errors and must
    keep surfacing as 500s. Only this one is the caller naming a model that
    isn't there -- the API layer maps it to 400. Subclasses ValueError so
    existing `except ValueError` consumers are unaffected.
    """


# The server config file: scan folders, the default model, the load limit and
# the operational settings ([settings]). Per-model settings live in each
# model's own model.heylook.toml. It was models.toml before v2.0.122; a
# leftover one is refused by name, never read.
CONFIG_FILENAME = "heylook.toml"
LEGACY_CONFIG_FILENAME = "models.toml"


class ModelRouter:
    """Manages loading, unloading, and routing to different model providers with an LRU cache."""
    def __init__(self, config_path: str, log_level: int, initial_model_id: Optional[str] = None):
        # Store config path for reload
        self.config_path = config_path

        # Load config (TOML only)
        self.app_config = self._load_config(config_path)

        self.providers = OrderedDict()
        self.max_loaded_models = self.app_config.max_loaded_models
        logging.info(f"Router configured to keep up to {self.max_loaded_models} models in memory.")

        # Log available providers
        if HAS_MLX:
            logging.debug("MLX provider is available")
        else:
            if 'MLX_IMPORT_ERROR' in globals() and 'mlx_vlm' in MLX_IMPORT_ERROR:
                logging.warning("MLX provider not available: mlx-vlm not installed. Run: uv sync")
            else:
                logging.debug("MLX provider not available. Install with: uv sync")

        self.log_level = log_level

        # Fine-grained locking: separate locks for cache access and model loading
        self.cache_lock = threading.RLock()  # For quick cache operations
        self.loading_locks: Dict[str, threading.Lock] = {}  # Per-model loading locks
        self.loading_locks_lock = threading.Lock()  # Protect loading_locks dict

        # Capacity reservations for in-flight loads. The capacity check and
        # the multi-hundred-ms load can't share one lock hold, so without a
        # reservation two concurrent different-model loads both pass the
        # check and hold two full models in memory (check-then-act TOCTOU,
        # OOM-class on boxes sized for max_loaded_models). A side-set rather
        # than a sentinel inside self.providers, so
        # self.providers always means "real, loaded providers" and reader
        # APIs need no filtering discipline.
        # model_id -> PROVIDER KIND ('mlx' / 'gguf'). A set until v2.0.49:
        # provider exclusivity has to see in-flight loads too, or two
        # concurrent loads of different engines both pass the check.
        self._loading: dict[str, str] = {}
        # Ceiling on how long a loader waits for another thread's in-flight
        # load to free capacity. Must exceed the slowest legitimate load
        # (100GB+ giants take minutes); its job is to turn a WEDGED load
        # into a loud error instead of silently blocking admission of every
        # other model forever (each blocked get_provider also pins an
        # asyncio default-executor thread).
        self._reservation_wait_timeout: float = 600.0

        # Idle-unload tracking (C2). time.time() of the last cache hit or load
        # per model_id. Consulted by ``unload_idle_models`` against each model's
        # effective threshold. Kept separate from self.providers' OrderedDict
        # position so unload decisions read from an explicit signal, not LRU
        # ordering.
        self._last_used_ts: Dict[str, float] = {}

        # Observability (S1.2). Set by api.py lifespan after construction.
        self.memory_manager: Optional[Any] = None

        # Startup pre-warm is OPT-IN and only ever explicit (`--model-id`).
        # `default_model` is a ROUTING fallback for requests that name no model
        # (see get_provider) -- it deliberately does NOT preload, so opening the
        # server doesn't pin a multi-GB model into RAM nobody asked for.
        initial_model_to_load = initial_model_id or None
        enabled_models = self.app_config.models
        if not enabled_models:
            logging.error("No models found (heylook.toml and the [scan] folders). Server cannot serve requests.")
            return

        # Validate the requested initial model
        if initial_model_to_load:
            model_config = self.app_config.get_model_config(initial_model_to_load)
            if not model_config:
                logging.warning(f"Initial model '{initial_model_to_load}' not found or disabled.")
                initial_model_to_load = None
            elif model_config.provider == "mlx" and not HAS_MLX:
                logging.warning(f"Initial model '{initial_model_to_load}' requires MLX provider which is not installed.")
                initial_model_to_load = None

        # Validate (never load) the routing default. Startup used to check it
        # implicitly by pre-warming it; now that preload is opt-in, an
        # unresolvable default would otherwise stay invisible until some
        # model-less request failed at runtime.
        if self.app_config.default_model and not self.app_config.get_model_config(
            self.app_config.default_model
        ):
            logging.warning(
                f"default_model '{self.app_config.default_model}' is not a known enabled model. "
                f"Requests that name no model will fail. Available: "
                f"{[m.id for m in enabled_models]}"
            )

        if not initial_model_to_load:
            logging.info("No startup model requested. Models will be loaded on first request.")

        if initial_model_to_load:
            try:
                logging.info(f"Pre-warming initial model: {initial_model_to_load}")
                self.get_provider(initial_model_to_load)
                logging.info(f"Successfully pre-warmed model: {initial_model_to_load}")
            except Exception as e:
                logging.error(f"Failed to pre-warm initial model '{initial_model_to_load}': {e}")
                logging.warning(f"Continuing without pre-warming. Model '{initial_model_to_load}' will be loaded on first request.")

    def _load_config(self, config_path: str) -> AppConfig:
        """Load configuration from TOML, fold in discovered models, validate.

        Both AppConfig construction sites (__init__ and reload_config) come
        through here, so discovery applies to a reload as well -- dropping a
        model into a scan folder and hitting reload is enough to serve it.
        The merge and validation are `model_registry.served`, the function
        `served_diff` also calls, so a diff cannot disagree with the server.

        The merge is LOAD-time only: the config file is never written. See
        model_registry for the rule (explicit entries win, matched by resolved
        model_path) and for why discovery can only ever add models, never
        change or remove the ones written down.
        """
        config_file = Path(config_path)
        if config_file.suffix != ".toml":
            config_file = config_file.with_suffix(".toml")
        if not config_file.exists():
            old = config_file.with_name(LEGACY_CONFIG_FILENAME)
            if config_file.name == CONFIG_FILENAME and old.exists():
                raise FileNotFoundError(
                    f"{config_file} not found, but {old} is there: the server config "
                    f"file is {CONFIG_FILENAME} since v2.0.122. Rename it.")
            raise FileNotFoundError(
                f"Config file not found: {config_file}. Create it with a [scan] "
                f"section naming the folders your models live in.")
        with open(config_file, 'rb') as f:
            return self._with_discovered(tomllib.load(f))

    # Set by every config load (_with_discovered): the ids with a heylook.toml
    # entry, and for each the config discovery derives for the same file. The
    # engine contract reads both so /v1/models never re-reads heylook.toml or
    # re-runs discovery to say which settings are really configured.
    written_ids: frozenset = frozenset()
    derived_configs: dict = {}
    # id -> where its stored values live, for every model that has any: a
    # heylook.toml entry, or its own model.heylook.toml.
    stored_in: dict = {}

    def _with_discovered(self, config_data: dict) -> AppConfig:
        """Fold `[scan].folders` discoveries into the parsed config."""
        from heylook_llm.model_registry import (
            derived_for_explicit,
            discover,
            served,
        )

        ModelRouter._audit_configured_paths(config_data)
        discovered = discover(config_data)
        self.written_ids = frozenset(
            str(e["id"]) for e in config_data.get("models") or [] if e.get("id"))
        self.derived_configs = derived_for_explicit(config_data, discovered)
        app = served(config_data, discovered)
        stored = {mid: "heylook.toml entry" for mid in self.written_ids}
        by_id = {str(e.get("id")): e for e in discovered if e.get("sidecar")}
        for m in app.models:
            e = by_id.get(m.id)
            if m.id not in stored and e is not None:
                stored[m.id] = "model.heylook.toml"
                self.derived_configs[m.id] = e["derived"]
        self.stored_in = stored
        return app

    # Last audit report emitted, so a reload re-reports only on CHANGE.
    # `None` (never audited) is deliberately distinct from `""` (audited,
    # nothing wrong) -- otherwise the first clean audit looks like a repeat.
    _paths_audited_report: "str | None" = None

    @staticmethod
    def _audit_configured_paths(config_data: dict) -> None:
        """Name every EXPLICIT entry whose configured paths no longer resolve.

        Reorganising a model directory does not touch heylook.toml, so a
        hand-written entry keeps pointing at the old layout and nothing says
        so until someone tries to chat with it -- at which point llama-server
        exits 1 with its output discarded (see the provider's spawn pre-flight
        for that half). Both halves shipped after a folder rename cost an
        evening on 2026-09-06.

        Only EXPLICIT entries are audited: a discovered one came off the
        filesystem a moment ago, so it cannot be stale in this way. Warn, never
        raise -- one dead entry must not stop the other models from serving,
        which is the same best-effort posture discovery itself takes.

        ONE SUMMARY, ONCE PER PROCESS. This warned per entry per field on every
        load, so a config with several stale entries printed the same block at
        startup and again on every reload -- and a warning that repeats
        unchanged is one people learn to scroll past, which is the opposite of
        what it is for. Owner ask 2026-09-08.

        It also says, per entry, whether deleting it would disturb anything
        else. That question is the one a reader actually has, and answering it
        HERE is only safe because a dead path is the narrow case: discovery
        cannot re-add an entry whose path does not exist, so removal takes the
        id and nothing more. The general question -- what a config edit does to
        the served set -- is NOT answerable from this loop and must not be
        guessed at from it: entries are matched to discovery on RESOLVED PATH
        while the served unit is an id, and those are not one-to-one. Two
        entries can claim one path (verified in this repo: a plain entry and a
        text-only twin), so deleting a live entry that reads as redundant can
        remove a model outright because the twin still claims its path. If a
        general "what would this edit change" answer is ever wanted, it belongs
        in a diff over the merge, not in an existence check.
        """
        FIELDS = ("model_path", "mmproj_path", "draft_model_path",
                  "chat_template_path")
        entries = config_data.get("models") or []
        # Which resolved paths more than one entry claims -- see the docstring.
        # RESOLVED, matching the merge: `merge_discovered` matches an entry to a
        # discovered model on `Path(...).resolve()`, so answering "does another
        # entry claim this" on `expanduser()` alone predicts a DIFFERENT
        # relation than the one the reader is about to act on. Two spellings of
        # one directory -- a vendor symlink in a model folder and the real path -- are
        # the case this repo has already hit, and expanduser reports them as
        # unrelated. Falls back to expanduser for a path that cannot resolve.
        def _identity(value: str) -> str:
            try:
                return str(Path(value).expanduser().resolve())
            except OSError:
                return str(Path(value).expanduser())

        claimed: dict[str, int] = {}
        for entry in entries:
            value = (entry.get("config") or {}).get("model_path")
            if value:
                # Bound once: _identity() calls resolve(), which walks symlinks
                # on the filesystem, and this ran it twice for every entry on
                # every startup AND every reload_config.
                key = _identity(value)
                claimed[key] = claimed.get(key, 0) + 1

        dead: list[str] = []
        for entry in entries:
            cfg = entry.get("config") or {}
            # exists(), not is_file(): an MLX model_path is a DIRECTORY.
            missing = [f for f in FIELDS
                       if cfg.get(f) and not Path(cfg[f]).expanduser().exists()]
            if not missing:
                continue
            model_path = cfg.get("model_path")
            shared = model_path and claimed.get(_identity(model_path), 0) > 1
            note = (" -- another entry also claims this model_path, so removing "
                    "this one leaves that one serving it"
                    if shared else
                    " -- removing it takes this id and nothing else")
            dead.append(f"  - {entry.get('id', '<unnamed>')}: "
                        f"{', '.join(missing)} does not exist{note}")

        # REPORT ON CHANGE, not once per process. The original silenced every
        # repeat because a warning that repeats unchanged is one people learn to
        # scroll past -- true, and it also silenced a warning that CHANGED. An
        # admin edit writing a bad path after startup goes through reload_config
        # -> _load_config -> here, and was then never reported for the life of
        # the process. Keyed on the report text, so a fixed entry going quiet
        # and a newly-broken one speaking up both work, while a reload that
        # changes nothing stays silent.
        report = "\n".join(dead)
        if report == ModelRouter._paths_audited_report:
            return
        ModelRouter._paths_audited_report = report
        if not dead:
            return
        logging.warning(
            "[config] entries in heylook.toml point at paths that no longer "
            "exist and will fail to load:\n%s\n"
            "Fix the path, or delete the entry -- a model under [scan].folders "
            "is served with derived defaults and needs no entry at all. This "
            "is reported again only when it changes.",
            report)

    def _get_or_create_loading_lock(self, model_id: str) -> threading.Lock:
        """Get or create a loading lock for a specific model."""
        with self.loading_locks_lock:
            if model_id not in self.loading_locks:
                self.loading_locks[model_id] = threading.Lock()
            return self.loading_locks[model_id]

    def _check_cache(self, model_id: str) -> Optional[BaseProvider]:
        """Check if model is in cache. Uses fine-grained locking."""
        with self.cache_lock:
            if model_id in self.providers:
                if self.log_level <= logging.DEBUG:
                    logging.debug(f"Cache hit for model: {model_id}. Reusing existing provider.")
                self.providers.move_to_end(model_id)
                self._last_used_ts[model_id] = time.time()
                return self.providers[model_id]
            return None

    def _teardown_provider(self, provider: BaseProvider) -> None:
        """Run a provider's unload + GC + Metal-cache clear sequence.

        Must be called without ``cache_lock`` held -- unloading MLX weights
        can take hundreds of ms and other threads need to continue cache
        reads. Shared between LRU eviction and idle unload so both paths
        stay in lockstep if teardown gains new steps (e.g. cache persistence
        in S3.1).
        """
        # provider_name is the BaseProvider class attribute (7a) -- the old
        # getattr(provider, "provider") gate matched nothing and made this
        # cache-clear dead code.
        is_mlx_model = getattr(provider, "provider_name", "") == "mlx"
        try:
            provider.unload()
        except Exception:
            logging.error("Provider unload failed", exc_info=True)
        del provider
        gc.collect()
        if HAS_MLX and is_mlx_model:
            mx.clear_cache()

    def _is_generating(self, model_id: str) -> bool:
        """Is this model in the middle of a generation right now?

        Tearing a model down mid-generation is unsafe in DIFFERENT ways per
        provider, which is why the guard belongs here and not in either one:
        MLXProvider.unload frees weights under a live Metal command buffer
        (its own docstring says that crashes), and LlamaServerProvider.unload
        SIGTERMs the llama-server subprocess with no wait for actives at all --
        `_active_generations` is MLX-local, so gguf had strictly less
        protection. One predicate at the router covers both.

        Reads BaseProvider's per-provider in-flight count, so it is exact for
        the model asked about rather than conservative across models -- the
        earlier spelling read the process-global generation gate, which gguf
        does not use at all (llama-server queues its own requests), so every
        gguf model looked permanently idle.

        Why this became load-bearing (v1.79.12): a generation used to exist
        only while a client watched it, so the window was small. Runs now
        outlive the response that started them, which widens it to the full
        length of every abandoned generation.
        """
        provider = self.providers.get(model_id)
        if provider is None:
            return False
        count = getattr(provider, "active_generations", 0)
        # int(), not truth-test: a bare Mock is truthy and would report every
        # mocked provider as permanently generating and un-unloadable.
        return isinstance(count, int) and count > 0

    def _provider_kind(self, model_id: str) -> Optional[str]:
        """The engine family holding ``model_id``'s weights, or None."""
        return getattr(self.providers.get(model_id), "_heylook_provider_kind", None)

    def _evict_foreign_providers(self, kind: str) -> bool:
        """Drop every resident model that is NOT of engine family ``kind``.

        Must be called with cache_lock held. Returns True if it evicted
        anything, so the caller re-runs its capacity check.

        WHY THIS IS A RULE AND NOT A TUNING CHOICE. MLX holds weights in THIS
        process's Metal working set; a gguf model is a llama-server SUBPROCESS
        with its own. The two engines do not agree on what the working-set
        ceiling means -- ram_fit's header spells it out: MLX treats
        ``max_recommended_working_set_size`` as HARD (over the line is a
        refusal, and a Metal fault can poison this process), while llama.cpp
        checks it as a debug warning and merely degrades into paging. Worse,
        ``mx.set_wired_limit`` is set ONCE at startup to the full
        recommendation and never shrinks when a subprocess takes residency, so
        MLX keeps believing it owns the whole budget. A mixed pair therefore
        degrades the gguf side gently and can kill the MLX side outright.

        ``max_loaded_models`` cannot express that: it is a COUNT, and at 2 it
        admits any pair. Two MLX models are the case the accounting gets right
        (one process, one wired limit, and usable_gb() sees both), so the rule
        is per PROVIDER -- mlx-lm and mlx-vlm are one process and one kind.

        EVICTING, not refusing: a cross-engine switch should cost a load, not
        return an error. Load cost is disclosed here, never confirmed.
        """
        foreign = [mid for mid in self.providers if self._provider_kind(mid) not in (kind, None)]
        if not foreign:
            return False

        busy = [m for m in foreign if self._is_generating(m)]
        if busy:
            from heylook_llm.providers.common.generation_gate import ModelBusyError
            raise ModelBusyError(
                f"MODEL_BUSY: loading a '{kind}' model needs {busy} unloaded "
                f"(one engine family at a time), but "
                f"{'it is' if len(busy) == 1 else 'they are'} generating. "
                f"Stop the generation or wait for it to finish."
            )

        for mid in foreign:
            was = self._provider_kind(mid) or "other"   # read BEFORE the pop
            provider = self.providers.pop(mid)
            self._last_used_ts.pop(mid, None)
            logging.info(f"Engine switch to '{kind}': evicting {was} model {mid}")
            diag_event("model_evict", model=mid)
            observability.record_event(
                "model_unload", tier="events", min_level="minimal",
                fields={"model": mid, "reason": "provider_switch"})
            from heylook_llm.memory import safe_mm_call
            safe_mm_call(self.memory_manager, "register_model_unload", mid, reason="provider_switch")
            self._teardown_provider(provider)
        return True

    def _evict_lru_model(self):
        """Evict the least recently used model that is not generating.

        Must be called with cache_lock held.
        """
        evict_id = None
        blocked_by_generation = []
        for model_id in self.providers:
            if self._is_generating(model_id):
                blocked_by_generation.append(model_id)
                continue
            evict_id = model_id
            break

        if evict_id is None:
            if blocked_by_generation:
                # MODEL_BUSY is the existing backpressure contract: the API
                # layer maps it to 503 + Retry-After and v3 already retries on
                # it with a "Server busy" line. Better than evicting a running
                # generation, and better than inventing a second vocabulary.
                from heylook_llm.providers.common.generation_gate import ModelBusyError
                raise ModelBusyError(
                    f"MODEL_BUSY: cannot make room -- {blocked_by_generation} "
                    f"{'is' if len(blocked_by_generation) == 1 else 'are'} "
                    f"generating. Stop the generation or wait for it to finish."
                )
            raise RuntimeError("Cannot evict to make room: no model is loaded.")

        lru_provider = self.providers.pop(evict_id)
        self._last_used_ts.pop(evict_id, None)
        logging.info(f"Cache full. Evicting model: {evict_id}")
        diag_event("model_evict", model=evict_id)
        observability.record_event("model_unload", tier="events", min_level="minimal",
                                   fields={"model": evict_id, "reason": "lru_evict"})
        from heylook_llm.memory import safe_mm_call
        safe_mm_call(self.memory_manager, "register_model_unload", evict_id, reason="lru_evict")

        self.cache_lock.release()
        try:
            self._teardown_provider(lru_provider)
        finally:
            self.cache_lock.acquire()

    def get_current_model_id(self) -> Optional[str]:
        """Get the most recently used model ID from cache, or None if no models loaded."""
        if self.providers:
            # OrderedDict keeps insertion order; last item is most recently used
            return next(reversed(self.providers))
        return None

    def get_loaded_models(self) -> Dict[str, BaseProvider]:
        """
        Get all currently loaded models.

        Returns:
            Dict mapping model_id to BaseProvider instance.
        """
        with self.cache_lock:
            # Return a copy to prevent external modification
            return dict(self.providers)

    def get_provider(self, model_id: str) -> BaseProvider:
        # Fallback logic when no model specified:
        # 1. Use currently loaded model (most recently used)
        # 2. Use default_model from config
        # 3. Raise error with available models
        if not model_id:
            model_id = self.get_current_model_id()
            if model_id:
                logging.debug(f"No model specified, using loaded model: {model_id}")
            elif self.app_config.default_model:
                model_id = self.app_config.default_model
                logging.debug(f"No model specified, using default: {model_id}")
            else:
                available = [m.id for m in self.app_config.models]
                raise ModelNotFound(f"No model specified and no default configured. Available: {available}")

        # Fast path: check cache first
        provider = self._check_cache(model_id)
        if provider:
            return provider

        # Get model-specific loading lock
        loading_lock = self._get_or_create_loading_lock(model_id)

        # Acquire loading lock for this specific model
        with loading_lock:
            # Double-check cache after acquiring lock (another thread might have loaded it)
            provider = self._check_cache(model_id)
            if provider:
                return provider

            # Model needs to be loaded
            load_start_time = time.time()

            # Get model config
            model_config = self.app_config.get_model_config(model_id)
            if not model_config:
                available = [m.id for m in self.app_config.models]
                raise ModelNotFound(f"Model '{model_id}' not found or disabled. Available: {available}")

            # Keep keys in sync with config.PROVIDER_CONFIG_CLASSES.
            provider_map = {}
            if MLXProvider:
                provider_map["mlx"] = MLXProvider
            from heylook_llm.providers.llama_server_provider import LlamaServerProvider
            provider_map["gguf"] = LlamaServerProvider

            provider_class = provider_map.get(model_config.provider)
            if not provider_class:
                if model_config.provider == "mlx" and not HAS_MLX:
                    raise ValueError(f"MLX provider requested but not installed. Run: uv sync")
                else:
                    raise ValueError(f"Unknown provider: {model_config.provider}")

            logging.info(f"Loading model '{model_id}' with provider '{model_config.provider}'...")

            # Show loading progress
            model_path = model_config.config.model_path if hasattr(model_config.config, 'model_path') else 'unknown'
            logging.info(f"Model path: {model_path}")

            try:
                # Reserve capacity BEFORE loading (evicting if needed) so the
                # capacity check and the load are one atomic commitment.
                # Without a reservation, two concurrent different-model loads
                # both pass the check and hold two full models in memory
                # (check-then-act TOCTOU). If capacity is held by other
                # threads' in-flight loads, wait (bounded) for one to publish.
                reservation_wait_start = time.time()
                while True:
                    with self.cache_lock:
                        # ONE ENGINE FAMILY AT A TIME, checked before the count:
                        # max_loaded_models is a count and at 2 would admit an
                        # MLX + gguf pair. See _evict_foreign_providers for why
                        # that pair specifically is unsafe.
                        self._evict_foreign_providers(model_config.provider)
                        # A foreign load already in flight cannot be evicted --
                        # it owns no provider object yet. Wait for it to publish
                        # (then it is evictable) rather than racing it.
                        foreign_inflight = sorted(
                            m for m, k in self._loading.items()
                            if k != model_config.provider and m != model_id
                        )
                        if (not foreign_inflight
                                and len(self.providers) + len(self._loading) < self.max_loaded_models):
                            self._loading[model_id] = model_config.provider
                            break
                        if foreign_inflight:
                            # Nothing to evict our way out of -- evicting a
                            # SAME-kind model here would throw away a model we
                            # want in order to wait for one we are about to
                            # evict anyway. Just wait for it to publish.
                            inflight = foreign_inflight
                        elif self.providers:
                            self._evict_lru_model()
                            continue
                        elif not self._loading:
                            raise RuntimeError(
                                f"Cannot make room for '{model_id}': nothing is loaded "
                                f"or loading, and max_loaded_models is {self.max_loaded_models}."
                            )
                        else:
                            inflight = sorted(self._loading)
                    if time.time() - reservation_wait_start > self._reservation_wait_timeout:
                        raise RuntimeError(
                            f"Timed out after {self._reservation_wait_timeout:.0f}s waiting "
                            f"for model-load capacity to free (in-flight loads: {inflight}). "
                            f"A load may be wedged."
                        )
                    time.sleep(0.05)

                # Create provider instance
                new_provider = provider_class(
                    model_config.id,
                    model_config.config.model_dump(),
                    self.log_level <= logging.DEBUG
                )
                # Which ENGINE FAMILY holds this model's weights, stamped on the
                # instance rather than looked up later: a reload can change a
                # model's provider in heylook.toml while it is resident, and the
                # exclusivity rule must reason about what is IN MEMORY, not what
                # the config now says. Stamped once here, and it dies with the
                # object -- no teardown path can leave a stale entry behind.
                new_provider._heylook_provider_kind = model_config.provider

                logging.info(f"Initializing {model_config.provider.upper()} provider...")

                # Load model (this is the expensive operation)
                new_provider.load_model()

                # Prime JIT caches before publishing the provider so concurrent
                # cache hits don't race a half-warmed model. `warmup()`'s contract
                # (BaseProvider docstring) requires it to swallow exceptions; no
                # wrapper needed here.
                new_provider.warmup()

                # Publish: the reservation becomes the real provider.
                with self.cache_lock:
                    self._loading.pop(model_id, None)
                    self.providers[model_id] = new_provider
                    self._last_used_ts[model_id] = time.time()

                load_time = time.time() - load_start_time
                logging.info(f"Successfully loaded model: {model_id} in {load_time:.2f}s")
                diag_event("model_load", model=model_id, provider=model_config.provider,
                           load_time_s=round(load_time, 2))
                observability.record_event("model_load", tier="events", min_level="minimal",
                                           fields={"model": model_id, "provider": model_config.provider,
                                                   "load_time_s": round(load_time, 2)})

                if self.memory_manager is not None:
                    from heylook_llm.memory import capture_model_metadata, safe_mm_call
                    try:
                        metadata = capture_model_metadata(
                            model_id,
                            new_provider,
                            getattr(model_config.config, "model_path", ""),
                        )
                    except Exception:
                        logging.debug("capture_model_metadata failed", exc_info=True)
                        metadata = None
                    if metadata is not None:
                        safe_mm_call(self.memory_manager, "register_model_load", metadata, load_time * 1000.0)

                # Log cache state after loading
                if self.log_level <= logging.DEBUG:
                    with self.cache_lock:
                        logging.debug(f"Router cache state after loading: {list(self.providers.keys())}")

                return new_provider

            except Exception as e:
                # Release the reservation so the failed load doesn't hold
                # capacity forever.
                with self.cache_lock:
                    self._loading.pop(model_id, None)
                load_time = time.time() - load_start_time
                logging.error(f"Failed to load model '{model_id}' after {load_time:.2f}s: {e}")
                raise e

    def list_available_models(self) -> list[str]:
        return [m.id for m in self.app_config.models]

    def clear_cache(self):
        """Clear all loaded models from cache."""
        with self.cache_lock:
            # Unload all models
            for model_id in list(self.providers.keys()):
                provider = self.providers[model_id]
                try:
                    provider.unload()
                    logging.info(f"Unloaded model: {model_id}")
                except Exception as e:
                    logging.error(f"Error unloading model {model_id}: {e}")

            # Clear the cache (OrderedDict maintains order automatically)
            self.providers.clear()
            self._last_used_ts.clear()
            logging.info("Model cache cleared")
    
    # Set by the app's lifespan, never by construction: unit tests build
    # routers directly over a mocked MLX tree, and a background thread
    # touching those mocks is the teardown-crash class tests/README warns of.
    warm_model_facts_on_load: bool = False

    def warm_model_facts(self) -> None:
        """Derive every model's row facts once, in the background.

        The derivation /v1/models and the admin row run (template parses,
        header reads, the engine contract's static half) is cached by file
        stamps, so the FIRST listing after a start or reload pays it cold.
        Warming here moves that cost off the user's first page load. Never
        fatal, never blocking: failures are logged and dropped, like
        discovery. A warm that races a later reload only fills entries whose
        stamps nobody reads.
        """
        def run(app_config):
            from heylook_llm.capabilities import derived_model_facts
            for mc in list(app_config.models):
                try:
                    derived_model_facts(mc, self)
                except Exception:  # noqa: BLE001 -- warming is best-effort
                    logging.debug("[router] warming row facts failed for %s",
                                  mc.id, exc_info=True)

        threading.Thread(target=run, args=(self.app_config,),
                         name="warm-model-facts", daemon=True).start()

    def reload_config(self):
        """Reload model configuration from file."""
        try:
            # Reload the configuration from stored path
            self.app_config = self._load_config(self.config_path)
            self.max_loaded_models = self.app_config.max_loaded_models
            self._refresh_per_request_defaults()
            if self.warm_model_facts_on_load:
                self.warm_model_facts()
            logging.info(f"Model configuration reloaded from {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            raise

    def _refresh_per_request_defaults(self):
        """Push per_request defaults from the fresh config into LOADED providers.

        A provider is constructed with a SNAPSHOT of its config dict and reads
        per_request defaults (enable_thinking, temperature,
        reasoning_effort, ...) from that snapshot at request time -- so without
        this, a PATCH to one of them returned "no reload required" while the
        loaded model kept serving the old default: the exact stale-snapshot
        lie the effect classification exists to prevent, relocated into the
        per_request bucket. Only per_request keys are touched -- spawn/load
        state (requires_reload, load_time_only) genuinely needs the reload the
        API reports. Plain dict-key writes; a request mid-flight reads each
        default at most once, and a torn read across two defaults is no worse
        than the request racing the PATCH itself.
        """
        from heylook_llm.config import (
            EFFECT_PER_REQUEST, PROVIDER_CONFIG_CLASSES, fields_by_effect,
        )
        with self.cache_lock:
            for model_id, provider in self.providers.items():
                # NOT get_model_config: that filters on `enabled`, and a
                # disabled-but-still-loaded model (toggle does not unload)
                # must keep receiving refreshes -- re-enabling it must not
                # resurrect stale defaults.
                model_config = self._any_model_config(model_id)
                if model_config is None:
                    continue  # entry removed while loaded; unload handles it
                # Guard the class match: re-import can change an entry's
                # provider without evicting the loaded provider, and pushing
                # one class's per_request keys into another class's snapshot
                # is silent cross-class bleed.
                if model_config.provider != getattr(provider, "provider_name", None):
                    continue
                cls = PROVIDER_CONFIG_CLASSES.get(model_config.provider)
                cfg_dict = getattr(provider, "config", None)
                if cls is None or not isinstance(cfg_dict, dict):
                    continue
                fresh = model_config.config.model_dump()
                for key in fields_by_effect(cls).get(EFFECT_PER_REQUEST, ()):
                    cfg_dict[key] = fresh[key]

    def _any_model_config(self, model_id: str):
        """The config entry for ``model_id`` regardless of ``enabled``."""
        for m in self.app_config.models:
            if m.id == model_id:
                return m
        return None

    def stale_reload_fields(self, model_id: str) -> list:
        """requires_reload fields whose saved value differs from what the
        LOADED provider was constructed with. Empty for unloaded models.

        This is the server-derived truth behind any "config changed --
        reload to apply" marker: client-side bookkeeping of the same fact
        dies on page remount and drifts on partial failures, while the
        snapshot comparison cannot (the provider's config dict IS what the
        process was built from; per_request keys are refreshed live and so
        never diff here).
        """
        from heylook_llm.config import (
            PROVIDER_CONFIG_CLASSES, reload_required_fields,
        )
        with self.cache_lock:
            provider = self.providers.get(model_id)
        if provider is None:
            return []
        model_config = self._any_model_config(model_id)
        if model_config is None:
            return []
        if model_config.provider != getattr(provider, "provider_name", None):
            # Provider changed under a loaded model (re-import): everything
            # about the process is stale; report the identity field.
            return ["provider"]
        cls = PROVIDER_CONFIG_CLASSES.get(model_config.provider)
        snap = getattr(provider, "config", None)
        if cls is None or not isinstance(snap, dict):
            return []
        fresh = model_config.config.model_dump()
        stale = sorted(
            key for key in reload_required_fields(cls)
            if snap.get(key) != fresh.get(key)
        )
        # The template binds at load too, but lives in a FILE (a sidecar
        # created, edited or deleted beside the weights), so no config field
        # moves when it changes. Same comparison the template editor's
        # `stale` makes: what the process loaded vs what a respawn would use.
        loaded = getattr(provider, "loaded_chat_template", None)
        if loaded is not None:
            from heylook_llm import chat_template_files
            try:
                if chat_template_files.view(model_id, model_config.provider,
                                            fresh, loaded).stale:
                    stale.append("chat_template")
            except Exception as e:  # noqa: BLE001 - a listing must not fail on a file read
                logging.debug(f"template staleness for {model_id} unknown: {e}")
        return stale

    def unload_model(self, model_id: str, force: bool = False) -> bool:
        """Explicitly unload a specific model from cache.

        Returns True if the model was loaded and is now unloaded, False if it wasn't loaded.
        Raises RuntimeError if the model is generating and force=False.
        """
        with self.cache_lock:
            if self._is_generating(model_id) and not force:
                raise RuntimeError(
                    f"Model '{model_id}' is generating. "
                    f"Stop the generation, or use force=True to override."
                )
            if model_id not in self.providers:
                return False

            provider = self.providers.pop(model_id)

        # Unload outside the cache lock to avoid holding it during slow ops.
        # provider_name is the BaseProvider class attribute (7a); the old
        # `provider.provider` gate matched nothing (dead cache-clear).
        is_mlx = getattr(provider, "provider_name", "") == "mlx"
        try:
            provider.unload()
            del provider
            gc.collect()
            if HAS_MLX and is_mlx:
                mx.clear_cache()
            logging.info(f"Explicitly unloaded model: {model_id}")
        except Exception as e:
            logging.error(f"Error unloading model {model_id}: {e}")

        return True

    def unload_all(self) -> List[str]:
        """Unload every loaded provider. For server shutdown.

        Never raises: this runs on the way out, and one provider that throws must
        not strand the ones behind it. Matters most for the gguf provider,
        where "loaded" IS a running llama-server subprocess in its own
        process group -- anything left here is a multi-GB orphan that
        outlives the server.

        Returns the ids it unloaded.
        """
        with self.cache_lock:
            items = list(self.providers.items())
            self.providers.clear()
            self._last_used_ts.clear()

        unloaded = []
        for model_id, provider in items:
            try:
                provider.unload()
                unloaded.append(model_id)
            except Exception:
                logging.error(f"Error unloading model {model_id} at shutdown", exc_info=True)
        if unloaded:
            logging.info(f"Unloaded {len(unloaded)} model(s) at shutdown: {', '.join(unloaded)}")
        return unloaded

    def get_model_status(self, model_id: str) -> dict:
        """Get load status and basic metrics for a model."""
        with self.cache_lock:
            loaded = model_id in self.providers

        status = {"loaded": loaded}

        if loaded:
            provider = self.providers.get(model_id)
            if provider:
                # Try to get memory info
                if hasattr(provider, 'get_memory_usage'):
                    try:
                        status["memory_mb"] = provider.get_memory_usage()
                    except Exception:
                        pass
                # Busy vs idle, on every engine: the provider's own count of
                # generations in flight (gguf included; the field was null
                # for every model before 2026-09-25), and the generation
                # gate's queue, which is process-wide.
                status["requests_active"] = provider.active_generations
                queue = provider.generation_queue_stats() or {}
                status["requests_waiting"] = queue.get("waiting")

        return status

    def _effective_idle_threshold(self, model_id: str) -> int:
        """Per-model override beats global default. ``0`` at either level means
        "disabled"; per-model non-zero override wins over a ``0`` global.
        Returns ``0`` when no idle-unload should happen for this model.
        """
        model_config = self.app_config.get_model_config(model_id)
        per_model = None
        if model_config is not None:
            per_model = getattr(model_config.config, "unload_after_idle_seconds", None)
        if per_model is not None:
            return int(per_model)
        return int(getattr(self.app_config, "idle_unload_seconds", 0))

    def unload_idle_models(self, now_ts: Optional[float] = None) -> List[str]:
        """Unload models whose idle window has elapsed.

        Driven by ``MemoryManager.tick()`` on the 60s resource-snapshot loop.
        Models with an effective threshold of ``0``
        (explicit per-model disable, or global disable with no per-model
        override) are never touched.

        ``now_ts`` defaults to ``time.time()``; tests inject a fake clock.
        Returns the list of ``model_id`` values that were unloaded.
        """
        if now_ts is None:
            now_ts = time.time()

        with self.cache_lock:
            candidates = []
            for model_id in list(self.providers.keys()):
                # An idle timer that fires mid-generation is the same teardown
                # hazard as an evict. NOT unreachable, and the stamp is why:
                # last_used is written at REQUEST time (get_provider), so a
                # generation longer than the threshold is ALREADY stale for its
                # whole second half -- routine since v1.79.12, when runs began
                # outliving the response that started them.
                #
                # Re-stamping is the other half. Skipping without it deferred
                # the unload by exactly one tick: the run finishes, the next
                # tick still sees a stale timestamp, and the model is torn down
                # the instant the reply lands -- immediately before the user's
                # next message. Generating IS use; say so.
                if self._is_generating(model_id):
                    self._last_used_ts[model_id] = now_ts
                    continue
                threshold = self._effective_idle_threshold(model_id)
                if threshold <= 0:
                    continue
                last_used = self._last_used_ts.get(model_id, now_ts)
                if now_ts - last_used > threshold:
                    candidates.append(model_id)

        unloaded = []
        for model_id in candidates:
            if self._unload_idle(model_id):
                unloaded.append(model_id)
        return unloaded

    def _unload_idle(self, model_id: str) -> bool:
        """Pop + tear down a single idle model. Unload runs outside the
        cache lock; weight release can take hundreds of ms and would stall
        concurrent cache reads otherwise.

        The busy check and the pop happen under ONE cache_lock hold: a
        request WAITING at the FIFO generation gate is neither 'active' nor
        recently-used (last_used was stamped at its cache hit, and gate
        waits can outlast the idle threshold) -- unloading then would tear
        the weights down under a request that's about to run. Any cache hit
        after this pop simply reloads.

        This re-check is the ONLY thing standing between the candidate scan
        and the pop: ``unload_idle_models`` computes candidates, RELEASES the
        lock, then calls this. Anything that becomes true in that window has
        to be caught here or not at all, so every condition the scan skips on
        is re-tested -- `_is_generating` included. It was
        missing, and the queue-stats test does not cover either: it returns
        None for the gguf provider (llama-server queues its own requests), so
        `if stats and ...` is False and a gguf generation started inside the
        window took a SIGTERM under an open stream -- exactly the "gguf had
        strictly less protection" hole `_is_generating` was written to close.
        """
        with self.cache_lock:
            provider = self.providers.get(model_id)
            if provider is None:
                return False
            stats = provider.generation_queue_stats()
            busy = None
            if self._is_generating(model_id):
                busy = "generating"
            elif stats and (stats.get("active", 0) > 0 or stats.get("waiting", 0) > 0):
                busy = (f"generation queue busy (active={stats.get('active', 0)}, "
                        f"waiting={stats.get('waiting', 0)})")
            if busy:
                # Every one of these means IN USE, so all three re-stamp. A
                # refusal without a stamp only defers by one tick: last_used is
                # written at REQUEST time, so the model is already stale, and
                # the tick after the work finishes unloads it the instant the
                # reply lands -- right before the user's next message.
                self._last_used_ts[model_id] = time.time()
                logging.info(f"Skipping idle unload of {model_id}: {busy}")
                return False
            self.providers.pop(model_id, None)
            self._last_used_ts.pop(model_id, None)

        logging.info(f"Idle timeout. Unloading model: {model_id}")
        diag_event("model_idle_unload", model=model_id)
        observability.record_event("model_unload", tier="events", min_level="minimal",
                                   fields={"model": model_id, "reason": "idle_timeout"})
        from heylook_llm.memory import safe_mm_call
        safe_mm_call(self.memory_manager, "register_model_unload", model_id, reason="idle_timeout")
        self._teardown_provider(provider)
        return True

    def is_loading(self, model_id: str) -> bool:
        """Whether a load of this model is in flight (capacity reserved but
        the provider not yet published to ``providers``). The reload route
        refuses on this instead of silently JOINING the in-flight load and
        reporting a reload that never happened -- unload_model returns False
        for a loading model (it's not in ``providers`` yet), so without this
        check the route cannot tell 'not loaded' from 'loading right now'."""
        with self.cache_lock:
            return model_id in self._loading
