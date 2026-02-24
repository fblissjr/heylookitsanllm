# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.19.0

### Fixed

- **Profile apply bug**: `ModelProfile.apply()` now unconditionally sets profile values instead of only filling gaps. Previously, smart defaults ran first and set `top_k`, `max_tokens`, `cache_type`, etc., so profiles could never override them. Precedence is now: base -> smart_defaults -> profile overrides -> user `--override`.
- **Sub-1B model size regex**: `Qwen3-0.6B` was reported as "(6B)" because the integer pattern `(\d+)b` matched before `(\d+\.\d+)b`. Swapped regex order so decimal patterns match first.

### Changed

- **Profiles moved to TOML files**: The 9 hardcoded Python profile definitions are now standalone TOML files in `profiles/`. Each file has `[meta]` (name, description) and `[defaults]` (flat key=value). Lambda-based dynamic values removed; size-based logic stays in `get_smart_defaults()`. Profiles are loaded once at module import via `tomllib`.
- **Profile renames**: `fast` -> `tight_fast`, `balanced` -> `moderate`, `quality` -> `wide_sampling`, `performance` -> `high_throughput`, `max_quality` -> `widest_sampling`, `background` -> `low_resource`, `memory` -> `quantized_kv`, `interactive` -> `conversation`, `encoder` -> `embedding`.
- **Dynamic profile discovery**: `--profile` choices in CLI are now discovered from `profiles/*.toml` filenames instead of a hardcoded list. Adding a new profile is just dropping a `.toml` file.
- **Profile values printed on import**: When `--profile` is used, the parameter table is printed before writing. The "Available profiles" listing now includes a parameter summary for each profile.

### Removed

- **`/v1/performance` stub endpoint**: Deleted the stub that returned "Removed in v1.17.1". No consumers.
- **`/v1/performance/profile/{time_range}` endpoint**: Deleted ~170 lines of DuckDB queries for performance profiling. The analytics SQL endpoint (`/v1/data/query`) provides equivalent ad-hoc access.
- **`test_performance_monitoring()` integration test**: Deleted stale test that tested the removed endpoint.
- **`mlx` optional extra**: Removed from pyproject.toml. `mlx`, `mlx-lm`, `mlx-vlm`, `transformers` are already in core `dependencies`; the extra duplicated them and added unused packages.
- **Unused dependencies from extras**: Removed `torch`, `torchvision`, `opencv-python`, `scipy` (never imported in production code). Moved `datasets` to `analytics` extra (only used by data loader). Moved `rich` to `scripts` extra (only used by `scripts/metrics_dashboard.py`).

### Added

- **`gguf` placeholder extra**: Empty extra with commented `llama-cpp-python` for Phase Next.
- **`scripts` extra**: Contains `rich` for dashboard scripts.
- **`load_profiles()` / `get_available_profiles()` API**: Public functions for programmatic profile access.
- **Contract test suite**: TestClient-based API tests (39 tests) covering `/v1/models`, `/v1/chat/completions`, `/v1/messages`, `/v1/admin/models/`, and OpenAPI conformance. Runs in-process, no server or models needed.
- **Profile unit tests**: 22 tests covering TOML profile loading, profile application with provider filtering, model size regex, and the `load_profiles()` caching API.
- **`httpx` test dependency**: Added to `[test]` extra for Starlette TestClient support.

## 1.18.1

### Added

- **`--interactive` flag for model import**: `heylookllm import --interactive` launches a TUI (via ConfigEditor) to customize sampler and KV cache settings for each discovered model before writing `models.toml`. Compatible with `--profile` (profile applies first, interactive tweaks override).

### Changed

- **Documentation refresh for v1.18.0**: Updated CLAUDE.md, architecture.md, mlx.md, mlx_optimization_plan.md, and TODO.md to reflect vision path unification via pre-filled cache pattern.

## 1.18.0

### Changed

- **Vision path unification**: Replaced `mlx_vlm.generate.stream_generate` with a pre-filled cache pattern (inspired by vllm-mlx). The full VLM model runs a single forward pass to encode vision + text into a KV cache, then the language model generates tokens using `generation_core.run_generation()` -- the same code path as text-only requests. Vision requests now get the full sampler suite (top_k, min_p, presence_penalty, logit_bias, XTC), abort support, and speculative decoding acceptance tracking. Eliminates the hardcoded Qwen `[1, 24, 24]` image grid -- `mlx_vlm.utils.prepare_inputs` handles grid dimensions natively per model.
- **Syntax check auto-discovery**: `scripts/syntax_check.py` now uses `glob.glob("src/heylook_llm/**/*.py")` instead of a hand-curated file list. Adding or removing source files no longer requires updating the script.
- **Cache miss logging**: Radix cache misses now log at INFO level with model ID for observability parity with cache hits.

### Removed

- **`vlm_generation.py`**: Deleted entirely (55 lines). The `stream_generate_vlm_vision()` wrapper around `mlx_vlm.generate.stream_generate` is replaced by the pre-filled cache approach in `VLMVisionStrategy`. The Qwen model-type string sniffing for `image_grid_thw` is no longer needed.
- **`BatchVisionEncoder` import**: Removed unused import from `mlx_provider.py`.

## 1.17.1

### Removed

- **`PerformanceMonitor`**: Deleted `performance_monitor.py` (234 lines) and all `@time_mlx_operation` decorators. No consumers, no tests, generator-wrapping overhead on every token. `/v1/performance` endpoint returns a stub response.
- **`VLMGeneratorWithSampling` class**: Flattened to standalone `stream_generate_vlm_vision()` function (~55 lines, down from 161). Dead text-only branch, `LanguageModelLogitsWrapper` cache, and `lm_stream_generate` import removed -- all unreachable after phase 5 unification.
- **Duplicate `_reconstruct_thinking`**: Deleted copy from `mlx_provider.py`. Canonical version lives in `providers/common/vlm_inputs.py`.

## 1.17.0

### Added

- **`generate_text()` entry point**: New high-level function in `generation_core.py` that builds sampler/processors internally and delegates to `run_generation()`. Strategies call this instead of building samplers externally, keeping sampler construction co-located with the generation loop.
- **Dynamic draft token tuning (`DraftTuner`)**: Module-level singleton in `generation_core.py` that dynamically adjusts `num_draft_tokens` per model based on rolling acceptance rate. Conservative policy: increase by 1 (max 8) when acceptance > 80%, decrease by 1 (min 1) when < 50%, over a 50-sample window. Integrated into `run_generation()` automatically.
- **Standalone VLM input preparation**: Extracted `VLMVisionStrategy._prepare_vlm_inputs_parallel` (92 lines) to `providers/common/vlm_inputs.py` as a standalone function. Testable without instantiating a full strategy.
- **Unified path equivalence tests**: Parameterized tests proving `UnifiedTextStrategy` produces equivalent `generate_text()` calls for `is_vlm=True` and `is_vlm=False`.

### Changed

- **Sampler construction moved out of `create_chat_completion()`**: `build_sampler()` no longer called in the routing layer. `UnifiedTextStrategy` uses `generate_text()` (builds sampler internally); `VLMVisionStrategy` builds its own sampler at the start of `generate()`. Strategy signatures no longer include `sampler`/`processors` parameters.
- **`run_generation()` consults DraftTuner**: When a draft model is active, `run_generation()` queries `DraftTuner` for the current token count before calling `lm_stream_generate`, and feeds acceptance data back in the `finally` block.

## 1.16.0

### Changed

- **Provider strategy unification**: Merged `TextOnlyStrategy` and `VLMTextOnlyStrategy` (~400 lines of duplication) into a single `UnifiedTextStrategy` (~130 lines) that dispatches on `is_vlm` for chat template application and model wrapping. All shared logic (cache config, prompt cache, generation loop, acceptance tracking, KV snapshot storage) extracted to `generation_core.run_generation()`. Strategy keys changed from `text_only`/`vlm_text`/`vlm_vision` to `text`/`vision`.
- **`LanguageModelLogitsWrapper` moved**: Relocated from `mlx_provider.py` to `providers/common/model_wrappers.py` to break circular import with `vlm_generation.py`.
- **Generation core extracted**: New `providers/common/generation_core.py` contains the single generation loop (`run_generation`), cache config construction (`_build_cache_config`), and prompt cache setup (`_setup_prompt_cache`). This is the integration point for future `mx.compile` optimization.
- **Simplified routing**: `_compile_strategies()` now creates 1-2 strategies (text, optionally vision) instead of 2-3. `create_chat_completion()` routing reduced to a simple `has_images` check.

## 1.15.0

### Fixed

- **`num_draft_tokens` passthrough**: Both `TextOnlyStrategy` and `VLMTextOnlyStrategy` now pass `num_draft_tokens` to `lm_stream_generate`. Previously, the configured value (default 6) was never forwarded, causing `speculative_generate_step` to use its hardcoded default of 2. New default changed from 6 to 3 (safe middle ground between mlx-lm's default of 2 and the overly aggressive 6).
- **`VLMTextOnlyStrategy` missing `model_config`**: Added `model_config` parameter so VLM text-only path can access model-level config (thinking mode, cache settings, etc.), matching `TextOnlyStrategy`.
- **`cache_config` ignoring model config**: The `kv_bits` fallback changed from hardcoded `8` to `None`, and `_apply_model_defaults()` now explicitly includes cache config fields (`cache_type`, `kv_bits`, `kv_group_size`, `max_kv_size`, `quantized_kv_start`, `num_draft_tokens`) from model config. Previously, if the request didn't specify cache params, the strategy's fallback defaults would override model config.
- **Analytics DB size limit enforced**: The cleanup thread now prunes the oldest 25% of records and runs VACUUM when the database exceeds `max_db_size_mb`, instead of just logging a warning.

### Added

- **Memory-pressure eviction for radix cache**: `RadixCache` accepts an optional `memory_pressure_fn` callback checked before each node insertion. When GPU memory exceeds 85% of the recommended working set, eviction triggers even if node count is below `max_nodes`. Keeps the radix cache pure (no MLX dependency in the data structure).
- **Speculative decoding acceptance tracking**: Both text-only strategies now count draft token acceptance/rejection during generation and log the acceptance rate in the finally block. Provides visibility into whether speculation is helping without changing behavior.
- **Startup disk usage logging**: Server startup now logs analytics DB size (with limit) and log directory size for disk usage visibility.
- **`num_draft_tokens` in smart defaults**: `get_smart_defaults()` now includes `num_draft_tokens=3` for MLX models, ensuring the field is always present when users configure `draft_model_path`.

### Changed

- **Profile cache threshold aligned**: The `balanced` profile's quantized KV cache threshold changed from >30GB to >13GB, matching `get_smart_defaults()`.

### Removed

- **`models.toml.example`**: Deleted. The `heylookllm import` command and smart defaults generate complete config. README updated to use `heylookllm import --hf-cache` instead of `cp models.toml.example`.

## 1.14.0

### Added

- **Radix-tree prefix cache**: New `RadixCache` data structure (`providers/common/radix_cache.py`) stores multiple cached prefixes per model simultaneously. Editing an earlier message, branching, or regenerating no longer invalidates the entire cache -- only the divergent suffix needs re-prefilling. Configurable block size (32 tokens), LRU leaf eviction, thread-safe.
- **KV snapshot helpers**: `snapshot_kv()` and `restore_kv_from_snapshot()` in `cache_helpers.py` capture and restore KV cache state for radix tree storage. Uses MLX copy-on-write semantics for cheap snapshots.

### Changed

- **Pure-MLX sampler**: `_mlx_unique()` in `samplers.py` reimplemented using `mx.sort` + `mx.cumsum` + scatter, replacing the numpy-based version that forced a full GPU-to-CPU sync on every token when presence penalty was active. Now only a single int32 scalar crosses the device boundary.
- **Prompt cache manager**: `PromptCacheManager` now uses a `RadixCache` per model as the persistent backing store instead of a single linear prefix cache. Public API (`get_or_create_cache`, `process_prompt_with_cache`) unchanged.

### Removed

- **numpy dependency from samplers**: `import numpy as np` removed from `samplers.py`. The presence penalty path now stays entirely on the Metal compute graph.

## 1.13.0

### Removed

- **Dead code**: Deleted `mlx_optimizations.py` (283 lines of hallucinated MLX APIs, zero imports anywhere)
- **Queue manager**: Removed `queue_manager.py` and all queue branches from `api.py`, `messages_api.py`, `router.py`. Queue was always disabled (`queue_config.enabled: false`) and fundamentally incompatible with streaming (`list(generator)` defeated the point). Removes ~400 lines of dead code.

### Added

- **Generation abort mechanism**: New `AbortEvent` (`providers/abort.py`) enables cooperative cancellation of in-flight MLX generation. When a new request arrives while another is generating, the current generation is aborted (per-token check) so the new request starts immediately instead of blocking. Client disconnect during SSE streaming also triggers abort, freeing GPU compute.

## 1.12.4

### Added

- **Models applet sorting**: Sort models by name (A-Z/Z-A), provider, or status (loaded first) via dropdown selector
- **Models applet tag filtering**: Data-driven tag chips extracted from model configs; click to filter by tag (OR logic)
- **Models applet preference persistence**: Sort and filter preferences persist to localStorage across sessions

## 1.12.3

### Changed

- **README.md rewrite**: Updated from v1.1-era to reflect current state -- dual API (OpenAI + Messages), 7 applets, llama-server subprocess (not llama-cpp-python), thinking blocks, logprobs, model management, hidden states. Removed dead links (`guides/SERVICE_SECURITY.md`, `docs/WINDOWS_INSTALL.md`) and stale installation instructions (`--extra llama-cpp`, `CMAKE_ARGS`/`llama-cpp-python` GPU section).
- **pyproject.toml version**: Bumped from 1.2.0 to 1.12.2 to match actual release.
- **performance extra trimmed**: Removed 6 unused packages (imagecodecs, blake3, diskcache, msgpack, aiofiles, aiocache). Kept xxhash, PyTurboJPEG, uvloop, cachetools.

## 1.12.2

### Fixed

- **Mobile delete button**: Tapping the trash icon on iOS Safari triggered conversation selection (and sidebar collapse) instead of deletion. Touch events now stop propagation on the delete button so they don't bubble to the parent's long-press handler.
- **Prompt cache stale KV entries**: After generation, the KV cache contained both prompt AND generated tokens, but token tracking only recorded the prompt. On regeneration or message editing, the trim calculation was wrong, leaving old response tokens in the KV cache and biasing the model toward repeating its previous output. Now tracks generated token IDs so trimming is accurate.

## 1.12.1

### Fixed

- **ModelImporter race condition**: `handleScan` used stale closure-captured `scanResults` after async `scanForModels()`. Now reads fresh state via `useModelsStore.getState().scanResults`.
- **model_service.py update_config mutation bug**: Updates were applied in-place to the models list before validation. Invalid updates corrupted in-memory state during lock hold. Now works on a deep copy and only commits if validation passes.
- **Path validation ordering**: `model_path` validation happened after the value was already written to the config dict. Moved to before any mutations.
- **default_model fallback**: When removing the default model, fallback now prefers enabled models instead of blindly using `models[0]`.
- **ModelDetail save/remove error handling**: `handleSave`, `handleConfigUpdate`, and `handleRemove` now catch errors and display them inline instead of leaving the UI in a broken state.
- **Import modal state persistence**: Local state (customPath, selectedIds, step, profile) now resets when the modal reopens.

### Added

- **Per-action loading states**: `actionLoading` store field tracks which model is being acted on. Load/unload buttons, toggle button, and preset buttons show spinners during operations.
- **Import modal error display**: Errors from scan/import now display inline in the modal instead of behind it.
- **PresetSelector empty state**: Shows "Failed to load profiles" instead of permanent "Loading..." when profile fetch fails. Uses `profilesLoaded` flag to distinguish loading from failure.
- **reload_config error handling**: All 6 admin API endpoints that call `router.reload_config()` now catch exceptions and include a warning in the response instead of crashing.
- **Models applet tests**: 70 tests across 3 files (modelsStore.test.ts, ModelList.test.tsx, ModelImporter.test.tsx). Total test count: 781.

### Changed

- **model_importer.py refactor**: Moved `PROFILES`, `ModelProfile`, `get_smart_defaults`, `get_hf_cache_paths` into `model_service.py` (single source of truth). `model_importer.py` reduced from 874 to 450 lines, now imports shared logic from `model_service`. Extracted `_detect_tags` helper to reduce duplication in MLX/GGUF entry creation.
- **Removed fragile route exclusion list**: `admin_api.py` no longer hard-codes sub-paths that the `{model_id:path}` catch-all must skip. FastAPI's two-router registration order handles this correctly.

## 1.12.0

### Added

- **Model Management System**: Full-stack model management with backend API and frontend applet
  - **ModelService** (`model_service.py`): Service layer for model discovery, validation, and config management with thread-safe atomic TOML writes, path validation, CRUD operations, scan/import, and profile application
  - **Admin API** (`admin_api.py`): 14 endpoints under `/v1/admin/models/` for CRUD, scan, import, validate, profiles, bulk-profile, load/unload, and status
  - **Router enhancements**: `unload_model()`, `get_model_status()`, `reload_single_model()` on ModelRouter
  - **Pydantic models**: `ScannedModelResponse`, `ModelScanRequest`, `ModelImportRequest`, `ModelUpdateRequest`, `AdminModelResponse`, `ModelStatusResponse`, `ProfileInfo`, etc.
- **Models applet** (`/models` route): 7th frontend applet for model management
  - Side-by-side list + detail layout (AppletLayout pattern)
  - Searchable, filterable model list with status pills (Loaded/Available/Disabled)
  - Full config editing with field-level reload indicators
  - Import workflow: scan filesystem/HF cache, select models, apply profile, import
  - Preset selector for quick profile application
  - Provider-specific config forms (MLX 17 fields, GGUF 13 fields, STT 6 fields)
  - Load/unload controls per model
  - Metadata editing (description, tags)
  - Config-only removal with confirmation

### Changed

- **AppNav**: Added Models entry with CubeIcon (7 nav items total)
- **MobileBottomNav**: Inherits Models entry via shared `navItems`
- **App.tsx**: Added lazy-loaded `/models` route

## 1.11.1

### Changed

- **Documentation restructure**: Reorganized flat `internal/` directory (36 files, 9 dead) into `backend/`, `backend/providers/`, `bugs/`, `research/`, `frontend/`, `session/`, `log/`. Deleted 9 obsolete files, consolidated 5 into 3.
- **CLAUDE.md rewrite**: Reduced from 322 lines to 87 lines. Now a nav hub that links out instead of duplicating content.
- **Stale CoreML STT references**: Updated all non-historical references to use MLX STT (docs, tests, scripts). CoreML STT was removed in v1.2.0.

## 1.11.0

### Added

- **Mobile bottom tab navigation**: New `MobileBottomNav` component provides access to all 6 applets on mobile (Chat, Batch, Token Explorer, Model Comparison, Performance, Notebook). Previously only Chat was reachable.
- **Shared `AppletLayout` component**: Reusable responsive wrapper for applets with left panels. Desktop shows inline panel; mobile hides it behind a toggle button with overlay drawer.
- **Model loading from any applet**: `ModelSelector` and right panels (Advanced, Settings) lifted from Chat-only `Layout` to `AppShell`, making model management available on every route.
- **iOS scroll fix**: `100dvh` height with `100vh` fallback, `-webkit-overflow-scrolling: touch` on chat scroll container.
- **Desktop content width constraints**: `max-w-3xl` on chat messages, `max-w-4xl` on batch/notebook/token-explorer, preventing unreadable line lengths on wide screens.

### Changed

- **AppShell absorbs shared chrome**: Header, SystemStatusBar, ModelSelector panel, AdvancedPanel, SettingsPanel, and mobile detection all moved from route-specific `Layout` to `AppShell`. `Layout` reduced to Chat sidebar wrapper only.
- **Header is route-aware**: Sidebar hamburger toggle only renders on `/chat`; other routes show a spacer.
- **ModelSelector no longer uses `onModelLoaded` callback**: Removed prop. ChatView now watches `loadedModel` store state directly to auto-create conversations.
- **AppNav exports `navItems`**: Shared between desktop sidebar and mobile bottom nav.
- **Applet LeftPanels stripped of outer wrappers**: `AppletLayout` now provides width, border, and responsive behavior.

### Frontend Tests

- 707 tests passing across 31 test files (was 686/28)
- New test files: `AppShell.test.tsx` (20), `MobileBottomNav.test.tsx` (6), `AppletLayout.test.tsx` (13)
- Rewrote `Layout.test.tsx` (sidebar-only), `Header.test.tsx` (route-aware), `ModelSelector.test.tsx` (removed callback tests)

## 1.10.1

### Fixed

- **`wired_limit` model mismatch in VLMTextOnlyStrategy**: `wired_limit()` was receiving the full VLM model (vision encoder + language model) but only the language model wrapper was running, causing incorrect Metal memory limit calculations. Now correctly passes `self._cached_wrapper` to match the model actually used for generation.
- **Generator detection in performance_monitor**: `time_operation` used `hasattr(result, '__next__')` which matched any iterator. Changed to `isinstance(result, types.GeneratorType)` for precise generator detection.
- **VLM vision path now forwards logits_processors and repetition_penalty**: `_stream_generate_vision_enhanced()` previously silently dropped `processors` and `repetition_penalty`. These are now forwarded to `vlm_stream_generate` as `logits_processors` and `repetition_penalty` kwargs (both supported by `mlx_vlm.generate.generate_step`).

### Removed

- Dead `_cached_generator` field from `VLMTextOnlyStrategy` (only used by `VLMVisionStrategy`)
- Duplicate `import threading` inside `MLXProvider.__init__` (already imported at module level)

## 1.10.0

### Fixed

- **VLM text-only sampling parity**: `VLMTextOnlyStrategy` now uses `lm_stream_generate` with full sampler/processor pipeline (top_k, min_p, presence_penalty, logit_bias, XTC). Previously bypassed all advanced sampling by calling `vlm_stream_generate` with raw temperature/top_p/repetition_penalty kwargs.
- **VLM text-only prompt caching**: Added prompt cache support to VLM text-only path (same pattern as `TextOnlyStrategy`), reducing token processing on follow-up requests.
- **Performance monitor generator timing**: `time_operation` decorator now correctly times generator functions from first to last yield, instead of measuring generator object creation time (microseconds).
- **`_apply_model_defaults` serialization**: Replaced `request.model_dump()` (serialized entire request including all messages) with direct `getattr()` for the 9 scalar parameter fields.

### Removed

- **`mlx_metal_tuning.py` module**: Deleted entirely. This module was called on every model load and caused active harm: cast all weights to float16 (destroying 4-bit quantized weights), pre-allocated unused KV cache buffers (wasting GPU memory), set wired limits that conflicted with per-generation context managers, and ran `subprocess.run(['sysctl', 'hw.memsize'])` on every load.
- **Broken `_content_cache`**: Removed image detection cache that used `id(messages)` as key (ephemeral per request, never hit, grew without bound).
- **Dead methods**: Removed `VLMTextOnlyStrategy._prepare_vlm_inputs`, `VLMVisionStrategy._prepare_vlm_inputs`, `VLMGeneratorWithSampling._apply_advanced_sampling`, `_get_vocab_size`, `_get_eos_token_id`, `supports_speculative_decoding`, and `vlm_stream_generate_with_sampling` convenience wrapper.
- **Unused imports**: Cleaned up `traceback`, `ABC`, `abstractmethod`, `vlm_generate`, `vlm_stream_generate`, `load_image`, `make_cache`, `BatchVisionStrategy` from mlx_provider.py. Removed `nn` from vlm_generation.py.

### Added

- **`mx.clear_cache()` after generation**: Added to `create_chat_completion` finally block to release MLX internal memory cache between requests, preventing memory accumulation.
- **`LanguageModelLogitsWrapper` caching in `VLMGeneratorWithSampling`**: Wrapper now created once in `__init__` instead of per-request.

## 1.9.0

### Added

- **`/v1/messages` endpoint**: Anthropic Messages-inspired API alongside existing `/v1/chat/completions`. Typed content blocks (text, image, thinking, logprobs), system prompt as top-level parameter, and structured SSE streaming with distinct event types (message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop). Uses `StreamingEventTranslator` state machine for event sequencing.
- **Testing infrastructure**: Root `tests/conftest.py` with shared fixtures (mock_mlx, mock_mlx_provider, mock_vlm_provider, sample requests). Reusable MLX mocking utilities in `tests/helpers/mlx_mock.py` for testing provider code on any platform without MLX installed.
- **MLX provider unit tests**: 26 tests covering initialization, strategy compilation, image detection, model defaults (including thinking mode defaults), metrics, cache clearing, unload safety, and error paths.
- **Glass Box backend tests**: 16 tests validating `_reconstruct_thinking()` round-trip behavior and assistant prefill convention. Covers thinking tag formatting, non-assistant message handling, None/empty thinking, and dict mutation semantics.
- **Config unit tests**: Rewrote `test_config.py` from 6-line script to 25 proper pytest tests covering ChatMessage, ChatRequest, ModelConfig, and AppConfig.
- **Messages API unit tests**: 21 tests for request/response converters and StreamingEventTranslator event sequencing.

### Removed

- Deleted `config_migration.py` (dead code -- YAML-to-TOML migration completed, file imported nowhere).

### Fixed

- Fixed `mlx_provider.py` header comment (was referencing old `mlx_provider_optimized.py` filename).

## 1.8.0

### Added

- **Glass Box: Universal Editability and Transparency**: Every token the model generates is now visible, editable, and round-trips correctly through the API.
  - **Backend thinking round-trip**: `ChatMessage` accepts `thinking` field; MLX provider reconstructs `<think>` tags before template application across all generation paths (text, VLM, batch). Assistant prefill support: when last message is `role: assistant`, sets `add_generation_prompt=False` for mid-response continuation.
  - **Shared `lib/messages.ts`**: Extracted from chatStore, now includes thinking on assistant messages in API payloads. Used by all applets.
  - **Shared `lib/stale.ts`**: Timestamp-based stale detection -- marks downstream messages when upstream edits occur. No stored flags.
  - **Editable ThinkingBlock**: Thinking blocks are now default-open and editable (save/cancel inline).
  - **Shared MessageActions**: Copy, edit, delete, regenerate, continue, next-turn actions in one component. Compact mode for tight layouts.
  - **StaleBadge**: Amber indicator on messages generated before upstream edits.
  - **Chat: Continue from message**: Prefill/append to partial assistant responses.
  - **Chat: Generate next turn**: Fresh assistant response from full history without a user message.
  - **Notebook: Thinking display**: ThinkingBlock appears above editor during and after generation with thinking models.
  - **Model Comparison: Editable results**: ThinkingBlock defaults open and editable on completed results. Compact message actions. New `editResult` store action.
  - **Token Explorer: Thinking token visibility**: Tracks thinking token boundary. Visual separator between thinking and response tokens in the stream.
- **PlayIcon, ForwardIcon**: Added to icon library (22 icons total).

## 1.7.1

### Fixed

- **Notebook persistence hardening**: Migrated document storage from localStorage (5MB limit) to IndexedDB, matching the pattern used by chatStore. Added `loaded` state flag to eliminate the 100ms setTimeout race condition on auto-document creation. One-shot migration reads existing localStorage data into IDB and removes the legacy key. Delete operations go directly to IDB (no debounce). Individual document saves via debounced put instead of full-array serialization.

## 1.7.0

### Added

- **Notebook Mode applet** (`applets/notebook/`): Sixth applet in the platform. Base-model text continuation simulator with a single monospace text buffer and cursor-based generation. The model continues from wherever the cursor is positioned, treating the text as a completion context. System prompt is visible and editable (not hidden). Optional image attachments provide vision model context. Documents persist to IndexedDB with debounced saves. Keyboard shortcuts: Cmd+Enter (generate), Escape (stop), Cmd+N (new document), Cmd+S (force save). Lazy-loaded at `/notebook`.
- **DocumentTextIcon**: Added to icon library (20 icons total).

## 1.6.0

### Added

- **Performance Dashboard applet** (`applets/performance/`): Fifth applet in the platform. Real-time system metrics (RAM, CPU, context usage) with color-coded thresholds. When analytics is enabled (`HEYLOOK_ANALYTICS_ENABLED=true`), displays timing breakdowns by operation type (queue, model load, image processing, token generation), throughput sparklines with TPS/request/error trends, and per-model performance table with response time and TTFT. Graceful degradation shows system metrics with a friendly message when analytics is disabled. Auto-polls system metrics at 5s and profile data at 30s. Lazy-loaded at `/perf`.
- **ChartBarIcon**: Added to icon library (19 icons total).
- **Sparkline component**: Reusable SVG sparkline with gradient fill for inline data visualization.
- **MiniBarChart component**: Reusable horizontal bar chart for comparing values with labeled bars.

### Changed

- **Type adapter layer**: `EnhancedUsage`, `GenerationTiming`, `GenerationConfig` in `types/api.ts` now re-export from `generated-api.ts` with optional-field normalization, reducing manual type drift surface.

## 1.5.1

### Fixed

- **MLX Provider safe unload**: Added reference counting (`_active_generations` counter) to prevent LRU cache eviction from unloading a model during active generation. `unload()` now waits up to 30 seconds for active generations to complete, with force-unload as a safety valve. Fixes potential Metal command buffer crashes when >2 models are requested concurrently (e.g., model comparison with 3+ models).

### Changed

- **Shared utility library**: Extracted 9 duplicated functions across 4 applets into `src/lib/` -- `generateId()`, `tokenFromLogprob()`, `displayToken()`, `probabilityToColor()`, `probabilityToBarColor()`. `ExplorerToken` and `ComparisonToken` are now type aliases for shared `LogprobToken`.
- **Map to Record migration**: `ComparisonRun.results` changed from `Map<string, ModelResult[]>` to `Record<string, ModelResult[]>` for JSON serialization and devtools compatibility. Extracted `updateRunResult()` helper to simplify store mutations.
- **Shared UI primitives**: Extracted `StatusBadge` (10 status variants), `StreamingCursor` (inline/block), `AlternativeBar` (default/compact), and `RunHistoryList` (generic collapsible) from applet-specific duplications.
- **OpenAPI streaming schema**: Backend now exposes `StreamChunk`, `StreamChoice`, `StreamDelta`, `TokenLogprobInfo`, `EnhancedUsage`, `GenerationTiming`, `GenerationConfig` Pydantic models in the OpenAPI spec. Frontend `types/api.ts` restructured as adapter layer with compile-time drift detection against `generated-api.ts`. `npm run generate:api` chains `openapi-typescript` with `sed` to convert `| null` to `| undefined` for TypeScript idiom compatibility.

## 1.5.0

### Added

- **Model Comparison applet** (`applets/model-comparison/`): Fourth applet in the platform. Run the same prompt against 2-6 models simultaneously, streaming responses side-by-side with per-model performance metrics (TTFT, tokens/sec, duration, token count). Supports batch mode (multiple prompts separated by `---`, navigated via tabs). Optional token probability visualization with colored chips and alternative token bars. Models execute sequentially via the backend LRU cache but appear concurrent in the UI. Includes run history, per-model stop controls, and keyboard shortcut (Escape to stop all). Lazy-loaded at `/compare`.
- **ScaleIcon**: Added to icon library (18 icons total).
- **DuckDB persistence stub**: `ComparisonPersistence` interface defined with no-op `sessionPersistence` adapter, ready for real implementation when analytics API is built.

## 1.4.0

### Added

- **Token Explorer applet** (`applets/token-explorer/`): Third applet in the platform. Stream tokens with real-time probability visualization using a continuous red-yellow-green HSL color scale. Click any token to see its rank and top-K alternative tokens as horizontal probability bars. Includes run history, keyboard navigation (arrow keys to move between tokens, Escape to deselect), and auto-scroll during streaming. Lazy-loaded at 13.8 kB.
- **Streaming logprobs callback**: Added `onLogprobs` to `StreamCallbacks` in `api/streaming.ts`. Extracts `logprobs.content` from SSE chunks and forwards to callback. Backward-compatible -- existing callers unaffected.
- **SparklesIcon**: Added to icon library (17 icons total).

## 1.3.0

### Added

- **Batch Processing applet** (`applets/batch/`): Second applet in the platform. Create batch jobs with multiple prompts, process via `/v1/batch/chat/completions` endpoint, view results in dashboard with per-prompt expandable cards and JSON export. Lazy-loaded for zero impact on chat page performance.
- **Shared SamplerControls component** (`components/composed/SamplerControls.tsx`): Reusable sampler parameter sliders (temperature, top_p, top_k, min_p, max_tokens, penalties, seed) extracted from SettingsPanel. Used by both chat settings and batch create form.
- **Shared ModelSelector component** (`components/composed/ModelSelector.tsx`): Moved from `features/models/` to shared layer. Now uses `onModelLoaded` callback prop instead of direct chatStore dependency, enabling reuse across applets.
- **Batch-related icons**: LayersIcon, BoltIcon, DownloadIcon added to icon library (16 icons total).
- **API schema module** (`src/heylook_llm/schema/`): New Anthropic Messages-inspired API schema with typed content blocks (TextBlock, ImageBlock, ThinkingBlock, LogprobsBlock, HiddenStatesBlock), structured streaming events, and bidirectional converters to/from the existing OpenAI-compatible format. Purely additive -- existing endpoints unchanged.
- **Pre-commit safety hook**: Rejects staged files in `internal/`, `coderef/`, `.claude/`, `.archive/`, `modelzoo/*`, `models.toml`, `.env`, and files containing personal filesystem paths.
- **Frontend platform documentation** (`internal/frontend/`): Applet catalog, platform architecture, API schema design, design system, and migration plan.
- **Frontend applet platform architecture**: AppShell with AppNav sidebar rail, react-router-dom routing with `/chat` and `/batch` routes, shared primitives (Slider, Toggle, EmptyState, Modal) and composed components (ThinkingBlock, SamplerControls, ModelSelector), icon component library. Chat restructured as first applet module under `applets/chat/` with own store.
- **Type generation pipeline**: openapi-typescript devDependency with `generate:api` script for auto-generating TypeScript types from the FastAPI OpenAPI spec.

### Changed

- **Dependency update**: Removed `huggingface-hub<1.0` upper bound pin to allow `transformers>=5.0` (required by latest mlx-lm from git)
- **Frontend architecture**: Chat moved from `features/chat/` to `applets/chat/`, chatStore moved to applet-owned `applets/chat/stores/`. Sidebar, ConfirmDeleteModal, AdvancedPanel now live in `applets/chat/components/`. Layout changed from `h-screen` to `h-full` (AppShell owns viewport). SettingsPanel refactored to use extracted SamplerControls.

### Removed

- Empty placeholder directories: `features/advanced/`, `features/batch/`, `features/settings/`, `components/common/`, `components/ui/`, `features/models/`

## 1.2.0

### Changed

- **Dependency modernization**: Added `[tool.uv.sources]` for mlx-lm, mlx-vlm, and parakeet-mlx to pull latest from git instead of PyPI
- **STT migration**: Replaced CoreML STT provider (coremltools + manual RNNT decoder) with MLX STT provider (parakeet-mlx high-level API)
- **Setup scripts**: Rewrote `setup.sh` and `update-heylook.sh` for uv-only workflow, removed all llama-cpp-python references (deprecated, replaced by llama-server subprocess)

### Removed

- `coreml_stt_provider.py` (465 lines) -- replaced by `mlx_stt_provider.py` (265 lines, parakeet-mlx)
- `CoreMLSTTModelConfig` and `coreml_stt` provider type from config
- `llama-cpp` from `[all]` extra (llama-server subprocess is the supported path)
- pip fallback in setup/update scripts (uv required for `[tool.uv.sources]` git resolution)

### Migration

- Users with `provider = "coreml_stt"` in models.toml must change to `provider = "mlx_stt"` and update model path to a HuggingFace repo ID (e.g., `mlx-community/parakeet-tdt-0.6b-v3`)
- `setup.sh` and `update-heylook.sh` now require uv (no pip fallback)

## 1.1.1

### Added

- **MLX Performance Optimizations**: Following mlx-lm reference patterns for better Metal utilization
  - Compiled presence penalty processor with `@mx.compile` decorator for Metal kernel optimization
  - Compiled vision preprocessing (normalize + transpose) for faster image encoding
  - Pre-computed ImageNet constants as module-level `mx.array` objects
  - Use `mx.broadcast_to` instead of list multiplication for memory-efficient grid creation
  - Replaced blocking `mx.eval` with `mx.async_eval` for better pipeline pipelining
  - Added buffer cache cleanup method with configurable retention
  - Release temporaries before sync points to reduce memory pressure

- **MLX Performance Test Suite**: Comprehensive tests for MLX optimizations
  - `tests/unit/mlx_perf/`: Compilation correctness, type consistency, sync boundary tests
  - `tests/integration/mlx_perf/`: Throughput benchmarks, TTFT tests, memory profiling
  - New pytest markers: `mlx_perf`, `slow` for selective test execution
  - Configurable test model path via `HEYLOOK_TEST_MODEL` env var

- **Structured Hidden States Endpoint**: New `/v1/hidden_states/structured` for server-side chat template
  - Accepts chat components separately (user_prompt, system_prompt, thinking_content, assistant_content)
  - Server applies Qwen3 chat template internally with `enable_thinking` support
  - Returns token boundaries for each section (system, user, think, assistant)
  - Returns token counts per section and total
  - Optional `formatted_prompt` field for debugging
  - Enables ablation studies and token attribution research for Z-Image

- **Model Capabilities Discovery**: Expose model capabilities in `/v1/models` response
  - New `capabilities` field in model config (e.g., `["hidden_states", "chat", "thinking", "vision"]`)
  - New `supports_thinking` and `thinking_token_ids` fields in MLXModelConfig
  - `/v1/models` now includes `provider` and `capabilities` when configured
  - Enables programmatic capability discovery for multi-model clients

- **Auto Model Selection**: Fallback to loaded model when no model specified in request
  - Uses most recently used model from LRU cache
  - Falls back to `default_model` from config if no models loaded
  - Provides clear error message with available models if no default

- **Token-Level Thinking Parser**: More efficient parsing of Qwen3 thinking blocks
  - New `TokenLevelThinkingParser` uses token IDs (151667/151668) for precise detection
  - New `HybridThinkingParser` auto-selects between token-level and text-based parsing
  - Eliminates regex buffering overhead when token IDs are available
  - Instant detection of `<think>`/`</think>` boundaries via special token IDs
  - Backwards compatible: falls back to text parsing for models without token IDs
  - Integrated into streaming response generator for automatic use

- **Hidden States Config Defaults**: Model-level defaults for hidden states extraction
  - New `default_hidden_layer` config option in MLXModelConfig (default: -2)
  - New `default_max_length` config option in MLXModelConfig (default: 512)
  - Hidden states endpoint now applies model config defaults when request uses defaults

- **Enhanced Streaming Metadata**: Detailed generation stats in final streaming chunk
  - New `stream_options: {include_usage: true}` parameter in ChatRequest
  - `usage` object with `prompt_tokens`, `completion_tokens`, `thinking_tokens`, `content_tokens`, `total_tokens`
  - `timing` object with `thinking_duration_ms`, `content_duration_ms`, `total_duration_ms`
  - `generation_config` object with sampler settings used (temperature, top_p, top_k, min_p, max_tokens, enable_thinking)
  - `stop_reason` field indicating why generation stopped: `eos_token`, `max_tokens`, `stop_sequence`, or `length`
  - Properly maps MLX `finish_reason` values to OpenAI-compatible stop reasons
  - Enables frontend display of thinking tokens, timing breakdown, and stop reason

- **OpenAI-Compatible Logprobs Support**: Return token log probabilities in chat completions
  - New `logprobs: bool` parameter to enable log probability output
  - New `top_logprobs: int` parameter to specify number of top alternatives (0-20)
  - Non-streaming responses include `choice.logprobs.content` array with per-token data
  - Streaming responses include `choice.logprobs` delta in each chunk
  - Each token entry includes: `token`, `token_id`, `logprob`, `bytes`, `top_logprobs[]`
  - Leverages mlx-lm's full vocabulary log-softmax for accurate probabilities
  - New `logprobs.py` module with `LogprobsCollector` and `StreamingLogprobsCollector`

### Fixed

- **TextOnlyStrategy model_config**: Fixed AttributeError when `enable_thinking` not in request
  - Added `model_config` parameter to `TextOnlyStrategy.__init__`
  - Changed `getattr()` to `.get()` for proper dict access
  - Fixes Qwen3 thinking mode fallback to model config defaults

- **Qwen3-VL-MOE Vision Support**: Fixed "Image features and image tokens do not match" error
  - Use `mlx_vlm.prompt_utils.apply_chat_template` for proper image token insertion
  - Removed manual image placeholder insertion that conflicted with library handling
  - Now properly passes `num_images` to prompt formatting

### Previously Added

- **Qwen3 Thinking Token Support**: Parse `<think>...</think>` blocks from Qwen3 model outputs
  - Non-streaming responses include `message.thinking` field with parsed reasoning content
  - Streaming responses emit `delta.thinking` during thinking, `delta.content` for response
  - Multiple thinking blocks are concatenated with `---` separators
  - New `thinking_parser.py` module for robust parsing
- **Model Configuration for Thinking Mode**: Add `enable_thinking` parameter to MLXModelConfig
  - When enabled, automatically applies Qwen3 optimal sampler defaults:
    - temperature=0.6 (greedy decoding causes repetition)
    - top_p=0.95
    - top_k=20
    - presence_penalty=1.5
  - Pass `enable_thinking` to chat template for Qwen3 tokenizers
- **Presence Penalty Support**: Add `presence_penalty` parameter to ChatRequest and MLXModelConfig
  - Discourages reuse of tokens that have already appeared
  - Recommended value 1.5 for Qwen3 thinking mode
  - Custom logits processor implementation for mlx-lm compatibility

### Changed

- Updated `models.toml.example` with thinking model configuration documentation
- Extended sampler builder to support presence_penalty processor
