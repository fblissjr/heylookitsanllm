# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.35.0]

### Added

- **batch-labeler rebuilt as a task-based CLI (app v0.2.0)**: subcommands
  `run` / `try` / `models` / `tasks` replace the single flag-soup invocation.
  Built-in task templates with curated system prompts (`label` structured
  taxonomy, `caption` training-style captions, `tags` keyword tags, `ocr`
  verbatim extraction), each carrying its own sampler preset
  (`vlm-extract`/`vlm-describe`), max_tokens, and required-key validation;
  custom tasks via `--task-file` TOML (unknown keys rejected). Now exposes
  the server capabilities the old client ignored: `--think/--no-think`
  (enable_thinking; thinking stored in its own record field), `--vision-tokens`
  (visual token budget), `--resize-max`/`--image-quality` (server-side
  resize), `--preset`, `--seed`. Model auto-pick when the server has exactly
  one vision model; transient-failure retries with backoff; `--limit` for
  sampling; `--dry-run` prints fully-resolved settings; records carry
  parse_ok/missing_keys, usage, performance, and a settings echo. Live tok/s
  in the progress bar. 37 new tests (62 total in the app suite).

### Fixed

- **`include_performance` on `/v1/chat/completions` was dead**: declared on
  ChatRequest but never read; `ChatCompletionResponse.performance` was never
  populated. Non-streaming responses now fill it from native mlx-lm telemetry
  (prompt_tps, generation_tps, peak_memory_gb) when requested. Contract test
  added; schema unchanged (field already existed as Optional).

## [1.34.66]

### Fixed

- **Review-pass fixes** (xhigh multi-agent review over the day's 13-commit
  range; findings adversarially verified before fixing):
  - `<think>`-family parser (HybridThinkingParser) now strips declared
    special tokens from routed text like every other parser -- with the
    decode-level hygiene gone, non-think control tokens could render
    literally for Qwen-family models; implemented with a rolling per-kind
    holdback because the inner parser's buffering can split a special
    across deltas.
  - Gemma channel parser: an abort landing mid-`<channel|>` no longer
    flushes literal partial-token garbage (final drain strips trailing
    partials of BOTH gemma control-token shapes; harmony's helper only
    knew `<|`-prefixed ones).
  - `effective_thinking_flag`'s absent-key fallback now mirrors the
    template-side resolution (True) so a raw un-normalized config dict
    can't produce a prefilled-thinking prompt with a content-state parser.
  - config API docstrings/OpenAPI descriptions no longer claim an
    env-override layer that settings.py deliberately abolished.
- **v3 touch/mobile fixes from the impeccable audit** (desktop + iPhone 17
  Pro Safari): attach-strip remove buttons get a 44px hit area (glyph
  stays 24px); composer icon buttons meet the 44px floor on touch
  devices; the composer placeholder no longer wraps and clips on phones
  (Enter hint is desktop-only).

## [1.34.65]

### Added

- **`tests/eval/` -- opt-in LLM behavior-eval harness** (13-task seed bank,
  7 programmatic judges), generalized from the 2026-07-20 live-verification
  scripts. Covers the four bug classes found that day: thinking split/leak,
  stop discipline, vision single/multi-image correctness, vision_tokens
  budgets. Needs a running server (never spawns one); not wired into
  /test-suite -- run it when touching templates, parsers, stop tokens, or
  the vision pipeline.

### Changed

- **Pyright noise triage.** Real fixes: deprecated `datetime.utcnow`;
  untyped `= None` defaults; batch responses could hand pydantic
  `model=None` (runtime 500) -- now coalesced; float re-binding in
  `_format_bytes`; route discovery duck-typed via getattr. Systemic:
  all 96 positional Pydantic `Field(default, ...)` calls converted to
  explicit `default=` (this pyright build only recognizes keyword form --
  the source of every false "arguments missing" constructor error).
  Idioms annotated per repo convention (`# type: ignore[...]`): request
  attr-attach, streaming-logprobs union, psutil optional import, RLM's
  provider-specific batch fast path. hasattr-ternary tokenizer access
  swapped for `getattr(x, "tokenizer", x)` (same runtime, narrows).

## [1.34.64]

### Added

- **Model-agnostic vision token budget** (`vision_tokens` on chat requests +
  per-model default in models.toml; v3 drawer control, cap-gated on
  `vision`): one wire knob mapped by duck-typing the loaded model's image
  processor -- gemma-4 discrete buckets (snapped: 70/140/280/560/1120 soft
  tokens ~ 0.16-2.58MP), qwen2/3-VL continuous pixel budget
  (tokens x (patch x merge)^2 -> max_pixels), unknown families degrade to
  processor defaults. Vision-feature cache key carries the budget (feature
  shapes differ per bucket). Verified live on gemma-4-31b + Qwen3.5-27B
  (pixel patches 630/2520/10080 at 70/280/1120; full think x images x
  budget matrix clean on both).

### Fixed

- **Qwen3.5 thinking never split into the thinking field**: its template
  PRE-FILLS `<think>\n` into the generation prompt when thinking is on, so
  the model output starts inside the block without an opening tag; the
  parser routed everything to content. New `prefills_thinking` template
  detection + `initial_thinking` parser state, armed per request from the
  effective thinking flag.
- **Removed hardcoded Qwen3 vocab ids from the thinking parser**
  (151667/151668): the token-level mode keyed on them silently failed to
  split for any other `<think>`-family vocabulary whenever token ids were
  present. Parser is text-based only now, same as the harmony/gemma
  channel parsers (tag strings are format grammar; token ids are
  per-model vocabulary).

## [1.34.63]

### Fixed

- **Gemma-4 on the mlx-vlm path generated past end-of-turn** (visible as
  answers that "keep going", thought-babble tails, repeated `<turn|>`):
  two stop-set gaps, both fixed data-driven from the model's own files.
  (1) Raw HF tokenizers don't absorb generation_config.json's eos list
  (gemma-4 declares [1, 106, 50] incl. `<turn|>`; the tokenizer said 1) --
  `extend_eos_from_generation_config` now unions it at load. (2) mlx-lm's
  `stream_generate` auto-wraps raw tokenizers with only the single
  `eos_token_id` -- `run_generation` now wraps raw tokenizers itself
  (`ensure_gen_tokenizer`) with the full resolved stop set.
- **Removed the decode-path special-token hygiene patch** (v1.34.5x
  `tokenizer_hygiene.py`): defaulting every `decode()` to
  `skip_special_tokens=True` stripped structural channel markers before
  the reasoning parser could see them (gemma-4 thinking could never split
  on the VLM path). The parsers themselves strip declared specials from
  ROUTED text (`strip_tokens`), which is the correct layer. The vision
  first-token decode also yields raw now (a leading `<|channel>` must
  reach the parser).
- Live-verified on gemma-4-31b (vision + text, think on/off, 1/2/large
  images): thinking splits into the thinking field everywhere, turns
  terminate, multi-image + thinking works. The previously reported
  multi-image + thinking gibberish did not reproduce on the fixed stack.

## [1.34.62]

### Fixed

- **Gemma-4 canonical template was rejected by the stop-token gate**, which
  silently disabled thinking parsing + capability sniffing (template_info
  emptied -> PassThroughParser -> thinking leaked inline as plain text with
  the channel markers stripped -- the "thought" first-line leak). Cause:
  `_read_eos_tokens` resolved eos ids only via tokenizer_config.json's
  `added_tokens_decoder`, which gemma-4 fast tokenizers don't have; the
  `<turn|>` terminator (eos id 106) lives in tokenizer.json's
  `added_tokens`. The id map now unions both files (same dual-source rule
  as `_read_special_tokens`). Generation itself was never affected
  (transformers loads the jinja directly); only the sniffing layer was
  blind. The existing v3 collapsible thinking block now receives the
  split thinking as designed.

## [1.34.61]

### Changed

- v3 chat composer: attach button is now an icon (was "+ Img"), and a
  thinking-toggle icon sits next to it for thinking-capable models (same
  capability gate + true/unset semantics as the drawer checkbox, kept in
  sync via onSettingsChange; pressed state styled off aria-pressed). New
  `.btn--icon` style (40px touch-target floor, currentColor SVGs).

## [1.34.60]

### Added

- **Gemma-4 thinking support, end to end.** The canonical gemma-4 template's
  `<|channel>thought ... <channel|>` reasoning format is now recognized:
  new `GemmaChannelParser` (streaming-safe state machine, `thought` channel
  -> `thinking` field, unknown channels -> content) selected via template
  sniffing (`has_gemma_channel_structure`); previously gemma thinking leaked
  inline into `content` on every path.
- **Thinking capability auto-detected from the template.** A chat template
  referencing `enable_thinking` (Qwen3 <think> blocks, gemma-4 thought
  channels -- transformers forwards extra template kwargs as variables) now
  reports the `thinking` capability on /v1/models without a manual
  models.toml flag -- the v3 thinking checkbox appears for these models.
- v3 chat: multi-image attach hardening -- cap at 8 with an aria-live
  announcement when exceeded, per-image accessible remove labels ("Remove
  image N"). (The strip, picker `multiple`, paste-append, N-block store/wire
  shipped in v1.34.20; this closes the cap + a11y gaps.)

### Fixed

- **VLM template path dropped `enable_thinking` entirely** (both the
  text-only-request leg and the image leg via `prepare_vlm_inputs_parallel`):
  the flag never reached `apply_chat_template`, so thinking was
  uncontrollable on VLM-loaded models (mlx-lm's TokenizerWrapper silently
  injects `true` when absent). Both legs now forward the resolved bool
  (request > model config default).
- **Messages API used a hardcoded `<think>`-only parser** in streaming AND
  non-streaming; both now route through `select_reasoning_parser` like
  chat/completions (harmony/gemma/think formats all split correctly).

## [1.34.59]

### Added

- **`mlx_cache_limit_gb` operational setting** (`/v1/admin/config`, default
  unset): opt-in cap on MLX's buffer cache. The allocator keeps freed buffers
  for reuse and never returns them to the OS, so server RSS pins at the
  prompt-spike high-water mark; the cap bounds idle RSS (useful when other
  memory-hungry jobs, e.g. lens fitting, share the box). Clearing the override
  restores MLX's own default, captured from the first apply.

### Fixed

- `DELETE /v1/admin/config/{key}` now re-applies settings immediately like PUT
  does; previously a reset only took effect after a restart while GET already
  reported the default as effective.

## [1.34.58]

### Changed

- **Per-document sampler settings unified across chat AND notebook** (no branched
  copies of the same wiring):
  - Backend: `notebooks` gain a `params` JSON column (like conversations);
    `_SCHEMA_VERSION` 4 -> 5. Conversations + notebooks share ONE encode/decode
    pair (`_encode_params`/`_decode_params` in db.py). Threaded through notebook
    create/update + `Notebook{Create,Update}` + `PUT /v1/notebooks/{id}`. 3 tests.
  - Frontend: one shared `bindDocumentParams({activeId, updateDoc, onError})` +
    `hydrateDocParams(doc)` in settings.js. Both chat.js and notebook.js call them
    (chat's bespoke `saveConversationParams` copy removed) -- sampler knobs bind to
    the active conversation/notebook's `params`, hydrate silently on select,
    debounce-PUT on change, and carry forward on create.

## [1.34.57]

### Changed

- **Sampler settings are now per-conversation** (v3, settings-storage unification
  frontend): the drawer's sampler knobs bind to the active conversation's `params`
  (server), mirroring the per-conversation system-prompt editor -- no longer
  browser-only global state. On conversation select the panel hydrates from
  `conversation.params` (silent, no re-PUT); a knob change / preset apply
  debounce-PUTs `{params}` to the conversation; new chats + first-send create with
  the current panel (`snapshotSettings()`), so knobs carry forward. `settings.js`
  gains `onSettingsChange` (mirrors `onDisplayChange`) + a `silent` hydrate option
  on `applySettings`. localStorage is now just the new-chat seed. Resolves the
  "some settings in browser, some on server" split (redesign note §3b).

## [1.34.56]

### Added

- **Per-conversation sampler settings** (settings-storage unification, backend):
  conversations gain a `params` JSON column (temperature, top_p, ... -- next to
  `system_prompt`, so per-conversation tuning lives WITH the conversation on the
  server instead of split into browser localStorage). Threaded through
  `create_conversation`/`update_conversation` + `Conversation{Create,Update}` +
  `PUT /v1/conversations/{id}`. `_SCHEMA_VERSION` 3 -> 4 (drops tables per the
  solo-deploy policy -- accepted). Frontend wiring (hydrate the settings drawer
  from `conversation.params`, PUT on change) is the v3 lane. 5 tests.

## [1.34.55]

### Fixed

Chat-template robustness -- a broken/corrupted `chat_template.jinja` no longer
causes silent runaway generation (owner hit this after a re-import force-installed
a gemma jinja that emitted `<|turn>model` with no `<end_of_turn>`, so the model
generated to the max_tokens cap):

- **Load-time self-heal** (`read_template_info`): a resolved template that renders
  none of the model's OWN stop tokens is rejected; resolution walks the remaining
  file sources for a valid one, and if none, installs nothing so the loader's
  built-in template stands (rescues already-broken configs without a re-import).
- **Import-time guard** (`detect_chat_template_source`): a stop-less
  `chat_template.jinja` is no longer recorded as `chat_template_source = jinja`.
- **Non-hardcoded** stop-token detection: `_read_eos_tokens` reads the model's
  declared `eos_token`/`eos_token_id` (resolved via `added_tokens_decoder`, across
  `tokenizer_config.json` + `generation_config.json`) -- we validate against the
  model's real stop set, never a hardcoded marker list. Conservative: an empty
  template, or a model whose stop set can't be determined, is never rejected.
- 6 tests.

## [1.34.54]

### Fixed

- **Aborted / manually-stopped streaming requests are now logged** (owner report):
  on client disconnect, `GeneratorExit`/`CancelledError` is thrown into the
  streaming generator mid-yield and unwinds past the normal finalizer, so a
  stopped request produced NO `request_complete` and left a silent gap. A new
  `except (GeneratorExit, CancelledError)` on the stream loop emits a partial
  `request_complete` (`success=false`, `stop_reason="abort"`, partial
  `completion_tokens` + elapsed) then re-raises. Sync-only in the handler (no
  await/yield during unwind). `provider` type derivation extracted to a shared
  `_provider_type` helper.

## [1.34.53]

### Fixed

/code-review pass on the observability + config work:

- **record_event field collisions** (correctness): the ingestion API took
  `**fields`, so a diag caller or v3 client sending a key named `source`/`tier`/
  `min_level`/`type` would either raise `TypeError` (500 + dropped batch on the
  telemetry endpoint; broke `diag_event`'s best-effort on the request path) or
  silently override the record's own `type`/`source`. `record_event` now takes an
  explicit `fields=<dict>` (reserved record keys spread last, always win); all
  call sites updated.
- **Rotation clobber**: two same-second rolls overwrote the earlier archive
  (`rename` is silent on POSIX). Rotation now picks a non-existing archive name.
- **Config stored raw value**: `PUT /v1/admin/config` persisted the request value,
  not the Pydantic-coerced one (`stored` could be `"30"` vs effective `30`); now
  stores the coerced value.

### Documentation

- CLAUDE.md, README, and `docs/observability_guide.md` updated for the spine
  (`logs/metrics.jsonl` + `logs/events.jsonl`), `observability_level` as the single
  control, the `logs/` location, and the accurate content model (metrics tier
  content-free; events tier may carry bounded error text -- `minimal` is NOT
  "content-free").

## [1.34.52]

### Changed

- **`observability_level` is now the master telemetry kill switch** (unifying the
  control, owner ask): setting it to `off` silences memory.py's legacy streams
  (`request_events`/`model_events`/`memory_baseline`) too, not just the spine --
  one place to turn ALL telemetry off. Additive gate; the legacy per-stream env
  toggles still work for granular control. (Full retirement of those toggles +
  removing the streams the spine now duplicates is deferred until the spine is
  fully live-verified -- deleting proven streams for an unverified replacement is
  the risk the live check just caught with the provider bug.)
- Tests: an autouse fixture resets the observability global level per test
  (deterministic; the level is a mutated module global).

## [1.34.51]

### Fixed

- Telemetry `provider` field was always null (found via live verification): the
  per-request emission + memory.py's `provider_type` read `getattr(provider,
  "provider")` on the provider OBJECT, which has no such attribute. Now derived
  from the provider class (`MLXEmbeddingProvider` -> `mlx_embedding`, else `mlx`).
  Live run confirmed the text/mlx-lm path: `request_complete` with
  `effective_loader=mlx-lm`, `is_vlm=false`, and real token/tps/memory metrics.

## [1.34.50]

### Changed

- **Operational settings are DB-authoritative -- the env-override layer is removed**
  (owner feedback: env silently overriding a value set in the admin UI is a
  footgun). `observability_level`/`retention` now resolve DB > default only; there
  is no `HEYLOOK_OBSERVABILITY_LEVEL`. Env vars are reserved for bootstrap paths
  that have no UI counterpart and thus can't conflict (`HEYLOOK_LOGS_DIR`,
  `HEYLOOK_DB_PATH`). The `/v1/admin/config` response drops `env_overrides`.

## [1.34.49]

### Added

- `POST /v1/telemetry/events` (redesign Phase 3, backend): frontend telemetry
  ingestion. v3's client logger batches events (JS errors, fetch failures, stream
  stalls) and posts them here; each is appended to the observability events stream
  with `source=frontend-v3`, level-gated, batch- and field-size-bounded. Metadata
  only. 5 tests.

## [1.34.48]

### Changed

Error/event consolidation (redesign Phase 2) -- one writer, one schema for
`logs/events.jsonl`:

- `diagnostic_logger.diag_event` now **delegates to `observability.record_event`**
  (events tier) instead of owning its own file writer + rotation. All api.py/
  router.py call sites are unchanged. Consequences: diag fields are **flattened**
  onto the record (queryable top-level keys) instead of nested under `data`;
  the diag `level` (severity) is carried as a field and mapped to the spine's
  verbosity gate (errors/warnings surface at `minimal`, info at `standard`); and
  events are now level-gated + rotated like the rest of the spine. `exception_detail`
  is unchanged. `HEYLOOK_DIAG_LOG` is retired (use `HEYLOOK_LOGS_DIR`).
- Tests: `HEYLOOK_LOGS_DIR` is isolated to a temp dir in the root conftest, so the
  suite no longer writes telemetry into the repo's `logs/` (also fixes a
  pre-existing `events.jsonl` pollution).

## [1.34.47]

### Added

- Per-request metrics emission (redesign Phase 1): completed requests now emit a
  content-free `request_complete` line to `logs/metrics.jsonl` (tokens, tps, ttft,
  timings, peak memory, kv bytes, cache, stop reason, image count), plus the
  frozen §4.3 registry dims (`provider`/`effective_loader`/`is_vlm`) read null-safely
  via `getattr` -- an embedding provider yields null, not a crash. This is the
  highest-value telemetry the spine carries. 3 tests.

## [1.34.46]

### Added

Observability spine wired live (redesign Phase 1, slices 2-5):

- **Startup wiring + disclosure**: the spine is `configure()`d from the settings
  layer at boot (level/retention resolved env > DB > default), and a startup log
  line discloses what's written and that it's local ("nothing transmitted").
  `/v1/admin/config` PUT refreshes the in-process cache so a level change takes
  effect immediately.
- **Env hardening**: `resolve_settings_safe` never raises -- a bad `HEYLOOK_*`
  value falls back to defaults + a warning (and is surfaced in the config API
  response) instead of crashing startup.
- **File rotation**: `logs/*.jsonl` streams roll past a size cap to timestamped
  archives; archives older than the retention window (default 30d) are swept
  hourly on the maintenance tick. Best-effort, never raises.
- **`internal/log/` reconcile**: `memory.py`'s telemetry streams now write under
  `logs/` (runtime data), not `internal/log/` (human session diaries). Overridable
  via `HEYLOOK_LOGS_DIR`.
- **First real emitters**: model load/unload/evict now emit `record_event` (events
  tier) alongside the existing diagnostics.

### Changed

- README monitoring paths updated to `logs/`.

## [1.34.45]

### Added

Observability spine core (redesign Phase 1, slice 1) -- the single JSONL
ingestion path. Internal foundation; wiring + emission from real call sites and
the `internal/log/` reconcile follow in subsequent slices:

- `observability.record_event(type, *, tier, min_level, source, **fields)` --
  appends one JSON line (with `ts` + local-time `iso`) to the right stream
  (`logs/metrics.jsonl` content-free, `logs/events.jsonl` correlated), gated by
  the configured verbosity (`off < minimal < standard < debug`). **Best-effort:
  never raises** (observability must not break inference). Level + log dir are
  cached in-process (`configure()`), so the hot path never touches the DB.
- 8 tests (write shape, tier routing, level gating, never-raises).

## [1.34.44]

### Added

Config foundation (observability + config redesign, Phase 0) -- the runtime-mutable
operational-settings layer that the JSONL observability spine will build on:

- App DB `settings` table (key -> JSON value), added additively alongside presets.
  Schema-stable (a new setting is a new row, never DDL), so it survives the
  drop/recreate schema policy without a drop-list carve-out; treated as config,
  not data. Store CRUD in `db.py` (`get_setting`/`get_all_settings`/`set_setting`/
  `delete_setting`).
- `settings.py`: the `SettingsSchema` Pydantic contract (types, defaults,
  `extra="forbid"`) + `resolve_settings()` with **env > DB > default** precedence
  (`HEYLOOK_<FIELD>` overrides stay the always-wins escape hatch). First fields:
  `observability_level` (off/minimal/standard/debug, default minimal) and
  `observability_retention_days` (default 30) -- consumed by the Phase 1 spine.
- `/v1/admin/config` router (GET effective+stored+env-overrides, PUT validated
  updates, DELETE resets a key) -- the backend for the v3 admin/settings config
  panel. 422 on unknown key / invalid value before anything persists.
- 23 tests (store, resolver precedence, HTTP contract); no new database (reuses
  the App DB), no registry change.

## [1.34.43]

### Added

- Model registry now describes modality and engine routing as two separate
  fields on `MLXModelConfig`, decoupling what a model IS from how it loads
  (Phase 6 refinement; see `docs/project/plan_2026-07.md`):
  - `modalities: list[str]` -- author-declared capability set
    (`text`/`vision`/`audio`/`video`), detected at import from the config's own
    `vision_config`/`audio_config` blocks + `*_token_id`/`image_token_index` keys
    (`model_importer.detect_modalities`), with `mmproj`-style files as a
    fallback. Represents genuinely multi-modal models (e.g. gemma-4 declares
    text+vision+audio) that a single `vision` bool could not. Validated against a
    19-model modelzoo audit (LLaVA/Mistral use `image_token_index`, not `_id`).
  - `loader: "auto" | "mlx-vlm" | "mlx-lm"` -- engine routing within
    `provider="mlx"`. `auto` picks mlx-vlm when the model declares vision AND
    mlx-vlm registers its `model_type`, else mlx-lm; it degrades to mlx-lm only
    on POSITIVE knowledge that mlx-vlm lacks the type (uncertainty keeps the
    historical vision->mlx-vlm default). An explicit value forces the engine
    (e.g. run a dual-capable VLM as text via `mlx-lm`). Resolution lives in
    `providers/common/loader_routing.py`; `is_vlm` + a new
    `MLXProvider.effective_loader` derive from it.
- `/v1/models` entries now carry `modalities` (full description); `capabilities`
  stays gated to what the server actually serves (image input) -- description !=
  served.

### Changed

- `vision: bool` (MLXModelConfig) is demoted to a derived mirror of
  `"vision" in modalities` (kept for back-compat readers of `config["vision"]`);
  `modalities` is authoritative. Absent `modalities` derives from `vision`, so
  existing `models.toml` entries and the provider load path are unchanged. The
  richer modality set (e.g. audio) lands on re-import.

## [1.34.42]

### Fixed

- jspace `/v1/jspace/analyze` crashed (`AttributeError: 'NoneType' object has no
  attribute 'offset'`) on hybrid mlx-vlm models -- specifically Qwen3.5 (the
  KVCache+ArraysCache GDN architecture). Their full-attention block dereferences
  `cache.offset` with no None-guard, so the cache-less inner forward the lens
  used for read-out (`ModelAdapter.logits` / `capture_residuals`) blew up. gemma
  was unaffected (its attention tolerates a missing cache). The adapter now
  sources a fresh, empty per-layer cache from the model's own `make_cache()`
  (length-matched to the block count) and passes it into every inner forward;
  each analyze forward re-prefills the whole sequence, so a throwaway offset-0
  cache reproduces the old no-cache semantics. Models without a matching
  `make_cache` still run cache-less, unchanged.

## [1.34.41]

### Changed

Diagnostic log (`logs/events.jsonl`) now actually explains errors:

- Every event carries a human-readable `iso` field (local time with UTC offset)
  alongside the epoch `ts` -- `ts` stays authoritative for sorting/latency math,
  `iso` makes the file legible without converting epoch seconds by hand.
- `request_error` events now record `error_type` (the exception class), `stage`
  (where in setup it failed: routing / provider_get / capacity_check /
  generator_create / streaming), `model`, and -- for wrapped errors -- a `chain`
  of the underlying causes. Previously the record held only `str(e)`, so the
  actual "why" (exception type, root cause) only reached the console logger.
- Mid-stream generation failures are now logged. The streaming `GenerationFailed`
  path previously wrote nothing to `events.jsonl`, leaving a `generation_start`
  with no matching completion; a new catch-all also logs unexpected mid-stream
  errors and closes the SSE stream cleanly (in-band error payload + `[DONE]`)
  instead of propagating a raw exception into an already-started response.
- Cause chains are captured via `traceback.format_exception_only` (type +
  message only, never frame locals), so prompt/response text cannot leak.

## [1.34.40]

### Fixed

Chat-template hardening batch (the quick-fix findings from the v1.34.38 review):

- Server-side import (`/v1/admin/models/import`) and the CLI wizard now share one
  detection helper (`template_info.detect_chat_template_source`) -- the two inline
  copies had already drifted -- and tilde/relative model paths no longer silently
  skip jinja detection (`expanduser`).
- `chat_template_source = "auto"` no longer takes the force-install branch (it is
  documented as fill-only-when-missing; force could clobber a natively-loaded dict
  of named templates). `"chat_template_json"` is now an accepted explicit value --
  it was a resolved-source label users could see in load logs but not configure.
- The missing-template error is decided from tokenizer state
  (`chat_template`/`has_chat_template`) instead of string-matching transformers'
  error prose (version-fragile), mentions all three supported sources, and now
  covers all three apply sites (chat, batch, hidden-states) instead of one.
  mlx-lm wrapper-level python templates (`chat_template_type`, e.g. DeepSeek-V3.2
  conversions) are recognized, so their render errors are no longer mislabeled.
- The load-time "NO chat template" warning consumes `install_chat_template()`'s
  result and checks the wrapper's `has_chat_template`, fixing both a missed-warning
  case (resolved template whose install failed) and a false-alarm case
  (`chat_template_type` models that render fine with a None tokenizer attr).

## [1.34.39]

### Fixed

- **`heatmap_top_k` is now schema-bounded (0-64)** on `POST /v1/jspace/analyze`: previously any
  int was accepted and clamped only to vocab size, letting one request decode band x positions x
  vocab tokens (a multi-GB response) while holding the process-global generation gate.
- v3 jspace interaction fixes from the review pass: arrow-key pin walking now respects the
  layer-range scope (it could land the pin on a hidden row); `scrollIntoView` on the pinned cell
  now fires only for keyboard navigation (strip clicks no longer yank the viewport to the
  heatmap's far-right onset column); non-answer logit bars no longer show a literal "undefined"
  tooltip; the echo highlight accepts the empty-string token (rendered as the empty-token glyph);
  slider drags dedupe unchanged ranges and route the aggregation repaint through the frame
  throttle.

### Tests

- Contract: out-of-range `heatmap_top_k` rejected (422). E2E pages suite grows to 35 checks:
  arrow-walk-respects-scope assertions and a heatmap-off analyze check (strip-only render,
  onset_strip aggregation fallback, strip-row pin) -- the path the review found uncovered.

## [1.34.38]

### Added

- **Chat-template resolution hardened as a registry concern.** The server-side
  scan+import route (`/v1/admin/models/import`, what the v3 models page uses) now
  applies the same `chat_template.jinja` auto-detection as the CLI import wizard,
  recording `chat_template_source = "jinja"` on the imported entry (request
  overrides still win). Template resolution (`template_info.py`) gains a last
  auto-fallback to processor-side `chat_template.json` -- previously a model
  shipping ONLY that file looked template-less at the tokenizer level. In auto
  mode the provider now installs the resolved template onto the tokenizer when
  the tokenizer loaded none (explicit `chat_template_source` still force-installs);
  the install logic is consolidated into a tested `install_chat_template()` helper.
  The v3 models page shows `template: <source>` in each model's meta line when set.

### Fixed

- A model folder with no chat template anywhere (no `chat_template.jinja`, no
  embedded `tokenizer_config.json` template, no `chat_template.json`) previously
  surfaced transformers' raw `ValueError` as the HTTP 500 detail on the first chat
  request. The provider now warns at load time and raises an actionable error
  naming the model and the fix (add a template file or set `chat_template_source`).

## [1.34.37]

### Added

- **Per-cell top-N analyze extension** (`POST /v1/jspace/analyze`): new `heatmap_top_k`
  (default 0 = unchanged) makes each heatmap cell carry `top_k: [{token, logit}...]`
  (reduced on-device via argpartition -- the full per-layer logits never leave the GPU
  path). The v3 jspace page sends it with the heatmap toggle, so pinning ANY cell now
  shows its full silent-token readout (spec section 4 updated in the same commit).
- **J-space visualizer sequence item 2 -- layer-range slider + aggregation** (v3, pure
  client-side): a slot-per-band-layer slider (click = one layer, drag = a contiguous
  range, hover = live preview, reset) scopes the strip and heatmap rows; the detail
  panel's unpinned state becomes a most-common-silent-tokens aggregation (top-k
  appearance counts over the scoped layers, heatmap-wide when per-cell top-k data
  exists), and clicking an aggregation row echo-highlights where that token wins in
  the grid.
- **Lens provenance surfaced**: `GET /v1/jspace/models` now returns
  `meta: {model: {provisional, fit_date, fit_source, n_prompts}}` from the lens
  sidecar (`provisional` = no own-fit `hf_model_name` stamp); the jspace page shows a
  "provisional lens" badge for such models. Pairs with the fitting-track change that
  stamps own-fit sidecars with provenance.

### Tests

- analyze() pipeline now unit-tested end-to-end on a tiny random-weight gpt2 (per-cell
  top-k shape/ordering + back-compat cell shape); registry provenance unit test;
  contract tests for heatmap_top_k forwarding and models-meta. E2E pages suite grows
  to 34 checks (aggregation panel, slider scoping/reset; the non-onset pin check now
  expects full bars).

## [1.34.36]

### Added

- **v3 design language seeded** (`apps/heylook-frontend-v3/DESIGN.md`, plan Phase 4 item 2):
  formalizes the token roles and the OKLCH data-strength chip formula that lived only in
  `css/app.css` comments, defines the selection/pin grammar, and records the j-space
  visualizer paradigm decision (matrix-first; Neuronpedia-style layer-range slider +
  aggregation sidebar as the growth path).
- **J-space visualizer sequence item 1 -- click-to-pin readout** (v3 `jspace` page, no
  backend change): workspace strip rows and heatmap cells pin a per-(layer, position)
  detail panel. Answer-onset pins (the strip; the heatmap's last column) show the full
  top-k silent tokens as logit bars with the first-answer-token emphasized; other cells
  show top-1 + entropy with a note that the per-cell top-N analyze extension (scoped in
  TODO.md) unlocks them. Esc unpins, arrow keys walk layers/positions, cells sharing the
  pinned cell's top token get an echo highlight, and the heatmap gains a prompt-token
  header row with an answer-onset column marker.

### Tests

- E2E pages suite extends the lens-gated jspace block to 32 checks total (heatmap render +
  onset marker, row pin, Escape unpin, non-onset cell pin/unpin); 32/32 live-green.

## [1.34.35]

### Changed

- `scripts/jspace_convert_lens.py` tolerates `--out-dir` already ending in the model id
  (avoids the double-nested `adapters/jspace/<id>/<id>/` that the registry never sees).

### Docs

- README, CLAUDE.md, and `docs/architecture/api.md` document the j-space feature + endpoints;
  CLAUDE.md gains the MLX off-event-loop / thread-stream, lazy-`mx.load`, and pipeline-`.layers`
  gotchas.

## [1.34.34]

### Fixed

- **`/v1/jspace/analyze` crashed the process from the frontend** (`There is no Stream(cpu, 0) in
  current thread`). Two causes, both fixed: (1) it ran the MLX forwards on starlette's ephemeral
  `run_in_threadpool` worker, which has no thread-local MLX stream (and a dying MLX thread aborts
  the process). Analyze now runs on a **pinned `mlx-stream` executor** (`streaming_utils._executor_pool`,
  same as generation) inside `mx.stream(generation_stream)`. (2) The lens `J` matrices are
  `mx.load`'d lazily (mmap-backed); their first eval landed on the worker thread and dispatched to
  the CPU default stream that thread lacks. `JSpaceLens` now force-evaluates them at load time
  (on the loading thread). Verified end-to-end on the served 26B MoE via a worker thread.

### Changed

- **Convert helper is easier to run** (`scripts/jspace_convert_lens.py`): carries PEP 723 inline
  deps, so `uv run scripts/jspace_convert_lens.py …` provisions torch+jlens itself (a clear error
  points there if run via the torch-less server venv); accepts a lens *directory* (finds the single
  `*_jacobian_lens.pt`, e.g. a neuronpedia model dir); rejects a path-shaped `--model-id`.

## [1.34.33]

### Fixed

J-space code-review fixes (xhigh review of the feature branch):

- **Capture is pipeline-safe (#1).** `ModelAdapter` now mutates the underlying block
  list on the text decoder (`inner.layers`/`.h`), not the top `model.layers` property,
  which for pipeline-parallel models (Qwen3.5, deepseek, glm4_moe) returns a FRESH slice
  each access — so capture was silently recording nothing → KeyError 500. `capture_residuals`
  now also raises a clear error if a forward never hits the recorders (instead of a silent
  empty read-out).
- **Analyze respects the concurrency invariants (#2/#3/#4).** `/v1/jspace/analyze` now pins the
  model (no LRU-evict / idle-unload mid-analyze) and runs the forwards under the process-global
  FIFO generation gate, so it serializes with generation and other analyze calls — no concurrent
  Metal command buffers (the documented crash class) and no racing mutation of the shared block list.
- **Unguarded 500s (#5/#6/#7).** Lens/normalizer/router load inside the error handler; `has()`
  requires the sidecar too (a partial convert no longer 404-passes then 500s); `router()` picks an
  available variant instead of KeyError-ing on a missing 'combined'; `format_prompt` accepts
  OpenAI content-block (list) messages instead of TypeError-ing on the default path.
- **Feature correctness (#8/#9/#10).** Reuse `resolve_stop_tokens` (honors plural `eos_token_ids`);
  `greedy_generate` returns the real first token even when it's a stop token (no redundant fallback
  forward); `workspace_readout` tolerates empty hedge sets and out-of-vocab ids.
- **E2E (#11).** The jspace mount check waits for the `/v1/jspace/models` fetch to resolve before
  asserting (was reading the select before it populated).

### Changed

- J-space efficiency + robustness: heatmap reduces to (top-1, entropy) on-device instead of
  materializing full-vocab float64 host arrays; lens/model `d_model` mismatch raises a clear error;
  `LensRegistry.from_env` falls back to `<cwd>/adapters/jspace` for non-editable installs; the
  convert script serializes the sidecar before the safetensors; the risk badge uses `Number.isFinite`.

## [1.34.32]

### Added

- **Lens convert+register helper + `adapters/` store.** `scripts/jspace_convert_lens.py` (git-tracked;
  torch+jlens, separate env) downloads/loads a jlens `.pt` and writes
  `adapters/jspace/<model_id>/lens.safetensors` + sidecar. `adapters/` is git-tracked via `.gitkeep`
  with gitignored contents (mirrors `modelzoo/`); `LensRegistry.from_env` now defaults there (repo
  root), so a converted lens is served with zero config (`HEYLOOK_JSPACE_DIR` still overrides).
- **E2E coverage for the J-Space page** (`tests/e2e/suites/pages.mjs`, lens-gated): mounts the page
  and, when a lens for the E2E model is installed, drives Analyze and asserts the workspace strip
  renders. Cuts the 26B-reload cost of future iteration.

### Docs

- **`docs/jspace_guide.md`** -- how-it-works + end-to-end tutorial (install a lens, call
  `/v1/jspace/analyze`, use the v3 J-Space page, interpret the output, enable risk). Indexed in
  `docs/README.md` + CLAUDE.md.

## [1.34.31]

### Added

- **v3 `J-Space` page** (`apps/heylook-frontend-v3/js/pages/jspace.js`, new `jspace` nav route).
  Model picker (lens-gated), prompt, `raw`/`chat` + heatmap toggles; renders the layer×top-k
  "silent words" strip (colored by within-layer rank), an optional layer×position heatmap (colored
  by confidence), and a hallucination-risk badge. Reuses the explore-chip OKLCH formula.

### Changed

- **`/v1/jspace/analyze` defaults to raw-completion prompting** (`chat=false`). The chat template's
  final position is the generation-prompt boundary, where the lens top-k is formatting junk; a raw
  completion reads a real content token so the workspace surfaces sensible tokens (verified on the
  served 26B MoE: "...the city of" -> Paris). `chat=true` (chat template) is kept for the risk
  features. Answer decode now strips special tokens.

## [1.34.30]

### Added

- **`/v1/jspace` interpretability API** (`jspace_api.py`, tag `JSpace`). `GET /v1/jspace/models`
  lists served models with a fitted lens; `POST /v1/jspace/analyze` formats the prompt exactly as
  the provider does (chat template + `<bos>`), greedily generates a short answer, captures the
  residual stream, and returns the Jacobian-lens workspace: per-band-layer top-k "silent" tokens
  at the answer-onset, an optional layer x position heatmap, workspace features, and (when a
  per-model normalizer + router are configured) a hallucination-risk score. New
  `jspace/analyze.py` (pipeline) + `jspace/registry.py` (`HEYLOOK_JSPACE_DIR/<model_id>/` lens
  cache; offline-converted safetensors). Lenses are loaded, never converted, at runtime.
  Registry unit tests + endpoint contract tests (routing/guards, no model needed).

## [1.34.29]

### Fixed

- **J-space `_Recorder` proxies attribute access to the wrapped block** (`capture.py`). The
  gemma-4 text forward reads `layer.layer_type` during mask construction, so the temporary
  capture wrapper must delegate attributes -- without this, residual capture on the served
  gemma-4 VLM raised `AttributeError`. Verified end-to-end on `gemma-4-26b-a4b-it-8bit-mlx`:
  the late-band workspace surfaces the correct entity (e.g. "Eiffel Tower ... city of" -> Paris),
  confirming the MoE capture point (through 128-expert routing) and 8-bit-lens transfer. NOTE:
  gemma requires an explicit `<bos>` (id 2) or the residual stream degrades to garbage.

### Changed

- **Bumped mlx-lm / mlx-vlm to latest upstream commits** (mlx-lm `a790972`: quantized-SDPA GQA
  batched-padding crash fix; mlx-vlm `05440cc5`). Same versions (0.31.3 / 0.6.5), newer commits.

## [1.34.28]

### Changed

- **J-space `ModelAdapter` now resolves multimodal-wrapper nesting** (`capture.py`). Walks
  `model` -> `.model` / `.language_model` -> `.model` to find the text decoder, softcap, and
  tied/untied head -- so it handles the served gemma-4 VLM (text stack under
  `model.language_model.model`, `final_logit_softcapping` on `language_model`) as well as the
  flat mlx-lm gpt2/gemma layouts. New unit test covers the nested-VLM resolution.

## [1.34.27]

### Added

- **J-space workspace features + hallucination-risk router** `src/heylook_llm/jspace/features.py`.
  Reproduces solarkyle/jspace's feature math: `workspace_readout` (per-band-layer ignition /
  rank / entropy / hedge-rank from lens logits), `router_feature_vector` (the 10 named features),
  `baseline_features` (output-confidence), and `HallucinationRouter` + `FeatureNormalizer`
  (per-model z-scored logistic regression predicting P(answer wrong)). Verified against the
  shipped e4b TriviaQA trace: AUC 0.795 workspace-only / 0.815 combined, both beating the
  first-token-logprob baseline (0.771) -- the paper's "workspace beats/adds to output confidence"
  result. Download-free unit tests in `tests/unit/test_jspace_features.py`.

## [1.34.26]

### Added

- **J-space (Jacobian lens) backend core module** `src/heylook_llm/jspace/` -- a post-hoc
  interpretability read-out of a model's verbalizable "workspace" (per-layer, which vocabulary
  tokens a residual is disposed toward), from Anthropic's July-2026 global-workspace work.
  `capture.py` (`ModelAdapter` architecture-introspection + `capture_residuals` via a temporary
  block wrapper) and `lens.py` (`JSpaceLens`: load converted safetensors + sidecar, transport,
  apply through the model's real head so gemma soft-cap / tied embeddings stay correct).
  Download-free unit tests in `tests/unit/test_jspace.py`; apply-parity verified cos ~1.0 vs the
  genuine reference `jlens` on gpt2 (no softcap) and gemma-2-2b (RMSNorm + softcap 30). Not yet
  wired to an endpoint -- see [docs/jspace_integration_plan.md](docs/jspace_integration_plan.md).

### Docs

- Promoted the j-space build + verifier plan into git-tracked `docs/jspace_integration_plan.md`
  (indexed in `docs/README.md` + CLAUDE.md Orient).

## [1.34.25]

### Removed

- **Legacy React frontend `apps/heylook-frontend/` deleted.** v3 (served at `/v3`) has
  parity and the app was no longer mounted by the backend. Retired with it: the OpenAPI
  drift guard that served only the legacy app -- `scripts/check_openapi_sync.sh`, the
  pre-commit OpenAPI block, and the `/openapi-regen` skill (v3 hand-writes `js/api.js`;
  nothing consumes the generated TS types). The live schema stays at `/openapi.json` +
  `/docs`. Also deleted the stale `docs/frontend_api_reference.md` (React-era integration
  guide, superseded by the v3 spec §4 + live OpenAPI).

### Docs

- CLAUDE.md, README.md, tests/README.md, docs/frontend_v3_spec.md, .gitignore, and a
  `perf_collector` docstring updated to drop references to the removed app/guard and to
  point at v3 (spec §4, `tests/e2e/`). New `internal/frontend/v3.md` (gitignored) maps
  v3's done/left + the backend<->v3 coupling; CLAUDE.md Orient now names
  `docs/project/plan_2026-07.md` as the roadmap.

## [1.34.24]

### Fixed

- **`/code-review` pass over v1.34.22-.23** (8 finder angles; 10 findings reported, 5 finders
  independently converged on the top two): the background preset-refresh repaint no longer fires when
  the list is unchanged or while the cursor is inside the panel (it destroyed uncommitted text on
  every open); `presets` removed from the schema-recreate drop list (a `_SCHEMA_VERSION` bump would
  have wiped saved presets despite the config-not-data promise — regression-tested); preset `params`
  that orjson can't serialize (>64-bit ints) now 400 instead of 500; `savePreset` decides
  create-vs-overwrite against a freshly fetched list (fixes the stale-cache 404 mirror of the 409 and
  the wrong "saved" toast, and replaces the nested 409 retry); the New button carries a pre-create
  draft prompt (send()'s implicit create already did); a prompt/preset applied while the first-send
  create is in flight is delivered to the new conversation instead of reverted; a stale sysprompt
  blur now PUTs to the conversation the textarea was built for instead of dropping the edit;
  `resetSettings()` = `applySettings({})`; spec §4 points at `PARAM_META` instead of re-enumerating
  knobs; E2E preset-option lookup deduplicated. Skipped as design/pre-existing: cap-gated
  `enable_thinking` pinning via global settings (predates presets; needs caps-aware settings),
  unknown-params round-trip stripping, apply-copies-null-prompt. 882 green; E2E 55/55 live.

## [1.34.23]

### Changed

- **`/simplify` pass over v1.34.22** (4 review angles: reuse, simplification, efficiency, altitude;
  applied 7, skipped 4 as deliberate design or house-pattern): settings.js `mergeKnown()` unifies the
  twice-inlined known-keys merge and `samplerParams()` now derives from `snapshotSettings()`; the
  one-caller `lead` option reverted in favor of `panel.prepend(...)` in chat.js; the open-panel
  freshness guard moved inside `rebuildSettingsPanel()` (call sites unconditional, plus a rebind after
  the implicit first-send create); the sysprompt textarea captures its conversation id at build so a
  stale blur can't write one conversation's prompt onto another; settings panel opens instantly from
  cached presets (refresh repaints in the background); preset save-by-name catches a stale-cache 409
  and retries as overwrite; preset router drops its duplicated field-allowlist (db layer enforces).
  Skipped by design: unifying user presets with the TOML sampler registry (client-side copy semantics
  is intentional), generic `_update_row` extraction, shared sysprompt component. E2E chat 28/28 live.

## [1.34.22]

### Added

- **Per-conversation system prompt editing + saved presets in v3** (LM-Studio-style). Backend: new
  `presets` table in the DuckDB store (additive `CREATE TABLE IF NOT EXISTS` — no schema-version bump,
  existing data untouched) holding named `system_prompt` + sampler-`params` bundles; name uniqueness
  enforced in code on the store's single serialized writer; presets deliberately survive
  `POST /v1/data/clear` (config, not data). New `/v1/presets` router (list/create/update/delete;
  409 on name collision, 400 on bad fields) — spec §4 + `generated-api.ts` updated in this commit.
  v3 chat settings panel: a per-conversation system-prompt editor (PUTs to the conversation on blur;
  a prompt typed before the first send rides along on create) and a preset bar (apply = copy params
  into the panel + prompt onto the conversation; save-by-name creates or overwrites; armed delete).
  These are UI-authored and expanded client-side — distinct from the server's TOML preset registry
  (`ChatRequest.preset`). Tests: +25 unit (store + HTTP), suites 880 green; E2E +3 checks, 55/55 live.

## [1.34.21]

### Fixed

- **Code-review pass over v1.34.20** (8 finder angles, 3 independently verified system claims; all fixes regression-tested, +6 tests, suites 855 green + E2E):
  - Store ops are now transactional (BEGIN/COMMIT with rollback-on-exception): DuckDB autocommits per statement, so a crash mid-operation could previously orphan rows or leave stale `updated_at`; an unhandled error could also have wedged the long-lived connection until ROLLBACK (verified live).
  - Store runs on its own dedicated single worker thread instead of asyncio's shared default executor, where multi-second model loads and full generation-consumption loops could starve trivial conversation reads (verified: `to_thread` = the shared pool; aiosqlite previously had its own thread). The threading.Lock became redundant and was removed.
  - `duckdb.connect` retries the file lock for up to 10s (parity with the old aiosqlite `timeout=10`); previously a restart racing the old process's lock hard-failed startup instantly (verified live).
  - Content blocks are validated at the storage boundary: `{"type":"text","text":null}` no longer poisons a row (flatten would TypeError on every subsequent read, making the conversation permanently unreadable -- repro'd), and a malformed image block (missing/invalid `source`) is a 400 instead of persisting and crashing the whole conversation render client-side. Unknown block types still pass through (forward-compatible). FK dropped from the schema (DuckDB's FK check rejects parent deletes even with children deleted in the same transaction -- documented limitation; integrity enforced in code) with a schema v3 recreate for any same-day v2 file.
  - v3 chat: staged images are cleared on conversation switch/new (previously a photo picked in conversation A silently attached to the next send in B); one `imageBlockUrl()` helper feeds both rendering and wire conversion and handles `url`-type sources (previously `data:undefined;base64,undefined`); Edit-save syncs `content_blocks` from the server response; Copy hidden on image-only messages (copied empty string); multi-file reads parallelized.
  - db.py: `_COLS` derived from `_NAMES` (zip could silently mispair on drift), `_touch_conversation` helper, `update_message` merges locally instead of re-SELECTing multi-MB rows; README + conversation_api docstring no longer say SQLite; `*.duckdb` added to the gitignore safety net.
- Noted, deliberately not fixed here (recorded in the plan): the schema module's flat `ImageBlock` vs the stored nested Anthropic shape (the STORED shape is the spec-correct one; Phase 3b conformance reconciles the schema module), per-turn base64 re-upload of full history (Phase 3b design input: server-side history resolution), keyed message rendering.

## [1.34.20]

### Changed

- **Q5 executed: conversations/notebooks store migrated SQLite/aiosqlite -> DuckDB, messages now persist as CONTENT BLOCKS** (`db.py` rewritten; same public surface). Every operation runs on a worker thread under a store lock with explicit statements -- the aiosqlite shared-implicit-transaction defect class is retired by construction (regression test: concurrent appends serialize with correct positions). DuckDB has no ON DELETE CASCADE, so conversation deletes cascade explicitly. No data migration by owner decision: fresh store at `data/conversations.duckdb` (`HEYLOOK_DB_PATH` still honored). aiosqlite dependency removed.
- **Conversation API accepts Messages-style content blocks** (additive): `content` on message create/update takes a string OR a block list (`[{type:"image",source:{type:"base64",media_type,data}},{type:"text",text}]`); responses carry both `content` (flattened text, back-compatible) and `content_blocks` (full list). Spec §4 updated in this commit; `generated-api.ts` regenerated.

### Added

- **Images in the v3 chat UI** (the point of pulling Q5 forward): attach via file picker (iPhone camera roll included) or paste; thumbnail strip with per-image remove; user messages with images stored as content blocks and rendered as images in history (reload included); generation converts stored blocks to OpenAI `image_url` data-URL parts (works against the VLM path today; the conversion disappears when v3 moves to /v1/messages). Editing is hidden on image messages (the text editor would silently drop the blocks) -- delete/regenerate still work. Verified: unit suite green (849), full E2E green, plus a live round-trip (store blocks -> reload byte-identical -> VLM correctly describes the image over the v3 wire shape).

## [1.34.19]

### Fixed

- **optloop-lib spec-decode baseline guard was incomplete** (found by a `/code-review` pass): the CLI-level `--reset-baseline`+`--spec-decode` refusal only caught the explicit flag, but per-model baselines (v1.34.14) mean a spec-decode run against a *not-yet-benched* model hits the implicit `baseline_data is None` branch and would silently write a **speculative** baseline (inflated gen_tps + mismatched fingerprints for all later comparisons). Moved the guard into `run_benchmark` where baseline presence is known, so it fires for both the explicit and implicit cases (before the prompt loop); removed the now-redundant CLI guard.

### Changed

- **`/simplify` cleanup of the session's E2E + optloop code** (4-angle review): shared `resolve_or_download()` in `bench_common.py` collapses the models.toml→HF-download fallback that was copy-pasted across three resolvers; spec-decode result metadata deduped via one `spec_meta` dict; stale text-model default id fixed. E2E harness: new `lib/dom.mjs` helpers (`waitForLabel` for the toggle-button idiom used ~7×, `findModelRow`/`modelRowState` for the models-row lookup duplicated 4× — the value-returning `modelRowState` avoids a handle-per-poll leak, `settingsInputValue`/`setSettingsInput` for the settings panel); `run.mjs` collapses the two identical suite-run blocks into a loop; magic literals (`STOP_TEST_MAX_TOKENS`, cadence thresholds) named. Behavior-identical; Python 70 tests green (re-run the Metal-gated E2E suite to confirm the JS refactors).

## [1.34.18]

### Changed

- **mlx-vlm bumped 0.6.3 -> 0.6.5 in the root venv** (git pin refreshed to upstream e9c5bd7): brings the gemma-4 video/audio-weight loading fixes and the Qwen3-VL mrope fix to the SERVER (they were previously only in optloop-lib's fork clones). Safety net held: the tests/contract/test_mlxvlm_surface.py pins (added v1.34.5 for exactly this moment) plus the full unit suite are green on 0.6.5, and the v3 E2E suite passed 52/52 against a server running it.

## [1.34.17]

### Fixed

- **Five confirmed v3 frontend defects** (from the triaged external review, plan Phase 4 item 3; E2E 52/52 after):
  - Router crash-guard (`js/app.js`): a page that fails to load/mount renders an in-place error panel instead of bricking navigation.
  - Perf page fetches now abort on page teardown (`{signal: ctx.signal}` passed to systemMetrics/perfProfile).
  - Chat settings panel rebuilds on model switch while open, so capability-gated controls (enable_thinking) track the selected model.
  - Notebook content is readOnly during generation -- the streaming painter overwrites textarea.value every frame, so mid-generation keystrokes were silently destroyed; the surface now locks honestly and unlocks on completion/stop/error.
  - Status lines (chat/notebook/explore) moved --ink-faint -> --ink-muted (~3.5:1 -> ~6:1 on white); --ink-faint documented as placeholders-only at the token definition.
  - NOT included on purpose: the enable_thinking tri-state (contract change, coupled to the Messages extension design -- Phase 3b).

## [1.34.16]

### Added

- **First real-vision VLM baselines: gemma-4 dense + MoE** (`docs/optimization_log.md`). The bench's vision path had never run against a real VLM before (the Mar-16 "VLM baseline" was a text model through the loader's text path). Updating the editable forks to the owner's synced versions — **mlx-vlm 0.6.5 (#1529), mlx-lm 0.31.3 (#1431)** (`uv sync` clean, mlx stayed 0.32.0) — resolved the multimodal-RoPE blocker that the stale Mar-15 fork had; gemma-4 dense/MoE and Qwen3-VL all run the manual pre-filled-cache vision path clean. Baselines (8-bit, 14 prompts incl. 9 real photos, runs=3): **dense gemma-4-31b 15.3 gen_tps / 1592ms vision / 33.3 GB**; **MoE gemma-4-26b-a4b 48.1 gen_tps / 524ms vision / 27.3 GB**. The MoE is ~3× faster decode + vision-encode despite similar total params (only ~4B active) — dense is bandwidth-bound, MoE is dispatch-bound, so library optimizations will score very differently on each (the reason to bench both). Per-model baselines keep them separate. Next: the MTP experiment (MoE + the `-assistant-bf16` drafter via mlx-vlm's `draft_kind="mtp"`).

## [1.34.15]

### Changed

- **E2E harness default model → `gemma-4-26b-a4b-it-8bit-mlx`** (`tests/e2e/run.mjs`, README). Any fast A4B MoE in `models.toml` works; override with `E2E_MODEL`. Unverified against this default (needs a run once it's in `models.toml`) — the streaming-cadence guard needs >30 tok/s, which an 8-bit A4B MoE should clear comfortably.

## [1.34.14]

### Added

- **optloop-lib: per-model baselines** so dense vs MoE (and their distinct fingerprints) don't clobber one another. Baselines now live in `data/<bench>/<model-slug>/baseline.json` via a shared `model_bench_dir()` helper. Also `resolve_model_name()` maps an HF cache snapshot path (`.../models--org--name/snapshots/<hash>`) back to the repo short name instead of the opaque hash (the cosmetic model-name bug), and `slugify_model()` keeps the dir filesystem-safe. Applied to both `bench_text.py` and `bench_vlm.py`; +5 unit tests (70 total). Groundwork for benching gemma-4 dense and MoE side by side.

## [1.34.13]

### Fixed

- **optloop-lib VLM bench: advanced the vision baseline past two blockers** (still one open). The `[bench.vlm]` model id was dead (`mlx-community/Qwen3.5-27B-mxfp8-mlx` -- a text model, not local); pointed it at the local vision model `Qwen3-VL-32B-Instruct-8bit`. This also revealed the bench's VISION path had never run against a real VLM -- the Mar-16 "VLM baseline" was a text model through the loader's text path. Ported the server's two transformers-5.x soft-patches (AutoVideoProcessor -> None, lenient ProcessorMixin video check) verbatim into `bench_vlm.py` so Qwen3-VL loads on a torch-free MLX venv. STILL BLOCKED (documented in `docs/optimization_log.md`, not fixed here): Qwen3-VL's 3D multimodal RoPE -- the bench's simplified pre-filled-cache vision path doesn't supply mrope position_ids, so cos/sin broadcast fails against image-expanded queries. Needs either routing vision through `mlx_vlm.generate` or porting the server's wrap_language_model/position-reset. No false baseline was written.

## [1.34.12]

### Added

- **Speculative-decoding baseline -- optloop-lib's first real run** (`docs/optimization_log.md`). Re-established the text baseline on mlx 0.32.0 (`gemma-3-27b-it-bf16`, 6 prompts incl. the new long_context workload; 11.7 gen_tps, matching the Mar-16 continuity point) and ran the first classic-draft speculative-decoding experiment (draft `gemma-3-1b-it-bf16`). Result: NET-NEGATIVE on this bandwidth-bound bf16 target (composite 0.91 at num_draft=2, 0.96 at num_draft=4). Nuance: `num_draft_tokens` dominates -- at 4, short-context prompts turn positive (short +10%), but the benefit collapses as context grows (long_context -40%), and greedy spec-decode is NOT bit-identical (batched-verify float order flips borderline argmaxes, so fingerprints diverge -- a distributional gate, not the fingerprint guard, is needed to certify a speculative run). Confirms the Direction thesis that the decode win is verification-based decoding (DFlash), not classic draft. Added a `--num-draft-tokens` flag to `bench_text.py` for the sweep; corrected the (wrong) "lossless/bit-identical" docstring. The harness validated itself: it flagged every regression and divergence.

## [1.34.11]

### Changed

- **mlx upgraded 0.31.2 -> 0.32.0 in the root venv (and 0.31.1 -> 0.32.0 in optloop-lib's)** -- v0.32.0 (released 2026-07-07) ships upstream PR #3628 "Fix threaded compile cache cleanup", the real fix for the CompilerCache TLS teardown abort we worked around in v1.31.2. Proven with a discriminating A/B repro (a compiled function returning a TUPLE, executed on a worker thread that then exits): SIGTRAP with the exact production `PyThreadState_Get`/GIL fatal error on 0.31.2, clean on 0.32.0 -- the tuple return was the ingredient the original minimal-repro attempt was missing. `_PinnedExecutorPool` stays regardless (it also bounds stream-registry growth, which the upstream fix does not address). Full suites green on 0.32.0: backend 839, optloop-lib 65, E2E 51/51 live.

### Fixed

- **Root venv extras**: plain `uv sync` had silently stripped the `performance`/`test` extras (pyturbojpeg -- the multipart JPEG decoder -- uvloop, xxhash, cachetools, pytest plugins). Restored with `uv sync --all-extras`; gotcha recorded in CLAUDE.md.

## [1.34.10]

### Added

- **optloop-lib: spec-decode baseline prep** (committing the prior session's tested work-in-progress). `bench_text.py` gains a genuinely long-context workload (~2.5-3k prompt tokens, fixed coherent document) so prefill scaling and a large KV cache are exercised, not just decode; `bench_vlm.py` gains folder-based real-photograph prompts (`data/vlm/photos/`, sorted, empty-safe -- adding/removing photos changes fingerprints, re-baseline after any change); `bench_config.toml` fixes the text-model HF id (the prior `google_gemma-3-27b-it-mlx-bf16` does not exist on HF; now `mlx-community/gemma-3-27b-it-bf16`) and adds `draft_model = gemma-3-1b-it-bf16` for the speculative-decoding comparison run (unset it for the plain baseline). Both additions land BEFORE the first baseline on purpose. optloop-lib suite: 65 green.

## [1.34.9]

### Added

- **Streaming-cadence regression guard in the E2E chat suite** (`tests/e2e/suites/chat.mjs`, now 25 checks): an in-page fetch to `/v1/chat/completions` measures client-observed inter-chunk arrival gaps and asserts median gap < 50ms and > 30 tok/s. The Phase 1 delivery fix (`asyncio.wait` instead of a 0.1s poll) is INVISIBLE to server-side telemetry -- only a client timing the stream can catch a revert to the ~100ms poll ceiling, so this is the sole automated guard for it. Live: 64 chunks, 10.8ms median, 92.2/s on the MoE. Requires a fast `E2E_MODEL` (the default MoE); a natively-slow dense model would false-fail by design.

### Changed

- **Root `.gitignore`: anchored `lib/`/`lib64/` to `/lib/`/`/lib64/`.** The bare setuptools-boilerplate `lib/` matched ANY nested source dir of that name (it had already forced a `!apps/heylook-frontend/src/lib/` negation and silently swallowed `tests/e2e/lib/`). Anchoring to the repo root keeps the build-artifact intent without eating source trees; the frontend negation is now unnecessary and removed.

## [1.34.8]

### Added

- **v3 frontend E2E harness (`tests/e2e/`)**: puppeteer-core + system Chrome driving the real `/v3` frontend against a spawned `heylookllm` with an isolated `HEYLOOK_DB_PATH`, so real conversations/notebooks are never touched (the suites clear all data). Two suites, 51 checks, green against a fast A4B MoE (~90 tok/s): **chat** (24 -- streaming, edit/regenerate/delete position-truncation, stop=partial-saved, post-abort health, settings + the `localStorage` `max_tokens` seed, conversation CRUD, 390px mobile) and **pages** (27 -- notebook autosave + generate-at-cursor tail preservation, explore logprob chips + keyboard nav, perf no-polling proof + range switching, models list/load/unload + HF scan + danger-zone clear). Own `package.json`/`bun.lock` (not repo root); run with `node run.mjs [chat|pages]`. Rebuilds the 52 browser checks lost with the v3 build scratchpad (plan Phase 4 item 1). Two gotchas encoded in the harness: settings are seeded via `localStorage` before boot then a forced reload (settings.js caches localStorage once at import, and a hash-only navigation is same-document, so a plain re-goto never re-reads the seed); `finishStream` flips the Send button to idle before it awaits the partial-save and sets the status, so stop-checks wait on the "Stopped" status, not the button.

## [1.34.7]

### Fixed

- **Teardown waiter-safety moved to the right depth**: `MLXProvider.unload()` now waits for gate WAITERS as well as active generations (the active counter decrements BEFORE `gate.release()` admits the next waiter, so an active-only wait could free weights exactly as a woken queued request started generating). Living in `unload()` means every teardown path -- LRU eviction, `clear_cache`, explicit unload, idle unload -- inherits the guarantee; the router-level check `_unload_idle` gained in v1.34.3 remains as an early skip. The gate is process-global, so with multiple loaded models this conservatively waits out other models' traffic too, bounded by the same 30s force-unload cap as before.
- **Capacity-reservation wait is bounded** (10 min default, `router._reservation_wait_timeout`): a wedged model load no longer blocks admission of every other model indefinitely (each blocked `get_provider` also pinned an asyncio default-executor thread), and the all-pinned `RuntimeError` can no longer be starved by an in-flight reservation -- the loop now raises a clear timeout error naming the in-flight loads.

## [1.34.6]

### Changed

- **Simplify pass over the v1.34.1-.5 diff** (4-angle review: reuse, simplification, efficiency, altitude). Router: the `_LoadingPlaceholder` sentinel inside `self.providers` (13 isinstance checks across 9 methods) is replaced by a `self._loading` side-set mirroring the existing `_pinned` precedent -- `self.providers` again always means "real, loaded providers", every reader keeps its filter-free form, and forgetting to skip a reservation becomes structurally impossible; behavior unchanged (same race tests pass untouched). Telemetry: the per-chunk getattr scrape hand-copied at 4 consume loops collapses into `perf_collector.ChunkTelemetry.absorb()`, and the twice-duplicated TTFT-minus-queue-wait formula into `net_ttft_ms()`. Scans: `scan_paths` computes the configured id/path identity once instead of once per source (was K TOML re-reads + K x N path resolves per scan). Tests: `QueueStatsProvider` subclasses the shared `MockProvider`; the two hand-rolled fake-chunk builders merge into `tests/unit/_fake_chunk.py`.

## [1.34.5]

### Added

- **mlx-lm/mlx-vlm surface contract tests** (`tests/contract/test_mlxvlm_surface.py`, 22 tests): executable pins for every private/undocumented library surface this server consumes -- `prepare_inputs` signature and return shape, `apply_chat_template` kwargs, the `encode_image`/`cached_image_features` pattern, `LanguageModelOutput` fields, the `_position_ids`/`_rope_deltas` attribute convention, `mlx_lm.utils._get_classes`, cache classes' `state`/`empty()` surface, and `GenerationResponse`'s exact field set + non-slotted runtime attachment. Each test names its consumption site, so an aggressive library upgrade fails loudly in tests instead of silently at runtime. This is item 1 of the mlx-vlm bus-factor strategy (plan Direction).

## [1.34.4]

### Fixed

- **Scan/import correctness**: `already_configured` now matches on the resolved weights path as well as the id (a rescan that derives a different id for already-configured weights no longer presents them as unconfigured; symlinked spellings compare equal). Re-import has PUT semantics: importing an id that already exists replaces that entry with the freshly built one (smart defaults + profile + overrides) instead of silently skipping -- refreshing an entry from a rescan no longer requires hand-editing models.toml.

## [1.34.3]

### Fixed

- **Router load-capacity TOCTOU closed.** The capacity check + LRU evict ran under `cache_lock` but the load and publish ran outside it, so two concurrent requests for two DIFFERENT models both passed the check and held two full models in memory at once (OOM-class on a box sized for `max_loaded_models`). `get_provider` now reserves a placeholder slot under `cache_lock` before loading; placeholders count toward capacity, are invisible to every reader API (`get_loaded_models`, `get_current_model_id`, `get_model_status`), are never evicted/unloaded, and are released on load failure. A loader that finds the cache full of other in-flight loads waits for one to publish instead of over-committing.
- **Idle unload no longer tears down a provider with queued requests.** A request waiting at the FIFO generation gate is neither "active" (that starts after gate acquire) nor recently-used (its `last_used` was stamped at cache hit, and gate waits can outlast the idle threshold), so the 60s idle tick could delete model weights out from under a request about to run. `_unload_idle` now checks the provider's `generation_queue_stats()` (active + waiting) under the SAME `cache_lock` hold as the pop, and skips busy providers until a later tick.

## [1.34.2]

### Fixed

- **Reasoning parser is now instantiated per request, not shared on the provider.** The parser (with its streaming buffer state) was built once at model load; every request called `reset()` and streamed through the shared instance, so two interleaved streams on the same model corrupted each other's buffers and request B's `reset()` clobbered request A mid-flight. Each request now gets its own parser via `select_reasoning_parser(provider._template_info)`. The load-time rationale (Mistral's ~1000-token strip-regex compile) is preserved by an `lru_cache` on `_compile_strip_pattern` -- the compiled pattern is stateless and shared; only buffers are per-request.
- **Embedding tokenizer pad-token guard**: decoder-only backbones without a `pad_token` broke the `padding=True` batch-encode call; the embedding provider now falls back to `eos_token` at load (warns if both are missing).

## [1.34.1]

### Fixed

- **Streaming delivery is no longer quantized to ~10 chunks/s.** The disconnect-watch loop in `streaming_utils.async_generator_with_abort` slept a fixed 100ms between `chunk_future.done()` checks, so every SSE chunk waited for the next poll boundary -- capping delivered (and recorded) throughput at ~10 tok/s regardless of model speed, and making e.g. a 60->48 tok/s regression invisible. The loop now blocks on the chunk future with a 100ms timeout (`asyncio.wait`), waking the moment a chunk is ready while keeping the disconnect-detection and keepalive cadence. This was also the measurement prerequisite for all Phase 5 perf work.
- **Headline perf metrics are honest now** (from the 2026-07-06 measurement audit): recorded tok/s comes from mlx-lm's native per-chunk `generation_tps` (measured tightly around the decode loop; previously never read anywhere in src/), with a wall-clock fallback that excludes FIFO queue wait; recorded TTFT excludes queue wait (admission pressure stays visible in its own `queue_wait_ms` field); `/v1/messages` non-streaming `prompt_tps` no longer divides prompt tokens by whole-request elapsed time (it reports native prefill tps); hourly trends and the 60s resource-snapshot rolling average aggregate successful requests only (failed/503 events recorded 0.0 tok/s and dragged averages toward zero). `RequestEvent` gains a `prompt_tps` field (defaulted, back-compat), which flows into `request_events.jsonl`.
- **Close-timed-out streaming executors are quarantined, not dropped.** Dropping the last reference let GC fire `ThreadPoolExecutor`'s weakref callback, which enqueues the shutdown sentinel -- so a wedged worker that eventually finished would EXIT its thread and hit the MLX TLS-teardown process abort the executor pool exists to prevent. The pool now holds quarantined executors for the process lifetime (cost: one leaked idle thread per wedge).

## [1.34.0]

### Removed

- **App-level optloop (`apps/optloop/`) retired.** A measurement audit found its benchmarks import mlx-lm/mlx-vlm directly and never exercise the `src/heylook_llm/` serving path they were chartered to optimize -- a change to the router, radix cache, or generation core scored exactly 1.0 either way -- and no optimization cycle had ever run end-to-end (results.tsv was header-only, `data/cycles/` empty; the only artifacts were `--reset-baseline` writes). The scoring/fingerprint harness itself was sound and lives on in optloop-lib. Serving-path benchmarking will instead be a thin HTTP bench against a running server, planned after the streaming-delivery and headline-metrics fixes (see `docs/project/plan_2026-07.md`, Phase 5 measurement section). Also removed: `docs/optloop_advanced.md` (its headline topics -- the bench activation gap and `.pth` monkey patching -- documented the retired app-level mechanism).

### Changed

- **optloop-lib is now the only optimization bench**, reframed as a manual benchmark tool first (agent-driven loop optional): new `apps/optloop-lib/CLAUDE.md` orientation doc, placeholder `AGENTS.md` deleted (cross-session knowledge consolidates in `docs/optimization_log.md`), `program.md` slimmed ("LOOP FOREVER"/"NEVER STOP" ceremony removed, stale references fixed), and `docs/optloop_guide.md` rewritten lib-only with the still-relevant advanced-guide content merged in. Fingerprinting docs now state the limitation plainly: greedy decode + token-ID fingerprint freezes behavior against the harness's own baseline but certifies nothing about output quality (no ground-truth metric exists).

### Added

- **optloop-lib: models.toml path resolution** (ported from the retired app-level harness before deletion): bench model IDs now resolve CLI `--model-path` > `bench_config.toml` id > the server's root `models.toml` local path (no re-download) > HF download fallback, with an org-prefix fallback match and 5 new unit tests (65 total).

## [1.33.0]

### Changed

- **Generation failures are now typed exceptions, not sentinel chunks.** The provider raises `GenerationFailed` (server-side) or `InvalidGenerationRequest` (client-side, e.g. images sent to a text-only model) instead of yielding an `is_error` chunk that each consumer had to remember to check -- the sentinel approach had already missed two consumers (`batch_processor.py` and `rlm.py` concatenated error text into results; RLM fed "Error: MLX generation failed..." back into its REPL loop as a sub-answer). Raising makes every consumer, present and future, fail loudly by default: batch requests now record `group.error` per-request instead of shipping fake content; RLM surfaces the failure through its own error handling. API translation: HTTP 500 / **400 for client errors (new)** on non-streaming; the same SSE error payload as before when streaming (headers already sent, so client errors also arrive as stream errors there). The wire contract for streaming clients is unchanged; non-streaming clients now get a proper 400 for never-going-to-work requests that previously returned a 200 with error text (pre-1.31.1) or a 500 (1.31.1+).

## [1.32.2]

### Fixed

- **1-in-60 flaky test identified and fixed**: `test_perf_collector.py::test_single_hour_bucket` used live `time.time()` with a +60s second event, so its two events straddled an hour bucket whenever the suite ran in the last minute of any hour (the previously-unexplained single-run failure on 2026-07-06). Now anchored to mid-hour.

### Added

- **Real coverage for two previously-untested production paths** (the deleted tautological tests only asserted their own inline math): `resolve_add_generation_prompt()` extracted from `_apply_template` (prefill convention: trailing assistant message = continue, no new generation prompt) with 4 tests, and `normalize_layer_index()` extracted from `_extract_from_layer` (negative layer indexing + bounds) with 7 tests -- both verified non-tautological by mutation (break helper -> tests fail). The batch path's duplicate inline prefill logic now reuses the helper. Sibling suites confirmed unaffected by the backend changes: legacy frontend 880/880, optloop-lib 60/60, batch-labeler 25/25.

## [1.32.1]

### Fixed

- **`mx.metal.device_info()` deprecation** (warned at every startup): migrated to `mx.device_info()` at all three call sites (`memory.py` startup record, `prompt_cache.py` memory-pressure check -- which would have broken outright when the alias is removed -- and `/v1/capabilities`). Verified warning-free with `-W error::DeprecationWarning`.

### Changed

- **Library-drift audit against installed mlx 0.31.2 / mlx-lm 0.31.3 / mlx-vlm 0.6.3 / transformers 5.5.4**: every load-bearing assumption verified against installed source -- `KVCache.state` laziness, `stream_generate` kwargs, `GenerationResponse` fields, mlx-vlm `apply_chat_template`/`prepare_inputs`/`LanguageModelOutput`, `_get_classes`, ArraysCache trim limitation: all still correct, no broken sites. Removed verified-dead code: two of four transformers compat patches (one now unreachable behind a backend gate, one silently no-oping since `transformers.utils.auto_docstring` became a function) and the `_load_vlm_with_weight_fix` TypeError-fallback + `load_model` monkeypatch (mlx-vlm `load()` has accepted `strict` for a long time).
- **Test-suite consolidation** (-26 tests, 760 passing): deleted `test_mlx_provider_safety.py` (its fake provider invented `unload(max_wait=...)` and `_content_cache`, neither exists on the real provider; real coverage in `test_mlx_provider.py::TestUnload`); removed two tautological classes that asserted their own inline math without touching production code (`TestPrefillConvention`, `TestLayerIndexNormalization` -- both note the real coverage gap for a follow-up); deduplicated tests across `test_speculative`/`test_vlm_inputs`/`test_admin`/`test_config`/`test_rlm`; merged `test_rlm.py`'s three identical-setup timeout tests into one; replaced a `time.sleep(10)` payload the interrupt mechanism couldn't break (leaked a live thread ~7s past the test) with an interruptible busy loop (~1s now). `tests/README.md` scrubbed of the false "pre-existing failures" claims; `CLAUDE.md` test-count reference updated.

## [1.32.0]

### Changed

- **Chat-sane request defaults.** The global sampler floor (what a request gets when neither the request, a preset, nor the model config says anything -- i.e. every freshly imported model) was `temperature 0.1, max_tokens 512`: near-greedy sampling and mid-sentence truncation of long answers. Now `temperature 0.7, max_tokens 4096` (`GLOBAL_SAMPLER_FLOOR`), and the batch fallbacks reference the same constant instead of a third hardcoded 512. Admin/CLI import now stamps `default_preset = "balanced"` (temp 0.7 / top_p 0.9) instead of the deprecated `moderate` alias.
- **Radix prefix cache is bypassed for non-standard KV caches.** Snapshot restore prefix-trims `keys[..., :N, :]`, which is wrong for `QuantizedKVCache` (packed tuple state) and impossible for rotating caches -- the risk was documented in `radix_cache.py` but unenforced (silent wrong output on partial prefix hits). Both lookup and store now gate on `cache_type == "standard"` with no `max_kv_size`.
- **Import size is real bytes, not a name regex.** The CLI import path parsed `size_gb` out of the model NAME (`Qwen-7B` -> "7 GB" -- billions of params masquerading as gigabytes; `-4bit` -> 4 GB) and fed it to the RAM-relative smart defaults. `size_gb` now always comes from the safetensors byte-sum (matching the admin scan path); the name regex only supplies the human label, and only from the directory name (the full-path match let parent-dir fragments win).
- **Config TUI cache defaults aligned** with the new policy (8-bit/group-64 when quantizing, no `max_kv_size` by default, truncation warning on the max-size prompt).

### Added

- **Strict model-config validation** (`MLXModelConfig`): `extra="forbid"` so models.toml typos fail at load instead of silently reverting to defaults; `kv_bits` constrained to 2/4/8 and `kv_group_size` to 32/64/128 (what MLX actually supports); `cache_type="rotating"` without `max_kv_size` now fails validation instead of the first generation; `max_queue_depth` is a real config field (it was read by the generation gate but silently dropped by pydantic, making it permanently 8).

### Removed

- **`quantized_kv_start`** config field: written by smart defaults and stored in every import, but never consumed by `_build_cache_config`/`make_cache` -- pure dead config. Existing models.toml entries carrying it must drop the key (this machine's file was migrated). Also removed: `num_draft_tokens` stamping on import (inert without a `draft_model_path`; the field itself remains) and five `RELOAD_REQUIRED_FIELDS` entries from a long-removed audio provider.

## [1.31.3]

### Changed

- **Import-time KV-cache defaults are now RAM-relative, and `max_kv_size` is never defaulted.** `get_smart_defaults` quantized the KV cache for any model over an absolute weight threshold (>13GB: 8-bit KV; >30GB: 8-bit KV *plus* `max_kv_size = 2048`) -- sized for a small-RAM machine and wrong on a 192GB Studio, where 11 of 14 configured models had auto-quantized KV and 6 carried the 2048 cap. The cap is the worst offender: it creates a RotatingKVCache that **silently drops context** beyond 2048 tokens (and rotating caches have known correctness limits with the radix prefix cache). Now: quantize only when weights exceed ~35% of total unified memory (`psutil`), and never emit `max_kv_size` -- context truncation is an explicit user choice. Existing `models.toml` entries are local data and were migrated by hand on this machine (standard cache below the threshold, quantized-but-uncapped above it).

## [1.31.2]

### Fixed

- **Server aborted (SIGTRAP, `Fatal Python error: PyThreadState_Get`) after a streaming request on models with compiled sampler / quantized-KV paths** (e.g. Qwen3-VL-32B with `cache_type = "quantized"`): `async_generator_with_abort` created a fresh single-worker executor per request and shut it down at stream end, so one MLX-tainted thread died per request. MLX keeps a thread-local `CompilerCache` whose entries hold Python objects when `mx.compile`d *Python* functions ran on that thread; pthread TLS cleanup runs after the Python thread state is destroyed, so the cache destructor deallocated those objects without the GIL -- `Py_FatalError` -> abort (confirmed by two macOS crash reports with identical stacks: `~CompilerCache()` -> `tupledealloc` -> `fatal_error` inside `_pthread_exit`). Fix: `_PinnedExecutorPool` in `streaming_utils.py` leases persistent single-thread executors instead of creating/destroying one per request -- generation stays pinned to one thread (unchanged invariant), but threads are reused, never torn down. A worker whose generator close times out is retired (leaked), not shut down. Repro was deterministic: one streaming request to Qwen3-VL-32B-8bit killed the server; 6/6 clean after the fix. Also removes the per-request thread churn noted as a follow-up in the 1.31.1 review (MLX stream-registry growth). Tests: `tests/unit/test_streaming_executor_pool.py`.

## [1.31.1]

### Fixed

- **Radix cache reuse crashed with "There is no Stream(gpu, N) in current thread"** (recurrence of the 1.30.5 bug class in a different spot): `snapshot_kv` published *lazy* KV slice nodes into the shared radix tree. Generation runs on a fresh single-worker thread per request (`streaming_utils`), and both our `generation_stream` and mlx_lm's are `mx.new_thread_local_stream` -- GPU thread-local streams are destroyed with their thread (verified by direct probe; CPU streams and per-thread *default* streams survive, which is why only this path crashed). When a later request on a different thread hit the cached prefix, mlx_lm's `mx.eval([c.state for c in prompt_cache])` couldn't resolve the dead thread's stream. Snapshots are now materialized (`mx.eval`) at store time on the generating thread, making cached entries thread-agnostic. Reproduced deterministically (4/4 identical-resend radix hits crashed without the fix, 4/4 clean with it) via seed-with-`max_tokens=1` + identical resend, which parks the snapshot on a block the resend fully covers. Regression test: `tests/unit/test_snapshot_thread_affinity.py` (skips off-Metal). Audited `vision_feature_cache` for the same hazard: safe -- its features are scheduled on a (globally registered) default stream, not a thread-local one.
- **Generation failures were streamed to clients as assistant content**: the provider yielded error text as a normal chunk, so frontends rendered and even persisted "Error: MLX generation failed: ..." as a model response -- the crash above shipped as a fake completion. `MLXErrorChunk` (now module-level, `is_error=True`) is surfaced properly: OpenAI streaming emits `data: {"error":{message, type:"server_error", code:"generation_failed"}}` then `[DONE]`; Messages API streaming emits an Anthropic-style `event: error`; non-streaming paths return HTTP 500 with the message in `detail`. Frontend v3's `streaming.js` routes error payloads to `onError`. Contract tests: `tests/contract/test_generation_errors.py`. (Legacy/v2 clients that only read `delta.content` now see an empty response instead of fake content.)

## [1.31.0]

### Added

- **Frontend v3 at `/v3`** (`apps/heylook-frontend-v3/`): from-scratch rewrite per `docs/frontend_v3_spec.md`, served alongside `/v2` until cutover. Vanilla JS ES modules, no framework, no build step. Five pages: chat, notebook, token explorer, models (admin), performance (on-demand only, no polling); batch page dropped per spec. Pretext virtualization is gone -- markdown rendering via the vendored marked + DOMPurify path only. Shared layer replaces v2's per-page boilerplate: `createPage` lifecycle (per-mount state, teardown AbortSignal, auto-cancelled rAF throttles, post-await guards), hash router with nav generated from route registration, route-table-generated `api.js`, `streaming.js` (SSE keepalive-comment handling, `reader.cancel()` on abort, abort-as-normal-completion), data-driven sampler settings panel (null = backend cascade, localStorage key `heylook-v3-settings`). Fresh OKLCH warm-minimal design system (pure-white surface, honey-bronze accent) with desktop + iPhone-Safari layouts. Verified end-to-end against a live backend: 25 chat checks + 27 page checks (streaming, position-truncation edit/regenerate, stop/abort partial save, 503-busy retry path, autosave, generate-at-cursor, logprob chips, admin load/unload/scan, clear-all).
- **`/v3` static mount in `api.py`**: duplicate of the `/v2` block (SPA fallback + path-traversal guard), plus contract tests for both mounts (`tests/contract/test_frontend_mounts.py`).

## [1.30.5]

### Fixed

- **All MLX generation failed with "There is no Stream(gpu, 0) in current thread."**: the dedicated generation stream in `mlx_provider.py` was created at import time with `mx.new_stream(mx.default_device())`. MLX streams are thread-local -- bound to the thread that creates them -- but generation runs on FastAPI's thread pool (`asyncio.to_thread` / `run_in_executor`), not the import thread. When `wired_limit` called `mx.synchronize(generation_stream)` on a pool worker, MLX raised `RuntimeError: There is no Stream(gpu, 0) in current thread.`, so every text and VLM request aborted before producing output (clients saw a fixed-length error string instead of a completion). Switched to `mx.new_thread_local_stream(...)`, which materializes the stream per-thread -- the same API `mlx_lm.generate` uses for its own `generation_stream`. Verified on real Metal across multiple concurrent pool workers.
- **Concurrent requests cannibalized each other**: the generation lock used a *preemption* policy -- a new request aborted the in-flight one to take the lock. Under any concurrency this meant only the newest request ever completed. The Batch applet (fires up to 4 concurrent), the batch-labeler client, and multiple frontends all aborted each other's generations. Replaced with a strict-FIFO admission gate (`providers/common/generation_gate.py`): requests queue in arrival order and each completes. Interactive "cancel on new message" is unaffected -- the frontend already aborts its own in-flight HTTP request, which the disconnect handler turns into a cooperative abort.
- **Generation slot could be held until GC**: the non-streaming, batch, and RLM consume loops never called `generator.close()`, so on a consumer-side exception the provider generator's `finally` (which releases the generation slot) only ran when the garbage collector eventually reclaimed it -- stalling every queued request until then. All consume paths now close the generator (via `contextlib.closing`). (The streaming path already did.)
- **One client's disconnect aborted another client's generation**: the cooperative abort signal was a single `_abort_event` shared per provider. Once FIFO made concurrent requests genuinely live, a disconnecting request set the shared flag and the *active* (unrelated) generation saw it and stopped early -- the connected client got a truncated response. The abort signal is now **per-request**: created by the route, passed to both the generation and the disconnect watcher, so a disconnect cancels only that request.
- **A queued request whose client disconnected still ran a full generation**: `acquire()` didn't watch for cancellation, and the per-generation `_abort_event.clear()` wiped the disconnect signal once the turn arrived. The gate's `acquire(cancel_check=...)` now drops a request from the queue (`GenerationCancelled`) when its client has gone, and the fresh per-request event means a disconnect set during the wait survives. The streaming disconnect wait is also bounded so it can't pin the coroutine.
- **Generation didn't serialize across models**: the gate (and the lock before it) was per-provider, so with `max_loaded_models>1` two models could run concurrent generations on the one GPU. The gate is now a process-global singleton shared by all MLX providers.
- **Metrics double-counted queued requests / `mx.clear_cache()` overlapped the next generation / `max_queue_depth=0` rejected everything**: `_active_generations` is now incremented after acquiring (so a queued request counts as `requests_queued`, not `requests_active`); the MLX cache is cleared before releasing the slot (cleanup completes before the next waiter runs); and `check_capacity()` accounts for the active holder so an idle gate admits the first request even at `max_queue_depth=0`.

### Added

- **`requests_queued` in `/v1/system/metrics`**: per-model count of requests waiting in the FIFO generation queue behind the active one (alongside the existing `requests_active`), for observing backpressure and tuning `max_queue_depth`.
- **Per-request queue-wait timing.** Each request's time blocked in the FIFO queue is measured (around `gen_gate.acquire()`), tagged on the generation chunks, and surfaced three ways: `queue_wait_ms` in the streaming usage chunk's `timing` (when `stream_options.include_usage=true`), a `queue_wait_ms` field on the per-request observability record (`request_events.jsonl`), and an average `queue_wait` in the per-model `bottlenecks` breakdown of the performance profile. Distinct from the existing `queue` metric, which is provider-acquisition / model-load time. Covers both `/v1/chat/completions` and the Messages API, streaming and non-streaming.
- **OpenAPI types drift guard.** `scripts/check_openapi_sync.sh` regenerates `generated-api.ts` from the FastAPI app's schema offline (`app.openapi()`, no running server) using the frontend's pinned `openapi-typescript`, and diffs against the committed file. Wired into the pre-commit hook (gated on staged top-level `src/heylook_llm/*.py`, `schema/`, or the generated file) so the types can't silently drift again; also runnable via `bun run check:api`. Degrades gracefully (skips, never false-blocks) when uv/bun/MLX are unavailable.

### Changed

- **503 backpressure responses now report real queue capacity.** The `model_overloaded` 503 previously hardcoded `X-RateLimit-Limit: 1` ("we can handle 1 concurrent request") and said "processing another request" -- both stale now that requests queue. The body says the generation queue is full, and `X-RateLimit-Limit` reflects actual capacity (`1 + max_queue_depth`) from the provider's live queue snapshot.
- **Generation is serialized FIFO with bounded-depth backpressure.** A single GPU + one loaded model + shared KV cache means one generation at a time; the new `GenerationGate` enforces this in arrival order. HTTP entry points call `provider.check_capacity()` before starting, returning **503 (`model_overloaded`, `Retry-After: 1`)** once `max_queue_depth` requests are already queued -- wiring up the `MODEL_BUSY` 503 path in `api.py`/`messages_api.py` that previously existed but was never triggered. Depth is configurable per model via `max_queue_depth` (default 8). Internal orchestration (batch, RLM) skips the capacity check and simply queues. Batch generation now shares the same gate as chat, so batch and chat can never run on the GPU concurrently.
- **Streaming generation is pinned to one thread.** `async_generator_with_abort` drove each `next()` through the default thread pool, so a single generation's tokens could hop worker threads -- fragile for MLX, whose per-generation stream and `wired_limit` context are entered on the first `next()` and synchronized on the last. Each streaming generation now runs start-to-finish on a dedicated single-thread executor, and the generator is closed on that same worker. Non-streaming paths already ran each generation on one thread.

## [1.30.4]

### Fixed

- **VLM warmup never primed the model**: `MLXProvider.warmup()` passed the full VLM model straight to `generate_text`. A VLM's forward pass returns a `LanguageModelOutput`, but mlx-lm's `generate_step` subscripts logits directly (`logits[:, -1, :]`), so every VLM warmup raised `'LanguageModelOutput' object is not subscriptable`. Warmup now routes the model through the text strategy's `_get_generation_model()` -- the same `LanguageModelLogitsWrapper` real requests use -- so VLMs are actually JIT-primed at load instead of paying compilation cost on the first request. Text-only models are unaffected (wrapper returns the raw model).

### Changed

- **Warmup failures now log at WARNING** (was INFO). A consistently-failing warmup means the model is never primed and the first request pays full JIT cost; logging it at WARNING surfaces the regression instead of burying it. Behavior is otherwise unchanged -- warmup stays best-effort and never blocks model loading.
- **Consolidated VLM language-model wrapping** into `wrap_language_model()` in `providers/common/model_wrappers.py`. The text and vision strategies share one definition of "wrap a VLM's language model for mlx-lm" instead of constructing `LanguageModelLogitsWrapper(model.language_model)` inline in two places. Warmup resolves its generation model through `UnifiedTextStrategy._get_generation_model()` -- the same path real requests use -- so it can't drift back into passing the raw VLM model.
- **Removed a redundant VLM mRoPE position reset** in `UnifiedTextStrategy.generate()`. The `_position_ids`/`_rope_deltas` reset already happens in `run_generation` via `_reset_vlm_positions()` (the wrapper forwards to the same language-model instance), so the inline copy was dead. No behavior change.

## [1.30.3]

### Added

- **Format-aware reasoning parser + template-info driven selection**: new `heylook_llm.reasoning_parser` module replaces the previous hardcoded thinking parser call sites. Two classes + a factory: `HarmonyChannelParser` for multi-channel formats (control tokens `<|channel|>`/`<|message|>`/`<|start|>`/`<|end|>`/`<|return|>`/`<|call|>` stripped; analysis/commentary channels route to `message.thinking`; final channel routes to `message.content`), `PassThroughParser` for formats without reasoning structure. Templates with `<think>...</think>` markers route through the existing `HybridThinkingParser` directly (no wrapper). `select_reasoning_parser(template_info)` picks by template-file signals. Non-streaming + SSE streaming paths both route through the factory.
- **Template-info loader** (`providers/common/template_info.py`): reads `chat_template.jinja` / embedded `tokenizer_config.json` template + unions specials from both `tokenizer.json` `added_tokens` and `tokenizer_config.json` `added_tokens_decoder`. Exposes `ModelTemplateInfo` with `chat_template`, `special_tokens`, `has_harmony_structure` (derived from `<|channel|>` + `<|message|>` literals), `has_thinking_markers` (from `<think>...</think>`), `template_source`. The model's on-disk files are the single source of truth; no tokenizer introspection, no format-name lookup table.
- **Decode-path special-token hygiene**: `apply_special_token_hygiene(tokenizer)` patches `tokenizer.decode` to default `skip_special_tokens=True`, closing the leak where `NaiveStreamingDetokenizer` calls `decode(tokens)` bare and control tokens render as literal strings. Patches both the wrapper and the inner HF tokenizer so either detokenizer reference path is covered. Callers that want raw specials (Token Explorer UI) still pass `skip_special_tokens=False`. Vision first-token and batch text decode sites updated in-place.
- **Chat-template source policy**: `MLXModelConfig.chat_template_source` field (`None`/`"auto"` / `"jinja"` / `"tokenizer_config"` / absolute path). Resolved at load time. Useful when a model ships multiple templates or the user wants to point at a custom `.jinja` for testing. Logged at load. Import wizard auto-detects `chat_template.jinja` in scanned folders and records `"jinja"` when present.
- **CLI `--chat-template` flag** on `heylookllm import`: overrides the auto-detection, recorded in generated `models.toml` so the user can edit post-import.
- **`HarmonyChannelParser` + `PassThroughParser` strip-tokens set**: consumed from the template-info's declared specials and compiled into a single alternation regex (sorted longest-first). Strips non-structural control tokens from output deltas as a defense against fast-detokenizer leaks. Optimized for tokenizers with hundreds-to-thousands of declared specials.

### Changed

- **Parser built once at model load** (`MLXProvider._reasoning_parser`), reset per request instead of rebuilt. Saves the regex compile cost for tokenizers with large reserved-token sets.
- **Harmony control-token scan** collapsed from six `.find()` calls per iteration to one module-level compiled regex (`_HARMONY_CONTROL_PATTERN`). Single `re.search` returns position + matched token.
- **`template_info` uses `orjson` + `read_bytes()`** for JSON parsing, matching project convention.
- **Removed `Qwen3ThinkingParser` adapter class**: `HybridThinkingParser` already conforms to the `ReasoningParser` protocol. Factory imports it directly in the thinking branch.

## [1.30.1]

### Added

- **Idle model unload (C2)**: non-pinned loaded models that go unused for longer than their idle window auto-unload. Global default `idle_unload_seconds = 1800` (30min) in `[defaults]` of `models.toml`; per-model `unload_after_idle_seconds` override on `MLXModelConfig`. `0` at either level disables (per-model override still wins for models that set their own non-zero value). `ModelRouter._last_used_ts` tracks last cache hit or fresh load per model; `unload_idle_models(now_ts)` scans and unloads; pinned models always exempt. Events flow through `model_events.jsonl` as `reason="idle_timeout"`. `MemoryManager.tick()` drives this from the existing 60s resource-snapshot loop -- no new thread.

### Changed

- **`max_loaded_models` schema default flipped from 2 to 1**. Apple Silicon is memory-bandwidth-bound so a second loaded-but-idle model doesn't help throughput; it just holds memory that could serve bigger KV caches or higher-resolution vision batches. Field stays configurable (`Field(1, ge=1)`); existing `models.toml` entries are unaffected (they set their own value). Matches the user's already-explicit `max_loaded_models = 1`.

## [1.30.0]

### Added

- **Runtime preset registry (C1)**: new `heylook_llm.presets` module + `src/heylook_llm/data/presets/*.toml` bundle (8 canonical presets: `balanced`, `creative`, `deterministic`, `code`, `thinking`, `moderate`, `vlm-describe`, `vlm-extract`). `ChatRequest.preset` references them at request time. Five-layer cascade in `MLXProvider._apply_model_defaults`: global floor -> thinking-model defaults -> model sampler fields -> request preset -> request explicit fields. Per-model sampler fields still act as defaults; presets overlay at request time; explicit fields still win. Unknown preset name -> HTTP 400. VLM presets deliberately omit `top_k`/`min_p`/`repetition_penalty` since mlx-vlm's `stream_generate` ignores them (prevents silent no-ops). `request_events.jsonl` now records the resolved preset name.
- **Inference API key (C1.5)**: optional bearer-token gate via `HEYLOOK_API_KEY` env var. When set, inference endpoints (`/v1/chat/completions`, `/v1/batch/chat/completions`, `/v1/messages/*`, `/v1/embeddings`, `/v1/hidden_states*`, `/v1/rlm/*`) require `Authorization: Bearer <value>`. Loopback (`127.0.0.1`, `::1`) is exempt by default -- same-machine dev tools don't need to carry the key; set `HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true` to close the carve-out for paranoid setups. `hmac.compare_digest` comparison, case-insensitive `Bearer` scheme per RFC 6750. Admin token (`HEYLOOK_ADMIN_TOKEN`) remains a separate gate on admin endpoints. Both default unset = open, matching the default single-user localhost UX.

## [1.29.0]

### Added

- **LAN hardening (S1.6)**: optional admin-token gate via `HEYLOOK_ADMIN_TOKEN` env var. When set, `/v1/admin/*`, `/v1/data/clear`, and `/v1/cache/clear` require a matching `X-Heylook-Admin-Token` header or return 401; unset/empty is a backward-compat no-op. Inference endpoints (`/v1/chat/completions`, `/v1/messages/*`, `/v1/embeddings`, `/v1/rlm/*`) are intentionally never gated so clients don't need to learn a shared secret. Startup log now advises on non-loopback binds and reports admin-token status. New `docs/lan_setup.md` walks through Caddy `tls internal` + `caddy trust` + hosts-file flow for HTTPS in front of a loopback-bound inference server; nginx alternative documented.

## [1.28.0]

### Added

- **Per-request peak memory + KV cache bytes telemetry**: `/v1/chat/completions` responses now expose `x-heylook-peak-memory-gb` and `x-heylook-kv-bytes` headers on non-streaming responses; streaming emits the same values in the usage chunk's `timing` object when `stream_options.include_usage=true` (SSE headers can't carry post-generation values). Frontend-v2 chat status bar renders "N tokens · P.PP GB peak · K KV" after each completion. `mx.reset_peak_memory()` is called at the top of `run_generation` to scope the counter per-request.
- **Three-stream observability with content invariant**: new `src/heylook_llm/memory.py` owns three disk-backed JSONL streams under `internal/log/` (gitignored) plus a one-shot startup record. `memory_baseline.jsonl` is the periodic resource snapshot (default hourly); `request_events.jsonl` is one line per completed request with sampler settings, timings, peak memory, cache hit rate, thinking/content token counts, stop reason; `model_events.jsonl` records load/unload with weights bytes, quantization, param count, context length. Configurable via `HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS` / `HEYLOOK_REQUEST_LOG_ENABLED` / `HEYLOOK_MODEL_EVENT_LOG_ENABLED` env vars. Content invariant -- numeric + metadata only, never prompts or responses -- is test-enforced via a recursive forbidden-key walk and a `RequestEvent` primitive-fields-only introspection test. See `docs/observability_guide.md`.
- **Vision feature cache byte cap**: `VisionFeatureCache` now evicts on both `max_entries` (20) and `max_bytes` (default 8 GB). Closes the documented leak vector where a few 8K-resolution images could consume multiple GB despite the entry-count cap. `stats()` exposes `bytes` + `max_bytes`; the hourly baseline aggregates across loaded providers.
- **Provider warmup + prefill_step_size passthrough**: `BaseProvider.warmup()` (no-op default) runs after each `load_model()`. `MLXProvider` override runs a ~30-token throwaway generation to prime Metal shader compilation, killing the 1-3s cold-start latency on first user request. `MLXModelConfig.prefill_step_size` (new optional field) flows into per-request `effective_request` and to `lm_stream_generate` when set; `None` lets mlx-lm use its own 2048 default. `MLX_RUNTIME_DEFAULT_FIELDS` is derived from `MLXModelConfig.model_fields` metadata (fields tagged `json_schema_extra={"is_runtime_default": True}`) so adding a new cache/speculative-decoding field auto-propagates without touching `_apply_model_defaults`.

### Changed

- **Deprecated `mx.metal.*` memory APIs** swapped to the top-level non-deprecated `mx.*` equivalents (`get_active_memory`, `get_peak_memory`, `get_cache_memory`, `reset_peak_memory`) across the observability code and tests. `mx.metal.device_info()` stays -- not aliased, not deprecated.
- **Router tests** migrated from YAML fixtures to TOML (matches `router.py:_load_config` which only dispatches on `.toml`).
- **`docs/FRONTEND_HANDOFF.md` renamed to `docs/frontend_api_reference.md`** (git-detected 98% similarity preserved). Cross-links updated across CLAUDE.md, tests/README.md, apps/heylook-frontend/README.md, internal/backend/api.md.
- **`setup.sh` installation menu** collapsed from 4 options to 2; removed every reference to `mlx_stt` / `parakeet-mlx` / the `stt` install extra (MLXSTTProvider was removed in Phase 2).

### Fixed

- **Double JSON serialization** on non-streaming `/v1/chat/completions`: switched from `JSONResponse(content=result.model_dump())` to `Response(content=result.model_dump_json(), media_type="application/json")` -- Pydantic's single-pass serializer saves ~1-2ms per response.
- **`_normalize_path_for_log`** strips the user's home-directory prefix from `ModelMetadata.path` before emitting to JSONL streams; keeps logs portable.
- **Deleted dead `mlx_batch_vision` test references** in `tests/unit/mlx_perf/` that prevented the suite from being green by default after the v1.23.0 batch-labeler extraction.

### Infrastructure

- **pytest-asyncio** added as a dev dependency so `tests/unit/test_conversation_api.py` and `test_notebook_api.py` run in the default suite (previously required manual install).
- **`sandbox.excludedCommands`** in `.claude/settings.local.json` now exempts `uv run pytest*`, `uv run python*`, `uv sync*`, `uv lock*`, `bun install*`, `bun run build*` so they don't trip the uv cache-access error.
- **`.gitignore` hardened** for runtime data: `*.db`, `*.db-*`, `*.sqlite`, `*.sqlite3`, `*.jsonl`, `/data/*` (with `!/data/.gitkeep`), `apps/*/data/*`.

## [1.27.0]

### Added

- **Conversation storage API**: New `/v1/conversations` CRUD endpoints backed by SQLite (aiosqlite). Stores conversations and messages server-side, enabling simpler frontends that don't need client-side persistence. Endpoints: list, create, get (with messages), update, delete (cascade), append message, edit message, truncate messages after position.

## [1.26.0]

### Added

- **Process-wide wired memory limit**: Server now calls `mx.set_wired_limit(max_recommended_working_set_size)` at startup, matching mlx-lm's server. Model weights stay wired between requests, reducing memory churn and improving time-to-first-token.
- **Vision feature cache**: VLM requests cache vision encoder outputs by image URL. Multi-turn conversations with the same image skip the 200-500ms vision tower forward pass. LRU cache with 20 entries, cleared on model unload. Uses mlx-vlm's `cached_image_features` / `encode_image` API.
- **Byte-level prompt cache budget**: New `--prompt-cache-bytes` CLI flag (e.g. `2G`, `512M`) caps total KV cache memory in the radix tree. Radix cache tracks snapshot sizes and evicts LRU leaves when over budget.
- **Cache stats in API responses**: `usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens were served from the radix cache, matching the OpenAI API format.
- **Segment-aware cache eviction**: Radix tree nodes are tagged with segment type (system/assistant). System prompt KV caches are evicted last, keeping shared system prompts alive longer across conversations.
- **SSE keepalive during long prefill**: Streaming responses emit SSE comments (`: keepalive`) every 5 seconds during prompt processing to prevent connection timeouts on long prompts.

## [1.25.2]

### Fixed

- **OOM on model swap**: Router evicted the old model AFTER loading the new one, holding both in memory simultaneously. With `max_loaded_models: 1`, swapping from a 27B to a 120B model would OOM-kill the process. Eviction now happens BEFORE the new model loads.

## [1.25.1]

### Fixed

- **Single source of truth for connection state**: Removed duplicate `isConnected` state in `App.tsx` -- connection status now lives exclusively in `connectionStore`, eliminating drift between the reconnection banner and the app's connection gate
- **Reconnection banner safe area**: Banner now uses `env(safe-area-inset-top)` so it isn't hidden behind iPhone notch/dynamic island
- **Duplicate listener registration**: `initReconnectionDetection()` is now idempotent -- StrictMode double-fire no longer registers two `visibilitychange` listeners
- **Dead streams on tab restore**: After successful reconnect, stale streaming state is cleaned up so the UI doesn't show a spinner indefinitely
- **Connection error message**: Replaced hardcoded "localhost:8080" with generic message that works for LAN connections
- **E2E sidebar assertion**: Replaced tautological `isOffscreen || isOverlay` check with actual verification that sidebar doesn't push main content off-screen
- **Event listener leak**: `_resetReconnectionState()` now removes the `visibilitychange` listener, preventing stacked listeners in tests and HMR

### Changed

- `chat.spec.ts` refactored to use shared `backendPage`/`modelPage` fixtures instead of inline `beforeEach` blocks
- `connectionStore` now uses `withDiagnostics` middleware for consistency with `modelStore`
- Reconnect module delegates to `connectionStore.checkConnection()` instead of calling `fetchModels` directly -- single code path for refreshing server state
- `modelStore.initialize()` owns the startup sequence (fetchModels + fetchCapabilities), called by both initial connection and reconnection
- Dead-stream cleanup in reconnect is now fire-and-forget (`.then()` instead of `await`) to stay off the critical path
- Safe-area banner uses `.pt-safe` CSS class (matching existing `.pb-safe`/`.mb-safe` pattern) instead of inline style
- Removed no-op `hmr.host: undefined` from `vite.config.ts`
- Consolidated duplicate imports in `persistence.spec.ts` and `conversation.spec.ts`
- Added `data-role` attributes to message wrappers in `MessageList.tsx` (fixes `conversationPage` fixture selector)
- Added idempotent init test to `reconnect.test.ts`

## [1.25.0]

### Fixed

- **iOS Safari meta tag**: Changed `mobile-web-app-capable` to `apple-mobile-web-app-capable` in `index.html` -- Safari now treats the app as a web app and is less aggressive about killing the tab under memory pressure
- **Vite HMR over LAN**: Configured `server.hmr` to auto-detect hostname from request and disabled error overlay to prevent reload loops when iOS Safari freezes and restores the tab's websocket
- **Playwright config**: Changed `npm run dev` to `bun run dev` to match actual package manager
- **Persistence test DB name**: Fixed `heylook-db` to `heylook` in `persistence.spec.ts` (tests were passing by accident since `indexedDB.open` creates any DB name)
- **Persistence test assertions**: Replaced weak `body.toBeVisible()` checks with actual data verification (message text presence, conversation count)

### Added

- **Reconnection detection**: New `reconnect.ts` module detects dead connections after iOS Safari tab restore. Pings `/v1/models`, retries with exponential backoff, shows "Reconnecting..." banner via `connectionStore.ts`
- **Shared E2E fixtures**: New `e2e/fixtures.ts` with `backendPage`, `modelPage`, and `conversationPage` fixtures -- eliminates copy-pasted `setupWithLoadedModel` across 4 test files
- **Navigation E2E tests**: All 7 applet routes tested for rendering, lazy loading, unknown-route redirect, and state preservation across navigation
- **Applet E2E coverage**: New test files for Notebook, Models, Batch, Token Explorer, Model Comparison, and Performance applets (previously zero E2E coverage)
- **Multi-turn E2E test**: Conversation test that sends a message, waits for response, then sends follow-up to verify context is maintained
- **Mobile E2E tests**: Viewport tests (layout at mobile width, sidebar behavior, touch target sizes) and persistence tests (visibilitychange flush, rapid-send-then-background)
- **Playwright browser matrix**: Added WebKit (Desktop Safari), Mobile Safari (iPhone 13), and Mobile Chrome (Pixel 5) projects with 60s timeout for mobile
- **Reconnection unit tests**: 6 tests covering ping success/failure, backoff retry, visibility change listener, and model list refresh
- **Optloop multi-turn prompts**: `bench_text.py` gains `multi_turn_short` (2-turn Q&A follow-up) and `multi_turn_long` (3-turn conversation with system prompt and growing KV cache). `bench_vlm.py` gains `vision_multi_turn` (image analysis followed by text follow-up). Both optloop and optloop-lib benchmark suites updated in sync.

## [1.24.2]

### Added

- **Optloop user guides**: `docs/optloop_guide.md` (user walkthrough, scoring, monitoring) and `docs/optloop_advanced.md` (bench activation gap, monkey patching, performance ceilings, failure modes, FAQ)
- **Optloop data artifact reference**: inventory table documenting what gets created, where it lives, persistence rules, and audience
- **Session-end protocol**: documented in program.md files, README, and user guides (teardown, analysis, optimization log update)
- **Cross-session memory references**: both program.md files now reference `docs/optimization_log.md` in setup and loop steps; optloop-lib also references AGENTS.md

## [1.24.1]

### Added

- **Optloop cross-session memory**: `docs/optimization_log.md` accumulates findings across optloop sessions (baselines, what worked/failed, technical gotchas). Optloop pre-flight, loop iteration, and session-end protocol updated to read and write it.

## [1.24.0]

### Fixed

- **VLM position state bleeding**: Qwen3.5 mRoPE models cache `_position_ids` and `_rope_deltas` on the language model instance. Stale values between fresh generations caused broadcast shape mismatches. Position state is now reset before each fresh generation in `run_generation()`

### Changed

- **DraftTuner**: `ensure_and_get()` replaces separate `_ensure_baseline` + `get_num_draft_tokens` (single lock acquisition per request)
- **LanguageModelLogitsWrapper**: simplified `__call__` hot path -- removed try/except, reduced to single `getattr` check

## [1.23.9]

### Added

- **RLM SHOW_VARS**: `SHOW_VARS()` function in REPL namespace lists user-defined variables with their types
- **RLM root prompt re-injection**: original query is appended to feedback after iteration 0, keeping the model on-task during long runs
- **RLM best partial answer tracking**: fallback paths (max_iterations, error_threshold, timeout) prefer the best text answer seen over raw code block text
- **RLM max_timeout**: wall-clock timeout for the entire RLM loop (`max_timeout` request field, `"timeout"` finish reason). Checked per-iteration
- **RLM llm_query_batched**: `llm_query_batched(prompts)` runs multiple sub-queries with GPU batching when available, sequential fallback otherwise
- **RLM rlm_query_batched**: `rlm_query_batched(prompts)` runs multiple recursive sub-calls sequentially (requires `max_depth >= 2`)
- **RLM custom tools**: `RLMEngine(router, custom_tools=[...])` injects server-registered Python functions into the REPL namespace. Propagates to child RLMs
- **RLM event callbacks**: `on_iteration_start`, `on_iteration_complete`, `on_subcall_start`, `on_subcall_complete` callbacks on `RLMEngine.__init__()` for programmatic monitoring

## [1.23.8]

### Added

- **RLM compaction**: history summarization when context fills up (`compaction`, `compaction_threshold`, `max_context_tokens` request fields). Prevents hitting context window limits on long runs while preserving REPL namespace
- **RLM recursive depth**: `rlm_query()` spawns child RLMs with their own REPL loops for divide-and-conquer over sub-problems (`max_depth` request field, `child_traces` in response metadata)
- **RLM max errors**: stop after N consecutive code execution errors to prevent infinite error loops (`max_errors` request field, `error_threshold` finish reason)

## [1.23.7]

### Added

- **RLM endpoint**: `POST /v1/rlm/completions` -- Recursive Language Model inference with sandboxed Python REPL, iterative code execution, `llm_query()` sub-calls, and SSE streaming support

## [1.23.6]

### Changed

- **Python version**: Bump minimum from 3.11 to 3.12 across all pyproject.toml files (main, optloop-lib, batch-labeler)
- **optloop-lib**: Remove stale dependency floor pins (bare package names, let uv resolve latest)
- **optloop-lib**: Switch build system from hatchling to setuptools (no package to build)

### Added

- **optloop-lib**: Smoke import tests for bench_text, bench_vlm, bench_analysis (60 tests total)

## [1.23.5]

### Added

- **optloop-lib**: Library-level inference optimizer targeting mlx-lm and mlx-vlm internals via editable installs from GitHub fork clones (`apps/optloop-lib/repos/`)

## [1.23.4]

### Fixed

- **Optloop JSON error handling**: `load_baseline()`, `load_cycles()`, and `load_json_runs()` now catch corrupt JSON instead of crashing the entire run (warn to stderr, skip bad files)
- **Optloop lazy MLX import**: `bench_common.py` defers `import mlx.core` to `sync_barrier()` so pure functions can be imported without triggering Metal initialization

### Added

- **Optloop test coverage**: 18 new tests covering `baseline_metrics_from_result`, `get_bench_params`, `get_constraints`, TTFT per-prompt regression, prefill/memory hard constraints, partial fingerprint matches, zero-variance guard, and corrupt JSON handling (52 total)

## [1.23.3]

### Fixed

- **Optloop prefill_tps guard**: bench_vlm.py now guards `prefill_tps` division with `if prefill_time_s > 0 else 0.0`, matching bench_text.py
- **Optloop dead code**: bench_analysis.py `print_rankings` removed redundant `and r.get("status") != "baseline"` condition (always true when first condition is true)

### Added

- **Optloop unit tests**: 34 tests for bench_common pure functions (scoring, variance, constraints, suspicion, fingerprinting, config extraction) in `apps/optloop/tests/`

## [1.23.2]

### Fixed

- **Optloop variance**: bench_common.py used population variance (N divisor) instead of sample variance (N-1), causing CV threshold to pass too easily with runs=3
- **Optloop zero-token guard**: bench_text.py and bench_vlm.py now raise RuntimeError if generation produces 0 tokens instead of silently recording all-zero metrics
- **Optloop atomic writes**: save_baseline, save_run, and save_cycle now write to tmp file and rename to prevent corruption on crash
- **Optloop weight validation**: compute_composite_score warns to stderr if scoring weights don't sum to 1.0
- **Optloop dead config docs**: bench_config.toml `[scoring.decision]` comments now clarify these values are read by the agent from program.md, not by bench scripts

### Changed

- **Optloop README**: full rewrite with detailed end-to-end tutorial, configuration reference, scoring explanation, verification walkthrough, and troubleshooting guide

## [1.23.1]

### Fixed

- **Optloop skill names**: program.md referenced `/mlx` and `/mlx-lm` instead of `/mlx-skills:mlx` and `/mlx-skills:mlx-lm`
- **Variance transposition shadowing**: list comprehension variable `prompt_runs` shadowed outer scope in both bench scripts
- **TimingContext `__exit__`**: renamed unused `*exc` to `*_exc` to suppress linter warning

### Added

- **Local source mode docs**: program.md documents how to use editable mlx-lm/mlx-vlm installs for library-level optimization
- **Commented coderef config**: bench_config.toml has commented-out `allowed_paths` and `banned_diff_patterns` for local source mode

## [1.23.0]

### Added

- **Output fingerprinting**: SHA-256 hash of token ID sequences for greedy decode correctness verification -- mismatch = auto-reject
- **Per-cycle structured logging**: `data/cycles/cycle_NNNN.json` with git info, optimizer hypothesis, verification results, cumulative drift tracking
- **Config-driven bench**: `bench_config.toml` controls scoring weights, constraint thresholds, model paths, and optimizer scope
- **Verification phase**: diff inspection, per-prompt regression checks, suspicion flags, variance checks built into the optimization loop
- **Optloop skill**: `/optloop` skill replaces the old `optloop.md` slash command

### Changed

- **Bench scripts relocated**: moved from `scripts/` to `apps/optloop/scripts/` following the batch-labeler self-contained app pattern
- **Scoring weights configurable**: `compute_composite_score()` accepts weights dict from config instead of module-level constants
- **Constraint thresholds configurable**: `check_hard_constraints()` reads thresholds from config
- **CLI args override config**: bench scripts load `bench_config.toml` as defaults, CLI flags take precedence

### Removed

- `scripts/optloop.md` (replaced by `apps/optloop/program.md` + `/optloop` skill)

## [1.22.0]

### Added

- **Bench harness**: Direct-load benchmark scripts for text (`bench_text.py`) and VLM (`bench_vlm.py`) inference paths -- no HTTP server required
- **Composite scoring**: Weighted metric (40% gen_tps, 25% TTFT, 20% prefill_tps, 15% memory) with hard constraint checks and baseline tracking
- **Bench analysis**: `bench_analysis.py` reads results.tsv and per-run JSON to produce summary tables and progress charts
- **Optimization loop**: `optloop.md` agent instructions for continuous autonomous inference optimization with dual-bench scoring

## [1.21.1]

### Fixed

- **VLM vision**: pass mask as `mask` kwarg (not `attention_mask`) to VLM models, fixing `Model.__call__() missing 1 required positional argument: 'mask'` for mistral3, pixtral, and llava_next architectures
- **VLM chat template**: flatten list content before applying tokenizer chat template, fixing `can only concatenate str (not "list") to str` for mistral3/pixtral models

## [1.21.0]

### Added

- **Packaging**: Package is installable from git via `uv pip install git+https://github.com/fblissjr/heylookitsanllm`. Profiles and service templates ship inside the wheel as package data (`heylook_llm.data.profiles`, `heylook_llm.data.services`).
- **Dynamic version**: Single source of truth in `heylook_llm.__version__`, read by setuptools at build time.
- **Platform guard**: `heylookllm` CLI exits with a clear error on non-macOS platforms.
- **Project URLs**: Homepage, repository, and issues links in package metadata.

### Changed

- **macOS-only deps**: `mlx`, `mlx-lm`, `mlx-vlm`, and `parakeet-mlx` now carry `sys_platform == 'darwin'` markers so pip can resolve the dependency tree on non-macOS (even though the server requires macOS to run).
- **Classifiers**: Removed Linux/Windows OS classifiers; added Python 3.11/3.12/3.13, FastAPI, and AI topic classifiers.
- **Data file paths**: `profiles/` and `services/` moved from repo root into `src/heylook_llm/data/`; path resolution uses `importlib.resources` instead of `__file__`-relative traversal (fixes broken paths when installed from wheel).
- **License metadata**: Switched from `license = { file = "LICENSE" }` to SPDX expression `license = "MIT"` per PEP 639.

## [Unreleased]

### Added

- **Model pinning**: `pin_model()`/`unpin_model()` on ModelRouter prevent LRU eviction of models in active use.
- **Dynamic embedding backbone**: Embedding provider loads any mlx-lm-supported architecture via `load_backbone()`, replacing the hardcoded Gemma3 import.
- **Pooling config**: Embedding models accept `pooling` field (`mean`, `cls`, `none`) for future multi-vector/ColBERT support.
- **Stop-token utility**: Shared `resolve_stop_tokens()` standardizes EOS token resolution across all generation paths.
- **Embedding weight sanitization**: Strips rotary embedding frequencies, vision tower weights, and multimodal projector keys for architecture-agnostic backbone loading.

### Changed

- Pydantic validators migrated from V1 (`@validator`) to V2 (`@field_validator`, `@model_validator`).

### Removed

- **STT provider**: Removed `mlx_stt` provider, `/v1/audio/transcriptions` endpoint, `parakeet-mlx` dependency, and `stt` optional dependency group.
- **Batch vision labeling pipeline**: Decoupled from backend into standalone client app at `apps/batch-labeler/`. Removes `batch_vision_pipeline.py`, 4 API endpoints, provider-specific prefix cache methods, and SQLite/threading infrastructure. The client app calls the existing `/v1/chat/completions` VLM endpoint instead.

### Fixed

- **Radix cache crash on VLM hybrid models (Qwen3.5)**: Editing/deleting messages caused `broadcast_shapes` ValueError when the radix cache restored a partial prefix match. Three root causes: (1) VLM `LanguageModel._position_ids` persisted across requests, causing stale position slicing; (2) KV snapshots contained entries beyond the matched prefix length, corrupting cache offsets; (3) failed prefill stored broken snapshots that cascaded into future requests. Fixed by resetting VLM position state per-request, trimming KVCache to matched prefix on restore, and skipping snapshot storage on generation errors.
- **VLM model loading crash with transformers 5.x**: Four bugs in transformers 5.x prevent VLM processor loading when torchvision is absent (the correct state for MLX-only setups). Patched at import time in `mlx_provider.py`: `VIDEO_PROCESSOR_MAPPING_NAMES` None values, `auto_docstring` IndexError, `AutoVideoProcessor.from_pretrained` hard-fail, and `ProcessorMixin` type-check rejection of optional `None` sub-processors. Qwen3.5 VLM models now load correctly.
- **Batch processor eos_token_ids null safety**: `hasattr` returns `True` when attribute is `None`; switched to `getattr(..., None) or set()` to handle tokenizers that define `eos_token_ids` as `None`.
- **transformers version**: Pinned `>=5.0.0` to match mlx-lm 0.30.8 requirement. Added `override-dependencies` in `[tool.uv]` to force latest regardless of transitive pins.
- **Mobile state persistence**: Chat and notebook stores now flush to IndexedDB on `visibilitychange`/`pagehide`, preventing data loss when mobile Safari kills the tab.
- **iOS delete button**: Sidebar delete button fires from `onTouchEnd` directly, working around iOS Safari not synthesizing click events for small tap targets in scrollable lists.
- **Gitignore silently dropping src/lib/**: Python `lib/` ignore rule was catching `apps/heylook-frontend/src/lib/`. Added exclusion so frontend lib modules are tracked.
- **DB connection retry**: `getDB()` no longer caches a rejected `openDB` promise forever. If IndexedDB open fails (quota, permissions), the next call retries instead of failing permanently.
- **Thread safety in unpin_model**: `unpin_model()` now acquires `cache_lock` to prevent TOCTOU race with `_evict_lru_model()`.

### Removed

- **torchvision dependency**: Removed unused `torchvision` from core dependencies (nothing imports it; it just pulled in PyTorch unnecessarily).

- **MLX Embedding Provider**: New `mlx_embedding` provider for EmbeddingGemma models. Produces contextual 768-dim embeddings via full bidirectional transformer forward pass with padding-aware attention masking, mean pooling, dense projections, and L2 normalization. Supports task-specific prefixes (query, document, code_retrieval, clustering) and quantized model loading (4bit, 8bit via nn.quantize). 30 unit tests.
- **EmbeddingGemmaModel**: Pure MLX encoder reusing mlx-lm Gemma3 internals with bidirectional attention and padding mask. Located in `src/heylook_llm/models/embedding_gemma.py`.
- **Embedding model import**: Model importer now detects embedding models (bidirectional attention config or `*_Dense` projection dirs) and imports them as `provider: "mlx_embedding"` with correct config (no vision/temperature/sampling params). Model service validates and imports `mlx_embedding` provider correctly.

### Fixed

- **Embedding padding attention**: Padding tokens no longer contaminate content token hidden states. EmbeddingGemmaModel now creates a (B, 1, 1, seq_len) additive padding mask instead of passing mask=None to all layers. Identical content with different padding now produces identical embeddings.
- **Diagnostic logging**: Frontend ring buffer (5000 events) with JSONL download from Settings panel. Backend writes structured events to `logs/events.jsonl`. Request IDs (`X-Request-ID` header) correlate frontend and backend events. Console verbosity adjustable via Settings or `window.__setLogLevel()` in devtools.
- **Stream timeout setting**: Configurable stream timeout (default 30s) in Generation Settings panel. Prevents permanently stuck streaming state when the backend hangs.

### Changed

- **Model selection consolidated**: Removed per-applet model dropdowns from Chat, Batch, and Token Explorer. All applets use the globally loaded model from the top-level selector. Model Comparison multi-select unchanged.
- **VLM guards**: Batch shows a warning when a VLM is loaded (batch mode is text-only, submit disabled). Notebook hides image attachment UI when a text-only model is loaded.
- **Tokenizer extraction consolidated**: Provider base class now exposes `get_tokenizer()` method; tokenizer extraction consolidated from 2 duplicated call sites in api.py.
- **Frontend re-render optimization**: Bare `useModelStore()` calls replaced with individual selectors in 9 components (reduces unnecessary re-renders).
- **Logprobs init deduplication**: Logprobs collector initialization extracted into shared `_init_logprobs_collector()` helper (removes ~30 lines of duplication between streaming and non-streaming paths).
- **Frontend package manager**: Migrated from npm to bun.

### Fixed

- **Generation lock deadlock after client disconnect**: `async_generator_with_abort` now calls `sync_gen.close()` in a finally block, deterministically releasing the provider's `_generation_lock` on all exit paths (disconnect, aclose, normal completion). Previously the abandoned generator only released the lock when GC collected it, causing the next request to hang for up to 30s.
- **Bare except handlers**: Replaced `except:` with `except Exception:` in Metal info query and STT cache cleanup (was swallowing SystemExit/KeyboardInterrupt).
- **Logprobs helper exceptions**: `_decode_token` and `_get_token_bytes` exception handlers narrowed to specific types instead of broad `except Exception`.
- **Non-streaming logprobs init error path**: Missing diagnostic event in non-streaming logprobs init `except` block (now shared via extracted helper).
- **Logprobs exception handling**: `add_token()` exception handler narrowed from `except Exception` to specific types (IndexError, ValueError, RuntimeError, TypeError).
- **Non-streaming logprobs diagnostic**: Non-streaming logprobs path now logs `logprobs_missing_data` diagnostic event (was streaming-only).
- **Redundant provider lookup**: Removed redundant `router.get_provider()` call in streaming logprobs initialization.
- **Token Explorer logprobs**: Added full traceback logging and diagnostic events to the logprobs pipeline. Silent exceptions in `logprobs.py` now log `exc_info=True`. Missing tokenizer logged at `WARNING` level instead of silently producing no logprobs.
- **Chat: concurrent stream guard**: Sending a message while a stream was already in-flight silently started a second stream without aborting the first. Both streams wrote to the store simultaneously. A new `ChatStreamManager` singleton ensures the previous stream is aborted before any new one starts.
- **Chat: AbortController leak**: The controller was only nulled on the success path; errors left a stale reference. The controller is now nulled in a `finally` block in all cases.
- **Chat: wrong-conversation targeting**: `finalizeStream` read `activeConversationId` at callback time. Switching conversations mid-stream caused streamed content to be written into the newly active conversation. The conversation ID is now pinned when the stream starts and passed through to `finalizeStream`.
- **Chat: orphaned streams on navigation**: Navigating away from the Chat applet did not stop in-flight streams. The backend kept generating and eventual callbacks wrote to a detached store. `ChatView` now calls `stopGeneration()` on unmount.
- **Chat: no timeout**: A hung backend (stuck MLX kernel, stalled connection) left `isStreaming: true` permanently with no way to recover without a page reload. A 30-second timeout is now applied via `AbortSignal.timeout()`, producing a user-visible error message.
- **Chat: "Failed to fetch" on cancel-and-resend**: Cancelling a streaming generation and immediately sending a new message caused `TypeError: Failed to fetch` because the browser HTTP connection from the aborted SSE stream was never released (relied on GC). The `ReadableStreamDefaultReader` is now explicitly cancelled on abort. `stopGeneration` saves partial content to the message. Stale-callback guards prevent old stream callbacks from corrupting a new stream's state.

## 1.22.0

### Added

- **Performance profiling backend**: `GET /v1/performance/profile/{time_range}` now returns real aggregated data (was a 503 stub). In-memory ring buffer (10K events, ~2MB) records timing breakdown, per-model bottlenecks, hourly trends, and resource timeline. Background task collects system snapshots every 60s. Performance applet in frontend now renders live data.
- **Benchmark script** (`scripts/benchmark.py`): HTTP-based benchmark for measuring TTFT, generation TPS, and memory against a running server. Supports both OpenAI and Messages endpoints, streaming and non-streaming modes, configurable prompt sets, and `--json` output for CI.

## 1.21.0

### Removed

- **llama.cpp provider and all GGUF support**: Deleted `LlamaCppProvider`, `LlamaCppModelConfig`, `LlamaCppEmbeddingExtractor`, `LlamaCppHiddenStatesExtractor`, and all associated config types, router entries, factory functions, test fixtures, and frontend type unions. The `gguf` pyproject extra is removed. Provider type narrowed from `mlx | llama_cpp | gguf | mlx_stt` to `mlx | mlx_stt` throughout backend and frontend. GGUF model entries removed from `models.toml`. A future `llama-server` subprocess provider will replace this.
- **Dead scripts and config**: Removed `setup_analytics.py`, `analytics_config.json`, `.env.example` (analytics system removed in v1.20.0), and broken `tests/run_tests.sh`.

### Changed

- **tests/README.md**: Full rewrite -- documents all 34 backend test files (unit, contract, integration), corrected coverage matrix, updated run commands to `uv run pytest`.

## 1.20.0

### Removed

- **14 dead/broken API endpoints**: Removed analytics endpoints (`/v1/data/summary`, `/v1/data/query`, `/v1/data/request/{id}`), evaluation endpoints (`/v1/eval/create`, `/v1/eval/run`, `/v1/eval/run/{id}`, `/v1/eval/list`), replay endpoint (`/v1/replay/{id}`), async batch processing (`/v1/batch/process`, `/v1/batch/{id}`), and server restart (`/v1/admin/restart`). All were broken at runtime or had no consumers.
- **6 dead files**: `data_endpoint.py`, `api_capabilities.py`, `openapi_enhancements.py`, `analytics_config.py`, `metrics_db.py`, `metrics_db_wrapper.py` -- never imported or only consumed by removed endpoints.
- **Analytics from core path**: Removed metrics database logging from chat completion request/response handlers and server startup initialization.
- **STT dead endpoints**: Removed broken `/v1/audio/translations`, stub `/v1/stt/stream` WebSocket, hardcoded `/v1/stt/models`. Simplified transcription response to `json` and `text` formats only (removed fake `srt`, `vtt`, `verbose_json`).
- **Sync streaming generator**: Removed unused `stream_response_generator()` (async version is what's actually used).

### Changed

- **`/v1/admin/reload` moved to admin_api.py**: New `admin_ops_router` with `/v1/admin` prefix, consistent with other admin endpoints.
- **Image resize logic extracted**: Duplicate ~25-line resize blocks in `create_chat_completion` consolidated into `_apply_image_resize()` helper.
- **Shared streaming utilities**: New `streaming_utils.py` with `async_generator_with_abort()`, `get_provider_or_503()`, and `consume_sync_generator()` -- used by both `api.py` and `messages_api.py`.
- **`/v1/performance/profile/{time_range}`**: Re-added as a stub returning 503, so the frontend Performance applet gets a clean error instead of 404.

## 1.19.0

### Fixed

- **Profile apply bug**: `ModelProfile.apply()` now unconditionally sets profile values instead of only filling gaps. Previously, smart defaults ran first and set `top_k`, `max_tokens`, `cache_type`, etc., so profiles could never override them. Precedence is now: base -> smart_defaults -> profile overrides -> user `--override`.
- **Sub-1B model size regex**: `Qwen3-0.6B` was reported as "(6B)" because the integer pattern `(\d+)b` matched before `(\d+\.\d+)b`. Swapped regex order so decimal patterns match first.
- **Admin `/status` route shadowed by catch-all**: `GET /v1/admin/models/{id}/status` was unreachable because the greedy `GET /{model_id:path}` catch-all was registered first. Reordered route registration so sub-resource routes (`/status`, `/toggle`, `/load`, `/unload`) register before catch-all routes.

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
