---
paths:
  - "src/heylook_llm/*_api.py"
  - "src/heylook_llm/{api,db,request_registry,observability,diagnostic_logger,settings,openapi_doc,frontend_static,busy_response,model_service,server}.py"
  - "src/heylook_llm/schema/**"
---

# API routes, the inference wire, the store

## App and routers

- `api.py` is app assembly only (lifespan, the MODEL_BUSY handler, router mounting). Every route lives in a `*_api.py` router. There is no CORS middleware (v2.0.123): the UI is same-origin and the other clients are not browsers; do not add a wildcard back. Admin writes refuse the provider-config fields flagged `file_only` (`config.FILE_ONLY_FIELDS`: a program path, raw argv, a template path); those are set in the model's `model.heylook.toml` only. The OpenAPI narrative is `openapi_doc.py`; the static frontend is `frontend_static.py`. A route added to `api.py` itself is in the wrong place.
- New endpoint or changed response model: a module with `APIRouter(tags=["Name"])`, the tag added to `openapi_tags`, and `app.include_router()` in `api.py`. The live schema is `/openapi.json`; there is no committed OpenAPI artifact. Update [docs/frontend_v3_spec.md](../../docs/frontend_v3_spec.md) §4 in the same commit.
- A Pydantic model with custom headers: `Response(content=model.model_dump_json(), media_type="application/json", headers=...)` (`JSONResponse` double-serializes).
- Pydantic `Field` defaults use keyword form (`Field(default=None, ...)`); `test_field_keyword_defaults.py` enforces it.

## One inference wire

- `/v1/messages` (Anthropic Messages-conformant plus documented heylook extensions) and `/v1/conversations/{id}/generate`, which shares its grammar. Do not re-add an OpenAI wire. `/v1/models` keeps the OpenAI list shape because clients read `data`; that is a shape, not a wire.
- `ChatRequest` stays as the internal request every provider takes, in OpenAI vocabulary; the rename to Anthropic's happens once, in `converters`. Nothing binds `ChatRequest` as a request body, so a guard refusing a removed or renamed field belongs on `MessageCreateRequest`. A test for such a guard must go through the route.
- Messages conformance: media blocks accept both nested `source` and flat `source_type` (`source` is a declared `MediaSource` field so it reaches the JSON Schema). Thinking blocks and `thinking_delta` carry text under both `thinking` and `text`; keep both. `stop_reason` comes from one table, `converters.STOP_REASON_FROM_FINISH_REASON`, and `TestStopReasonHasOneMapper` asserts it is the only writer across both routes. An aborted generate run reports `max_tokens`. `error` is not a stop reason. The `/v1/conversations` store accepts only nested `source`. Deliberate differences from Anthropic's spec are hand-listed in `docs/api_integration.md` and can be wrong.
- Cancellation (`request_registry.py`): `DELETE /v1/requests/{request_id}` sets a run's `AbortEvent`. It makes a run stoppable, not self-stopping; disconnect polling was deliberately not built (owner call). The id comes from `X-Request-ID` via the shared `resolve_request_id` (bounded and charset-restricted). The registry maps an id to a set. Register a streaming body by wrapping the generator (`tracked_stream`), never with a `with` around the return.

## DuckDB store

- `db.py` holds conversations, notebooks, presets and `settings`, with a single serialized writer thread and transactional ops. `HEYLOOK_DB_PATH` overrides the location.
- Dynamic field names are gated by allowlists: the `_UPDATABLE_*_FIELDS` frozensets and the public `UPDATABLE_CONVERSATION_FIELDS`, which the update route also uses (never a second copy).
- Schema changes: add a table with `CREATE TABLE IF NOT EXISTS`; for a real change, bump `_SCHEMA_VERSION`, which drops all tables. `settings` and `presets` are additive, drop-safe and not in the drop list. No migration code.
- `applied_preset_id` is written only on explicit Apply/Update/Save-as-new; a document that merely matches a preset is labelled by live client-side matching and never stamped.

## Observability

- One ingestion path, `record_event(type, *, tier, min_level, source, fields=<dict>)` in `observability.py`, writing level-gated JSONL under `logs/`. `metrics.jsonl` is content-free; `events.jsonl` may carry bounded error text but never prompts, responses or token IDs. `fields` is an explicit dict, not `**kwargs`. It never raises. `diag_event` delegates here.
- Control is one setting, `observability_level` (off|minimal|standard|debug), resolved from the config file's `[settings]` > default, default `off`: file logging is opt-in (owner rule). No env override; env is bootstrap-only (`HEYLOOK_LOGS_DIR`, `HEYLOOK_DB_PATH`, `HEYLOOK_READONLY_MODEL_CONFIG`). `off` also silences memory.py's streams and the llama-server `.log`, which is decided at spawn, so capturing llama-server output needs the level raised before load, then a reload. `minimal` is not content-free; only the metrics tier is.
- Operational settings live in heylook.toml's `[settings]` table (owner decision 2026-09-23: one server config file), contract in `settings.py` (`SettingsSchema` + `resolve_settings`), CRUD via `/v1/admin/config`, written through `ModelService._write_toml` so comments survive. A read-only instance (`HEYLOOK_READONLY_MODEL_CONFIG`) holds its writes in memory for its own lifetime (`config_api._MEMORY_ONLY`) and never writes the shared file. Level and retention are cached in-process (`observability.configure`) and refreshed at startup and on PUT. The per-stream logging toggles and their env overrides retired; `memory.BASELINE_INTERVAL_SECONDS` is a constant.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "API and store".
