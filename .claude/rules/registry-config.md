---
paths:
  - "src/heylook_llm/{model_registry,config,router,toml_comments,model_importer,admin_api,config_api,modality_detect,cache_defaults}.py"
  - "models.toml"
---

# Model registry and models.toml

- models.toml is override-only (`model_registry.py`): anything under `[scan].folders` is served with derived defaults. The merge happens at load (`ModelRouter._load_config`, startup and reload) and never writes models.toml. A `[[models]]` entry is served exactly as written and always wins; discovery can only add. A model is added only by placing it under a watch folder; nothing writes a new entry except an admin config edit (`update_config`).
- Match models by resolved `model_path` (`.resolve()`), never by id.
- Discovery is best-effort: a failing scan is logged and dropped, never fatal.
- Admin edits (`update_config` / `toggle_enabled`) materialize an entry, writing the whole derived config. Reads never materialize. `remove_config` deliberately does not materialize.
- An explicit entry receives none of discovery's derived fields. Adding one field means hand-writing every other field that model needs. Before adding a field to an entry, check what discovery was giving that model (`merge_discovered(data, discover(data))`) and carry it, or the edit is a silent capability removal.
- The router keeps `max_loaded_models=1` by default (LRU evict, pin, idle-unload via `idle_unload_seconds` / `unload_after_idle_seconds`).
- Every provider-config field declares when a change takes effect (`json_schema_extra={"effect": ...}`, classes in `config.EFFECT_CLASSES`). The reload set and `/v1/admin/model-options` derive from it; never hand-maintain a second copy. A new field must be classified or `config.py` refuses to import.
- Config invariants (design record: [docs/architecture/config.md](../../docs/architecture/config.md)): `reload_config()` pushes per_request defaults into loaded providers; admin responses serialize config with `exclude_unset`, and a validator that assigns derived fields must restore `__pydantic_fields_set__`; `stale_reload_fields` is server-derived and never rebuilt client-side.
- models.toml comments survive admin writes (`toml_comments.py`) only while their anchor is unchanged; a comment on a value you patch is deliberately dropped. Keep provenance for a value you are changing in these rule files, `sharp_edges.md` or `internal/`, not beside the value. `tomli_w` stays authoritative; tomlkit is read-only for comment extraction. Never graft comments into a tomlkit-parsed document.
- Pydantic `Field` defaults use keyword form (`Field(default=None, ...)`, never `Field(None, ...)`); this pyright build flags positional defaults as missing constructor arguments. `test_field_keyword_defaults.py` enforces it.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "Model registry and config".
