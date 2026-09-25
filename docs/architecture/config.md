# Configuration System

Last updated: 2026-07-28

This document explains the configuration system, `models.toml` structure, Pydantic schemas, and how to configure models for each provider.

**Not the same thing as operational settings.** This document covers
`models.toml` / `MLXModelConfig` -- the model registry, loaded at startup
and reloaded via `POST /v1/admin/reload`. Runtime-mutable server behavior
(observability verbosity, the MLX buffer-cache cap) is a separate system,
`settings.py` / `SettingsSchema`, resolved DB-over-default and edited via
`/v1/admin/config` -- schema + resolution in `src/heylook_llm/settings.py`.
The two are unrelated: a `models.toml` edit needs a reload; a
`/v1/admin/config` change applies immediately with no reload.

## What changed 2026-07-05/06

Three fixes landed against the model import/config/loading system this
window (full findings: `internal/log/log_2026-07-06.md`, audit-to-fix
narrative starting "audit: model import / config / loading system"):

- **Import-time KV-cache defaults are RAM-relative, and `max_kv_size` is
  never defaulted** (v1.31.3) -- see "Smart Defaults at Import" below.
  (Both retired v2.0.88 with the MLX KV fields.)
- **Chat-sane request defaults + strict config validation** (v1.32.0) --
  see "Sampler Defaults and the Effective-Request Cascade" and
  "Validation" below. `quantized_kv_start` was removed as dead config.
- **Importer `size_gb` is real safetensors bytes, not a name regex**
  (v1.32.0) -- see "Model Importer" below.

## Overview

heylookitsanllm uses TOML configuration for model management and Pydantic for validation:
- `models.toml` -- model configuration (user-editable, gitignored)
- `src/heylook_llm/config.py` -- Pydantic schemas and validation (Pydantic V2)

Supports hot-reloading without server restart via `POST /v1/admin/reload`.

## models.toml Structure

### Basic Structure

```toml
# No default_model (removed v2.0.150): every request names its model, and
# startup loads nothing unless `--model-id` asks for it.
max_loaded_models = 2

# THIN by design (derive-at-load, v1.47+): id + provider + model_path is a
# complete entry. Anything else you write is an explicit override. Live
# reference: models.example.toml at the repo root (validated against the
# real schema by tests/unit/test_config.py::TestModelsExampleToml).
[[models]]
id = "model-identifier"       # Unique ID for API requests
provider = "mlx"              # mlx | gguf
enabled = true                # Include in /v1/models?

  [models.config]
  model_path = "path/to/model"
  # optional operator overrides only -- see the field table below
```

### Top-Level Fields

#### `id` (required)
Unique identifier used in API requests. Must be unique, no spaces, case-sensitive.

#### `provider` (required)
Valid values: `"mlx"`, `"gguf"`

| Provider | Text | Vision | Platforms |
|---|---|---|---|
| `mlx` | Yes | Yes | macOS (Apple Silicon) |
| `gguf` | Yes | Yes (mmproj) | anywhere a llama-server binary runs |

#### `enabled` (optional, default `true`)
Whether the model appears in `/v1/models`. Set to `false` to hide experimental models.

#### `capabilities` (optional)
Array of feature flags for client-side discovery. Common values: `"chat"`, `"vision"`, `"thinking"`.

#### `config` (required)
Provider-specific configuration (see sections below).

---

## MLX Provider Configuration (`provider = "mlx"`)

### Examples

The minimal entry IS the normal entry (derive-at-load: modalities, chat
template, sampling and cache defaults all come from the model's own files;
text vs vision needs no declaration for a local dir with a config.json):

```toml
[[models]]
id = "my-model-4bit"
provider = "mlx"
enabled = true

  [models.config]
  model_path = "/path/to/models/my-model-4bit"
```

Fields you write are explicit overrides (each a deliberate choice):

```toml
[[models]]
id = "my-tuned-model"
provider = "mlx"
enabled = true

  [models.config]
  model_path = "/path/to/models/my-tuned-model"
  temperature = 0.6         # beats the model's generation_config.json
  max_tokens = 8192
```

More shapes (GGUF sidecars): `models.example.toml`.

### MLX Config Fields

`MLXModelConfig` (`src/heylook_llm/config.py`, lines 134-208) sets
`model_config = ConfigDict(extra="forbid")` -- any key not in this table
fails config load/import validation. (`trust_remote_code` was removed from
this table because it was never a real field here -- it does not exist on
`MLXModelConfig`. Under the pre-v1.32.0 default Pydantic behavior it was
silently dropped; under `extra="forbid"` it would now fail validation.
Do not put it in `models.toml`.)

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | required | HuggingFace model ID or local path |
| `modalities` | list[str] | derived | Author-declared capability set (`text`/`vision`/`audio`/`video`); `text` always present. **Derive-at-load (v1.47.0)**: when absent, detected at CONFIG-LOAD time from the model dir's `config.json` blocks (`vision_config`/`audio_config` + `*_token_id`, shared detector `modality_detect.py`); a stored value is an explicit override and wins. No config.json to read -> legacy derivation from `vision`. The importer no longer materializes it (MLX entries; GGUF still does -- no config.json to probe at load). |
| ~~`loader`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| `vision` | bool | `false` | **Derived mirror** of `"vision" in modalities`, retained for back-compat. Setting it seeds `modalities` when `modalities` is omitted; if both are set, `modalities` wins. Load routing goes through `loader`/`effective_loader`, not this flag. |
| `context_length` | int (`gt=0`), none | none = read config.json | The model's context window when `config.json` does not tell the truth (a YaRN-scaled checkpoint ships the ORIGINAL `max_position_embeddings` with the factor in `rope_scaling`). Absent = `capabilities.model_context_length` reads the file. Read once at load into `MLXProvider.context_length`, the number `run_generation` refuses an over-length prompt against and the admin row / `/v1/models` report; requires a reload. MLX only -- gguf's window is what the process was spawned with (`ctx_size`). |
| `max_tokens` | int | none | Default maximum tokens to generate. Unset falls through the effective-request cascade to `GLOBAL_SAMPLER_FLOOR['max_tokens']` (16384) -- see below. |
| `temperature` | float | none | Default sampling temperature. Unset falls through to the cascade (0.7 floor). |
| `top_p` | float | none | Nucleus sampling threshold |
| `top_k` | int | none | Top-k sampling |
| `min_p` | float | none | Min-p sampling |
| `repetition_penalty` | float | none | Repetition penalty |
| `presence_penalty` | float | none | Presence penalty |
| ~~`cache_type`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| ~~`max_kv_size`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| ~~`kv_bits`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| ~~`kv_group_size`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| `max_queue_depth` | int (`ge=1`) | `8` | Requests admitted behind the active generation before 503 backpressure. A real config field as of v1.32.0 -- previously read by the generation gate but not declared on `MLXModelConfig`, so it was silently dropped by Pydantic and permanently 8 regardless of `models.toml` |
| `enable_thinking` | bool | `false` | Thinking-mode default for this model (any thinking-capable template, not Qwen3-specific -- see "Sampler Defaults" below for the request-time cascade) |
| ~~`vision_tokens`~~ | -- | -- | REMOVED v2.0.64 (measured dead on Qwen-family MLX: mlx-vlm drops the `max_pixels` kwarg; cap pixels client-side instead). Was: per-model default visual token budget per image (16-16384). A request's own `vision_tokens` overrides; `none` leaves the image processor's own default. Mapped per model family by `providers/common/vision_budget.py` (gemma-4: discrete `max_soft_tokens` bucket; qwen2/3-VL: `max_pixels`) |
| ~~`supports_thinking`~~ | -- | -- | REMOVED v1.46.0 (MLX only; the GGUF config keeps its flag). MLX thinking capability is derived: `enable_thinking`, else template probe, else the explicit `ModelConfig.capabilities` override. |
| ~~`draft_model_path`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| ~~`num_draft_tokens`~~ | -- | -- | REMOVED v2.0.88 (plan W10 stage 3): every MLX model runs on mlx-vlm's engine, which builds its own caches, has no mlx-lm drafter path, and needs no library routing. An entry still carrying it fails validation. |
| `unload_after_idle_seconds` | int, none | none | Per-model idle-unload override. `None` = use `AppConfig.idle_unload_seconds`; `0` = never idle-unload this model |
| `chat_template_source` | string, none | none | `"auto"` / `"jinja"` / `"tokenizer_config"` / absolute path -- overrides chat-template source detection. Since v1.47.0 the importer records this ONLY for an explicit CLI `--chat-template` override; absent = load-time auto resolution (template_info.py, same policy the detection duplicated). |

**Removed field: `quantized_kv_start`.** Written by the pre-v1.31.3 smart
defaults and stored in every quantized-cache import, but never consumed by
`_build_cache_config`/`make_cache` -- pure dead config (confirmed by
grepping the codebase for consumers: none). Removed from `MLXModelConfig`
entirely in v1.32.0. If an older `models.toml` still carries this key,
`extra="forbid"` will now reject it at load -- strip the key.

### model_path formats

```toml
# HuggingFace model ID
model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Local path (relative to the repo, or absolute; tilde expanded)
model_path = "/path/to/models/qwen-2.5-custom"
```

### Sampler Defaults and the Effective-Request Cascade

A chat request's actual sampler values are resolved by the shared
`resolve_effective_sampling` (`src/heylook_llm/samplers.py`) -- ONE
implementation used by BOTH providers (`MLXProvider._apply_model_defaults`
wraps it to add the cached vendor-layer read and MLX runtime-default fields;
`LlamaServerProvider._build_payload` calls it directly).

The layers below are the whole cascade since v2.0.30, each overriding only the
fields it sets. The shape of the answer is: *the model's own settings, then this
model's overrides, then what the request said outright* -- with a hardcoded
fallback only where all three are silent.

1. **Floor** -- deliberately small, and three different KINDS of value that
   are kept apart in `samplers.py` because they rot differently:
   - `FALLBACK_TEMPERATURE = 1.0` / `FALLBACK_TOP_P = 0.95` -- the only two
     that are an OPINION about sampling (v1.79.60 owner ruling: low
     temperature flattens generative prose; 0.7/1.0 before that, 0.1/512
     before that). They apply ONLY where the model's metadata is silent.
   - `DEFAULT_MAX_TOKENS = 16384` -- not taste. llama-server's `n_predict`
     default is UNLIMITED, so a request naming no cap runs to the end of the
     context. A stop, not a preference.
   - `KNOBS_OFF` (`top_k 0`, `min_p 0.0`, `repetition_penalty 1.0`,
     `presence_penalty 0.0`) -- each means "this knob is OFF", not "we prefer
     this". Load-bearing anyway, because the ENGINES' own defaults are not
     neutral: llama.cpp ships `top_k = 40` and applies it to any request that
     omits the key, so dropping these would hand each engine its taste back
     and let the two diverge on identical input.
   1. **Vendor layer** -- the model's OWN published settings, and the layer
      that should normally decide. MLX reads `temperature`/`top_p`/`top_k`
      from the model dir's `generation_config.json` (`load_vendor_sampling`);
      gguf reads the same three from the GGUF header's `general.sampling.*`
      (`gguf_metadata.vendor_sampling`, v2.0.23), which converters write FROM
      that same generation_config.json. gemma-4 runs 1.0/64/0.95, Qwen3.6
      1.0/20/0.95, each from its own file. Best-effort: a missing or malformed
      source yields nothing and never blocks a load.
2. **Model sampler fields** from `models.toml` (per-model overrides in the
   table above).
3. **Request-level explicit field values** -- always win.

The **thinking switch is still resolved here** -- request `enable_thinking`,
else the model config's, else the capability -- because both engines must read
the same bool. It just no longer drags a sampler value with it.

**Removed in v2.0.32: the thinking anti-loop overlay.** It applied
`presence_penalty 1.5` to every thinking-capable model on both engines. The
value came from a "Qwen3-style" bundle in July 2026 and became automatic in
the same commit that slimmed that bundle away, on the strength of one observed
gemma MoE repetition loop. It was never measured, it was the sole survivor of
a set whose other values the vendor layer replaced (surviving only because no
vendor ships a `presence_penalty` -- it is not an HF generation_config field
at all), and Qwen's own guidance for a THINKING model is 0.0, with 1.5 being
their NON-thinking value.

Repetition control did not go with it: `presence_penalty` is settable per
model in `models.toml` and per request. gguf gained the per-model field in the
same change, having had none -- removing a global default without leaving a
per-model lever would have been a capability loss rather than a
simplification. A model that genuinely loops is now fixed on that model.

**Removed in v2.0.30: the bundled sampler registry.** Layers 3b (models.toml
`default_sampler`) and 4 (`ChatRequest.sampler`) named entries in a
`SamplerRegistry` loaded from five TOMLs under `data/samplers/`. All of it is
gone -- the TOMLs, the registry, both wire fields, `/v1/admin/models/samplers`,
`/v1/capabilities.samplers`, `bulk-default-sampler`, `request_guards.py` and
the `--sampler`/`--preset`/`--profile` CLI arguments.

It shipped generic guesses applied to every model, which is the opposite of
what the vendor layer does. Three of the five had no consumer anywhere; the
`thinking` entry was provably a no-op because the cascade hardcoded the same
constant as a fallback; and `balanced` -- stamped on every imported model --
carried `temperature = 0.7`, the value the owner had explicitly overturned
when raising the floor to 1.0. No frontend code ever sent `sampler` or read
either roster endpoint, and the e2e suite asserts the generate wire stays
sampler-free. A request still sending `sampler` or `preset` now gets a 422
naming the removal.

Named bundles that a USER wants still exist as the `/v1/presets` DuckDB
system, which is editable and client-expanded -- a preset reaches the wire as
explicit sampler fields, so it arrives at layer 4 and needs no layer of its own.

### Smart Defaults at Import

> **Historical (removed v2.0.88).** `get_smart_defaults` and the
> RAM-relative KV-cache defaults went with the MLX KV fields when the
> mlx-vlm engine replaced mlx-lm's caches (plan W10 stage 3; owner decision:
> f16 KV everywhere). The section below records what they did.

`get_smart_defaults()` (`src/heylook_llm/model_service.py`, lines
153-193) computes **load-time** defaults (`cache_type`, `kv_bits`,
`kv_group_size`) at import. It does not touch sampler fields -- those are
the request-time cascade's job (above).

Before v1.31.3, KV quantization triggered on an absolute weight-size
threshold (`>13GB` -> 8-bit KV; `>30GB` -> also `max_kv_size = 2048`).
This is wrong on any machine whose unified memory isn't implicitly
assumed by the threshold: a 40GB model is "large" on a 64GB laptop and
trivial on a 192GB Studio. On a 192GB machine, this threshold had
auto-quantized the KV cache for 11 of 14 configured models, 6 of which
also carried the 2048 cap -- silently, because both defaults were applied
at import with no user-visible warning.

The `2048` cap was the worse of the two: it creates a `RotatingKVCache`
(`cache_helpers.make_cache`, `cache_type == "rotating"`), which **silently
drops context** beyond the cap -- older tokens are evicted from the
window entirely, not just quantized. A model configured this way answers
about content it can no longer see, with no error.

Now (`model_service.py`, lines 172-193):

```python
if size_gb > _system_ram_gb() * 0.35:
    defaults["cache_type"] = "quantized"
    defaults["kv_bits"] = 8
    defaults["kv_group_size"] = 64
else:
    defaults["cache_type"] = "standard"
```

`_system_ram_gb()` (lines 144-150) reads total unified memory via
`psutil`, falling back to a conservative `64.0` GB if `psutil` is
unavailable. Quantization now triggers only when model weights alone
claim over ~35% of unified memory (leaving headroom for KV, vision
towers, and the OS) -- RAM-relative, not an absolute GB figure.

**`max_kv_size` is deliberately never set by smart defaults, at any
size.** Context truncation via a rotating cache is now an explicit,
user-chosen `models.toml` edit, never an automatic side effect of
importing a large model.

### Validation

`MLXModelConfig` (`src/heylook_llm/config.py`, lines 134-208) is stricter
than it used to be:

- **`model_config = ConfigDict(extra="forbid")`**: a typo'd or renamed key
  in `models.toml` (e.g. `temperatue`) now fails config load loudly
  instead of being silently ignored and falling back to defaults.
- **`kv_bits: Optional[Literal[2, 4, 8]]`** and
  **`kv_group_size: Literal[32, 64, 128]`**: constrained to the bit
  widths and group sizes MLX's `QuantizedKVCache` actually supports.
  Anything else previously validated cleanly and failed at first
  generation.
- **`_rotating_requires_max_kv_size`** (`@model_validator(mode="after")`,
  lines 201-208): `cache_type = "rotating"` without `max_kv_size` now
  fails validation at load/import time. Enforced here because
  `cache_helpers.make_cache` raises for exactly this at first generation
  -- a config guaranteed to fail should never validate cleanly.
- **`max_queue_depth: int = Field(default=8, ge=1)`**: promoted from
  "read by the generation gate, absent from the schema" (silently
  dropped by Pydantic, permanently 8 regardless of what `models.toml`
  said) to a real, validated field.

---

## Pydantic Schemas

**File**: `src/heylook_llm/config.py` (Pydantic V2, `@field_validator` / `@model_validator`)

```python
class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "gguf"]
    config: Union[MLXModelConfig, GGUFModelConfig]
    description: Optional[str]
    tags: List[str] = []
    enabled: bool = True
    capabilities: List[str] = []

class AppConfig(BaseModel):
    models: List[ModelConfig]
    max_loaded_models: int = 2
```

The `config` field is discriminated on `provider`: `"mlx"` parses as `MLXModelConfig`, `"gguf"` as `GGUFModelConfig`.

---

## Model Profiles

> **The bundled sampler registry this section described was REMOVED in
> v2.0.30.** See "Sampler Defaults and the Effective-Request Cascade"
> above for what replaced it: the model's own published settings as the
> primary source, a two-value fallback beneath, and `/v1/presets` as the
> one remaining named-bundle system.

## Field-effect metadata (v1.52+, design record)

Every provider-config field declares WHEN a change to it takes effect, as
`json_schema_extra={"effect": ...}` on the field itself -- six classes:
`identity` (a different entry), `requires_reload` (teardown/respawn),
`load_time_only` (a reload cannot fix it; carries a `reason`),
`applies_live` (router re-reads while loaded), `per_request` (a model-level
default), `descriptive` (changes what we advertise, not the process).
Field-local ON PURPOSE: every drift this replaced (an MLX-shaped reload set
naming no gguf field; an import allowlist that silently dropped five)
existed because the fact lived somewhere other than the declaration it
described. `arg` alongside it is the llama-server spelling, pinned to the
emitted argv by `tests/unit/test_gguf_argv_matches_metadata.py`; `ui`/
`shape`/`reason` are pass-through UI hints.

Derived consumers (never hand-maintain a second copy):
- `reload_required_for(provider)` -- the PATCH route's
  `reload_required_fields`, provider-aware.
- the gguf import allowlist (`configurable_fields`, "not identity").
- `GET /v1/admin/model-options` -- the full six-way distinction, consumed by
  v3's config editor. This is the consumer that makes a misclassification
  VISIBLE; the in-process two collapse the classes to a binary.
- `_validate_effect_declarations()` refuses to import with an unclassified
  or misspelt field.

Invariant added 2026-08-11 (v1.55.0): **`per_request` means the loaded
process re-reads the default** -- `router.reload_config()` pushes
per_request keys from the fresh config into every loaded provider's config
dict, because providers are constructed with a snapshot and read these
defaults from it at request time. Without the refresh, "applies
immediately" was false for every per_request field on a loaded model.
`requires_reload` keys deliberately stay snapshots: the reported reload is
their real cost. Pinned both ways by
`tests/unit/test_per_request_refresh.py`.

Wire contract corollary (same date): admin responses serialize `config`
with `exclude_unset` -- absent IS how a default is spelled in models.toml,
and set-vs-default must survive the wire for any UI to render it honestly.

## Configuration Management

### Loading

```python
from heylook_llm.router import ModelRouter

router = ModelRouter(config_path="models.toml")
# Loads configuration at startup; models are NOT loaded until requested
```

### Hot Reload

```bash
curl -X POST http://localhost:8000/v1/admin/reload
```

Reloads `models.toml` without restarting the server. Currently loaded models stay in cache.

### Adding models

Put the model under a `[scan].folders` watch folder; discovery serves it with
no entry. `heylookllm import` was retired in v2.0.72.

---

## Troubleshooting

### Model Not Found

```
ValueError: Model 'unknown-model' not found in configuration
```

Check `/v1/models`, and that the model sits under a `[scan].folders` watch folder.

### Validation Error

```
ValidationError: provider
  Input should be 'mlx' or 'gguf'
```

`llama_cpp` (embedded), `mlx_stt` and `mlx_embedding` (v2.0.41) providers have been removed. Valid providers: `"mlx"` (text/vision), `"gguf"` (llama-server subprocess, v1.41+).

### Model Load Failure

```
RuntimeError: Failed to load model: No such file or directory
```

Check that `model_path` points to a valid local path or valid HuggingFace model ID.

---

## Related Documentation

- [mlx_provider.md](./mlx_provider.md) -- MLXProvider details
