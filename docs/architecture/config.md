# Configuration System

Last updated: 2026-07-20

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
default_model = "model-id"
max_loaded_models = 2

[[models]]
id = "model-identifier"       # Unique ID for API requests
provider = "mlx"              # Provider: mlx, mlx_embedding
enabled = true                # Include in /v1/models?
capabilities = ["chat"]       # Feature flags for clients

  [models.config]
  model_path = "path/to/model"
  # ... provider-specific options
```

### Top-Level Fields

#### `id` (required)
Unique identifier used in API requests. Must be unique, no spaces, case-sensitive.

#### `provider` (required)
Valid values: `"mlx"`, `"mlx_embedding"`

| Provider | Text | Vision | Embeddings | Platforms |
|---|---|---|---|---|
| `mlx` | Yes | Yes | No | macOS (Apple Silicon) |
| `mlx_embedding` | No | No | Yes | macOS (Apple Silicon) |

#### `enabled` (optional, default `true`)
Whether the model appears in `/v1/models`. Set to `false` to hide experimental models.

#### `capabilities` (optional)
Array of feature flags for client-side discovery. Common values: `"chat"`, `"vision"`, `"thinking"`, `"hidden_states"`.

#### `config` (required)
Provider-specific configuration (see sections below).

---

## MLX Provider Configuration (`provider = "mlx"`)

### Text Models

```toml
[[models]]
id = "qwen-2.5-3b"
provider = "mlx"
enabled = true

  [models.config]
  model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"
  vision = false
  max_tokens = 2048
  temperature = 0.7
  cache_type = "standard"
```

### Vision Models

```toml
[[models]]
id = "qwen-vl-7b"
provider = "mlx"
enabled = true

  [models.config]
  model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
  vision = true
  max_tokens = 1024
```

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
| `modalities` | list[str] | derived | Author-declared capability set (`text`/`vision`/`audio`/`video`); `text` always present. Detected at import from the model's `config.json` blocks (`vision_config`/`audio_config` + `*_token_id`). Absent -> derived from `vision`. **Authoritative** description of the model. |
| `loader` | `auto`\|`mlx-vlm`\|`mlx-lm` | `auto` | Engine routing (within `provider="mlx"`). `auto`: mlx-vlm iff the model declares vision AND mlx-vlm registers its `model_type`, else mlx-lm (degrades only on positive non-support). Explicit forces the engine (e.g. run a dual-capable VLM as text via `mlx-lm`). |
| `vision` | bool | `false` | **Derived mirror** of `"vision" in modalities`, retained for back-compat. Setting it seeds `modalities` when `modalities` is omitted; if both are set, `modalities` wins. Load routing goes through `loader`/`effective_loader`, not this flag. |
| `max_tokens` | int | none | Default maximum tokens to generate. Unset falls through the effective-request cascade to `GLOBAL_SAMPLER_FLOOR['max_tokens']` (4096) -- see below. |
| `temperature` | float | none | Default sampling temperature. Unset falls through to the cascade (0.7 floor). |
| `top_p` | float | none | Nucleus sampling threshold |
| `top_k` | int | none | Top-k sampling |
| `min_p` | float | none | Min-p sampling |
| `repetition_penalty` | float | none | Repetition penalty |
| `presence_penalty` | float | none | Presence penalty |
| `cache_type` | `Literal["standard", "rotating", "quantized"]` | `"standard"` | KV cache implementation. `"rotating"` requires `max_kv_size` (validated at load, not first generation) |
| `max_kv_size` | int | none | Rotating-cache size cap. **Never set by smart defaults** -- see "Smart Defaults at Import" below. |
| `kv_bits` | `Literal[2, 4, 8]` | none | KV quantization bits -- constrained to what MLX's `QuantizedKVCache` actually supports |
| `kv_group_size` | `Literal[32, 64, 128]` | `64` | KV quantization group size -- constrained to what MLX supports |
| `max_queue_depth` | int (`ge=1`) | `8` | Requests admitted behind the active generation before 503 backpressure. A real config field as of v1.32.0 -- previously read by the generation gate but not declared on `MLXModelConfig`, so it was silently dropped by Pydantic and permanently 8 regardless of `models.toml` |
| `enable_thinking` | bool | `false` | Thinking-mode default for this model (any thinking-capable template, not Qwen3-specific -- see "Sampler Defaults" below for the request-time cascade) |
| `vision_tokens` | int | none | Per-model default visual token budget per image (16-16384). A request's own `vision_tokens` overrides; `none` leaves the image processor's own default. Mapped per model family by `providers/common/vision_budget.py` (gemma-4: discrete `max_soft_tokens` bucket; qwen2/3-VL: `max_pixels`) |
| `supports_thinking` | bool | `false` | Capability metadata flag |
| `default_preset` | string | none | Preset name applied when a request doesn't specify one -- see "Sampler Defaults and the Effective-Request Cascade" below |
| `draft_model_path` | string | none | Path to draft model for speculative decoding |
| `num_draft_tokens` | int | `3` | Draft tokens for speculative decoding. The importer no longer stamps this on every import (v1.32.0) -- it's inert without `draft_model_path`, so writing it on every model was dead config. The field and its default of 3 remain; only the automatic import-time write was removed. |
| `default_hidden_layer` | int | `-2` | Layer for hidden state extraction |
| `default_max_length` | int | `512` | Max sequence length for hidden states |
| `unload_after_idle_seconds` | int, none | none | Per-model idle-unload override. `None` = use `AppConfig.idle_unload_seconds`; `0` = never idle-unload this model |
| `chat_template_source` | string, none | none | `"auto"` / `"jinja"` / `"tokenizer_config"` / absolute path -- overrides chat-template source detection |

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

# Local path (tilde expanded)
model_path = "~/models/qwen-2.5-custom"
```

### Sampler Defaults and the Effective-Request Cascade

A chat request's actual sampler values are resolved by
`MLXProvider._apply_model_defaults` (`src/heylook_llm/providers/mlx_provider.py`,
lines 758-836) as a six-layer cascade, each layer overriding only the
fields it sets:

1. **Global hardcoded floor** -- `GLOBAL_SAMPLER_FLOOR` (mlx_provider.py,
   lines 495-503): `temperature 0.7`, `top_p 1.0`, `top_k 0`, `min_p 0.0`,
   `max_tokens 4096`, `repetition_penalty 1.0`, `presence_penalty 0.0`.
   This is what a request gets when nothing else in the cascade says
   anything -- the de-facto behavior for a freshly imported model with no
   `default_preset`.
2. **Thinking-mode defaults**, applied only when the model config sets
   `enable_thinking = true` (sourced from the `thinking` preset).
3. **Model sampler fields** from `models.toml` (per-model overrides in the
   table above).
   1. **Model `default_preset`** -- applied only when the request has no
      explicit preset. Unknown preset name logs and skips the layer
      rather than failing (models are validated at server startup; an
      unknown name here means the preset registry changed post-startup).
4. **Request preset** (`ChatRequest.preset`). Overrides the model's
   `default_preset`. Unknown name raises `PresetNotFound`, translated to
   HTTP 400 by the route handler.
5. **Request-level explicit field values** -- always win.

Before v1.32.0, the global floor was `temperature 0.1, max_tokens 512` --
near-greedy sampling and 512-token truncation for every model with no
`default_preset` set, which in practice meant every CLI-imported model
(the 512 cap also existed independently in `moderate.toml` and the batch
processor's fallback). This was silently truncating and flattening output
for models the user had never explicitly tuned. `GLOBAL_SAMPLER_FLOOR` is
now `0.7 / 4096`, and the batch fallback paths (`mlx_provider.py` lines
854, 919) reference the same constant instead of a third hardcoded `512`.

Admin/CLI import also used to stamp `default_preset = "moderate"` --
`moderate.toml` (removed 2026-07-20) described itself as "the deprecated
back-compat alias for the pre-preset-split default; new users should
prefer 'balanced'." Admin import (`ModelService.import_models`,
`src/heylook_llm/model_service.py`, line 642) now defaults to
`profile_name = "balanced"` instead.

### Smart Defaults at Import

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

## MLX Embedding Provider Configuration (`provider = "mlx_embedding"`)

```toml
[[models]]
id = "embeddinggemma-300m"
provider = "mlx_embedding"
enabled = true

  [models.config]
  model_path = "/path/to/google_embeddinggemma-300m"
  max_length = 2048
```

### Embedding Config Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | required | HuggingFace model ID or local path |
| `max_length` | int | `2048` | Maximum tokenization length |

The embedding provider uses dynamic backbone loading -- it supports any architecture that mlx-lm's `_get_classes()` can resolve, not just Gemma3. Design notes live in `src/heylook_llm/models/embedding_model.py`'s docstring.

---

## Pydantic Schemas

**File**: `src/heylook_llm/config.py` (Pydantic V2, `@field_validator` / `@model_validator`)

```python
class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "mlx_embedding"]
    config: Union[MLXModelConfig, MLXEmbeddingModelConfig]
    description: Optional[str]
    tags: List[str] = []
    enabled: bool = True
    capabilities: List[str] = []

class AppConfig(BaseModel):
    models: List[ModelConfig]
    default_model: Optional[str]
    max_loaded_models: int = 2
```

The `config` field is discriminated on `provider`: `"mlx"` parses as `MLXModelConfig`, `"mlx_embedding"` parses as `MLXEmbeddingModelConfig`.

---

## Model Profiles

> **This section describes the current (post-"C4") preset system.** It
> supersedes an older load-time "profiles" design that baked sampler
> fields directly into `models.toml` at import; that design, and the
> `src/heylook_llm/data/profiles/` directory this doc used to point at,
> no longer exist. This drift predates and is unrelated to the
> 2026-07-05/06 work in this document's other sections -- noted here
> because it was found while verifying this file against the code.

Presets are named sampler-field bundles resolved at **request time**, not
baked into `models.toml` at import. They live under
`src/heylook_llm/data/presets/` as TOML files, loaded by the
`PresetRegistry` (`src/heylook_llm/presets.py`). Current presets:
`balanced` (import-default profile), `deterministic` (repro/eval),
`thinking` (auto-applied by the cascade for `enable_thinking` models),
`vlm-describe` / `vlm-extract` (VLM-safe field subsets -- mlx-vlm's
`stream_generate` ignores top_k/min_p/repetition_penalty; used by
batch-labeler's tasks).

The flavor presets `moderate` (back-compat alias for the pre-preset-split
default), `code`, and `creative` were removed 2026-07-20: they had no
consumer anywhere in the stack (the v3 frontend's user-preset system owns
interactive sampler preferences), and their only references were tests
asserting their own existence. The registry keeps only presets that encode
mechanism (model-family or library knowledge) or are wired as defaults.
`test_preset_registry.py` pins the exact roster -- adding a preset means
naming its consumer there.

**Discovery** (2026-07-20): `GET /v1/capabilities` advertises the registry
(`sampler_presets: {available, request_field, model_default_field}`) so
scripted clients can enumerate names without admin access. The admin list
lives at `GET /v1/admin/models/sampler-presets`.

**Terminology** (unified 2026-07-20): "sampler preset" is the single term;
the import/admin paths historically said "profile" for the same registry.
Renames: routes `/v1/admin/models/profiles` -> `/sampler-presets` and
`/bulk-profile` -> `/bulk-default-preset` (body field `profile` ->
`preset`); `ModelImportRequest.profile` -> `default_preset`;
`model_service.py` symbols (`ModelProfile` -> `SamplerPreset`,
`load_profiles` -> `load_sampler_presets`, `get_profiles` ->
`get_sampler_presets`, `bulk_apply_profile` -> `bulk_set_default_preset`,
`get_available_profiles` -> `available_sampler_presets`). The import CLI's
`--profile` survives only as an alias for `--preset`. Do not reintroduce
"profile" for this concept -- it collides with `/v1/performance/profile`
(perf) and invites confusion with `/v1/presets` (user presets, DuckDB).

Each preset has `[meta]` and `[defaults]` tables:

```toml
[meta]
name = "balanced"
description = "Middle-ground sampling for everyday chat. Works for most non-specialized workloads."

[defaults]
temperature = 0.7
top_p = 0.9
top_k = 40
min_p = 0.05
max_tokens = 1024
repetition_penalty = 1.05
```

`ModelService.import_models` (or `heylookllm import --preset NAME`)
records the preset name on the model as `default_preset` -- it does not
copy the preset's fields into `models.toml`. See "Sampler Defaults and
the Effective-Request Cascade" above for where `default_preset` sits in
the resolution order, and how it interacts with a request's own
`ChatRequest.preset`.

Apply a preset at import via CLI (`--profile` is accepted as an alias for
`--preset`):
```bash
heylookllm import --hf-cache --preset balanced
```

---

## Configuration Management

### Loading

```python
from heylook_llm.router import ModelRouter

router = ModelRouter(config_path="models.toml")
# Loads configuration at startup; models are NOT loaded until requested
```

### Hot Reload

```bash
curl -X POST http://localhost:8080/v1/admin/reload
```

Reloads `models.toml` without restarting the server. Currently loaded models stay in cache.

### Model Importer

```bash
# Scan HuggingFace cache and generate models.toml
heylookllm import --hf-cache --preset balanced

# Scan a directory
heylookllm import --folder ~/models --output models.toml

# Interactive mode
heylookllm import --interactive
```

`ModelImporter._get_model_size` (`src/heylook_llm/model_importer.py`,
lines 239-270) returns two independent values that must not be conflated:
a human-facing **label** parsed from the model directory name (e.g. `7B`,
`4bit`), and the actual **`size_gb`** fed to `get_smart_defaults`. Before
v1.32.0, `size_gb` also came from the name regex -- `Qwen-7B` produced
`size_gb = 7.0`, which is 7 billion *parameters*, not 7 *gigabytes*, and a
`-4bit` suffix produced `size_gb = 4.0` the same way. Feeding a
params-count into a RAM-relative GB threshold (see "Smart Defaults at
Import" above) is a straightforward unit error. `size_gb` now always
comes from the safetensors byte-sum on disk (`sum(f.stat().st_size for f
in path.rglob("*.safetensors")) / 1024**3`), matching the admin scan path
(`ModelService._raw_to_scanned`) that already did this correctly. The
name regex now only supplies the label, and only matches the model
**directory** name -- matching the full path let size-looking fragments
in parent directories (e.g. a temp dir containing `680b`) win.

---

## Troubleshooting

### Model Not Found

```
ValueError: Model 'unknown-model' not found in configuration
```

Check `/v1/models` or run `heylookllm import` to regenerate `models.toml`.

### Validation Error

```
ValidationError: provider
  Input should be 'mlx' or 'mlx_embedding'
```

`llama_cpp` and `mlx_stt` providers have been removed. Update `models.toml` to use `"mlx"` for text/vision or `"mlx_embedding"` for sentence-transformer models.

### Model Load Failure

```
RuntimeError: Failed to load model: No such file or directory
```

Check that `model_path` points to a valid local path or valid HuggingFace model ID.

---

## Related Documentation

- [mlx_provider.md](./mlx_provider.md) -- MLXProvider details
