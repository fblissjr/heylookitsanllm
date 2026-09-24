---
paths:
  - "src/heylook_llm/providers/{llama_server_provider,gguf_describe,llama_cache_witness}.py"
  - "src/heylook_llm/{gguf_metadata,ram_fit}.py"
  - "scripts/{build_llama,gguf_probe,convert_gguf,ram_report}.py"
  - "scripts/gpu_wired_limit.sh"
  - "models.toml"
---

# gguf and llama-server

## Templates

- llama-server runs `--jinja` with reasoning pre-split. The provider's `template_info()` is None, which routes heylook's parsers to pass-through. Never re-parse another engine's split output.
- The template is a four-rung ladder resolved at spawn, and every spawn logs which rung won: explicit `chat_template_path` (`--chat-template-file`) > the operator override `chat_template.heylook.jinja` > a `chat_template.jinja` sidecar beside the .gguf > the template embedded in the GGUF. `use_sidecar_chat_template = false` keeps the embedded one without deleting a file from a downloaded snapshot.
- A template can change with no models.toml change (drop a file next to the weights). Any measurement that varies by prompt format must first establish which template each arm ran against.
- `chat_template_source` is MLX-only and does not reach this provider; it is a different mechanism with a deliberately different name.
- All template settings are `requires_reload` (llama-server takes the template at spawn). There is no per-request or per-preset form; the per-request template levers are the wire fields `thinking` and `reasoning_effort` (sent to llama-server as `chat_template_kwargs`; `/v1/messages` refuses that name itself with a 422).
- Publishers differ on the same weights (official vs unsloth templates accept different system-message shapes, and a raised jinja exception is a 500 from llama-server). Try the shape; do not assume a template is permissive.

## Context, memory, errors

- `ctx_size` absent means `-c 0`: the model's training context, then `--fit` shrinks unset args to device memory. The engine contract carries `engine.context.length` (GGUF header, the ceiling) and `engine.context.running` (`/props` at ready, what the process got). `POST /v1/admin/models/{id}/reload?ctx_size=N` persists through the one config writer and then loads (0 = Auto = drop the key; unchanged and resident = plain load, no restart).
- Always send `max_tokens`; llama-server's default is unlimited. `-np 1` is our choice.
- Micro-batch is automatic (`n_ubatch` None): the provider sizes the model like the fit panel (`ram_fit`) and spawns `-ub 2048` only when headroom clears `ram_fit.THIN_HEADROOM_GB`, else inherits llama-server's 512, and logs which. A stored value wins both ways. `scripts/gpu_wired_limit.sh` (root-only, raises `iogpu.wired_limit_mb`) is the one lever that enlarges the working set.
- A llama-server error mid-stream is a `data: {"error":...}` frame after a 200; the adapter raises on it. "Compute error." on Metal almost always means the working set ran out, so the raised message carries headroom and the sysctl.
- KV cache stays f16 by default. `cache_type_k/v` are for headroom emergencies, not a default.
- A non-causal projector's image must fit one micro-batch or llama-server aborts. At spawn the provider reads `clip.projector_type` and passes `--image-max-tokens` = the effective micro-batch when that projector's default maximum exceeds it (never raising a default). The set is the hand-copied `NON_CAUSAL_IMAGE_PROJECTORS` table (owner call: table plus test, no build-time parser); `test_non_causal_table_matches_the_build` pins it against the built tree's source, found through the build manifest, never `coderef/`.
- The spawn sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` to `METAL_RESIDENCY_KEEP_ALIVE_S` (keep weights resident for the process life; heylook's idle unload is what ends it). It is llama.cpp's residency heartbeat, not heylook's idle unload; keep the two distinct in names and docs. Never derive it from the idle threshold (stale on a live change) and never exceed the `atomic_int` 5 ms tick range (it wraps and turns residency off). An inherited value wins, with a warning.
- `ram_fit` sizes gguf by file bytes and ignores placement fields, so it overstates GPU need for models that keep tables host-side. Owner decision: `ram_fit` is not rewired to `llama-fit-params`; the workaround is an explicit `n_ubatch`. Revisit only if expert offload starts being used or the false warning grates.

## Speculative decoding

- Spec decode is per-model opt-in and stays off for a new model unless you have checked it is a win on your model at your context. It is off because it is unproven here, not because it is known harmful.
- Carve-out, do not undo it: the owner decided the DeepSeek V4 Flash entry keeps spec decode on. "Default off" means do not enable it elsewhere without checking; it never means disable what is already running.
- `draft_model_path` is the switch, not `spec_type`. The provider emits `-md` on that field alone, and llama.cpp infers the type from the drafter's own header when `--spec-type` is absent. `spec_type` only pins the type; it is required only for a sharded drafter.
- Discovery auto-pairs `draft_model_path` for a sidecar named `mtp-`/`dspark-`/`dflash-`/`eagle3-` and leaves `spec_type` unset, so such a model runs spec decode at llama.cpp's defaults with no models.toml entry. To find those models, walk `merge_discovered(data, discover(data))` for configs carrying `draft_model_path`. An embedded MTP head (Qwen3.6) pairs no sidecar and gets no spec decode. To keep spec decode off, the drafter must not be paired.
- Checking it yourself: never at temp 0 for a throughput claim; match prompt length, generation length, seed, sampling, binary and prompt-cache state across arms; tune `spec_draft_n_max` and `spec_draft_p_min` together; do not trust short prompts or short generations for ranking; matching controls only rules out the confounds you imagined. Spot observations are not performance testing.
- Drafter packaging: read the GGUF, not vendor docs. gemma drafters are sidecar `mtp-*.gguf` (llama's `-hf` sibling discovery does not work for local files); Qwen3.6's MTP is embedded. A LoRA applies to the target only, so it erodes any spec-decode win; `--spec-type` is a spawn flag while `lora` is per-request.
- Local detail and conditions: `internal/research/`.

## Binary and build

- The llama-server binary is the canonical local build, written only by `scripts/build_llama.py` (owner rule: one build, one source; update = re-run the script). `server_binary` / `$HEYLOOK_LLAMA_SERVER` are experiment escape hatches that warn at every spawn, naming the build they shadow. Load fails loudly only if no binary exists anywhere.
- `scripts/build_llama.py` is the only thing that clones or builds llama.cpp (`uv sync` cannot). It builds the newest `b<N>` release tag by default (`--rev` for anything else) and never touches pyproject or uv.lock. It also builds `llama-bench` and `llama-fit-params` from the same commit; the server calls neither. llama.cpp is not vendored and not a submodule (`dir` / `$HEYLOOK_LLAMA_CPP_DIR` relocate the tree). Build flags and their rationale: `scripts/README.md`.
- `llama-fit-params` is llama.cpp's memory projector (per-device split from metadata, no weights loaded). It is not a llama-server flag, and it refuses `--mmproj`, so a projector is costed separately.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "gguf and llama-server".
