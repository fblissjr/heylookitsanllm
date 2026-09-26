---
paths:
  - "src/heylook_llm/providers/{llama_server_provider,gguf_describe,llama_cache_witness}.py"
  - "src/heylook_llm/{gguf_metadata,ram_fit}.py"
  - "scripts/{build_llama,gguf_probe,convert_gguf,ram_report}.py"
  - "scripts/gpu_wired_limit.sh"
  - "heylook.toml"
---

# gguf and llama-server

## Templates

- llama-server runs `--jinja` with reasoning pre-split. The provider's `template_info()` is None, which routes heylook's parsers to pass-through. Never re-parse another engine's split output.
- The template is a four-rung ladder resolved at spawn, and every spawn logs which rung won: explicit `chat_template_path` (`--chat-template-file`) > the operator override `chat_template.heylook.jinja` > a `chat_template.jinja` sidecar beside the .gguf > the template embedded in the GGUF. `use_sidecar_chat_template = false` keeps the embedded one without deleting a file from a downloaded snapshot.
- A template can change with no heylook.toml change (drop a file next to the weights). Any measurement that varies by prompt format must first establish which template each arm ran against.
- `chat_template_source` is MLX-only and does not reach this provider; it is a different mechanism with a deliberately different name.
- All template settings are `requires_reload` (llama-server takes the template at spawn). There is no per-request or per-preset form; the per-request template levers are the wire fields `thinking` and `reasoning_effort` (sent to llama-server as `chat_template_kwargs`; `/v1/messages` refuses that name itself with a 422).
- Publishers differ on the same weights (official vs unsloth templates accept different system-message shapes, and a raised jinja exception is a 500 from llama-server). Try the shape; do not assume a template is permissive.

## Context, memory, errors

- `-fa` is always passed, off unless `flash_attn` is set (owner call, 2026-09-26), on for a quantized `cache_type_v` (llama.cpp refuses it off). `effective_flash_attn` is the one decision; spawn and report both call it. `auto` is an explicit value, llama-server's device probe.
- `ctx_size` absent means `-c 0`: the model's training context, then `--fit` shrinks unset args to device memory. The engine contract carries `engine.context.length` (GGUF header, the ceiling) and `engine.context.running` (`/props` at ready, what the process got). `POST /v1/admin/models/{id}/reload` with a JSON body of `load_setting` fields (`{ctx_size, flash_attn}`, null = the default = drop the key) persists through the one config writer and then loads; unchanged, resident and nothing stale = plain load, no restart.
- Always send `max_tokens`; llama-server's default is unlimited. `-np 1` is our choice.
- Micro-batch is automatic (`n_ubatch` None): the provider sizes the model like the fit panel (`ram_fit`) and spawns `-ub 2048` only when headroom clears `ram_fit.THIN_HEADROOM_GB`, else inherits llama-server's 512, and logs which. A stored value wins both ways. `scripts/gpu_wired_limit.sh` (root-only, raises `iogpu.wired_limit_mb`) is the one lever that enlarges the working set.
- A llama-server error mid-stream is a `data: {"error":...}` frame after a 200; the adapter raises on it. "Compute error." on Metal almost always means the working set ran out, so the raised message carries headroom and the sysctl.
- KV cache stays f16 by default. `cache_type_k/v` are for headroom emergencies, not a default.
- A non-causal projector's image must fit one micro-batch or llama-server aborts. At spawn the provider reads `clip.projector_type` and passes `--image-max-tokens` = the effective micro-batch when that projector's default maximum exceeds it (never raising a default). The set is the hand-copied `NON_CAUSAL_IMAGE_PROJECTORS` table (owner call: table plus test, no build-time parser); `test_non_causal_table_matches_the_build` pins it against the built tree's source, found through the build manifest, never `coderef/`.
- The spawn sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` to `METAL_RESIDENCY_KEEP_ALIVE_S` (keep weights resident for the process life; heylook's idle unload is what ends it). It is llama.cpp's residency heartbeat, not heylook's idle unload; keep the two distinct in names and docs. Never derive it from the idle threshold (stale on a live change) and never exceed the `atomic_int` 5 ms tick range (it wraps and turns residency off). An inherited value wins, with a warning.
- `ram_fit` sizes gguf by file bytes and ignores placement fields, so it overstates GPU need for models that keep tables host-side. Owner decision: `ram_fit` is not rewired to `llama-fit-params`; the workaround is an explicit `n_ubatch`. Revisit only if expert offload starts being used or the false warning grates.

## Speculative decoding

- Spec decode is ON by default whenever a model ships a drafter (owner decision 2026-09-24, replacing "per-model opt-in"): a drafter file, or an MTP head built into the GGUF. The off switch is per model: `unset = ["draft_model_path", "spec_type"]` in the model's `model.heylook.toml`, or an explicit heylook.toml entry without those fields. A model served with a LoRA (steering included) should have it off, because the drafter never gets the adapter and spec decode is fixed when the process spawns.
- Discovery finds it, in this order, first rung wins (`ModelImporter._pick_spec`): a drafter file (named `mtp-`/`dspark-`/`dflash-`/`eagle3-`) beside the weights or, for a per-quant variant folder, at the repo root; a drafter file in an immediate subfolder (`MTP/`); an MTP head built into the weights (`spec_type = "draft-mtp"`, no file); a drafter file in a neighbouring folder whose header names the same model (`gguf_metadata.model_names`). Each pairing is logged with its reason at scan. The built-in-head test is llama.cpp's own rule (`gguf_metadata.spec_type_from_gguf`, pinned to the build by `test_spec_rule_matches_the_build`), read across every split. An explicit heylook.toml entry receives none of it; the registry sidecars (`docs/project/plan_registry_sidecars.md`) are what let it reach those models.
- A drafter gives way to fit, at spawn (`LlamaServerProvider._drop_drafter_if_short`): when the model alone fits live reclaimable RAM (ram_fit plus its headroom) and the model plus drafter does not, the spawn drops `draft_model_path`/`spec_type` and warns with both numbers (`drafter_skipped`). Live RAM, not the wired limit: llama.cpp's `--fit` sizes against the Metal working set and does not see a total-RAM shortfall. A model that does not fit alone is left to fail as before.
- A drafter the build cannot load costs one retry, never the model (`LlamaServerProvider.load_model`): when llama-server exits during load with a drafter set, the spawn is retried once without it, and a success records the (drafter, binary) pair in `_UNLOADABLE_DRAFTERS` for the life of the process. Known case at build 4e416ee73: Qwen3.8-Flash-Next's `MTP/mtp-...-shared-*.gguf` is a split-out MTP head with no `token_embd.weight`, and llama.cpp loads a `-md` file as a whole model. The permanent off switch is `unset` in the model's `model.heylook.toml`.
- `draft_model_path` is the switch, not `spec_type`. The provider emits `-md` on that field alone, and llama.cpp infers the type from the drafter's own header when `--spec-type` is absent. `spec_type` only pins the type; it is required only for a sharded drafter.
- Discovery pins `spec_type` only where llama.cpp cannot infer it from the drafter's first split: a built-in head, a sharded drafter, and eagle3. Otherwise a paired model runs spec decode at llama.cpp's defaults with `spec_type` unset. To list those models, walk `model_registry.scan(data).entries` for configs carrying `draft_model_path` or `spec_type`.
- A built-in MTP head runs with `--spec-type draft-mtp` and no `-md`: llama.cpp then loads the head from the target itself (`load_mtp` in `common/common.cpp` of the build). The provider sends `--spec-type` whenever `spec_type` is set, so that config alone reaches it.
- Measuring whether it pays on a model: never at temp 0 for a throughput claim; match prompt length, generation length, seed, sampling, binary and prompt-cache state across arms; tune `spec_draft_n_max` and `spec_draft_p_min` together; do not trust short prompts or short generations for ranking; matching controls only rules out the confounds you imagined. Spot observations are not performance testing.
- Drafter packaging: read the GGUF, not vendor docs. gemma drafters are sidecar `mtp-*.gguf` (llama's `-hf` sibling discovery does not work for local files); Qwen3.6's and Qwen3.8-27B's MTP heads are built into the GGUF. A LoRA applies to the target only, so it erodes any spec-decode win; `--spec-type` is a spawn flag while `lora` is per-request.
- Local detail and conditions: `internal/research/`.

## Binary and build

- The llama-server binary is the canonical local build, written only by `scripts/build_llama.py` (owner rule: one build, one source; update = re-run the script). `server_binary` / `$HEYLOOK_LLAMA_SERVER` are experiment escape hatches that warn at every spawn, naming the build they shadow. Load fails loudly only if no binary exists anywhere.
- `scripts/build_llama.py` is the only thing that clones or builds llama.cpp (`uv sync` cannot). It builds the newest `b<N>` release tag by default (`--rev` for anything else) and never touches pyproject or uv.lock. It also builds `llama-bench` and `llama-fit-params` from the same commit; the server calls neither. llama.cpp is not vendored and not a submodule (`dir` / `$HEYLOOK_LLAMA_CPP_DIR` relocate the tree). Build flags and their rationale: `scripts/README.md`.
- `llama-fit-params` is llama.cpp's memory projector (per-device split from metadata, no weights loaded). It is not a llama-server flag, and it refuses `--mmproj`, so a projector is costed separately.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "gguf and llama-server".
