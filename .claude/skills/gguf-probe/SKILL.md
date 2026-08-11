---
name: gguf-probe
description: Direct llama-server diagnostics for one GGUF model (no FastAPI/DB/router) -- /props modalities, thinking-template on/off diff, one-shot generation with tps + draft acceptance, LoRA adapter attach + off/on A/B, clean teardown. Use for provider debugging, new-GGUF triage, MTP A/B measurements (e.g. the open gemma-4 12B TODO), and checking whether a LoRA applies and what it costs.
---

# gguf-probe

Thin wrapper: the probe lives GIT-TRACKED at `scripts/gguf_probe.py` (read
its header for details). The layer BELOW the dev-server skill -- spawns
`LlamaServerProvider` directly, so it inherits llama.cpp flag/lifecycle
updates from the provider instead of duplicating them.

```bash
# All must run UNSANDBOXED (Metal + localhost + modelzoo traversal).

# A model DIR: sidecars (mmproj / mtp- drafter) auto-paired via the
# importer's own pickers (spec_type stays OFF unless you pass it):
uv run python scripts/gguf_probe.py modelzoo/gguf/unsloth_gemma-4-E4B-it-qat-GGUF

# MTP A/B (run once without, once with):
uv run python scripts/gguf_probe.py modelzoo/gguf/unsloth_gemma-4-12B-it-qat-GGUF --no-gen
uv run python scripts/gguf_probe.py modelzoo/gguf/unsloth_gemma-4-12B-it-qat-GGUF \
  --spec-type draft-mtp --spec-draft-n-max 4

# LoRA: attach an adapter, then A/B it off vs on in ONE process/model load.
# Vendor-recommended sampling -- `--temp 0` is REFUSED by the probe.
uv run python scripts/gguf_probe.py <base-dir> --lora <adapter>.gguf --lora-ab \
  --temp 1.0 --top-p 0.95 --top-k 64 --seed 1234
```

What it reports (and why each matters):
- `/props modalities` -- the LIVE capability truth (config `modalities` is
  description only).
- apply-template on/off/unset diff -- `unset != off` means thinking-off
  must be sent EXPLICITLY (gemma's empty-thought prefill; the bug-shaped
  behavior found 2026-07-26).
- generation tok/s + `draft_accepted/draft_tokens` from the chunk stream,
  plus the subprocess log's `draft acceptance = ...` lines (the ONE fragile
  grep if llama.cpp ever rewords that line -- everything else rides stable
  surfaces).
- with `--lora-ab`: tok/s, draft-acceptance and output deltas between adapter
  off and on, toggled over `POST /lora-adapters` so both runs share one model
  load. IDENTICAL output proves the adapter did nothing; differing output is
  only consistent with it working, so repeat at a pinned `--seed` before
  trusting either reading. NEVER measure at `--temp 0` -- the probe refuses it.
  Greedy is a DIFFERENT decoding regime (speculative acceptance becomes exact
  argmax matching instead of rejection sampling), so a conclusion drawn there
  can name the WRONG setting, not merely an imprecise one. Use each model's
  vendor-recommended sampling (unsloth.ai/docs/models/&lt;model&gt;), but VERIFY
  against the local files -- those docs have already been observed to disagree
  with what is on disk (they call gemma-4 MTP embedded and Qwen3.6 sidecar;
  locally it is the reverse). An acceptance collapse means the draft path is UNADAPTED (an
  embedded MTP head is not covered by the adapter) -- worth knowing before
  pairing a LoRA with spec decode. `--lora` rides `extra_args`, the same raw
  passthrough a models.toml gguf entry uses, so a working flag transfers
  verbatim to a server config.

Non-negotiables: one model at a time (each run spawns/kills its own
subprocess; RAM per the model's size); never point it at a long-running
server instance -- it doesn't touch existing servers, it spawns its own;
teardown is automatic via `finally`.
