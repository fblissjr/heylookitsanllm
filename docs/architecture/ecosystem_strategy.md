# MLX ecosystem strategy — context for performance work

Last updated: 2026-07-06

Audience: Claude sessions building or refactoring heylookitsanllm, especially
performance work (plan_2026-07.md Phase 5 and beyond). This doc explains the
POSTURE toward mlx / mlx-lm / mlx-vlm and why. Read it before proposing
upstream workarounds, backend changes, or perf investments.

Facts below were verified 2026-07-06 (release pages, PyPI feeds, commit logs,
primary announcements). Staleness warning: re-verify anything load-bearing
after ~2026-09.

## Overview

The distilled message to any session working on performance right now:

**Posture: the Python MLX stack is the right rail, but treat it as maintenance-mode infrastructure, not a living upstream.** mlx-lm is release-starved (20+ unreleased commits, 259 open PRs, HF quietly co-maintaining) while Apple's feature frontier ships in mlx-swift-lm. So the operating rules are: SHA-pin rather than wait for PyPI releases, check the open-PR backlog before writing any workaround, and expect new capabilities (spec decoding especially) to arrive via sidecar packages like dflash-mlx — never via an mlx-lm release.

**For perf work specifically, the ordering logic is:** fix measurement first (the recorded numbers are quantized and queue-inflated — optimizing against them is theater), then the dormant draft-model path (biggest lossless win, already wired, text-only), then prefix-cache hit-rate (attacks MLX's real weakness on our prefix-heavy workloads), then memory-budget admission and batched decode (the 192GB box is underscheduled). Everything else stays evidence-gated. And keep the two regimes separate: the VLM pre-filled-cache path is incompatible with both radix and speculative prefill, so vision perf lives in the feature cache, not decode tricks.

**The one invariant worth protecting above all: `run_generation()` as the single chokepoint and the provider seam in config.py.** That seam is why none of this ecosystem turbulence forces a rewrite — it converts backend risk into a config entry, whether the future is a DFlash sidecar, a Swift shim, or llama-server for coverage gaps.

The doc also carries the tripwires (next `mlx` core wheel is the load-bearing signal; day-one support location for the next frontier model is the diagnostic one) and the alternatives-considered section so the "why not Swift/llama.cpp/fork" reasoning doesn't get relitigated every session.

## Verified ecosystem state (as of 2026-07-06)

- **MLX core is healthy and strategically backed.** Ollama switched its Apple
  Silicon engine to MLX (2026-03-30). WWDC26 shipped `MLXLanguageModel` in the
  Foundation Models framework (open-sourcing "later this summer"), Metal 4
  support, RDMA-over-Thunderbolt. M5 GPU Neural Accelerators are hardware
  co-designed for MLX compute patterns (~4x TTFT vs M4 — note: does NOT reach
  our M2 Ultra). NVIDIA contributes NVFP4 + CUDA-backend work.
- **Awni Hannun left Apple 2026-02-27 (→ Anthropic).** Team remains:
  angeloskath, davidkoski, jagrit06, Cheng Zhao, Ronan Collobert.
- **mlx-lm (Python) is in maintenance mode, not abandoned.** Last release
  v0.31.3 (2026-04-22); no release since despite ~20 commits on main. May 2026
  had ONE commit; June a trickle (CVE-2026-5843 fix, Qwen3.5 pipelining,
  Mellum 2, server/tool-call fixes, `return_logprobs` in `batch_generate`).
  259 open PRs / 185 open issues = maintainer bottleneck, not dead community.
  pcuenca (Hugging Face) now commits — HF is informally co-maintaining.
- **Apple's LM feature frontier moved to mlx-swift-lm.** 3.31.4 (released
  ~2026-06-30, davidkoski) merged ~90 PRs in the same window: batched
  inference, ChatSession speculative decoding + telemetry, Gemma 4 MTP
  speculative decoding, ParoQuant, extensible KV-cache compression (kvScheme),
  pipelined prefill w/ asyncEval, runtime LoRA, Swift model conversion API,
  deep in-repo VLM work (MLXVLM). New capabilities now debut Swift-first.
- **mlx-vlm (Blaizzy, community) is active but single-maintainer.** v0.6.x
  cadence through June 2026 (agent APIs, spec decoding, new VLM archs).
  Same structural position mlx-lm is now drifting toward.
- **Speculative decoding is the current capability wave.** DFlash (block-
  diffusion drafting, arXiv 2602.06036, ~4x lossless on MLX ports) and DSpark
  (DeepSeek, 2026-06-27, 60-85% faster, byte-identical) converged on:
  cheap parallel draft + target verification, draft models shipping as
  standard release artifacts. It reaches Python via SIDECAR packages
  (dflash-mlx) — the mlx-lm commit log says upstream won't ship it soon.

## Strategic posture (the "why" behind the rules)

1. **Stay on Python MLX. No pivot to llama.cpp/GGUF, no Swift rewrite.**
   Decode speed, memory efficiency, and hardware trajectory favor MLX on
   Apple Silicon; the Python bindings are first-party and carry the largest
   downstream ecosystem (LM Studio, vLLM-MLX, dflash-mlx, mlx-community).
   Our orchestration layer (FastAPI/SQLite/RLM/router) is compute-agnostic
   glue; a rewrite buys nothing and burns the encoded correctness knowledge
   in the postmortems and our invariants.
2. **The engine's identity is a single-box inference SCHEDULER, not a model
   runner.** Bare "OpenAI endpoint over mlx-lm" is commodity (8+ competitors,
   Ollama, first-party Apple path). Our differentiation: router/admission,
   radix + vision-feature caches, observability, RLM, multi-client serving
   (ComfyUI batch, scripts, frontend). Perf work should compound THAT, not
   re-derive what upstream does.
3. **Treat mlx-lm and mlx-vlm as pinned, contract-tested, cherry-pickable
   dependencies.** Release starvation means PyPI pins fall behind real fixes.
   Supported responses, in order: (a) SHA-pin via
   `mlx-lm @ git+https://github.com/ml-explore/mlx-lm@<sha>` to a vetted
   commit; (b) cherry-pick from the 259 open PRs (check there BEFORE writing
   any workaround — the fix likely exists unmerged); (c) local patch, documented,
   with an upstream issue filed. Never silently diverge.
4. **The provider seam is the hedge; preserve it.** `generation_core.
   run_generation()` as the single chokepoint and the provider Literal in
   config.py are what make backends swappable. Planned option purchase (not
   migration): a weekend Swift sidecar spike wrapping mlx-swift-lm ChatSession
   behind the provider HTTP contract, to validate the seam while it's cheap.
   The llama-server subprocess provider (TODO P2) is the other exit.
5. **Contribute upstream where we hold unique knowledge.** Our VLM/cache
   postmortems are integration knowledge single-maintainer projects lack.
   Filing distilled issues (incl. the CompilerCache TLS/GIL bug) buys early
   warning and influence — cheap bus-factor insurance.

## Rules for performance work specifically

- **Measurement before optimization — hard gate.** Recorded perf numbers are
  untrustworthy until plan Phase 1 lands: streaming tok/s quantized by the
  100ms poll loop, headline numbers include queue-wait, mlx-lm's native
  per-chunk `prompt_tps`/`generation_tps` never read. Do not cite, compare,
  or optimize against current numbers. Fix the instruments first.
- **Perf investment ranking (why in this order):**
  1. *Speculative decoding via the dormant draft path.* `draft_model_path` /
     `draft_tuner` are wired but unused. Classic draft first (baseline on OUR
     hardware), then DFlash via dflash-mlx sidecar behind per-model config.
     Biggest available decode win (1.5-4x, lossless); text-only — the VLM
     pre-filled-cache pattern is incompatible with speculative prefill.
  2. *Prefix/radix cache hit-rate.* Prefill is MLX's documented weakness and
     our workloads (agents, RLM, shared system prompts) are prefix-heavy.
     Extends existing machinery; SSD persistence tier only with evidence of
     re-prefill pain from request_events.jsonl.
  3. *Memory-budget admission + batched decode.* 192GB box, max_loaded_models
     is count-based; evolve to fits-in-budget. Batched decode for homogeneous
     same-model traffic (ComfyUI captioning). Note upstream `batch_generate`
     on main now returns logprobs/token_ids (unreleased — SHA pin to use it).
  4. *Everything else* (prefill_step_size tuning, quantized-cache radix) is
     evidence-gated per plan Phase 5.
- **Do NOT reimplement mlx-swift-lm features in Python** (ParoQuant, kvScheme,
  MTP heads). If a Swift-only capability becomes load-bearing, that's the
  sidecar spike's job — wrap, don't port.
- **Verify a library is actually broken before working around it** (standing
  rule). Now with the corollary: check open PRs and recent main commits first;
  maintenance-mode repos accumulate fixes without releases.
- **Vision perf is a different regime.** Vision tower cost is attacked by the
  vision feature cache (encode_image / cached_image_features), NOT by decode
  tricks. Spec decoding and radix don't apply to the pre-filled-cache VLM path
  (radix bypassed, spec prefill incompatible). Don't conflate the two paths
  when reading benchmarks.

## Tripwires / watch items

- **Next `mlx` core wheel on PyPI** — the load-bearing signal. Core bindings
  silent since 0.31.2 (2026-04-22). Resumption (esp. with the Foundation
  Models open-sourcing) = foundation fine, only the LM-zoo layer moved.
  Silence into autumn = escalate: schedule the sidecar spike immediately and
  re-evaluate posture.
- **Day-one support for the next frontier open model** (next Qwen/Llama/
  DeepSeek): if official support lands in mlx-swift-lm while mlx-lm gets only
  an unmerged community PR, the Swift-first inversion is fully confirmed.
- **mlx-lm PR backlog** (259 as of 2026-07-06): draining = co-maintenance
  materialized; growing = plan around SHA pins permanently.
- **mlx-vlm bus factor**: watch for VLM support being absorbed into official
  Python packages, or Blaizzy gaining co-maintainers. If the project stalls,
  contingency = vendor the specific model defs we use (self-contained MLX
  modules) or route vision via the llama-server provider.
- **Draft models as release artifacts**: when an official draft/MTP head ships
  with a model we run, wire it into the draft path that week.

## Alternatives considered (and why not)

- **Pivot to llama.cpp**: loses 1.4-3x decode + memory efficiency on our
  hardware and the direction of hardware co-design; keep as narrow provider
  option for coverage gaps only.
- **Swift rewrite**: ~6-12 solo months, discards Python-binding scar tissue,
  targets the echo of the Python model zoo for models we run today. Sidecar
  provider achieves the same optionality for ~a weekend.
- **Fork mlx-lm/mlx-vlm**: maintenance cost of a fork exceeds SHA-pin +
  cherry-pick posture; forking is a last resort if upstream goes truly dark.
- **Wait for upstream to ship spec decoding**: contradicted by observed
  commit flow; sidecar packages are the delivery mechanism now.
