# Plan: runtime visibility and one behaviour across engines

last updated: 2026-09-23 (APPROVED by the owner; W8 + W9 shipped v2.0.70)

## Context

The 2026-09-23 audit ([testing/gguf_runtime_audit_2026-09-23.md](../testing/gguf_runtime_audit_2026-09-23.md))
set out to check that llama-server is built and spawned optimally for vision
and thinking models. The build and spawn were close to right. What it found
instead was a set of things that were **happening without anyone being able
to see them**:

- **A hand-placed chat template silently cost one model its multi-turn prompt
  cache for weeks.** Every turn re-encoded the previous image and
  re-processed the previous reply. It was fixed the same day. Nothing in the
  product would ever have shown it.
- **The thinking-depth dropdown sends values some models 500 on and others
  ignore.** It cannot reach the strongest setting of one family, or the
  depth controls of two others.
- **`models.toml` entries freeze a full copy of their derived config.** Any
  later improvement to discovery never reaches them.
- **Cache hits and misses and the prompt-cache RAM budget are captured and
  then dropped** before any client sees them. Of speculative decoding, only
  a ratio reaches the wire, and it means different things on the two engines;
  the raw counts are dropped.
- **The engines differ in what they cache**, and the differences are
  invisible:
  - On MLX a conversation with an image anywhere in its history gets no
    cross-request cache at all, for every family.
  - Adding one image re-encodes every earlier image.
  - Qwen hybrid models get no reuse even on text, because of the current
    cache design, not because of anything in MLX.

Session postmortem (local, gitignored):
`internal/postmortems/2026-09-23_session_gguf-runtime-audit.md`.

This plan is the follow-up. Its governing rule, from the owner:
**`models.toml` must not become a black box of settings nobody knows are in
force.** Every effective setting has to be explainable from the UI: what it
is, where it came from, and what it would be if left alone.

## Principles

1. **Derived by default, override by exception.** A new setting defaults to a
   derived "auto" that is never written to disk. Only an explicit choice is
   written. Every override is shown beside its derived value.
2. **Derive, never hand-copy.** Thinking values come from the in-force
   template. Image limits come from the engine (probe) or the processor
   config. Cache limits come from the running process. A hand-maintained
   table of any of these is a defect with a delay (CLAUDE.md).
3. **Report per engine, per model, never per family.** The same model family
   behaves differently on MLX and llama.cpp (image token caps, rounding,
   cache mechanics).
4. **Null means unknown, 0 means measured zero.** Never collapse the two in a
   report.
5. **Disclose cost; gate only loss** (existing owner rule). For example,
   changing depth mid-conversation costs a full re-process on some templates.
   Say so; do not confirm.

## Workstreams

### W0. Registry sidecars + per-field provenance (foundation)

The structural fix is already planned and owner-approved:
[plan_registry_sidecars.md](./plan_registry_sidecars.md) (2026-09-08).

- Per-model entries leave `models.toml`.
- A model's own settings live in a sidecar file in its directory, layered over
  the derived defaults.
- It is gated on Phase 0's served-set diff.

That plan considered per-field overlays inside `models.toml` and rejected them
(overlay makes best-effort discovery load-bearing). **This workstream is that
plan, executed, and does not re-decide it.** The one thing this plan adds on
top is making the result explainable from the UI:

- **Provenance on the admin row.** Each effective field carries a source:
  - `derived:<what>` (GGUF header, template, projector, processor config);
  - `override` (the sidecar's value, with the derived value alongside);
  - `engine_default`.

  An override that contradicts a derivation is flagged.
- The models page renders provenance generically, off the same schema-driven
  editor (`model-config.js`).

It comes first because W1, W2, W4 and W6 all add derived settings, which
today's design would freeze into every existing entry.

### W1. Load settings: flash attention and a real load panel

- **Backend.**
  - A `flash_attn` field (`auto|on|off`; unset = auto, never written),
    `requires_reload`, arg `-fa`.
  - Tag each field that belongs in a load panel (`json_schema_extra`
    `"load_setting": true`) rather than listing them.
- **Reload route.** `POST /v1/admin/models/{id}/reload` takes a JSON body
  of any load-setting fields, persisted through the one config writer. It is
  the same shape as today's `ctx_size` query parameter, which becomes one
  field of it.
- **Frontend.** A load panel beside the model select in chat, and on the
  models page, generated from the tagged fields. For each field it shows:
  - the value that will be used and its provenance;
  - the auto value;
  - a reset.

  The context-size select moves into it. The status bar keeps the live
  pre-first-token status it has today.
- **Default: auto, not off.** Forcing off made the vision encode markedly
  slower and nothing faster. The field exists so a new architecture can be
  tested, not to change the default.

### W2. Thinking controls: detect from the template, show its own values

**Owner decision (2026-09-23): no hardcoded thinking levels.** The controls
show what the in-force template offers, in its own spellings. There is no
heylook scale, no step-to-value mapping and no clamping, and no value is
translated between models. Detail and reasoning:
`internal/claude/w2/preset_portability.md`.

- **Detection (both engines) is the only source of values.** Render the
  in-force template once per candidate value (the template's own string
  literals plus a small fixed set) and group by output. It recovers:
  - the thinking switch variable, or none;
  - the depth variable (`reasoning_effort`, `reasoning_strength`,
    `thinking_mode`, ...);
  - the distinct values, in the template's own spellings, with aliases
    grouped under one spelling;
  - the default (the group matching the absent render);
  - strictness (garbage raises, is ignored, or is pasted in);
  - **where depth enters the prompt**: the character offset of the first
    divergence between two depth renders (`changes_prefix`). Early means a
    mid-conversation change re-processes everything.

  Cache the result by template-body hash, on the template that WILL be used
  (`chat_template_files.view`, both ladders including the override), at
  row-derivation time. Never write it to `models.toml`.
- **API: it fills W13's `engine.thinking` slot**, not a separate top-level
  block:

      "thinking": {
        "switch": "enable_thinking" | null,
        "depth": {
          "variable": "reasoning_effort",
          "values": ["low", "medium", "xhigh"],   // the template's own spellings
          "default": "xhigh",
          "strict": true,
          "changes_prefix": true
        } | null
      }

  `thinking_default` / `sampler_defaults` stay the cascade's own answer; the
  depth default is the model's own value. The `reasoning_effort` capability
  becomes `engine.thinking.depth != null`.
- **Removed:** the hand-copied `ReasoningEffort` Literal and
  `PARAM_META.reasoning_effort.options` in `settings.js`. No list of levels
  exists anywhere but detection.
- **The request carries one depth field; the server supplies the variable.**
  The wire field stays `reasoning_effort`, as a bounded string. Both providers
  emit `template_kwargs[depth.variable] = value`, which makes Muse
  (`reasoning_strength`) and MiniMax (`thinking_mode`) depth reachable with no
  new field. The MLX rule that it never rides `base_kwargs` carries over to
  whatever the variable is named.
- **Validation lives in one place: the model's own list.**
  `check_thinking_depth(value, depth)` runs inside
  `samplers.resolve_effective_sampling`, covering the request value and a
  models.toml per-model default alike. A detected value or alias passes;
  anything else raises `InvalidGenerationRequest` naming the model's values,
  a 400 instead of llama-server's 500. Both routes pre-check the request value
  from W13's static half, so the 400 comes before the 200, for unloaded models
  too. An invalid stored default is a W13 settings problem, never a startup
  failure, and is not sent.
- **Presets store the value they were saved with.** A value means what that
  template says it means, and there is no honest cross-model equivalence.
  `samplerParams(caps)` extends its existing capability drop to the VALUE: a
  depth the current model does not offer is left off the request, so the
  model runs at its own default. The panel cache keeps it, and switching back
  restores it. `valid()` accepts any bounded string, since validity is per
  model; legacy stored values need no migration. Presets keep saving system
  prompts; nothing here touches `presets.system_prompt`.
- **Disclosure, no confirm** (nothing is lost). The dropdown is "auto
  (<default>)" plus `depth.values` in template order; a stored value the model
  does not offer shows as a disabled "xhigh (not offered by this model)". The
  preset status line on Apply names the default used instead. A thinking
  toggle appears only when `switch` is non-null; if a template offers `off` as
  a depth value, it is just a value and the template decides. When
  `changes_prefix` is true, a mid-conversation change is disclosed
  ("re-processes the conversation").
- **Be honest about depth.** The template's words are instructions, not
  guarantees: measured on three models, levels above low overlapped run to
  run (the audit, §6). W7's budget is the control that actually bounds length.
- `supports_thinking` for gguf is derived from the in-force template, not read
  once at import from the embedded one.
- **Capability inference moves here from `capabilities.py`** (W13 deferred it
  to W2, which changes its answers): the per-engine capability branches move
  into the engine describers as part of this workstream.
- **Checks (lean):** one property test over descriptors detected from
  templates on disk (fixtures copied into tests): every detected value and
  alias passes, a value from another model's list or an unknown string
  raises. One route test through `/v1/messages`: an unoffered value is a 400
  before any stream. One e2e check: apply a preset carrying a depth value,
  switch to a model that does not offer it, and confirm the request omits it,
  the disclosure shows, and switching back restores it (prompt preview).
  Model pair chosen when W2 starts; the models with real depth vocabularies
  are large.
- **Aliases (owner, 2026-09-23):** the server accepts every spelling the
  template accepts; the dropdown lists one option per distinct render, in the
  template's own spelling; nothing is added beyond what detection finds.
- **Not in scope: editing templates to add levels.** A level is prompt text
  the model was trained on; an invented level is untrained text.

### W3. Templates: every copy visible, copy-to-override, lint

Extends the existing template panel (`GET/PUT/DELETE
/v1/admin/models/{id}/chat-template`).

- **Sources list.** Every copy present, with origin and provenance:
  - embedded in the GGUF;
  - `chat_template.jinja`;
  - `tokenizer_config.json`;
  - `chat_template.json`;
  - the override;
  - an explicit path.

  Each is marked downloaded (repo@commit from the download metadata), modified
  since download, hand-placed, or embedded, with a hash. The in-force hash goes
  into the spawn log line.
- **Copy to override.** Start an override from any listed source, then edit.
  The existing validate-then-atomic-write path does the rest.
- **Prefix-stability lint.** Render turn 1 with the generation prompt, then
  the same conversation plus the assistant reply and a new user turn, and check
  the history prefix is stable. Run it on every source and on every override
  write. It is a warning, not a refusal. This is the check that would have
  caught the defect in the audit.
- The W2 detection results are shown beside the in-force template.
- **Owner op:** move the five in-place-edited MLX gemma templates into
  overrides, so a re-download cannot revert them.

### W4. Image geometry on the API

- **`vision.image_geometry` per model**, on `/v1/models` and the admin row:

      "image_geometry": {
        "engine": "llama.cpp" | "mlx-vlm",
        "unit_px": 32, "patch_px": 16, "merge": 2,
        "min_tokens": 8, "max_tokens": 4096,
        "min_pixels": ..., "max_pixels": ...,
        "fit": "pad" | "stretch", "rounding": "half_up" | "half_even",
        "resample": "bicubic", "max_aspect": null | 8,
        "tokens_per_image_overhead": 2,
        "rule": "smart_resize" | "budget_fill" | "grid_aspect" | "dsv4_block",
        "provenance": {"unit_px": "mmproj header", "max_tokens": "probed", ...},
        "verified": true
      }

  - **gguf.** The unit comes from the mmproj header. min/max are probed at
    spawn-ready through llama-server's
    `POST /v1/chat/completions/input_tokens` with synthetic solid PNGs (it
    runs no vision encode). The result is cached on disk keyed by
    (binary version, mmproj hash), so an unloaded model still answers after
    its first load. The probe picks up any `--image-*-tokens` override
    automatically.
  - **MLX.** From `preprocessor_config.json` / `processor_config.json`,
    merged as mlx-vlm merges them.
- **`POST /v1/models/{id}/image-plan`** with `{sizes: [[w, h], ...]}` returns
  the target size and token count per image. On gguf the count is exact from
  llama-server; the resize rule is a replica, self-checked against probes, with
  `verified: false` on mismatch.
- **Frontend.** Resize at send time to the planned size for the CURRENT model:
  one resample, no pad bars, no wasted pixels. The fixed long-edge cap goes. A
  staged original is kept until send, so a model switch can re-plan instead of
  upscaling a shrunken blob.
- It cannot know the lead pad DeepSeek adds per image (0-3 tokens, it depends
  on the preceding text), or whether more tokens help quality.

### W5. Cache and speculative reporting, end to end

- **Per request.** `GenerationChunk` gains slotted `cache: CacheReport` and
  `spec: SpecReport` fields, which `ChunkTelemetry.absorb` latches on
  not-None.
  - `CacheReport`: `prompt_tokens`, `cached_tokens`, `processed_tokens`,
    `prefill_ms`, `outcome` (hit|partial|miss|ineligible), `miss_reason`
    (cold, evicted, no_common_prefix, trim_refused, vision_path, mrope,
    template_diverged, budget_skipped, unknown), `source`, `image_features`
    {reused, encoded}.
  - `SpecReport`: `active`, `kind`, `drafted`, `accepted`, `emitted`,
    `acceptance_rate` (only when drafted is known), `draft_share`,
    `disabled_reason`.
  - Today one `draft_acceptance` name carries two different quantities on the
    two engines; it is split.
- **Wire.** `performance.cache` / `performance.speculative` on
  `message_stop`. `heylook_saved.timing` is built by the same builder instead
  of a second copy.
  - Whether to add Anthropic's `usage.cache_read_input_tokens` is an open
    decision. Anthropic's `input_tokens` excludes cached tokens; heylook's does
    not. Either align, or list it as a deliberate difference in
    `docs/api_integration.md`.
- **gguf "why".**
  - llama-server reports only the final `cache_n`, not whether it came from
    the slot, the RAM cache or a checkpoint.
  - heylook keeps an in-memory per-model fingerprint of the last request (a
    hash per message) and labels probable causes (`probable_*`). An example:
    same leading messages but `cache_n` far short means the template diverged.
  - The two default-level WRN lines (entry over the RAM budget, eviction) are
    read from the subprocess pipe (not a file, so observability `off` still
    writes nothing) with a parser pinned by a fixture.
  - The durable fix is an upstream PR adding a cache source to `timings`.
- **Per model.** A `cache` block on the admin row:
  - gguf: effective RAM budget, checkpoint count and spacing, reuse class
    (full attention vs checkpointed SWA/hybrid, from the header), KV-shift
    disabled with mmproj, spec decode configured and how.
  - MLX: reuse eligibility and reason, byte budget, slot size, vision-cache
    counts.
- **Storage.** Counts and enums only in the request telemetry (content-free).
  Per-message stats in an additive `message_stats` table, so numbers survive a
  reload with no schema bump.
- **Frontend.**
  - An always-visible muted stats line per assistant message, e.g.
    "cache 12.3k/14.0k · draft 71% (340/480)". A miss states its reason in
    text.
  - A "Cache" section per loaded model on the perf page.
  - A token-weighted cache column in the trends table.
  - The mislabelled "KV" figure is replaced.

- **Live cache-reuse check in `tests/smoke`** (it also serves as W10's
  acceptance test). Per engine arm, against a running server:
  - turn 2 of an image conversation processes about the new content (new
    image plus question), not the history;
  - a text-only follow-up processes a handful of tokens;
  - a repeated system prompt is reused across requests.

  Assert on those relationships, never on absolute counts. The MLX vision arm
  reports "known gap: W10" rather than failing, until W10 lands and it becomes
  a real pass/fail. It needs W5 first, because MLX's cached count reaches the
  wire only through W5. The audit's template defect would have failed this
  check on its first run.

### W6. Prompt-cache RAM budget: derived default

**Evidence-gated (2026-09-23).** On the hybrid Qwen3.8 the default budget held
a 10k-token image conversation comfortably, because its per-token KV is small.
So build this only when W5's reporting shows over-budget skips in real use.
Until then, W5 surfaces the budget and every skip.

- `cache_ram_mb` unset = auto, never written. The derivation is sized from
  the model's measured state:
  - bytes per token from the KV geometry;
  - checkpoint size from the recurrent/SWA state;
  - both confirmed at spawn from llama-server's own figures.

  The target is "hold N conversations of the model's context", capped by the
  RAM left after weights and KV.
  - A mid-size model on this machine gets a generous budget.
  - A model near the working-set ceiling keeps llama.cpp's default.
- The chosen value and the reasoning are shown on the admin row. Over-budget
  skips are reported through W5.

### W7. Thinking budget: a hard cap per request

- The Messages API already has `thinking.budget_tokens`. heylook honours it:
  - on gguf as llama-server's per-request `reasoning_budget_tokens`;
  - on MLX with a logits processor that closes the thinking block at the
    budget.
- Exposed in the chat settings panel.
- The quality cost of a hard cut is unmeasured. The control is disclosed as a
  cap, not a quality-neutral setting.
- **Re-point the eval bank's `thinking_requested_split`.** Until a hard
  budget exists it measures verbosity, not the split: a small model that
  thinks past the task's token budget fails it with a clean split (seen on
  Qwen3.5-0.8B at the v2.0.71 port). With W7, the task checks that the budget
  is honoured and the split is clean.

### W8. Non-causal image decode guard (shipped v2.0.70)

- At spawn, when the projector decodes images non-causally (`gemma4v`
  26B/31B, `gemma4uv`, `gemma3`, `deepseek4v`), heylook passes
  `--image-max-tokens` no larger than the effective micro-batch.
- The set and the limits are NOT derived: llama.cpp exposes them only in C++.
  Owner call 2026-09-23: a small table plus a test that reads the built
  tree's source, not a build-time parser that could block a llama.cpp update
  ([sharp_edges.md#gguf-non-causal-images](../architecture/sharp_edges.md#gguf-non-causal-images)).
- This closes a `GGML_ASSERT` abort. It is latent today (no gemma-4 gguf is
  served).
  - DeepSeek's 384-token cap and gemma3's fixed 256 are safe at the default
    micro-batch.
  - Both become unsafe if anyone raises `--image-max-tokens` above the
    micro-batch.

### W9. Metal residency keep-alive (shipped v2.0.70)

- A server setting that passes `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` at spawn.
  This is llama.cpp's heartbeat that keeps weights resident; it is not heylook's
  idle unload.
- **Measured worthwhile (2026-09-23).** On the 145 GB DeepSeek-V4-Vision,
  the first request after an idle gap past the default 180 s paid a
  first-token delay many times the warm one. With the keep-alive raised to an
  hour, it answered as fast as a warm repeat. On a ~30 GB model the effect was
  small.
- The default should be derived, not fixed: keep resident for as long as
  heylook's own idle unload would keep the model loaded, since holding the
  model while letting its pages go cold is the worst of both.
- As shipped: resident for the life of the process (a thirty-day constant),
  because a value derived from the idle threshold at spawn goes stale when
  the threshold changes live or the model is pinned, and heylook's unload
  ends the process anyway. The idle CPU cost was measured first (owner
  condition) and judged negligible
  ([sharp_edges.md#gguf-metal-residency-keep-alive](../architecture/sharp_edges.md#gguf-metal-residency-keep-alive)).

### W10. MLX prompt cache: checkpoints, vision, and hybrids

**The problem.** On MLX:
- a conversation with an image anywhere in its history gets no cross-request
  cache at all, for every family;
- the vision feature cache is keyed by the whole image list, so adding one
  image re-encodes all of them;
- qwen3_5 hybrids get no reuse even on text.

"Caching cannot work for qwen3_5 on MLX" is true of heylook's current
single-slot, after-generation design only:
- the mRoPE blocker is a rope-delta seed, derivable from token ids and image
  grids with no vision tower;
- the trim blocker disappears once state is snapshotted at prompt positions.

**The larger finding (2026-09-23).** heylook straddles two MLX stacks that are
diverging. Vision models run mlx-vlm's model code and preprocessing but mlx-lm's
decode loop (`stream_generate`). Most MLX sharp edges in `CLAUDE.md` live on
that seam:
- the raw tokenizer versus the TokenizerWrapper, and the detokenizer class;
- manual position resets;
- the prefill-all-but-last handoff;
- the dropped `max_pixels` kwarg;
- the attention-mask trap.

Meanwhile mlx-vlm has become a full stack of its own:
- it no longer depends on mlx-lm (the kernels it needed are vendored);
- it has its own generation loop, cache, restorable checkpoints (APC) and
  speculative verifier;
- 150 of its 227 model directories are text-only;
- it uses the fast paths at least as heavily as mlx-lm;
- it is far more active, but more concentrated (a few people make most of the
  commits).

The seam will widen, not close. This also settles an open forward item from
the 2026-08-07 postmortem: "adopt mlx-vlm's generator, or port its cache and
prefill into `run_generation`".

**Step 1: spike (decide before building).** Can heylook drive mlx-vlm's own
generator with its checkpoints, keeping heylook's hooks?
- the generated-only penalty wrapper;
- prefill progress;
- mid-prefill cancel and the abort path;
- per-request telemetry.

The spike ends in exactly one named outcome:

| Outcome | When | Result |
|---|---|---|
| **A1** mlx-vlm drives vision models; mlx-lm stays for text-only models | The hooks fit | The seam is gone for vision, and W10's checkpoints come mostly from upstream. Two engines remain, but never mixed within one request |
| **A2** mlx-vlm drives everything; mlx-lm is removed | The hooks fit AND the text-model check passes (below) | One MLX engine and one pin. heylook deletes the loader routing between engines, the two tokenizer shapes, the prefill handoff, the manual position resets and its own prompt cache (replaced by APC). The smoke suite goes from two MLX arms to one. It trades Apple's broadly maintained code for a faster, more concentrated project, so it is an owner decision on evidence |
| **B1** heylook builds llama.cpp-style checkpoints itself | The hooks do not fit | Partial-state snapshots at the end of the system prompt and at the history/generation boundary, keyed by per-image hashes on the vision path. The rope delta is seeded on restore and `_mrope_reuse_safe` is removed. The seam stays |
| **B2** upstream the missing hooks to mlx-vlm, then A1/A2 | The hooks do not fit, but they are small and general | Less local code; the timing depends on the maintainers |

- **Text-model check for A2.** mlx-vlm must load a plain text-only
  checkpoint through its normal loader. Its text ports must also match
  mlx-lm's on the text models actually served: same tokens at greedy, and
  matched decode speed under matched controls. A model that fails stays on
  mlx-lm (that is A1).
- **Who owns model code.** Never heylook. heylook has no architecture ports
  and keeps it that way. Text-only models are implemented by mlx-lm (A1/B) or
  by mlx-vlm's ports (A2). A model neither library supports is ported
  upstream, not here.
- **Fallback while any of this is in flight:** gguf is already the working
  multi-turn vision engine for the same families, now that the Qwen3.8
  template is fixed.

**Step 2: build the chosen outcome.** In every outcome:
- **Per-image vision feature cache**, so a new image stops re-encoding the old
  ones (unless APC under A1/A2 already covers it).
- **The decision logic is a pure table**, separate from the code that applies
  it and testable without MLX. mlx-swift-lm's `PromptCacheReusePolicy.swift`
  is the structural reference (extend, append media, rewind, rebuild).
  - **It is not the mechanism:** for a non-trimmable hybrid whose template
    re-renders history differently, it rebuilds, with no checkpoints. The
    mechanisms that solve that case are mlx-vlm's APC and llama.cpp's context
    checkpoints.

**Why.** It makes the two engines behave the same on hybrid and vision
multi-turn. It is the owner's daily path, and it also serves the Qwen-Image
PE encoders' long fixed system prompts.

**Verification.**
- W5's live cache-reuse smoke check is the acceptance test: the MLX vision arm
  flips from its named known gap to a real pass.
- Plus the multi-hop greedy chain probe (single hops passed before, so only a
  chain discriminates) and `scripts/vlm_parity_probe.py` on vision restores.
- Under A2, add the text-model check above.

### W12. Profile before any native-code question

Independent of W10's outcome, and small. It answers whether a native (Swift
or C++) piece is ever justified.
- Profile decode on the models as actually loaded (through mlx-vlm for vision
  models): the fraction of wall time outside MLX evaluation.
- Measure image preprocessing time per image at the owner's usual sizes. It
  runs through transformers, PIL and OpenCV on the CPU, which is the likelier
  Python cost for vision.

The outcomes are named in advance:
- **Both small** → the native-server question closes; stay Python.
- **Preprocessing dominates vision TTFT** → MLX-native preprocessing, upstream
  in mlx-vlm or in heylook. Check first whether mlx-vlm already has it for the
  served families. A small native sidecar only if that fails.
- **Overhead matters only on tiny models** → targeted fixes (a compiled
  sampler), not a rewrite.

Data goes to `internal/claude/perf/` with conditions attached. No figures in
tracked docs.

Out of scope for this plan, recorded so it is not re-derived: a **native
on-device client** (mlx-swift on iPhone/iPad, speaking the same Messages wire,
running small models locally and falling back to heylook). It is a capability
track, not a server speedup, and mlx-swift-lm has the small vision families it
would need.

### W13. One engine contract (first cut shipped v2.0.73)

**The goal.** Every engine answers the same questions through its provider,
and everything else consumes one shape. Today's single request type,
single output type, single wire and single sampler cascade are the start of
this. What is not unified is everything *about* an engine:
- `capabilities.py`, the admin routes and the frontend answer thinking
  support, vendor sampling and other facts with per-engine branches;
- that is where the drift bugs came from (the gguf vendor layer landed in the
  provider while the capability report still said MLX-only).

W2, W4, W5 and W1 each add per-engine facts. Built independently, they would
each add branches in their own place. This workstream makes them
implementations of one contract instead.

**Shape:**
- `provider.describe()` returns thinking controls (W2), image geometry (W4),
  cache profile (reuse class, budget, known limits; W5), load settings (W1)
  and the template in force (W3).
- Every generation chunk can carry the per-request cache and speculative
  reports (W5's schema).
- `/v1/models`, the admin row and the frontend read that one shape, with no
  engine switches. The existing per-engine branches move behind the providers,
  which is net deletion. A future engine implements the contract and nothing
  else changes.

**Limits, stated so they are not re-litigated:**
- **Unified cache means one vocabulary, one report, and one decision table
  for MLX (W10). It does not mean shared cache storage.** MLX and llama.cpp
  hold their state in incompatible formats, and llama-server decides its own
  reuse. For gguf, heylook observes (W5) and influences (stable templates,
  spawn flags).
- **Routing one model's request across engines** (for example sending
  multi-turn vision to a gguf copy until W10 lands) is not in scope. Each
  model id maps to one engine. Revisit only if W10 fails.

**Adopted refinements (2026-09-23, agreed across sessions and with the owner):**
- **Two halves in code, provenance on every value on the wire.**
  - The static half is functions over the config, never a loaded provider,
    so it answers for unloaded models. It is cached by a stamp over EVERY
    input that can change the answer, not just the weights: the template
    ladder files (override and sidecar included), the mmproj, and for
    gguf-derived facts the build manifest's commit. Each engine declares its
    inputs beside the resolvers that read them, so the stamp is derived.
  - The observed half comes from the running process and is null until the
    model is loaded.
  - Each value carries its provenance (`derived`, `observed`,
    `observed_cached`, or unknown) rather than consumers inferring it from
    which half produced it. `observed_cached` is W4's spawn-time probe,
    cached on disk by binary version and mmproj hash.
- **A load report, engine-neutral from the start.** What the provider
  decided at load becomes visible in the product instead of only in logs:
  - gguf: the auto micro-batch, the W8 image cap, the template rung, the W9
    keep-alive, and which binary;
  - MLX: the effective loader, prompt-cache eligibility and the reason when
    ineligible, and the template rung.
- **One conformance test, on the carrying path.** It asserts on the
  `/v1/models` and admin-row responses through their routes, not on
  `describe()` alone: the same key set on both engines, each key a value or an
  explicit null. One property, not a family of examples.

**Order.** Before W5, which is the first workstream to add per-engine data.
Define the contract with W5's report as its first member, then W2/W4/W1
extend it. The types are sketched for the owner's review before any code.

**Shipped (v2.0.73):** the contract (`providers/contract.py`), one static
describer per engine, `describe_observed()` on both providers, `engine` on
both model lists replacing `effective_loader`/`context_length`/
`context_running`, "configured" meaning stored and different from derived,
and the route-level conformance test.

**Remaining, in order:**
1. **Done (v2.0.74).** Move vendor sampling and context length out of
   `capabilities.py` into the describers, with one registry still naming
   which engine reads which vendor layer. Checked by byte-identical route
   output before and after; `test_vendor_layer_reaches_the_report_on_every_engine`
   kept its assertion (its mocks now patch the source readers).
2. **Done (v2.0.75).** One effective micro-batch function
   (`LlamaServerProvider.effective_ubatch`, both llama.cpp clamps) that the
   image-cap decision and both halves of the report call, over the same
   inputs the argv carries (argv keeps the request; llama.cpp clamps it). The
   report names the clamp when it changed the answer.
3. Warm the static describe cache in the background after each config load
   (never fatal; stamp-keyed, so a warm that races a reload is ignored).
4. The generic frontend renderer, before W5: the full panel on the models
   page row (ALL settings, per_request included, grouped by `effect`, with
   provenance, since per-request defaults quietly set in models.toml are the
   black box this plan exists to open), and a compact chat popover (runtime,
   context, template origin, a link to the models page row; room for W5's
   cache line). Chat shows no per_request settings; its sampler panel owns
   them. Schema
   `ui:"hidden"` fields are omitted only when not configured; a configured one
   shows read-only.
- **Capability inference moves with W2**, which changes its answers (the
  `reasoning_effort` capability becomes `engine.thinking.depth != null`).

### W14. Activation steering (gated on the research track)

A research track lives outside the repo (`internal/claude/steering/`, local).
It steers a model with a rank-1 LoRA on stock llama-server; no fork, no heylook
code yet.
- **Step 1 passed on 2026-09-21:** the adapter loads, a scale of 0 matches no
  adapter, negative scales work, and a scale change re-processes the prompt.
- **Step 2** (one real verbosity direction) is next.
- **Nothing here is built until step 2 shows a real, measured effect.** This
  section exists so W13, W5 and W0 are built with steering's needs in mind,
  instead of being reworked later.

**How it fits once the gate passes:**
- **W13, the contract:** `describe()` lists a model's steering directions and
  their allowed scale range. It states that steering is gguf-only for now.
- **W5, the report:** a scale change forces a full prompt re-process. The
  cache report records that as the reason, not as an unexplained miss.
- **A provider rule, enforced in code:** a steered gguf model gets its
  explicit adapter list on every request, never a bare request, and heylook
  never uses llama-server's global `POST /lora-adapters`. Step 1 found a
  request that omits the list reuses cache computed at the previous scale, and
  poisons the next request too. That must be structurally impossible, not a
  convention.
- **W0, sidecars:** a model's steering adapters and direction metadata live in
  its own directory, like the chat-template override. They are discovered, not
  listed in `models.toml`.
- **Speculative decoding conflict:** llama.cpp applies a LoRA to the target
  model only, never the drafter. Steering and a drafter on the same model
  undercut each other, and `describe()` says so.
- **Captured data uses the template in force.** Traces for step 2 onward go
  through heylook's render path, with the fixed Qwen3.8 override. The
  pre-fix template would bake its whitespace defect into the captured
  sequences.
- **MLX:** out of scope until the gguf path proves out. It would need its own
  mechanism, and W10's cache work must know about scale-dependent state first.

## Sequencing (revised 2026-09-23, after the measurements)

The first order put W0 first. W0 is gated on its own Phase 0 and two open
questions, and most workstreams do not depend on it: W5, W2/W3, W4 and W10
derive at runtime and write no settings. Only W1, W6 and the provenance
display touch stored config. So W0 runs in parallel instead of blocking.

1. **W8 + W9. Shipped v2.0.70.** Both are small. W9 is now measured: raising the keep-alive
   removed the idle first-request delay on the 145 GB model.
2. **W-1, the eval-bank port. Shipped v2.0.71** (`docs/project/TODO.md`, "Port the eval bank to
   /v1/messages"). Steps 3-5 change exactly the subsystems unit tests cannot
   certify (templates, thinking, cache state), and the bank is dead until it
   speaks the Messages wire.
3. **W13 (first cut shipped v2.0.73), then W5.** Finish W13's remaining
   items (its section, in order: narrow capabilities move, micro-batch
   function, cache warming, generic renderer), then build W5 backend and wire,
   then frontend. This comes first so every later change is observable in the
   product, not only in a harness, and so W2/W4/W1 extend one contract rather
   than adding per-engine branches.
4. **W10**, moved up. On MLX every turn of a conversation with an image
   anywhere in its history re-processes everything, for every family. That is
   the owner's daily path and probably the largest single win here, and W5
   makes it measurable.
5. **W2 + W3, with W7.** One template/thinking pass. W7 moved up because
   thinking length proved the dominant and least predictable latency cost,
   varying several-fold at a fixed level, and a hard budget is the only
   reliable control.
6. **W4**, backend then frontend resize. It is independent.
7. **W0**, from Phase 0 onward; it may start in parallel from step 1. Then
   **W1**, which renders W0's provenance.
8. **W6 only if W5 shows budget skips.** The default budget held a
   10k-token conversation on the hybrid model comfortably, because its
   per-token KV is small. So auto-sizing waits for evidence.
- **W11 (optional, any time): upstream llama.cpp.**
  - A cache-source field in `timings` (slot, RAM cache or checkpoint), which
    W5 otherwise has to infer.
  - Place a checkpoint at the generation-prompt boundary rather than a fixed
    four tokens from the end. The fixed offset sat one position past the
    divergence a template produced in the audit, which is why a one-newline
    defect cost whole turns.
- **W12 (any time, small): profile before any native-code question.** Its
  outcomes are named in its section. W10 starts with its own spike (outcomes
  A1/A2/B1/B2) before anything is built.
- **W14 (gated): activation steering integration.** Only after the research
  track's step 2 shows a measured effect. Its integration points are written
  into W13, W5 and W0 so those land steering-ready.

Each workstream ships with its CHANGELOG entry and `frontend_v3_spec.md` §4
updates in the same commit as any contract change.

## Owner decisions (2026-09-23)

- **W10: reopened.** It lands after W5, so its effect is visible in the
  product. This supersedes the 2026-09-20 "leave prompt caching alone" call.
- **Sequencing:** the revised order above (approved the same day).
- **W5: align with Anthropic.** `usage.input_tokens` = processed tokens and
  `usage.cache_read_input_tokens` = cached tokens. heylook's detail rides in
  `performance.cache`. **This is a contract change**: the number existing
  clients read changes meaning. Before shipping, check v3's readers and the
  owner's other project, and record it in `docs/api_integration.md` and spec
  §4.
- **Muse-Glimmer's hand-written `supports_thinking = true` removed from
  `models.toml`.** Its template reads no thinking switch.

- **W2: no hardcoded thinking levels.** The controls show the in-force
  template's own values; no heylook scale, no mapping, no cross-model
  translation. Presets keep saving system prompts.

Still open, to settle when the workstream starts:
- W0: the two questions `plan_registry_sidecars.md` already lists as due
  before its Phase 2 (read-only model directories; the twin).
