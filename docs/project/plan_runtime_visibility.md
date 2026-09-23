# Plan: runtime visibility and one behaviour across engines

last updated: 2026-09-23 (APPROVED by the owner; nothing shipped yet)

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

### W2. Thinking controls: detect from the template, map one scale

- **Detection (both engines).** Render the in-force template once per
  candidate value (the template's own string literals plus a small fixed set)
  and group by output. It recovers:
  - the thinking switch variable, or none;
  - the depth variable (`reasoning_effort`, `reasoning_strength`,
    `thinking_mode`, ...);
  - accepted values and aliases;
  - the default (the group matching the absent render);
  - strictness (garbage raises, is ignored, or is pasted in);
  - **where depth enters the prompt**: the character offset of the first
    divergence between two depth renders. Early means a mid-conversation change
    re-processes everything.

  Cache the result by template-body hash. Always run it on the template that
  WILL be used (`chat_template_files.view`, which already resolves both
  engines' ladders including the override), at row-derivation time. Never
  write it to `models.toml`.
- **API.** `/v1/models` and the admin row gain a `thinking` block:

      "thinking": {
        "switch": "enable_thinking" | null,
        "depth": {
          "variable": "reasoning_effort",
          "values": ["low", "medium", "xhigh"],   // the model's own spellings
          "default": "xhigh",
          "strict": true,
          "scale": {"off": null, "low": "low", "medium": "medium",
                    "high": "xhigh", "max": "xhigh"},
          "changes_prefix": true                  // depth edits re-process history
        } | null,
        "source": {"template_sha": "...", "origin": "heylook_override"}
      }

  `thinking_default` / `sampler_defaults` report the depth default, so the
  existing "auto (x)" label names it.
- **Mapping.** The request keeps one field on a fixed heylook scale
  (off, low, medium, high, max). The server maps it onto the model's own
  variable and value via `scale`, falling back to the nearest value at or below
  on the ordered vocabulary.
  - A raw model value is also accepted.
  - An unsupported value is a 400 naming the model's values, never a
    llama-server 500.
  - The hand-copied `ReasoningEffort` Literal goes away.
  - **Be honest about what the scale does.** Measured on three models, only
    off, a shortened low, and "on" separated reliably; medium, high and xhigh
    overlapped, and run-to-run variation at one level exceeded the gap between
    adjacent levels. The UI must not imply that a higher step reliably means
    longer thinking. It shows the model's own value and says the ordering
    above low is the template's instruction, not a guarantee. W7's budget is
    the control that actually bounds length.
- **Frontend.** The dropdown is built from the row. It shows the heylook step
  and, muted, the model's own value ("high → xhigh"). A thinking toggle
  appears only when `switch` is non-null. When `changes_prefix` is true, a
  mid-conversation change is disclosed ("re-processes the conversation").
- `supports_thinking` for gguf is derived from the in-force template, not read
  once at import from the embedded one.
- **Not in scope: editing templates to add levels.** A level is prompt text
  the model was trained on. An invented level is untrained text, and measured
  even the trained levels above "low" barely separate.

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

### W8. Non-causal image decode guard

- At spawn, when the projector decodes images non-causally (`gemma4v`
  26B/31B, `gemma4uv`, `gemma3`, `deepseek4v`), heylook passes
  `--image-max-tokens` no larger than the effective micro-batch. The set and
  the limits are derived from the projector, never listed.
- This closes a `GGML_ASSERT` abort. It is latent today (no gemma-4 gguf is
  served).
  - DeepSeek's 384-token cap and gemma3's fixed 256 are safe at the default
    micro-batch.
  - Both become unsafe if anyone raises `--image-max-tokens` above the
    micro-batch.

### W9. Metal residency keep-alive

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

## Sequencing (revised 2026-09-23, after the measurements)

The first order put W0 first. W0 is gated on its own Phase 0 and two open
questions, and most workstreams do not depend on it: W5, W2/W3, W4 and W10
derive at runtime and write no settings. Only W1, W6 and the provenance
display touch stored config. So W0 runs in parallel instead of blocking.

1. **W8 + W9.** Both are small. W9 is now measured: raising the keep-alive
   removed the idle first-request delay on the 145 GB model.
2. **W-1, the eval-bank port** (`docs/project/TODO.md`, "Port the eval bank to
   /v1/messages"). Steps 3-5 change exactly the subsystems unit tests cannot
   certify (templates, thinking, cache state), and the bank is dead until it
   speaks the Messages wire.
3. **W5**, backend and wire, then frontend. This comes first so every later
   change is observable in the product, not only in a harness.
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

Still open, to settle when the workstream starts:
- W0: the two questions `plan_registry_sidecars.md` already lists as due
  before its Phase 2 (read-only model directories; the twin).
- W2: the heylook scale's step names (off, low, medium, high, max proposed).
