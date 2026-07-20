# Persistent TODOs

Cross-session task backlog organized by priority.

*Last reviewed: 2026-07-20*

## Upstream-borrow follow-ups (vllm-metal scan + delta review, 2026-07-20)

From the coderef/vllm-metal optimization scan + mlx-lm/mlx-vlm delta review
(session log 2026-07-20). Verdicts validated against the plan's Phase 5
ordering and the sole-user/minimal-custom-code posture.

- [ ] **Gemma-4 MTP self-speculation** (P2, the live lead): mlx-lm PR **#1276**
  (open, "Add Gemma 4 assistant (MTP drafter) model class") is the adoption
  vehicle -- model class only, no generation wiring yet. PROBED LIVE 2026-07-20
  (`internal/research/mtp_probe/`): the zoo's 26B-A4B assistant head drafting
  for the daily MoE measures **50.8% greedy acceptance** (lower bound; scaled
  embeddings are the trained convention), drafter ~1ms vs ~11ms/step ->
  projected ~1.4x decode ceiling at n_predict=1. Matches vllm-metal's published
  draft-model wins. Build list + caveats in the probe README. Watch #1276;
  when it merges (or from the fork), build the verify loop provider-internal
  behind `run_generation`. Greedy-only; fingerprint gate invalid (doctrine).
  OWNER ACTION (optional, with data): comment on #1276 -- `AssistantAttention`
  uses `num_key_value_heads` for both layer types; the 26B assistant needs
  `num_global_key_value_heads=2` on full-attention layers or SDPA mis-shapes
  (their network test uses the E4B where counts coincide). Acceptance numbers
  are shareable.
- [x] **mlx-vlm pin bump -- DONE 2026-07-20** (0bfe60b): bumped mlx-lm
  a790972->15b522f (+1 commit, server XTC fix, N/A to us) and mlx-vlm
  8e2638b->c9e27b08 (0.6.5->0.6.6, 53 commits: gemma-4 bf16 dtype-leak fix,
  `prepare_inputs` mask preservation, qwen3-vl PIL video-frame normalization).
  Consumed surface confirmed signature-stable (22 contract tests green);
  transformers deliberately stays pinned at 5.5.4 via `override-dependencies`
  (mlx-vlm HEAD declares `>=5.14` -- decoupled on purpose, contract tests are
  the gate, not the floor). Verified: full suite 1085 green + a live eval-bank
  A/B (old vs new pin) on gemma-4-31b + Qwen3.5-27B, no regressions.
- [ ] **optloop experiments** (P3, both gated on prerequisites): (a) re-test
  classic draft spec-decode with a ~4B gemma draft (the closed negative result
  used a 1B draft; vllm-metal's 1.36-1.48x with a favorable pairing says the
  ratio matters -- needs a gemma-3-4b download first); (b) n-gram
  prompt-lookup prototype (mlx-lm has no equivalent; zero draft cost, wins on
  repetitive/structured output, greedy-only).
- [ ] **Gemma-4 QAT q4_0 swap candidate** (P3, owner download decision):
  Google's QAT collection (2026-07 blog) ships `*-qat-q4_0-unquantized`
  checkpoints for ALL sizes incl. 26B-A4B + matched QAT ASSISTANT drafter
  heads. MLX conversions exist (mlx-community OptiQ 4-bit for 26B/31B,
  template-synced 2026-07-20; lmstudio-community 26B QAT-MLX-4bit). Halves
  memory vs the daily 8-bit with QAT-held quality; verify OptiQ loads on the
  pinned mlx-lm before adopting. The QAT assistant head also pairs with the
  MTP item above.
- [ ] **E2E checks for the 2026-07-20 v3 features** (P2, from the xhigh
  review's coverage table -- all four currently have ZERO e2e coverage):
  (1) attach 9 files -> 8 thumbs + aria-live cap message; (2) thinking
  toggle visible on capable model, aria-pressed true/false round-trip,
  unchecked sends null not false; (3) drawer `#set-vision_tokens` present
  (min 16 max 16384) + localStorage round-trip; (4) thinking `<details>`
  block appears with non-empty body on a thinking generation, summary
  toggles open. Also update tests/e2e/README.md suite descriptions.
- [ ] **LLM behavior-eval harness follow-ups** (P2, owner-directed 2026-07-20):
  `tests/eval/` (opt-in, seed bank generalized from the 2026-07-20 live
  verification scripts) is the base. Direction the owner wants explored:
  (a) eval as an API surface (trigger runs / read results server-side) and
  (b) an eval page in frontend v3 (run the bank against loaded models,
  compare results -- fits the introspection identity). Design with the
  Phase 6 admin surface + observability pages, not a bolt-on; results
  storage could ride the observability JSONL + DuckDB-over-files pattern.
  Also queue: run the eval bank as the optional gate for changes touching
  templates/parsers/stop-tokens/vision (the 4 bug classes it was built on).
- [x] **Pyright noise triage -- DONE 2026-07-20** (0e236e4): real fixes
  (deprecated `datetime.utcnow`, untyped `= None` defaults, a latent bug where
  batch responses could hand pydantic `model=None` -> runtime 500 now
  coalesced, a float re-binding bug in `_format_bytes`, duck-typed route
  discovery) plus a systemic sweep of all 96 positional Pydantic
  `Field(default, ...)` calls to explicit `default=` keyword form (this
  pyright build only recognizes the keyword form -- the source of most false
  "arguments missing" complaints). `pyrightconfig.json` already existed from
  an earlier session; this pass didn't need a new one. The
  `LogprobsCollector.add_token_and_get_delta` stale-reference suspicion was
  not confirmed as real during this pass -- re-check if it resurfaces.
- [x] **Vision token budget knob -- SHIPPED GENERALIZED 2026-07-20**
  (v1.34.64, ahead of the Q8 spike): model-agnostic `vision_tokens`
  (request + models.toml per-model default + v3 drawer control) mapped by
  duck-typing the processor (gemma buckets / qwen pixel budget); cache key
  carries the budget; live-verified on gemma-4-31b + Qwen3.5-27B. The Q8
  ACCURACY question (does 1120 improve detail QA?) remains open -- the
  eval harness vision tasks are the vehicle.
- [ ] **Optional v3 nano-feature: per-attachment estimated token cost in the
  composer** (P3, leftover from the shipped vision-budget item above): the
  resize math is ~15 lines client-side (gemma bucket snap / qwen pixel
  formula, mirroring the server's own duck-typed mapping). Not required --
  the budget itself is already a first-class request/config/UI knob.
- [x] **Gemma-4 canonical template refresh (2026-07-09 "less laziness" fix)**
  -- DONE 2026-07-20: verified upstream state (template-ONLY fix -- no IT
  weight re-uploads; commits 07-15 "null handling, reasoning preservation,
  turn-tag balance"); owner refreshed 26B/31B jinja, E4B fetched + installed
  from upstream (E-series variant of the canonical). mlx-community 8-bit
  conversions are stale (07-05) -- local chat_template.jinja files are the
  fix, auto-picked-up at load (verified: transformers prefers the jinja
  file; no embedded template in these checkpoints).
- [x] **`mlx_cache_limit_gb` operational setting** -- DONE 2026-07-20
  (v1.34.59): opt-in MLX buffer-cache cap via /v1/admin/config (bounds idle
  RSS at the cost of realloc on the next spike; MLX default restored on
  reset). Plus fix: DELETE on config keys now re-applies immediately.
  Borrowed shape from vllm-metal's measured-overhead cache cap; ours is a
  manual knob, their auto-measurement is overkill for one box.

## J-space / jlens-mlx (from jspace_integration_plan.md Part 2)

Fitting lives in the `jlens-mlx` sibling repo; this server applies. Apply feature +
baseline fitter are GREEN (see CURRENT.md 2026-07-10).

- [x] **Refit the band lens on the fixed corpus** (DONE 2026-07-12): `band-n12`/`band-n12b` were
  degenerate (mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects
  `enable_thinking=True` -> every on-policy completion collapsed into shared CoT-preamble
  boilerplate, 62% of fitted positions). Both fits' results are DISCARDED (method stack unaffected).
  jlens-mlx now has explicit `enable_thinking` control (default False) + a diversity gate
  (`238826e`/`951dd76`/`232b98b`). REFIT COMPLETE: `out/band-n14-fixed` (11 items, band 16-47,
  `identity_ok: true`, ~4.25h, zero SIGKILLs). Qualitative readout done -- L40-42 surface meaningful
  tokens (Paris/city/France), L45-47 degenerate but that's the model's own degeneracy. See the
  "Fidelity gate" item below for the metric caveat.
- [x] **Abliteration diff -- DONE 2026-07-13** (P1): stock lens (`out/band-n14-stock`,
  `identity_ok: true`, same corpus as `band-n14-fixed`, item 10 skipped in both) fit clean overnight;
  `diff_lenses.py` run BOTH substrate directions (stock + heretic). FINDING (robust,
  substrate-independent, layer-for-layer agreement): the abliterated transport surfaces safety/refusal
  vocab MORE in the mid-late band (L32-42: `Safety`/`unsafe`/`unethical`/`dangerous`/`Cannot`/
  `violations` + CJK `安全风险`/`违反` + Russian `безопасность`), and SUPPRESSES geography (China/Europe)
  + retrieval verbs. Counterintuitive on purpose: abliteration edits the TRANSPORT, not the readout --
  Heretic (confirmed by reading its source) orthogonalizes the residual-WRITING matrices (every
  layer's `attn.o_proj`/`mlp.down_proj`) against `r = mean(harmful) - mean(harmless)`, tail blocks
  INSIDE the fitted Jacobian (`model.norm` is untouched). **Interpretation RETRACTED 2026-07-13
  (second correction, same day):** a per-prompt re-run (below) falsified the content-conditional
  "disposition preserved" reading -- the diff is abliteration's STATIC WEIGHT-EDIT FINGERPRINT,
  readable on any input, not a content-conditional internal state. Cross-validated by an independent weight-footprint analysis
  (`scripts/abliteration_footprint.py`, jlens sibling repo; `out/abliteration_footprint.txt`): edit ~6x
  concentrated in residual-writing matrices, vision tower bit-identical, weight-delta peak (L33/L36)
  co-localizes with the transport-diff safety cluster (L32-42). Write-up + explainer live in the jlens
  research repo: `docs/abliteration_diff.md` + `docs/abliteration_diff_explainer.html`. Two open caveats -> two
  follow-ups below.
- [x] **Abliteration diff -- per-prompt benign floor** (DONE 2026-07-13, P2): `scripts/per_prompt_diff.py`
  (`out/per_prompt_diff.txt`, jlens sibling repo) re-ran `diff_lenses.py` one prompt at a time instead of
  pooled. RESULT: FALSIFIED the benign floor. The benign weeknight-recipe prompt lights up the same
  L32-42 safety band just as strongly as the safety-adjacent prompts (mean l2 596 vs 524-571) and
  surfaces the same refusal vocab (Nothing/Impossible/cannot/unsafe). The effect is PROMPT-INDEPENDENT
  -- this retracts the "disposition preserved" content-conditional reading above; see the interpretation
  update. Converter-match now CLOSED 2026-07-13: self-converted the base (mlx-vlm 0.6.5) diffed vs the
  mlx-community base is uniform ~0.003-0.004 drift, no tent, with `o_proj`/`down_proj` among the LOWEST
  (~8x below the abliteration signal, structureless) -- converter asymmetry cannot manufacture the
  finding (`scripts/abliteration_footprint.py`; `out/converter_drift_base_vs_mlxcommunity.txt`). Both
  original caveats on this finding are now resolved.
- [~] **A genuinely disposition-aware metric is STILL OPEN** (P2, updated 2026-07-12): the KL/top-k
  identity tripwire ships. BUT the qualitative readout on `band-n14-fixed` proved the
  `verify.legibility_report` metric ALSO MISLEADS -- it ranked the degenerate deep layers J_45/46/47
  HIGHEST (0.91-0.93) while the meaningful J_40 scored 0.85, because the degenerate ' __'/' ____'
  readouts "agree" with the model's own degenerate next-token output. This is the SAME failure mode
  as the old final-logit fidelity gate, now reproduced with a clean corpus AND the new metric -- so
  it is a metric problem, not a corpus problem. For now, judge readouts QUALITATIVELY (`readout.py`).
  An actually-disposition-aware metric (penalize format/junk-token readouts) is unsolved.
- [x] **Fit/apply capture parity -- numerical check** (DONE 2026-07-12: BIT-EXACT, rel_err 0.0 at 9 layers incl. band edges on the served 27B; gate script `check_capture_parity.py` in the jlens sibling repo, jlens commit 36d859b -- rerun it at the top of every refit session; 2026-07-11, re-affirmed 2026-07-12 as the
  top open correctness IOU by an architecture review): fitting captures residuals cache-less; apply
  uses a fresh cache (the hybrid served qwen3_5 crashes cache-less). Both are causal-from-scratch so
  they SHOULD match, but it's asserted, never verified -- and it's the foundation of served-model
  lens correctness. Cheap check: capture `h_l` both ways on one input, assert allclose. (Does not
  invalidate current lenses; identity KL~0 is consistent.)
- [~] **Post the PR comments** (P2, 2026-07-12; harness port DONE -- `bench/upstream_pr_eval/` on the pushed jlens `upstream-pr-eval` branch, links filled in the drafts): the
  2026-07-12 upstream mlx-lm GDN differentiability eval (PRs #1389/#1217, both numerically correct,
  see the plan doc) was run ad hoc; port it to a jlens-mlx `upstream-pr-eval` branch, then the owner
  posts data-backed comments on #1217 (full dataset) and #1389 (the log-domain fp32 `dg`-gradient
  finding -- not a bug, cancels at the parameter leaves). Draft comments live in jlens's internal
  folder.
- [ ] **Audit GDN cache-slice captures for the #1077 `mx.contiguous()` pattern** (P3, 2026-07-12):
  upstream mlx-lm #1077 (merged) fixed a shared-buffer memory leak by adding `mx.contiguous()` on
  GDN cache slices. Check our code (jspace capture path + any other raw GDN cache-slice reads) for
  the same unguarded-shared-buffer pattern.
- [ ] **Watch #1217 merge before any mlx-lm pin bump** (P2, 2026-07-12): #1217 adds a `training=`
  kwarg passed unconditionally at every qwen3_5 call site upstream; jlens-mlx's `gdn_fit_patch`
  already absorbs unknown future kwargs (`951dd76`) so a bump won't TypeError mid-fit, but confirm
  before bumping the served-side pin too.
- [ ] **Consider #1515/#1532, #1486/#1456, #1526 on the next serving-side mlx-lm bump** (P3,
  2026-07-12): #1515+#1532 add anchor-stride prefix reuse for non-trimmable hybrid caches (large
  TTFT claims, relevant to qwen3_5 serving); #1486/#1456 fix hybrid ArraysCache trimmability for
  speculative decoding (issue #1446); #1526 fixes `max_kv_size` being silently dropped for models
  with their own `make_cache` (qwen3_5 still needs the analogous one-line fix upstream). None of
  these are fitting-path, all are serving-path -- triage on the next pin bump, not now.
- [~] **Fit memory levers (de-brittle the fit; CORRECTED 2026-07-12 PM)** (P1): peak scales with
  FITTED POSITIONS, not sequence length. The earlier "~1.7GB/token of sequence" slope was a corpus
  confound (short items had both few positions AND short sequences). The real model: **~63GB base
  + ~2.1GB per fitted position** (flat across seq 72-78 at 47 positions; validated live -- item 11
  at 56 positions -> 174.6GB). The forward runs over the full sequence; the backward/Jacobian runs
  only over the fitted positions, and that sets peak. On-policy items fit ~47 generated tokens
  (capped by `on_policy_max_tokens=48`). Two distinct problems tonight:
  - **(a) transition SIGKILLs (exit 137), DONE (real fix).** MLX's caching allocator never
    returns freed buffers to the OS, pinning RSS at the run's max-item high-water (~161GB) for
    the whole process lifetime, tripping the macOS jetsam killer at item transitions on the
    192GB box. Fixed with `mx.clear_cache()` between items (jlens commit `e56fad6`) — drops RSS
    between items, negligible cost. (`reset_peak_memory` resets the counter, not the pool.)
  - **(b) item 10 dropped -- but LIKELY UNNECESSARY.** Item 10 (seq 126) was dropped via the new
    `JLENS_MAX_FIT_SEQ` env (jlens commit `073cc04`), on the WRONG (sequence-slope) extrapolation
    to ~245GB. Under the corrected positions model it has ~47 fitted positions -> ~163GB peak and
    would have FIT. So `JLENS_MAX_FIT_SEQ` is the wrong knob (positions / `on_policy_max_tokens`
    is the lever); `band-n14-fixed` is an 11-item lens that could be re-fit to 12. Not urgent.
  The chunk 128->64 lever is FALSIFIED as a memory lever (measured 2.8% reduction — dim-batch
  memory is chunk-independent).
  NOT urgent (tonight's fit is unblocked), but a standing liability for longer-context transfer
  experiments, the stock-model diff, and item-batching, all of which want headroom. Deeper
  follow-ups (NOT done, parked for a future session, do M2 before M3; tracked in jlens
  `docs/fit_metrics.md` §3):
  - **M2 — instrument the memory.** Sample `mx.get_active_memory`/`get_cache_memory` around each
    chain-sweep phase to find WHERE the per-token memory lives. First-principles estimates range
    34-320GB depending on assumptions, none match the measured ~161GB, and chunk-independence
    rules out the obvious dim-batch-cotangent hypothesis — the footprint is genuinely
    unexplained. Cheap; the prerequisite for any real reduction and the honest end of guessing.
  - **M3 — the checkpointing bench.** `feat/checkpoint` (built, equality-gated, unproven at real
    scale) is the one lever that could reduce a SINGLE item's peak — the real headroom fix for
    genuinely high-position items (long-context transfer experiments), the case `JLENS_MAX_FIT_SEQ`
    only papers over. Bench it on the real 27B.
  NB the T<=128 GDN kernel brittleness is a separate, unrelated issue (see the fit-speedup item
  below) — it sunsets when mlx-lm PR #1389/#1217 merge (we delete the kernel + monkey-patch).
  Memory-brittle: (a) fixed for real, (b) was an over-drop; deeper headroom fix (M2/M3) still open
  for high-position items. Kernel-brittle: wait + adopt.
- [ ] **Fit speedup: seq-tile the GDN scan** (P3, 2026-07-11): the chain fit is ~44min/item / a full
  band ~7-8h; a designer+verifier pass found NO config-level 2-3x (`chunk` is a dead knob). The real
  lever is the GDN kernel `MAX_T=128` cliff -- tile the recurrence across 128-tok blocks (EXACT) so
  long items stay on the fast kernel. Delicate; must re-pass `check_chain_vs_direct` (cos 1.0). A
  guardrail (warn + kernel-eligibility sidecar metadata) already stops the silent slow path.
- [ ] **Standing golden gate for `/v1/jspace/analyze`** (P3): freeze onset top-k + features,
  tie-aware calibrated epsilon, mutation-checked -- turns the one-time V1/V2 parity into a
  wired regression gate.
- [ ] **Visualizer track** (P3): gate cleared 2026-07-10 -- `apps/heylook-frontend-v3/DESIGN.md`
  seeded (OKLCH strength/chip system formalized; paradigm = matrix-first, Neuronpedia-style
  layer-range slider + aggregation sidebar as the growth path). SHIPPED so far
  (v1.34.36-.37): item 1 click-to-pin readout (strip rows + heatmap cells, Esc/arrow
  walk, echo highlight, onset marker), the per-cell top-N analyze extension
  (`heatmap_top_k` -- every cell pins its full readout now), item 2 layer-range
  slider + aggregation view, and a "provisional lens" badge off the sidecar
  provenance (`/v1/jspace/models` meta). Remaining, in order: live streaming
  (new SSE analyze endpoint) -> steer/swap/ablate interventions (last).
  **Fold into the streaming rework** (both live in the analyze grid loop it
  will rewrite; from the 2026-07-10 review): (a) unify the onset column's two
  numeric paths -- onset_strip uses float64 np.argsort, the heatmap's last
  column uses float32 argpartition, so near-tied logits can show different
  top-1 tokens for the same position (breaks the echo highlight); one shared
  per-position reduce fixes it. (b) batch the per-layer device-to-host syncs
  (~4 x band_layers sequential np.asarray evals under the gen gate) into one
  mx.eval, and memo tok.decode per request (~5k redundant single-id decodes).
- [ ] **Confirm coverage for the deleted `verify_endpoint.py` / `probe_thread.py`** (P3,
  updated 2026-07-12): they were git-rm'd from jlens (its `migrated_from_scratch/` is fully
  dissolved; recoverable from jlens git history). This repo's `tests/contract/test_jspace_api.py`
  + `tests/unit/test_jspace_analyze.py` likely cover the same ground -- diff the checks, recover
  from history only if a gap shows.
- [ ] **HF lens repo** (P3): publish OUR fitted lenses post-own-fit; gated -- don't
  republish the converted third-party lenses (Gemma ToU).
- [ ] **Stale docstrings** (P4): `tests/unit/test_jspace.py`, `test_jspace_features.py`,
  `src/heylook_llm/jspace/capture.py` still name `coderef/jspace_scratch/` (dissolved into
  `jlens-mlx/migrated_from_scratch/`) -- fix when next touching those files.

## Presets/system-prompt follow-ups (from the v1.34.22-.24 review passes)

- [ ] **Cap-gated `enable_thinking` can be pinned invisibly** (P2): the global
  settings cache keeps cap-gated keys when the current model can't display
  the control (`requiresCap` hides the checkbox but `samplerParams()` still
  emits the flag on every chat/notebook/explore request until Reset). Predates
  presets -- switching models always had this gap -- but preset apply is a
  one-click entry point into it. Needs a caps-aware settings design (drop or
  grey cap-gated keys when unsupported), not a preset-side patch.
- [ ] **Presets vs TOML registry: dual-source by name** (P3, design decision):
  user presets are client-expanded only; `ChatRequest.preset` resolves just
  the bundled TOML registry. If saved presets should ever work by name from
  the raw API/CLI, make the registry dual-source (TOML + DB rows, name-unique
  across both) instead of growing a second wire path. Deliberately NOT done
  in v1.34.22 -- client-side copy semantics (LM Studio) was the owner's ask.
- [ ] **Notebook page preset bar** (P3): chat-first was deliberate; the
  sections prepend cleanly onto notebook's settings panel. Notebook already
  has its own sysprompt input -- decide whether preset apply writes it.
- [ ] **"Panel drifted from selected preset" indicator** (P3): applying copies
  with no live binding; after edits the select still shows the preset name.
  LM Studio shows a modified-dot. Cosmetic, deferred.
- [ ] **Unknown `params` keys stripped on apply-then-save** (P3, edge): a
  preset authored via the API with keys outside `PARAM_META` loses them if
  the UI applies then re-saves it (panel state is the source of truth).
  Fine for UI-authored presets; document or merge-through if API authoring
  becomes real.
- [x] **v3 frontend doc** (P2, docs): DONE 2026-07-09 -- the v3 map (what's
  done/left + backend<->v3 coupling + remaining backend work) now lives
  git-tracked at `docs/frontend_v3.md` (renamed + promoted out of
  `internal/frontend/v3.md`); the 5 stale React-frontend docs were archived to
  `internal/frontend/archive/`. CLAUDE.md Orient-first + architecture paragraph
  point at it and at the plan as the roadmap.
- [x] **DRY settings drawer** (P2, plan Phase 4): DONE 2026-07-11 -- the chat settings UI
  extracted into an app-shell **global slide-over drawer** (`js/settings-drawer.js`) shared
  by all 6 pages; `registerSettings(contribution)` with the sampling / global display prefs /
  per-page extras taxonomy (DESIGN.md §6). jspace's toggles + samplers-greyed, notebook/chat
  sysprompt as sections, presets on chat. Edge cases preserved, browser-verified, code-reviewed.
- [x] **Impeccable design-quality pass (plan Phase 4 item 2)**: DONE 2026-07-11 -- ran
  `/impeccable audit` + `polish` across all 6 pages + app shell + settings drawer. Slop-detector
  clean; technical score 17/20. Fixed the mobile + a11y cluster: **delete/rename were unreachable
  on iPhone** (hover-gated, no `@media(hover:none)` fallback -- a genuine touch bug); status lines
  now `role=status` / error surfaces `role=alert` (honest states reach screen readers); sampler +
  display inputs get `<label for>`/id; drawer became a real modal (seals `#app` via `inert`);
  explore chips carry a `title` (non-color access, DESIGN.md §2); `aria-current` on active nav;
  `--radius-sm` undefined-token nit fixed. **Mobile settings gear moved FAB -> trailing `⚙` item
  in `#bottom-nav`** (owner-chosen: a FAB collided with chat's Send button). Verified at an
  iPhone-17-Pro viewport with touch-media emulation (19/19 checks) + full E2E (zero regressions vs
  baseline). Rules recorded in DESIGN.md §7. Deferred P3s (single-user pragmatic floor): conv/
  notebook list items not keyboard-focusable (`<div>`, not button); jspace layer-range slider is
  pointer-only.
- [x] **E2E suite stale vs the settings-drawer refactor** (P2): DONE 2026-07-11. The suite predated
  `42a1769` (drawer unification) + the jspace route -- it clicked a `.chat__bar` "Settings" button,
  poked on-page `.chat__sysprompt`/`#jspace-heatmap`, and asserted 5 routes, so 14 checks failed on
  BOTH HEAD and the design branch (never regressions). Added `openDrawer`/`closeDrawer` helpers
  (`lib/dom.mjs`) that drive settings/presets/sysprompt/jspace-toggles through the shared drawer, and
  fixed the route count (6). The drawer is modal (inert #app + covering backdrop), so the helpers are
  transition-aware: reset a leaked-open drawer, fire the gear via `evaluate` (a real click hits the
  fading backdrop), and wait for the slide-in + the backdrop's *delayed* hide to settle -- documented
  in the E2E README's new "Settings drawer" gotcha. Also hardened two pre-existing latent races
  (assistant-reply-persists polled instead of racing finishStream's save; post-abort waits for the
  conv list). Result: 62/63 green; the sole miss is the load-sensitive streaming-cadence perf check
  (Mac Studio throttled after many back-to-back 26B spawns -- passes idle, README notes it).
- [ ] **Wire `show_special_tokens` render consumers** (P2, 2026-07-11): the display pref exists +
  is honesty-first `true`, but is gated (`wired:false`) out of the UI because NO surface reads it --
  the token-rendering paths still strip specials (DESIGN.md §6 "known violation"). Wire chat /
  notebook / token-explorer to read `getDisplayPref` + subscribe to `onDisplayChange` (unsubscribe on
  teardown), then flip `wired:true`. Edit surfaces must ALWAYS show raw tokens regardless (§6).
- [ ] **jspace viz: chat-turn default + special tokens + prefill/token-walker** (P2/P3, 2026-07-11):
  see `docs/jspace_integration_plan.md` Part 2 (2026-07-11 refinements). Flip analyze to chat-turn-
  default (verify the "format-dominated onset" claim -- likely a provisional-lens artifact), show
  special tokens, add prefill/edit-the-assistant + per-token selection (the `coderef/mlxui-core`
  possibility-horizon walker collapses prefill+selection into one primitive). Activation patching
  (steer/swap/ablate) = port `mlxui-core`'s op-semantics via forward-hooks, NOT its per-arch subclassing.

## Chat-template resolution follow-ups (from the v1.34.38 review, 2026-07-10)

v1.34.38 made template resolution a registry concern (server import detects jinja,
chat_template.json fallback, auto-install-when-missing, actionable errors). The
review (10 verified findings) split into a quick hardening batch and design items.

- [x] **Quick hardening batch**: DONE v1.34.40 (same day) -- shared
  `detect_chat_template_source()` helper used by both import paths (+ expanduser,
  fixing tilde-path detection); `"auto"` no longer force-installs and
  `"chat_template_json"` became an accepted explicit source; the missing-template
  error is decided from tokenizer state (not transformers' error prose), respects
  wrapper-level python templates (`has_chat_template`), and covers all three apply
  sites (chat, batch, hidden-states); the load warning consumes install's return
  and no longer false-alarms on `chat_template_type` models (this also closed the
  "wrapper-level templates false-alarm the warning" design item).

**Design items** (need a decision, not just a patch):

- [ ] **List-form `chat_template` silently dropped** (P3): HF's legacy named-template
  list (`[{"name","template"}]`, still read AND written by transformers 5.5.4; real
  repos ship it, e.g. command-r-plus conversions) is treated as no-template by
  `_read_embedded_template` -- empty template for harmony/thinking detection and a
  false "lacks a chat_template field" warning under explicit source. Decide: pick
  the "default" entry, or keep string-only and log the list case explicitly.
- [ ] **chat_template.json fallback can change response shape** (P4, note-only?):
  a json-only model whose template carries <think> markers now selects the thinking
  parser (split `thinking` field) where it previously streamed inline -- arguably
  correct (the processor already applied that template) and no local model hits it
  (all json-shipping folders also have jinja), but it's an undocumented behavior
  change; consider a CHANGELOG amendment when touching this area next.

## Model registry: modalities/loader follow-ups (from the v1.34.43 split, 2026-07-11)

v1.34.43 split the overloaded `vision: bool` into `modalities` (description,
`model_importer.detect_modalities`) + `loader` (routing,
`providers/common/loader_routing.py`); `is_vlm`/`MLXProvider.effective_loader`
derive from it. Design + decision recorded in `plan_2026-07.md` Phase 6
("Refinement 2026-07-11"). Shipped, simplified (v1.34.43+), and audited against a
19-model modelzoo sweep. Three deferred items, none urgent, all Phase-6-coupled:

- [ ] **Registry entry `kind` (chat vs draft/MTP vs embedding)** (P2, needs a
  field + a decision): `provider` (mlx|mlx_embedding) doesn't capture that some
  `provider="mlx"` entries are NOT servable chat models -- e.g.
  `gemma-4-26B-A4B-it-assistant-bf16-mlx` is an MTP/draft head ("no chat template
  on purpose"), flagged only by a `draft` tag. Because it inherits gemma's
  `vision_config`/`audio_config`, `detect_modalities` OVER-CLAIMS
  `[text,vision,audio]` for it (routing stays correct via the positive-knowledge
  degrade -> mlx-lm; the over-claim is cosmetic, and only on re-import since the
  toml entry is hand-set `vision=false`). Fix: a `kind` field so UI /
  `/v1/models` capabilities / telemetry / `detect_modalities` don't treat a draft
  head as a chat model. This is the clearest driver for the Phase 6 "entry KIND
  is under-modeled" note.
- [ ] **Manual `loader` override isn't durable until the Phase 6 tomlkit merge**
  (P2, coupled): `loader` is honored at load, but a re-scan REGENERATES
  `models.toml` and wipes a hand-set value (Option 2 by design -- reserve `auto`
  now, durable editing when the non-clobbering merge lands). Sharp edge until
  then; don't advertise the manual override in the UI before it's durable.
- [ ] **Remove the `vision` derived-mirror once readers migrate to `modalities`**
  (P3, cleanup): `vision` is kept as a validator-maintained mirror of
  `"vision" in modalities` for back-compat. Migrate readers to `modalities`
  (grep `config.vision` / `.get("vision")` / `config["vision"]` -- known:
  `api.py::_infer_model_capabilities`, `model_importer` entry-build,
  `loader_routing._modalities_of` raw-dict fallback), then drop the mirror + the
  bool. Do NOT do piecemeal -- it's a coordinated removal.

See also the Phase 6 "per-model SIDECAR ARTIFACTS" note (draft model / j-space
lens / future LoRA managed as a group on the admin CRUD surface).

## Observability + config redesign (2026-07-11)

Full design + status: `internal/research/observability_and_config_redesign.md`
(local-only). Backend spine + config layer landed this session
(v1.34.44-.55).

**Done (backend):**
- [x] Config foundation: App-DB `settings` table + `/v1/admin/config` (env > DB >
  default, then made DB-authoritative -- no env override for operational settings).
- [x] Observability spine: `observability.py` `record_event` -> `logs/*.jsonl`
  (metrics + events tiers), level-gated (`observability_level`), file rotation,
  startup disclosure; `diag_event` delegates to it; per-request + model-lifecycle
  emission; `POST /v1/telemetry/events` for v3 client events.
- [x] `internal/log/` -> `logs/` reconcile; `observability_level=off` master kill
  switch over memory.py's legacy streams.
- [x] Aborted/stopped streaming requests now logged (`stop_reason=abort`).
- [x] Chat-template robustness: reject stop-less templates, validate vs the
  model's own eos tokens (v1.34.55).

**Backend TODO (mine):**
- [x] **Per-DOCUMENT sampler settings** DONE 2026-07-11 (v1.34.56-.58): `params`
  JSON column on BOTH `conversations` (v4) and `notebooks` (v5), shared
  `_encode_params`/`_decode_params`; threaded through create/update + API models +
  PUT allowed sets. Unifies "settings in browser vs server" -- sampler knobs join
  the system prompt on the server. Frontend below shares ONE binding.
- [ ] **memory.py stream CONSOLIDATION** (P2): spine now duplicates memory.py's
  `request_events`/`model_events` streams. Once live-verified, remove the dupes +
  retire the 3 legacy env toggles (`HEYLOOK_REQUEST_LOG_ENABLED` /
  `_MODEL_EVENT_LOG_ENABLED` / `_BASELINE_LOG_INTERVAL_SECONDS`); resource snapshot
  moves to the spine. Gated on full live verification.
- [ ] **Live-verify the spine end-to-end** (P2): confirm `provider=mlx` +
  `effective_loader` (text=mlx-lm / vision=mlx-vlm) + `stop_reason=abort` in
  `logs/metrics.jsonl` from real runs before removing memory.py streams.
- [ ] **`modalities` dim in `request_complete`** (P3): only
  `provider`/`effective_loader`/`is_vlm` captured; `modalities` needs
  `model_config` threaded to `_maybe_log_request_event`.
- [ ] **Never-stops health signal** (P3): flag models whose requests consistently
  hit `stop_reason=length`/`abort` (surfaces broken templates in the metrics).

**Frontend TODO (v3):**
- [x] **Per-document settings UI** DONE 2026-07-11 (v1.34.57-.58): ONE shared
  `settings.bindDocumentParams`/`hydrateDocParams`; chat.js + notebook.js both use
  it (no branched copy). Sampler drawer binds to the active conversation/notebook's
  `params`, hydrates silently on select, debounce-PUTs on change, carries forward
  on create; localStorage demoted to new-chat seed. **Browser/E2E UX check still
  recommended** (v3 has no unit tests).
- [ ] **v3 observability CONFIG + VIEW pages** (P2, owner-required): admin panel
  edits `observability_level`/retention via `/v1/admin/config`; a read page
  surfaces `logs/*.jsonl` (recent events/errors + metric summaries via
  `read_json_auto`). `js/telemetry.js` client logger -> `POST /v1/telemetry/events`.
- [ ] **Default sampler temp** (P3, owner: 1.0): no code default exists today
  (`settings.js emptySettings()` -> null; the `1.2` seen was a saved localStorage
  value). If a code default is wanted, set it in the new-chat defaults (frontend)
  or as a `None`->1.0 backend fallback -- decide where.

**Docs / rot:**
- [ ] `docs/observability_guide.md` has a redesign banner + corrected paths but
  its body still describes the legacy `internal/log/` 4-stream layout -- full
  rewrite pending (P3).

## Recently Completed (Phase 2 -- 2026-03-13)

- [x] Remove STT provider (`mlx_stt_provider.py`, `stt_api.py`, parakeet-mlx dep)
- [x] Narrow provider type to `Literal["mlx", "mlx_embedding"]`
- [x] Rename `embedding_gemma.py` → `embedding_model.py`, `EmbeddingGemmaModel` → `EmbeddingModel`
- [x] Dynamic backbone loading via `load_backbone()` using `mlx_lm.utils._get_classes()`
- [x] Pydantic V2 migration (`@field_validator`, `@model_validator`)
- [x] Stop-token utility extracted to `providers/common/stop_tokens.py`
- [x] Fix transformers 5.x VLM processor loading (4 patches in `_apply_transformers_patches()`)
- [x] Fix `eos_token_ids` null safety in `mlx_batch_text.py`

## P0 - Critical (blocks other work)

None currently.

## Slice 1 (in flight)

- [x] S1.1 -- per-request peak memory + KV bytes (v1.28.0, `be0f15f`)
- [x] S1.2 -- three-stream observability (v1.28.0, `2f9b03d`/`f28d52d`/`3641cf0`)
- [x] S1.3 -- byte cap on VisionFeatureCache (v1.28.0, `312db4e`)
- [x] S1.4 -- provider.warmup() + prefill_step_size (v1.28.0, `31e59a2`/`915dab6`)
- [ ] S1.5 -- batched docs + cleanup (in progress: STT removal + test cleanup + cache_keys refactor done; docs-audit items landing next)
- [ ] S1.6 -- LAN hardening (Caddy reverse proxy + optional `HEYLOOK_ADMIN_TOKEN`)

## Slice 1 gated work

- [x] S1.2b -- preset + import redesign: SHIPPED (presets.py registry, default_preset cascade -- see "C1 of S1.2b" in presets.py docstring). Stale-gated entry closed 2026-07-06 during docs audit; the import-defaults follow-through landed in v1.31.3/v1.32.0.

## P1 - High Priority (do soon)

### TOML Migration Completion
- [x] Integrate `--interactive` flag into model_importer.py (DONE v1.18.1)
- [x] Wire up ConfigEditor in import workflow (DONE v1.18.1)
- [x] Move profiles to TOML files, rename profiles, dynamic discovery (DONE v1.19.0)
- [x] Fix `ModelProfile.apply()` precedence bug (DONE v1.19.0)
- [ ] Test with: `heylookllm import --folder ~/test --interactive` (manual)

### Dependency Cleanup
- [x] Remove `mlx` optional extra (duplicated core deps) (DONE v1.19.0)
- [x] Purge unused deps: torch, torchvision, opencv-python, scipy (DONE v1.19.0)
- [x] Move `datasets` to `analytics` extra, `rich` to `scripts` extra (DONE v1.19.0)

### Stale Code Removal
- [x] Delete stale integration test `test_performance_monitoring` (DONE v1.19.0)
- [x] Delete `/v1/performance` stub and `/v1/performance/profile/{time_range}` endpoint (DONE v1.19.0)

## P2 - Medium Priority (scheduled)

### Build v1.20.0: Models Config TUI + CI Foundation

#### Models Config Command (Phase 4)
- [ ] Create `src/heylook_llm/commands/models_config.py`
- [ ] Implement TUI menu for model management (ConfigEditor in `config_tui.py` exists, 344 lines -- needs CLI wrapper)
- [ ] Add subparser to `server.py`
- [ ] Integrate into server.py CLI

#### llama-server Provider (Phase 5)
- [ ] Create `src/heylook_llm/providers/llama_server_provider.py`
- [ ] Subprocess management for llama-server binary
- [ ] HTTP client with SSE streaming
- [ ] Router integration
  - Note: LlamaCppProvider was removed in v1.21.0. llama-server provider would re-add GGUF support via subprocess rather than embedded library.

### Optimization Plan Doc Refresh
- [x] Update `docs/mlx_optimization_plan.md` -- phase 5 updated for v1.18.0 pre-filled cache pattern
- [ ] Mark plan as historical or rewrite deferred items as standalone proposals

## P3 - Nice to Have (opportunistic)

### CI/CD Pipeline
- [ ] Add GitHub Actions workflow (`.github/workflows/test.yml`, `.github/workflows/lint.yml`)
- [ ] Automated testing on commit/push
- [ ] Coverage reporting

### Benchmark Script
- [ ] Create `scripts/benchmark.py` (standalone, useful before shape bucketing)
- [ ] Token throughput, TTFT, memory usage metrics

### Build v1.21.0: llama-server Provider + GGUF + Benchmark
- [ ] llama-server provider complete
- [ ] GGUF extra activation (uncomment in `pyproject.toml`, blocked on provider)
- [ ] Benchmark script validates cross-platform performance

## Deferred (blocked on upstream)

### MLX Engine Optimization (remaining)
- [ ] Shape bucketing for prefill (needs attention mask correctness verification)
- [ ] `mx.compile` on decode step (deprioritized -- marginal gains vs. complexity; revisit if mlx-lm adds native compile support)
- [ ] Automatic draft model selection (vocabulary compatibility checking)
- [x] Full vision path unification (DONE v1.18.0 -- pre-filled cache pattern, no `inputs_embeds` needed)
- [ ] Vision + speculative decoding (pre-filled cache incompatible with speculative prefill)
- [ ] Radix cache for vision (pre-filled cache bypasses radix tree)

- [x] Error-chunk altitude: DONE in v1.33.0 -- provider raises typed GenerationFailed/InvalidGenerationRequest; batch/RLM now fail loudly; non-streaming client errors return 400. See docs/architecture/postmortems + CHANGELOG 1.33.0.