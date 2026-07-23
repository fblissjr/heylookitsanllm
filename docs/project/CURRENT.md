# Current Work

Last updated: 2026-07-23 (v1.39.1-.5 -- v3 preset/sysprompt data-loss fix +
shared preset bar + drift indicator + full E2E rebuild; jlens-mlx side-track
below unchanged since 2026-07-13)

UPDATE 2026-07-23 (v1.39.1-.5, frontend v3 + E2E, all heylook code):

- **Chat system-prompt data loss FIXED (v1.39.1) -- SOLID, live-verified.**
  The editor committed only on `change` (blur); closing the settings drawer
  with Escape or a route change removed the focused textarea before `change`
  could fire, discarding the typed prompt from both state and the server (a
  preset saved in that window captured `system_prompt: null` and would later
  erase the conversation's prompt on apply). Editor now commits state on
  every keystroke (notebook parity) and debounces the PUT (400ms, flushed on
  blur) -- no close path can outrun the save. Same session: chat top-bar gear
  added (in-context opener for the shared drawer, alongside the existing
  sidebar-foot/bottom-nav gears), sysprompt section now always expanded
  (was collapsed-when-empty), editor grew 3 rows -> 9rem-min auto-growing,
  drawer widened 22rem -> 26rem.
- **v1.39.2-.4: the drawer-close data-loss class fixed at its root, not just
  for sysprompt.** v1.39.4 found the v1.39.1 fix was a point patch: ANY
  commit-on-`change` field (sampler number inputs included) loses an
  unblurred edit when `close()` removes it while focused. `close()` now
  blurs the focused field before clearing the drawer body -- one fix at the
  convergence point all close paths share, not per-field patches.
- **Preset apply made explicit + legible (v1.39.2).** Preset select is now
  INERT (no apply-on-`change`); a dedicated Apply button, armed-confirmed
  ("Replace prompt?") only when it would overwrite a differing non-empty
  prompt. A live drift line ("Matches current settings." / "Differs from
  current settings...") updates in place on prompt keystrokes and sampler
  edits, `role="status"` (v1.39.5) so the flip reaches screen readers.
- **Shared preset bar (v1.39.3) -- notebook is no longer chat-only.** The
  preset section was extracted to `apps/heylook-frontend-v3/js/preset-bar.js`
  (`createPresetBar` + a `getPrompt`/`setPrompt`/`onStatus` adapter); the
  notebook page now contributes the same bar ahead of its sysprompt section,
  identical grammar to chat. Decision: apply writes the notebook's system
  prompt too (a preset is a prompt+sampler bundle everywhere).
- **`tests/e2e/` full suite run live 2026-07-23 -- 65/66 on the full run,
  every check green across runs (chat re-run 30/30); all new
  preset/sysprompt/gear checks passed first try.** The one failure is a
  model-behavior flake, now documented (`tests/e2e/README.md`): post-abort,
  the model can emit EOS immediately on the next turn, so an empty reply
  saves nothing and the "new send completes normally" check flags -- not a
  UI bug (`finishStream` deliberately drops empty completions).
- Both prior "REMAINING" follow-ups from the 2026-07-09 preset entry below
  are now closed: notebook reuses the preset bar (this session), and the
  "panel drifted from selected preset" indicator shipped as the drift line
  (v1.39.2).

UPDATE 2026-07-20 (v1.34.59-.65 + dep bump 0bfe60b -- all heylook code; jlens-mlx
entries below are unchanged from 2026-07-13):

- **Gemma-4 thinking, end to end (v1.34.60-.63) -- SOLID + live-verified.** Root
  cause chain: the stop-token gate rejected gemma-4's canonical template because
  `_read_eos_tokens` only read `added_tokens_decoder` (tokenizer_config.json),
  missing the `<turn|>` terminator (id 106) that lives in tokenizer.json's
  `added_tokens` -- template_info emptied, thinking leaked inline as plain text
  with channel markers stripped (v1.34.62 fix: union both files, same
  dual-source rule as `_read_special_tokens`). New `GemmaChannelParser`
  (streaming state machine, `thought` channel -> `thinking` field) selected via
  template sniffing. VLM path additionally dropped `enable_thinking` on both
  legs (text-only + `prepare_vlm_inputs_parallel`) and generated past
  end-of-turn (raw HF tokenizers don't absorb `generation_config.json`'s eos
  list; mlx-lm's `stream_generate` auto-wraps with only the single
  `eos_token_id`) -- fixed by `extend_eos_from_generation_config` at load +
  `run_generation` wrapping raw tokenizers itself (`ensure_gen_tokenizer`) with
  the full resolved stop set (v1.34.63). Also removed the decode-path hygiene
  patch (`tokenizer_hygiene.py`, v1.34.5x) that defaulted
  `skip_special_tokens=True` and stripped channel markers before the parser
  could see them. Live-verified on gemma-4-31b: vision + text, think on/off,
  1/2/large images -- the previously reported multi-image + thinking gibberish
  did not reproduce.
- **Qwen3.5 thinking fixed + hardcoded vocab ids removed (v1.34.64).** Its
  template pre-fills `<think>\n` into the generation prompt, so output started
  mid-block with no opening tag and the parser routed everything to content --
  new `prefills_thinking` detection + `initial_thinking` parser state. The
  token-level thinking mode keyed on hardcoded Qwen3 ids (151667/151668),
  silently failing to split for any other `<think>`-family vocabulary; the
  parser is text-based only now (same as the harmony/gemma channel parsers).
- **Thinking capability auto-detected from the template (v1.34.60).** A chat
  template that references `enable_thinking` (Qwen3 `<think>` blocks, gemma-4
  thought channels) now reports the `thinking` capability on `/v1/models`
  without a manual `models.toml` flag -- the v3 checkbox/composer icon appear
  for these models automatically. Messages API's hardcoded `<think>`-only
  parser also replaced with `select_reasoning_parser` (streaming and
  non-streaming), matching chat/completions.
- **Model-agnostic vision token budget (v1.34.64) -- SOLID, live-verified,
  ahead of the Q8 spike.** `vision_tokens` request field + per-model
  models.toml default + v3 drawer control (cap-gated on `vision`), mapped by
  duck-typing the loaded processor: gemma-4 discrete buckets (snap to
  70/140/280/560/1120), qwen2/3-VL continuous pixel budget, unknown families
  degrade to processor defaults. Vision-feature cache key carries the budget.
  Verified on gemma-4-31b + Qwen3.5-27B (pixel patches 630/2520/10080 at
  70/280/1120). Closes the TODO.md research item (collapsed there into this
  entry). The Q8 ACCURACY question (does a bigger budget improve detail QA?)
  is still open -- the new eval harness's vision tasks are the vehicle.
- **v3 composer polish (v1.34.60-.61).** Multi-image attach capped at 8 with
  an aria-live announcement when exceeded + per-image "Remove image N" labels
  (the attach strip/picker/paste/N-block store shipped earlier, v1.34.20 --
  this closes the cap + a11y gaps). Attach button and a new thinking-toggle
  button are now icon buttons (`.btn--icon`, 40px touch floor, same cap-gate +
  true/unset semantics as the drawer checkbox, kept in sync via
  `onSettingsChange`); toggle state is styled off `aria-pressed`, not a class
  (pattern recorded in `apps/heylook-frontend-v3/DESIGN.md` §7).
- **`mlx_cache_limit_gb` operational setting (v1.34.59).** Opt-in cap on MLX's
  buffer cache via `/v1/admin/config` (bounds idle RSS -- the allocator never
  returns freed buffers to the OS -- at the cost of realloc on the next spike;
  clearing restores MLX's own default). Plus a real bug fix: `DELETE
  /v1/admin/config/{key}` now re-applies immediately instead of only taking
  effect after a restart.
- **`tests/eval/` -- opt-in LLM behavior-eval harness (v1.34.65), NEW.** 13
  tasks / 7 programmatic judges (`color_mention`, `marker_leak`, `repetition`,
  `token_budget_exhausted`, `exact_word_count`, `non_empty_non_gibberish`,
  `substring_present`), generalized from that day's live-verification scripts.
  Covers the four bug classes fixed today: thinking split/leak, stop
  discipline, vision single/multi-image correctness, vision_tokens budgets.
  Needs an already-running server (never spawns one); not wired into
  `/test-suite` -- opt-in via `uv run python tests/eval/run.py`. Direction for
  an API surface + a v3 eval page: TODO.md.
- **Pyright noise triage (v1.34.65).** Real fixes: deprecated
  `datetime.utcnow`, untyped `= None` defaults, a latent bug (batch responses
  could hand pydantic `model=None` -> runtime 500, now coalesced), a float
  re-binding bug in `_format_bytes`, duck-typed route discovery. Systemic: all
  96 positional Pydantic `Field(default, ...)` calls converted to explicit
  `default=` keyword form (this pyright build only recognizes the keyword
  form -- the source of every false "arguments missing" constructor
  complaint). `pyrightconfig.json` already existed from a prior session; this
  pass didn't need a new one, just fixed the real signal it was surfacing.
- **Dependency bump (0bfe60b): mlx-lm a790972->15b522f, mlx-vlm
  8e2638b->c9e27b08 (0.6.5->0.6.6, 53 commits).** Consumed surface verified
  signature-stable (22 contract tests green); transformers stays pinned to
  5.5.4 via the deliberate `override-dependencies` (mlx-vlm HEAD declares
  `>=5.14`; decoupled on purpose, contract tests are the gate, not the
  version floor). Verified: full suite 1085 green + a live eval-bank A/B (old
  vs new pin) on gemma-4-31b + Qwen3.5-27B, no regressions (two chronic
  trivial-prompt degeneration flaps exist on BOTH pins; one budget-exhaustion
  flap is harness calibration, not model drift).
- **MTP research probe (no server code shipped; `internal/research/mtp_probe/`,
  local-only).** First live measurement of the gemma-4 MTP assistant head
  (open mlx-lm PR #1276's model class, vendored standalone) drafting for the
  daily-driver MoE: 50.8% greedy acceptance with thinking on (CoT preamble
  flatters it), 36.2% with thinking off (still net-positive, ~1.30x decode
  ceiling projected at ~5-10% drafter overhead). Found and documented a PR
  #1276 gap (`AssistantAttention` doesn't do per-layer-type KV head selection,
  mis-shapes SDPA on this pair's `num_global_key_value_heads=2`); commenting on
  the PR with the data is an optional owner action, tracked in TODO.md. Feeds
  the existing "Gemma-4 MTP self-speculation" TODO item; doctrine still
  applies (greedy-only, fingerprint gate invalid for any batched-verify run).

No obsoletions found in this file from today's work -- all additive. The
Phase 4 item 3 "enable_thinking tri-state (auto/on/off)" REMAINING note below
is still accurate and unaffected: today's composer icon shipped the same
binary true/unset semantics as the existing drawer checkbox, not a tri-state.

UPDATE 2026-07-13 (jlens-mlx: abliteration diff finding + explainer, sibling repo):
- **The abliteration diff landed -- and it's the finding the project chases.** Stock lens
  (`out/band-n14-stock`, `identity_ok: true`) fit clean overnight on the SAME corpus as the
  abliterated `band-n14-fixed`; `diff_lenses.py` run BOTH substrate directions.
- **Result (robust, substrate-independent -- both directions agree layer-for-layer):** the
  abliterated transport surfaces safety/refusal vocab MORE in the mid-late band (L32-42:
  `Safety`/`unsafe`/`unethical`/`dangerous`/`Cannot`/`violations`, CJK `安全风险`/`违反`, Russian
  `безопасность`), and SUPPRESSES geography (China/Europe) + retrieval verbs. The tallest raw-l2 bars
  are the illegible early band (`*`-junk); the SIGNAL is the coherent mid-late safety cluster (read
  the shape, not the peak -- same lesson as the legibility metric).
- **Why MORE, not less (the headline) -- CORRECTED 2026-07-13:** abliteration edits the TRANSPORT,
  not the readout. Heretic (confirmed by reading its source) does directional ablation:
  it orthogonalizes the residual-WRITING matrices (every layer's `attn.o_proj`/`mlp.down_proj`)
  against `r = mean(harmful) - mean(harmless)` -- tail blocks INSIDE the fitted Jacobian, not
  outside it. `model.norm` is untouched (bit-identical), so the diff is a PURE TRANSPORT
  difference. **Interpretation RETRACTED 2026-07-13 (second correction, same day):** the diff is
  NOT a content-conditional "disposition preserved" reading -- a per-prompt re-run
  (`scripts/per_prompt_diff.py`, `out/per_prompt_diff.txt`) showed a benign weeknight-recipe
  prompt lights up the same L32-42 safety band just as strongly (mean l2 596 vs 524-571 for the
  safety-adjacent prompts) and surfaces the same refusal vocab (Nothing/Impossible/cannot/unsafe).
  The effect is PROMPT-INDEPENDENT -- no benign floor, no evidence of a content-conditional
  internal state. The diff recovers abliteration's STATIC WEIGHT-EDIT FINGERPRINT: a refusal
  direction in vocabulary space, readable on any input because the weight edit is always present,
  localized to WHERE the edit lives (L33/L36, matching the weight footprint).
- **Cross-validated by an independent weight-footprint analysis** (`scripts/abliteration_footprint.py`,
  jlens sibling repo; `out/abliteration_footprint.txt`): dequantizing both 8-bit builds shows the
  vision tower bit-identical (LM-only edit) and the edit ~6x concentrated in the residual-writing
  matrices vs input matrices (at the quant floor); the per-layer weight-delta peak (L33/L36)
  co-localizes with the transport-diff safety cluster (L32-42) -- two independent measurements
  agreeing on WHERE abliteration lives.
- **Artifacts (jlens research repo):** raw diffs in `out/` (`diff_ablit_vs_stock.txt` + `…_hereticsub.txt`);
  tracked write-up + visual explainer at `docs/abliteration_diff.md` + `docs/abliteration_diff_explainer.html`.
  Both original caveats RESOLVED: benign floor FALSIFIED on the old pair then REVERSED 2026-07-14 on a clean
  matched pair (magnitude prompt-independent, content prompt-conditional, floor HOLDS); quant-converter match
  CLOSED 2026-07-13 -- self-converted the base (mlx-vlm 0.6.5) vs the mlx-community base is uniform
  ~0.004 drift, no tent, o_proj/down_proj at the floor (~8x below the abliteration signal,
  structureless) -- converter asymmetry cannot manufacture the finding
  (`out/converter_drift_base_vs_mlxcommunity.txt`).
- **Own controlled abliteration + clean matched pair (2026-07-13 PM).** Owner ran Heretic on their
  4090 to abliterate the base (Trial 144: 41/100 refusals from 89 baseline, KL 0.0282, tent centers
  o_proj 45.6 / down_proj 57.9). Self-converted BOTH base and `heretic-ours` to 8-bit MLX with
  mlx-vlm 0.6.5 (VLM-aware; keeps the vision tower) -> `modelzoo/Qwen/Qwen3.5-27B-{8bit,heretic-8bit}-ours`.
  Clean footprint (`out/footprint_ourpair_trial144.txt`): untouched matrices EXACTLY 0.0 (same base+
  converter), only o_proj/down_proj/linear_attn.out_proj change (zero leakage). TWO results: (a) the
  footprint method is CALIBRATED against ground truth -- it recovers the known tent centers within ~2
  layers; (b) the ablation locus is RECIPE-DEPENDENT -- Trial 144 peaks DEEP (L42-59) vs coder3101's
  SHALLOW (L33/36), so "L33/36" was coder3101-specific, not a model property. **Overnight: two band
  lens fits running detached** (`jlens out/run_matched_pair_fits.sh`) -> by AM, diff+footprint on the
  clean pair; prediction = safety cluster deeper (~L42-59), still prompt-independent.

UPDATE 2026-07-12 PM (jlens-mlx: first clean own-fit lens + the memory saga, sibling repo):
- **First clean-corpus full-band own-fit lens produced.** `out/band-n14-fixed`: 11 items, band
  16-47, `identity_ok: true`, ~4.25h, zero SIGKILLs. Qualitative readout (the honest test) on the
  Eiffel probe: L40-42 surface meaningful tokens (Paris/city/France) -- the lens works where it
  matters; L45-47 collapse to degenerate ` __`/` ____`, but that's the MODEL's own degeneracy
  (abliterated+quant), not a fit fault.
- **The legibility metric MISLEADS -- same failure as the old fidelity gate.** It ranked the
  degenerate J_45/46/47 HIGHEST (0.91-0.93) and the meaningful J_40 at 0.85, because the degenerate
  readouts agree with the model's own degenerate output. Reproduced with a clean corpus AND the new
  disposition-aware metric -> it's a metric problem, not a corpus one. Judge readouts qualitatively;
  a real disposition-aware metric is still open.
- **Memory model CORRECTED: peak scales with FITTED POSITIONS, not sequence length** (~63GB base +
  ~2.1GB/position; validated live). The caching-allocator SIGKILLs (transition exit-137) were fixed
  for real with `mx.clear_cache()` between items (jlens `e56fad6`); item 10's drop via
  `JLENS_MAX_FIT_SEQ` was an over-drop on the wrong (sequence-slope) model -- it would have fit.
- **Abliteration diff set up as the overnight run.** A stock-model lens (`Qwen3.5-27B-8bit-mlx`) is
  fitting on the SAME corpus (`out/band-n14-stock`, same token sequences) so only the model varies;
  by morning, `diff_lenses.py` (ablit vs stock) yields the transport-geometry difference abliteration
  introduced. Also this session: capture parity verified BIT-EXACT (`36d859b`), fit metrics store +
  dashboard finished, both fit-math branches code-reviewed clean.

UPDATE 2026-07-12 (jlens-mlx corpus incident + upstream GDN PR eval, sibling repo):
- **Corpus incident found + fixed.** The `band-n12`/`band-n12b` own-fits on the served abliterated
  Qwen3.5-27B were degenerate: mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects
  `enable_thinking=True`, so every on-policy completion collapsed into shared CoT-preamble
  boilerplate (62% of all fitted positions, 71% of on-policy). `band-n12b` was stopped at 9/11
  items (checkpoint kept); both fits' results are discarded — the fitter math itself (chain-vs-
  direct, kernel parity) is unaffected. Fixed in jlens-mlx (`238826e`/`951dd76`/`232b98b`):
  explicit `enable_thinking` (default False, matching heylook's own served default), role-aware
  off-policy spans, a 16-token sink floor, and a diversity gate that hard-fails a corpus this
  degenerate. Refit on the fixed corpus is next.
- **Upstream mlx-lm GDN differentiability PRs validated.** PRs #1389 and #1217 (both add
  differentiable gated-delta ops) are numerically correct (3-way gradient agreement, rel ≤2.6e-7)
  and remove the GDN kernel's `T≤128` cap that's been capping fit corpus length — 27B QLoRA
  ~145-150 tok/s / 38-39GB on either PR vs ~50 tok/s / 117.5GB on main, no inference regression.
  **Decision: do not fork mlx-lm for the fit path** — jlens-mlx's outer-layer design keeps fit-side
  numerics identical to the SHA the server serves; the fork stays eval-only. Full detail + the
  serving-relevant PR triage (#1486/#1456, #1515/#1532, #1526, #1077, #997):
  `docs/jspace_integration_plan.md` § "Observations & watch-items", 2026-07-12 subsection.
- No heylook (this repo) code changed today — pure jlens-mlx + docs session. TODO.md updated with
  the follow-up items.

UPDATE 2026-07-11 (jlens-mlx fitting + DRY settings drawer):
- **Fitting pipeline matured** (jlens-mlx sibling repo): exact reverse-mode CHAIN fitter
  (verified == direct, cos 1.0; the default), cotangent dim-batching (2.4x), a corpus builder
  (weighted strata + on-policy generation + role/think position masks), a sequence-length cap
  (drop-not-truncate), per-item checkpoint/resume + JLENS_FINALIZE, and `decode_corpus` (stores a
  readable corpus by default). Own-fits on the served abliterated Qwen3.5-27B: band-5L done;
  `band-n12b` (band layers 16-47, N=12, cap 128 = all on-kernel) running as of this date.
- **Perf reality (designer+verifier pass):** the chain fit is ~44 min/item, a full band ~7-8h, and
  there is NO config-level 2-3x. `chunk_size` is a dead knob; the trap is the GDN kernel **MAX_T=128
  cliff** -- items >128 tokens fall to the slow differentiable ops fallback (~10x slower + memory
  blowup, OOM-adjacent at 143GB/192). Added a guardrail (warn + kernel-eligibility metadata in the
  sidecar); rule of thumb: cap corpora <=128 for the served qwen. Real speedup = seq-tile the GDN
  scan (scoped, deferred -- exact since it's a recurrence).
- **Fidelity gate misleads -> legibility metric:** the final-logit-agreement gate ranked a degenerate
  near-target layer ABOVE a meaningful mid-band one (band-5L: ' __' junk beat ' Paris'). New
  disposition-aware `verify.legibility_report` ranks band layers by real-content-vs-junk, wired into
  the fit output + sidecar (15 tests).
- **Abliteration study tooled + ready:** control = `mlx-community/Qwen3.5-27B-8bit` (downloaded, same
  base+quant, differs only in abliteration; heretic KL 0.065 from base). `scripts/diff_lenses.py`
  drives `verify.diff`. Fit both bands, then diff.
- **Fit/apply capture parity is ASSERTED, not verified** (fit captures cache-less; apply uses a fresh
  cache the hybrid qwen3_5 requires) -> a cheap go-forward numerical check. Does NOT invalidate current
  lenses (identity KL~0 is consistent). The old "capture.py must be byte-identical" invariant was false,
  corrected.
- **DRY settings drawer (v3 Phase 2 / plan Phase 4):** the chat settings UI extracted into an app-shell
  **global slide-over drawer** shared by all 6 pages (sampling / global display prefs / per-page extras
  taxonomy). Edge cases preserved (focus guard, preset fingerprint diff, stale-textarea sysprompt),
  browser-verified, code-reviewed. `show_special_tokens` display pref gated (`wired:false`) until a
  render surface honors it (DESIGN.md §6).
- **Docs:** an end-to-end research report (`internal/research/jspace_jlens_end_to_end.md`) written, then
  two-agent critiqued + corrected; `docs/jspace_integration_plan.md` Part 2 extended (Neuronpedia
  refinements, `coderef/mlxui-core` prior-art for activation patching, the fit/apply parity item).

UPDATE 2026-07-10 (j-space + jlens-mlx): the j-space **apply** feature
(Jacobian-lens workspace readout) shipped in the ~v1.34.31-.35 range (per
CHANGELOG) -- `/v1/jspace/{models,analyze}`, the v3 `jspace` page, V1/V2
apply-parity (cos 1.0), the V4 hallucination router, gen-gate coordination, a
thread-stream crash fix. Lens **fitting** was EXTRACTED to a new sibling repo
**`jlens-mlx`** (this server only APPLIES -- same lean-scheduler pattern as Q6's
`rlm-heylook`). Two GREEN milestones there: apply-path parity + a baseline fitter
(Anthropic direct-VJP design -- norm outside J, no chain, no closed-form seed)
cross-checked vs Anthropic's torch `jlens` (J cos 1.0). Go-forward:
`docs/jspace_integration_plan.md` Part 2. Known issue: the served
`Qwen3.5-27B-abliterated` lens is `hf_model_name=""` / likely fit on STOCK Qwen --
treat its readouts as provisional until we own-fit.

UPDATE 2026-07-09 (v1.34.22): per-conversation system prompt editing +
saved presets shipped in v3. Backend: new `presets` table in the DuckDB
store (db.py), added ADDITIVELY via `CREATE TABLE IF NOT EXISTS` --
deliberately no `_SCHEMA_VERSION` bump (a version mismatch drops and
recreates every table, which would nuke existing data for an additive
change). Name uniqueness enforced in code on the store's single serialized
writer (`PresetNameTaken` -> HTTP 409), same rationale as the earlier
dropped messages FK. Presets are deliberately EXCLUDED from
`clear_all_data`/`POST /v1/data/clear` (config, not data) -- a test pins
this. New router `preset_api.py`: `/v1/presets` GET/POST,
`/v1/presets/{id}` PUT/DELETE, tag "Presets" (server's 8th API router).
409 name collision, 400 bad/empty fields, 404 unknown id. These are
UI-authored bundles (`{name, system_prompt, params}`) expanded
CLIENT-side into explicit request fields when applied -- deliberately
distinct from the server-side TOML sampler registry (now `samplers.py`
/ `ChatRequest.sampler`, renamed from "presets" 2026-07-20); no wire relationship. Frontend v3: chat settings
panel gained a per-conversation system-prompt editor (details/textarea,
PUTs `system_prompt` to the conversation on blur; a prompt typed before
the first send rides along on the implicit conversation create) and a
preset bar (select applies a preset = COPIES params into the sampler
panel + system prompt onto the conversation, LM Studio semantics, no
live binding; name input + Save creates or overwrites by name; armed
Del). `settings.js` gained `snapshotSettings()`/`applySettings()` and a
`lead` option on `buildSettingsPanel`. Tests: +16 store unit
(`tests/unit/test_preset_store.py`), +9 HTTP-level router tests
(`tests/unit/test_preset_api.py` -- the repo's first HTTP-level unit test
for a storage router, minimal FastAPI app + httpx ASGITransport); suites
880 green. E2E chat suite +3 checks (system-prompt persist, preset
save/apply round-trip, armed delete): 55/55 live. Follow-ups at the time:
notebook page could reuse the preset bar (the `lead` hook makes it cheap;
chat-first was deliberate); a "panel drifted from selected preset"
indicator was deferred. BOTH CLOSED 2026-07-23 (v1.39.2-.3, see the update
above): notebook now contributes the shared preset bar, and the drift line
is the drifted-panel indicator.

Phase: **PHASE 1 COMPLETE + E2E rebuilt (51 checks live-green) + mlx 0.32.0
upgrade DONE** (v1.34.11): v0.32.0 ships the real CompilerCache-teardown fix
(mlx#3619/PR#3628) -- proven by a discriminating A/B repro (tuple-output
compiled fn on a worker thread: SIGTRAP on 0.31.2, clean on 0.32.0); both
venvs upgraded; suites green (839 backend + 65 optloop-lib + E2E; the
cadence guard has a documented cold-shader false positive after mlx bumps,
see tests/e2e/README.md). Spec-decode TEXT baseline + first experiment
DONE (v1.34.12, mlx 0.32.0): classic 1B->27B draft decode is NET-NEGATIVE on
the bandwidth-bound bf16 target (composite 0.91 at num_draft=2, 0.96 at 4;
short-context turns +10% at nd=4 but long_context -40%) -- confirms the
"verification-based decoding, not classic draft" thesis. Full numbers +
verdict in docs/optimization_log.md; harness validated itself. TEXT was
photo-independent so it did NOT wait on photos.
VLM VISION BASELINES DONE 2026-07-07 (v1.34.16). The mrope "blocker" was just
the stale Mar-15 mlx-vlm fork -- pulling the owner's synced forks (mlx-vlm
0.6.5 #1529, mlx-lm 0.31.3 #1431; uv sync clean, mlx stayed 0.32.0) runs
gemma-4 dense/MoE AND Qwen3-VL clean through the manual vision path. No
wrap_language_model port needed (that v1.34.13 TODO is moot). Owner's synced
forks + downloaded models (gemma-4 8bit dense/MoE + assistant drafter),
per-model baselines (v1.34.14), model-name scrub (v1.34.15),
3 models safe-merged into models.toml (append; 13 tuned entries intact).
Baselines: dense gemma-4-31b-it-8bit = 15.3 gen_tps / 1592ms vision / 33.3GB;
MoE gemma-4-26b-a4b-it-8bit = 48.1 gen_tps / 524ms vision / 27.3GB. MoE ~3x
faster (bandwidth-bound dense vs dispatch-bound MoE = distinct optimization
profiles). Full numbers in docs/optimization_log.md.
NEXT (optloop lane): the MTP experiment -- MoE + gemma-4-26B-A4B-it-assistant-
bf16 drafter via mlx-vlm's draft_kind="mtp" (model exposes speculative_draft_
hidden / speculative_logits_from_hidden). Needs the bench's decode routed
through mlx-vlm's generate (not mlx-lm stream_generate) -- own focused pass;
the shot at the verification-based-decoding win classic 1B-draft (net-negative)
missed. Minor: bench_config vlm still points at Qwen3-VL (v1.34.13); switch to
a gemma-4 default with the MTP work. Branch: `main` (v1.34.25). The E2E/optloop
lane has wrapped; the version counter has since advanced through the v3 lane
(DuckDB+images v1.34.20-.21, presets v1.34.22-.24, legacy-app deletion +
drift-guard retirement v1.34.25). The MTP experiment above is the one item still
open in the optloop lane.

## Phase 1 results (this session, all TDD, each its own commit)

- **Item 1 (v1.34.1) SOLID + LIVE-VERIFIED**: streaming delivery
  unquantized (asyncio.wait on the chunk future). Live A/B on the MoE
  gemma-4-26B-A4B: 88.3 chunks/s client-observed, 11.1ms median gap
  (old ceiling: ~10/s, ~100ms). 31B dense gemma measures ~10.4 tok/s
  NATIVE -- it genuinely decodes at the old ceiling, don't use it to
  demo this fix.
- **Item 2 (v1.34.1) SOLID + LIVE-VERIFIED**: recorded tok/s = native
  mlx-lm generation_tps (perf_collector.headline_tps; fallback excludes
  queue wait); TTFT excludes queue_wait (kept as own field);
  /v1/messages prompt_tps formula fixed; trends + resource-snapshot
  averages success-only (live: a failed request left the trend at 50.0
  avg over 2 successes instead of dragging it to 33). RequestEvent
  gains prompt_tps -> flows to request_events.jsonl.
- **Item 3 (v1.34.1) SOLID**: close-timed-out executors quarantined
  (strong ref forever), never dropped-to-GC.
- **Item 4 (v1.34.2) SOLID**: reasoning parser per-request from
  _template_info (shared instance raced concurrent streams); strip
  pattern lru_cached; provider no longer builds a load-time parser.
- **Item 5 (v1.34.3) SOLID**: router TOCTOU (placeholder slot reserved
  under cache_lock; concurrent different-model loads can't over-commit)
  + idle unload checks generation_queue_stats (active+waiting) under
  the same lock as the pop.
- **Item 6 (v1.34.2) SOLID**: embedding pad_token -> eos_token at load.
- **Item 7 (v1.34.5) SOLID**: tests/contract/test_mlxvlm_surface.py, 22
  tests pinning the consumed mlx-lm/mlx-vlm surface, each naming its
  consumption site; anti-contamination guard vs the contract conftest's
  session mocks.
- **Item 8 (v1.34.4) SOLID**: already_configured matches resolved paths
  too (symlink-safe); re-import = PUT semantics.
- **Item 9 DRAFTED, filing = owner action**: ready-to-file upstream MLX
  issue at internal/backend/upstream_mlx_compilercache_issue.md (stacks
  re-verified from both .ips reports; minimal-repro attempt documented
  negative). Public post under owner's name -- review and paste.

Suite: 762 unit + 72 contract green. Two broken model configs noticed
during live verification (pre-existing, NOT from this work):
Ministral-3-3B-Base-2512 (mlx-vlm tokenizer_utils crash:
PixtralImageProcessorPil has no .vocab) and
gemma-4-E4B-...-int8-affine (missing k_proj/v_proj weights at load).
Worth pruning or re-quantizing.

This file is the handoff. Statuses are graded honestly:
**SOLID** = built + verified end-to-end · **HALF-BAKED** = works but a known gap remains ·
**UNCERTAIN** = done but verification was partial · **STUB/NOT DONE** = planned, not built.

ECOSYSTEM POSTURE (read before perf/provider work):
`docs/architecture/ecosystem_strategy.md` -- Python MLX stack is
the right rail but maintenance-mode upstream (SHA-pin, check the open-PR
backlog before workarounds, expect spec-decode etc. via sidecars like
dflash-mlx, never an mlx-lm release; Apple's frontier ships in mlx-swift-lm
-- feeds the Swift tripwire watch). Invariant to protect: run_generation()
single chokepoint + the provider seam in config.py. Distilled into the
plan's Direction section too.

THE PLAN for what's next: `docs/project/plan_2026-07.md` -- phased
(0: owner decisions, 1: correctness debt, 2: consolidation, 3: v2
retirement, 4: v3 hardening incl. rebuilding E2E, 5: perf). Start there.

Daily detail: `internal/log/log_2026-07-05.md` (v3 build) and `log_2026-07-06.md`
(crash fix, defaults audit, this cleanup). Older Slice-1 status that used to live
here: see git history of this file (all landed long ago).

## Evening additions (2026-07-06, after the doc above was written)

All in the plan with full rationale; one-liners here so nothing is missed:

- **Phase 0: every decision made except Q3 (E2E timing -- decide during
  re-plan).** Q4 Messages-first + namespaced extensions (consumer inventory
  done: shrug-prompter has a batch-parse bug to fix client-side); Q5 DuckDB
  app-state + JSONL analytics; Q6 RLM -> sibling repo `rlm-heylook`
  (exists, empty; client+research shape recommended); Q7 radix ->
  single-slot; images server-owned w/ ACCURACY-primary spike; batch
  collapses onto native upstream generators.
- **Swift rewrite: NO** -- decision + 4 tripwires in plan Direction;
  BaseProvider contract is the sidecar hedge.
- **mlx-vlm dependency strategy adopted** -- contract tests are Phase 1
  item 6 (transcribe the 07-06 drift audit); per-model shrink; upstream
  postmortems; contingency doc.
- **Measurement reality check (plan Phase 5, READ before any perf work):**
  server telemetry cannot detect a ~20% streaming regression (100ms poll
  quantization, queue-wait conflation, error-polluted means, native mlx-lm
  tps never read). Phase 1 streaming fix is now ALSO a measurement
  prerequisite; new honest-headline-metrics item; spec-decode baseline
  protocol pinned.
- **STATUS CHANGE -- optloop: app-level DELETED (v1.34.0).** Its benches
  never imported heylook_llm; zero cycles ever ran. optloop-lib is the only
  bench (new CLAUDE.md there; models.toml resolution ported in, 65 tests).
  Remaining tail: real-photo + long-context workloads BEFORE first
  baseline; thin HTTP serving-path bench AFTER Phase 1 telemetry fixes.
  NOTE: root pyproject pins UPSTREAM mlx-lm/mlx-vlm, not the local forks --
  fork-side wins don't reach the server until upstreamed/repointed.

## The plan (as executed 2026-07-05/06)

1. Rewrite frontend v2 as v3 per `docs/frontend_v3_spec.md` (the complete build
   contract -- §4 is the authoritative backend API contract, §8 the build order).
2. Fix whatever the build shook loose in the backend (§9 co-evolution ladder).
3. Audit the months-old model import/config/loading system.
4. Clean up the test suite + verify assumptions against current mlx/mlx-lm/mlx-vlm.

## 1. Frontend v3 (`apps/heylook-frontend-v3/`, served at /v3; /v2 untouched)

- **SOLID -- chat**: conversations CRUD, streaming w/ thinking blocks,
  edit/regenerate/delete via position truncation, stop = partial saved,
  status telemetry line, mobile drawer. 25 browser E2E checks + a /simplify
  pass. The most-verified surface in the app.
- **SOLID -- shared layer**: `js/page.js` createPage lifecycle (read FIRST
  before touching any page), hash router, table-generated `api.js`,
  `streaming.js` (keepalive comments, reader.cancel, abort-as-completion,
  built-in 503 retry via `onRetryWait`, mid-stream `{"error":...}` payloads
  -> onError), settings panel (null = backend-cascade contract).
- **SOLID -- notebook, models, perf, explore**: built by delegated agents
  against the spec, reviewed, 27 E2E checks (autosave, generate-at-cursor,
  scan/import/load/unload, no-polling perf, logprob chips + keyboard nav,
  390px viewports).
- **DONE -- visual design (2026-07-11)**: the impeccable `audit` + `polish`
  gates ran across all 6 pages + shell + drawer (slop-clean, scored 17/20).
  Fixed a mobile + a11y cluster -- notably delete/rename were unreachable on
  iPhone (hover-gated, no touch fallback) -- plus aria-live status/error, `<label
  for>` association, a real drawer focus-trap (`inert` #app, closes on hashchange),
  the mobile settings gear (FAB -> bottom-nav), explore chip titles, aria-current.
  Load-bearing rules new UI must honor: `apps/heylook-frontend-v3/DESIGN.md` §7.
  iPhone-Safari verified via viewport + touch-media emulation (19/19), not a real
  device.
- **HALF-BAKED -- j-space visualizer track (v1.34.36-.37)**: DESIGN.md gate
  cleared; SHIPPED + live-green in E2E: item 1 click-to-pin readout (strip
  rows + heatmap cells pin a detail panel with logit bars +
  first-answer-token emphasis; Esc/arrow-key walk; same-top-token echo;
  onset marker + token header row), the `heatmap_top_k` analyze extension
  (every heatmap cell now pins its full top-k -- reduced on-device), item 2
  layer-range slider (slot-based; click/drag/hover-preview/reset; pure
  client-side) + most-common-silent-tokens aggregation in the unpinned
  detail panel (click a row = echo-highlight where it wins), and a
  "provisional lens" badge from `/v1/jspace/models` meta (consumes the
  fitting track's sidecar provenance stamps). Next per TODO.md: live
  streaming (new SSE endpoint), interventions last.
- **NOT DONE -- cutover**: retiring v2 / promoting v3 is deliberately open
  until the owner has lived in /v3. Nothing blocks it.
- **DONE -- E2E in repo (v1.34.8)**: rebuilt under `tests/e2e/` (puppeteer-core
  + system Chrome; claude-in-chrome refuses localhost). 51/51 live-green vs the
  MoE gemma-4-26B-A4B: chat 24 + pages 27. Spawns its own server with an
  isolated HEYLOOK_DB_PATH so real data is untouched; each suite clears the temp
  DB, pages ends on the danger-zone clear. `cd tests/e2e && bun install` then
  `node run.mjs [chat|pages]`. Must run UNSANDBOXED (Chrome profile dir + Metal).
- **STUB -- batch page**: dropped from v3 scope on purpose (spec §6); the
  backend endpoint remains.

## 2. Backend changes v3 depends on (all committed, v1.31.1-v1.32.0)

- **SOLID -- crash fixes** (each root-caused from evidence, deterministic
  repro before/after, regression-tested):
  - radix snapshots materialized on the generating thread (v1.31.1) --
    "no Stream(gpu,N)" on prefix reuse; A/B proven 4/4 crash -> 4/4 clean.
  - persistent pinned-executor pool (v1.31.2) -- MLX threads are NEVER
    destroyed; per-request thread teardown aborted the whole process via
    MLX's thread-local CompilerCache destructor (SIGTRAP, two matching
    .ips crash reports). See tests/unit/test_streaming_executor_pool.py.
- **SOLID -- SSE error contract** (v1.31.1): generation failures are
  `data: {"error":{...,"code":"generation_failed"}}` + `[DONE]` (OpenAI),
  `event: error` (Messages), HTTP 500 (non-streaming) -- never assistant
  content. Contract-tested. v3 handles it; **v2/legacy do NOT** (empty
  response on failure -- accepted for retiring UIs).
- **SOLID -- error altitude (v1.33.0)**: provider now RAISES typed
  GenerationFailed / InvalidGenerationRequest (400) instead of yielding an
  is_error sentinel; batch and RLM fail loudly through their existing
  handlers instead of concatenating error text. Streaming wire contract
  unchanged (v3 needed no edits). Contract-tested; 774 passed.
- **SOLID -- request defaults** (v1.32.0): global sampler floor 0.1/512 ->
  0.7/4096 (`GLOBAL_SAMPLER_FLOOR`); imports stamp default_preset
  "balanced" (was deprecated "moderate"). UI-visible: bare chat requests no
  longer near-greedy or truncated at 512 tokens. Unit-tested; live-verified
  standard cache; the long-generation live check timed out (slow 32B model)
  so 4096-cap behavior is unit-verified only.
- **SOLID -- import defaults** (v1.31.3): KV quant is RAM-relative (>35% of
  unified memory), max_kv_size NEVER defaulted (it silently truncates
  context). models.toml migrated: everything <67GB -> standard cache; the
  122GB/155GB giants keep 8-bit KV uncapped; max_loaded_models=2,
  idle_unload_seconds=0 for this 192GB box.
- **SOLID -- config strictness** (v1.32.0): extra="forbid" (typos fail at
  load), kv_bits Literal[2,4,8], kv_group_size Literal[32,64,128],
  rotating-requires-max_kv_size validator, max_queue_depth is a real field.
  `quantized_kv_start` REMOVED (was dead config) -- strip it from any old
  models.toml. Importer size_gb now from safetensors bytes (name regex was
  returning params-count as GB).
- **UNCERTAIN -- radix gate for non-standard caches** (v1.32.0): lookup+store
  now bypassed unless cache_type=standard with no max_kv_size (fixes a
  documented silent-wrong-output risk). Unit-tested via mocks; NOT yet
  exercised live against a quantized-cache giant (122B/235B) -- behavior is
  "radix off for them", which is safe but means no prefix reuse there.

## 3. Model import/config audit -- findings fixed vs deferred

Fixed: everything in section 2 above. Deferred (documented, not built):
- typed GenerationFailed refactor (TODO.md, batch/RLM behavior change);
- re-import is skip-not-update and `already_configured` matches by id only;
- update-on-reimport flow;
- upstream MLX bug report (CompilerCache TLS destructor drops Python refs
  without the GIL -- .ips reports in ~/Library/Logs/DiagnosticReports).

## 4. Test-suite + library-drift cleanup -- DONE (v1.32.1)

- **SOLID -- library drift**: every load-bearing mlx/mlx-lm/mlx-vlm
  assumption verified against the INSTALLED libraries (0.31.2/0.31.3/0.6.3,
  transformers 5.5.4): no broken sites. Fixed 3x mx.metal.device_info
  deprecation; deleted 2 dead transformers patches + the dead VLM
  strict-load fallback.
- **SOLID -- test consolidation**: -26 tests without coverage loss (drifted
  fake provider file deleted, tautological classes removed, duplicates
  folded, uninterruptible timeout payload fixed). 760 passing (713 unit).
  tests/README.md scrubbed of false "pre-existing failures" claims.
- **KNOWN GAPS left on purpose** (noted in-place in the test files): real
  coverage for the prefill-convention logic (mlx_provider) and
  _extract_from_layer index math (hidden_states) needs small refactors to
  expose testable helpers -- the deleted tests only asserted their own math.
- One unexplained single-run flake earlier on 2026-07-06 (1/778, never
  reproduced; name not captured). Watch for recurrence.

## Invariants that bite (learned the hard way)

- tests/unit green with 0 unexpected skips (Metal-gated skips OK); any
  failure is a regression.
- NEVER destroy a thread that ran MLX work (process abort); executors are
  leased from _PinnedExecutorPool, never shut down.
- Radix snapshots must be mx.eval'd before insert; radix only for
  cache_type=standard without max_kv_size.
- Verification of /v3 in a browser: puppeteer-core + system Chrome;
  claude-in-chrome blocks localhost by policy.

## Blockers

None.
