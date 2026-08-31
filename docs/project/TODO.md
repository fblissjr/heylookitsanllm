# Persistent TODOs

Cross-session task backlog organized by priority.

*Last reviewed: 2026-08-30 (caught up through v1.79.43 on frontend branch);
docs-twins entry added 2026-08-31 without a full backlog pass*

## Observability follow-ups (2026-08-19, from the startup-record review)

- [ ] **Pre-warm load telemetry is dropped** (P3): a `--model-id` startup
  load runs in server.py BEFORE the lifespan resolves `observability_level`
  from the DB, so with telemetry enabled the most expensive load of the run
  is missing from events.jsonl/model_events.jsonl while every later load is
  recorded. Same pre-configure class as the fixed startup-record bug; fix =
  resolve settings (or replay the load event) before/after the pre-warm.
- [ ] **No test pins the lifespan ordering** (P3): log_startup_info must run
  AFTER apply_runtime_settings in api.py's lifespan; both unit tests call it
  directly with the level already set, so a refactor moving it back beside
  MemoryManager construction regresses silently. Needs a contract test that
  seeds the settings DB before app startup.

## Docs twins (2026-08-31)

- [x] **The `heylook-provider` skill has moved with the wire** -- CLOSED
  2026-08-31, verified locally rather than on report. The marketplace clone on
  this machine now reads `verified_against: "heylookitsanllm 1.79.50"` and
  carries `DELETE /v1/requests` in `SKILL.md`, `wire_reference.md` and its
  README; it read 1.79.42 with no such string while the work was unpushed. The
  skill session had reported it shipped three times before this and each report
  was unverifiable from here, which was the right call to hold -- the commits
  were real but local, so nothing on this machine could distinguish "shipped"
  from "not shipped" until the owner pushed. Standing note for next time: a
  peer's "shipped" means shipped in a checkout you may not be able to read.

## Test-harness + coverage gaps (2026-08-29, worked 2026-08-30)

Carried here so they outlive the `CURRENT.md` handoff block, which is
rewritten every session. Two of the three closed on 2026-08-30.

- [ ] **Rewrite `tests/e2e/suites/chat.mjs` around where chat state actually
  lives** (P2): the suite was RUN on 2026-08-30 and is still 33/46, so the
  v1.79.41 "the two selector rots were the whole static gap" claim is
  refuted. The real cause is architectural: `tests/e2e/lib/browser.mjs` seeds
  sampler settings into `localStorage` and expects the chat settings panel to
  reflect them, but since v1.65-66 chat hydrates that panel from the DOCUMENT
  (`hydrateDocParams` -> `applySettings(doc.params)`), so selecting a
  conversation overwrites the seed before the first assertion. The preset and
  system-prompt checks rest on the same stale model. This is a rewrite, not a
  patch; until it lands the suite is not a gate. The app is NOT implicated --
  `bun run e2e:render` drives the same real `/v3` page at 102/102 including
  its uncaught-page-error check.
- [x] **The gguf smoke arm -- COVERED 2026-08-30, no code fix needed.** The
  `llama-server exited with code 1` failure was the model's architecture
  (`qwen4exp`); the canonical build has since been rebuilt from a checkout
  that supports it. 30/30 on each of two models -- the conformance rows split
  across them because no single served model has both audio and a thinking
  block. The handoff's own diagnosis recipe (raise `observability_level`,
  reload, then load) was confirmed to produce the missing log.
- [x] **`/v1/models` over-reporting `vision` -- FIXED v1.79.43.** The
  capability now derives through `effective_loader_for_config`, the same
  resolver `MLXProvider.__init__` calls, so the advertised capability and the
  provider's image guard cannot disagree. Fails open on an unreadable
  `config.json`, inheriting the loader router's "only positive non-support
  degrades" rule rather than inventing a second policy.

## gguf chat templates (2026-08-30)

- [ ] **Re-check the `reasoning_effort` 500 claim** (P3): CLAUDE.md states
  Qwen3.8 accepts `xhigh|medium|low` and RAISES otherwise, surfacing as a 500
  from llama-server. A consuming session measured all four of
  `low|medium|high|xhigh` accepted on `unsloth_Qwen3.8-27B-UD-Q8_K_XL` at
  v1.79.42 with no 500 at any level, and has since WITHDRAWN it as a
  correction -- four values on one model does not generalise to a claim about
  the model or the server version when the variable is the chat TEMPLATE.
  One unknown is now closed: they re-checked `/v1/capabilities` and their
  server was still v1.79.42, so their arm ran against the EMBEDDED template,
  pre-sidecar-precedence. That model has a sidecar, so on v1.79.43 the same
  calls render through a different template.
  What is left is a genuinely open question, not a contradiction: the existing
  claim and their measurement can BOTH be true, about different templates.
  `reasoning_effort` is a chat-template variable and the documented 500 is a
  jinja exception raised inside the template, so "which values raise" is a
  property of the template, never of the model or the server. Any re-check
  must record which template was in force -- v1.79.43 logs it at every spawn
  -- or it varies a control neither arm held.

## API asks from a consuming client (2026-08-30, measured, UNDECIDED)

A session integrating heylook as a second inference provider sent
measurements against v1.79.42. Recorded, not accepted -- each is an owner
call. Its numbers are specific and reproducible; the reasoning about what
they imply is theirs, not a decision here.

- [ ] **Cancelling a NON-streaming request is impossible today** (P2, real):
  `/v1/messages` non-streaming builds the whole response before writing
  anything, so nothing polls `request.is_disconnected()` and an abandoned
  client's generation runs to completion. Observed by a consuming client: an aborted
  non-streaming run kept the GPU busy for its full remaining length and the
  next request waited behind it, while the same abort on a STREAMING request
  freed the server at once. (Their timings are deliberately not recorded --
  uncontrolled for model, quant, context and machine, and this repo keeps
  performance numbers out of tracked files.) On a server that
  serialises generation, that abandoned run blocks everything behind it. Two
  candidate shapes, neither chosen: poll `is_disconnected()` between tokens on
  the non-streaming path, or `DELETE /v1/requests/{request_id}` keyed on the
  `X-Request-ID` clients are already told to send. NB the client chose
  non-streaming deliberately -- a late refusal arrives in-band as an `error`
  event after headers flush, which a naive reader renders as model output --
  so "just stream" is not a neutral answer for them.
- [x] **Constrained decoding -- DECIDED AGAINST 2026-08-30, do not re-raise.**
  OWNER RULING: grammar-constrained decoding makes output quality worse in
  general, so this is not a feature this server wants -- not "not yet", and
  not a sizing question. Recorded as a decision rather than deleted, because
  it arrived with numbers attached and will otherwise be re-proposed by the
  next person who reads them.
  The reporter INDEPENDENTLY WITHDREW it the same day, and their reasoning is
  the part worth keeping: they had framed it as "shape errors become
  unrepresentable", which treats schema conformance as the only axis, and
  their own app's first invariant is descriptive prose quality. In their
  words, they "optimised for the checkable one because it was the one I had
  numbers for". They also retracted the evidence: the shape-failure table is
  still accurate as data but most of it was fixed prompt-side (an unstated
  precondition -- with no reference slots attached their prompt said nothing
  about references and the model invented a citation), so it was never demand
  for a grammar feature.
  The only residual either side would entertain, and NOBODY is asking for it:
  an opt-in per-request, per-model-capability form, so a caller could pick
  shape-safety for a job where prose quality is irrelevant (extraction,
  classification) while leaving ordinary generation unconstrained. That is a
  different, smaller feature and would have to earn its own case -- not
  inherit this one's.
- [ ] **`requests_active` is null for gguf, so no client can tell busy from
  idle** (P3): `GET /v1/admin/models/{id}/status` reports the MLX-side
  generation gate, and llama-server queues its own requests, so the field is
  genuinely null for that provider rather than accidentally unpopulated. The
  client fell back to timing a trivial generation -- which queues, so the
  probe changes what it measures. Exposing queue depth somewhere honest would
  close it.

## Sidecar-template follow-ups (2026-08-30 code review, v1.79.43)

Acted-on findings are in the changelog. These three were judged real and left
open; each names what would settle it.

- [ ] **A sidecar swap is invisible to `stale_reload_fields`** (P2): that
  field is the repo's own "saved value differs from what the running process
  has" truth, and it derives from models.toml. A `chat_template.jinja`
  created, edited or deleted next to a loaded model's weights changes what a
  respawn would use while models.toml is untouched, so the Models page shows
  the model fully in sync. The spawn log says what you GOT; nothing says what
  is running has gone stale. Done when a template file's mtime/presence feeds
  the staleness answer, or the limitation is recorded as accepted with a
  reason.
- [ ] **Re-measure the Qwen3.8-27B template facts** (P3): CLAUDE.md records,
  as live-measured, that ggml-org embeds the official template at 8952 bytes
  and unsloth a patched 9993, and that the difference decides whether two
  leading system messages render or 500. That entry's sidecar is 9708 bytes --
  a THIRD template -- so as of v1.79.43 the entry no longer runs the template
  that prose describes. Done when the two-leading-system-messages probe is
  re-run against the sidecar and CLAUDE.md either confirms or splits the
  claim. Note this compounds the `reasoning_effort` re-check above: both are
  template-dependent and the default flip moved the variable underneath them.
- [ ] **`_build_args` reaches the filesystem now** (P3): it is documented as
  pure and exercised by `tests/unit/test_gguf_argv_matches_metadata.py` with
  paths that do not exist. Sidecar discovery probes the model file's parent,
  and that test uses `model_path="/tmp/model.gguf"` -- so a
  world-writable `/tmp/chat_template.jinja` would make every parametrized
  case emit a spurious `--chat-template-file`. Nothing creates that file
  today. Done when the drift test builds under a tmp_path it owns, or
  `_build_args` takes the template as a required argument so it cannot probe.
- [ ] **"Fails open" is overstated for the vision capability** (P3): the
  guarantee is that only POSITIVE non-support drops `vision`, but
  `mlx_vlm_supports` returns a clean False when mlx-vlm is merely absent or
  broken -- so an install condition reads as a model property and every
  `loader=auto` VLM loses the capability at once, with one INFO line per
  model_type as the only signal. Sharpened by the uncommitted uv.lock already
  bumping mlx-vlm, which makes `uv lock --upgrade` a thing that can change
  which models advertise vision. Done when an import failure is
  distinguished from an unregistered model_type, or the docs stop claiming
  the stronger guarantee.

## Chat reliability (2026-08-13, plan: `plan_chat_orchestration.md`)

- [x] **Stream-end blank frame -- DONE v1.72.2**: post-stream adoption now
  assigns from `heylook_saved`'s rows synchronously; the wholesale GET is
  the no-rows fallback only; list DOM structure is renderMessages-only.
  Guarded by the render-suite swap check (shown red against pre-fix tree).
- [x] **Media by reference -- DONE v1.73.0 (schema v7)**: base64 relocates
  to a content-addressed per-conversation blob store at every message
  write; rows carry url sources (`/{id}/media/{media_id}`, immutable +
  cacheable); the generate saga inlines blob bytes at wire build; GC on
  last-reference delete (retention-safe direction). Conversation reads are
  text-sized now.
- [x] **Delete means delete -- DONE v1.73.0**: single-row
  `DELETE /{id}/messages/{msg_id}`; the v3 button calls it; positions keep
  gaps. `?after` truncation stays API-only. Render check pins
  neighbor-survival (shown red against the truncation-era client).
- [ ] **Intermittent Metal fault cascade / load-hang under load-unload
  churn** (P2, observed 2026-08-18 during Q7 verification): three
  incidents in one afternoon on a hot machine -- a mid-generation wedge, a
  0.8B weight-load hang after a MoE unload, and a
  kIOGPUCommandBufferCallbackErrorSubmissionsIgnored cascade that 500'd
  every later MLX request. The cascade REPRODUCED ON STASHED BASELINE
  code, so it predates the v1.75.0 cache rewrite; morning runs of the
  same bank were clean. Related: the streaming_utils quarantine
  warnings. Needs its own investigation (candidate mitigations: detect
  the fault signature and refuse-with-restart-hint; py-spy needs sudo).
- [ ] **mRoPE cache gate: fail-open + no config escape** (P3, review
  finding 2026-08-18): the gate keys on two private upstream attribute
  names; a rename fails OPEN (reuse re-enabled on a broken family) and
  there is no per-model `cache_reuse` config override to gate manually.
  Follow-up: a models.toml field (effect-classified) honored ahead of the
  attribute sniff.
- [ ] **Prompt-cache reuse for quantized/rotating CONFIGS** (P3, deliberate
  non-widening in v1.75.0): the config-level gate (cache_type standard, no
  max_kv_size) predates Q7 and was kept verbatim. Under the snapshot+
  native-trim design, EXTENSION reuse would be sound for those configs
  too (nothing is sliced); widening needs its own live verification on a
  kv-quantized model, not a ride-along.
- [ ] **retrySave drops media blocks** (P3, pre-existing): the unsaved-row
  Retry save re-POSTs `msg.content` (flattened text), so a media message's
  blocks would not survive it. Unreachable today -- generation refuses
  while an unsaved row exists and unsaved rows are only minted by
  generation flows -- but the day an unsaved row can carry media, Retry
  save must POST `content_blocks`.
- [x] **Per-message model attribution UI -- DONE v1.74.0**: a muted
  per-row label (`.message-model-note`) rendered only while the thread
  MIXES models; rides msgSignature so rows rebuild exactly when the label
  appears. Data since v1.73.0 (fresh-row commits stamp; continuations keep
  the anchor's stamp).
- [x] **/v1/models is ~1.7s on a 29-model registry -- FIXED v1.74.1**:
  the reasoning_effort template probe shipped uncached (its sibling was
  lru_cached) and re-read every MLX model's template files per call.
  Measured 1650ms -> 1ms live. Cache-property tests pin BOTH probes
  (shown red against the uncached version).

- [x] **Phase 0 hardening -- DONE v1.64.0**: loud stream-guards (+ a
  `pendingSave` latch closing the Stop-then-act window between stream
  release and the partial save landing), unsaved-row "Retry save"/"Discard"
  honesty + destructive-op-and-send lock, reconcile-on-saga-end re-fetch
  (unsaved rows survive adoption), thinking-block editing in the message
  editor. All guarded by `e2e:render` (23 checks, each new one shown red
  against a pre-fix tree first), incl. an iPhone-emulation boot (viewport +
  touch + hover:none via CDP) for touch reachability.
- [x] **Phase 0.5 iOS hidden-row bug -- DONE v1.79.1**: reproduced on an
  iOS 26.5 simulator (Safari 26.5) with a control: with `content-visibility:
  auto` the thread opened 1046px above its end and the saved row landed
  mostly under the composer; with it off, the landing was exact. The row
  was never blank -- it was displaced: skipped rows report their 3rem
  estimate until WebKit's lazy relevance check lays them out, and WebKit
  has no scroll anchoring (`overflow-anchor` unsupported) to absorb the
  shift. Fix: the optimization is gated on `@supports (overflow-anchor:
  auto)`; editor close re-aims at its row. Instrument that saw it (no
  suite can): a same-origin injecting proxy + `simctl openurl` +
  screenshot loop, recipe in the 2026-08-20 session log. v1.79.2 added
  chat's resume sync (`refreshAfterResume`: visibilitychange/pageshow
  re-adopt the store) + the prompt editor's flush-on-hide; NOTEBOOK still
  lacks the resume refetch (it gets the flush via the shared factory) --
  port it when notebook is next touched. Open follow-up:
  the keyboard-dismissal drift after Save is only re-aimed, not prevented
  (viewport `interactive-widget` is not in Safari 26's notes).
- [x] **Phase 1 backend -- DONE v1.65.0**: `POST/DELETE
  /v1/conversations/{id}/generate` (append/regenerate/continue, Messages
  SSE + `heylook_saved` final event, per-conversation 409 arbitration,
  server-owned persistence incl. disconnect, truncation commits only with
  its replacement row). 17 contract tests; spec §4 updated.
- [x] **Phase 2 client cutover -- DONE v1.66.0**: chat generates via
  `/generate` (streamGenerate parser in streaming.js; Stop = DELETE;
  teardown = fetch abort + server disconnect-persist). Verified: render
  23/23 unchanged through the cutover, live chat suite 45/45 (three checks
  rewritten off the old wire), gguf matrix 7/7 on DeepSeek V4 Flash.
- [x] **Disconnect-persistence test -- DONE v1.74.0**: live chat suite
  reloads the page mid-stream (no Stop; the beforeunload dialog is
  accepted) and asserts the detached server task persisted the partial,
  server-side by id and client-side after reload. Green live 2026-08-18.
- [x] **Muse-Glimmer 30B -- RESOLVED 2026-08-13 (the parse was never
  broken)**: the "all output in reasoning_content, content empty" report
  was an always-reasoning model given a 100-token budget -- it burned it
  all in the harmony ANALYSIS channel and never reached the final channel.
  With a real budget, llama.cpp b10353+ parses it perfectly (the model
  card confirms; we're on b10416). Entry corrected: `supports_thinking =
  true` (it reasons every turn; template ignores enable_thinking), the
  since-withdrawn `--reasoning-format none` workaround REMOVED, and
  `--chat-template-file` points at the fixed HF template downloaded beside
  the weights (this GGUF predates the card's template fix -- 7167 vs 9992
  chars; the fix normalizes "Reasoning effort"->"Reasoning strength" in
  system prompts; drop the flag after re-downloading the GGUF). Vendor
  sampling temp 1.0 / top-p 0.95 / top-k 64; never stop on `<|eom|>`.
  Generate matrix 7/7 incl. thinking persistence and analysis-only abort
  partials. LESSON, again: starved budgets mislead about CORRECTNESS, not
  just perf -- an empty content with a fat reasoning_content means "ran
  out mid-think", not "parser broken". Alternative if upstream
  stalls: heylook's own harmony-channels parser via a gguf template_info --
  but that cuts against the "never re-parse another engine's split"
  invariant and only makes sense with format=none (engine not splitting),
  design it deliberately if at all.
- [ ] **At-rest encryption for the conversation store** (future state,
  owner ask 2026-08-13): conversations/messages (prompt inputs/outputs)
  sit plaintext in the DuckDB file. VERIFIED on the installed duckdb
  1.5.5: native encryption works -- `ATTACH 'file' (ENCRYPTION_KEY ...)`
  round-trips and a key-less open is refused. Vehicle when picked up
  (owner direction 2026-08-13): connect-then-ATTACH in db.py, key held
  ONLY in 1Password and fetched at server start via `op read` (the owner's
  gemini-bridge pattern in fb-claude-skills) -- the Touch ID popup at
  launch is the human-in-the-loop: with the server down, ANY attempt to
  read the key (Claude included) fires a popup the owner can deny, which
  the silent-same-user Keychain cannot offer. One fingerprint per server
  start; launchd auto-start becomes impossible (a boot service cannot
  fingerprint); a down 1Password agent blocks start, fail-loud. iOS Safari
  is UNAFFECTED -- the server holds the key and decrypts, the phone never
  touches 1Password. Per-ACCESS fingerprinting was considered and
  rejected: approvals render on the Mac (stalls every phone read) and the
  server must see plaintext to build prompts at all -- an LLM chat server
  is structurally a plaintext-processing machine while it runs. Turning
  encryption on is a fresh start or one-time manual copy per the
  no-migration policy.
  THREAT-MODEL BOUNDARY (do not oversell, discussed 2026-08-13): this
  protects the FILE at rest -- backups, cloud sync of the raw .duckdb,
  other user accounts, copied disk images (FileVault covers only the
  powered-off disk). It does NOT protect against any live process running
  as the owner: the key must be readable at server start, and the running
  server serves decrypted conversations over the loopback API, whose
  conversation routes carry no auth and whose opt-in API key exempts
  loopback by default. "An agent with my shell can't read it" requires
  harness-side sandbox deny rules + enforced API auth, not encryption.

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
- [x] **Gemma-4 QAT q4_0 swap -- CONVERTED + A/B DONE 2026-07-20, swap is
  the owner's call**: own conversions from the owner-downloaded unquantized
  QAT checkpoints (own-convert beat adopting OptiQ), 4-bit affine group-32
  matching the q4_0 32-block lattice per docs/mlx_conversion_guide.md:
  `modelzoo/google/gemma-4-{26B-A4B,31B}-it-qat-4bit-g32-mlx` (16G / 19G,
  ~5.2 bits/weight avg, vision towers bf16 via auto-skip), registered with
  Google-recommended sampler defaults (temp 1.0, top_k 64, top_p 0.95 --
  also added to the two 8-bit dailies). VERDICT: on the fixed v1.38.1 stack
  the full eval bank is **13/13 for all four models** (both QAT vs both
  8-bit dailies) -- parity. Every deficit seen mid-A/B (synthetic-image
  refusals, two-image failures, 31B thinking-skip, CJK degeneration on long
  thinking) traced to the server, not the quants: the transformers-5
  stop-set regression (fixed v1.36.1) let generation run past `<turn|>` far
  out of distribution, compounded by the pre-refactor sampler cascade. RAM
  halves (26B: 16G vs ~28G; 31B: 19G vs ~33G). Caveat before making QAT the
  daily: the bank's images are synthetic; run a real-photo spot-check.
  The QAT assistant heads (downloaded, unconverted) pair with the MTP item.
- [x] **finish_reason reports "stop" on budget-exhausted responses -- DONE
  v1.39.14** (P3, found during the 07-20 QAT A/B). Both non-streaming
  builders hardcoded "stop"; mlx-lm's own reason is now scraped in
  `ChunkTelemetry.absorb` (the one-scrape rule) and latched, so a trailing
  empty chunk cannot erase it, and both `/v1/chat/completions` and
  `/v1/messages` report it. The Messages converter already mapped length ->
  stop_reason -- it was never fed one. The eval bank's token-count
  workaround is retired (kept only as a fallback for servers that report no
  reason). Live-verified: max_tokens=12 -> "length", natural -> "stop".
- [ ] **Trim the two surviving architecture KEEPs** (P3, from the 07-20
  architecture audit; exact section lists in the session log): config.md
  (drop the field tables / TOML examples / schema dump / troubleshooting;
  keep the three design-reversal records + validation rationale) and
  mlx_provider.md (drop 1.4-1.6 key-file/function inventories + 2.1 + 5;
  keep all of section 4 -- the invariants). Philosophy: design records
  only, mirrors die.
- [ ] **frontend_v3_spec.md post-cutover slimming** (P3, gated on Phase 3
  v2 retirement): the spec is a historical BUILD contract; once v2 dies,
  slim it to section 4 (the living API contract) + the decision records,
  and let frontend_v3.md carry the map.
- [x] **Unify the parser strip/holdback layer -- DONE v1.39.12** (P2, from
  /simplify 2026-07-20): declared-specials stripping is now ONE
  implementation -- `reasoning_parser.StripSpecials`, composed by the
  factory over whichever routing parser it picked (and only when the model
  declares specials). Its rolling per-kind holdback is sized by the STRIP
  SET (longest tail that is still a proper prefix of some declared
  special), so the harmony/gemma boundary leak and PassThrough's total
  absence of holdback are both closed, and the holdback now covers
  non-`<`-shaped specials (Mistral's `[INST]` family) that the old
  `rfind("<")` scan could never hold. Fixed alongside: harmony's dead
  final-flush partial-strip (the abort-mid-token garbage flush, gemma's
  2026-07-20 fix ported to a shared `_strip_partial_token`), and the
  duplicated `_safe_prefix_len` bodies collapsed to one free function.
  Eight failing-first tests (harmony abort pair, cross-parser boundary
  straddles, non-`<` specials); suites green. Design record:
  docs/parser_strip_unification.md.
- [x] **E2E checks for the 2026-07-20 v3 features -- DONE v1.39.9** (all
  four landed in chat.mjs's capability/thinking/image section, live-green
  73/73): (1) attach 9 -> 8 thumbs + aria-live cap message; (2) thinking
  toggle aria-pressed round-trip + wire contract (enable_thinking absent
  when off -- never false -- true when on, asserted on captured request
  bodies); (3) `#set-vision_tokens` present with min 16 / max 16384 +
  localStorage round-trip, absent on a non-vision model (negative model
  discovered from /v1/models, never loaded); (4) thinking `<details>` block
  renders with non-empty body on a real thinking generation (branch-on-
  empty per the suite hardening principle). Plus an image-message
  round-trip (content_blocks persist + render + survive reload) beyond the
  original list. README descriptions updated.
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
- [x] **Eval-gate reminder hook + /eval-ab skill -- APPROVED + BUILT
  2026-07-20** (local `.claude/` config, not committable by design; the
  tracked piece is `scripts/dev_server.sh`, the isolated live-server harness
  both lean on). Scoping that was approved: owner's constraints: no over-testing,
  no auto-spawning heavy servers, no 3-hour runs on small changes, no
  one-model-one-prompt tunnel vision. Scoped design that satisfies them:
  (1) the hook is INERT TEXT ONLY (hookify context injection on edits to
  reasoning_parser/thinking_parser/template_info/stop_tokens/vision_budget/
  generation_core) -- it never runs anything; (2) it names the MINIMAL
  category for the touched subsystem via `run.py --tasks` (parsers/template
  -> thinking,stop; vision_budget/vlm -> vision; stop_tokens/generation_core
  -> stop) -- a smoke tier of 2-4 tasks x 1 fast model, ~1-2 min, not the
  13-task bank; (3) it only suggests running against an ALREADY-RUNNING
  server (dev-server skill's reuse-first rule) -- if none is up, the
  suggestion is "queue it", never "spawn one"; (4) full bank x multiple
  models stays explicit-ask-only (pin bumps, QAT A/B, releases). /eval-ab
  is the explicit-ask wrapper: baseline JSONL storage + diff + the
  rerun-flapped-tasks-before-concluding discipline from the 07-20 pin bump.
  Build neither until the owner approves this scoping.
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
- [ ] **Engine coverage, phases 1-4** (P2): one shared engine classifier instead of the
  copy now in `tests/smoke` + `tests/eval`; both harnesses REPORT which engines a run
  spanned and refuse to call an absent one green; then the same-feature-two-mechanisms
  checks (audio must 400 loudly on MLX, thinking capability, thinking depth, chat-template
  source). Plan: `docs/project/plan_engine_coverage.md`. Phase 0 = `tests/smoke/` (done).
- [ ] **Standing golden gate for `/v1/jspace/analyze`** (P3): freeze onset top-k + features,
  tie-aware calibrated epsilon -- turns the one-time V1/V2 parity into a wired
  regression gate.
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

- [x] **Cap-gated `enable_thinking` can be pinned invisibly -- FIXED
  v1.39.10**: `samplerParams(caps)` drops capability-gated keys
  (enable_thinking, vision_tokens) at request-build time when the current
  model lacks the cap; chat/notebook/explore each pass their page's caps.
  The cache deliberately KEEPS the value (switch back to a capable model
  and the control + value return) -- only the wire is filtered. E2E: the
  cap-filter wire check asserts on an intercepted+aborted request to a
  never-loaded negative model.
- [ ] **Presets vs TOML registry: dual-source by name** (P3, design decision):
  user presets are client-expanded only; `ChatRequest.sampler` resolves just
  the bundled TOML registry. If saved presets should ever work by name from
  the raw API/CLI, make the registry dual-source (TOML + DB rows, name-unique
  across both) instead of growing a second wire path. Deliberately NOT done
  in v1.34.22 -- client-side copy semantics (LM Studio) was the owner's ask.
- [x] **Notebook page preset bar** -- DONE v1.39.3: the bar was extracted to
  the shared `preset-bar.js` (createPresetBar + getPrompt/setPrompt/onStatus
  adapter) and notebook contributes it ahead of its sysprompt section.
  Decision: apply DOES write the notebook's system prompt (a preset is a
  prompt+sampler bundle everywhere; the armed confirm guards the overwrite).
- [x] **"Panel drifted from selected preset" indicator** -- DONE v1.39.2: live
  drift line under the preset select ("Matches/Differs from current settings"),
  updated in place on prompt keystrokes + sampler edits; apply became an
  explicit button (selection is inert), armed-confirmed only when it would
  replace a differing non-empty prompt.
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
- [x] **Wire `show_special_tokens` render consumers -- DONE v1.79.6** (commits
  `a7a3b3f`, `447cb32`, `b489ab1`): `show_special_tokens` display pref wired
  in v3 settings drawer on chat & notebook; new opt-in request field on
  `POST /v1/messages` and `POST /v1/conversations/{id}/generate` bypassing
  the declared-specials filter; assistant-stored declared specials stripped
  before replay to prevent control-token prompt injection. Display panel
  gated to pages declaring support (`displayPrefs`). Request-schema parity
  guard (`tests/unit/test_request_schema_parity.py`) pins wire consistency.
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
  `capabilities.py::infer_model_capabilities`, `model_importer` entry-build,
  `loader_routing._modalities_of` raw-dict fallback), then drop the mirror + the
  bool. Do NOT do piecemeal -- it's a coordinated removal.
  (`model_service._raw_to_scanned` was migrated 2026-08-07 -- it read
  `config["vision"]`, which no entry builder writes any more, and so reported
  `vision:false` for every scanned model of both providers.)
- [ ] **Audit the other five v3 pages for the status-area shape** (P2, found
  2026-08-07): the models page wrote an error and then awaited an internal
  refetch that cleared it on success, so it had never once shown a load failure
  since the page was built (fixed v1.50.2 via a `keepStatus` flag). For chat,
  notebook, explore, perf and jspace: find any handler that writes an error and
  then awaits a refresh whose success path clears it. Unknown whether any share
  the shape -- this item is the check, not a claim that they do.
- [ ] **`test-audit` over the payload/cascade tests** (P2, found 2026-08-07):
  two green-but-blind escapes in one session, both assertions written from the
  perspective of the case the author had in mind. One actively PINNED the bug
  as correct (`"chat_template_kwargs" not in unset`). Specific question, not a
  general sweep: does any other test assert a key's ABSENCE where the sent
  VALUE is the contract? Grep `not in payload` / `not in body` across `tests/`
  and review each hit.
- [ ] **Derive gguf `modalities`/`supports_thinking` at LOAD, like MLX**
  (P2, found 2026-08-07, Wave-1 derive-at-load coupled): the gguf importer
  STORES both in the entry, so an entry written before v1.49.4/.6 under-reports
  forever. Measured on the local fleet: every gemma-4 GGUF's entry says
  `["chat","vision"]` while a fresh scan of the same files derives
  `["text","vision","audio"]` + `supports_thinking: true`. Since v3 gates all
  modality UI on capabilities, those models show no thinking toggle and no audio
  attach even though the server serves both. `gguf_metadata` already reads
  headers cheaply (stops at the last requested key) and `modality_detect` has the
  mtime-cache precedent, so a `GGUFModelConfig` validator mirroring
  `MLXModelConfig._resolve_modalities` is the shape. NOTE this does not fix
  EXISTING entries: a stored value is indistinguishable from a deliberate
  override, so those need a re-import (or the two keys deleted) either way --
  deriving is what stops it recurring.

See also the Phase 6 "per-model SIDECAR ARTIFACTS" note (draft model / j-space
lens / future LoRA managed as a group on the admin CRUD surface).

## Qwen3.5-27B thinking-path repetition collapse (found 2026-08-07, P1)

**Observation.** `Qwen3.5-27B-8bit-mlx` degenerates into repetition on the
thinking path at large budgets. Same prompt ("What is 12 times 13?"),
`enable_thinking = true`, non-streaming:

| max_tokens | result |
| --- | --- |
| 600  | `thinking=1349ch content=0ch`, coherent opening |
| 2500 | `thinking=13070ch content=0ch`, opens `ejahterejahterejahter Consor...` |

Reproduces with `sampler="thinking"` too (`thinking=48ch` then junk).
`Qwen3.5-0.8B-MLX-8bit` shows the same shape (`门门门门...`, `款款款款...`), so it
is not obviously size-specific.

**Not caused by the v1.50.x work.** For an explicit `enable_thinking=true`
request the resolution is byte-identical before and after (old:
`effective_thinking_flag(True, provider)` -> True; new: cascade -> True), and
the prompt side never changed. What DID change is visibility: the trace now
lands in the `thinking` field instead of being mislabelled as `content`, so
this was always happening and was simply harder to see.

**Why it is worth real time.** This is the daily-driver-class model, the
collapse is in the reasoning trace rather than the answer, and two of the
project's own subsystems are plausible causes.

**Hypotheses, roughly in order of suspicion.** Each is falsifiable:

1. **The `thinking` sampler's anti-loop overlay is causing the loop.** The
   overlay applies `presence_penalty = 1.5` (`src/heylook_llm/data/samplers/thinking.toml`,
   slimmed to loop control in v1.45.0) and was tuned against a GEMMA repetition
   loop, never against Qwen3.5. A penalty that pushes hard away from recent
   tokens can drive a model into novel-token gibberish, which is what
   `ejahter`/`Consor` look like. FALSIFIED IF collapse rate is unchanged at
   `presence_penalty = 0.0`.
2. **The vendor layer picks bad values for this checkpoint.** v1.45.0 reads
   temp/top_p/top_k from the model's own `generation_config.json` above the
   floor (`samplers.load_vendor_sampling`). Check what it actually reads for
   this model dir and whether those values are sane for long generations.
   FALSIFIED IF collapse persists with vendor values overridden by the floor.
3. **Long-context degradation of the checkpoint/quant itself.** 8-bit Qwen3.5
   at multi-thousand-token self-generated context. FALSIFIED IF a different
   quant or `Qwen3.5-27B-8bit-ours` behaves differently under identical
   sampling.
4. **Radix/prompt-cache interaction.** Qwen3.5 is HYBRID (KVCache+ArraysCache)
   and CLAUDE.md already records limited radix correctness there — ArraysCache
   cannot trim to a prefix. FALSIFIED IF collapse reproduces with the radix
   gate off (it should already be off: the gate requires `cache_type=standard`
   with no `max_kv_size`, so CONFIRM that first rather than assuming).

**Do this first, before any hypothesis.** Run thinking OFF at the same budgets
(600/1200/2500). If it collapses there too, this is not a thinking-path bug at
all and hypotheses 1-2 are dead on arrival — the name of the item is wrong and
the search should start at 3-4. This is one server start and three requests.

**Measurement discipline (learned the hard way twice on 2026-08-07).** Single
unseeded runs on this stack are SAMPLES, not measurements: the DSpark A/B this
morning came out 11.7 acceptance points apart on two nominally identical runs,
and an E2E check turned out to fail 1 run in 6 from model nondeterminism alone.
So: pin the seed, repeat each cell at >= 3 seeds, and report the SIGN before
the magnitude. Report "collapsed / did not collapse" as a count out of N, not a
character length from one run.

**Related open thread:** `Qwen3.5-27B-8bit-ours` and the abliterated pair are
the jlens study models. If the collapse is checkpoint-specific, that matters
to the lens work too -- see `docs/jspace_integration_plan.md`.

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

**Config-editor / audit follow-ups (2026-08-11, sourced from the four-agent
audit; design context `internal/research/expert_offload_design_frontend.md`
+ `_backend.md`):**
- [x] **Chat model-switch hardening G1+G3 -- DONE v1.57.0** (G2 v1.55.0):
  caps-gated `toWireContent` with per-message drop disclosure (staged
  attachments still block -- asymmetry commented at both sites); residency
  dots + pre-switch warning (Cancel / Switch anyway; Send with the
  unconfirmed target commits) + Load button on the chat bar. NB the
  LOAD-COST half of that warning was removed in v1.62.3 (owner rule:
  only loss gates, cost is disclosed) -- only incompatible-media
  warnings gate a switch now; the cost is stated by the dots, the Load
  button, and a live pre-first-token status on Send. Still open
  from doc §15: G4 context estimate, G5 per-message attribution (waits for
  a `_SCHEMA_VERSION` bump), F14 switch-lock during a pending load.
- [x] **Fit meter -- DONE v1.60.0** (P2, the frontend design doc's stated
  heart): `heylook_llm.ram_fit` (extracted from `ram_report.py`, which now
  renders it), `POST /v1/admin/models/{id}/fit` with the provider-derived
  `hard_working_set` (MLX-FAIL/gguf-WARN), v3 Memory-fit section + Load
  gating on FAIL. Fit stays server-computed. The §5 "observed after load"
  line landed in v1.62.0.
- [x] **Server-owned `POST /v1/admin/models/{id}/reload?warm=true` -- DONE
  v1.62.0** (P2, ask #4): one route sharing load's exact body (warm contract
  can't fork); v3's "Reload now" points at it. The fit meter's §5 observed
  line (resident memory, measured after load) landed in the same version.
- [ ] **Continuation loose ends** (P3, from v1.61.0-1.61.1; the feature is
  complete and review-hardened, these are parity/depth): MessageCreateRequest
  has no explicit `continue_final_message` field (the auto trailing-assistant
  convention works on /v1/messages and block-form prefill is flattened, but
  there is no explicit control for OpenAI-endpoint parity); image-history
  continuation is a deliberate 400 (the vision strategy has no open-turn
  spelling yet); the eval bank's thinking/stop tasks have not been run over
  the template change (explicit-ask tier per testing-cost discipline --
  normal-path rendering is covered by the suite + live E2E).
- [x] **update_deps hardenings -- DONE v1.57.1** (P2, from the KEEP-WITH-FIXES verdict):
  re-read pyproject before the final write and abort/re-apply on
  concurrent change (the C++ build makes the window minutes long);
  roll back the pyproject write when the follow-up `uv lock` fails;
  unit tests for the write path (latest-channel sources = git+rev only;
  tomlkit round-trip comment-preservation).
- [ ] Small backend nits (P3): `extra_args` schema reports `default: null`
  (default_factory not serialized) and array fields carry no `items` type;
  top-level `description`/`tags` not clearable via null (asymmetric with
  config nulls); admin `/import` still log-skips invalid entries while the
  CLI importer refuses loudly; `$HEYLOOK_LLAMA_SERVER` pointing at an older
  build than the default-location one gets no staleness warning; consider
  `-t`/`--threads` field (design doc's trigger -- experts-on-CPU landed)
  and widening `n_gpu_layers` to accept `auto`/`all` (b10362 idiom);
  `num_draft_tokens`/`prefill_step_size` effect-class re-read now that
  per_request means "refreshed live into loaded providers".
- [ ] Small editor nits (P3): armed reload label could name the recorded
  cost (retain last warm_ms per model); F8 field-level 422 mapping onto the
  offending input; F6b warn when an `extra_args` token collides with a
  managed field's declared `arg`; per-group collapse on phone widths.

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
- [x] `docs/observability_guide.md` -- rewritten 2026-07-20, then DELETED
  same day under the owner's docs philosophy (code-inferable; the streams are
  self-describing JSONL + /v1/admin/config is the knob).

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
- [x] S1.6 -- LAN hardening: Caddy reverse proxy guidance in server.py + optional `HEYLOOK_ADMIN_TOKEN` (extended to gate `/v1/admin/config` in v1.79.0)

## Slice 1 gated work

- [x] S1.2b -- preset + import redesign: SHIPPED (presets.py registry, default_preset cascade -- see "C1 of S1.2b" in presets.py docstring). Stale-gated entry closed 2026-07-06 during docs audit; the import-defaults follow-through landed in v1.31.3/v1.32.0.

## P1 - High Priority (do soon)

### TOML Migration Completion
- [x] Integrate `--interactive` flag into model_importer.py (DONE v1.18.1)
- [x] Wire up ConfigEditor in import workflow (DONE v1.18.1)
- [x] Move profiles to TOML files, rename profiles, dynamic discovery (DONE v1.19.0)
- [x] Fix `ModelProfile.apply()` precedence bug (DONE v1.19.0)
- [x] ~~Manual test of `import --folder ... --interactive`~~ MOOT: `--interactive`
  and the config TUI were retired in v1.47.0 (derive-at-load); import is
  non-interactive now.

### Dependency Cleanup
- [x] Remove `mlx` optional extra (duplicated core deps) (DONE v1.19.0)
- [x] Purge unused deps: torch, torchvision, opencv-python, scipy (DONE v1.19.0)
- [x] Move `datasets` to `analytics` extra, `rich` to `scripts` extra (DONE v1.19.0)

### Stale Code Removal
- [x] Delete stale integration test `test_performance_monitoring` (DONE v1.19.0)
- [x] Delete `/v1/performance` stub and `/v1/performance/profile/{time_range}` endpoint (DONE v1.19.0)

## P2 - Medium Priority (scheduled)

### Build v1.20.0: Models Config TUI + CI Foundation

#### Models Config Command (Phase 4) -- RETIRED / MOOT
- [x] ~~Create `src/heylook_llm/commands/models_config.py`~~ MOOT: config_tui.py
  was retired in v1.47.0 (derive-at-load thin models.toml entries); superseded
  by the v3 Web admin config editor (v1.54.0+).

#### llama-server Provider -- SUPERSEDED 2026-07-26 (and then BUILT same day)
Absorbed into `plan_2026-07.md` Phase 7 and executed v1.40.0-1.44.2 (7a-7e
detection all DONE). Dossiers:
`internal/research/gguf_provider_viability_2026-07.md` +
`gguf_driving_models_2026-07.md`. Driving models at `modelzoo/gguf/`.

#### gguf follow-ups (post-Phase-7 loose ends, 2026-07-26)
- [ ] GGUF-metadata reading in the importer: auto-detect audio modality +
      thinking capability from the file's own metadata (today: audio via
      manual `modalities`, thinking via `supports_thinking` flag).
- [ ] gemma-4 12B MTP A/B on Metal (E4B measured net loss, Qwen3.6 +21%;
      12B unmeasured -- entry has the drafter paired, spec_type set:
      measure before trusting).
- [ ] E2E: no browser-suite coverage for the audio attach flow (needs a
      gguf E2E_MODEL; today's coverage = contract tests + live eval bank).
- [ ] llama-server binary lifecycle: coderef build is pinned by checkout,
      not recorded anywhere machine-readable -- consider stamping the
      build SHA into logs at provider load.
- [ ] Importer scan currently iterates each gguf dir 3x (pickers) --
      fold into one listing pass next time that file is open (flagged by
      /simplify efficiency review; scan-only cost, deliberately deferred).

### Optimization Plan Doc Refresh
- [x] Update `docs/mlx_optimization_plan.md` -- phase 5 updated for v1.18.0 pre-filled cache pattern
- [ ] Mark plan as historical or rewrite deferred items as standalone proposals

## P3 - Nice to Have (opportunistic)

### CI/CD Pipeline
- [ ] Add GitHub Actions workflow (`.github/workflows/test.yml`, `.github/workflows/lint.yml`)
- [ ] Automated testing on commit/push
- [ ] Coverage reporting

### Benchmark Script
- [x] Create `scripts/benchmark.py` (DONE -- HTTP benchmark measuring TTFT, TPS, memory across OpenAI and Messages APIs)
- [x] Token throughput, TTFT, memory usage metrics (DONE in `scripts/benchmark.py`)

### ~~Build v1.21.0: llama-server Provider + GGUF + Benchmark~~ RETIRED 2026-07-26
Stale bucket (predates the v1.21.0 removal; the referenced commented
pyproject extra no longer exists). Superseded by plan Phase 7.

## Deferred (blocked on upstream)

### MLX Engine Optimization (remaining)
- [ ] Shape bucketing for prefill (needs attention mask correctness verification)
- [ ] `mx.compile` on decode step (deprioritized -- marginal gains vs. complexity; revisit if mlx-lm adds native compile support)
- [ ] Automatic draft model selection (vocabulary compatibility checking)
- [x] Full vision path unification (DONE v1.18.0 -- pre-filled cache pattern, no `inputs_embeds` needed)
- [ ] Vision + speculative decoding (pre-filled cache incompatible with speculative prefill)
- [ ] Radix cache for vision (pre-filled cache bypasses radix tree)

- [x] Error-chunk altitude: DONE in v1.33.0 -- provider raises typed GenerationFailed/InvalidGenerationRequest; batch/RLM now fail loudly; non-streaming client errors return 400. See docs/architecture/postmortems + CHANGELOG 1.33.0.