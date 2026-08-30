# CLAUDE.md

<!-- Nav hub. Repo-specific only -- global conventions (uv, orjson, no-emoji,
conventional commits, TDD, path-privacy, docs) live in the user-level CLAUDE.md
and still apply. Don't duplicate them here. Last verified: 2026-07-09 -->

Personal MLX inference server on Apple Silicon: FastAPI backend + a vanilla-JS
frontend (v3, served at `/v3` -- the ONLY frontend since v1.77.0).

## Orient first

- **Roadmap** -- the master plan, phased 0-7 (§"v3 frontend guardrails" + Phase 4 = v3 hardening; Phase 3b = Messages-API migration; Phase 7 = gguf/llama-server provider, done 2026-07-26 except the Phase-6-coupled registry substrate): [docs/project/plan_2026-07.md](./docs/project/plan_2026-07.md).
- **Status + backlog**: [docs/project/CURRENT.md](./docs/project/CURRENT.md) (graded done/left narrative), [docs/project/TODO.md](./docs/project/TODO.md). Read before starting.
- **Engine coverage plan** -- why provider != engine (`"mlx"` is TWO upstream repos) and how live coverage stops being incidental: [docs/project/plan_engine_coverage.md](./docs/project/plan_engine_coverage.md). Phase 0 shipped as `tests/smoke/`.
- **v3 frontend map** -- what's done/left + the backend<->v3 coupling: [docs/frontend_v3.md](./docs/frontend_v3.md) (git-tracked). Build contract: [docs/frontend_v3_spec.md](./docs/frontend_v3_spec.md) (§4 = API contract). USER-FACING behaviour (the three-layer settings model behind "preset vs ad-hoc change", the generation lifecycle, editing, and a standing rough-edges list): [docs/frontend_v3_user_guide.md](./docs/frontend_v3_user_guide.md) -- read it before changing chat UX, and update its rough-edges section when you close one.
- Deep dives: **backend reference** is git-tracked in [docs/architecture/](./docs/architecture/) (config, mlx_provider, ecosystem_strategy -- design records + invariants only, the live surface is code + /openapi.json; see its [README](./docs/architecture/README.md)); crash **postmortems** (read before touching providers) are in [docs/architecture/postmortems/](./docs/architecture/postmortems/). Local-only in [internal/](./internal/): `log/`, `research/`, `thoughts/`, and stale subsystem notes (batch/logprobs/thinking) pending refresh. The old React-frontend docs are in `internal/frontend/archive/`.
- **Research / design**: Jacobian-lens ("j-space") interpretability feature -- how-to + tutorial [docs/jspace_guide.md](./docs/jspace_guide.md); build + verifier plan [docs/jspace_integration_plan.md](./docs/jspace_integration_plan.md) (Phase 5-ish; V1/V2 parity GREEN; go-forward plan = its Part 2). Lens **fitting** lives in the `jlens-mlx` sibling repo; this server only **applies** (the old `coderef/jspace_scratch/` spike was relocated there).
- Setup/commands [README.md](./README.md) · tests [tests/README.md](./tests/README.md). (The v3 API contract is spec §4; the live schema is at `/openapi.json` + `/docs`.)
- `internal/`, `models.toml`, `coderef/` are gitignored -- local-only, never committed.

## Architecture

**Backend `src/heylook_llm/`** -- Three providers (`Literal["mlx", "mlx_embedding", "gguf"]`,
single source of truth `config.PROVIDER_CONFIG_CLASSES`; router provider_map must stay
key-synced): MLXProvider (text+vision), MLXEmbeddingProvider, LlamaServerProvider
(gguf -- ONE llama-server SUBPROCESS per loaded model; "loaded" == "running process",
so LRU/idle-unload = spawn/SIGTERM; pure stdlib, no MLX import). Provider OUTPUT is the
owned `GenerationChunk` (providers/base.py, slotted -- new telemetry = a FIELD there
absorbed in perf_collector.ChunkTelemetry, never an attr-patch; `thinking` carries
engine-PRE-SPLIT reasoning, e.g. llama-server's reasoning_content; errors RAISE
GenerationFailed/InvalidGenerationRequest, never chunks). gguf gotchas: llama-server
runs `--jinja` + reasoning pre-split by default (provider template_info()=None routes
heylook's parsers to pass-through -- never re-parse another engine's split output);
the template is resolved as a THREE-WAY LADDER at spawn (v1.79.43) and every spawn logs
which rung won: explicit `chat_template_path` (v1.68.0, `--chat-template-file`) beats a
`chat_template.jinja` DISCOVERED beside the .gguf, which beats the one EMBEDDED IN THE
GGUF. Sidecar-beats-embedded is the default because the embedded template is whatever
the quantizer baked in while a sidecar is the file you can read, diff and edit;
`use_sidecar_chat_template = false` keeps the embedded one WITHOUT deleting a file out of
a downloaded snapshot dir. Consequence worth holding: a template can now change with no
models.toml change at all -- dropping a file next to the weights is enough -- which is
why the spawn log exists, and why any measurement that varies by prompt format must
establish WHICH TEMPLATE each arm ran against before concluding anything about the model.
`chat_template_source` (MLX-only, template_info.py) still does NOT reach this provider and
is deliberately a different name for a different mechanism. All of it is requires_reload
because llama-server takes the template at SPAWN, which is also why no per-request/
per-preset form exists (per-request lever = `chat_template_kwargs`).
Publishers differ on the SAME weights and it is not cosmetic (measured live on
Qwen3.8-27B, both templates, 2026-08-17): ggml-org embeds Qwen's official template
byte-identically (8952 bytes), unsloth a patched one (9993) adding a `developer` role
and MERGING up to two leading system messages -- two leading system messages render
under unsloth and are a 500 under official, because a raised jinja exception is a 500
from llama-server. Both still reject a system message appearing MID-conversation, so
"unsloth's is permissive" is false in general -- try the shape, do not assume it.
always send max_tokens (server default is UNLIMITED); `-np 1` is our choice; spec decode
(`spec_type = "draft-mtp"`) is per-model opt-in and should stay OFF unless you have
checked it a win on YOUR model at YOUR context. NO PERFORMANCE NUMBERS IN TRACKED DOCS
-- 2026-08-10 produced a string of figures that were each confidently wrong in turn,
and the ones that survived were spot OBSERVATIONS, not performance testing: nothing
controlled for quant version, which llama.cpp produced the quant, build flags, thermal
state, memory pressure, or a second machine. What is left after a day of chasing it:
on the one case examined most carefully (a gemma-4 MTP model, vendor sampling,
realistic context, matched warm cache, LONG generation), spec on/off is a WASH --
indistinguishable from noise. Every larger effect seen that day dissolved when one more
variable was controlled: a big tuning win was a greedy artifact, a clear cost was a
short-generation artifact, a dramatic context effect was a cache-ordering mistake, and
a "broken drafter" was refuted by the drafter's own output. Default OFF for a NEW model because it is unproven
here, NOT because it is known harmful -- and expect your own case to need its own check.
CARVE-OUT, do not undo it: the owner has decided the DeepSeek V4 Flash entry KEEPS spec
decode on, on community evidence and their own judgement rather than anything measured
here. "Default OFF" means do not ENABLE it elsewhere without checking; it does not mean
disable what is already running. Nothing above transfers to that entry anyway -- it is
a different spec type on a bandwidth-bound MoE, which is the regime most likely to win
and the opposite of the cheap dense target these observations came from.
Local detail, conditions and the history: `internal/research/`.
CHECKING IT YOURSELF (a day of wrong answers produced these, and they are the durable
part): never at temp 0 -- greedy acceptance is exact argmax matching while temp>0 is
rejection sampling, a different regime, not a quieter one (temp 0 is fine for
reproducibility, never for a throughput claim; the probe warns). Match prompt length,
generation length, seed, sampling, WHICH BINARY, and PROMPT-CACHE STATE across arms --
an unmatched cache produced a large phantom that survived repeats and looked exactly
like a finding. Tune `spec_draft_n_max` and `spec_draft_p_min` TOGETHER; they interact
and a 1D sweep finds a different, wrong optimum. Short prompts and short generations mislead about
RANKING, not just magnitude. And matching every control you thought of does not
make a result sound -- it only rules out the confounds you imagined: the runs that
produced the since-dissolved cost matched prompt, cache, seed and sampling, and
were still measuring the wrong thing because nobody had varied generation length.
DRAFTER PACKAGING (read the GGUF, do not trust vendor docs -- unsloth.ai currently
states the OPPOSITE): gemma-4 12B ships a SIDECAR `mtp-gemma-4-12B-it.gguf` with zero
nextn tensors in the main file; Qwen3.6-27B has `blk.64.nextn.*` EMBEDDED and no
sidecar. A LoRA further ERODES whatever win exists: `common_set_adapter_lora` has ONE
call site in tools/server and applies to `ctx_tgt` only, so the drafter proposes the
base distribution while the target generates the adapted one -- structural, no adapter
escapes it, magnitude unknown and n=1. `--spec-type` is a SPAWN flag while `lora` is
per-REQUEST, so one process cannot suit both kinds of traffic -- but that tradeoff only
matters if spec decode is a win at all, which is not in evidence. gemma drafters are SIDECAR `mtp-*.gguf` (auto-paired by the
importer into draft_model_path -- llama's own `-hf` sibling discovery does NOT work for
local files), Qwen3.6's MTP is EMBEDDED in the main GGUF; the llama-server BINARY
is the canonical local build (fixed dir under home, written only by
scripts/build_llama.py -- ONE build, one source, owner rule 2026-08-13; update =
re-run the build script). `server_binary` / `$HEYLOOK_LLAMA_SERVER` remain as
escape hatches for experiments but WARN AT EVERY SPAWN naming the canonical
build they shadow -- the silent-rot mode (an exported var pointing at a stale
working-tree binary while a fresh canonical build sits unused) shadowed exactly
that on 2026-08-13 and is retired: the owner's shell export is gone, and any
re-introduction announces itself. Only if no binary exists anywhere does load
fail loudly. CLI
`heylookllm import` MERGES with an existing models.toml by default (existing entries +
top-level keys go right back out verbatim, comments re-injected via toml_comments;
scans only APPEND new ids; `--fresh` = old wholesale rewrite) -- so a hand-written
`server_binary` survives reimport. llama-server is BUILT by
`scripts/build_llama.py` (the only thing that clones/builds it; `uv sync` cannot -- it
is C++, not a uv package; newest `b<N>` release tag by default, and llama.cpp's
releases ARE those tags -- `--rev` for anything else; it never touches
pyproject/uv.lock). llama.cpp is NOT vendored and NOT
a submodule: the clone + build tree live OUTSIDE the repo (fixed dir under the user's
home; `dir`/`$HEYLOOK_LLAMA_CPP_DIR` relocate), so upstream source can never be
committed or packaged and there is nothing to `submodule init`. Audio input
(`input_audio` parts, gguf-only) must fail LOUDLY on MLX (audio towers are stripped at
load) -- the 400 guard lives in MLXProvider.create_chat_completion.
THINKING DEPTH: `reasoning_effort` (v1.71.0) is a CHAT-TEMPLATE VARIABLE, not a sampler
knob -- it rides `chat_template_kwargs` beside `enable_thinking` (gguf) / apply_chat_template
kwargs (MLX). Sent WHENEVER SET, never gated on enable_thinking: gpt-oss/harmony reads it
unconditionally and has no enable_thinking at all, so a gate made it unreachable for the
one family the docs name as taking low|medium|high. Its CAPABILITY is separate from
`thinking` for the same reason (Qwen3.5 reads one, gpt-oss the other): MLX probes the
template file precisely, gguf rides `supports_thinking` because the template is inside
GGUF metadata. The accepted SET IS PER MODEL (Qwen3.8: xhigh|medium|low and it RAISES
otherwise; harmony: low|medium|high), so the schema Literal is their union and a wrong-for-
this-model value reaches the template -- where llama-server turns a raised jinja exception
into a 500. Absent = send nothing, leaving the template default (xhigh on Qwen3.8, which is
why the field exists). In MLX it must NOT ride `base_kwargs`: the TypeError retry re-passes
those verbatim and strips only the explicitly-named kwargs, so a narrow TokenizerWrapper
has to be able to lose it the same way it loses enable_thinking.
MODEL REGISTRY (v1.69.0, `model_registry.py`): models.toml is OVERRIDE-ONLY --
anything under `[scan].folders` is served with derived defaults, so a new download
needs no import, no symlink, no edit. The merge is LOAD-time (`ModelRouter._load_config`,
so startup AND reload get it) and NEVER writes models.toml; a `[[models]]` entry is
served exactly as written and always wins, discovery can only ADD. Matching is the
RESOLVED `model_path` (`.resolve()` follows symlinks), never the id -- an id is DERIVED
from the directory name, so a hand-renamed entry stops matching itself, and
`modelzoo/<vendor>` symlinks make one file reachable by two spellings sharing no prefix;
that pair silently duplicated a Muse-Glimmer entry (with a wrong `supports_thinking`)
on 2026-08-17, which is also why the importer now dedups on resolved path. Discovery is
BEST-EFFORT: a failing scan is logged and dropped, never fatal. Admin edits MATERIALIZE
an entry on write (`update_config`/`toggle_enabled`/`bulk_set_default_sampler`) because
editing IS the override; reads never do, or browsing the models page would grow the
file. `remove_config` deliberately does not materialize -- the next scan would serve it
back, so a "removed" model that reappears is worse than a clear no.
Router keeps `max_loaded_models=1`
by default (LRU evict + pin + idle-unload via `idle_unload_seconds`/`unload_after_idle_seconds`);
config in `models.toml`. Every provider-config FIELD declares when a change
takes effect (`json_schema_extra={"effect": ...}`, six classes; reload set +
import allowlist + `/v1/admin/model-options` all DERIVE from it -- never
hand-maintain a second copy; new field = classify it or import refuses).
Invariants (v1.55-56, design record docs/architecture/config.md):
`reload_config()` pushes per_request defaults into LOADED providers (they
are construction-time snapshots otherwise -- "applies immediately" was a lie
before this); admin responses serialize config `exclude_unset` (absent IS
the default's spelling -- a validator that ASSIGNS derived fields must
restore `__pydantic_fields_set__` or they leak back as "stored");
`stale_reload_fields` on admin responses is the server-derived
"saved-but-process-runs-old-value" truth (never rebuild it client-side).
API routers (counts rot; the list is the point): messages, rlm, conversation, notebook,
preset, admin, admin_ops, scan_import, jspace (Jacobian-lens interpretability), config (operational
settings), telemetry (frontend ingestion). DuckDB store (`db.py`: conversations +
notebooks + presets + `settings`, single serialized writer thread, transactional ops;
`HEYLOOK_DB_PATH` override; dynamic field names gated by `_UPDATABLE_*_FIELDS`
frozensets; a `_SCHEMA_VERSION` bump DROPS all tables -- `settings`/`presets` are
additive + drop-safe, key->value/config, not in the drop list). DB/config POLICY (solo
deploy, no data to preserve): NEVER write migration code -- dropping, recreating,
or truncating any DuckDB store or config on a schema change is fine and preferred.
Use additive `CREATE TABLE IF NOT EXISTS` only when you just need to ADD a table
(cheap, non-destructive); for an actual schema CHANGE, bump `_SCHEMA_VERSION` /
drop / recreate. RLM (`rlm.py`): recursive inference with sandboxed REPL.
- [docs/architecture/](./docs/architecture/) (config, mlx_provider, ecosystem_strategy + postmortems -- design records and invariants only; live surface = code + /openapi.json) · [docs/rlm_guide.md](./docs/rlm_guide.md) · converting checkpoints to MLX: [docs/mlx_conversion_guide.md](./docs/mlx_conversion_guide.md)

**Frontend v3 `apps/heylook-frontend-v3/`** -- the current frontend: vanilla
JS, no build, served at `/v3`. 6 pages (chat, notebook, models, perf, explore, jspace);
chat generates over `POST /v1/conversations/{id}/generate` (v1.65-66: the
server builds the request FROM THE STORE and owns persistence incl. abort +
disconnect; Messages SSE grammar + a final `heylook_saved` event with the
authoritative rows; the client's post-stream state is ADOPTION, never
position arithmetic -- notebook/explore speak `/v1/messages` since v1.74.0,
so NO v3 page uses `/v1/chat/completions`), takes image (and gguf audio) input + renders
image content blocks out of the DuckDB store. The page is a MIRROR of the store
with exactly two invalidation points: document select, and RESUME (`ctx.onResume` ->
`refreshAfterResume`, v1.79.2) -- nothing polls, and re-clicking the active
conversation deliberately does not refetch. Resume exists because iOS Safari brings a
backgrounded tab back with the heap it had, and every write the page makes is
whole-value from that mirror (prompt keystroke PUT, params PUT, preset Save snapshot),
so a stale mirror re-plays old state over newer edits. The lifecycle edges are
`createPage`'s `ctx.onHide`/`ctx.onResume` (each binds BOTH event spellings -- never
hand-wire `visibilitychange`/`pagehide`/`pageshow` in a page). Every debounced
writer OWNS its hide flush the way it owns its teardown flush (prompt-section,
`bindDocumentParams` via its `onHide` arg, notebook's `scheduleSave`) -- a consumer
that has to remember it is how one shipped without. Hide flushes send with
`keepalive` and are dispatched AHEAD of the PUT chain (a request queued behind an
in-flight PUT is never sent if the page unloads). A prompt section's hide hook lives
as long as the section -- detached is NOT dead (notebook keeps one through every
drawer close); chat, which builds one per drawer render, `release()`s the one it
replaces. Resume fetches the conversation body only when the list's `updated_at`
moved and commits the new stamp only AFTER everything it covers is adopted (else one
failed fetch is a permanent "unchanged" -- nothing else ever refetches the active
conversation); adopts via the same `adoptConversationMeta` select uses; never touches
a live stream's rows, the prompt while its box is being typed in (only that field --
every other drawer field commits on change), or the sidebar during a rename. THREE attach inputs -- picker,
paste, drop (v1.72.0) -- funnel through ONE `addFiles` -> `addPendingFiles`
routine, which is where the cap gate, the count cap and the aria-live
announcement live; adding a fourth input means calling addFiles, never
re-deriving any of that beside it (paste was image-only for exactly as long as
it had its own copy). Only the picker has an `accept` list, so that routine is
also the backstop for everything that has no accept list to respect. PASTE
LISTENS ON `document`, not on the page root, and that is not a preference:
clicking a message leaves focus on `document.body`, which is an ANCESTOR of the
chat root, so a root-scoped listener never sees the event -- it only ever
fired when a field or an in-thread selection held focus, i.e. the case that
already worked. Any "paste anywhere" feature has this shape; verify the target
rather than the listener (a synthetic event dispatched at a convenient node
proves nothing, which is how v1.72.0 shipped this broken WITH a passing check).
Document scope is also why the other-editable guard is load-bearing -- the
drawer's system-prompt box is a body child OUTSIDE `#app`. And `preventDefault`
waits until something will really stage: a clipboard payload carries text AND
an image, so cancelling on a refusal eats the text too.
Capability-gated chrome (attach button, picker accept list, thinking toggle,
drop-overlay label) is refreshed AFTER `modelSelect.value` moves, never before
-- it all reads `currentCaps()` off that select, and `selectConversation` had
the order backwards, so every one of them described the conversation being left. Drag/drop
is desktop-only ON PURPOSE and is not a §7 violation: it duplicates paths that
exist on the phone rather than being the sole route to anything.
A STREAMING message is rendered INCREMENTALLY (`markdown-stream.js`, v1.79.9)
and this is load-bearing, not a micro-optimization: the painter used to
re-parse the whole accumulated response through marked+DOMPurify into
`innerHTML` every animation frame, and marked's parse is SUPERLINEAR in
length, so per-frame cost grew with the response and a long generation
saturated the main thread -- on a phone that is heat and battery, and NO
check that renders a finished document can see it. MarkdownStream cuts at a
boundary no markdown construct can span (not inside a fence; at column 0
after a blank line; never a list marker or `>`, which would merge with a
block above), renders each segment ONCE into a committed prefix whose nodes
are then never touched again, and re-renders only the tail; a
link-reference/footnote definition disables splitting for that message
because it reaches forward arbitrarily far. The boundary rule is a PROPERTY,
not a set of examples -- `tests/e2e/render.mjs` grows generated documents one
chunk at a time and diffs against a whole-document render, the same technique
and the same reason as the backend's `TestParserInvariants`. Painters whose
cost scales with the document use
`ctx.throttleTime` (~15/s), never `ctx.throttle` (per-frame, correct only for
cheap work like explore's token strip). Scroll-follow is measured at the TOP of
the painter, BEFORE it mutates -- both halves of that are load-bearing. Before
the write the reads are cache hits (layout is still clean from the last paint)
rather than a forced re-layout, which matters most on iOS, where
nothing skips off-screen rows (content-visibility was REMOVED in v1.79.18 --
it moved `scrollTop` behind the app's back on every engine, not just WebKit,
stranding the tail-follow and opening conversations thousands of px above
their end; the layout it saved was one-time and desktop-only) so a forced
layout walks every row. And only before the write is it HONEST: measured after,
one paint appending more than the slack (a code block, a table) reads as "the
reader scrolled away" and strands the view for the rest of the generation. A
CACHED flag fed by scroll events was tried and is WRONG -- pinning coalesces
scroll events to a handful across a whole generation, so the flag goes stale
exactly when the viewport changes under it, which on a phone is every keyboard
open (`tests/e2e/render.mjs` resizes mid-stream and was shown red against it).
Chat also has a per-document system-prompt editor + a saved-preset bar (TWO shared
drawer sections, `prompt-section.js` + `preset-bar.js`, used by chat AND
notebook -- fix a bug in the shared factory, never in one page's copy).
The system prompt is an OVERRIDE BOX (owner rule, v1.62.3): a preset OWNS a
prompt and carries it, but a preset with an EMPTY one makes no claim and
leaves the document's prompt alone -- empty NEVER means "set it to empty",
which is what turned one blank Save into two presets losing their prompts.
Only a carrying preset can arm "Replace prompt?" or count as drift.
BOTH directions are armed (v1.79.20): Apply overwrites the DOCUMENT (recoverable
-- re-apply the preset), Save overwrites the STORED PRESET with an UPDATE that
keeps no history, so Save is the one only-loss actually names. It was the bare
one, and the select PRE-FILLS the save-as name box, so picking a preset to look
at it armed that preset as Save's target -- one click wrote the document's
prompt over a 35k-char stored one on 2026-08-28. Save's guard is an ORDERED set
of questions in `wouldOverwritePresetPrompt` -- read it THERE, it carries the
order, the reasons and its own known boundary (this paragraph restating the
branch list is exactly the hand-copied second copy this file warns about). The
shape: only a save onto a preset the document is NOT running arms, so the
apply/edit/save-back ITERATE LOOP stays one click -- v1.79.20 armed all of them
and thereby charged the loop for the accident, the same click-through failure
the rule exists to prevent, reintroduced by the fix for it -- while BLANKING
always arms, because a NULL write leaves an override-box preset present but
inert ("my preset disappeared"). Enter in the name box goes straight to Save
as new, which is CORRECT now and was not before: that rule existed because a
second entry point past an ARM is the same hole with a keyboard on it, and
Save as new has no arm to get past -- it cannot overwrite anything. Update,
which can, is reachable only by its own button. AN ARM IS A PROMISE ABOUT ONE ACTION, and
that is enforced in the PRIMITIVE: `armedConfirm` takes a `target()` describing
destination+payload, captures it at arm time and re-reads it on the confirming
click, re-arming instead of firing if it moved. It cannot live in consumer
wiring -- Save's payload is the DOCUMENT prompt, edited in a different drawer
section the bar gets no events from, so "arm, clear the prompt box, confirm"
blanked a preset straight past the blanking guard and no `disarm()` call in the
bar could have seen it. `disarm()` stays for VISIBLE honesty (a button still
reading "Overwrite prompt?" while aimed elsewhere is a lie even once clicking
it is safe), and each control disarms only what IT re-aims: the select moves
all three buttons, the name box moves Save alone. The reason that click happened at
all is structural: the drawer renders the preset section directly above the
per-document prompt box, which shows the DOCUMENT's prompt whatever the select
says, so every preset looked like it held the same text. The section now carries
a read-only preview of the SELECTED preset's own prompt, and the document's box
names its owner (`label` on createPromptSection's adapter). The select also
FOLLOWS the document's `applied_preset_id` until an explicit pick, and a pick is
remembered against the document it was made on. A prompt
typed before any conversation exists is parked in localStorage until a
conversation adopts it (it was page-state-only, so a reload ate it while the
sampler params beside it survived), and `.chat__sysprompt-chip` states what
is in force -- including the "No system prompt" case, which is rendered, not
hidden.
Since v1.54-1.57: the models page EDITS per-model config (`js/model-config.js`,
schema-driven off `/v1/admin/model-options` -- a new backend config field
appears in the UI with no frontend change; `ui:"hidden"` on a field is what
keeps it out), and chat switches models mid-conversation honestly: history
media the current model can't take is dropped AT THE WIRE with a per-message
disclosure while staged attachments still BLOCK (deliberate asymmetry,
commented at both sites -- do not unify). Since v1.72.0 the cap is checked at
STAGING time too, so that send-side block's ONLY remaining case is media staged
on a capable model and then switched away -- a drop or paste onto a model
without the cap now refuses immediately and stages nothing, because a staged
blob the user must later hunt down and clear is worse than a straight no.
Chat also consumes
`/v1/admin/models` (residency) + `load?warm=true` (its Load button). Only
LOSS gates a switch: load cost is DISCLOSED, never confirmed (owner call,
v1.62.3) -- residency dots, the Load button, and a live pre-first-token
status say what is happening. Choosing a model IS choosing to pay for it, so
a confirm there only trains click-through; the one removed fired hardest with
NOTHING resident, where its "may evict the resident model" was false.
Build contract: [docs/frontend_v3_spec.md](./docs/frontend_v3_spec.md) (§4 =
the authoritative backend API contract -- update it in the same commit as any
contract change); orientation + backend coupling: [docs/frontend_v3.md](./docs/frontend_v3.md).
Read `js/page.js` (createPage lifecycle) before touching any page. Design system +
the load-bearing a11y/mobile-parity rules new UI MUST honor -- touch-reveal
fallbacks (`@media (hover:none)`; hover-only affordances are unreachable on
iPhone), the settings drawer as a **modal** (seals `#app` with `inert`, closes on
`hashchange`), aria-live states, label association -- are [DESIGN.md](./apps/heylook-frontend-v3/DESIGN.md) §7.

(Retired frontends: `apps/heylook-frontend-v2/` + its `/v2` mount deleted at
cutover 2026-08-18, v1.77.0; the older legacy React app 2026-07-09. Both live
in git history; a contract test pins that `/v2` stays 404.)

**Optloop-lib `apps/optloop-lib/`** -- library-level bench for mlx-lm/mlx-vlm fork experiments (app-level optloop retired 2026-07-06: it bypassed the server code it claimed to measure). [docs/optloop_guide.md](./docs/optloop_guide.md) · its [CLAUDE.md](./apps/optloop-lib/CLAUDE.md). NB: root pyproject pins UPSTREAM mlx-lm/mlx-vlm, not its forks -- fork-side bench wins don't reach the server until upstreamed/repointed.

## MLX / library gotchas (the things you'll get wrong without knowing)

- All text+vision generation routes through `generation_core.run_generation()` -> `mlx_lm.generate.stream_generate`. Vision uses a pre-filled-cache pattern: the VLM forward pass fills the KV cache, then `run_generation()` continues.
- A VLM's forward returns a `LanguageModelOutput`, not raw logits, and caches `_position_ids`/`_rope_deltas` on its language model. Wrap it with `wrap_language_model()` (model_wrappers.py) before driving it with mlx-lm; position state is reset in `run_generation` via `_reset_vlm_positions()`.
- VLM prompt formatting: `mlx_vlm.prompt_utils.apply_chat_template`; inputs: `mlx_vlm.utils.prepare_inputs`. `prepare_vlm_inputs_parallel()` returns a 4-tuple `(images, formatted_prompt, has_images, image_urls)`.
- Vision feature cache (`providers/common/vision_feature_cache.py`): models with `encode_image()` accept `cached_image_features` to skip the vision tower; LRU keyed by image URL (pixel-hash fallback for base64).
- Load-library selection is `MLXProvider.effective_loader` (`providers/common/loader_routing.py`), derived from the config's `modalities` + `loader` fields -- NOT the raw `vision` bool, which is now a derived mirror of `"vision" in modalities`. `is_vlm = (effective_loader == "mlx-vlm")`. `loader="auto"` routes vision->mlx-vlm iff mlx-vlm registers the `model_type`, else mlx-lm (degrades only on POSITIVE non-support; an explicit `loader` forces the engine). Modality DESCRIPTION (`model_importer.detect_modalities`: config `*_config` blocks + `image_token_id`/`image_token_index`/`audio_token_id`...) is deliberately separate from this library-aware routing. The REPORTED `vision` capability derives from it too (v1.79.43, capabilities.py): the provider's image guard reads `is_vlm`, so reading the checkpoint's DECLARATION instead let `/v1/models` advertise images a 400 then refused -- one resolver for both surfaces is what makes them agree by construction. `modalities` still carries the declaration; description and served capability are different fields on purpose. It is ON THE WIRE as `effective_loader` on the `/v1/admin/models` row (v1.79.31), derived via `effective_loader_for_config` so it answers for UNLOADED models -- the provider ATTRIBUTE is null unless the model is resident, which is the opposite of what a live harness picking engine arms needs. Null for every non-mlx provider (gguf is one engine, named by `provider`). Because it reads each model dir's `config.json`, the two admin READ routes that build a model response are plain `def` (threadpool), not `async def`.
- Embedding backbone: `mlx_lm.utils._get_classes(config_dict)` (private API, takes a dict) -> extract `.model`. Gemma needs `sqrt(hidden_size)` embedding scaling (gated on `model_type.startswith("gemma")`).
- Prompt cache is a per-model SINGLE SLOT of immutable (state, meta_state) snapshots (v1.75.0, Q7 -- the radix tree is deleted): extension continues, divergence goes through mlx-lm's `trim_prompt_cache`, and non-trimmable layers (hybrid ArraysCache, rotated windows) re-prefill rather than slice -- hybrids are now CORRECT, not "limited". NEVER store or hand out live cache OBJECTS: arrays are immutable, objects are not, and a quarantined zombie generator keeps mutating its own (that was live-verified process-poisoning). See [docs/architecture/mlx_provider.md](./docs/architecture/mlx_provider.md) §4.2.
- `mlx_lm.generate.GenerationResponse` is a non-slotted dataclass -- attach per-request metadata via `response.X = value` (`# type: ignore[attr-defined]`), read via `getattr`.
- `mx.set_wired_limit(...)` is set at startup, but the per-generation `wired_limit()` CM is still needed for stream sync. Call `mx.reset_peak_memory()` at `run_generation` start to scope `mx.get_peak_memory()` per request.
- Verify a library is actually broken before working around it.
- Perf numbers are honest as of v1.34.1: recorded tok/s = native mlx-lm `generation_tps`, TTFT/tok-s exclude queue-wait (own `queue_wait_ms` field), trends are success-only. Per-chunk scraping goes through `perf_collector.ChunkTelemetry.absorb()` -- add new chunk fields THERE, not at call sites (batch_processor.py still has 3 unconverted scrape sites, moot when Phase 2 collapses it).
- The FIFO generation gate is a PROCESS-GLOBAL singleton shared by all providers (`_get_generation_gate`); `generation_queue_stats()` reports gate-wide traffic, not per-model. Any "is this model busy" logic built on it is conservative across models. `unload()` waits for actives AND gate waiters (30s cap) -- the active counter decrements before `gate.release()`, so never gate teardown on actives alone.
- Any NEW code path running MLX forwards off the event loop (analysis endpoints etc.) must run on `streaming_utils._executor_pool` (a pinned, REUSED thread) inside `with mx.stream(generation_core._get_generation_stream())`, AND acquire the process-global gen gate + `router.pin_model()`. Starlette's `run_in_threadpool` has no thread-local MLX stream ("There is no Stream(cpu/gpu, 0)") and a dying MLX thread aborts the PROCESS. Verify on a real worker thread, not the main thread (see `jspace_api.py`).
- `mx.load` is lazy/mmap-backed -- `mx.eval()` the arrays at load time if they'll first be used on a different (worker) thread, else the first eval crashes on that thread's missing CPU stream.
- A model's `.layers` can be a fresh-slice `@property` (pipeline-parallel Qwen3.5/deepseek/glm4_moe: `pipeline_layers = self.layers[start:end]`), NOT the list the forward iterates -- to hook/mutate blocks use the underlying list on the inner decoder (`inner.layers`/`.h`), not `model.layers`.
- Live-verifying streaming/latency changes: the 31B dense gemma natively decodes ~10 tok/s (looks identical to the old delivery cap); use the MoE `gemma-4-26B-A4B` (~90 tok/s) as the discriminating model.
- Stop-token/eos resolution has the same dual-source trap as chat templates: a model's full eos set can be split across tokenizer_config.json's `added_tokens_decoder` and tokenizer.json's `added_tokens` (gemma-4's `<turn|>` terminator lives only in the latter) -- read both. Raw HF tokenizers also don't absorb `generation_config.json`'s eos list, and mlx-lm's `stream_generate` auto-wraps a raw tokenizer with only the single `eos_token_id` -- `run_generation` wraps it itself (`ensure_gen_tokenizer`) with the full resolved stop set, or a model generates past its own end-of-turn.
- MESSAGES-WIRE CONFORMANCE (v1.79.39-40): `/v1/messages` is Anthropic
  Messages-SHAPED and was not Messages-CONFORMANT for three payloads, each of
  which failed silently for a client written against Anthropic's spec. Media
  blocks now accept BOTH the nested `source` object and the original flat
  `source_type` form (`content_blocks._flatten_source`, gated on `source_type`
  being absent so a flat block carrying an unrelated `source` key is left
  alone); `source` is a DECLARED `MediaSource` field, not validator-only,
  because a `mode="before"` validator contributes NOTHING to the generated
  JSON Schema -- `/openapi.json` advertised only the flat form while the docs
  recommended the nested one. Thinking blocks and `thinking_delta` carry the
  text under BOTH `thinking` (Anthropic) and `text` (v3's `streaming.js`
  reads `text` in two places); dropping either breaks one of the two readers.
  `stop_reason` is Anthropic's vocabulary via ONE table,
  `converters.STOP_REASON_FROM_FINISH_REASON` -- providers speak OpenAI's
  `finish_reason` because the internal ChatRequest does, and the rename
  happens at that boundary. THERE ARE TWO ROUTES ON THIS GRAMMAR:
  `/v1/messages` and `/v1/conversations/{id}/generate` share
  `StreamingEventTranslator`, so block payloads agree by construction -- but
  each assigned `stop_reason` itself, and fixing one left the other emitting
  `"length"` for a commit. 1700 per-path tests stayed green throughout:
  PER-PATH BEHAVIORAL TESTS ARE STRUCTURALLY BLIND TO CROSS-PATH DIVERGENCE,
  which is why `TestStopReasonHasOneMapper` asserts the shared mapper is the
  only writer rather than asserting either path's output. An ABORTED generate
  run reports `max_tokens`, not `end_turn` -- Anthropic has no cancellation
  value and `end_turn` positively asserts the model finished. `error` is NOT
  a stop reason: it was added on an untraced claim that api.py set it, and a
  non-streaming failure raises HTTPException so no MessageResponse exists at
  all. Deliberate remaining differences are enumerated in
  `docs/api_integration.md`; that list is hand-maintained and has been wrong.
  ASYMMETRY: the `/v1/conversations` store accepts ONLY the nested `source`,
  so nested is the spelling that works on every surface.
- Reasoning parsers (`reasoning_parser.py`): four ROUTING parsers (harmony/gemma channels, `<think>` markers, pass-through) that never strip anything themselves -- declared-specials stripping is ONE wrapper, `StripSpecials`, composed over the selected parser by `select_reasoning_parser` (and only when the model declares specials, so a bare parser is the no-strip case). Its rolling holdback is sized by the STRIP SET, not by any parser's own control tokens, and is prefix-set based because declared specials are not all `<`-shaped (Mistral's `[INST]` family). Behavior is pinned by PROPERTIES, not just examples (`TestParserInvariants`): output is invariant to how the stream was chunked, and text carrying no structural tokens survives intact. Both 2026-07-23 parser bugs were violations of those two properties. Design record: `docs/parser_strip_unification.md`.
- Chat-template resolution: `providers/common/template_info.py` is the single source of truth (per-model `chat_template_source` in models.toml; auto = `chat_template.jinja` > embedded `tokenizer_config.json` > `chat_template.json`; explicit source force-installs on the tokenizer at load, auto only fills a missing one). Traps: mlx-lm `chat_template_type` python templates live on the TokenizerWrapper -- the inner tokenizer's `chat_template` attr stays None, so never gate "has a template" on that attr alone; HF's legacy list-form `chat_template` isn't parsed (string-only); import-time jinja detection is the shared `detect_chat_template_source()` (used by BOTH the CLI wizard and the `/v1/admin` import route -- don't re-inline it). An MTP/spec-decode head registered as a model legitimately has NO chat template -- the load warning is expected there.
- mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects `enable_thinking=True` when the kwarg is ABSENT -- always pass an explicit bool. The kwarg is the cross-model thinking control: transformers forwards extra apply_chat_template kwargs as template variables (Qwen3 renders `<think>`, gemma-4 renders thought channels; others ignore it), and "template references enable_thinking" is the thinking-capability signal.
- Upstream posture (details: `docs/architecture/ecosystem_strategy.md`): mlx-lm is release-starved -- SHA-pin rather than wait for PyPI, check its open-PR backlog BEFORE writing any workaround, expect new capabilities via sidecar packages.

## Repo conventions (beyond the global ones)

- New endpoint or changed response model: module with `APIRouter(tags=["Name"])`, add the tag to `openapi_tags` + `app.include_router()` in `api.py`. (The OpenAPI drift guard -- `generated-api.ts` / `scripts/check_openapi_sync.sh` / the pre-commit block / `/openapi-regen` -- was retired 2026-07-09 with the legacy React app that consumed the generated TS types; v3 hand-writes `api.js`. The live schema stays at `/openapi.json` and `/docs`.)
- Removing a provider/feature: grep the repo, then check `config.py` (Literal+Union), `router.py`, `api.py`, README/ARCHITECTURE, `pyproject.toml` extras, frontend type unions, test fixtures.
- TWO named-bundle systems, one word each -- don't conflate or re-alias: **samplers** = bundled TOMLs `src/heylook_llm/data/samplers/` via `SamplerRegistry` (`samplers.py`), resolved in the provider cascade (`thinking` auto-applies for `enable_thinking` models; `vlm-*` encode mlx-vlm's ignored-param subset), reachable as `ChatRequest.sampler` / models.toml `default_sampler` / `/v1/admin/models/samplers`, discoverable via `/v1/capabilities.samplers`; roster is exact-pinned by `test_sampler_registry.py` (a new sampler must name its consumer). **Presets** = `/v1/presets` DuckDB prompt+sampler bundles (v3's preset bar -- shared `preset-bar.js`, chat + notebook, client-expanded) -- "preset" means ONLY this system now. Retired names for the registry: "profile" (collides with `/v1/performance/profile`) and "preset" (collided with the above; a ChatRequest sending `preset` gets a 400 with a migration hint, not a silent drop). CLI `--preset`/`--profile` survive only as aliases for `--sampler` on import. Details: `docs/architecture/config.md`. Which preset a document is RUNNING is `applied_preset_id` on conversations/notebooks (schema v6) -- written on explicit Apply/Update/Save-as-new only; a document whose state merely matches a preset is labelled by live client-side matching and never stamped (storing a derived association can bind stale state to the wrong document).
- A document's `params` is the SAMPLER BAG and everything in it reaches the model -- non-sampler state may never be stashed there (v3 keeps display prefs in a separate store for exactly this reason, and preset provenance got its own column rather than a params key). The same rule is why `samplerParams(caps)` filters capability-gated keys at the wire.
- A HAND-COPIED CONSTANT LIST IS A DEFECT WITH A DELAY, not a style issue. This repo
  already derives rather than copies -- the reload set, the import allowlist and
  `/v1/admin/model-options` all come off `effect` metadata precisely because a
  hand-written second copy drifted. Three MORE copies drifted in one session
  (2026-08-17): `conversation_generate_api._SAMPLER_KEYS` was a copy of
  `samplers.REQUEST_SAMPLER_FIELDS` and silently dropped `reasoning_effort` on the ONLY
  surface that generates server-side; the resolved-path identity rule existed in three
  places; the reasoning-effort `Literal` union in three more. When you add a field to a
  cascade, `grep` for a tuple/set/frozenset that ENUMERATES its siblings -- each one is a
  place the field has to be added by hand, and the failure is silent in every case.
  Derive it (`X = SHARED_TUPLE + ("extra",)`) or expect the drift.
- - The security hook false-positives on `mx.eval` (MLX graph materializer, not Python's eval) -- prefer `mx.async_eval` or acknowledge it.
- Observability spine (`observability.py`): ONE ingestion path `record_event(type, *, tier, min_level, source, fields=<dict>)` -> level-gated JSONL under `logs/` (`metrics.jsonl` content-free/aggregatable; `events.jsonl` correlated, may carry BOUNDED error text -- type+message+cause-chain, still NEVER prompts/responses/token IDs). `fields` is an explicit dict (NOT `**kwargs`) so caller/client keys can't collide with the reserved kwargs. Best-effort: never raises (inference must not break). `diag_event` (diagnostic_logger.py) delegates here; `memory.py`'s legacy streams also write under `logs/` and are gated by the master off switch (see below).
- Observability CONTROL is a single knob: `observability_level` (off|minimal|standard|debug), an operational SETTING resolved **DB > default**, and the default is `off` -- FILE LOGGING IS OPT-IN (owner rule 2026-08-13: no files under `logs/` unless the level is raised; `logs/` resolves CWD-relative, so an on-by-default level sprinkled log dirs wherever the server was started). No env override -- env silently beating the admin UI is a footgun; env is bootstrap-only: `HEYLOOK_LOGS_DIR`, `HEYLOOK_DB_PATH`. `off` is the master kill switch (silences the spine, memory.py's streams, AND the llama-server subprocess `.log` -- the gguf provider checks the level at spawn, so capturing llama-server output needs level>off at LOAD time, then a reload). Settings live in the App-DB `settings` table (`db.get_setting`/`set_setting`), contract in `settings.py` (`SettingsSchema` + `resolve_settings`), CRUD via `/v1/admin/config`; the level+retention are cached in-process (`observability.configure`), refreshed at startup + on PUT. Rotation is file-based (size + age, hourly on the tick). Content invariant is level-INDEPENDENT: `minimal` is NOT "content-free" (its events carry error text) -- only the metrics tier is guaranteed content-free.
- Route MemoryManager calls through `memory.safe_mm_call(...)` (no-op when None, swallows errors); use `sampler_summary_from_request` (memory.py) for "what was this configured with". Redesign status + the memory.py-stream consolidation follow-ups: `docs/project/TODO.md` + `internal/research/observability_and_config_redesign.md`.
- Pydantic `Field` defaults MUST use keyword form (`Field(default=None, ...)`, never `Field(None, ...)`) -- this pyright build only recognizes the keyword form; positional defaults make every model constructor flag false "arguments missing" errors (repo-wide sweep done 2026-07-20).
- Pydantic model + custom headers: `Response(content=model.model_dump_json(), media_type="application/json", headers=...)` (`JSONResponse` double-serializes). SSE post-generation telemetry (peak mem, cache bytes) goes in the usage chunk's `timing` (client needs `stream_options.include_usage=true`).
- `models.toml` comments SURVIVE admin writes (v1.58.0, `toml_comments.py`) -- but only while their ANCHOR is unchanged, so a note can never outlive what it describes: a comment on a top-level key needs that key's value unchanged; every comment inside a `[[models]]` entry needs that whole model byte-identical (normalized through `tomli_w`, so old hand-formatting doesn't pin anything); a block sitting above a `[[models]]` header additionally needs that FOLLOWING model unchanged and still next. Consequence: a comment on the value you are PATCHING is deliberately dropped -- provenance for a value you're changing still belongs in CLAUDE.md or `internal/`, not next to the value. Mechanism invariants: `tomli_w` stays authoritative for values/layout/order; tomlkit is used STRICTLY READ-ONLY for comment extraction, and comments are injected as lines into the fresh render, gated on the merged text parsing to exactly the fresh render's values -- any doubt degrades to a comment-less write, never a refusal. NEVER graft comments into a tomlkit-parsed document instead: mutating ANY item of a parsed array-of-tables (even comment trivia) makes `tomlkit.dumps` re-render the AoT as an inline array -- malformed for nested tables; that is the failure mode that sank the first attempt against `test_import_reimport.py`.
- Never commit runtime data: `*.db`, `*.jsonl`, `/data/*`, `apps/*/data/*` are gitignored; package data at `src/heylook_llm/data/` is intentionally NOT ignored.
- `.claude/` is local-by-default with a PER-FILE tracked allowlist (2026-07-26): tracking a NEW rule/agent/skill/hookify file requires edits in TWO places -- the `.gitignore` negation block AND the pre-commit hook's `ALLOWED_PATHS` (a bare dir negation would silently publish future files; hookify filenames must keep `.local.md` -- its loader glob is hardcoded). Files INSIDE the four tracked skill dirs track by default (publish-intent boundary, deliberate). `modelzoo/` and `adapters/` (j-space lenses; `HEYLOOK_JSPACE_DIR` default `adapters/jspace/<model_id>/`) are git-tracked dirs with gitignored contents (only their `.gitkeep`).
- PARALLEL SESSIONS are normal in this repo -- assume another Claude may have uncommitted work. Stage files EXPLICITLY (never `git add -A`/`-u`); before committing, `git status` and leave any file you didn't touch unstaged. After any scripted string-replace (version bumps especially), verify the edit actually landed -- a concurrent edit to the same string makes the replace silently no-op.
- Release bookkeeping: bump `src/heylook_llm/__init__.py` `__version__` in the same commit as the CHANGELOG entry -- it feeds `/v1/capabilities.server_version` and package metadata (was hardcoded-stale for weeks before 2026-07-20).
- rich (batch-labeler, scripts): square brackets in dynamic text are MARKUP and vanish silently -- wrap model output/prompts in `Text(...)` or `rich.markup.escape()` before printing.
- Commits fine without asking; never push unless told. Update `internal/log/log_YYYY-MM-DD.md` before ending a session.
- The roadmap/status/backlog (`plan_2026-07.md`, `CURRENT.md`, `TODO.md`) + the v3 map live git-tracked in `docs/project/` and `docs/` -- git IS their history, edit them directly. `internal/` is still unversioned (gitignored): it holds the local-only docs (research/, log/, thoughts/, scratch/, frontend/archive/). Before destructively rewriting a long-lived doc that remains under `internal/`, copy the old version to an `archive/` subdir first -- that copy IS the history.
- CLAUDE.md carries MECHANISMS (how things work, what bites); STATUS (what's done, counts, "until X lands") lives in `docs/project/CURRENT.md` + the plan. Status lines here rot into being actively wrong -- the perf-distrust note did exactly that within a day.

## Tests

- **There is NO "make it fail first" rule here, and adding one back is a
  regression.** Write the check, run the suite, move on. A deliberate red
  (mutate the code, confirm the check goes red, restore) is a TOOL you may
  reach for when you personally doubt a specific check can fail -- an indirect
  oracle, an assertion of absence, one a vacuous implementation would pass --
  never a step, never a batch ritual, and not something to narrate. It was
  asked for twice (2026-08-17, 2026-08-28); the second time the ask was
  explicitly "entirely", carve-outs included, because the habit had been
  codified into this file, `tests/e2e/README.md`, `render.mjs` and
  `plan_chat_orchestration.md` AFTER being rejected, and then justified itself
  on every read. `E2E_V3_ROOT` remains available for the times you want it.
- Run via `/test-suite` (backend only -- there is NO frontend unit suite: the legacy React app that carried one was deleted 2026-07-09; v3 is no-build vanilla JS checked by the opt-in browser E2E below). `tests/unit/` + `tests/contract/` are fully green (Metal-gated skips OK) -- any failure is a regression, investigate it. There is no pre-existing-failure allowlist. (No counts here on purpose: they rot; green-is-the-invariant doesn't.)
- **Behavioral eval bank** (`tests/eval/`, opt-in): 13 tasks covering thinking split/leak, stop discipline, vision correctness, vision-token budgets. Run for changes touching templates/parsers/stop-tokens/vision -- `uv run python tests/eval/run.py --server <url> --models <ids>` against a RUNNING server (never spawns one). Unit tests cannot certify these subsystems (the 07-20 turn-overrun + thinking-leak bugs passed 1000+ of them). BUT the bank runs `stream=False`, so it structurally cannot see chunk-boundary behavior -- that class is owned by `TestParserInvariants` instead; reach for the bank for MODEL behavior, not parser plumbing.
- **Live smoke** (`tests/smoke/`, opt-in, never spawns a server): the half the
  browser suite cannot see. THAT suite drives real `/v3` against a STUBBED
  `/v1`, which left the store's own rules and the generation lifecycle
  unverified; this one talks to a real server and no stub at all. Arms are
  ENGINES, not providers: `"mlx"` routes to TWO separate upstream repos
  (mlx-lm text / mlx-vlm vision, separate release trains) via
  `effective_loader`, so a text arm and a vision arm are different code;
  "covered mlx" is a claim about a config value. A missing arm reports as
  UNCOVERED, never green -- and "served but not run" prints differently from
  "no model of this engine exists", because the first is the quiet one.
  `--contract-only` is seconds and loads nothing.
  Point it at an ISOLATED server (`scripts/dev_server.sh`) -- it writes presets
  and conversations. WHAT "COVERED" MEANS FOR A RELEASE (Phase 4 standard, not
  a CI gate -- there is no CI here and a gate nobody can run is worse than a
  rule somebody follows): before a release touching PROVIDER, LOADER, TEMPLATE
  or LIFECYCLE code, `tests/smoke/` runs green on all three arms, and an
  UNCOVERED arm is named in the changelog rather than passed over. Same for a
  Phase 3 mechanism reported uncovered -- an unmet precondition is a gap with a
  name, and the standing one is thinking DEPTH on both MLX arms (the only
  served MLX model advertising `reasoning_effort` is gpt-oss-120b).
  The taxonomy is `tests/helpers/engines.py`, SHARED with
  `tests/eval/run.py` (whose `fetch_models` it replaced); it reads
  `effective_loader` off the admin row, so an engine is named by the server
  rather than inferred from the vision capability. eval also reports how many
  tasks ran on NO model: its `required_capabilities <= model_caps` filter is
  how a text-only `--models` list ran zero vision tasks under a full green.
- **Browser E2E** (`tests/e2e/`, v1.34.8+): puppeteer-core + system Chrome (claude-in-chrome refuses localhost). Spawns its own server with an isolated `HEYLOOK_DB_PATH` (real data untouched); each suite clears its temp DB; load+warm readiness is the server-owned `POST /v1/admin/models/{id}/load?warm=true` (same contract as `scripts/dev_server.sh` -- never hand-roll poll/warm logic in a harness). NOT wired into `/test-suite` (Metal/GPU-gated + slow + spawns a server) -- opt-in: `cd tests/e2e && bun install`, then `bun run e2e[:chat|:pages]`, MUST run UNSANDBOXED (bun's non-interactive script shell resolves the real node binary; bare `node run.mjs` from an interactive-derived shell hits the nvm lazy-load function, which `export PATH` cannot beat -- the harness itself still executes under node by design, via the package.json scripts). Carries a client-side streaming-cadence guard -- the ONLY automated check for the Phase 1 delivery fix (server telemetry can't see it); needs a fast `E2E_MODEL` (default MoE gemma-4-26B-A4B). A THIRD entry, `bun run e2e:render`, is model-free and server-free (real `/v3` page, stubbed `/v1`, seconds): it guards that the chat message list is RECONCILED, not rebuilt -- `.message` is `content-visibility: auto`, so a row's laid-out height lives on the NODE, and a rebuild collapses `scrollHeight` mid-tick so any pixel-based scroll aims at a list about to grow underneath. Deliberately NOT part of `bun run e2e` (whose Metal/model prerequisites it does not share). `E2E_V3_ROOT` points it at a copy of the frontend, which is how each check was shown to FAIL against a deliberately broken one.
- NEVER apply an MLX `sys.modules` mock at module level with `.start()`; use `with patch.dict(...)` or the `mock_mlx` fixture. A module-level start leaks mocks across the whole session and fakes ~50 "Metal context" failures (the bug that produced the old allowlist).
- `test_mlx_provider.py` SEGFAULTS at GC teardown when run in near-ISOLATION (MLX `unload`/`__del__` flakiness) but passes clean in any multi-file batch / the full suite -- not a regression; run it batched, not alone.
- A SEPARATE interpreter-teardown crash, `Fatal Python error: gilstate_tss_set: failed to set current tstate (TSS)` (exit 134, printed AFTER the pass count), is NOT that MLX-GC class: it needs the MagicMock MLX tree, and reproduces model-free and pytest-free in a bare interpreter -- `import heylook_llm.api` under `patch.dict(sys.modules, create_mlx_module_mocks())` aborts at finalization, while the same import with real MLX exits 0 and the mock tree without that import exits 0. No stray Python thread survives the import, so the foreign thread doing it was not identified (timeboxed). Contract runs on Apple hardware no longer hit it at all since `mlx_mocks` stopped patching there (v1.77.1); it remains a residual on the mocked path.
- Real-MLX failures POISON later real-MLX tests in the same process: one `RuntimeError: [read] Unable to read from file` inside `test_embedding_provider.py` also took down an unrelated `test_jspace.py` case, which passed the moment the first was fixed. Chase the FIRST such failure, not the count. Its cause is worth knowing generally: `mx.load` is lazy/mmap-backed, so `mx.save_safetensors` back over the SAME path without `mx.eval`-ing the loaded arrays first corrupts them -- latent in that test until mlx 0.32.1 surfaced it.
- Provider unit tests build `MLXProvider` from RAW config dicts (bypassing `MLXModelConfig` validation; production passes `model_config.config.model_dump()`), so provider/loader code must tolerate un-normalized config (e.g. missing `modalities`) -- a back-compat branch that looks dead in the router path may be live only in tests.
- Backend: `uv run pytest tests/unit/ tests/contract/ -v`. INVOCATION ORDER IS
  NOT LOAD-BEARING (v1.77.1, verified on Apple hardware in both directions and
  each directory alone) -- historically it was: tests/contract/conftest.py's
  SESSION-scoped sys.modules MLX mock tore down only at the end of the whole
  run, so contract-first left every later unit test looking at MagicMock arrays
  (~57 failures + 8 collection errors that all passed in isolation). The fix is
  that `mlx_mocks` now SKIPS the patch when real MLX imports
  (`helpers.mlx_mock.real_mlx_available`), because contract tests drive
  FakeProvider and only ever needed the mock so imports would succeed where MLX
  is absent. Narrowing the fixture's SCOPE would not have sufficed: a heylook
  module first-imported under the patch binds MagicMocks into its own namespace
  permanently, since the module object outlives the patch. Where MLX is genuinely
  absent the session mock still applies and that residual order-sensitivity
  stands, untestable from here. `--timeout` is not
  installed. `settings.local.json` exempts `uv run pytest`/`uv sync`/`uv lock`/`bun install`/`bun run build` from the sandbox.
- The `helpers.mlx_mock` tree must cover every DOTTED module path a heylook
  module imports at module level (`import a.b` consults `sys.modules['a.b']`;
  a MagicMock `a` is not a package) -- but ONLY those. Adding paths that
  product code probes OPTIONALLY makes the absent-dependency branch untestable:
  mocking `mlx_vlm.generate.diffusion` turned `_detect_diffusion`'s
  returns-False-when-unavailable test green-to-red. Attribute pulls off an
  already-mocked module are free.
- Root venv: plain `uv sync` is the whole story now (v1.39.17). The performance stack (uvloop, cachetools) are CORE deps (questionary retired 2026-07-28 with config_tui; pyturbojpeg + xxhash dropped 2026-08-18 -- both only flipped a status flag, never called), and dev tooling (pytest+plugins, httpx, pyyaml, rich, py-spy) is the `dev` dependency-group uv installs by default (build/twine dropped 2026-08-18 -- `uv build`/`uv publish` cover both) -- there are NO optional extras anymore, and no `--all-extras` to forget. `uv sync --no-dev` for a runtime-only install. Dependency updates are PLAIN UV -- there is no updater script: `uv lock --upgrade[-package X]` + `uv sync`; pyproject.toml is a hand-maintained manifest of published releases. Running an upstream's git commit is a MACHINE-LOCAL experiment: a `[tool.uv.sources]` entry + relock (recipes incl. the zero-footprint `uv run --with`: `internal/deps.md`), and the committed pyproject/uv.lock stay on releases -- `scripts/guard_stable_channel.sh` (pre-commit) blocks committing a pin, because uv honors no gitignored home for source pins and pins always propagate into uv.lock; a dirty pyproject/uv.lock while pinned is the DESIGN, not a mess. llama-server is built by `scripts/build_llama.py` (see the Architecture section above). Build flags + their rationale (why no LTO, no OpenMP, and why `GGML_METAL_NDEBUG` stays OFF): `scripts/README.md`.
- Separate venvs (cd first): batch-labeler (`uv sync --dev`), optloop-lib (`uv sync`).
- GPG signing needs the 1Password agent; if a commit fails on socket errors use `git -c commit.gpgsign=false commit` (`-c` before `commit`).
- Sandbox traps: `ENV=x uv run ...` does NOT match the uv exemption (env-var prefix changes the command match -> sandboxed, no Metal); sandboxed `curl` can't reach localhost (probe via `uv run python` + urllib); never launch the server piped to `head` (SIGPIPE wedges it -- redirect to a file). To verify schema-neutrality of a change (no committed OpenAPI artifact exists -- deliberate), export `app.openapi()` from a HEAD~1 worktree and byte-compare. Sandboxed `find` can silently return nothing traversing `modelzoo/` (files present per `ls`) -- enumerate model dirs with `ls` or `uv run python` glob/`os.walk` instead.
