# Plan: retire per-model entries from models.toml

last updated: 2026-09-08 (PROPOSED — nothing built; Phase 0 is the gate.
Revised after two independent reviews: the template precedent does NOT
generalise, read-only model directories are a functional cliff, the twin is the
only mechanism available rather than a convenience, and the sidecar TRADES the
discovery-is-load-bearing objection rather than answering it. Third revision:
Phase 0's signature is constrained by two verified hazards, and retiring the
twin has a cost the smoke suite pays. Fourth: a resolver claim corrected, and a
"why do this at all" section added ahead of the bug list.)

## The decision

**Per-model entries should not exist in models.toml.** Owner call, 2026-09-08.
A model's own settings belong with the model; the central file keeps only what
is genuinely not per-model.

The case for this rests on the failures below, which have happened, and NOT on
the chat-template override that inspired it. That precedent was quoted as the
plan's foundation in the first draft and its author refused the generalisation
— correctly. See "The precedent does not generalise" below before citing it.

## Why do this at all

The bug list below says what goes wrong. It does not say why the work is worth
its cost, and the cost is real: a new config surface, a migration of an
unversioned file, the admin editor rewritten, a CLI command retired, and a
documented model inverted in several places. Nobody is blocked today. Every
incident so far was individually survivable. So the case has to be better than
"these were annoying", and it is.

**The principle, which is the owner's and not derived from anything here:** a
model's own settings belong with the model. Everything below is the engineering
case for it, but the principle stands on its own and would be a sufficient
reason even if the case were weaker.

**The compounding argument, which is the strongest one.** Materialization
freezes a snapshot of every derived value the moment a model is first edited.
That was tolerable while derivation was static. It is not any more, and the
day this was written is the proof: derivation gained an automatic micro-batch
sized against the live Metal working set, and a vendor sampling layer read from
the GGUF header. **Both silently skip any model somebody once configured.** A
context-size edit made months ago now costs that model a decode setting its own
file asks for, and nothing anywhere says so.

That cost is not fixed, it grows. Every future improvement to derivation
inherits the same hole, and it widens with each one, for models nobody touches
again. A design where a single past edit permanently opts a model out of all
future improvement is the thing to fix — not the individual bugs it produces.

**What the change actually buys, concretely.** Settings travel with the model:
move or copy the directory and they come along; delete the model and its
configuration goes with it, so dead entries cannot exist by construction.
Reverting is deleting a file. There is no central file to prune, no rename to
keep in sync, no path stored in one place that can drift from a file in
another. And derivation improvements reach every model, permanently, because
nothing has frozen a copy of the old answer.

**The cheaper alternative, and why it loses.** Fix the derivation gaps in Phase
1, prune the file once carefully, and keep the current design. That is a
fraction of the work and it fixes today's symptoms. It loses on two counts.
Materialization immediately starts re-creating the problem for any model
anyone edits, so the pruned file drifts back. And the prune itself is the
dangerous operation — it is an irreversible bulk edit to a gitignored file,
resting on exactly the reasoning that produced the twin bug. If that
alternative is chosen, Phase 0 is still required, which is a good sign that
Phase 0 is the right first step under either plan.

**What would make this not worth doing.** If the read-only-directory question
below resolves badly, per-model entries have to survive for that case, and the
plan degrades from "retire them" to "retire most of them" — still worth
something, but much less. That question should be answered before Phase 2, not
discovered during it.

## Why the current shape keeps producing bugs

models.toml is override-only: `[scan].folders` IS the registry, and
`merge_discovered` folds everything found there into the served set. An entry
is not a layer over the discovered record — it REPLACES it, matched on resolved
path. Three consequences, each of which has cost something real:

- **An explicit entry receives none of discovery's derived fields.** Bit twice:
  a context-size edit cost a vision model its projector (a thin materialized
  entry had no `mmproj_path`), and enabling speculative decode required writing
  the drafter path by hand because the pairing discovery would have done is not
  contributed to an entry that exists.
- **Materialization has to write everything**, which freezes a snapshot of
  derived values at the moment of the first edit. Every later improvement to
  derivation — the auto micro-batch, the gguf vendor layer — stops reaching
  that model, silently.
- **Two entries can claim one path.** The small Qwen appears twice on purpose
  (plain, and with a text-only loader for the smoke arm). The plain entry's
  config is byte-identical to what discovery derives, so it reads as redundant.
  Delete it and the model disappears entirely, because the twin still claims
  the path. Found by simulating a prune; reasoning from the documented rule
  ("an explicit entry always wins, discovery can only add") predicts the
  opposite.

### The cause, not the symptom

Every incident above is someone predicting the merge rather than running it —
and **prediction is the only option available at the point of decision**,
because the merge has no observable form there. A person editing an entry has
nothing to consult but a rule.

That reframing (credit: testclaude, 2026-09-08) rules out the fix already
tried. This family is documented at length in CLAUDE.md — resolved-path
matching, discovery-can-only-add, an entry receiving no derived fields, the
duplicate-by-symlink case — and the incidents kept happening anyway, including
two more the day this was written. Stronger still: three checks that could not
fail were written that same day by the two sessions who had spent it hunting
checks that could not fail. Knowing a rule, having just written it down, does
not prevent the failure the rule describes.

So the defence belongs at the decision point. Documentation is the explanation
of what the tool shows, never the guard.

## The precedent does not generalise

The chat-template override (v2.0.22) is one file in the model's own folder, and
its reasoning reads like this plan's. It is not. Its author names three
premises, all template-specific (2026-09-08):

- the vendor already ships a file of exactly that kind in that directory, so a
  sidecar beside it is in-idiom;
- both engines already DISCOVERED files in that directory, so it added one rung
  to two existing ladders and built no discovery mechanism;
- the thing overridden IS a file, so overriding it with a file is like-for-like
  and revert is deleting one, the vendor's copy never having been touched.

None hold for arbitrary per-model config. There is no vendor file to override,
no existing per-field discovery to extend, and the thing overridden is a TYPED
value carrying an `effect` class — which the reload set, the import allowlist
and `/v1/admin/model-options` are all derived from. A file body carries none of
that.

So generalising means BUILDING a discovery mechanism and a format. The force of
the original argument was "use the ladder already there and add nothing", and
that is exactly the part that does not survive. Cite it as a narrower instance
of the same shape, never as evidence that this is cheap.

## What replaces an entry

A **per-model sidecar in the model's own directory**, discovered by the same
scan, layered over the derived defaults. Same shape and same argument as
`chat_template.heylook.jinja`.

models.toml keeps only what is not per-model: scan folders, the default model,
the load limit. Once it lists no models, it should stop being named as if it
does.

### The trade this makes, which the first draft called an answer

Making entries per-field OVERLAYS was considered and rejected on 2026-09-08,
mainly because overlay makes discovery load-bearing: a degraded scan would
silently strip an entry's fields, and discovery is best-effort by construction.
The first draft claimed the sidecar DISSOLVED that, because an unreadable
directory means the model is not served anyway, so the failure modes move
together.

They do move together — toward BOTH being lost, where today one survives
(testclaude, 2026-09-08). Discovery is best-effort in more ways than an
unreadable directory: a failing scan is logged and DROPPED, never fatal. Today
an explicit entry is a durable statement that this model should be served, and
it outlives a scan that degrades. Under sidecars, a dropped scan takes the
model and its settings together.

That bites hardest on exactly what discovery cannot derive. A
speculative-decode carve-out and a context size are POLICY, and under sidecars
a model on an unmounted volume takes its policy with it silently instead of
erroring. This is a TRADE, not a refutation, and the plan states it as one.
Whether it is acceptable is the owner's call, and it should be made knowingly.

The other rejections (a config migration, a suppression gap, determinism loss,
an inverted rule in several docs) are answered by this shape, except the
migration, which Phase 0 exists to make safe.

## Phases

Ordered by dependency, not by appetite. **Every phase below changes which
models get served**, which is exactly what nobody can currently see before
doing it.

### Phase 0 — the served-set diff (GATE)

Nothing else starts until this exists.

**The argument that actually buys the gate is not the one the first draft made.**
"Every phase changes the served set" argues for a migration TOOL, which could
be a throwaway script. The load-bearing fact is that **models.toml is
gitignored** (verified: `.gitignore` names it). Phase 4 is an irreversible bulk
edit to an UNVERSIONED file, and the reasoning it would rest on — "the great
majority are byte-identical to what discovery derives" — is word for word the
reasoning that produced the twin bug. Same sentence, same confidence, wrong,
and with no history to recover from. That alone buys the gate. (testclaude,
2026-09-08.)

**Permanent surface, or scaffolding?** Deliberately deferred, and named here so
the ambiguity does not survive into the code. If sidecars genuinely remove
prediction, the diff's job ends with the migration and a script suffices. If
they do not, it is a maintained API and must be designed as one. That is not
knowable until sidecars exist. So: build Phase 0 as a LIBRARY FUNCTION usable
either way, and decide the wired-in API after Phase 2, when there is evidence
rather than a guess.

- `served_diff(before, after)` → gained / lost / effective-config-changed, per
  id. It CALLS `merge_discovered`; a second implementation of the matching
  rule would let the tool disagree with the server, which is worse than no
  tool.
- Wire it into every mutation (update, remove, toggle, materialize) as a
  RETURNED DISCLOSURE, not a refusal. The one existing refusal stays: removing
  a disabled override silently re-enables a model, and that inverts a decision
  the operator made. Nothing else earns a block — sometimes losing the model is
  the point of the deletion.
- A dry run, so the question can be asked before the edit rather than after.

The mechanism is already proven here: `ModelService.remove_config` builds a
candidate config, runs the merge and checks whether the model survives. It is
this exact instrument, scoped to a single case.

**Test shape is a PROPERTY, not a table of cases** (credit: testclaude): for
any config and any single-entry edit, every id served before is either still
served after or named in `lost`. That catches the twin without anyone having
imagined twins, which is the entire problem — the prune that found the twin
predicted three losses and got four, because the fourth was unimagined. Repo
precedent: `TestParserInvariants` over chunk boundaries, and the render suite
growing generated documents against a whole-document render. Both exist because
example tables missed the case nobody thought of. The known cases (twin, thin
materialization, dead entry) become regression rows UNDER the property.

**Fixtures carry no private or machine data** (owner constraint, via
testclaude): synthetic configs with invented paths. The real models.toml holds
absolute home paths. The pure-function design makes this free if decided now.

**`served_diff` MUST take validated configs, and this is a signature
constraint rather than a caution.** Verified here after testclaude traced it:
`effective_loader_for_config` given a RAW merged dict returns mlx-lm for every
mlx model — all of them — because the "auto" rule is gated on the model
DECLARING vision, and that declaration is materialised during pydantic
validation, not by the merge. Through `AppConfig` the same computation matches the live listing. No
exception, no warning: a confident wrong answer shaped exactly like a right
one. A diff that accepted a raw dict on either side would report engine churn
that did not happen, from the one tool whose entire value is being trusted
about what an edit does. Take `AppConfig`, not dicts — an API that will not
compile against a shape it cannot answer beats one that answers it wrongly.

**Validation MUTATES the structure it validates.** Found while checking the
above: `AppConfig(**merged)` replaces each entry's nested `config` dict with a
model instance IN PLACE, so a later read of the same structure gets a model
where a dict was. A diff that validates one snapshot and then reads the other
from a shared structure is reading something the first call rewrote. Copy
before validating, or build both sides from independent reads.

**One cause, not two — and a claim this plan got wrong.** An earlier revision
said the resolver "degrades to mlx-lm when it cannot read the model dir",
citing the dead-directory entries. That is wrong on both halves and the
correction matters, because it points the guard at a different layer.

Read `resolve_effective_loader`: the vision branch is gated on the DECLARATION
and returns the text loader BEFORE `model_type` is ever consulted. So the
dead entries land on mlx-lm because they declare text-only, not because their
directories are unreadable — verified, all three carry `['text']`, and none of
them declares `modalities` in models.toml at all, so that value is the schema's
silent fallback. (One of the three is a VL model reporting text-only.)

Better still, the unreadable-directory degradation this plan claimed does not
exist by design: the resolver's own comment says uncertainty (config.json
unreadable, `model_type` None) TRUSTS the vision declaration rather than
silently degrading a working VLM. The claim was not merely unobserved, it is
contradicted by the code.

So both failures are ONE: **the resolver's answer depends on `modalities`, and
`modalities` has a silent text-only fallback.** Raw merged dict, declaration
absent → every model reads mlx-lm. Unreadable directory, declaration
underivable → the fallback stands and the model is silently text. Stated for
the plan: a config in which `modalities` is absent or underivable resolves
every model to mlx-lm, silently and without error, so a diff computed across
such a config reports engine changes that are artefacts of the missing
declaration rather than of the edit.

That is why the constraint above is "take `AppConfig`" and not "guard the
`model_type` read" — guarding the read catches neither path, because the
divergence happens before it. (testclaude, 2026-09-08, measured.)

**Risk to design against:** discovery is best-effort, so a scan that degrades
between the before and after snapshots reports every discovered model as lost —
a catastrophic-looking lie from the one feature whose whole value is trust.

The first draft proposed detecting this by magnitude (an after-scan finding
materially less than the before). That is the same absence-based reasoning this
repo keeps getting caught by: a degraded scan and a real mass deletion are
indistinguishable by count, so the threshold is a heuristic dressed as a check.
Do it directly instead — `model_registry.discover` ALREADY catches its failures
per source and logs them, it just does not return them. Have it report them,
and let the diff say "the scan of this folder failed, so this comparison is
unreliable" instead of inferring trouble from a number.

### Phase 1 — close the derivation gaps

Each one removes a reason an entry exists. Cheap, independent, and they shrink
the migration.

- **Thinking capability on always-reasoning families.** `gguf_metadata.supports_thinking`
  asks whether the embedded template mentions the thinking variable. A harmony
  model always reasons and has no such variable, so the probe asks a question
  that family cannot answer and reports a model that always thinks as unable
  to. This is why one entry hand-writes the flag. Fix the probe, delete the
  entry. OWNED by the session that added the other header readers, and
  sequenced after its sampler-registry removal so two threads are not in
  `gguf_metadata` at once. Likely shape: probe for EITHER thinking variable,
  since harmony reads `reasoning_effort` unconditionally and has no
  `enable_thinking` at all — but check that against a real harmony template
  rather than assert it.
- **Sidecars in subdirectories.** Discovery pairs mmproj / MTP / dspark files
  sitting BESIDE the weights, verified working for all three kinds. A drafter
  in an `MTP/` subdirectory is missed. A rule, not a design limit.

### Phase 2 — the sidecar format and reader

- One optional file per model directory, layered over derived defaults; the
  same scan that finds the model finds it.
- Read-only path first. Nothing writes it until Phase 3.
- Precedent to follow for naming and revert semantics:
  `chat_template.heylook.jinja`.

### Phase 3 — move the writers

- The admin config editor writes the sidecar instead of materializing an entry.
- **Materialization is deleted**, and with it the trap it caused twice. TWO
  call sites to re-home, not three: `update_config` and `toggle_enabled`. The
  third, `bulk_set_default_sampler`, exists only to stamp a bundled-sampler
  name and is going away independently of this plan — the owner approved a full
  removal of the bundled-sampler system on 2026-09-08, which also takes
  `stamp_default_sampler`, `available_samplers`, `get_samplers` and two
  import-time stamp blocks out of the same module. **Write Phase 3 against that
  file AFTER the cut lands, not against a sketch of it now.**
- `heylookllm import` retires. Its derivation half MUST stay: `model_registry.discover`
  imports `ModelImporter`, so discovery and the importer are one derivation
  called from two places. What retires is the WRITE half — `generate_toml` has
  exactly one production caller, the CLI entry point — along with the
  merge-with-existing semantics, the append-only-new-ids rule and `--fresh`.
  Comment preservation (`toml_comments`) stays; admin writes use it.
- **Re-home the property that dies with it.** `generate_toml` also has test
  callers, and one is not incidental: it pins that the writer never emits a
  field `GGUFModelConfig` forbids — a real guard against the writer and the
  schema drifting apart. The sidecar writer has the identical failure mode and
  must inherit it, or this is coverage lost in a refactor rather than a guard
  retired with its subject. (testclaude, 2026-09-08.)

### Phase 4 — migrate and prune

Driven by Phase 0's diff, never by reading the file.

Of the entries present when this was written, the great majority are
byte-identical to what discovery derives, a few point at directories that no
longer exist and warn at every startup, and only a handful carry a real
override. Two of those are policy choices discovery cannot know (a
speculative-decode carve-out, a context size), one is the derivation bug Phase
1 removes, and one is the twin.

## Converging evidence from another direction

The owner also redirected `GLOBAL_SAMPLER_FLOOR` on 2026-09-08: it should
become per-model, with a bare two-value fallback only where the model's own
metadata is silent. That is this plan's argument arriving from a different
problem — the model's own files as the primary source, central config as the
last resort — and it is better evidence than the reasoning here because it was
not reached by reasoning about the registry at all. It also reframes the vendor
layer as the main event rather than a layer, which makes a per-model entry the
leftover rather than the norm.

## Open, and not to be hand-waved

- **The twin is the ONLY mechanism available, not a convenience, and the
  first draft's preferred fix does not work.** `loader` is a model-config field
  and appears nowhere on the request or sampler surface (verified), so a
  harness cannot force an engine per request — it needs a SERVED ID whose
  config carries that loader. Under sidecars that leaves two options: a second
  directory with its own sidecar, or an explicit exception to "no per-model
  entries".
  **Retiring it has a cost the smoke suite pays, and the plan should not
  pretend otherwise.** Five served ids resolve to mlx-lm, but three are the
  dead entries whose paths do not exist and cannot load, so the text arm has
  exactly two real sources: the twin, and gpt-oss-120b. Remove the twin and the
  cheap text arm becomes a 120B model, in a suite whose value is being cheap
  enough to run before every release. Against that: gpt-oss-120b is the only
  served MLX model advertising `reasoning_effort`, which is the standing
  UNCOVERED gap on thinking depth named in the last two smoke reports, so
  moving the arm there would close it. Expensive arm, one fewer permanent hole.
  Decide it deliberately, not as a side effect of retiring an entry.
  (testclaude, 2026-09-08.)
  Note also that the dead entries are not merely log noise: they count as
  mlx-lm arm sources to any static analysis that does not check whether the
  path resolves.
  If the second directory is chosen, **sidecar identity must be the directory
  path AS GIVEN, never RESOLVED.** Two directories symlinking to one weights
  directory would collapse to one model under resolution, rebuilding the twin
  bug in a new coordinate system — and this repo already carries that scar in
  the duplicate-by-symlink entry. The shape that works is a real directory
  holding symlinked FILES plus its own sidecar. (testclaude, 2026-09-08.)
- **Read-only or shared model directories are a FUNCTIONAL CLIFF, and may be
  the reason models.toml keeps per-model entries after all.** The asymmetry
  with the template case is the whole point (raised by its author,
  2026-09-08): there, a read-only directory costs you the ability to override
  and leaves you the ability to SEE — the write is gated, the resolved template
  still renders, and the panel says why saving is unavailable. Here, a
  read-only directory means the model cannot be configured AT ALL, and the
  fallback anyone would reach for is the central file this plan removes.
  Not live today (`watch_hf_cache` is off, nothing is disabled), so this can be
  decided rather than discovered — but it must be decided BEFORE Phase 2, and
  "models.toml stays for exactly this case" is a legitimate answer.
  One datum in the design's favour, verified 2026-09-08: `huggingface_hub`
  prunes nothing on re-download — no `unlink`, no `rmtree` in the snapshot path
  — so a sidecar survives, provided its name never collides with something the
  vendor ships.
- **Disabling a model** currently needs an entry. Nothing is disabled today, so
  the case is theoretical, but the sidecar has to answer it.
- **Docs invert in several places.** CLAUDE.md's registry section, the wiki's
  override-only architectural principle, and comments in `model_registry.py`
  and `model_service.py` all describe the replacement model. They change in the
  same commit as the behaviour, or they become the stale claims this repo keeps
  naming.

## Sizing, honestly

Materially larger than a session: a new config surface, a migration of existing
entries, the admin editor rewritten, a CLI command retired, and a documented
model inverted in several places. Phase 0 alone is worth building even if the
rest is deferred — it closes the two cases that have already cost capability,
and it is what any prune is blocked on.
