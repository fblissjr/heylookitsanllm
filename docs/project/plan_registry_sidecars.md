# Plan: retire per-model entries from models.toml

last updated: 2026-09-08 (PROPOSED — nothing built; Phase 0 is the gate.)

## The decision

**Per-model entries should not exist in models.toml.** Owner call, 2026-09-08.
A model's own settings belong with the model; the central file keeps only what
is genuinely not per-model.

The case rests on the failures below, which have happened, and NOT on the
chat-template override that inspired it — see "The precedent does not
generalise".

## Why do this at all

The bug list says what goes wrong. It does not say why the work is worth its
cost, and the cost is real: a new config surface, a migration of an unversioned
file, the admin editor rewritten, a CLI command retired, and a documented model
inverted in several docs. Nobody is blocked today. Every incident so far was
individually survivable. So the case has to be better than "these were
annoying", and it is.

**The principle, which is the owner's and not derived from anything here:** a
model's own settings belong with the model. Everything below is the engineering
case for it, but the principle stands on its own.

**The compounding argument, which is the strongest one.** Materialization
freezes a snapshot of every derived value the moment a model is first edited.
That was tolerable while derivation was static. It is not any more, and the day
this was written is the proof: derivation gained an automatic micro-batch sized
against the live Metal working set, and a vendor sampling layer read from the
GGUF header. **Both silently skip any model somebody once configured.** A
context-size edit made months ago now costs that model a decode setting its own
file asks for, and nothing anywhere says so.

That cost is not fixed, it grows. Every future improvement to derivation
inherits the same hole, and it widens with each one, for models nobody touches
again. A design where a single past edit permanently opts a model out of all
future improvement is the thing to fix — not the individual bugs it produces.

**What the change buys, concretely.** Settings travel with the model: move or
copy the directory and they come along; delete the model and its configuration
goes with it, so dead entries cannot exist by construction. Reverting is
deleting a file. There is no central file to prune, no rename to keep in sync,
no path stored in one place that can drift from a file in another. And
derivation improvements reach every model, permanently, because nothing has
frozen a copy of the old answer.

**The cheaper alternative, and why it loses.** Fix the derivation gaps in Phase
1, prune the file once carefully, and keep the current design. That is a
fraction of the work and it fixes today's symptoms. It loses on two counts.
Materialization immediately starts re-creating the problem for any model anyone
edits, so the pruned file drifts back. And the prune itself is the dangerous
operation — an irreversible bulk edit to a gitignored file, resting on exactly
the reasoning that produced the twin bug. If that alternative is chosen, Phase 0
is still required, which is a good sign that Phase 0 is the right first step
under either plan.

**What would make this not worth doing.** If the read-only-directory question
below resolves badly, per-model entries have to survive for that case, and the
plan degrades from "retire them" to "retire most of them" — still worth
something, but much less. Answer it before Phase 2, not during it.

## Why the current shape keeps producing bugs

models.toml is override-only: `[scan].folders` IS the registry, and
`merge_discovered` folds everything found there into the served set. An entry is
not a layer over the discovered record — it REPLACES it, matched on resolved
path. Three consequences, each of which has cost something real:

- **An explicit entry receives none of discovery's derived fields.** Bit twice:
  a context-size edit cost a vision model its projector (a thin materialized
  entry had no `mmproj_path`), and enabling speculative decode required writing
  the drafter path by hand because the pairing discovery would have done is not
  contributed to an entry that exists.
- **Materialization has to write everything**, which freezes a snapshot of
  derived values at the moment of the first edit. Every later improvement to
  derivation — the auto micro-batch, the gguf vendor layer — stops reaching that
  model, silently.
- **Two entries can claim one path.** The small Qwen appears twice on purpose
  (plain, and with a text-only loader for the smoke arm). The plain entry's
  config is byte-identical to what discovery derives, so it reads as redundant.
  Delete it and the model disappears entirely, because the twin still claims the
  path. Found by simulating a prune; reasoning from the documented rule ("an
  explicit entry always wins, discovery can only add") predicts the opposite.

### The cause, not the symptom

Every incident above is someone predicting the merge rather than running it —
and **prediction is the only option available at the point of decision**,
because the merge has no observable form there. A person editing an entry has
nothing to consult but a rule.

That rules out the fix already tried. This family is documented at length in
CLAUDE.md — resolved-path matching, discovery-can-only-add, an entry receiving
no derived fields, the duplicate-by-symlink case — and the incidents kept
happening anyway, including more the day this was written. Stronger still:
checks that could not fail were written that same day by the sessions who had
spent it hunting checks that could not fail. Knowing a rule, having just written
it down, does not prevent the failure the rule describes.

So the defence belongs at the decision point. Documentation is the explanation
of what the tool shows, never the guard.

## The precedent does not generalise

The chat-template override (v2.0.22) is one file in the model's own folder, and
its reasoning reads like this plan's. It is not. Its author names three
premises, all template-specific:

- the vendor already ships a file of exactly that kind in that directory, so a
  sidecar beside it is in-idiom;
- both engines already DISCOVERED files in that directory, so it added a rung to
  two existing ladders and built no discovery mechanism;
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

### The trade this makes

Making entries per-field OVERLAYS was considered and rejected, mainly because
overlay makes discovery load-bearing: a degraded scan would silently strip an
entry's fields, and discovery is best-effort by construction. An earlier draft
claimed the sidecar DISSOLVED that, because an unreadable directory means the
model is not served anyway, so the failure modes move together.

They do move together — toward BOTH being lost, where today one survives.
Discovery is best-effort in more ways than an unreadable directory: a failing
scan is logged and DROPPED, never fatal. Today an explicit entry is a durable
statement that this model should be served, and it outlives a scan that
degrades. Under sidecars, a dropped scan takes the model and its settings
together.

That bites hardest on exactly what discovery cannot derive. A speculative-decode
carve-out and a context size are POLICY, and under sidecars a model on an
unmounted volume takes its policy with it silently instead of erroring. This is
a TRADE, not a refutation, and the plan states it as one. Whether it is
acceptable is the owner's call, and it should be made knowingly.

The other rejections (a config migration, a suppression gap, determinism loss,
an inverted rule in several docs) are answered by this shape, except the
migration, which Phase 0 exists to make safe.

## Phases

Ordered by dependency, not by appetite. **Every phase changes which models get
served**, which is exactly what nobody can currently see before doing it.

### Phase 0 — the served-set diff (GATE)

Nothing else starts until this exists.

**The argument that buys the gate** is not "every phase changes the served set"
— that argues for a migration TOOL, which could be a throwaway script. The
load-bearing fact is that **models.toml is gitignored**. Phase 4 is an
irreversible bulk edit to an UNVERSIONED file, and the reasoning it would rest
on — "the great majority are byte-identical to what discovery derives" — is word
for word the reasoning that produced the twin bug. Same sentence, same
confidence, wrong, and with no history to recover from. That alone buys the
gate.

#### Signature

```
served_diff(before, after) -> {gained, lost, changed}
```

where each side is a `(config_data, discovered)` pair:

- `config_data` is parsed models.toml, exactly what `merge_discovered` takes.
- `discovered` is the list `model_registry.discover()` returns, **passed in by
  the caller**.

This is the part an earlier draft got wrong, and getting it wrong would have
sent an implementer down a blind alley. That draft said three things that cannot
all hold: that the function CALLS `merge_discovered`, that it TAKES `AppConfig`,
and that it is a PURE function whose fixtures are therefore free.
`merge_discovered` takes dicts, and `AppConfig` is the *output* of validating
the merge — so "take AppConfig" is unimplementable as written.

Injecting `discovered` resolves it. `discover()` is the only part that touches
the filesystem; `merge_discovered` is already pure over its two arguments. So
the diff calls the real merge (never a second implementation of the matching
rule — a tool that can disagree with the server is worse than no tool), stays
pure, and takes synthetic fixtures.

#### Two hazards the signature has to respect

**Validate each side internally, and validate before reading anything.**
`effective_loader_for_config` needs a resolved capability declaration. Given a
raw merged dict it used to answer `mlx-lm` for every model including vision
ones, with no exception and no log line. As of v2.0.30-era `45a03d1` it RAISES
instead, which is strictly better — a diff built on raw dicts now fails loudly
rather than reporting engine churn that never happened — but the requirement is
unchanged: merge, then validate, then compare.

**Validation MUTATES the structure it validates.** `AppConfig(**merged)`
replaces each entry's nested `config` dict with a model instance IN PLACE, so a
later read of the same structure gets a model where a dict was. Copy before
validating, or build both sides from independent reads. (Verified.)

#### Behaviour

- Wire it into every mutation (update, remove, toggle, materialize) as a
  RETURNED DISCLOSURE, not a refusal. The one existing refusal stays: removing a
  disabled override silently re-enables a model, and that inverts a decision the
  operator made. Nothing else earns a block — sometimes losing the model is the
  point of the deletion.
- A dry run, so the question can be asked before the edit rather than after.

The mechanism is already proven here: `ModelService.remove_config` builds a
candidate config, runs the merge and checks whether the model survives. It is
this exact instrument, scoped to a single case.

#### Test shape is a PROPERTY, not a table of cases

For any config and any single-entry edit, every id served before is either still
served after or named in `lost`. That catches the twin without anyone having
imagined twins, which is the entire problem — the prune that found the twin
predicted fewer losses than it got, because the extra one was unimagined. Repo
precedent: `TestParserInvariants` over chunk boundaries, and the render suite
growing generated documents against a whole-document render. Both exist because
example tables missed the case nobody thought of. The known cases (twin, thin
materialization, dead entry) become regression rows UNDER the property.

**Fixtures carry no private or machine data**: synthetic configs with invented
paths. The real models.toml holds absolute home paths. The pure-function design
makes this free if decided now.

#### Risk to design against

Discovery is best-effort, so a scan that degrades between the before and after
snapshots reports every discovered model as lost — a catastrophic-looking lie
from the one feature whose whole value is trust.

Do not detect this by magnitude. A degraded scan and a real mass deletion are
indistinguishable by count, so a threshold is a heuristic dressed as a check.
`model_registry.discover` ALREADY catches its failures per source and logs them;
it just does not return them. Have it report them, and let the diff say "the
scan of this folder failed, so this comparison is unreliable".

#### Permanent surface, or scaffolding?

Deliberately deferred, and named here so the ambiguity does not survive into the
code. If sidecars genuinely remove prediction, the diff's job ends with the
migration and a script suffices. If they do not, it is a maintained API. That is
not knowable until sidecars exist. Build it as a LIBRARY FUNCTION usable either
way, and decide the wired-in API after Phase 2.

### Phase 1 — close the derivation gaps

Each one removes a reason an entry exists. Cheap, independent, and they shrink
the migration.

**Both bullets below were mis-described in an earlier draft. Re-derive before
building either.**

- **A hand-written `supports_thinking` that discovery contradicts.** The earlier
  draft blamed the harmony family: a harmony model always reasons and has no
  `enable_thinking` variable, so `gguf_metadata.supports_thinking` reports it as
  unable to think. That is a real property of the probe, but it is **not** why
  any entry here carries the flag — the harmony model served here is MLX, and
  that probe is gguf-only, so it never sees it. Re-derived: the entries that
  hand-write the flag are all gguf, and discovery already agrees with all but
  one. The exception is the Muse-Glimmer entry, whose discovered sibling probes
  FALSE — and that entry is also the duplicate-by-symlink case. So the live gap
  is one specific model whose template the probe reads differently, not a family
  problem. Establish which before changing the probe.
- **Sidecars in subdirectories.** The earlier draft called a missed drafter in
  an `MTP/` subdirectory "a rule, not a design limit". It is a documented
  decision: `_pick_draft` and `_iter_root_gguf_files` state that an `MTP/`
  subdirectory holds additional PRECISION VARIANTS of the same drafter, that a
  servable pairing needs exactly one drafter path, and that only a root-level
  file is that path. The repo-root-one-level-up case (DeepSeek's `dspark-*`
  beside the quant folders) is already handled. Making this a gap requires a
  POLICY for choosing among precision variants, which this plan does not supply.
  Either supply one or drop the bullet.

### Phase 2 — the sidecar format and reader

- One optional file per model directory, layered over derived defaults; the same
  scan that finds the model finds it.
- Read-only path first. Nothing writes it until Phase 3.
- Precedent to follow for naming and revert semantics:
  `chat_template.heylook.jinja` — and note its filename rule, that a distinct
  name survives a re-download while the vendor's own does not.

### Phase 3 — move the writers

**Writable now.** An earlier draft said to write this phase only after the
bundled-sampler cut landed. It has landed, and the prediction it made is now
fact: `_materialize_discovered` is called from `update_config` and
`toggle_enabled`, and nowhere else. `bulk_set_default_sampler` is gone, along with
`stamp_default_sampler`, `available_samplers` and `get_samplers`.

- The admin config editor writes the sidecar instead of materializing an entry.
- **Materialization is deleted**, and with it the trap it caused twice.
- `heylookllm import` retires. Its derivation half MUST stay:
  `model_registry.discover` imports `ModelImporter`, so discovery and the
  importer are one derivation called from two places. What retires is the WRITE
  half — `generate_toml` has exactly one production caller, the CLI entry point
  — along with the merge-with-existing semantics, the append-only-new-ids rule
  and `--fresh`. Comment preservation (`toml_comments`) stays; admin writes use
  it.
- **Re-home the property that dies with it.** `generate_toml` also has test
  callers, and one is not incidental: it pins that the writer never emits a
  field `GGUFModelConfig` forbids — a real guard against the writer and the
  schema drifting apart. The sidecar writer has the identical failure mode and
  must inherit it, or this is coverage lost in a refactor rather than a guard
  retired with its subject.

### Phase 4 — migrate and prune

Driven by Phase 0's diff, never by reading the file.

**This plan deliberately states no inventory here.** models.toml is gitignored,
so any figure written down is unverifiable by anyone but the owner and rots the
moment the file is edited — an earlier draft's inventory was already wrong on
every count within the same afternoon, including undercounting the entries whose
deletion removes a model, which is the exact class this phase is dangerous
about. Phase 0's dry run IS the inventory, and producing it from the tool rather
than from prose is the whole point of the gate.

What is known structurally, and does not depend on counting: most entries
duplicate what discovery derives; a small number carry real policy overrides
that discovery cannot know (a speculative-decode carve-out, a context size); and
some entries cannot be deleted safely because another entry or a symlinked
spelling claims the same resolved path.

## Converging evidence from another direction

The owner also redirected `GLOBAL_SAMPLER_FLOOR` on 2026-09-08: it should become
per-model, with a bare fallback only where the model's own metadata is silent.
That is this plan's argument arriving from a different problem — the model's own
files as the primary source, central config as the last resort — and it is
better evidence than the reasoning here because it was not reached by reasoning
about the registry at all. It also reframes the vendor layer as the main event
rather than a layer, which makes a per-model entry the leftover rather than the
norm.

## Open, and not to be hand-waved

- **The twin is the ONLY mechanism available, not a convenience.** `loader` is a
  model-config field and appears nowhere on the request surface (verified), so a
  harness cannot force an engine per request — it needs a SERVED ID whose config
  carries that loader. Under sidecars that leaves two options: a second
  directory with its own sidecar, or an explicit exception to "no per-model
  entries".

  **Retiring it has a cost the smoke suite pays.** The mlx-lm text arm has
  exactly two real sources: the twin, and gpt-oss-120b. Remove the twin and the
  cheap text arm becomes a very large model, in a suite whose value is being
  cheap enough to run before every release. Against that: gpt-oss-120b is the
  only served MLX model advertising `reasoning_effort`, which is the standing
  UNCOVERED gap on thinking depth, so moving the arm there would close it.
  Expensive arm, one fewer permanent hole. Decide it deliberately, not as a side
  effect of retiring an entry.

  If the second directory is chosen, **sidecar identity must be the directory
  path AS GIVEN, never RESOLVED.** Two directories symlinking to one weights
  directory would collapse to one model under resolution, rebuilding the twin
  bug in a new coordinate system — and this repo already carries that scar. The
  shape that works is a real directory holding symlinked FILES plus its own
  sidecar.
- **Read-only or shared model directories are a FUNCTIONAL CLIFF**, and may be
  the reason models.toml keeps per-model entries after all. The asymmetry with
  the template case is the whole point: there, a read-only directory costs you
  the ability to override and leaves you the ability to SEE — the write is
  gated, the resolved template still renders, and the panel says why saving is
  unavailable. Here, a read-only directory means the model cannot be configured
  AT ALL, and the fallback anyone would reach for is the central file this plan
  removes.

  Not live today (`watch_hf_cache` is off, nothing is disabled), so this can be
  decided rather than discovered — but it must be decided BEFORE Phase 2, and
  "models.toml stays for exactly this case" is a legitimate answer. One datum in
  the design's favour: `huggingface_hub` prunes nothing on re-download, so a
  sidecar survives, provided its name never collides with something the vendor
  ships.
- **Disabling a model** currently needs an entry. Nothing is disabled today, so
  the case is theoretical, but the sidecar has to answer it.
- **Docs invert in several places.** CLAUDE.md's registry section, the wiki's
  override-only architectural principle, and comments in `model_registry.py` and
  `model_service.py` all describe the replacement model. They change in the same
  commit as the behaviour, or they become the stale claims this repo keeps
  naming.

## Provenance

The session that produced the first draft of this plan, and the audit that
rewrote it, are written up in `internal/` (local, gitignored). Relevant to a
reader: the same failure this plan names — answering locally a question another
function owns — was committed repeatedly while writing it, and several claims
that reached the first draft had to be corrected against the code.

## Sizing, honestly

Materially larger than a session: a new config surface, a migration of existing
entries, the admin editor rewritten, a CLI command retired, and a documented
model inverted in several places. Phase 0 alone is worth building even if the
rest is deferred — it closes the cases that have already cost capability, and it
is what any prune is blocked on.
