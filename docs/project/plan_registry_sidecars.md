# Plan: retire per-model entries from models.toml

last updated: 2026-09-08 (PROPOSED — nothing built; Phase 0 is the gate)

## The decision

**Per-model entries should not exist in models.toml.** Owner call, 2026-09-08.
A model's own settings belong with the model; the central file keeps only what
is genuinely not per-model.

This is not a new idea in this repo, it is the general case of one already
taken. The chat-template override (v2.0.22) is deliberately ONE file in the
model's own folder rather than a models.toml value, and its stated reasoning is
the whole of this plan: nothing writes central config, so there is no path to
materialize an entry, nothing can drift between a stored path and a file, and
revert is deleting one file. That argument does not stop at templates.

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

## What replaces an entry

A **per-model sidecar in the model's own directory**, discovered by the same
scan, layered over the derived defaults. Same shape and same argument as
`chat_template.heylook.jinja`.

models.toml keeps only what is not per-model: scan folders, the default model,
the load limit. Once it lists no models, it should stop being named as if it
does.

This also dissolves the objection that killed an earlier proposal. Making
entries per-field OVERLAYS was considered and rejected on 2026-09-08, mainly
because overlay makes discovery load-bearing: a scan that degrades would
silently strip an entry's fields, and discovery is best-effort by construction.
A sidecar in the model's own directory has no such split — if the directory is
unreadable the model is not served at all, so the failure modes move together
instead of apart. The other rejections (a config migration, a suppression gap,
determinism loss, an inverted rule in several docs) are answered by this shape
too, except the migration, which Phase 0 exists to make safe.

## Phases

Ordered by dependency, not by appetite. **Every phase below changes which
models get served**, which is exactly what nobody can currently see before
doing it.

### Phase 0 — the served-set diff (GATE)

Nothing else starts until this exists. It is the instrument that makes the rest
safe, and it is the piece a prune is blocked on.

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

**Risk to design against:** discovery is best-effort, so a scan that degrades
between the before and after snapshots reports every discovered model as lost —
a catastrophic-looking lie from the one feature whose whole value is trust. If
the after-scan finds materially less than the before, report the comparison as
unavailable rather than report a catastrophe.

### Phase 1 — close the derivation gaps

Each one removes a reason an entry exists. Cheap, independent, and they shrink
the migration.

- **Thinking capability on always-reasoning families.** `gguf_metadata.supports_thinking`
  asks whether the embedded template mentions the thinking variable. A harmony
  model always reasons and has no such variable, so the probe asks a question
  that family cannot answer and reports a model that always thinks as unable
  to. This is why one entry hand-writes the flag. Fix the probe, delete the
  entry.
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
- **Materialization is deleted**, and with it the trap it caused twice.
- `heylookllm import` retires. Its derivation half MUST stay: `model_registry.discover`
  imports `ModelImporter`, so discovery and the importer are one derivation
  called from two places. What retires is the WRITE half — `generate_toml` has
  exactly one production caller, the CLI entry point — along with the
  merge-with-existing semantics, the append-only-new-ids rule and `--fresh`.
  Comment preservation (`toml_comments`) stays; admin writes use it.

### Phase 4 — migrate and prune

Driven by Phase 0's diff, never by reading the file.

Of the entries present when this was written, the great majority are
byte-identical to what discovery derives, a few point at directories that no
longer exist and warn at every startup, and only a handful carry a real
override. Two of those are policy choices discovery cannot know (a
speculative-decode carve-out, a context size), one is the derivation bug Phase
1 removes, and one is the twin.

## Open, and not to be hand-waved

- **The twin has no home in this design.** One directory is one model; a
  per-directory sidecar cannot produce two served ids from one file. The twin
  exists only so `tests/smoke/` has a text-arm model on weights that would
  otherwise route to the vision library. Preference: fix it in the harness
  (pass the loader explicitly) rather than invent a variants mechanism to
  preserve a test need in the serving config. Not yet decided.
- **Read-only or shared model directories break the sidecar.** Not live —
  `watch_hf_cache` is off and nothing is currently disabled — but writing into
  an HF cache snapshot is wrong, and the obvious fallback (a heylook-owned file
  keyed by resolved path) reintroduces exactly the stored-path drift the
  sidecar exists to avoid. Know the boundary before turning cache scanning on.
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
