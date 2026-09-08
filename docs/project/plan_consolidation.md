# Consolidation plan

last updated: 2026-09-08

## What this is

A day of audit work across three parallel sessions left a lot of small,
correct changes and no single moment where anyone confirmed the repo is
coherent. This is the
sequence that gets it back to a state you could release from, and then stops.

**It is explicitly not more auditing.** The two audits in
[docs/testing/](../testing/) found what a first pass finds — a blocking
destructor, a scheme guard nothing watched, three oracles that could not fail —
and every one of them came from aiming at something specific. Nothing came from
sweeping. The remaining unprobed surface would cost a great deal for probably
little, and the habit that found all of it is already the repo's rule: when you
doubt a particular check, plant the defect once and watch it go red. That is a
tool for doubt, not a step to institutionalise, and it must not become one.

## Where this starts

Observed 2026-09-08 at `f3af1c4`, re-derived at `7abced7`. Only `scripts/`
and `docs/` changed between them, so the suite rows still hold without a
re-run; the table says how each was established rather than asking you to
trust it:

| | |
|---|---|
| Unpushed commits on `main` | a day's worth, from THREE sessions |
| Versions written today | 2.0.19 through 2.0.28, sequential, no gaps |
| `__version__` vs top changelog section | agree |
| Working tree | clean except `uv.lock`, which the owner has claimed |
| `uv lock --check` | resolves clean |
| Backend + render suites | green |
| `tests/smoke/` since provider code changed | **not run** |

The last row is the one that matters. Everything else is tidy-up.

## Phase 1 — settle the working tree

**Resolved 2026-09-08: `uv.lock` is the owner's and is intended.** It moves
`anyio` and `numpy` by a patch each, and `uv lock --check` resolves clean. No
action; land it whenever the rest lands.

The reason it earned a phase is worth keeping even though the answer was
mundane. It sat dirty across three sessions for a day. Two of them looked at
it, each concluded it belonged to somebody else, and neither asked — so a file
nobody had read was carried through every commit of the day. **A permanently
dirty file stops being information and becomes noise, and it hides the next
real change underneath it.** The cost here was zero. The next one will not be.

Standing rule, in Phase 5: resolve it or commit it, but do not carry it. When
it is not yours, ask whose it is rather than working around it.

**Done when** `git status --short` shows nothing you cannot account for.

## Phase 2 — run the release standard this repo already has

CLAUDE.md sets the bar and today's work crossed it: a release touching
**provider, loader, template or lifecycle** code runs `tests/smoke/` green on
all three engine arms, and an uncovered arm is *named in the changelog* rather
than passed over. Provider code changed across sessions —
`providers/base.py`, `mlx_provider.py`, `llama_server_provider.py`,
`mlx_embedding_provider.py`, `providers/common/template_info.py`.

**In flight 2026-09-08:** another session claimed this run rather than both of
us paying for it, using a warm setup from an earlier full pass that morning.
Note for whoever reads the result: `scripts/dev_server.sh` — the script step 1
depends on — was itself changed by a THIRD session mid-flight (`929be2c`, how
`stop` resolves its target). A server started before that commit is being
managed by a different script than the one now on disk.
That earlier pass is stale by exactly the argument this phase makes — it
predates the provider-teardown change — which is why it is being re-run rather
than cited.

This phase exists because the destructor change is the one thing today that
alters runtime behaviour on a path no unit test can see: `unload()` now takes
`drain`, and `__del__` refuses to tear down when work is in flight. The unit
checks pin the decision; only a live run exercises the real teardown.

1. Start an isolated server — `scripts/dev_server.sh`, never the real one.
2. Cheap first: `uv run python tests/smoke/run.py --server <url> --contract-only`.
   Seconds, loads nothing, and catches a broken contract before you pay for a
   model load.
3. Then the full run against that server, once per engine arm. Arms are
   **engines, not providers**: `"mlx"` routes to two separate upstream repos
   via `effective_loader`, so a text arm and a vision arm are different code.
   A missing arm reports UNCOVERED, never green.
4. Exercise teardown specifically, since that is what changed: load a model,
   generate, then evict it (load a second model past `max_loaded_models`, or
   let `idle_unload_seconds` fire). Watch for the new warning — it should not
   appear. If it does, a provider is being collected while it believes it is
   generating, which is the leaked-counter or dropped-reference case the
   warning names, and it is a real bug rather than noise to tune.
5. Name the result in the changelog, including any arm that came back
   UNCOVERED.

**Done when** the smoke result and every uncovered arm are written into the
changelog entry.

## Phase 3 — make the changelog true

**Precondition: the changelog must be settled first.** It was dirty in another
session's tree when this phase was last checked, and auditing an artifact while
someone is writing it produces findings about a draft. Wait for
`git status --short` to show `CHANGELOG.md` clean, AND for Phase 2's smoke
result to be written into it — that entry is part of what this phase audits.

The day's version sections were written by three sessions, partly from a
subagent's report, partly from measurements taken mid-session. It is the
artifact most likely to be quietly wrong, and this repo has been burned by
exactly that before.

1. Run `/claim-audit:claim-audit` over the prose added across those sections.
   It re-derives every count, status and attribution **by executing commands**
   rather than by reading, and labels what cannot be derived.
2. Apply the owner's no-numbers-in-prose rule to what it surfaces. The order is
   **delete, then bind, then derive** — deleting beats correcting, and a
   measurement worth keeping goes where its conditions travel with it
   (`internal/research/`), not into a sentence. Entries written earlier today
   predate that rule and carry counts in prose.
3. Fix the one stale claim already known: `tests/e2e/render.mjs`'s own header
   still describes the suite as taking "a few seconds". The README was
   corrected; the file's own comment was not.

**Done when** every claim in the day's entries is either re-derived or deleted.

## Phase 4 — leave one list of what is open, then stop

The audits already carry their own open items, marked closed inline as they
land. Do not restate them anywhere else — a second copy is the drift this repo
names for hand-copied constants. Confirm each is still accurately marked, and
leave it at that.

Standing open, with the reason to leave each alone:

- **The block-quote boundary rule in `markdown-stream.js`** — removing it kills
  nothing, but the document that would justify it could not be constructed.
  Unverified is not dead. Keep the rule.
- **The capability-gated sampler filter** — no render-suite coverage. Check
  whether the server-driven chat suite covers it *before* writing anything new.
- **The `mock_mlx` envelope gap** — real, and not a one-line fix. The contract
  suite needed mocks only so imports would succeed; unit tests use the mock
  objects' behaviour, so removing the patch breaks tests rather than revealing
  drift.
- **Most of the backend suite unprobed** — deliberate. See the opening section.

**Done when** nothing new has been added to this list.

## Phase 5 — the protocol that stops today recurring

Parallel sessions are normal here, and every collision today was avoidable.
These are rules, not observations:

- **Stage explicitly. Never `git add -A` or `-u`.** Run `git status` immediately
  before committing and leave anything you did not touch unstaged. Edits were
  swept into another session's commit today; the content survived, the
  authorship did not.
- **Never `git checkout <file>` to undo a probe.** It reverts your own
  uncommitted work in that file along with it — which happened today, silently.
  Undo a probe by the inverse edit, and verify the revert by **blob hash
  against HEAD**, not by `git status`: the dirty set moves under you while
  another session commits.
- **Re-read `__version__` immediately before bumping it, and verify the edit
  landed.** A concurrent bump makes a scripted replace a silent no-op.
- **A pin that fires because the code moved is the pin working.** Move it with
  the call; do not delete it. `test_mlxvlm_surface.py`'s `vlm_inputs.py` pin
  went red in one tree and green in another for exactly this reason, and it
  exists because the previous pin stayed put when the code moved.
- **A permanently dirty file is a blind spot.** Resolve it or commit it; do not
  carry it. When it is not yours, **ask whose it is** — three sessions each
  assumed `uv.lock` belonged to one of the others and nobody asked for a day.
  See Phase 1.
- **Attribute by asking, not by inferring — and `git log` cannot answer it.**
  Two separate misattributions happened today, in both directions: a fix
  credited to a session that did not make it, and a `cp -R` diagnosis that was
  confidently wrong about another session's setup. The first version of this
  rule said to consult `git log`; that is wrong here and was corrected by the
  session it would have misled. **Every commit today carries the same author**,
  because the author is git config, not the session — so the log settles the
  order and content of commits and says nothing about which session made them.
  The only record of WHAT a session did is that session's own. So: ask, and
  keep a record worth asking for. Append your own commit hashes to the day's
  `internal/log/log_YYYY-MM-DD.md` as you go, which is the one place three
  sessions can reconcile after the fact.
- **Say what you are about to run, before running it.** The expensive shared
  step is a smoke run; two sessions doing it independently is pure waste, and
  one session assuming the other did it is worse.

## Done means

Working tree clean. Smoke run on every arm, with uncovered arms named in the
changelog. Every claim in the day's changelog entries re-derived or deleted.
The audits' open lists accurate and not duplicated. Then stop — and if the next
session wants to extend coverage, it aims at something it doubts, one defect at
a time.
