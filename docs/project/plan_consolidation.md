# Consolidation plan

last updated: 2026-09-08

## What this is

A day of audit work and a parallel session left a lot of small, correct changes
and no single moment where anyone confirmed the repo is coherent. This is the
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

Observed 2026-09-08, at `f3af1c4`:

| | |
|---|---|
| Unpushed commits on `main` | 11, from two sessions in one day |
| Versions written today | 2.0.19 through 2.0.28, sequential, no gaps |
| `__version__` vs top changelog section | agree |
| Working tree | clean except `uv.lock` |
| `uv lock --check` | resolves clean |
| Backend + render suites | green |
| `tests/smoke/` since provider code changed | **not run** |

The last row is the one that matters. Everything else is tidy-up.

## Phase 1 — settle the working tree

`uv.lock` has been modified since before the session and was carried, unread,
through every commit. That is how it stayed invisible: a permanently dirty file
stops being information and starts being noise, and it hides the next real
change underneath it.

1. Read what actually moved: `git diff uv.lock | grep -E '^[+-](name|version) '`.
   At the time of writing it is two transitive patch bumps and nothing else.
2. Confirm the manifest still agrees: `uv lock --check`.
3. Decide, and act on the decision rather than deferring it again:
   - **Keep** — commit it alone, and record the change in
     `internal/log/log_YYYY-MM-DD.md` using the dependency table from the
     doc-conventions skill. That log entry is the only place dependency
     changes are written outside the manifest and lock.
   - **Discard** — `git checkout uv.lock`, and expect it back on the next
     `uv run` if something in the environment is regenerating it. If it comes
     back, that recurrence is itself the finding and belongs in the log.

**Done when** `git status --short` is empty.

## Phase 2 — run the release standard this repo already has

CLAUDE.md sets the bar and today's work crossed it: a release touching
**provider, loader, template or lifecycle** code runs `tests/smoke/` green on
all three engine arms, and an uncovered arm is *named in the changelog* rather
than passed over. Provider code changed across both sessions —
`providers/base.py`, `mlx_provider.py`, `llama_server_provider.py`,
`mlx_embedding_provider.py`, `providers/common/template_info.py`.

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

Ten version sections were written by two sessions in one day, partly from a
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
  carry it. See Phase 1.

## Done means

Working tree clean. Smoke run on every arm, with uncovered arms named in the
changelog. Every claim in the day's changelog entries re-derived or deleted.
The audits' open lists accurate and not duplicated. Then stop — and if the next
session wants to extend coverage, it aims at something it doubts, one defect at
a time.
