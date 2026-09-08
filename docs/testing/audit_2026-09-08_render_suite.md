# Render suite: mutation audit

**A dated record, not a living document.** Measured 2026-09-08 against
`3669cf1..66d6131`. Re-derive before trusting anything here; §1 makes that
cheap. Items closed since are marked inline.

Question asked: how many of the checks in `tests/e2e/render.mjs` are testing
something, and how would we know?

## 1. Method (repeatable)

`E2E_V3_ROOT` points the suite at a different copy of the frontend. So: copy
`frontend/`, plant ONE deliberate defect in the copy, run the suite against it,
and record which checks go red. A check that goes red has been shown to catch a
real defect. A check that never goes red has not.

    cp -R frontend "$SCRATCH/M01" && <edit one line in the copy>
    E2E_V3_ROOT="$SCRATCH/M01" bun run e2e:render

Runs are safe in parallel — the stub server binds an ephemeral port and
`launchBrowser` uses no fixed profile — but see §4 on load-induced flake.

Three rules the run itself taught, each earned:

- **Run an unmutated copy first.** If the copy path breaks anything, every
  mutation reads as "killed everything" and the matrix is noise.
- **A mutation that kills nothing is evidence only after you prove it reached
  its subject.** Two of them were duds here, and both looked exactly like
  coverage holes (§3).
- **Score the page-error canaries separately.** They go red on any throw, so
  they inflate the kill count of every unrelated mutation.

## 2. Result

| | | |
|---|---:|---|
| Verified load-bearing | 25 | caught a planted defect |
| Page-error canaries | 18 | caught none of the planted defects |
| Not reached by any mutation | 73 | an audit gap, not a verdict |

The unreached group is unexamined, not condemned. Attachments and staging,
thinking loss, media rows, editing, model switching, touch parity and context
select were never targeted. Reading that row as "checks we should delete" would
be exactly the false confidence this exercise exists to prevent.

The kill matrix per mutation, and the unreached checks listed by name, are in
the published report:
https://claude.ai/code/artifact/1efda03f-b6ac-4689-9e93-c0a11f80c2cc

## 3. What it found

**The link half of `markdown.js`'s scheme guard was invisible to the suite.**
Deleting that guard outright left the suite green. Not a mutation that missed:
with DOMPurify removed *as well* the check goes red, and with DOMPurify removed
and the guard intact it stays green — each layer independently satisfies an
oracle that asserts on the resolved protocol, so that oracle cannot see either
one regress alone. The guard's own comment called those checks its only
coverage. The image half was genuinely covered.

**Closed** in `66d6131`. The two layers refuse *differently* — the guard drops
the anchor and keeps the label, DOMPurify keeps the anchor and strips the
attribute — so "no anchor survives" is a property only the guard satisfies.

**One check was a race rather than an oracle.** `the reset button says it
clears overrides` read the first `button` in the Sampling panel, and the panel
carries a hidden per-row reset for every sampler control ahead of the one it
meant. It passed sequentially and failed under CPU load with the button
entirely correct; a probe inside a *passing* run found the per-row resets
sitting in front of it. **Closed** in `dee159f`, anchored on
`button:not(.settings-row__reset)`, verified red against a relabelled copy.

**Two mutations were duds, and both read as coverage holes until checked.**
`replaceChildren(...nodes)` passes the *same node objects*, so identity
survives and no reconciliation check can notice. The display-pref mutation hit
`previewPrompt`'s overrides, not the `generate` request the check reads.
Corrected, both killed checks immediately.

## 4. Still open

- **The block-quote boundary rule in `markdown-stream.js` is unverified.**
  Removing `if (line[0] === '>') return false;` kills nothing, while the
  sibling list-marker rule does kill checks. The property generator already
  emits adjacent quotes. An attempt to construct the document that would
  justify the rule — a fenced block inside a quote — failed, because quoted
  lines are never blank so the scanner never reaches a split point there. That
  does not make the rule dead. **Recommendation: keep it.** Deleting a safety
  rule on an absence of evidence is the wrong trade.
- **The capability-gated sampler filter has no coverage here.** Disabling
  `requiresCap` filtering in `samplerParams` produced no red. The
  server-driven chat suite was not audited and may cover it; check there
  before adding anything.
- **First-match selectors elsewhere.** The reset-button race is a shape, not
  an incident: `$eval` takes the first match, so any check anchored on a bare
  element inside a container is one markup addition from reading a different
  control. `PRESET`'s selectors are already title-anchored with a comment
  explaining why; the settings panel should follow.

### Closed since

- **The per-context page-error canaries are one check** (`c8a4857`).
  `openChat` registers every context it hands out and one check at the foot of
  the run speaks for all of them, naming the `render.mjs` line each offending
  page was opened at. It also asserts that contexts were registered, so it
  cannot go quietly vacuous. It covers contexts that previously had no
  assertion at all. Verified against a copy of the frontend that throws on
  every chat page.
- **The superseded-stream check reads `chat.js`'s status prefixes** instead of
  restating them (`c8a4857`). `GENERATING_PREFIX` and `MODEL_SWITCH_PREFIX`
  are exported now. The hand-copy had been wrong twice and stayed green both
  times.

## 5. What this audit cannot see

- **The mutation set is one person's.** A defect nobody thought to plant is
  indistinguishable here from a defect no check covers.
- **Kill counts overstate independent coverage in the preset chain.** Those
  checks share one page in sequence, so an early break cascades into the ones
  after it.
- **One check was load-flaky and was excluded from the matrix as baseline.**
  That was the reset-button race, since fixed. If another appears, calibrate
  against control runs at the same parallelism rather than trusting a single
  clean run.
- **Stubbed `/v1`.** Nothing here observes the DuckDB store's own rules, the
  generation lifecycle against a real server, or any provider. That half
  belongs to `tests/smoke/`, unaudited.
- **One renderer, one viewport family.** Headless Chrome, with a couple of
  checks resizing to phone width. A WebKit-only defect is unreachable;
  `e2e:ios` exists for that and is recorded as unrun.
