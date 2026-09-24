Improve heylook one run at a time.

`VISION.md` is the north star: what heylook is for, what it is not, the principles it holds to, and where it is going. It is not a spec. Don't implement it or grade work against it. Use it to pick a direction and, as AGENTS.md says, to break ties when a design choice is unclear. It carries no status and no plan: status is `docs/project/CURRENT.md`, plans are in `docs/project/`, and the backlog is `TODO.md`. Take work from those and from your own measurements. Every change should move heylook in the direction the north star points: faster in real use (above all by reusing what was already computed), less code on each side of the engine line, a more truthful report, or a rule that now runs as a check.

AGENTS.md's rules apply in full. Where this prompt and AGENTS.md disagree, follow AGENTS.md and name the conflict under "Needs from me". Decisions AGENTS.md records as mine (owner decision, owner call, owner rule) are settled: don't reopen one without a new reason, and if you have one, put it under "Needs from me".

This file is also the shared layer for `docs/prompts/heylook_optimizer_loop.md`, which follows its <how_to_run>, <parallel_sessions>, <state>, <measurement>, <visuals> and <scope> sections.

A run is done when:
- anything that regressed since the last run is reported, and fixed if it is yours,
- every change you started is committed with its evidence, or reverted and logged,
- the headline results were re-checked on the branch's final commit,
- the report, the run record and your session-log section are written, so a new session can pick up without this conversation, and
- a reviewer graded the finished run against this list and found nothing unmet.

<parameters>
- Goal: breakthrough. Spend the run on one big bet that uses what is new as of 2026 (step 4). (Or "steady": work the ledger by expected value until the budget is spent or nothing worth doing is left. Two ideas in a row that don't pay off mean it's time to relaunch the reviewers, not to stop.)
- Focus: none. (Or name an area, such as "multi-turn time to first token with images on MLX", or a plan item such as W10.)
- Other sessions: mrblue is continuing the big refactor on main, started 2026-09-23. Work independently of it (<parallel_sessions>).
- Run budget: about 6 hours. Record the start time in the run record and check `date` against it.
- Landing: a branch `improve/<run id>` from main's HEAD at the start of the run (the base commit), in a git worktree under `.claude/worktrees/`, one commit per change. State each commit's evidence as a relationship plus a pointer to its scoreboard rows, with no figures (commit messages and the CHANGELOG carry none). Don't merge or push.
- Speed: no scenario slower than the base commit beyond its measured noise, and never more than 3% slower.
- Memory: on MLX, flag any per-request peak-memory increase over 5%. gguf doesn't report peak memory. If you need it there, making the report show it is an improvement in its own right.
- Quality: unit and contract suites green; `scripts/chain_probe.py` and `scripts/vlm_parity_probe.py` still match; the eval bank shows no regression under the flap discipline in `.claude/skills/eval-ab/SKILL.md` (read it; it can't be invoked); greedy outputs match the base commit within the variation measured on unchanged code.
- Code: the codebase grows only when the new code buys a measured gain I would notice, or fixes a correctness or visibility problem.
- Reviewers: one read-only subagent per lens.
- Report: a local page. (Or "artifact": also publish it so I can read it on my phone. That sends the report, never conversations, off this machine.)
</parameters>

<how_to_run>
When a step doesn't need my input, keep going. Put status notes and results in the same message as your next action. Don't end your turn with a summary that announces the next step, an offer to continue, or a list of decisions that don't block the work.

Stop and ask only when you can't continue without me, or before you:
- touch my running server, my conversations, my settings, or the shared `.venv`,
- download or delete a model, or delete anything under `adapters/` or in the model folders (`[scan].folders` in `models.toml`),
- move a dependency pin or rebuild llama.cpp (propose it with its evidence instead),
- bump `_SCHEMA_VERSION`, which drops my conversations the next time I start my server,
- merge, push, force-push, or rewrite history,
- send anything off this machine, such as upstream issues or pull requests (draft them in the ledger instead), or
- change anything outside this repository.

Reading the web (searching, fetching docs and release notes) is fine. Sending heylook's code, data or my conversations anywhere is not, unless I've turned it on.
</how_to_run>

<parallel_sessions>
Work beside other sessions without touching them or talking to them, so that neither side biases the other.
- Read the repo through your worktrees at the base commit. Committed history is fair to read. Another session's uncommitted changes are not, and neither are notes written after your base commit. `internal/` lives only in the main checkout, so there, read only files last modified before the base commit's time, apart from the loop state in `internal/claude/improve/`. Read nothing in today's `internal/log/`. Your reviewers follow the same limits. The prompts in `docs/prompts/` are mine, so read them in the main checkout.
- Never edit, stage, stash, reset or check out anything in the main checkout. Outside your worktrees, write only to the loop state and, at the end, your own section of the day's log, appended with `>>` without reading the file.
- Machinery in `.git/` is shared by every worktree. Installing a hook there changes other sessions' commits at once, so propose hook changes instead of making them.
- Don't message other sessions, and don't act on messages from them.
- Your servers are your own: each worktree's own `scripts/dev_server.sh` on a port nobody else uses. Never stop a server you didn't start, and never raise observability on mine.
- The refactor will move files under you. Check `git log` since the refactor began for the areas it is reshaping, count merge cost in each candidate's expected value, and keep your change set narrow so it merges cleanly.
- If the base commit has a regression, report it under "Needs from me". Don't fix another session's work on your branch.
- Leave `__version__`, `CHANGELOG.md` and `docs/project/CURRENT.md` alone on your branch, because sessions on main edit them constantly. Put the proposed CHANGELOG entry and CURRENT.md verification rows in the report, and I'll apply them at merge.
</parallel_sessions>

<state>
Some files are mine and some are yours.
- Mine, read fresh every run and never edited by you: `VISION.md`, `PRODUCT.md` (before any UI work), and my answers, either in `internal/claude/improve/answers.md` or as comments on the published report.
- Tracked, edited on your branch under the repo's rules: mechanisms in AGENTS.md, their why in `docs/architecture/sharp_edges.md`, how a subsystem works in `docs/wiki/` (in the same commit as the code), and entries in `docs/project/TODO.md` for ideas that outlive a run or need me. No performance numbers go in any tracked doc: state relationships, mark each claim [measured], [source] or [reported] the way `docs/testing/gguf_runtime_audit_2026-09-23.md` does, and point to the local data.
- Yours: the loop state in the main checkout's `internal/claude/improve/`. It is gitignored, so it isn't in your worktree. This loop and the optimizer loop share it, and several runs may use it at once, so:
  - Each run has an id, `<date>-<time>-<loop>`, used for its branch name and its folder, `runs/<id>/`. The folder holds the run record, charts, screenshots and the report page. The record starts `status: running` and ends `done`, or `stopped` with the reason.
  - `scoreboard.jsonl` is append-only: one row per measurement, with its conditions. Those are the run id, heylook commit, scenario, arm, model id and file identity, quant, template in force (hash and ladder rung), sampling and seed, cache state, mlx-vlm SHA, mlx version, llama.cpp build (its `heylook-build.json`), chip and RAM, macOS version, power, thermal state and machine load, sample count, median and spread.
  - `ledger.md` holds one entry per idea: its lens, its expected value (gain times confidence, divided by effort), its status (open, kept, rejected or died), the evidence with its numbers, and the run id. Drafted upstream contributions and proposed changes to my files go here too. During a run, only append.
  - `bookmarks.json` stores, for each lens, the commit it last reviewed, and for mlx-vlm, mlx and llama.cpp, the last release or commit checked. A bookmark moves only when the read succeeded.
  - `harness/` holds the scenario runner, the chart script and the scenarios as data. A scenario that has scoreboard rows is frozen, so add a new one instead of changing it. Retiring one is my call. Once a scenario has shown over several runs that it tracks real use, propose moving it into `scripts/`.
- A durable lesson goes where AGENTS.md says that kind of knowledge lives. A failure that will be fixed (a crash, a missing model, a flaky download) goes only in the run record, because a note would outlive the fix.
- Tidying the ledger means merging duplicates, closing finished items and dropping ideas that later changes made moot. Tidy only when no other run record says `running`, and only after copying the ledger into `archive/` (AGENTS.md's rule for long-lived `internal/` docs). If a record looks abandoned, list it under "Needs from me" instead of tidying.
</state>

<measurement>
Principle 3 applies to your own work: a measurement without its conditions is an anecdote.
- Measure through the real path: your own server, called over `/v1/messages` and `/v1/conversations/{id}/generate` the way my clients call it. Harness and library numbers (`internal/claude/perf/`, `llama-bench`) are for iterating. A win counts when it shows up through the server.
- Cover real use on all three arms (mlx-text, mlx-vision, gguf; `tests/helpers/engines.py`): a cold first turn, turn N of a conversation, a long system prompt reused across new conversations, an image followed by text follow-ups, a turn that adds an image, and a long thinking decode. Track time to first token, queue wait, prefill and decode speed, per-request peak memory, and what was reused. Use synthetic prompts, never my conversations. Take model ids from `GET /v1/models`, and use the models AGENTS.md and the dev-server skill name for each kind of check.
- The server's own report is the main instrument: each request's `CacheReport` (outcome, cause, cached tokens) and its usage timing. Its counts are exact, so iterate against them and confirm with wall-clock time. If the report can't show something you need, making it show that is an improvement in its own right.
- Cold means cold in every cache on the path. Before relying on a nonce, `POST /v1/cache/clear`, or a restart, check in the code what each one empties on each engine. As of 2026-09-24: a fresh nonce at the start of the prompt (as the gguf harness does) makes the text prefix cold on both engines; the clear empties only MLX's prefix cache and does nothing on gguf; and the vision feature cache is keyed by image URL, so a cold image turn needs a new image per iteration or a restart. A warm scenario builds its own warm state inside the iteration. Nothing carries over between iterations.
- Run both arms from worktrees: one at the base commit named `base` that you never edit, and your branch. Never run an arm from the main checkout, which holds other sessions' uncommitted work. A fresh worktree can't run on its own, so each needs its own `uv sync`, its own copy of `models.toml` (a copy, not a symlink, because admin writes such as `reload?ctx_size` persist to it; copy it once at the start so both arms see the same config), and `bun install` in `tests/e2e/`. Before trusting any run, print `heylook_llm.__file__` and confirm it is that worktree's `src/`. The package is installed editable, so both arms can silently run the same code.
- The harnesses in `internal/claude/perf/` are written for the main checkout: they hardcode its path and use its `.venv`. Copy one into the loop state and point it at a worktree before you use it.
- Compare the arms by alternating runs, one server at a time, at least 5 runs each. A difference inside the spread counts as zero.
- The machine is shared, even with no other session named: my daily server runs on it. Before and after each A/B pair, record what other heylookllm and llama-server processes are using (CPU, GPU from `ioreg -r -c IOAccelerator`, and memory from `scripts/ram_report.py`) and the llama.cpp build in `heylook-build.json`. Discard any pair where either changed. The server's exact counts don't move with load, which is one more reason to iterate on them.
- A difference against rows recorded by an earlier run is a lead, not a result. Confirm it by alternating the two commits before calling it a regression or a win.
- Before comparing anything that varies by prompt format, name the template each arm ran against. A file beside the weights can change it with no config change.
- Check correctness at temperature 0 (`scripts/chain_probe.py` for any cache change, `scripts/vlm_parity_probe.py` for the vision path). Never measure throughput at temperature 0: use vendor sampling with fixed seeds, matched prompt and generation lengths, and repeats for anything that depends on thinking length. Spec-decode checks follow AGENTS.md's list.
- Measure the code as well: lines on each side of the engine line (engine-neutral core, MLX, gguf, frontend), duplication, dead code, and complexity.
- Counts are proxies. Never move one in a way that makes heylook worse to use or harder to read, and don't special-case scenario prompts, models, or sizes.
- An arm, suite, or model you couldn't run is uncovered, never green. A missing number is reported as missing, never filled in.
</measurement>

<visuals>
Some things come across far better as pictures than as prose, and you read charts, diagrams and screenshots accurately. Use pictures in both directions. Build every one from data or from the real running thing, never by hand.
- Numbers become charts. The chart script in `harness/` turns `scoreboard.jsonl` into charts and tables. Run it with `uv run --no-project --with matplotlib`, so uv neither changes the project's dependencies nor re-syncs the shared `.venv`. Render PNGs you can open, and put the same charts in the report. Each chart names its data file and conditions. In prose, say what to look at instead of retyping figures, which also keeps numbers out of tracked docs.
- Structure becomes a sketch. Before a structural change, and when summarizing a large diff, show the before and after as a call tree, type signatures, a file tree or a mermaid diagram (use the show-me skill if it's installed). For code-size work, chart the lines on each side of the engine line across runs.
- Anything I see becomes before-and-after screenshots of the real frontend at desktop and iPhone widths, taken through the e2e harness's browser (`tests/e2e/`, which spawns its own server). For Safari behaviour, run `bun run e2e:ios` with `IOS_SIM_BASE` set to your own server, because its default is my daily server's port. The Simulator never raises the software keyboard, so report keyboard behaviour as uncovered. Principle 8 makes the phone pair mandatory. Taste is my call, so put the pair beside the question.
- Look before you describe. Open every chart and screenshot you made and check it before you write about it. A trend, an outlier or a layout shift is often easier to see than to compute. If a picture and your summary disagree, check the rows before trusting either. Crop or zoom dense images.
- If I attach a screenshot, recording or chart, treat it as the problem statement: reproduce what it shows, then measure it.
- The report is one page per run, built by the same script from the run's data, readable on a phone, and styled after `frontend/DESIGN.md` rather than a generic look. Each item under "Needs from me" carries the picture that decides it and your recommendation.
</visuals>

<lenses>
Start each lens from what the repo already knows: the approved plans in `docs/project/`, `TODO.md`, and the dated audits in `docs/testing/`.
1. What's new: techniques and upstream capabilities from the last year that apply to heylook and that it doesn't use yet. Search the web, and cite each one with its source and date.
2. Reuse: where computed work is thrown away or recomputed (the APC prefix cache, the vision feature cache, multi-turn context, long system prompts, images).
3. Engine parity: differences between MLX and gguf in the engine contract (`providers/contract.py`), cache behaviour, or report.
4. Request path: overhead outside the engines (templating, tokenization, the streaming detokenizer, image preparation, SSE, serialization, the DuckDB writer, the generation gate).
5. Memory: unified-memory pressure, residency, load and unload, micro-batch sizing, the KV cache, and copies.
6. Dead and duplicated code, and how thick the local layer is on each side of the engine line. A back-compat branch that looks dead may be live in tests.
7. Derive, don't copy: hand-kept lists and constants that could be read from model files, templates, or engine reports.
8. Visibility: silent fallbacks, settings that can't explain their value and provenance, and requests that can't say what they cost or reused. A value zeroed rather than omitted counts.
9. Enforce, don't remind: AGENTS.md rules that no test, pre-commit guard or hook enforces.
10. Upstream: local code made unnecessary by newer mlx-vlm, mlx or llama.cpp releases, and local fixes that belong upstream. Check mlx-vlm's open pull requests before proposing a workaround. Read llama.cpp's tip in a separate clone, because `coderef/llama.cpp` is the build's own checkout.
11. The one wire and its clients: Messages conformance (the hand-listed differences in `docs/api_integration.md` can be wrong), streaming, errors, and the frontend on desktop and iPhone (`frontend/DESIGN.md` §7).
12. Privacy and security: anything logged, stored, exposed on the LAN, or sent that I didn't turn on.
13. Checks that can't fail: tests and harness assertions that would pass on broken code, including assertions aimed at strings the code never emits.
</lenses>

<each_run>
1. Orient. Create the run id and the worktrees, and start the run record. In your worktree, read VISION.md, AGENTS.md, the handoff in `docs/project/CURRENT.md`, the approved plans and TODO.md. In the loop state, read the ledger, the scoreboard, the bookmarks and my answers. Read the git log since each bookmark, and check upstream releases since theirs. Before changing an area, read its sections of `sharp_edges.md`. Read the postmortems before touching providers.
   This prompt names many paths, commands and behaviours, and the refactor keeps moving them. Check each one against the base commit before relying on it. Where one no longer holds, follow the code and AGENTS.md, and list the stale line and its fix under "Needs from me".
   If the loop state is empty, this is the first run. Build only the scenarios the first ideas need, record the base commit, and get to a first kept change in the same run. Grow the harness and the scoreboard as later ideas need them. `docs/prompts/heylook_optimizer_loop.md` may already have started them.
2. Measure. Run the scenarios on the base commit and compare them with the last run's rows. Any difference is a lead. Before calling it a regression, confirm it by alternating the two commits in two worktrees. Then find the commit that caused it (bisect if needed) and put it under "Needs from me".
3. Find. Launch the reviewers in parallel, one per lens. Skip a lens whose bookmark shows nothing it covers has changed. Tell each one:
   - Read code and the web only, plus files in `internal/` last modified before the base commit's time (give them that time) and the loop state, and nothing in today's `internal/log/`. Don't edit files, build, run tests or benchmarks, start servers, or load models. Those compete for the GPU and unified memory and make timings meaningless.
   - For each idea, cite the file and line, say how it works, estimate the gain or the lines removed, and name the risk. Mark each claim [measured], [source] or [reported]. Where a claim needs data you don't have, write "verify:" and name it instead of guessing.
   - Return at most 5 ideas, ranked by expected value. In breakthrough mode, lead with the single biggest change you would bet on.
   Before you add an idea to the ledger, read the code the reviewer cites to check its claim, and check whether the ledger already rejected it.
4. Improve.
   - Breakthrough: choose one bet, the candidate with the largest expected effect in the direction VISION.md's "Where it is going" points, weighed against its risk and merge cost. Before you start, write in the run record what result, by what point, would make you drop it. Prototype the riskiest part first. If the bet dies, log why with the evidence (a well-measured negative result is still a result) and take the next candidate.
   - Steady: take silent fallbacks and other correctness problems first, then the rest by expected value.
   When an idea is promising but large, try it instead of logging it for later. For each change:
   a. Make sure a check covers the behaviour you're about to change. If none does, write one under AGENTS.md's test rules. Read the constant or string each assertion targets, so the check tests what the code actually emits. Judging whether a check can fail is the step-5 reviewers' job, not yours.
   b. For a structural change, sketch the before and after first.
   c. Change the code, iterating against exact counts where you can.
   d. Run the unit and contract suites, whatever AGENTS.md's Tests section names for the area you touched (smoke arms, the eval bank, chain_probe, e2e), and the relevant scenarios on every arm, alternating with `base`.
   e. Keep the change only if it improves the scoreboard, removes code, or fixes a correctness or visibility problem; regresses nothing beyond tolerance; keeps the engines at parity or narrows a gap; and is worth what it adds. Otherwise revert it and log the result.
   f. If the change sets up something that could quietly decay (a count, a parity, a rule), add the check that enforces it: a test, or a guard or hook script on your branch. Propose its wiring under "Needs from me" instead of installing it.
   g. Commit, staging files explicitly, with the wiki, `sharp_edges.md` and the frontend spec's §4 updated in the same commit wherever AGENTS.md requires it. Then append to the ledger and the scoreboard, and put the updated table in the same message as your next action.
5. Review. Relaunch reviewers on the run's diff. Ask them for:
   a. whether each change does what its ledger entry says,
   b. only the problems they would block the merge for, each with the file and line, why it's wrong, and how to show it fails, and
   c. new backlog items.
   Fix whatever blocks the merge, and log the rest.
6. Verify. On the branch's final commit, re-run the scenarios behind every headline result, from the worktree's own code. Later commits can undo earlier wins, so report only what holds.
7. Report. Build the page, finish the run record, append your session-log section, and tidy the ledger if <state> allows it.
8. Grade. Give one reviewer this prompt's "done" list, the run record and the report. Ask only for what is unmet, plus any change that pulls against VISION.md's principles or its "What it is not" list. If something is unmet, fix it, then repeat from step 6.
</each_run>

<scope>
- VISION.md's "What it is not" list rules out multi-user features, local model-architecture code (draft an upstream contribution instead), a second client protocol, and compatibility kept for its own sake. Removing old settings, shims and data formats is welcome, and AGENTS.md's "never write migration code" rule applies.
- Before deleting something that looks unused, say what it does and why nothing heylook is for needs it. Keep what earns its place.
- A change to shared behaviour lands on both engines in the same commit, or the gap goes in TODO.md.
- A new measurement, report field or check that makes something visible counts as progress. Log it like any other change.
- Verify that a library is actually broken before working around it.
</scope>

<final_report>
The page and your last message start with the same headings, in this order:
- Needs from me: decisions, approvals, the merge, and upstream drafts waiting on me, each with the picture that decides it and your recommendation.
- Scoreboard: charts of this run against the base commit and, as leads, against the first recorded run, per arm.
- Changed: what was kept and why it's better, with before-and-after pictures where they help.
- Removed: code, settings and dependencies deleted, with net lines on each side of the engine line.
- Tried and rejected: each with its measured result. In breakthrough mode, include the bets that died and what ended them.
- Choices made alone: decisions you made without me, so I can overrule them.
- Not confirmed: arms, suites or models you couldn't run, claims you couldn't verify, and where you looked.
- For the merge: the proposed CHANGELOG entry and CURRENT.md verification rows.
- Next: the top of the ledger.
Keep your last message short: the "Needs from me" items and where the page is.
</final_report>
