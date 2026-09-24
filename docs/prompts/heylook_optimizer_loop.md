Improve heylook, one run at a time. `VISION.md` is the spec: what heylook is for, what it is not, its principles, and where it is going. Every change should move the project toward it. That can mean faster real use, less code, a more truthful report, or a rule that now runs as a check.

A run is done when:
- anything that regressed since the last run is fixed or reported,
- every change you started is either committed with its evidence or reverted and logged,
- the ledger and scoreboard are up to date, and
- nothing worth doing is left for this run: two ideas in a row didn't pay off, or the run budget is spent.

The next run starts from the ledger, so leave it in a state a new session can pick up without this conversation.

<parameters>
- Focus: none. Pick work by expected value. (Or name an area, such as "multi-turn time to first token with images on MLX".)
- Run budget: about 4 hours. Record the start time in the ledger and check `date` against it.
- Landing: a branch `improve/<date>` off main, one commit per change, with the evidence in each commit message. Don't merge. A run counts as accepted once I merge it, so main's last scoreboard entry is the reference point.
- Speed: no scenario slower than on main beyond its measured noise, and never more than 3% slower.
- Memory: flag any peak-memory increase over 5%.
- Quality: with greedy decoding, outputs match main's, within the run-to-run variation you measure on unchanged code. A request gives the same output with and without cache reuse.
- Code: the codebase grows only when the new code buys a measured gain I would notice.
- Reviewers: one read-only subagent per lens below.
</parameters>

<how_to_run>
When a step doesn't need my input, keep going. Put status notes and tables in the same message as your next action. Don't end your turn with a summary that announces the next step, an offer to continue, or a list of decisions that don't block the work.

Stop and ask only when you can't continue without me, or before you:
- touch my running server, my conversations, or my settings (run your own instance on a free port with a scratch data directory),
- download or delete a model,
- merge, force-push, or rewrite history,
- post anything outside this machine, including upstream issues and pull requests (draft them in the ledger instead), or
- change anything outside this repository.
</how_to_run>

<ledger>
Keep these in `improve/`, or in the repo's existing equivalents if it has them:
- `LEDGER.md`: the backlog as a checklist, with each item's lens and expected value; what was tried, kept, or rejected, with the evidence; drafted upstream contributions; and one short entry per run.
- `scoreboard.jsonl`: one row per measurement, with its conditions: commit, scenario, engine, model and file hash, quant, engine versions, chip and RAM, macOS version, power and thermal state, sample count, median, and spread.
- Scenarios, stored as data. Add new ones freely. Don't change an existing scenario during a run, and retire one only with a logged reason.
</ledger>

<measurement>
Principle 3 applies to your own work: a measurement without its conditions is an anecdote.
- Measure through the real path: your own server instance, called over its API the way my clients call it. Internal microbenchmarks are for iterating. A win counts only when it shows up through the real path.
- Cover real use on both engines: a cold first turn, turn N of a multi-turn conversation, a long system prompt reused across new conversations, an image followed by several text follow-ups, and a long thinking decode. Track time to first token, prefill and decode speed, total latency, peak memory, and tokens reused. Use synthetic prompts, never my conversations.
- Use the server's own per-request report of cost and reuse as the main instrument. Its counts are exact, so one run gives a clean signal. Iterate against them, then confirm with wall-clock time. If the report can't show something you need, making it show that is an improvement in its own right.
- Every iteration starts from the state its scenario declares. A cold scenario restarts the server with empty caches. A warm scenario builds its own warm state inside the iteration. Nothing carries over between iterations.
- Macs drift as they heat up. Run on power and note the thermal state. Compare against main in a worktree by alternating runs, one server instance at a time. Do at least 5 runs each, and count any difference inside the spread as zero.
- Keep engine versions and model files fixed within a comparison. A version bump is its own change and gets measured on its own.
- Measure code with standard tools for the repo's language: lines on each side of the engine line (engine-neutral core, MLX, llama.cpp), duplication, dead code, and complexity.
- Counts are proxies. Never improve a number in a way that makes heylook worse to use or harder to read. Don't special-case scenario prompts, models, or sizes.
</measurement>

<lenses>
1. Reuse: where already-computed work (multi-turn context, long system prompts, images) is thrown away or recomputed.
2. Engine parity: differences between MLX and llama.cpp in the contract, cache behavior, or report.
3. Request path: overhead outside the engines, such as templating, tokenization, image preprocessing, streaming, and serialization.
4. Memory: unified-memory pressure, model load and unload, KV cache sizing, and copies.
5. Dead and duplicated code, and how thick the local layer is on each side of the engine line.
6. Derive, don't copy: hand-kept facts that could be read from model files, templates, or engine reports.
7. Visibility: silent fallbacks, settings that can't explain their value and source, and requests that can't say what they cost or reused.
8. Enforce, don't remind: rules in docs, comments, or CLAUDE.md that no check enforces.
9. Upstream: local code made unnecessary by newer releases of what heylook pins, and local fixes that belong upstream.
10. The one wire and its clients: contract conformance, errors, streaming, and phone and browser use.
11. Privacy and security: anything logged, stored, exposed on the network, or sent that I didn't turn on.
</lenses>

<each_run>
1. Orient. Read VISION.md, CLAUDE.md, the ledger, and the git log since the last run. Check upstream release notes for everything heylook pins. A release that makes local code unnecessary goes near the top of the backlog. If there's no ledger yet, this is the first run: build the scenarios and the scoreboard, and record main as the first accepted run before anything else.
2. Measure. Run the scoreboard on main and compare it with the last accepted run. If something regressed, find the commit that caused it (bisect if needed). Then fix it first, or log it under "Needs from me".
3. Find. Launch the reviewers in parallel, one per lens. Skip a lens if the ledger shows a recent review and nothing it covers has changed since. Tell each reviewer:
   - Read code and the web only. Don't edit files, run builds, tests, or benchmarks, or load models. Those compete for the GPU and unified memory and make timings meaningless.
   - For each idea, cite the file and line, explain how it works, estimate the gain or the lines removed, and name the risk. Leave out generic advice.
   - Return at most 5 ideas, ranked by expected value.
   Read the code each reviewer cites and check its claim before you add the idea to the backlog.
4. Improve. Take silent fallbacks and other correctness problems first, then the rest by expected value. For each item:
   a. Make sure tests cover the behavior you're about to change. If they don't, add them first.
   b. Change the code, iterating against exact counts where you can.
   c. Run the tests and the relevant scenarios on both engines.
   d. Keep the change only if it improves the scoreboard or removes code, regresses nothing beyond tolerance, keeps the engines at parity or narrows a gap, and is worth what it adds. Otherwise revert it and log the result.
   e. If the change sets up something that could quietly decay (a count, a parity, a rule), add the check that enforces it.
   f. Commit, update the ledger and scoreboard, and move on.
5. Review. Before finishing, relaunch reviewers on the run's diff. Ask them for (a) whether each change does what its ledger entry claims, (b) only the problems they would block the merge for, each with the file and line, why it's wrong, and how to show it fails, and (c) new backlog items. Fix whatever blocks the merge and log the rest.
</each_run>

<scope>
- The vision rules out multi-user features, local model-architecture code (draft an upstream contribution instead), a second client protocol, and compatibility kept only for its own sake. You may remove old settings, shims, and data formats that exist only for compatibility; say so in the commit. If a removal would make my existing conversations or settings unreadable, ask first.
- Before deleting something that looks unused, say what it does and why nothing in the vision needs it. Keep what earns its place.
- A change to shared behavior lands on both engines in the same commit. If it can't, log the gap in the ledger.
- A new measurement or check that makes something visible counts as progress. Log it like any other change.
- When an idea is promising but large, try it on the branch instead of logging it for later.
</scope>

<final_report>
End each run with these headings, in this order:
- Needs from me: decisions, approvals, merges, and upstream drafts waiting on me.
- Scoreboard: this run compared with main and with the first recorded run.
- Changed: what was kept, and why it's better.
- Removed: code, settings, and dependencies deleted, with the net line count.
- Tried and rejected: each with its measured result.
- Not confirmed: anything you couldn't verify, and where you looked.
- Next: the top of the backlog for the next run.
</final_report>
