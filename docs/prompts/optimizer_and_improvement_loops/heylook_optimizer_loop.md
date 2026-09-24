Find a breakthrough in heylook's speed in real use, in one session, and measure every step. VISION.md, the project's north star, points speed at real use: reusing what was already computed (multi-turn context, long system prompts, images) matters more than raw throughput. Use it for direction and to break ties, not as a spec.

Done means:
- the target below is met on every arm,
- nothing regresses beyond the tolerances,
- the checks AGENTS.md names for what you touched are green,
- every headline result was re-checked on the branch's final commit, and
- a final review finds no significant untried idea.

The target is a floor, not the finish line. Keep going while ideas are still paying off.

This prompt shares its machinery with `docs/prompts/optimizer_and_improvement_loops/heylook_improvement_loop.md`. That file is mine, so read it in the main checkout. Follow its <how_to_run>, <parallel_sessions>, <state>, <measurement>, <visuals> and <scope> sections, plus the Speed, Memory, Quality and Code lines of its parameters, exactly as written. Ignore the rest of it (the lenses, the run cycle and the final report). AGENTS.md's rules apply in full and win any conflict.

<parameters>
- Target: time to first token at least 1.2× faster than the base commit on the reuse scenarios (turn N of a conversation, a long system prompt reused across new conversations, text follow-ups after an image), on the mlx-text, mlx-vision and gguf arms. Past that, as fast as you can get it.
- Other sessions: mrblue is continuing the big refactor on main, started 2026-09-23.
- Run budget: about 6 hours. Record the start time and check `date` against it.
- Landing: a branch `perf/<run id>` from main's HEAD at the start (the base commit), in a git worktree under `.claude/worktrees/`, one commit per kept change. State each commit's evidence as a relationship plus a pointer to its scoreboard rows, with no figures. Don't merge or push.
- Reviewers: one read-only subagent per lens.
</parameters>

<context>
Assume the easy wins are gone. A breakthrough usually changes the shape of the work rather than tuning it: work that is reused instead of recomputed, work that is never needed, fewer copies across boundaries, or a new approach built for heylook's specific problem. Draw on what's new: techniques and upstream capabilities from the last year. Be bold by default. When a promising idea is large, prototype it instead of logging it for later. Measurements decide what stays, and a well-measured dead end is still a result.
</context>

<baseline>
1. Create the run id and the worktrees, and start the run record. In your worktree, read VISION.md, AGENTS.md, the handoff in `docs/project/CURRENT.md`, and the `sharp_edges.md` sections and postmortems for the areas you expect to touch. In the loop state, read the scoreboard and the ledger; ideas already rejected there aren't worth retrying blind. Check each path and command this prompt and the improvement loop name against the base commit, and list any that no longer hold under "Needs from me".
2. If the loop state's `harness/` has scenarios, use them. Otherwise build the smallest runner there that covers the target's scenarios, so later runs can reuse it.
3. Measure the base commit and append the rows, with their conditions, to the scoreboard.
4. Pick the exact counts you'll iterate on (cached and processed tokens from `CacheReport`, image encodes, and the like). Before relying on each one, show that moving it moves time to first token.
5. Show me the baseline as a chart and a table.
</baseline>

<hypotheses>
Launch the reviewers in parallel. Each one is read-only and gets one lens:
- What's new as of 2026. Search the web, and cite the source and date.
- Prefix and prompt-cache reuse on each engine.
- The vision feature cache and image turns.
- The request path outside the engines.
- Memory and residency.
- The gguf spawn and its flags.
- Engine parity in what gets reused and reported.
- mlx-vlm, mlx and llama.cpp capabilities heylook doesn't use yet. Check mlx-vlm's open pull requests first.

Give them the <parallel_sessions> limits, and tell each one:
- Read code and the web only. Don't build, run tests or benchmarks, start servers, or load models. Those compete for the GPU and unified memory and make timings meaningless.
- For each idea, cite the file and line, say how it works, estimate the gain and name the risk. Mark each claim [measured], [source] or [reported]. Where a claim needs data you don't have, write "verify:" instead of guessing.
- Return at most 5 ideas, leading with the single biggest change you would bet on.

Before an idea goes into the ledger, check its claim against the code and check whether the ledger already rejected it. Work the ideas in order of expected value.
</hypotheses>

<optimize>
Start with the biggest bet. Before you start it, write in the run record what result, by what point, would make you drop it. Then prototype its riskiest part first. If the bet dies, log why with the evidence and move to the next one. Take the quick wins between bets.

For each change:
1. Confirm the path is hot, through the server's own report (`CacheReport`, usage timing) or a profile.
2. Make sure a check covers the behaviour you're about to change. If none does, write one under AGENTS.md's test rules.
3. For a structural change, sketch the before and after first.
4. Change the code, iterating against the exact counts, then confirm with wall-clock time.
5. Run the checks AGENTS.md names for the area you touched, and the target's scenarios on every arm, alternating with `base`.
6. Keep the change only if it moves toward the target, stays within every tolerance, keeps the engines at parity or narrows a gap, and is worth the code it adds. A correctness fix you find along the way is kept too. Otherwise revert the change. Either way, append it to the ledger.
7. Where a count backs the win, add a test that fails if the count gets worse than the new value.
8. Append the rows to the scoreboard and redraw the chart. Look at the chart, then put the table and the chart in the same message as your next action, with a line on what to look at.
</optimize>

<finish>
When the backlog is empty, relaunch reviewers on the diff against `base`. Ask each one for:
a. whether each change does what it claims,
b. only the problems they would block the merge for, each with the file and line, why it's wrong, and how to show it fails, and
c. ideas the ledger doesn't already have.
Fix whatever blocks the merge, and send new ideas back through <optimize>.

Stop when the reviewers have nothing significant left and the last two rounds together gained less than 2%, or when the budget is spent. Re-run every headline scenario on the branch's final commit, and report only what holds. Finish the run record, append your section to the day's log in `internal/log/`, and tidy the ledger if <state> allows it. End with these headings, in this order:
- Needs from me: decisions, approvals and the merge, each with the picture that decides it and your recommendation.
- Results: charts and table against `base`, per arm.
- Changed: what was kept, and why it's faster.
- Tried and rejected: each with its measured result, including the bets that died and what ended them.
- Not confirmed: arms or models you couldn't run, claims you couldn't verify, and where you looked.
- For the merge: the proposed CHANGELOG entry and CURRENT.md verification rows.
- Next ideas: what's still worth trying.
</finish>
