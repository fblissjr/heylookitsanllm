Make this project as fast as it can go: its core library and every public interface or binding (Python, Node, C ABI, etc.), without making correctness, output quality, or safety worse.

Done means: every primary benchmark is at least 1.2× faster than the True Performance Baseline, nothing regresses beyond the tolerances below, all tests pass, and a final review round finds no significant untried ideas. The 1.2× target is a floor, not the finish line. Keep going while ideas are still paying off.

<parameters>
- Target: every primary benchmark at least 1.2× faster than baseline.
- Speed regressions: no benchmark, including small-input and edge-case sizes, more than 3% slower beyond measured noise.
- Quality regressions: no quality or accuracy metric more than 0.1% (relative) worse. Where the API promises deterministic output, the output stays the same.
- Memory: flag any peak-memory increase over 10%.
- Safety: no memory-unsafe code (Rust `unsafe`, unchecked pointer arithmetic in C/C++, ctypes or buffer tricks in Python).
- Reviewers: 7–12 read-only subagents.
- Converged: two rounds in a row that together gain less than 2%.
</parameters>

<context>
This codebase is already heavily optimized, so the easy wins are mostly gone. Reaching the target will probably take structural changes: better algorithmic complexity, different data layouts, fewer copies across the binding boundary, parallelism, vectorization, or new algorithms built for this specific problem. You're encouraged to make them. When an idea is promising but large, try it instead of logging it as future work. Measurements decide what stays.
</context>

<how_to_run>
When a step doesn't need my input, keep going. Put status notes and results tables in the same message as your next action. Don't end your turn with a summary that announces the next step, an offer to continue, or a list of decisions that don't block the work. Stop and ask only when you can't continue without me, or before anything destructive: deleting data, force-pushing, rewriting history, or changing anything outside this repository.

Keep `PERF_LOG.md` as the source of truth: the baseline, the backlog as a checklist, what was kept or rejected and why, and the latest results table. Older turns get summarized as your context fills, so resume from this file, not from memory.
</how_to_run>

<phase_1_baseline>
1. Read the README, CONTRIBUTING, CLAUDE.md/AGENTS.md, the build/CI config, and the git history of past performance work. Learn how to build, test, and benchmark each part, and what has already been tried.
2. Build the release configuration that users actually install. On the unmodified code, run the test suite and every benchmark (core, bindings, and competitors if the repo has them). Run each benchmark at least 5 times and record the median and spread. Treat any difference inside the noise as zero.
3. Record the True Performance Baseline in `PERF_LOG.md` with the commit hash, machine info, and exact commands. Show it to me as a Markdown table with both speed and quality metrics.
</phase_1_baseline>

<phase_2_hypotheses>
Launch the reviewer subagents in parallel. Each one is read-only and gets a different lens:
1. Algorithmic complexity and scaling at large n (look for O(n²) and worse)
2. Memory layout, allocation, and cache behavior
3. Parallelism and concurrency
4. SIMD/vectorization and branch behavior
5. The binding/FFI boundary: copies, conversions, GIL/lock handling, per-call overhead
6. I/O, parsing, and serialization
7. Build and compiler configuration for the shipped artifact
8. Bespoke algorithm design for this specific problem
9. Numerical approach: precision, and approximation within the quality tolerance
10. Security and robustness, including what the other ideas might break
11. API usability and ergonomics

Tell every reviewer:
- Don't run builds, tests, or benchmarks. Parallel runs compete for CPU and make the timings meaningless.
- For each idea, cite file and line, explain how the speedup works, and estimate the gain and the risk. Leave out generic advice.
- Return at most 5 ideas, ranked by expected gain × confidence ÷ effort.

When a reviewer reports back, read the code it cites and check its claim before you add the idea to the backlog.
</phase_2_hypotheses>

<phase_3_optimize>
Work through the backlog, highest expected value first. For each item:
1. Profile the path and confirm it's hot.
2. Make sure tests cover the path's current behavior. If they don't, add tests first.
3. Change the library code. To iterate faster than noisy wall-clock timings allow, use a deterministic measure where you can: instruction counts (Valgrind/cachegrind `Ir`, `perf stat`), allocation counts, or call counts. Before relying on one, show that driving it down also lowers wall-clock time, and drop any measure that doesn't track it.
4. Run the tests, then the benchmarks.
5. Keep the change only if it moves toward the target, stays within every tolerance, and is worth the code it adds. A small gain that needs a lot of hard-to-maintain code isn't worth keeping. Commit each kept change on its own, with the measured difference in the commit message. Otherwise revert it and log the result so the idea isn't retried.
6. Where a deterministic count backs the win, add a test that fails if the count rises above the new value, so later changes can't quietly undo it.
7. Update the table (absolute and relative to baseline) in `PERF_LOG.md` and in your next message, then move on.

If the numbers start to look inconsistent, re-run the baseline from a clean worktree of the baseline commit.
</phase_3_optimize>

<rules>
Benchmark integrity:
- Optimize only library code. You may add new benchmarks, tests, and instrumentation, but don't change existing benchmarks, their inputs, iteration counts, or harness config, and don't special-case benchmark inputs.
- Every benchmark iteration must be independent. Nothing may persist between iterations and make later ones faster: no process-global memo tables, no caches keyed on inputs, no lazily built indexes reused across calls. Caches scoped to a single call or object are fine.
- A build setting only counts if it applies to the artifact users install. Keep that artifact portable: prefer runtime CPU-feature detection over native-CPU targeting.

Dependencies:
- New general-purpose dependencies (SIMD, hashing, allocators, parallelism) are fine. Don't add or copy in a library that already implements this project's core algorithm. Write the core yourself.
- Size-adaptive strategies, meaning different code paths for small and large inputs, are encouraged.

If the target can't be reached without breaking one of these rules, stop and tell me what you found.
</rules>

<phase_4_review>
When the backlog is empty, relaunch the reviewers with the diff against the baseline commit and `PERF_LOG.md`. Ask each one to:
(a) check that the code correctly carries out the ideas that were adopted,
(b) list only problems they would block the merge for, each with file and line, why it's wrong, and how to show it fails, and
(c) suggest new ideas that aren't already in the log.
Check their evidence, then send new ideas back through phase 3. When the convergence condition is met, finish.
</phase_4_review>

<final_report>
End with these headings, in this order:
- Needs from me: decisions or approvals you're waiting on.
- Results: the final table against baseline.
- Changed: what was kept, and why it's faster.
- Tried and rejected: each with its measured result.
- Not confirmed: anything you couldn't verify, such as gains inside the noise or input sizes and platforms you didn't test, and where you looked.
- Next ideas: what's still worth trying.
</final_report>
