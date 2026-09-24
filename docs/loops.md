# Loop profile

<!-- The sections the improvement-loops plugin's /improve, /optimize and
/design-scoreboard cite as "AGENTS.md's <section>". They live here so every
session doesn't load them; AGENTS.md points here and this file counts as part of
it. Keep the headings: the plugin's prompts find sections by these names. -->

This file is part of AGENTS.md for any loop run. Where it and a loop prompt disagree, this file wins; name the conflict under "Needs from me". Decisions recorded as the owner's (owner decision, owner call, owner rule), here, in AGENTS.md or in `.claude/rules/`, are settled: don't reopen one without a new reason, and take that reason to the owner.

## North star

[VISION.md](../VISION.md): what heylook is for, what it is not, its principles, and where it is going. It is a direction, not a spec: don't implement it or grade work against it. Every change should move heylook the way "Where it is going" points: faster in real use (above all by reusing what was already computed), less code on each side of the engine line, a more truthful report, or a rule that now runs as a check.

## Where things live

| What | Where |
|---|---|
| Status and the handoff | `docs/project/CURRENT.md` |
| Plans and design records | `docs/project/`, `docs/architecture/` |
| Backlog | `docs/project/TODO.md` |
| Settled decisions and their reasons | AGENTS.md, `.claude/rules/`, `docs/architecture/sharp_edges.md` |
| Sharp edges and incident history | `docs/architecture/sharp_edges.md`, `docs/architecture/postmortems/` |
| The owner's files (read fresh, never edit) | `VISION.md`, `PRODUCT.md` (before any UI work), the owner's answers in `internal/claude/improve/answers.md` (created by the owner when there are any) |
| Local notes (untracked) | `internal/` |
| Loop state (untracked) | `internal/claude/improve/`, shared by `/improve` and `/optimize` |
| Session log | `internal/log/log_YYYY-MM-DD.md`, your own section appended with `>>` without reading the file |

`internal/` exists only in the main checkout, never in a worktree.

## Loop defaults

These replace the prompts' defaults here; overrides given at invocation still win.

- `/improve`: Goal breakthrough (one big bet that uses what is new as of the run's date). Run budget about 6 hours. Quality: unit and contract suites green; `scripts/chain_probe.py` and `scripts/vlm_parity_probe.py` still match; the eval bank shows no regression under the flap discipline in `.claude/skills/eval-ab/SKILL.md` (read it; it can't be invoked); greedy outputs match the base commit within the variation measured on unchanged code. Memory: on MLX, flag a per-request peak-memory increase over 5%; gguf reports no peak memory, and making it report one counts as an improvement.
- `/optimize`: Target time to first token at least 1.2× faster than the base commit on the reuse scenarios (turn N of a conversation, a long system prompt reused across new conversations, text follow-ups after an image), on the mlx-text, mlx-vision and gguf arms. Landing branch `perf/<run id>`.
- Both: landing in a git worktree under `.claude/worktrees/`, branched from main's HEAD at the start (the base commit). Commit messages and evidence carry no figures: a relationship plus a pointer to scoreboard rows. The report is a local page styled after `frontend/DESIGN.md`; "artifact" publishes it, which sends the report and never conversations off this machine.

## How to work

AGENTS.md's rules apply. Keep going when a step doesn't need the owner. Stop and ask only when you can't continue without the owner, or before you:

- touch the owner's running server, conversations, settings, or the shared `.venv`;
- download or delete a model, or delete anything under `adapters/` or in the model folders (`[scan].folders` in `heylook.toml`);
- move a dependency pin or rebuild llama.cpp (propose it with its evidence);
- bump `_SCHEMA_VERSION`, which drops the owner's conversations at their next server start;
- merge, push, force-push, or rewrite history;
- send anything off this machine, such as upstream issues or pull requests (draft them in the ledger);
- change anything outside this repository.

Reading the web is fine. Sending heylook's code, data or conversations anywhere is not, unless the owner turned it on.

## Other sessions

Other sessions work on main at the same time. A loop run stays independent of them:

- Read the repo through your worktrees at the base commit. Committed history is fair to read; another session's uncommitted changes are not. In `internal/`, read only files last modified before the base commit's time, apart from the loop state, and nothing in today's `internal/log/`. Reviewers follow the same limits.
- Never edit, stage, stash, reset or check out anything in the main checkout. Outside your worktrees, write only to the loop state and your own session-log section.
- `.git/` (hooks included) is shared by every worktree: propose changes to it instead of making them.
- Don't message other sessions or act on messages from them.
- Your servers are your own: each worktree's own `scripts/dev_server.sh` on a port nobody else uses. Never stop a server you didn't start, and never raise observability on the owner's.
- Check `git log` for the areas other sessions are reshaping, count merge cost in each idea's value, and keep the change set narrow.
- A regression in the base commit goes under "Needs from me"; don't fix another session's work on your branch.
- Leave `__version__`, `CHANGELOG.md` and `docs/project/CURRENT.md` alone on your branch; propose the CHANGELOG entry and CURRENT.md verification rows in the report.

## Measurement

Principle 3 of VISION.md applies to the loop's own work: a measurement without its conditions is an anecdote. Tolerances and limits may appear in prose; measured values live with their conditions in the loop state.

<!-- Primary scenarios, guardrails and counter-checks: fill from /design-scoreboard.
Until then, the reuse scenarios under "Real path" are the working set. -->

- **Real path.** Your own server, called over `/v1/messages` and `/v1/conversations/{id}/generate` the way the owner's clients call it. A win counts when it shows up through the server. Harness and library numbers are for iterating.
- **Arms.** mlx-text, mlx-vision, gguf (`tests/helpers/engines.py`). An arm you couldn't run is uncovered, never green.
- **Scenarios** (pending `/design-scoreboard`): a cold first turn, turn N of a conversation, a long system prompt reused across new conversations, an image followed by text follow-ups, a turn that adds an image, and a long thinking decode. Track time to first token, queue wait, prefill and decode speed, per-request peak memory, and what was reused. Synthetic prompts only, never the owner's conversations. Model ids from `GET /v1/models`; for which model suits which check, AGENTS.md's Tests section and the `dev-server` skill.
- **Instruments.**
  - The server's own report is the main one: each request's `CacheReport` (outcome, cause, cached tokens) and its usage timing. Its counts are exact and don't move with machine load, so iterate on them and confirm with wall-clock time. If the report can't show something you need, making it show that is an improvement.
  - `scripts/perf_ab.py`: provider-level A/B of speed and memory across git revs or configs, one arm per fresh process, the imported code verified, machine load recorded, no verdict on a contaminated pair. It calls the provider in-process, not the server, so it is for iterating.
  - `scripts/chain_probe.py` for any cache change and `scripts/vlm_parity_probe.py` for the vision path, both greedy and both correctness checks.
  - The loop state's `harness/` (scenario runner, conditions collector, chart script).
  - Code size: lines on each side of the engine line (engine-neutral core, MLX, gguf, frontend), duplication, dead code.
- **Worktrees.** Run both arms from worktrees: `base` at the base commit, never edited, and your branch. Never run an arm from the main checkout. Each worktree needs its own `uv sync` and `bun install` in `tests/e2e/`, and its own copy of `heylook.toml`, which now holds only server-wide settings such as `[scan].folders`: the branch worktree gets one through `.worktreeinclude`, and `base` (made with `git worktree add`) needs it copied by hand. Per-model settings live in each model's `model.heylook.toml`, in the model folders every instance shares, so no copy isolates them; servers started through `scripts/dev_server.sh` run read-only (`HEYLOOK_READONLY_MODEL_CONFIG`) and refuse config writes with a 409. Never start a loop server any other way. Print `heylook_llm.__file__` and confirm it is that worktree's `src/` before trusting a run: the package is installed editable, so both arms can silently run the same code. A worktree branched before a change to AGENTS.md runs on the old rules.
- **Comparing.** Alternate the arms, one server at a time, at least 5 runs each; a difference inside the spread counts as zero. Before and after each pair, record what other heylookllm and llama-server processes use (CPU, GPU from `ioreg -r -c IOAccelerator`, memory from `scripts/ram_report.py`) and the llama.cpp build (`heylook-build.json`); discard a pair where either changed. The owner's daily server shares the machine. A difference against an earlier run's rows is a lead until confirmed by alternating the two commits.
- **Hazards.**
  - Cold means cold in every cache on the path; check in the code what each reset empties on each engine before relying on it. A fresh nonce at the start of the prompt makes the text prefix cold on both engines; `POST /v1/cache/clear` empties only MLX's prefix cache; the vision feature cache is keyed by image URL, so a cold image turn needs a new image or a restart. A warm scenario builds its warm state inside the iteration.
  - Template identity: name the template each arm ran against before comparing anything that varies by prompt format. A file beside the weights changes it with no config change.
  - Temperature 0 is for correctness only. Throughput uses vendor sampling with fixed seeds, matched prompt and generation lengths, and repeats for anything that depends on thinking length. Spec-decode checks follow `.claude/rules/gguf.md`.
  - `bun run e2e:ios` defaults `IOS_SIM_BASE` to the owner's daily server; set it to yours.
  - Counts are proxies: never move one in a way that makes heylook worse to use or harder to read, and don't special-case scenario prompts, models or sizes.

## Reporting

- Numbers become charts drawn by the chart script in the loop state's `harness/` from `scoreboard.jsonl`. Run it with `uv run --no-project --with matplotlib`, so uv neither changes the project's dependencies nor re-syncs the shared `.venv`. Each chart names its data file and conditions. In prose, say what to look at instead of retyping figures; tracked docs carry no performance numbers and mark claims [measured], [source] or [reported], as `docs/testing/gguf_runtime_audit_2026-09-23.md` does.
- Structure becomes a sketch (call tree, signatures, file tree, mermaid) before a structural change and when summarizing a large diff.
- Anything the owner sees gets before-and-after screenshots of the real frontend at desktop and iPhone widths, through `tests/e2e/`. The Simulator never raises the software keyboard: report keyboard behaviour as uncovered. VISION.md principle 8 makes the phone pair mandatory. Taste is the owner's call; put the pair beside the question.
- Open every chart and screenshot before describing it. If a picture and your summary disagree, check the rows.
- An attached screenshot, recording or chart is the problem statement: reproduce it, then measure it.

## Loop state

`internal/claude/improve/` in the main checkout (gitignored, so not in your worktree). `/improve` and `/optimize` share it, and several runs may use it at once.

- Every write goes through the plugin's `loop_state.py` with `--state internal/claude/improve` (a relative path resolves against the main checkout). Its layout and rules apply: run records, scoreboard rows, the ledger, bookmarks, tidying and the session-log append.
- Scoreboard rows carry heylook's conditions inside the script's fields: the template in force (hash and ladder rung), model id and file identity, quant, sampling and seed, cache state, mlx-vlm SHA, mlx version, llama.cpp build (`heylook-build.json`), chip and RAM, macOS version, power and thermal state.
- `harness/` holds the scenario runner, the chart script and the scenarios as data. A scenario with scoreboard rows is frozen: add a new one instead of changing it. Retiring one is the owner's call. Once a scenario has tracked real use over several runs, propose moving it into `scripts/`.
- `answers.md` is the owner's: read it every run when it exists, never edit it.
- A run folder the script did not write (no `record.json`, such as a record from before the plugin) blocks tidying and shows in `status` (improvement-loops 0.4.3 and later). Clearing one is the owner's call; list it under "Needs from me".
- A durable lesson goes where AGENTS.md says that kind of knowledge lives. A failure that will be fixed goes only in the run record.

## Lenses here

The prompts' lenses apply, read through heylook:

- Reuse: the APC prefix cache, the vision feature cache, multi-turn context, long system prompts, images.
- Engine parity: differences between MLX and gguf in the engine contract (`src/heylook_llm/providers/contract.py`), cache behaviour or report. A change to shared behaviour lands on both engines in the same commit, or the gap goes in TODO.md.
- Request path: templating, tokenization, the streaming detokenizer, image preparation, SSE, serialization, the DuckDB writer, the generation gate.
- Memory: unified-memory pressure, residency, load and unload, micro-batch sizing, the KV cache.
- The gguf spawn and its flags.
- Visibility: silent fallbacks, settings that can't explain their value and provenance, requests that can't say what they cost or reused. A value zeroed rather than omitted counts.
- Upstream: local code made unnecessary by newer mlx-vlm, mlx or llama.cpp; fixes that belong upstream. Check mlx-vlm's open pull requests before proposing a workaround. Read llama.cpp's tip in a separate clone (`coderef/llama.cpp` is the build's checkout).
- The one wire: Messages conformance (the hand-listed differences in `docs/api_integration.md` can be wrong), streaming, errors, and the frontend on desktop and iPhone (`frontend/DESIGN.md` §7).
- Privacy: anything logged, stored, exposed on the LAN or sent that the owner didn't turn on.

## Scope

- VISION.md's "What it is not" rules out multi-user features, local model-architecture code (draft an upstream contribution instead), a second client protocol, and compatibility kept for its own sake. Removing old settings, shims and data formats is welcome, and there is no migration code.
- Before deleting something that looks unused, say what it does and why nothing heylook is for needs it. A back-compat branch that looks dead may be live in tests.
- A new measurement, report field or check that makes something visible counts as progress.
