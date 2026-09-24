# AGENTS.md

<!-- Repo-specific operating guide, loaded every session. Global conventions live in
the user-level CLAUDE.md and still apply; don't duplicate them here. Rules that matter
only while editing one area live in .claude/rules/ (table below) and load when a
matching file is read. The why and the incident history behind every rule is
docs/architecture/sharp_edges.md. -->

Personal MLX inference server on Apple Silicon: a FastAPI backend (`src/heylook_llm/`)
with two providers, MLX (mlx-vlm, text and vision) and gguf (one llama-server
subprocess per loaded model), and a vanilla-JS frontend (`frontend/`, served at `/`).

## Where to look first

- [VISION.md](./VISION.md): what the project is for. It breaks ties when a design choice is unclear; it carries no status and no plan.
- [docs/wiki/](./docs/wiki/README.md): how the system works end to end. Start here to learn a subsystem.
- [docs/project/CURRENT.md](./docs/project/CURRENT.md) and [TODO.md](./docs/project/TODO.md): status and backlog. Read before starting.
- [docs/architecture/sharp_edges.md](./docs/architecture/sharp_edges.md): why each rule here and in `.claude/rules/` exists. Read an area's section before changing it, and the [postmortems](./docs/architecture/postmortems/) before touching providers.
- [docs/frontend_v3_spec.md](./docs/frontend_v3_spec.md) §4: the authoritative API contract.
- Everything else: [docs/README.md](./docs/README.md).
- `internal/` (logs, research, local notes), `models.toml` and `coderef/` are gitignored and never committed. `coderef/llama.cpp` is the checkout the running llama-server was built from (its `heylook-build.json` names the commit); read upstream tip from a separate clone.

## Rules by area

Each file below holds the mechanisms that bite in its area. Claude Code loads it when a matching file is read; other agents read it before editing there. These files are part of this guide: a reference to "AGENTS.md's rules" includes them, and decisions they mark as owner decision, owner call or owner rule are settled. Don't reopen one without a new reason, and take that reason to the owner.

| Area | Rule file |
|---|---|
| Provider and engine contract, thinking, sampling, template overrides | [.claude/rules/engine-contract.md](./.claude/rules/engine-contract.md) |
| gguf: llama-server templates, context, memory, spec decode, binary | [.claude/rules/gguf.md](./.claude/rules/gguf.md) |
| MLX engine: mlx-vlm, threads, caches, tokenizers, parsers | [.claude/rules/mlx.md](./.claude/rules/mlx.md) |
| Model registry, models.toml, config effect classes | [.claude/rules/registry-config.md](./.claude/rules/registry-config.md) |
| API routes, the Messages wire, DuckDB store, observability | [.claude/rules/api-and-store.md](./.claude/rules/api-and-store.md) |
| Frontend | [.claude/rules/frontend.md](./.claude/rules/frontend.md) |
| Test harnesses, mocks, e2e, smoke, eval | [.claude/rules/tests.md](./.claude/rules/tests.md) |
| pyproject, uv.lock, the mlx-vlm pin | [.claude/rules/dependencies.md](./.claude/rules/dependencies.md) |
| scripts/, apps/ | [.claude/rules/scripts-and-apps.md](./.claude/rules/scripts-and-apps.md) |

## Commands

- Set up: `uv sync` (dev tooling included). Serve: `heylookllm --log-level INFO` (API and UI on :8000). Isolated server for live checks: the `dev-server` skill (`scripts/dev_server.sh`).
- Backend suite: `uv run pytest tests/unit/ tests/contract/ -v`.
- Browser E2E (opt-in, unsandboxed): `cd tests/e2e && bun install && bun run e2e[:chat|:pages]`; `bun run e2e:render` needs no model and no server.
- Eval bank, against a running server: `uv run python tests/eval/run.py --server <url> --models <ids>`.
- Live smoke, against a running server: `tests/smoke/` (see [tests/README.md](./tests/README.md)).
- llama-server: `scripts/build_llama.py` builds it; nothing else does.

## Done means

- `tests/unit/` and `tests/contract/` are fully green (Metal-gated skips are fine). Any failure is a regression to investigate; there is no pre-existing-failure allowlist.
- The checks the Tests section names for what you touched have run, and you say which did not.
- Docs moved in the same commit: the matching wiki page when a subsystem's behaviour changed, spec §4 when an API contract changed, and `sharp_edges.md` when a new rule has a story behind it.
- `CHANGELOG.md` has the entry and `src/heylook_llm/__init__.py` `__version__` matches it (a pre-commit guard checks this).
- `internal/log/log_YYYY-MM-DD.md` is updated before the session ends.

## Tests

- There is no "make it fail first" rule here, and adding one back is a regression. Write the check, run the suite, move on. The author of a check shares its blind spot; an independent review catches a check that cannot fail.
- When a check asserts on text or a protocol, read the constant it is supposed to match. An assertion aimed at a string the code never emits is indistinguishable from a fixed bug.
- The dated audits in `docs/testing/` found checks that could not fail. Read them before concluding a suite covers something.
- Which check for which change:
  - templates, parsers, stop tokens, vision: the eval bank (unit tests cannot certify these);
  - any cache, KV or position change: `scripts/chain_probe.py` per model class;
  - the MLX vision path: `scripts/vlm_parity_probe.py`;
  - frontend: the E2E suites that cover the page, and `e2e:render` for the chat message list;
  - streaming or latency, live: the MoE `gemma-4-26B-A4B` (the dense 31B is slow enough to look like a delivery bug).
- Release standard (not a CI gate): a release touching provider, loader, template or lifecycle code runs `tests/smoke/` green on all three arms (mlx-text, mlx-vision, gguf) and names any uncovered arm in the changelog. It also runs `scripts/vendor_frontend.py --check` and names the answer, and names any unmet Phase 3 precondition (the standing one is thinking depth on both MLX arms).

## Repo rules

- Derive, never hand-copy. A hand-copied constant list is a defect with a delay. When you add a field to a cascade, grep for any tuple, set or frozenset that enumerates its siblings and derive from it (`X = SHARED_TUPLE + ("extra",)`).
- Repo rules are enforced, not reminded: a deterministic rule belongs in a pre-commit guard (`scripts/check_version_sync.py`), a unit test (`test_field_keyword_defaults.py`) or a Claude Code hook whose logic lives in `scripts/hooks/` and whose wiring is `.claude/settings.json`.
- Never write migration code (solo deploy, no data to preserve). Dropping, recreating or truncating a DuckDB store or config on a schema change is fine and preferred.
- Parallel sessions are normal: assume another session has uncommitted work. Stage files by name (never `git add -A`/`-u`), run `git status` before committing, and leave files you did not touch unstaged. After any scripted string-replace, check the edit landed.
- Where knowledge goes: mechanisms in this file and `.claude/rules/`; status (what is done, counts, "until X lands") in `docs/project/`; rationale and incident history in `sharp_edges.md`; how a subsystem works in `docs/wiki/`, which carries no figures. `internal/` is unversioned: copy a long-lived doc there to an `archive/` subdir before rewriting it destructively.
- `.claude/` is local by default. Tracking a new file there takes two edits: the `.gitignore` negation block and `ALLOWED_PATHS` in the local pre-commit hook.
- GPG signing needs the 1Password agent; on socket errors use `git -c commit.gpgsign=false commit`.

## Sandbox traps

- `ENV=x uv run ...` does not match the uv sandbox exemption, so it runs sandboxed with no Metal.
- Sandboxed `curl` cannot reach localhost; probe with `uv run python` and urllib.
- Never launch the server piped to `head` (SIGPIPE wedges it); redirect to a file.
- Sandboxed `find` can silently return nothing under `modelzoo/`; use `ls` or a `uv run python` glob.
- To verify a change is schema-neutral, export `app.openapi()` from a HEAD~1 worktree and byte-compare.
