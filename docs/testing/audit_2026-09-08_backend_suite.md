# Backend suite: runtime and mutation audit

**A dated record, not a living document.** Measured on 2026-09-08 against
`fb1bb8c..90a04f2`, on `uv run pytest tests/unit/ tests/contract/`.

Most of the evidence below was gathered by a subagent. Every claim is marked
**[verified here]** (re-derived independently in the main session) or
**[reported]** (the subagent's instrumented measurement, not re-run). Treat the
second kind as a lead, not a fact.

## 1. Runtime: the profiler could not see where the time went

**[verified here]** One test left `_active_generations = 3` and never restored
it. At garbage collection `BaseProvider.__del__` calls `unload()`, whose drain
loop polls `time.sleep(0.1)` until the active count reaches zero or a 30s cap
expires. Nothing was generating, so it burned the cap on every run.

| | before | after |
|---|---:|---:|
| `tests/unit/test_mlx_provider.py` alone | 33.51s | 3.29s |
| CPU during that run | 10% | 90% |
| full suite | 64.56s | 35.20s |

**[reported]** Instrumented at the call site, effectively all of the sleeping
came from one `__del__` entry in that file; the rest slept not at all.

The transferable part is why nobody found it from the suite's own output:
**`--durations` cannot see one second of it.** The stall happens inside
`__del__` during GC, outside every phase pytest times. That file's duration entries summed to a small fraction of its own wall time,
and the slowest test pytest would name ran about a second. **When a suite's
reported times do not add up to its wall clock, the gap is the finding.**

Fixed in `fb1bb8c`.

## 2. Verdicts

**[reported]** Almost every valid mutation killed something. Verified
load-bearing by mutation: the retired-field wire guard, reasoning-parser
invariants and the shared strip holdback, MODEL_BUSY propagation (both the AST
guard and the behavioural pair), the frontend mount shape, stop-reason
single-writer, vlm-inputs media attribution, observability level gating, and
the config `effect` metadata. No scar tissue was found; no test file imports a
deleted module.

**[verified here]** These oracles were defective and are now repaired
(`90a04f2`):

- `test_all_routes_have_schema_entries` walked `app.routes`, where most entries
  are `_IncludedRouter` objects with no `.path`. It saw a single `/v1` path
  while the schema published the whole surface, so its assertion body was
  reachable for one route. Planting a route with `include_in_schema=False` — the
  defect it names — left it green. Replaced with a walk that recurses through
  `original_router` and compares both directions; red on the planted route and
  naming it.
- `test_endpoint_count` asserted a floor far below the number of operations
  actually published, so most of the API could be deleted under it. Removed.
- `test_every_stop_reason_write_goes_through_the_mapper` read only the text
  right of the `=`, so hoisting the mapped value into a variable failed a
  passing test **on correct code**. Rewritten on AST. Verified green on the
  hoisted form and red on a raw provider value.

One correction worth recording, because it was nearly acted on: the deletion of
the first was justified by "the banner test subsumes it". **It does not.**
Planting the hidden route left `test_startup_banner.py` green too, because
`server.get_api_endpoints` reads `app.openapi()` — comparing it to the schema
compares the schema to itself for that case. The banner test is still
load-bearing for *its own* claim — reverting `get_api_endpoints` to the old
walk turns its tests red — it simply never claimed to catch a hidden route.
Delete became replace on the strength of one experiment.

## 3. Envelope: what this suite structurally cannot see

- **[verified here] Duration-based triage is blind to teardown and GC work.**
  §1 is the worked example. Defects of the kind "a `__del__` or teardown path
  started blocking" are unreachable by `--durations`.
- **[verified here] The unit `mock_mlx` fixture patches unconditionally.**
  `tests/contract/conftest.py` skips the patch when real MLX imports;
  `tests/conftest.py` has no such guard, and real MLX *is* importable on this
  machine. So defects of the kind "the real mlx / mlx-lm / mlx-vlm API changed
  shape" are unreachable in the unit suite on the one machine where they would
  otherwise be reachable. **This is not a one-line fix** — the contract suite
  only ever needed the mocks so imports would succeed, while unit tests use
  the mock objects' behaviour, so removing the patch would break tests rather
  than reveal drift.
- **[verified here] The real-library surface test pinned `mlx_provider.py`'s
  call site but not `vlm_inputs.py`'s**, which is where per-message media
  attribution has lived since v2.0.18 — so the file that decides how images
  are attributed was unwatched against the library surface this module tracks.
  **Closed** in `c8a4857`.
- **[reported] The contract suite has no per-test isolation.** `app`, `client`
  and `MockRouter` are session-scoped and `MockRouter.providers` accumulates
  with no reset. Verified by the subagent as not currently biting — all
  contract files pass alone — but structurally reachable.
- **[reported] Source-text oracles constrain spelling, not behaviour.**
  A number of files assert on `read_text()` / `inspect.getsource` / AST. The
  false-positive direction is now proven, not theoretical (§2). Others were
  not audited.
- **[verified here] No coverage tooling and no xdist** are installed. There is
  no in-repo mechanism to find untested lines; this audit bisected by hand.
  Parallelism is also bounded below by the slowest single file, so it was not
  the lever it looked like even before §1 landed.

## 4. Still open

Ranked by what the finding is worth, not by effort.

1. ~~**`BaseProvider.__del__` can block.**~~ **Closed** in `c8a4857`.
   `unload(drain=...)` is now the caller stating whether it can afford to
   wait; `__del__` passes `drain=False`, which leaves live work alone and
   warns rather than tearing down mid-decode. Pinned by two checks (collection
   with traffic returns at once and stays loaded; collection with nothing in
   flight still unloads), verified red against the old destructor.
2. **Most of the suite was never probed.** The largest unprobed files are `test_rlm.py`, `test_llama_server_provider.py`,
   `test_conversation_generate.py` and `test_template_info.py`. The kill rate
   above applies to the groups there was reason to suspect, not to the suite.
3. **`test_mlx_provider.py` is now spot-verified, not fully probed.** It could
   not be mutated during the original pass — another session held
   `mlx_provider.py` and `base.py` dirty throughout. Re-run once the tree was
   clean (2026-09-08, `c8a4857`), mutating the thread-local generation stream,
   image detection, VLM strategy compilation and diffusion detection. **Every
   one killed a test**, so the file's routing claims are load-bearing rather
   than decorative. This is a spot check, not coverage: the load path, the
   prompt cache and the sampler cascade in that file remain unprobed.
4. ~~**Extend `test_mlxvlm_surface.py` to `vlm_inputs.py`.**~~ **Closed** in
   `c8a4857`; verified red against a renamed kwarg. Note the first version of
   that pin asserted a kwarg the call does not pass and failed on its first
   run — the useful kind of red, and the reason to run a new pin before
   trusting it.
5. **[reported] `model_registry.discover` sleeps** account for a few seconds
   of what remains. Worth a look only after (1).
6. **[reported] Stale pytest temp garbage** produces warnings on every run,
   from an orphaned directory belonging to a test that no longer exists. A
   one-time removal under the user's temp directory, not a code change.

## 5. What was NOT done

No test was deleted on reasoning alone. Every deletion and rewrite in §2 was
preceded by planting the defect it claimed to catch and watching what happened.
The one case where that discipline changed the outcome is recorded in §2, and
is the reason to keep doing it.
