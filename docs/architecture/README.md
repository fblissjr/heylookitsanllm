# Backend architecture reference

Last updated: 2026-07-09

Git-tracked reference for how the heylookitsanllm backend works. Promoted out of
the local-only `internal/` tree on 2026-07-09 so it carries version history and
is readable on GitHub. Roadmap/status live in [../project/](../project/); the
frontend map is [../frontend_v3.md](../frontend_v3.md).

| Doc | Covers |
|-----|--------|
| [config.md](./config.md) | `models.toml` config system + the sampler cascade |
| [mlx_provider.md](./mlx_provider.md) | MLXProvider deep-dive (text + vision); the pre-filled-cache VLM path |
| [ecosystem_strategy.md](./ecosystem_strategy.md) | MLX ecosystem posture -- **READ before perf / provider work** |

## Postmortems

Crash postmortems referenced throughout these docs (read before touching the
provider / cache layers):

| Doc | Covers |
|-----|--------|
| [postmortems/mlx_thread_teardown_abort.md](./postmortems/mlx_thread_teardown_abort.md) | SIGTRAP process abort on generation-thread teardown (v1.31.2) |
| [postmortems/radix_thread_affinity.md](./postmortems/radix_thread_affinity.md) | "no Stream(gpu, N)" crash on radix cache reuse + the v1.32.0 eligibility gate |

Still local-only in `internal/` (not refreshed for tracking): the stale
subsystem notes (`internal/backend/{batch,logprobs,thinking}.md`). A few prose
references into `internal/` from these docs resolve for maintainers with the
full checkout but dangle when browsing on GitHub.
