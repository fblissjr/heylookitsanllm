# Backend architecture reference

Last updated: 2026-07-09

Git-tracked reference for how the heylookitsanllm backend works. Promoted out of
the local-only `internal/` tree on 2026-07-09 so it carries version history and
is readable on GitHub. Roadmap/status live in [../project/](../project/); the
frontend map is [../frontend_v3.md](../frontend_v3.md).

| Doc | Covers |
|-----|--------|
| [overview.md](./overview.md) | Backend architecture overview -- providers, request flow, module map |
| [api.md](./api.md) | Full API / endpoint reference (routers, request/response shapes) |
| [router.md](./router.md) | Model router: caching, LRU eviction, pinning, idle-unload, the FIFO generation gate |
| [config.md](./config.md) | `models.toml` config system + the sampler cascade |
| [mlx_provider.md](./mlx_provider.md) | MLXProvider deep-dive (text + vision); the pre-filled-cache VLM path |
| [mlx_embedding.md](./mlx_embedding.md) | MLXEmbeddingProvider (dynamic backbone via mlx-lm) |
| [ecosystem_strategy.md](./ecosystem_strategy.md) | MLX ecosystem posture -- **READ before perf / provider work** |

Still local-only in `internal/` (not refreshed for tracking): the bug
postmortems (`internal/bugs/`) and stale subsystem notes
(`internal/backend/{batch,logprobs,thinking}.md`). Links from these docs into
`internal/` resolve for maintainers with the full checkout; they dangle when
browsing on GitHub.
