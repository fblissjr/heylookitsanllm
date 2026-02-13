# AGENTS.md

Quick navigation for Claude Code and specialized agents working with heylookitsanllm.

## Essentials

- [CLAUDE.md](./CLAUDE.md) - Nav hub: architecture links, code style, agent rules
- [internal/](./internal/) - Organized docs: backend/, frontend/, bugs/, research/, session/, log/

## Session Continuity

- [Current Work](./internal/session/CURRENT.md) - Active tasks and quick resume instructions
- [Persistent TODOs](./internal/session/TODO.md) - Cross-session task backlog

## Specialized Agents

When to use each agent:

| Agent | Use When |
|-------|----------|
| **windows-platform-architect** | Windows support, cross-platform compatibility, llama.cpp builds, CUDA/Vulkan configuration |
| **security-privacy-auditor** | Security review, dependency audit, privacy concerns, OWASP checks |
| **internal-docs-coordinator** | After completing features - update internal docs and maintain living documentation |
| **ui-ux-designer** | Frontend/React work, UI design, responsive layouts (for companion apps) |

## Key Internal Docs

| Topic | Document |
|-------|----------|
| Architecture overview | `internal/backend/architecture.md` |
| MLX provider | `internal/backend/providers/mlx.md` |
| API endpoints | `internal/backend/api.md` |
| Model routing | `internal/backend/router.md` |
| Configuration | `internal/backend/config.md` |
| **VLM bug (READ FIRST)** | `internal/bugs/vlm_vision_bug.md` |

### Critical Lessons Learned

Before modifying MLX/VLM code, read:
- `internal/bugs/vlm_vision_bug.md` - Don't reimplement library APIs
- `internal/backend/mlx_review.md` - API compatibility notes

## Platform Support

| Platform | Backend | Notes |
|----------|---------|-------|
| macOS | MLX, llama.cpp, MLX STT | Full support |
| Linux | llama.cpp | CUDA or CPU |
| Windows | llama.cpp | CUDA or Vulkan |

---

*This file provides quick navigation. For detailed guidance, see [CLAUDE.md](./CLAUDE.md).*
