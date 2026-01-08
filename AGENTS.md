# AGENTS.md

Quick navigation for Claude Code and specialized agents working with heylookitsanllm.

## Essentials

- [CLAUDE.md](./CLAUDE.md) - Code style, commands, key concepts, platform notes
- [internal/00_INDEX.md](./internal/00_INDEX.md) - Full documentation navigation

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

## TDD Workflow

Use `/uv-tdd` skill for Python test-driven development:
- Creates project structure with tests
- Runs tests with `uv run pytest`
- Follows test-first methodology

## Documentation Updates

After completing significant work:
1. Invoke `internal-docs-coordinator` agent to update living docs
2. Update `internal/session/CURRENT.md` with progress
3. Move completed items to `internal/session/TODO.md` if applicable

## Quick Reference

### External Reference Notation

This project uses `[EXTERNAL:repo]` notation for external code references:
```
[EXTERNAL:mlx-lm] = https://github.com/ml-explore/mlx-lm
[EXTERNAL:mlx-vlm] = https://github.com/Blaizzy/mlx-vlm
```

### Key Internal Docs

| Topic | Document |
|-------|----------|
| Architecture overview | `internal/01_architecture_overview.md` |
| MLX provider | `internal/02_provider_mlx.md` |
| API endpoints | `internal/05_API_ARCHITECTURE.md` |
| Model routing | `internal/06_ROUTER_SYSTEM.md` |
| Configuration | `internal/07_CONFIGURATION.md` |
| **VLM bug (READ FIRST)** | `internal/VLM_VISION_BUG_2025-11-17.md` |

### Critical Lessons Learned

Before modifying MLX/VLM code, read:
- `internal/VLM_VISION_BUG_2025-11-17.md` - Don't reimplement library APIs
- `internal/MLX_CODE_REVIEW_2025.md` - API compatibility notes

## Platform Support

| Platform | Backend | Notes |
|----------|---------|-------|
| macOS | MLX, llama.cpp, CoreML STT | Full support |
| Linux | llama.cpp | CUDA or CPU |
| Windows | llama.cpp | CUDA or Vulkan |

## Project Commands

```bash
# Server
heylookllm --log-level DEBUG

# Testing
cd tests && pytest -m unit
cd tests && pytest -m integration  # requires server

# Model import
heylookllm import --folder ~/models --output models.toml
```

---

*This file provides quick navigation. For detailed guidance, see [CLAUDE.md](./CLAUDE.md).*
