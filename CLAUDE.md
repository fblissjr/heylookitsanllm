# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Internal Documentation - READ FIRST

**IMPORTANT**: Before making changes to provider code, MLX integration, or VLM functionality, check the `internal/` directory for known issues and lessons learned:

**Key Rule**: Always use library high-level APIs (mlx-vlm's `stream_generate`, mlx-lm's `stream_generate`) instead of reimplementing with low-level functions. If something seems broken in a library, verify it's actually broken before implementing a workaround.

## Ongoing Work - CHECK FIRST

**ACTIVE DEVELOPMENT**: If there is a `NEXT_STEPS.md` file in the root directory, **READ IT FIRST** before making any changes. It contains:
- Current work in progress
- What's been completed
- What needs to be done next
- Important notes and gotchas
- Clear continuation instructions

This ensures continuity across sessions and prevents duplicate work.

## Code Style Guidelines

- No emojis in code, display names, or documentation
- No hype language: avoid "Enhanced", "Advanced", "Optimized", "Ultimate" prefixes
- Use descriptive technical names that explain what the code does
- Follow PEP-8 with 120-character lines
- Use type hints throughout
- Import order: standard → third-party → local (separated by blank lines)

## Platform Support

- **macOS**: All backends (MLX, llama.cpp, CoreML STT)
- **Linux**: llama.cpp backend
- **Windows**: llama.cpp backend (CUDA, Vulkan, or CPU)

## Quick Commands

### Installation

```bash
# macOS/Linux (recommended - uses uv sync for proper dependency resolution)
./setup.sh

# Windows
.\setup.ps1

# Manual with uv sync (recommended)
uv sync                           # Base install
uv sync --extra mlx               # macOS only
uv sync --extra llama-cpp         # All platforms
uv sync --extra stt               # macOS only
uv sync --extra analytics         # DuckDB analytics

# Alternative: pip-style install (doesn't use lockfile)
uv pip install -e .[mlx,llama-cpp]
```

### Server

```bash
heylookllm --log-level DEBUG
heylookllm --port 8080
```

### Testing

```bash
cd tests && pytest -m unit
cd tests && pytest -m integration  # requires server
python tests/test_stt_integration.py
```

### Model Import

```bash
heylookllm import --folder ~/models --output models.toml
heylookllm import --hf-cache --profile fast
heylookllm import --folder ~/models --interactive  # Interactive configuration wizard
```

### Background Service (macOS/Linux)

```bash
# Install service (localhost only by default - secure)
heylookllm service install

# Install for LAN access (behind VPN)
heylookllm service install --host 0.0.0.0

# Manage service
heylookllm service status
heylookllm service start
heylookllm service stop
heylookllm service restart
heylookllm service uninstall

# Linux: system-wide service (requires sudo)
sudo heylookllm service install --host 0.0.0.0 --system-wide
```

See `guides/SERVICE_SECURITY.md` for firewall configuration and security guidance.

## Project Structure

```
src/heylook_llm/
├── server.py           # CLI entry point
├── api.py              # OpenAI-compatible endpoints
├── stt_api.py          # Speech-to-Text endpoints
├── router.py           # Model routing with LRU cache
├── config.py           # Pydantic models
├── service_manager.py  # Background service management (macOS/Linux)
└── providers/
    ├── mlx_provider.py         # Apple Silicon (macOS)
    ├── llama_cpp_provider.py   # GGUF models (all platforms)
    └── coreml_stt_provider.py  # STT (macOS)

services/                       # Service templates
├── heylookllm.service.template          # systemd template (Linux)
└── com.heylookllm.server.plist.template # launchd template (macOS)
```

## Key Concepts

### Model Routing
- LRU cache holds max 2 models in memory
- Automatic loading/unloading based on API requests
- Provider selection based on `models.toml` configuration

### API Compatibility
- **OpenAI**: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/batch/chat/completions`, `/v1/hidden_states`
- **Admin**: `/v1/admin/restart`, `/v1/admin/reload`
- **Batch**: `/v1/batch/chat/completions` - Batch text generation (2-4x throughput, text-only models)

### Provider System
- **BaseProvider**: Abstract interface for LLM providers
- **MLXProvider**: Text + vision models on Apple Silicon
- **LlamaCppProvider**: GGUF models via llama-cpp-python (legacy, thread-safe with mutex)
- **LlamaServerProvider**: GGUF models via llama-server subprocess (recommended, in development)
- **CoreMLSTTProvider**: Speech-to-text via CoreML

### Configuration
- `models.toml` (or `models.yaml` for backward compatibility) defines all available models
- Fields: `id`, `provider` (mlx/llama_cpp/llama_server/coreml_stt), `enabled`, `config`
- Models load on-demand when requested via API
- TOML is the preferred format; YAML is deprecated but still supported

## Error Handling

- Raise `HTTPException` for API errors
- Use structured logging (not bare `print()`)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Provide meaningful error messages to users

## Testing Strategy

**Test Organization:**
- `tests/unit/` - Fast isolated tests (3 tests)
- `tests/integration/` - Server-dependent tests (7 tests)
- `tests/fixtures/` - Shared test data
- `tests/archive/` - Old/superseded tests (58 archived)

**Running Tests:**
```bash
cd tests && pytest                # All tests
cd tests && pytest unit/          # Unit only
cd tests && pytest integration/   # Integration (requires server)
cd tests && pytest -m unit        # By marker
```

**Critical Test Gaps (High Priority):**
- MLX Provider unit tests (no coverage)
- CoreML STT Provider unit tests (no coverage)
- Error handling tests (minimal coverage)
- LRU cache edge cases

See `tests/README.md` for complete test documentation and coverage matrix.

## Documentation

### Primary Docs
- `README.md` - Quick start and overview
- `docs/WINDOWS_INSTALL.md` - Complete Windows setup guide
- `docs/WINDOWS_QUICK_REFERENCE.md` - Windows command reference
- `docs/CLIENT_INTEGRATION_GUIDE.md` - API integration guide

### API Documentation
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI schema: `http://localhost:8080/openapi.json`

### Internal Design Docs
See comprehensive internal documentation in `internal/` directory - use `internal/00_README.md` as navigation hub.

## Working with Dependencies

Dependencies are in `pyproject.toml` with optional groups:
- `[mlx]` - Apple Silicon models
- `[llama-cpp]` - GGUF models
- `[stt]` - Speech-to-text
- `[performance]` - orjson, xxhash, turbojpeg, uvloop
- `[analytics]` - DuckDB for metrics
- `[profile]` - py-spy, memray
- `[all]` - Everything

Use `uv sync --extra <group>` for installation (preferred), or `uv pip install -e .[group]` as alternative.

## Git Workflow

- Conventional commits: `feat(router):`, `fix(mlx):`, `chore(docs):`
- No emojis in commit messages
- User handles commits - don't create commits unless explicitly requested

## Common Tasks

### Adding a New Endpoint

1. Add route handler to `api.py` or `stt_api.py`
2. Define Pydantic request/response models in `config.py`
3. Add OpenAPI documentation (summary, description, examples)
4. Write unit tests in `tests/`
5. Update relevant documentation

### Adding Provider Support

1. Create provider in `src/heylook_llm/providers/`
2. Inherit from `BaseProvider` or create appropriate interface
3. Implement required methods (`load_model`, etc.)
4. Add provider option to `models.yaml` schema
5. Update router to handle new provider type
6. Document in CLAUDE.md and README.md

### Troubleshooting Models

```bash
# Check model loading
heylookllm --log-level DEBUG

# Test specific model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model-id","messages":[{"role":"user","content":"test"}]}'

# Check available models
curl http://localhost:8080/v1/models
```

## Platform-Specific Notes

### macOS
- MLX requires Apple Silicon (M1/M2/M3)
- CoreML STT uses Neural Engine
- scipy requires gfortran: `brew install gcc`

### Linux
- Only llama.cpp backend supported
- CUDA compilation requires CUDA Toolkit
- Use CMAKE_ARGS for GPU acceleration

### Windows
- Only llama.cpp backend supported
- Requires Visual Studio Build Tools for compilation
- CUDA (NVIDIA) or Vulkan (AMD/Intel) for GPU
- See `docs/WINDOWS_INSTALL.md` for complete setup
- Use PowerShell for setup: `.\setup.ps1`

## Performance Considerations

### Model Loading
- First request to a model: 2-30s load time (depends on size)
- Subsequent requests: instant (cached)
- LRU eviction when cache full (max 2 models)

### Vision Models
- Use `/v1/chat/completions/multipart` for raw images (57ms faster per image)
- Images processed in parallel
- Base64 encoding has overhead

### GGUF Models
- Thread-safe via mutex lock (serialized requests)
- `model.reset()` clears KV cache between requests
- Single model instance per model ID

## Security & Privacy

- No authentication by default (local deployment)
- Validate image URLs and sizes
- No secrets in code or config
- Use environment variables for sensitive config

## When to Push Back

Challenge the user when:
- Request would break existing functionality
- Better alternative exists
- Approach is known to fail
- Security/privacy concerns
- Platform incompatibility ignored

Propose alternatives and explain reasoning.
- When searching the codebase, always use `ripgrep` (`rg`) instead of `grep` for better performance and respecting `.gitignore`.

## Documentation Index

### User Documentation (docs/)

**Getting Started**
- `README.md` - Project overview and quick start
- `guides/SERVICE_SECURITY.md` - Background service setup and security (macOS/Linux)
- `docs/WINDOWS_INSTALL.md` - Complete Windows setup guide
- `docs/WINDOWS_QUICK_REFERENCE.md` - Windows command reference
- `docs/CLAUDE_APP_QUICK_REFERENCE.md` - Quick command reference

**API Integration**
- `docs/CLIENT_INTEGRATION_GUIDE.md` - Complete API integration guide
- `docs/CLIENT_DEVELOPER_GUIDE.md` - Developer guide for API consumers
- `docs/CLIENT_CAPABILITY_DETECTION.md` - Feature detection guide
- `docs/OPENAPI_DOCUMENTATION_UPDATE.md` - API documentation changelog

**API Endpoints & Features**
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- `docs/embeddings_api.md` - Embeddings API documentation
- `docs/heylookllm_embeddings_spec.md` - Embeddings specification
- `docs/BATCH_PROCESSING.md` - Batch processing reference

**Testing & Development**
- `docs/TESTING_GUIDE.md` - Testing guide for developers
- `docs/QUEUE_INTEGRATION.md` - Queue integration reference

**Archives**
- `docs/archive/` - 29 archived files (3+ months old, superseded by current docs)

### Internal Documentation (internal/)

**Navigation Hub**
- `internal/00_README.md` - Documentation index with question-based navigation

**Core Architecture** (Complete 01-07 sequence - start here)
- `internal/00_CODEBASE_STRUCTURE.md` - Complete module map
- `internal/01_architecture_overview.md` - System architecture
- `internal/02_provider_mlx.md` - MLX provider design
- `internal/03_provider_llama_cpp.md` - llama.cpp provider design
- `internal/04_provider_unification_strategy.md` - Provider abstraction
- `internal/05_API_ARCHITECTURE.md` - FastAPI and all endpoints
- `internal/06_ROUTER_SYSTEM.md` - Model caching and routing
- `internal/07_CONFIGURATION.md` - Configuration system

**Critical Lessons Learned** (READ BEFORE MAKING CHANGES)
- `internal/VLM_VISION_BUG_2025-11-17.md` - CRITICAL: Don't reimplement library functionality
- `internal/MLX_CODE_REVIEW_2025.md` - MLX compatibility review

**Features & Capabilities**
- `internal/QWEN3_VL_MLX_IMPLEMENTATION.md` - Vision model implementation
- `internal/BATCH_TEXT_IMPLEMENTATION_PLAN.md` - Batch text generation
- `internal/BATCH_VISION_IMPLEMENTATION_PLAN.md` - Batch vision processing
- `internal/DUAL_PATH.md` - Dual-path optimization design
- `internal/IMAGE_TRANSFER_OPTIMIZATION.md` - Image optimization

**Platform Support**
- `internal/WINDOWS_SUPPORT_DESIGN_AND_PLAN.md` - Windows implementation
- `internal/LLAMA_CPP_ROADMAP.md` - Cross-platform llama.cpp

**Performance & Optimization**
- `internal/PERFORMANCE_OPTIMIZATION_GUIDE.md` - Performance notes
- `internal/IMPLEMENTATION_COMPLETE_2025-11-14.md` - Implementation summary

**Analytics & Monitoring**
- `internal/HEYLOOK_ANALYTICS_TECH_SPEC.md` - Analytics technical spec
- `internal/HEYLOOK_ANALYTICS_TEST_GUIDE.md` - Analytics testing
- `internal/dependency_analysis_report.md` - Dependency analysis

**Project History**
- `internal/LOG.md` - Development timeline

**How to Use Internal Documentation**
- New to codebase? Start with `00_README.md` → `00_CODEBASE_STRUCTURE.md` → `01_architecture_overview.md`
- Adding an endpoint? Read `05_API_ARCHITECTURE.md`
- Working with models? Read `06_ROUTER_SYSTEM.md` and `07_CONFIGURATION.md`
- Touching VLM code? Read `VLM_VISION_BUG_2025-11-17.md` first
- Need specific info? Use question-based navigation in `00_README.md`

### Configuration Files
- `CLAUDE.md` - Claude AI assistant guidelines (this file)
- `models.toml` - Model configuration (TOML format, preferred)
- `models.yaml` - Legacy model configuration (deprecated, still supported)
- `models.toml.example` - Example TOML configuration with full documentation
- `pyproject.toml` - Python package configuration
- `NEXT_STEPS.md` - Ongoing work tracker (check if exists)

## Living Documentation Philosophy

This project follows living documentation principles:

**internal/ - Design Documentation (Always Current)**
- Architecture decisions and their rationale
- Provider implementation details
- Critical bug postmortems with lessons learned
- Performance optimization strategies
- Implementation plans and reviews

**docs/ - User-Facing Documentation**
- Installation and setup guides
- API integration guides
- Feature documentation
- Quick reference guides

**When to Update Documentation:**
- After completing features → Document in internal/, create user guide in docs/
- After fixing critical bugs → Write postmortem in internal/
- After design decisions → Document rationale in internal/
- After API changes → Update docs/ guides

**Archive Policy:**
- Content 3+ months old → Archive to docs/archive/ or tests/archive/
- internal/ does NOT archive - living docs stay current via updates
- Archive READMEs explain what was archived and why

See `internal/DOCUMENTATION_PRINCIPLES.md` for complete methodology.
