# CLAUDE.md

<!-- Full agent coordination and documentation navigation: see AGENTS.md -->

This file provides guidance to Claude Code when working with code in this repository.

## Internal Documentation - READ FIRST

**IMPORTANT**: Before making changes to provider code, MLX integration, or VLM functionality, check the `internal/` directory for known issues and lessons learned:

**Key Rule**: Always use library high-level APIs instead of reimplementing with low-level functions:
- `mlx_vlm.generate.stream_generate` for VLM generation
- `mlx_vlm.prompt_utils.apply_chat_template` for VLM prompt formatting (handles image tokens)
- `mlx_lm.generate.stream_generate` for text-only generation

If something seems broken in a library, verify it's actually broken before implementing a workaround.

## Ongoing Work - CHECK FIRST

**ACTIVE DEVELOPMENT**: Check `internal/session/CURRENT.md` before making changes. It contains:
- Current work in progress
- Quick resume instructions
- What's been completed
- Blockers and context

For persistent cross-session tasks, see `internal/session/TODO.md`.

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

See `docs/SERVICE_SECURITY.md` for firewall configuration and security guidance.

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
- **Auto model selection**: If no model specified in request, uses loaded model or `default_model` from config

### API Compatibility
- **OpenAI**: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/batch/chat/completions`, `/v1/hidden_states`
- **Admin**: `/v1/admin/restart`, `/v1/admin/reload`
- **Batch**: `/v1/batch/chat/completions` - Batch text generation (2-4x throughput, text-only models)
- **Logprobs**: `logprobs: true` + `top_logprobs: N` in chat completions returns token probabilities
- **Streaming Usage**: `stream_options: {include_usage: true}` returns token counts in final streaming chunk
- **Thinking Mode**: Qwen3 `<think>` blocks parsed and returned as `message.thinking` (non-streaming) or `delta.thinking` (streaming)

### Provider System
- **BaseProvider**: Abstract interface for LLM providers
- **MLXProvider**: Text + vision models on Apple Silicon
- **LlamaCppProvider**: GGUF models via llama-cpp-python (legacy, thread-safe with mutex)
- **LlamaServerProvider**: GGUF models via llama-server subprocess (recommended, in development)
- **CoreMLSTTProvider**: Speech-to-text via CoreML

### Configuration
- `models.toml` defines all available models
- Fields: `id`, `provider` (mlx/llama_cpp/coreml_stt/mlx_stt), `enabled`, `config`
- Models load on-demand when requested via API
- See `models.toml.example` for full parameter documentation
- **MLX-specific config options**:
  - `enable_thinking`: Enable Qwen3 thinking mode with optimal sampler defaults
  - `default_hidden_layer`: Default layer for hidden states extraction (default: -2)
  - `default_max_length`: Default max sequence length for hidden states (default: 512)

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
- `docs/WINDOWS.md` - Complete Windows installation and reference
- `docs/FRONTEND_HANDOFF.md` - Complete API reference for frontend developers
- `docs/SERVICE_SECURITY.md` - Background service setup and security

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
4. Add provider option to `models.toml` configuration
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

## Search and Navigation

- When searching the codebase, use `ripgrep` (`rg`) for better performance
- For documentation navigation, see `AGENTS.md` and `internal/00_INDEX.md`
