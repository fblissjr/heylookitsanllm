# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Internal Documentation - READ FIRST

**IMPORTANT**: Before making changes to provider code, MLX integration, or VLM functionality, check the `internal/` directory for known issues and lessons learned:

**Key Rule**: Always use library high-level APIs (mlx-vlm's `stream_generate`, mlx-lm's `stream_generate`) instead of reimplementing with low-level functions. If something seems broken in a library, verify it's actually broken before implementing a workaround.

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
# macOS/Linux
./setup.sh

# Windows
.\setup.ps1

# Manual
uv pip install -e .
uv pip install -e .[mlx]          # macOS only
uv pip install -e .[llama-cpp]    # All platforms
uv pip install -e .[stt]          # macOS only
```

### Server

```bash
heylookllm --api openai --log-level DEBUG
heylookllm --api ollama --log-level DEBUG
heylookllm --api both --port 8080
```

### Testing

```bash
cd tests && pytest -m unit
cd tests && pytest -m integration  # requires server
python tests/test_stt_integration.py
```

### Model Import

```bash
heylookllm import --folder ~/models --output models.yaml
heylookllm import --hf-cache --profile fast
```

## Project Structure

```
src/heylook_llm/
├── server.py           # CLI entry point
├── api.py              # OpenAI/Ollama endpoints
├── stt_api.py          # Speech-to-Text endpoints
├── router.py           # Model routing with LRU cache
├── config.py           # Pydantic models
└── providers/
    ├── mlx_provider.py         # Apple Silicon (macOS)
    ├── llama_cpp_provider.py   # GGUF models (all platforms)
    └── coreml_stt_provider.py  # STT (macOS)
```

## Key Concepts

### Model Routing
- LRU cache holds max 2 models in memory
- Automatic loading/unloading based on API requests
- Provider selection based on `models.yaml` configuration

### API Compatibility
- **OpenAI**: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/batch/chat/completions`
- **Ollama**: `/api/tags`, `/api/chat`, `/api/generate`, `/api/embed`
- **Admin**: `/v1/admin/restart`, `/v1/admin/reload`
- **Batch**: `/v1/batch/chat/completions` - Batch text generation (2-4x throughput, text-only models)

### Provider System
- **BaseProvider**: Abstract interface for LLM providers
- **MLXProvider**: Text + vision models on Apple Silicon
- **LlamaCppProvider**: GGUF models via llama-cpp-python (thread-safe with mutex)
- **CoreMLSTTProvider**: Speech-to-text via CoreML

### Configuration
- `models.yaml` defines all available models
- Fields: `id`, `provider` (mlx/llama_cpp/coreml_stt), `enabled`, `config`
- Models load on-demand when requested via API

## Error Handling

- Raise `HTTPException` for API errors
- Use structured logging (not bare `print()`)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Provide meaningful error messages to users

## Testing Strategy

- Unit tests: `@pytest.mark.unit` - no server required
- Integration tests: `@pytest.mark.integration` - server must be running
- Use pytest fixtures and parametrization
- Performance tests validate optimizations

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
- `internal/WINDOWS_SUPPORT_DESIGN_AND_PLAN.md` - Windows implementation plan
- `docs/OPENAPI_DOCUMENTATION_UPDATE.md` - API doc changelog

## Working with Dependencies

Dependencies are in `pyproject.toml` with optional groups:
- `[mlx]` - Apple Silicon models
- `[llama-cpp]` - GGUF models
- `[stt]` - Speech-to-text
- `[performance]` - orjson, xxhash, turbojpeg, uvloop
- `[analytics]` - DuckDB for metrics
- `[profile]` - py-spy, memray
- `[all]` - Everything

Always use `uv pip install -e .[group]` for installation.

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
- `docs/WINDOWS_INSTALL.md` - Complete Windows setup guide
- `docs/WINDOWS_QUICK_REFERENCE.md` - Windows command reference
- `docs/CLAUDE_APP_QUICK_REFERENCE.md` - Quick command reference

**API Integration**
- `docs/CLIENT_INTEGRATION_GUIDE.md` - Complete API integration guide
- `docs/CLIENT_DEVELOPER_GUIDE.md` - Developer guide for API consumers
- `docs/CLIENT_CAPABILITY_DETECTION.md` - Feature detection guide
- `docs/COMFYUI_INTEGRATION_GUIDE.md` - ComfyUI integration guide
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

### Internal Documentation (internal/)

- `internal/MLX_CODE_REVIEW_2025.md` - MLX provider architecture and best practices

**Known Issues & Lessons Learned**
- `internal/VLM_VISION_BUG_2025-11-17.md` - **READ FIRST**: VLM vision bug caused by reimplementing library functions. Critical: Don't reimplement library functionality. Use mlx-vlm's built-in APIs.

**Architecture & Design**
- `internal/01_architecture_overview.md` - System architecture overview
- `internal/02_provider_mlx.md` - MLX provider design
- `internal/03_provider_llama_cpp.md` - Llama.cpp provider design
- `internal/04_provider_unification_strategy.md` - Provider unification strategy
- `internal/LOG.md` - Project development history
- `internal/MLX_CODE_REVIEW_2025.md` - Latest MLX compatibility and performance review (2025-01-14)

**Implementation Plans**
- `internal/WINDOWS_SUPPORT_DESIGN_AND_PLAN.md` - Windows support design
- `internal/QWEN3_VL_MLX_IMPLEMENTATION.md` - Vision model implementation
- `internal/BATCH_VISION_IMPLEMENTATION_PLAN.md` - Batch vision processing
- `internal/BATCH_TEXT_IMPLEMENTATION_PLAN.md` - Batch text generation (2025-01-14)
- `internal/LLAMA_CPP_ROADMAP.md` - Llama.cpp roadmap

**Performance & Optimization**
- `internal/DUAL_PATH.md` - Dual-path optimization design
- `internal/PERFORMANCE_OPTIMIZATION_GUIDE.md` - Performance notes
- `internal/IMAGE_TRANSFER_OPTIMIZATION.md` - Image optimization details
- `internal/IMPLEMENTATION_COMPLETE_2025-11-14.md` - Latest implementation summary (wired_limit + batch text)

**Analytics & Monitoring**
- `internal/HEYLOOK_ANALYTICS_TECH_SPEC.md` - Analytics technical spec
- `internal/HEYLOOK_ANALYTICS_TEST_GUIDE.md` - Analytics testing
- `internal/dependency_analysis_report.md` - Dependency analysis

### Configuration Files
- `CLAUDE.md` - Claude AI assistant guidelines (this file)
- `models.yaml` - Model configuration
- `pyproject.toml` - Python package configuration
