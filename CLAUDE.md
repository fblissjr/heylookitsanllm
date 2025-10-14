# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

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
- **OpenAI**: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`
- **Ollama**: `/api/tags`, `/api/chat`, `/api/generate`, `/api/embed`
- **Admin**: `/v1/admin/restart`, `/v1/admin/reload`

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
