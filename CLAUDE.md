# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Server Operations
```bash
# Start server in OpenAI API mode (default port 8080)
heylookllm --api openai --log-level DEBUG

# Start server in Ollama API mode (default port 11434)
heylookllm --api ollama --log-level DEBUG

# Start server with both APIs (specify port)
heylookllm --api both --port 8080 --log-level DEBUG

# Custom host/port
heylookllm --host 0.0.0.0 --port 4242
```

### Testing
```bash
# Run all tests with server running
./tests/run_tests.sh

# Run pytest unit tests only
cd tests && pytest -m unit

# Run integration tests (requires server)
cd tests && pytest -m integration

# Run specific test file
python tests/test_api.py
```

### Installation & Setup
```bash
# Install package in development mode (includes all dependencies)
uv pip install -e .

# Install with performance optimizations (orjson, xxhash, turbojpeg, uvloop)
uv pip install -e .[performance]

# Compile llama-cpp-python for Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Model Import
```bash
# Import models from a directory
heylookllm import --folder ~/modelzoo --output my_models.yaml

# Scan HuggingFace cache
heylookllm import --hf-cache --profile fast

# Import with specific profile (fast, balanced, quality, memory, interactive)
heylookllm import --folder ./models --profile quality

# Import with custom overrides
heylookllm import --folder ~/models --profile fast --override temperature=0.5 --override max_tokens=1024
```

## Project Architecture

### Core Components
- **`src/heylook_llm/`** - Main package containing all server logic
- **`src/heylook_llm/server.py`** - CLI entry point and FastAPI app creation
- **`src/heylook_llm/api.py`** - REST API endpoints (OpenAI + Ollama compatible)
- **`src/heylook_llm/router.py`** - Model routing with LRU cache (max 2 models in memory)
- **`src/heylook_llm/providers/`** - Model provider backends (MLX, llama.cpp)
- **`models.yaml`** - Model registry and configuration
- **`tests/`** - Test suite with pytest configuration

### Provider System
The server supports two backend providers:
- **MLX Provider** (`mlx_provider_optimized.py`) - For Apple Silicon MLX models (text + vision)
- **Llama.cpp Provider** (`llama_cpp_provider.py`) - For GGUF models via llama-cpp-python

Both providers implement the `BaseProvider` interface with `load_model()` and `create_chat_completion()` methods.

### Model Configuration
Models are defined in `models.yaml` with the following key fields:
- `id` - Unique model identifier for API calls
- `provider` - Either "mlx" or "llama_cpp"
- `enabled` - Boolean to enable/disable models
- `config` - Provider-specific settings (model_path, vision capabilities, etc.)

### API Compatibility
- **OpenAI API** - `/v1/models`, `/v1/chat/completions`, `/v1/capabilities`, `/v1/performance`, `/v1/chat/completions/multipart`
- **Ollama API** - `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`, `/api/embed`

## Key Design Patterns

### LRU Model Caching
The `ModelRouter` maintains an LRU cache of loaded models (default max: 2). When cache is full, the least recently used model is unloaded to make space for new models.

### Middleware Translation
Ollama API requests are translated to OpenAI format via `OllamaTranslator` middleware, processed through the same backend providers, then translated back to Ollama response format.

### Thread-Safe Loading
Model loading uses threading locks to prevent race conditions when multiple requests try to load the same model simultaneously.

### Streaming vs Non-Streaming
Both streaming and non-streaming responses are supported. Streaming uses Server-Sent Events (SSE) format.

## Development Conventions

### Code Style
- Follow PEP-8 with 120-character lines
- Use type hints throughout
- Import order: standard → third-party → local (separated by blank lines)
- Use existing logging setup, not bare `print()` statements

### Naming Conventions
- NEVER use vague marketing terms like "Enhanced", "Optimized", "Improved", "Resilient" in class/function names or log messages
- Use descriptive technical names that explain what the code does
- Instead of "OptimizedLanguageModelWrapper", use "LanguageModelLogitsWrapper" or similar
- Instead of "EnhancedVLMGenerator", use "VLMGeneratorWithSampling" or similar
- Instead of "resilient loading", use "loading with fallback strategies" or similar
- Names and messages should describe the technical behavior, not claim superiority or robustness

### Error Handling
- Raise `HTTPException` for API errors
- Log errors with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Provide meaningful error messages to users

### Testing Strategy
- Unit tests for individual components (marked with `@pytest.mark.unit`)
- Integration tests requiring running server (marked with `@pytest.mark.integration`)
- Performance tests for optimization validation
- Use pytest fixtures and parametrization

## Security & Performance Notes

### Resource Management
- Models are automatically unloaded when evicted from cache
- Vision models require additional memory for image processing
- Use quantized models (4-bit) for memory efficiency

### Model Swapping
- Hot-swappable models via API calls without server restart
- Model loading time varies (2-30s depending on model size)
- Pre-warming of default model on server startup

### Vision Model Support
- Both MLX and llama.cpp providers support vision models
- Image inputs via base64 encoding or URLs
- Automatic validation of vision capabilities per model

This architecture enables efficient local LLM serving with dual API compatibility and intelligent resource management.

## OpenAPI/Swagger Documentation

### Interactive API Documentation
The server provides built-in OpenAPI documentation:
- **Swagger UI**: `http://localhost:8080/docs` - Interactive API testing interface
- **ReDoc**: `http://localhost:8080/redoc` - Alternative documentation UI
- **OpenAPI Schema**: `http://localhost:8080/openapi.json` - Machine-readable API specification

### Client SDK Generation
Generate client libraries using the OpenAPI schema:
```bash
# Download schema
curl http://localhost:8080/openapi.json > openapi.json

# Generate Python client
openapi-generator generate -i openapi.json -g python -o ./python-client

# Generate TypeScript client
openapi-generator generate -i openapi.json -g typescript-axios -o ./ts-client
```

### Client Integration Resources
- **`CLIENT_INTEGRATION_GUIDE.md`** - Comprehensive guide for integrating with the API
- **`CLAUDE_APP_QUICK_REFERENCE.md`** - Quick reference for client applications
- **`src/heylook_llm/openapi_examples.py`** - Example requests/responses for documentation

### Key API Features for Clients
1. **Dual API Support**: OpenAI and Ollama compatible endpoints
2. **Vision Model Support**: Process images with vision-capable models
3. **Streaming Responses**: Real-time token generation with SSE
4. **Batch Processing**: Process multiple prompts efficiently
5. **No Authentication**: Local server requires no API keys
6. **Image Resizing**: Server-side image resizing to optimize token usage
7. **Multipart Upload**: Fast image upload endpoint (57ms faster per image)
8. **Performance Monitoring**: Real-time performance metrics and optimization status

### Example Client Usage
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen2.5-coder-1.5b-instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)