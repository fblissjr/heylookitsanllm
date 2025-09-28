# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Clean, simple names that describe what they do

## Working with 3rd Party Libraries
- Ensure you search for the latest llms.txt or Python docs for the library you are using or proposing
- If the latest docs are not available, search for the latest version of the library on the official website or GitHub repository.
- Prioritize the latest code and docs over your own training data.
- Use the latest version of the library over older versions.
- Prioritize libraries that are actively maintained and have a large community.

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

# Enable file logging with different levels for console and file
heylookllm --log-level INFO --file-log-level DEBUG --log-dir logs

# Full logging options
heylookllm --api openai \
  --log-level WARNING \           # Console shows warnings and errors only
  --file-log-level DEBUG \         # File logs everything
  --log-dir custom_logs \          # Custom log directory
  --log-rotate-mb 200 \            # Rotate at 200MB
  --log-rotate-count 5             # Keep 5 rotated files
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

# Test GGUF concurrent request handling
python test_mutex_simple.py        # Basic mutex test
python test_parallel_stress.py 10  # Stress test with 10 parallel requests

# Test new features
python tests/test_keepalive.py     # Test SSE keepalive during long prompts

# Test STT integration
python tests/test_stt_integration.py  # Test Speech-to-Text with Parakeet model
```

### Installation & Setup

**IMPORTANT: Dependency management has been consolidated into pyproject.toml with security updates.**
All requirements.txt files are now deprecated. Use the commands below or the automated setup script.

```bash
# Option 1: Automated setup (recommended)
./setup.sh

# Option 2: Manual installation
# Install package in development mode (minimal core dependencies only)
uv pip install -e .

# Install with specific backends
uv pip install -e .[mlx]          # For MLX models (macOS)
uv pip install -e .[llama-cpp]    # For GGUF models via llama.cpp
uv pip install -e .[stt]           # For CoreML STT models (Speech-to-Text)
uv pip install -e .[mlx,llama-cpp,stt] # All backends

# Note: MLX installation includes scipy for mlx-vlm. On macOS, if you get scipy build errors:
# brew install gcc  # Installs gfortran needed for scipy

# Install with performance optimizations
uv pip install -e .[performance]  # orjson, xxhash, turbojpeg, uvloop, imagecodecs, etc.

# Install with analytics support
uv pip install -e .[analytics]    # duckdb

# Install with profiling tools
uv pip install -e .[profile]      # py-spy, memray

# Install everything
uv pip install -e .[all]          # includes all optional dependencies

# GPU acceleration for llama-cpp (optional, after installing [llama-cpp])
# macOS Metal:
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
# NVIDIA CUDA:
CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Logging & Analytics

Two independent systems for different needs:

**File Logging** (debugging/troubleshooting):
```bash
# Separate console and file log levels
heylookllm --log-level INFO --file-log-level DEBUG --log-dir logs
```

**Analytics DB** (performance/metrics):
```bash
# Setup and enable
python setup_analytics.py
export HEYLOOK_ANALYTICS_ENABLED=true
heylookllm --api openai

# Analyze after running
python analyze_logs.py
```

**Both** (production):
```bash
python setup_analytics.py
export HEYLOOK_ANALYTICS_ENABLED=true
heylookllm --api openai --file-log-level DEBUG
```

Output locations:
- Text logs: `logs/heylookllm_*.log` (with --file-log-level)
- Analytics: `logs/analytics.db` (when enabled via config)

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

## New Features

### 1. Keepalive Support
- SSE keepalive comments during long prompt processing prevent client timeouts
- Automatically sends `: keepalive` messages every 10 seconds during prompt processing
- Particularly useful for vision models with large image processing times

### 2. Cross-Request Prompt Cache Persistence
- MLX models now cache KV states across requests for faster multi-turn conversations
- Common prefix detection automatically reuses cached tokens (system prompts, conversation history)
- Cache trimming support for partial prefix matches
- Up to 10x speedup for conversations with shared context

### 3. Quantized KV Cache Configuration
- Configure memory-efficient 8-bit quantized caches in `models.yaml`
- Reduces memory usage by 50% with minimal quality impact
- Example configuration:
```yaml
cache_type: "quantized"
kv_bits: 8              # 8-bit quantization
kv_group_size: 64       # Standard group size
```

### 4. Speech-to-Text (STT) Support
- CoreML-based STT provider for on-device speech recognition
- OpenAI Whisper-compatible API endpoints
- Support for Parakeet TDT v3 (0.6B) model with Neural Engine acceleration
- API Endpoints:
  - `/v1/audio/transcriptions` - Transcribe audio files (WAV, MP3, M4A, FLAC, OGG, WebM)
  - `/v1/stt/models` - List available STT models
  - `/v1/stt/stream` - WebSocket for real-time streaming transcription
- Configuration in `models.yaml`:
```yaml
- id: parakeet-tdt-v3
  provider: "coreml_stt"
  config:
    model_path: ~/Storage/parakeet_coreml
    compute_units: "ALL"  # Use Neural Engine
    sample_rate: 16000
    max_audio_seconds: 15
```

## Project Architecture

### Core Components
- **`src/heylook_llm/`** - Main package containing all server logic
- **`src/heylook_llm/server.py`** - CLI entry point and FastAPI app creation
- **`src/heylook_llm/api.py`** - REST API endpoints (OpenAI + Ollama compatible)
- **`src/heylook_llm/router.py`** - Model routing with LRU cache (max 2 models in memory)
- **`src/heylook_llm/providers/`** - Model provider backends (MLX, llama.cpp)
- **`src/heylook_llm/providers/common/prompt_cache.py`** - Cross-request prompt caching
- **`src/heylook_llm/providers/common/cache_helpers.py`** - KV cache configuration
- **`models.yaml`** - Model registry and configuration
- **`tests/`** - Test suite with pytest configuration

### Provider System
The server supports three backend providers:
- **MLX Provider** (`mlx_provider.py`) - For Apple Silicon MLX models (text + vision)
- **GGUF Provider** (`llama_cpp_provider.py`) - For GGUF models via llama-cpp-python
- **CoreML STT Provider** (`coreml_stt_provider.py`) - For CoreML speech-to-text models

LLM providers implement the `BaseProvider` interface with `load_model()` and `create_chat_completion()` methods.
The STT provider implements `load_model()` and `transcribe()` methods for audio processing.

### Model Configuration
Models are defined in `models.yaml` with the following key fields:
- `id` - Unique model identifier for API calls
- `provider` - "mlx", "gguf"/"llama_cpp", or "coreml_stt"
- `enabled` - Boolean to enable/disable models
- `config` - Provider-specific settings (model_path, vision capabilities, audio settings, etc.)

### API Compatibility
- **OpenAI API** - `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/capabilities`, `/v1/performance`, `/v1/chat/completions/multipart`, `/v1/audio/transcriptions`
- **Ollama API** - `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`, `/api/embed`
- **STT API** - `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/stt/models`, `/v1/stt/stream`
- **Admin API** - `/v1/admin/restart`, `/v1/admin/reload` (for development use)

## GGUF/llama.cpp Provider Details

### Concurrent Request Handling
The GGUF provider uses a mutex lock to serialize requests because llama-cpp-python's KV cache is not thread-safe. Key implementation:
- `threading.Lock()` ensures only one request uses the model at a time
- `model.reset()` clears KV cache before each request to prevent corruption
- Broken models are automatically marked and reloaded
- See `guides/LLAMA_CPP_PROVIDER.md` for full technical details

### Known Limitations
- No true parallel processing for GGUF models (serialized via mutex)
- KV cache corruption if mutex is disabled
- Single model instance per model ID (memory constraint)

## Key Design Patterns

### LRU Model Caching
The `ModelRouter` maintains an LRU cache of loaded models (default max: 2). When cache is full, the least recently used model is unloaded to make space for new models.

### Default Max Tokens
- MLX Provider: 512 tokens (src/heylook_llm/providers/mlx_provider.py:826)
- Llama.cpp Provider: 512 tokens (src/heylook_llm/providers/llama_cpp_provider.py:122)
- API fallback: 1000 tokens (various places in api.py)
- Can be overridden in models.yaml or per-request

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
- Use descriptive business logic names that explain what the code does
- Avoid overhyping or mishyping
- Names and messages should describe the technical behavior, not claim superiority or robustness
- Never use emojis in code or in markdown files or in any documentation or analysis

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
- Both MLX and GGUF providers support vision models
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
9. **Embeddings API**: Extract real model embeddings for semantic search and similarity

### Example Client Usage

#### Chat Completions
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="dolphin-mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Embeddings
```python
# Generate embeddings for semantic search
response = client.embeddings.create(
    input="Your text here",
    model="dolphin-mistral"
)
embedding = response.data[0].embedding

# Batch processing
response = client.embeddings.create(
    input=["text1", "text2", "text3"],
    model="dolphin-mistral"
)
embeddings = [d.embedding for d in response.data]
```

## Embeddings API

### Overview
The `/v1/embeddings` endpoint provides real model embeddings extraction for both MLX and llama.cpp models. Unlike approaches that ask LLMs to generate embedding arrays (which produces meaningless random numbers), this endpoint extracts actual embeddings from the model's internal representations.

### Key Features
- **Real Embeddings**: Extracts actual model embeddings from internal representations
- **Multi-Provider Support**: Works with both MLX and GGUF backends
- **Vision Model Support**: Can extract embeddings from vision-language models
- **Batch Processing**: Process multiple texts in a single request
- **Dimension Truncation**: Optionally truncate embeddings to specific dimensions
- **Normalized Output**: Embeddings are L2-normalized by default

### Implementation Details
- **MLX Models**: Extracts embeddings from token embeddings layer or hidden states
- **GGUF Models**: Uses built-in `create_embedding` method (requires `embedding=True` during model init)
- **TokenizerWrapper**: Properly handles MLX's TokenizerWrapper by accessing `_tokenizer` attribute
- **Pooling Strategies**: Supports mean, cls, last, and max pooling (MLX only)

### Testing
```bash
# Test embeddings endpoint
python tests/test_embeddings.py

# Test without server
python tests/test_embeddings_direct.py
```

## Documentation and Configuration Management

### Configuration Tracking
- Always document endpoint or configuration changes
- Update OpenAPI documentation for API-related modifications
- Maintain clear tracking of configuration updates in relevant documentation
- Ensure configuration changes are traceable and reproducible

### Documentation Best Practices
- Keep documentation updated and synchronized with code changes
- Use clear, concise language when describing configurations and endpoints
- Include context and rationale for significant configuration updates

## HeylookPlayground Apps

### HeylookPlayground-web (Primary - Desktop Web App)
**Location**: `apps/HeylookPlayground-web/`
**Documentation**: See `apps/HeylookPlayground-web/README.md` for complete setup and development guide

A clean, desktop-first web application for interacting with heylookitsanllm server.

#### Quick Start
```bash
cd apps/HeylookPlayground-web
npm install
npm run dev         # Start development server on http://localhost:3000
npm run test:visual # Run Playwright visual tests
```

#### Key Features
- **Desktop-optimized interface**: Fixed sidebar navigation, proper layouts
- **Chat Interface**: Real-time chat with model selection
- **Visual Testing**: Playwright integration for automated UI testing
- **Pure Web Stack**: Vite + React + TypeScript (no React Native)

#### Tech Stack
- **Build Tool**: Vite for fast development
- **Framework**: React with TypeScript
- **Styling**: CSS modules with custom styles
- **Testing**: Playwright for visual regression testing
- **API Client**: Axios for heylookitsanllm server communication
- When I'm trying to do something that makes no sense, or is just flat out dumb or a bad idea, or you *know* it's a dead end, push back on me and challenge me and propose alternatives - don't just go down the path knowing it won't work.