# Hey Look, It's an LLM

OpenAI-compatible API server for local LLM inference with MLX, llama.cpp, and CoreML.

A lightweight API server for running Apple MLX models, GGUF models, and a bunch of quality of life additions, using a modified OpenAI endpoint (can't really say it's fully compatible anymore given the additions), with on-the-fly model swapping and optional analytics.

## Key Features

- **OpenAI-Compatible API**: Works with existing OpenAI client libraries - Anthropic API compatible functionality coming soon
- **Multi-Provider**:
  - **MLX**: Optimized for Apple Silicon (with additional Metal acceleration)
      - Huge thanks to both the [MLX team](https://github.com/ml-explore) for [mlx-lm](https://github.com/ml-explore/mlx-lm) and [Blaizzy](https://github.com/Blaizzy) for [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Nothing here would work without them.
  - **llama.cpp**: Cross-platform support for GGUF models (CUDA, Vulkan, CPU)
      - Grateful for the continuous work put in by the core maintainers of [llama.cpp](https://github.com/ggerganov/llama.cpp), who've been pushing ahead since the first llama model.
  - **CoreML**: *Experimental* Speech-to-Text on Apple Silicon (Neural Engine)
- **Vision Models**: Process images with vision-language models (VLM), with support to push down / resize from the client
- **Performance Optimized**:
  - Metal acceleration on macOS
  - Async processing and smart caching
  - Fast multipart image endpoint
  - Batch processing for 2-4x throughput
- **Hot Swapping**: Change models on the fly without restarting, specified in the request body
- **Analytics**: Optional tracking and performance metrics

## Platform Support

- **macOS**: Both backends (MLX, llama.cpp)
- **Linux**: llama.cpp backend (CUDA, CPU)
- **Windows**: llama.cpp backend (CUDA, Vulkan, CPU)

## Quick Start

### Installation

**macOS/Linux**:
```bash
./setup.sh
```

**Windows**:
```powershell
.\setup.ps1
```

See [Windows Installation Guide](docs/WINDOWS_INSTALL.md) for detailed Windows setup.

### Manual Installation

```bash
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm

# Recommended: use uv sync for proper dependency resolution
uv sync                            # Base install
uv sync --extra mlx                # macOS only
uv sync --extra llama-cpp          # All platforms
uv sync --extra stt                # macOS only (CoreML) - very experimental
uv sync --extra analytics          # DuckDB analytics
uv sync --extra all                # Install everything

# Alternative: pip-style install (doesn't use lockfile)
uv pip install -e .
uv pip install -e .[mlx,llama-cpp]
```

### GPU Acceleration

**macOS (Metal)**
Included by default with `mlx`. For `llama-cpp`, run:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

**Linux/Windows (CUDA)**
```bash
# Linux
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python

# Windows (PowerShell)
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Start Server

```bash
# Configure models first
cp models.toml.example models.toml
# Edit models.toml with your model paths

# Start server
heylookllm --log-level INFO
heylookllm --port 8080
```

### Automatic Import

Scan directories for models and auto-generate configuration:

```bash
heylookllm import --folder ~/models --output models.toml
heylookllm import --hf-cache --profile fast
```

## API Documentation

Interactive docs available when server is running:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Schema**: http://localhost:8080/openapi.json

### Key Endpoints

**Core Endpoints** (`/v1`)
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion (text and vision)
- `POST /v1/chat/completions/multipart` - Fast raw image upload (57ms faster per image)
- `POST /v1/batch/chat/completions` - Batch processing (2-4x throughput)
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/hidden_states` - Extract hidden states from intermediate layers (MLX only)

**Speech-to-Text** (macOS only)
- `POST /v1/audio/transcriptions` - Transcribe audio
- `POST /v1/audio/translations` - Translate audio to English
- `GET /v1/stt/models` - List STT models

**Analytics and Admin**
- `GET /v1/capabilities` - Discover server capabilities and optimizations
- `GET /v1/performance` - Real-time performance metrics
- `GET /v1/data/summary` - Analytics summary (requires analytics enabled)
- `POST /v1/data/query` - Query analytics data
- `POST /v1/admin/restart` - Restart server
- `POST /v1/admin/reload` - Reload model configuration

### Example Usage (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# Chat
response = client.chat.completions.create(
    model="qwen2.5-coder-1.5b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
embedding = client.embeddings.create(
    input="Your text here",
    model="qwen2.5-coder-1.5b"
)
print(embedding.data[0].embedding[:5])  # First 5 dimensions
```

### Hidden States (MLX only)

Extract intermediate layer hidden states for use with diffusion models:

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/hidden_states",
    json={
        "model": "Qwen/Qwen3-4B",
        "input": "A photo of a cat",
        "layer_index": -2,  # Second-to-last layer
        "encoding_format": "base64"  # or "float" for JSON array
    }
)
result = response.json()
print(f"Shape: {result['data'][0]['shape']}")  # [seq_len, hidden_dim]
```

### Batch Processing

Process multiple requests efficiently with 2-4x throughput improvement:

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/batch/chat/completions",
    json={
        "requests": [
            {
                "model": "qwen-14b",
                "messages": [{"role": "user", "content": "Prompt 1"}],
                "max_tokens": 50
            },
            {
                "model": "qwen-14b",
                "messages": [{"role": "user", "content": "Prompt 2"}],
                "max_tokens": 50
            }
        ]
    }
)
```

## Analytics (Optional)

Track performance metrics and request history.

1. **Setup**: `python setup_analytics.py`
2. **Enable**: Set `HEYLOOK_ANALYTICS_ENABLED=true`
3. **Analyze**: `python analyze_logs.py`

## Troubleshooting

**Model not loading**
```bash
heylookllm --log-level DEBUG
```

## License

MIT License - see [LICENSE](LICENSE) file
