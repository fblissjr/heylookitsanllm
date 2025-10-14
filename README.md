# Hey Look, It's an LLM

OpenAI and Ollama compatible API server for local LLM inference with MLX, llama.cpp, and CoreML.

A lightweight API server for running Apple MLX models, GGUF models, and CoreML STT behind OpenAI/Ollama compatible endpoints, with on-the-fly model swapping and optional analytics.

**Platform Support**
- macOS: All backends (MLX, llama.cpp, CoreML STT)
- Linux: llama.cpp backend
- Windows: llama.cpp backend (CUDA, Vulkan, CPU)

## Quick Start

### Installation

```bash
# macOS/Linux: Run setup wizard
./setup.sh

# Windows: Run PowerShell setup
.\setup.ps1
```

See [Windows Installation Guide](docs/WINDOWS_INSTALL.md) for detailed Windows setup.

### Manual Installation

```bash
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm

# Base install
uv pip install -e .

# Add backends
uv pip install -e .[mlx]           # macOS only
uv pip install -e .[llama-cpp]     # All platforms
uv pip install -e .[stt]           # macOS only
uv pip install -e .[all]           # Everything
```

### GPU Acceleration

**macOS (Metal)**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

**Linux/Windows (CUDA)**
```bash
# Linux
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python

# Windows
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
```

**Windows (Vulkan - AMD/Intel)**
```powershell
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
$env:FORCE_CMAKE = "1"
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
```

See [Windows Installation Guide](docs/WINDOWS_INSTALL.md) for Visual Studio Build Tools and SDK setup.

### Start Server

```bash
# Configure models
cp models.yaml.example models.yaml
# Edit models.yaml with your model paths

# Start server
heylookllm --api openai --log-level DEBUG
heylookllm --api ollama --log-level DEBUG
heylookllm --api both --port 8080
```

## Configuration

Edit `models.yaml` to define your models:

```yaml
models:
  - id: qwen2.5-coder-1.5b
    provider: mlx                    # or llama_cpp, coreml_stt
    enabled: true
    config:
      model_path: models/qwen-mlx
      max_tokens: 512
      temperature: 0.7
```

Or use automatic import:

```bash
heylookllm import --folder ~/models --output models.yaml
heylookllm import --hf-cache --profile fast
```

## API Documentation

Interactive API docs available when server is running:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Schema**: http://localhost:8080/openapi.json

See [Client Integration Guide](docs/CLIENT_INTEGRATION_GUIDE.md) for detailed API usage.

### Key Endpoints

**OpenAI Compatible**
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion
- `POST /v1/embeddings` - Extract embeddings
- `POST /v1/audio/transcriptions` - Speech-to-text (macOS)
- `POST /v1/chat/completions/multipart` - Fast image upload

**Ollama Compatible**
- `GET /api/tags` - List models
- `POST /api/chat` - Chat endpoint
- `POST /api/generate` - Text generation
- `POST /api/embed` - Generate embeddings

### Example Usage

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

# Embeddings
embedding = client.embeddings.create(
    input="Your text here",
    model="qwen2.5-coder-1.5b"
)

# Speech-to-text (macOS only)
with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="parakeet-tdt-v3",
        file=f
    )
```

## Features

### Model Management
- LRU cache holds 2 models in memory
- Automatic model loading/unloading
- Hot-swap models without restart

### Performance
- Metal acceleration (macOS)
- CUDA/Vulkan acceleration (Linux/Windows)
- Async request processing
- Fast multipart image endpoint (57ms faster per image)
- Cross-request prompt caching (MLX)

### Analytics (Optional)

```bash
# Setup
python setup_analytics.py
export HEYLOOK_ANALYTICS_ENABLED=true
uv pip install -e .[analytics]

# Run server
heylookllm --api openai

# Analyze
python analyze_logs.py
```

See analytics configuration in [README Analytics section](https://github.com/fblissjr/heylookitsanllm#analytics--logging).

## Platform-Specific Notes

### macOS
- MLX requires Apple Silicon (M1/M2/M3)
- CoreML STT uses Neural Engine
- Install gfortran for scipy: `brew install gcc`

### Linux
- Only llama.cpp backend
- CUDA requires CUDA Toolkit

### Windows
- Only llama.cpp backend
- Requires Visual Studio Build Tools
- See [Windows Installation Guide](docs/WINDOWS_INSTALL.md)
- See [Windows Quick Reference](docs/WINDOWS_QUICK_REFERENCE.md)

## Troubleshooting

**Model not loading**
```bash
heylookllm --log-level DEBUG
```

**GPU not working**
```bash
python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

**Port in use**
```bash
# macOS/Linux
lsof -i :8080

# Windows
Get-NetTCPConnection -LocalPort 8080

# Use different port
heylookllm --port 8081
```

**Windows scipy build errors (macOS)**
```bash
brew install gcc
```

See platform-specific guides for detailed troubleshooting:
- [Windows Installation Guide](docs/WINDOWS_INSTALL.md)
- [Client Integration Guide](docs/CLIENT_INTEGRATION_GUIDE.md)

## Documentation

- [Windows Installation Guide](docs/WINDOWS_INSTALL.md) - Complete Windows setup
- [Windows Quick Reference](docs/WINDOWS_QUICK_REFERENCE.md) - Windows commands
- [Client Integration Guide](docs/CLIENT_INTEGRATION_GUIDE.md) - API integration

## License

MIT License - see LICENSE file
