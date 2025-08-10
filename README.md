# Hey Look, It's an LLM (!)
OpenAI API and ollama API compatible LLM and Vision LLM (VLM) / multimodal server for mlx + llama.cpp

a lightweight (and lighthearted, but still aiming for quality), OpenAI-compatible API server that runs both vision and text Apple MLX models (text via `mlx-lm`, and vision via `mlx-vlm`, with some `mlx` stitching) and GGUF models (via `llama-cpp-python`) behind one endpoint, with live on-the-fly model swapping via API calls. trying to take the best of what's out locally and put it under one roof in a smart, performant way. allows for running in openai api mode, ollama mode, or both together. also (optionally) includes an analytics / eval app and embedded duckdb database for monitoring and evaluating model performance.

*note*: llama-cpp-python will by default install the cpu binary, but you can compile it yourself by following the instructions in the llama.cpp repo.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Model Import](#model-import)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Documentation](#api-documentation)
- [Analytics & Metrics](#analytics--metrics)
- [Running Tests](#running-tests)

## Quick Start

### Setup Wizard (Recommended)

```bash
# Run the interactive setup wizard
./setup.sh
```

The setup wizard will:
- Detect your OS (macOS/Linux/Windows)
- Let you choose which backends to install (MLX, llama.cpp, or both)
- Optionally configure GPU acceleration for llama.cpp
- Create initial models.yaml configuration

### Manual Installation

```bash
# Install base package (minimal dependencies)
uv pip install -e .

# Install with your preferred backend(s)
uv pip install -e .[mlx]           # For MLX models (macOS)
uv pip install -e .[llama-cpp]     # For GGUF models via llama.cpp
uv pip install -e .[mlx,llama-cpp] # Both backends
uv pip install -e .[all]           # Everything (default)

# Configure models (edit models.yaml with your paths)
cp models.yaml.example models.yaml

# Run server
heylookllm  # defaults to OpenAI mode on port 8080
```

### Server Modes

```bash
# OpenAI API mode (default)
heylookllm --api openai --log-level DEBUG

# Ollama API mode (port 11434)
heylookllm --api ollama --log-level DEBUG

# Both APIs on single port
heylookllm --api both --port 8080 --log-level DEBUG
```

## Installation

### Prerequisites
- Python 3.9+
- macOS (for MLX models) or Linux/Windows (for GGUF models)
- 8GB+ RAM recommended otherwise you're limited to models that are no fun and even 8GB is pushing it
- Metal (macOS) or CUDA (NVIDIA) for GPU acceleration

### Step 1: Clone & Install

```bash
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm
```

Choose one of these installation methods:

#### Option 1: Setup Wizard (Recommended)
```bash
./setup.sh
```

#### Option 2: Manual Installation
```bash
# Base install (minimal dependencies)
uv pip install -e .

# Add backends as needed:
uv pip install -e .[mlx]               # MLX models (macOS)
uv pip install -e .[llama-cpp]         # GGUF models
uv pip install -e .[mlx,llama-cpp]     # Both backends
uv pip install -e .[all]               # Everything

# Add optional features:
uv pip install -e .[mlx,performance]   # MLX + performance optimizations
uv pip install -e .[mlx,analytics]     # MLX + analytics
```

**Note for macOS users**: The MLX backend includes mlx-vlm which requires scipy. If you encounter scipy build errors:
```bash
brew install gcc  # Installs gfortran needed for scipy compilation
```

The `[performance]` option includes:
- **orjson**: 3-10x faster JSON operations (matters when you're sending images)
- **xxhash**: 50x faster image hashing for deduplication
- **turbojpeg**: 4-10x faster JPEG encoding/decoding
- **uvloop**: faster async event loop (linux/macos only)

These are optional but recommended if you're doing anything serious with vision models or high throughput.

### Step 2: GPU Acceleration (llama-cpp only)

If you installed the `[llama-cpp]` backend, you'll need to recompile llama-cpp-python for GPU acceleration. By default, it installs with CPU-only support.

#### macOS (Metal)
```bash
# After installing with pip install heylookitsanllm[llama-cpp]
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

#### NVIDIA (CUDA)
```bash
# After installing with pip install heylookitsanllm[llama-cpp]
CMAKE_ARGS="-GGML_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

#### CPU Only
```bash
# No additional steps needed - CPU version is installed by default with [llama-cpp]
```

### Step 3: Verify Installation

```bash
# Check llama.cpp (should show Metal/CUDA if compiled)
python -c "import llama_cpp; print('llama.cpp version:', llama_cpp.llama_cpp_version())"

# Check MLX packages
python -c "from mlx_lm import __version__; print('mlx_lm version:', __version__)"
python -c "from mlx_vlm import __version__; print('mlx_vlm version:', __version__)"
```

### Keeping Dependencies Updated

Since our key dependencies are actively developed, here's how to stay current:

#### Option 1: Automated Update Script (Recommended)
```bash
# Run the update script - it detects your OS and updates everything
./update-heylook.sh
```

This script will:
- Update all MLX packages (mlx, mlx-lm, mlx-vlm)
- Detect your OS and GPU configuration
- Automatically recompile llama-cpp-python with the appropriate acceleration (Metal/CUDA/CPU)

#### Option 2: Manual Updates
```bash
# Update MLX packages
uv pip install --upgrade mlx mlx-lm mlx-vlm

# Update llama-cpp-python (requires recompilation for GPU support)
# For Metal:
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

# For CUDA:
CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

# Update heylookitsanllm itself
cd /path/to/heylookitsanllm
git pull
uv pip install -e .[your-options]  # e.g., .[mlx] or .[all]
```

## Configuration

All models are defined in the **`models.yaml`** file. You **must** edit this file to point to your local models. Note you can also use the `import` command to automatically generate model configurations. See [Model Import](#model-import) for more right below.

Here's some of the key fields:

- `id`: unique model identifier for API calls
- `provider`: must be either `mlx` or `llama_cpp`
- `enabled`: boolean to enable/disable models
- `config`: provider-specific settings (model_path, vision capabilities, etc.)

Example model configuration:
```yaml
- id: qwen2.5-coder-1.5b
  enabled: true
  provider: mlx
  config:
    model_path: models/Qwen2.5-Coder-1.5B-Instruct-MLX-4bit
    max_tokens: 4096
    temperature: 0.7
```

My setup is to use the included `modelzoo` directory and set up a symbolic link to it. On Mac and Linux and WSL2, you can do this: `ln -s /my_models/live/here/* .` and you'll have them all there without duplication.

See the provided `models.yaml` for examples of different model setups with mlx, llama.cpp / gguf, including vision, text, both, etc.

## Model Import

tired of manually editing models.yaml? we got you. the import command scans directories and generates model configs with smart defaults:

```bash
# Scan a directory (follows symlinks)
heylookllm import --folder ~/modelzoo

# Scan HuggingFace cache
heylookllm import --hf-cache

# Use profiles for different use cases
heylookllm import --folder ~/models --profile fast      # speed optimized
heylookllm import --folder ~/models --profile quality   # quality optimized
heylookllm import --folder ~/models --profile memory    # low memory usage

# Fine-tune with overrides
heylookllm import --folder ~/models --profile fast --override temperature=0.5
```

Profiles:
- `fast`: aggressive sampling, quantized cache for speed
- `balanced`: default, good middle ground
- `quality`: conservative sampling, standard cache
- `memory`: maximum memory savings
- `interactive`: optimized for chat use

The importer detects:
- model size from filenames or file sizes
- vision support (mmproj files, config flags)
- quantization (4bit, 8bit, etc)
- model family (llama, qwen, gemma, mistral)

All imported models start disabled, so you can review before enabling.

## Running the Server

Once configured, start the server from the root `heylookitsanllm` directory. Note that if you want to access the server from another machine on your local network, set the host to `0.0.0.0`. Otherwise, it defaults to `127.0.0.1`.

```bash
# Basic usage
heylookllm

# Network accessible
heylookllm --host 0.0.0.0 --port 8080

# Debug mode with metrics
heylookllm --log-level DEBUG
```

## API Documentation

The server provides interactive API documentation at:
- **Swagger UI**: `http://localhost:8080/docs` - Interactive API testing interface
- **ReDoc**: `http://localhost:8080/redoc` - Alternative documentation UI
- **OpenAPI Schema**: `http://localhost:8080/openapi.json` - Machine-readable API specification

### Key Endpoints

- `/v1/models` - List available models
- `/v1/chat/completions` - OpenAI-compatible chat endpoint
- `/v1/embeddings` - Extract real model embeddings for semantic search
- `/v1/chat/completions/multipart` - Fast multipart upload for images (57ms faster per image)
- `/v1/capabilities` - Server capabilities and optimization status
- `/v1/performance` - Real-time performance metrics
- `/v1/data/query` - Execute SQL queries on analytics (when enabled)
- `/v1/data/summary` - Get analytics dashboard summary (when enabled)

### Example Usage

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="qwen2.5-coder-1.5b-instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")

# Embeddings for semantic search
embeddings = client.embeddings.create(
    input="Your text here",
    model="qwen2.5-coder-1.5b-instruct-4bit"
)
vector = embeddings.data[0].embedding
```

**curl:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b-instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Analytics & Logging

The server offers two independent logging systems for different needs:

### Quick Decision Guide

| Use Case | Solution | Command |
|----------|----------|---------|
| **Debugging issues** | File logging | `heylookllm --file-log-level DEBUG` |
| **Performance analysis** | Analytics DB | `export HEYLOOK_ANALYTICS_ENABLED=true && heylookllm` |
| **Production monitoring** | Both | `python setup_logging.py && python start_with_logging.py` |
| **Quick testing** | Neither | `heylookllm` |

### File Logging (Text Logs)
**Best for:** Debugging, troubleshooting, error tracking
- Real-time tailing with `tail -f`
- Grep-able text format
- Stack traces and error details
- Server internals and execution flow

```bash
# Enable file logging with separate console/file levels
heylookllm --log-level WARNING --file-log-level DEBUG

# Options:
# --file-log-level    Set file log level (DEBUG/INFO/WARNING/ERROR)
# --log-dir           Directory for logs (default: logs)
# --log-rotate-mb     Max MB per file (default: 100)
# --log-rotate-count  Rotated files to keep (default: 10)
```

### Analytics Database (DuckDB)
**Best for:** Performance metrics, model comparison, usage analysis
- SQL queries for complex analysis
- Request/response history
- Token usage and latency tracking
- Model A/B testing

```bash
# Quick setup
python setup_logging.py  # Creates analytics_config.json

# Enable analytics
export HEYLOOK_ANALYTICS_ENABLED=true
heylookllm --api openai
```

### Installation

```bash
# Analytics support (includes DuckDB)
uv pip install -e .[analytics]

# With performance optimizations
uv pip install -e .[performance,analytics]
```

Configuration options:
- `HEYLOOK_ANALYTICS_ENABLED` - Enable/disable analytics (default: false)
- `HEYLOOK_ANALYTICS_STORAGE_LEVEL` - Data collection level:
  - `none`: No data collection
  - `basic`: Only counters and timing metrics
  - `requests`: Request metadata without content
  - `full`: Complete requests/responses for replay & eval
- `HEYLOOK_ANALYTICS_DB_PATH` - Database file path (default: analytics.db)
- `HEYLOOK_ANALYTICS_RETENTION_DAYS` - Data retention period (default: 30)
- `HEYLOOK_ANALYTICS_MAX_DB_SIZE_MB` - Max database size (default: 1000)
- `HEYLOOK_ANALYTICS_LOG_IMAGES` - Store image data (default: false)
- `HEYLOOK_ANALYTICS_ANONYMIZE_CONTENT` - Anonymize PII (default: false)

### Analytics Features

When enabled, you get:
- **Request/Response Logging**: Query your conversation history
- **Performance Metrics**: Token usage, response times, model comparisons
- **SQL Interface**: Custom analytics queries
- **Dashboard API**: Pre-computed metrics for visualization
- **Request Replay**: Re-run requests with different models (full storage mode)
- **Eval Tracking**: Compare outputs across models and parameters

Example queries:
```sql
-- Find slow requests
SELECT model, total_time_ms, prompt_tokens + completion_tokens as total_tokens
FROM request_logs
WHERE total_time_ms > 1000
ORDER BY total_time_ms DESC;

-- Compare model performance
SELECT model,
       AVG(tokens_per_second) as avg_tps,
       AVG(first_token_ms) as avg_first_token
FROM request_logs
GROUP BY model;

-- Find errors
SELECT model, error_type, COUNT(*) as count
FROM request_logs
WHERE success = false
GROUP BY model, error_type;
```

Analyze logs after running:
```bash
python analyze_logs.py  # Generates performance reports and exports to CSV
```

### HeylookAnalytics Dashboard

A web dashboard is available at `apps/HeylookAnalytics-Web`:
```bash
cd apps/HeylookAnalytics-Web
npm install
npm start
```

Access at `http://localhost:3000` to view:
- Real-time request metrics
- Model performance comparisons
- Interactive playground for testing
- Model comparison tool
- Conversation tester
- SQL query interface

## Example Client Apps

- **comfyui nodes**: [shrug-prompter](https://github.com/heylookitsanllm/shrug-prompter) - now with multipart support
- **web dashboard**: HeylookAnalytics with real-time streaming, model comparison, batch processing

## Client SDK Generation

Generate client libraries using the OpenAPI schema:
```bash
# Download schema
curl http://localhost:8080/openapi.json > openapi.json

# Generate Python client
openapi-generator generate -i openapi.json -g python -o ./python-client

# Generate TypeScript client
openapi-generator generate -i openapi.json -g typescript-axios -o ./ts-client
```

## Troubleshooting

### Common Issues

**scipy build errors on macOS (MLX installation)**
```bash
# Install gfortran compiler
brew install gcc

# Then retry installation
uv pip install -e .[mlx]
```

**llama-cpp-python not using GPU**
- Make sure to recompile with the appropriate CMAKE flags after installation
- Check GPU support: `python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`

**Model not loading**
- Check your `models.yaml` paths are correct
- Ensure the model file exists and is readable
- Check logs with `--log-level DEBUG` for detailed error messages

**Port already in use**
```bash
# Find process using port 8080
lsof -i :8080

# Use a different port
heylookllm --port 8081
```

## License

MIT License - see LICENSE file for details
