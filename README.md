# Hey Look, It's an LLM

Local multimodal LLM API server with dual OpenAI-compatible and Anthropic Messages-style endpoints, a React web UI, and on-the-fly model swapping.

Built on Apple MLX for text, vision, and speech-to-text.

## Key Features

- **Dual API**: OpenAI-compatible `/v1/chat/completions` and Anthropic Messages-style `/v1/messages` with typed content blocks (text, image, thinking, logprobs, hidden states)
- **Multi-Provider**:
  - **MLX**: Text and vision-language models on Apple Silicon ([mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm))
  - **MLX STT**: Speech-to-text via [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
- **Thinking Blocks**: Qwen3-style `<think>` parsing with token-level detection, round-trip editing, and streaming
- **Logprobs**: Per-token log probabilities with top-K alternatives (OpenAI-compatible format)
- **Hidden States**: Extract intermediate layer representations for diffusion model conditioning or research
- **Model Management**: Scan, import, configure, load/unload models from the web UI or API
- **Vision Models**: Image processing with VLMs, client-side resize, fast multipart upload
- **Batch Processing**: 2-4x throughput for multi-prompt workloads
- **Hot Swapping**: LRU cache holds up to 2 models, swaps on request
- **Performance**: Metal acceleration, async processing, prompt caching, compiled logit processors

## Web UI

7 applets built with React + Zustand + Vite:

- **Chat** -- Streaming conversation with thinking blocks, message editing, continue/regenerate
- **Batch** -- Multi-prompt batch jobs with result dashboard
- **Token Explorer** -- Real-time token probability visualization with top-K alternatives
- **Model Comparison** -- Side-by-side generation from 2-6 models
- **Performance** -- System metrics, timing breakdowns, throughput sparklines
- **Notebook** -- Base-model text continuation with cursor-based generation
- **Models** -- Scan, import, configure, and load/unload models

See [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) for frontend architecture.

## Platform Support

- **macOS (Apple Silicon)**: MLX backend + MLX STT

## Quick Start

### Installation

```bash
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm

# Base install (MLX on macOS)
uv sync

# Optional extras
uv sync --extra stt          # Speech-to-text (macOS, parakeet-mlx)
uv sync --extra analytics    # DuckDB analytics
uv sync --extra performance  # xxhash, uvloop, turbojpeg, cachetools
uv sync --extra all          # Everything
```

### Start Server

```bash
# Import models from your HuggingFace cache (auto-generates models.toml)
heylookllm import --hf-cache
# Or from a specific directory:
heylookllm import --folder ~/models

# Start
heylookllm --log-level INFO
heylookllm --port 8080
```

### Run as Background Service (macOS/Linux)

```bash
heylookllm service install            # localhost only
heylookllm service install --host 0.0.0.0  # LAN access
heylookllm service status|start|stop|restart|uninstall
```

### Adding Models

There are three ways to add models:

**Web UI** -- Open the Models applet (`/models`) in the browser. Click Import, scan a directory or your HuggingFace cache, select the models you want, pick a profile, and import. Models are added to `models.toml` and available immediately.

**CLI** -- Scan a directory or HF cache and generate config:
```bash
heylookllm import --folder ~/models --output models.toml
heylookllm import --hf-cache --profile tight_fast
```

**API** -- Scan then import programmatically (server must be running):
```bash
# Scan a directory for models
curl -X POST http://localhost:8080/v1/admin/models/scan \
  -H "Content-Type: application/json" \
  -d '{"paths": ["/path/to/models"], "scan_hf_cache": true}'

# Import selected models from scan results
curl -X POST http://localhost:8080/v1/admin/models/import \
  -H "Content-Type: application/json" \
  -d '{"models": [{"model_path": "mlx-community/Qwen3-4B-4bit"}], "profile": "tight_fast"}'
```

If you edit `models.toml` directly while the server is running, reload the config:
```bash
curl -X POST http://localhost:8080/v1/admin/reload
```

## API

Interactive docs at `http://localhost:8080/docs` when the server is running.

## Troubleshooting

```bash
heylookllm --log-level DEBUG
```

## License

MIT License -- see [LICENSE](LICENSE)
