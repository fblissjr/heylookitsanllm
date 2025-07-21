# macOS Performance Guide for heylookitsanllm

getting the most out of your apple silicon for local LLMs. this guide covers performance optimizations specific to macOS and Metal.

## Quick Start

```bash
# install with all performance optimizations
uv pip install -e .[performance]

# compile llama.cpp with metal support (required for GGUF models)
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## What the Performance Package Gets You

### 1. orjson (3-10x faster JSON)
- critical for vision models where image data is base64 encoded
- reduces API response times significantly
- handles large batch requests better

### 2. xxhash (50x faster hashing)
- image deduplication for batch processing
- cache key generation for repeated requests
- minimal CPU overhead

### 3. turbojpeg (4-10x faster JPEG ops)
- faster image encoding/decoding for vision models
- reduces preprocessing time for multimodal requests
- lower memory usage during image operations

### 4. uvloop (faster async)
- drop-in replacement for asyncio event loop
- 2-4x faster for high concurrency scenarios
- automatic on macOS/Linux

## Metal-Specific Optimizations

### Memory Management

the server automatically detects and optimizes for your Metal GPU:

```python
# automatic cache sizing based on available memory
# for M1 Max with 64GB unified memory:
- MLX cache limit: ~36GB
- Wired memory limit: ~25GB
```

### Model Loading Strategy

1. **Pre-warming**: default model loads on startup to avoid first-request latency
2. **LRU caching**: keep your most-used models in memory (configurable via `max_loaded_models`)
3. **Smart eviction**: least recently used models unload automatically

### Quantization Recommendations

for apple silicon, these quantization levels work best:

| Model Size | Recommended | Memory Usage | Quality |
|------------|-------------|--------------|---------|
| <7B | 4-bit or 8-bit | 2-4GB | Excellent |
| 7B-13B | 4-bit | 4-8GB | Very Good |
| 13B-30B | 4-bit | 8-16GB | Good |
| 30B-70B | 4-bit with quantized KV cache | 16-36GB | Good |
| 70B+ | 4-bit with aggressive KV quantization | 36GB+ | Acceptable |

### Cache Strategies

models.yaml configuration for different scenarios:

```yaml
# small model (<7B) - maximum quality
cache_type: "standard"

# medium model (7B-30B) - balanced
cache_type: "standard"  # or "quantized" if memory tight

# large model (30B+) - memory optimized
cache_type: "quantized"
kv_bits: 4
kv_group_size: 32
quantized_kv_start: 512
max_kv_size: 2048
```

## Vision Model Optimization

### Image Preprocessing

the server includes automatic image resizing to optimize token usage:

```bash
# via API parameters
resize_max: 512     # resize to max 512px maintaining aspect ratio
resize_max: 768     # good balance for most vision models
resize_max: 1024    # higher quality, more tokens

# or use the multipart endpoint for 57ms faster per image
POST /v1/chat/completions/multipart
```

### Batch Processing

for multiple images, use batch modes:

```python
# parallel processing for independent images
"processing_mode": "parallel"

# sequential with shared context
"processing_mode": "sequential_with_context"
```

## Monitoring Performance

### Real-time Metrics

```bash
# check optimization status
curl http://localhost:8080/v1/performance

# see what's available
curl http://localhost:8080/v1/capabilities
```

### Key Metrics to Watch

1. **Token Generation Speed**: aim for >2 tok/s for 70B models
2. **Image Load Time**: should be <5ms with optimizations
3. **Memory Pressure**: monitor via Activity Monitor
4. **Cache Hit Rate**: check model router logs

## Troubleshooting

### Metal Errors

if you see Metal assertion failures during concurrent requests:

```python
# models.yaml - add generation lock for problematic models
config:
  use_generation_lock: true  # serializes inference
```

### Memory Issues

for "out of memory" errors:

1. reduce `max_loaded_models` to 1
2. enable quantized KV cache
3. reduce `max_kv_size`
4. use smaller models or more aggressive quantization

### Slow First Token

this is normal for large models. solutions:

1. use speculative decoding with a draft model
2. pre-warm models on startup
3. keep frequently used models loaded

## Advanced Tuning

### Speculative Decoding

dramatically speeds up generation for larger models:

```yaml
# in models.yaml
config:
  draft_model_path: "path/to/small/model"
  num_draft_tokens: 4  # for 70B models
  num_draft_tokens: 6  # for 13B models
  num_draft_tokens: 8  # for 7B models
```

### Metal Shader Cache

mlx automatically caches compiled Metal shaders. first run of a new model will be slower while shaders compile.

### Network Optimizations

for multi-machine setups:

```bash
# bind to all interfaces
heylookllm --host 0.0.0.0

# increase uvicorn workers (if RAM permits)
# note: each worker loads models independently
uvicorn heylook_llm.api:app --workers 2
```

## Best Practices

1. **start simple**: begin with standard settings, optimize based on actual usage
2. **monitor actively**: use activity monitor + server metrics
3. **profile before optimizing**: understand your bottlenecks
4. **test with your workload**: synthetic benchmarks != real usage

remember: the fastest model is the one that fits comfortably in memory. when in doubt, use smaller models with less aggressive optimization rather than pushing the limits with larger ones.