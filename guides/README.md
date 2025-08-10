# Provider Architecture Guide

## Overview

heylookitsanllm uses a unified provider architecture to support multiple model backends (MLX and GGUF/llama.cpp) through a common interface. This design enables seamless model switching, consistent API behavior, and efficient resource management across different model formats.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   API Layer                         │
│         (OpenAI + Ollama Compatible)                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 ModelRouter                         │
│   - Model selection & routing                       │
│   - LRU cache management (max 2 models)            │
│   - Provider lifecycle management                   │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼────────┐
│  MLX Provider  │          │  GGUF Provider  │
│                │          │  (llama.cpp)    │
└────────────────┘          └─────────────────┘
```

## Base Provider Interface

All providers inherit from `BaseProvider` and must implement:

```python
class BaseProvider:
    def __init__(self, model_id: str, config: Dict, verbose: bool)
    def load_model(self) -> None
    def create_chat_completion(self, request: ChatRequest) -> Generator
    def unload(self) -> None
```

### Key Responsibilities

1. **`load_model()`**: Initialize model, load weights, configure handlers
2. **`create_chat_completion()`**: Process chat requests, yield streaming responses
3. **`unload()`**: Clean up resources, free memory, clear caches

## Provider Implementations

### MLX Provider (`mlx_provider.py`)

**Purpose**: Native Apple Silicon support via MLX framework

**Key Features**:
- Optimized for M-series chips (M1/M2/M3)
- Support for vision models (mlx-vlm)
- Speculative decoding with draft models
- Quantized KV cache for large models
- Real embeddings extraction

**Architecture Highlights**:
```python
class MLXProvider(BaseProvider):
    # Uses mlx_lm for text generation
    # Uses mlx_vlm for vision understanding
    # Supports quantized caches for 30B+ models
    # Thread-safe by default (MLX handles it)
```

**Concurrency**: Thread-safe, supports parallel requests naturally

### GGUF Provider (`llama_cpp_provider.py`)

**Purpose**: Cross-platform support for quantized GGUF models

**Key Features**:
- Wide model compatibility (any GGUF format)
- CPU and GPU acceleration
- Custom chat templates (Jinja2)
- Vision support (Llava format)
- Memory-mapped loading

**Architecture Highlights**:
```python
class LlamaCppProvider(BaseProvider):
    # Uses llama-cpp-python bindings
    # Mutex lock for thread safety
    # KV cache reset between requests
    # Automatic error recovery
```

**Concurrency**: Serialized via mutex lock (KV cache not thread-safe)

## Model Router

The `ModelRouter` class manages provider lifecycle and request routing:

### Key Components

1. **Provider Registry**:
```python
self.providers = {}  # Active provider instances
self.provider_map = {
    "mlx": MLXProvider,
    "gguf": LlamaCppProvider,
    "llama_cpp": LlamaCppProvider  # Backwards compatibility
}
```

2. **LRU Cache Management**:
```python
# Maximum 2 models in memory (configurable)
self.max_loaded = config.max_loaded_models
self.lru_order = []  # Track usage order

def evict_lru_model():
    # Unload least recently used model
    # Free memory before loading new model
```

3. **Thread-Safe Loading**:
```python
self.loading_locks = {}  # Per-model locks
# Prevents race conditions during model loading
```

## Request Flow

### 1. API Request Arrives
```
POST /v1/chat/completions
{
  "model": "dolphin-mistral",
  "messages": [...],
  "stream": true
}
```

### 2. Router Selection
```python
# ModelRouter.get_provider(model_id)
1. Check if provider already loaded
2. If not, check cache capacity
3. Evict LRU model if needed
4. Load new provider
5. Update LRU order
6. Return provider instance
```

### 3. Provider Processing
```python
# Provider.create_chat_completion(request)
1. Validate and prepare request
2. Process through model
3. Yield streaming chunks
4. Handle errors gracefully
```

### 4. Response Streaming
```python
# API layer streams back to client
for chunk in provider.create_chat_completion(request):
    yield f"data: {json.dumps(chunk)}\n\n"
```

## Configuration System

### Model Definition (`models.yaml`)
```yaml
models:
  - id: "model-identifier"
    provider: "mlx" | "gguf"
    enabled: true
    config:
      model_path: "path/to/model"
      # Provider-specific settings
```

### Provider Selection Logic
1. Request specifies model ID
2. Router looks up model config
3. Config determines provider type
4. Provider class instantiated with config

## Memory Management

### Cache Strategies

**MLX Provider**:
- Standard cache (default)
- Quantized cache (4-bit/8-bit for large models)
- Rotating cache (experimental)

**GGUF Provider**:
- RAM cache only
- Reset between requests (mutex protection)
- No cache sharing between requests

### Eviction Policy

```python
def handle_memory_pressure():
    1. Check current loaded models
    2. Identify LRU model
    3. Call provider.unload()
    4. Remove from cache
    5. Garbage collect
```

## Error Handling

### Provider-Level
- Model loading failures
- Generation errors
- Memory errors
- Broken pipe handling

### Router-Level
- Provider not found
- Model not enabled
- Cache management errors
- Thread safety violations

### Recovery Mechanisms
1. **Automatic Reload**: Broken models marked and reloaded
2. **Graceful Degradation**: Errors returned to client
3. **Resource Cleanup**: Always attempt cleanup on errors

## Performance Optimizations

### MLX Optimizations
- Metal GPU acceleration
- Quantized operations
- Speculative decoding
- Flash attention

### GGUF Optimizations
- GPU layer offloading
- Memory-mapped files
- Batch processing
- Thread tuning

### Router Optimizations
- LRU caching
- Lazy loading
- Thread pooling
- Lock minimization

## Thread Safety

### MLX Provider
- **Naturally thread-safe**: MLX framework handles concurrency
- **Parallel requests**: Multiple requests can process simultaneously
- **No special handling**: Framework manages resources

### GGUF Provider
- **Mutex protection**: Single request at a time
- **Cache reset**: Clear KV cache between requests
- **Sequential processing**: Trade-off for stability

### Router Level
- **Loading locks**: Per-model to prevent races
- **Cache locks**: Thread-safe cache operations
- **Atomic updates**: LRU order updates are atomic

## Provider Comparison

| Feature | MLX Provider | GGUF Provider |
|---------|--------------|---------------|
| Platform | macOS (Apple Silicon) | Cross-platform |
| Parallelism | Full parallel support | Serialized (mutex) |
| Memory Efficiency | Excellent with quantization | Good with mmap |
| Vision Support | Native (mlx-vlm) | Via Llava |
| Custom Templates | Limited | Full Jinja2 support |
| Embeddings | Native support | Requires flag |
| GPU Support | Metal only | Metal/CUDA/CPU |
| Model Formats | MLX weights | GGUF files |

## Adding New Providers

To add a new provider:

1. **Create Provider Class**:
```python
# src/heylook_llm/providers/new_provider.py
from .base import BaseProvider

class NewProvider(BaseProvider):
    def load_model(self):
        # Load your model
    
    def create_chat_completion(self, request):
        # Process request
        yield response_chunks
    
    def unload(self):
        # Cleanup
```

2. **Register in Router**:
```python
# src/heylook_llm/router.py
self.provider_map = {
    "mlx": MLXProvider,
    "gguf": LlamaCppProvider,
    "new": NewProvider  # Add here
}
```

3. **Update Config Schema**:
```python
# src/heylook_llm/config.py
class ModelConfig(BaseModel):
    provider: Literal["mlx", "gguf", "new"]
    # Add config class if needed
```

4. **Document Configuration**:
```yaml
# models.yaml
- id: "new-model"
  provider: "new"
  config:
    # Provider-specific config
```

## Best Practices

### Provider Development
1. **Inherit from BaseProvider**: Ensures interface compatibility
2. **Handle errors gracefully**: Don't crash the server
3. **Clean up resources**: Implement proper unload()
4. **Log appropriately**: Use consistent logging levels
5. **Document configuration**: Clear parameter descriptions

### Memory Management
1. **Set reasonable limits**: max_loaded_models = 1-2
2. **Monitor memory usage**: Check system resources
3. **Use quantization**: For models > 30B parameters
4. **Enable mmap**: For GGUF models when possible

### Concurrency
1. **Understand limitations**: GGUF requires serialization
2. **Use appropriate patterns**: Mutex for GGUF, native for MLX
3. **Test thoroughly**: Use provided test scripts
4. **Monitor performance**: Track response times

## Testing Providers

### Unit Tests
```python
# Test individual provider methods
pytest tests/test_providers.py
```

### Integration Tests
```python
# Test with running server
pytest tests/test_integration.py
```

### Concurrency Tests
```bash
# Test parallel handling
python test_mutex_simple.py      # Basic test
python test_parallel_stress.py   # Stress test
```

## Troubleshooting

### Common Issues

**MLX Provider**:
- Out of memory: Use quantized cache
- Slow generation: Enable flash attention
- Import errors: Check MLX installation

**GGUF Provider**:
- KV cache corruption: Ensure mutex is working
- Slow loading: Enable mmap
- GPU not used: Check n_gpu_layers setting

**Router**:
- Model not found: Check models.yaml
- Cache thrashing: Reduce max_loaded_models
- Thread deadlock: Check loading locks

## Future Enhancements

### Short Term
- Provider pooling for GGUF
- Better error recovery
- Performance metrics
- Request queuing

### Long Term
- Distributed providers
- Model sharding
- Dynamic batching
- Provider plugins

## Related Documentation

- [`LLAMA_CPP_PROVIDER.md`](LLAMA_CPP_PROVIDER.md) - GGUF provider details
- [`../CLAUDE.md`](../CLAUDE.md) - Development guidelines
- [`../README.md`](../README.md) - Project overview
- [`../models.yaml`](../models.yaml) - Model configuration