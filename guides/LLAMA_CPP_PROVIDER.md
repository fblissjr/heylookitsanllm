# GGUF/llama.cpp Provider Technical Guide

## Overview

The GGUF provider (`llama_cpp_provider.py`) enables support for quantized GGUF models via the llama-cpp-python library. This guide covers the technical implementation details, known issues, and solutions for concurrent request handling.

## Architecture

### Provider Class Structure

```python
class LlamaCppProvider(BaseProvider):
    def __init__(self, model_id: str, config: Dict, verbose: bool)
    def load_model(self)
    def create_chat_completion(self, request: ChatRequest) -> Generator
    def unload(self)
```

### Key Components

1. **Model Loading**: Uses `llama_cpp.Llama` class with configurable parameters
2. **Chat Handlers**: Supports custom Jinja2 templates and vision models (Llava15)
3. **Cache Management**: Uses `LlamaRAMCache` for KV cache storage
4. **Thread Safety**: Implements mutex lock for serialized access

## Concurrent Request Handling

### The Problem

llama-cpp-python maintains a stateful KV cache that is NOT thread-safe. When multiple requests attempt to use the same model instance concurrently, you'll encounter:

```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 1752
 - the tokens for sequence 0 in the input batch have a starting position of Y = 4
```

This error occurs because:
1. Request A starts processing and fills the KV cache
2. Request B arrives and tries to use the same model
3. The cache still contains Request A's data
4. Position mismatch causes a fatal error

### Our Solution: Mutex Lock with Cache Reset

```python
class LlamaCppProvider(BaseProvider):
    def __init__(self, ...):
        self._generation_lock = threading.Lock()  # Mutex for thread safety
    
    def create_chat_completion(self, request):
        with self._generation_lock:  # Only one request at a time
            try:
                # Reset model state before each request
                if self.model is not None:
                    self.model.reset()  # Clear KV cache
                    
                # Process request...
```

**Key Points:**
- **Serialization**: Only one request can use the model at a time
- **Cache Reset**: `model.reset()` clears the KV cache before each request
- **Automatic Release**: Context manager ensures lock is always released
- **Error Recovery**: Broken models are marked and reloaded on next request

### Alternative Approaches Considered

1. **Multiple Model Instances**: Memory prohibitive for large models
2. **Request Queuing**: Added complexity without performance benefit
3. **Native llama.cpp Server**: Would require major architecture changes
4. **Graph Reuse (PR #14482)**: Optimization for compute graphs, doesn't solve KV cache issue

## Configuration

### models.yaml Parameters

```yaml
- id: "gpt-oss-20b"
  provider: "gguf"  # or "llama_cpp" for backwards compatibility
  config:
    # Required
    model_path: "path/to/model.gguf"
    
    # Memory/GPU
    n_ctx: 16384          # Context window size
    n_gpu_layers: -1      # -1 for all layers on GPU
    n_batch: 1024         # Batch size for prompt processing
    n_threads: 22         # CPU threads (optimal: cores - 2)
    use_mmap: true        # Memory-mapped file loading
    use_mlock: true       # Lock model in RAM (if available)
    
    # Generation
    temperature: 1.0
    top_p: 0.95
    top_k: 50
    min_p: 0.05
    repeat_penalty: 1.0
    max_tokens: 512
    stop: ["<|endoftext|>", "<|return|>"]
    
    # Special Features
    embedding: true       # Enable embeddings extraction
    mmproj_path: "path/to/vision.gguf"  # For vision models
    chat_format: "chatml" # Built-in chat format
    chat_format_template: "templates/custom.jinja2"  # Custom template
```

### M2 Ultra Optimizations

For 192GB RAM systems:
- `n_gpu_layers: -1` - Use all Metal GPU layers
- `n_ctx: 32768` - Large context for smaller models
- `n_batch: 2048` - Leverage high memory bandwidth
- `use_mlock: true` - Lock model in RAM
- `n_threads: 22-24` - Optimal thread count

## Error Handling

### Broken Pipe Errors

The provider handles pipe disconnection errors gracefully:

```python
except BrokenPipeError as e:
    logging.error(f"Llama.cpp connection lost: {e}")
    self._model_broken = True  # Mark for reload
    yield error_response
```

### Model Recovery

When a model is marked as broken:
1. Current request returns an error
2. Next request triggers cleanup
3. Model is reloaded automatically
4. Processing continues normally

## Testing

### Basic Mutex Test

```bash
python test_mutex_simple.py
```

Tests sequential vs concurrent requests to verify mutex protection.

### Stress Test

```bash
python test_parallel_stress.py 10
```

Launches 10 parallel requests to stress-test the mutex implementation.

### Expected Behavior

With mutex protection:
- First request processes normally
- Concurrent requests wait for lock
- Each request gets clean KV cache
- No position mismatch errors
- All requests eventually complete

## Performance Considerations

### Throughput Limitations

- **Single-threaded**: Only one request processes at a time
- **Latency**: Average response time × number of concurrent requests
- **No batching**: Each request processes independently

### Optimization Strategies

1. **Model Caching**: Keep frequently used models loaded
2. **Quantization**: Use 4-bit/8-bit models for memory efficiency
3. **Context Size**: Reduce `n_ctx` if full context not needed
4. **Draft Models**: Use speculative decoding for faster generation

## Known Issues

1. **No True Parallelism**: Mutex serializes all requests
2. **Memory Leaks**: Some models may leak memory on reload
3. **Cache Corruption**: Without mutex, KV cache becomes corrupted
4. **Template Limitations**: Vision models can't use custom templates

## Future Improvements

### Short Term
- Implement request queuing with timeout
- Add metrics for wait time and throughput
- Improve error recovery mechanisms

### Long Term
- Multiple model instance pool
- Native llama.cpp server integration
- Batch processing support
- Distributed model serving

## Debugging

### Enable Debug Logging

```bash
heylookllm --log-level DEBUG --file-log-level DEBUG
```

### Common Error Messages

**KV Cache Corruption:**
```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions
```
**Solution**: Ensure mutex lock is working and cache is reset between requests

**Model Loading Failed:**
```
RuntimeError: Model not loaded. Call load_model() first.
```
**Solution**: Check model path and available memory

**Broken Pipe:**
```
BrokenPipeError: [Errno 32] Broken pipe
```
**Solution**: Model will auto-reload on next request

## References

- [llama-cpp-python Documentation](https://github.com/abetlen/llama-cpp-python)
- [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
- [GGUF Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Thread Safety Discussion](https://github.com/abetlen/llama-cpp-python/issues/438)