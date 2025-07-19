# src/heylook_llm/openapi_enhancements.py
"""
OpenAPI documentation enhancements for better clarity and examples.
"""

from typing import Dict, Any, List

# Enhanced descriptions for endpoints
ENDPOINT_DESCRIPTIONS = {
    "list_models": {
        "summary": "List Available Models",
        "description": """
List all language models currently available on this server.

**Purpose**: Discover which models are loaded and ready for inference.

**Use Cases**:
- Check available models before making completion requests
- Verify a specific model is loaded
- Get model metadata for client configuration

**Response includes**:
- Model IDs (e.g., "qwen2.5-coder-1.5b-instruct-4bit")
- Model type (always "model" for compatibility)
- Owner (always "user" for local models)

**Note**: Only models marked as `enabled: true` in models.yaml are shown.
        """,
        "response_description": "List of available models in OpenAI-compatible format"
    },
    
    "create_chat_completion": {
        "summary": "Create Chat Completion",
        "description": """
Generate text completions from chat messages using the specified model.

**Purpose**: Main inference endpoint for text and vision models.

**Supports**:
- ✅ Text-only models (MLX and GGUF)
- ✅ Vision models (with base64 encoded images)
- ✅ Streaming responses (SSE format)
- ✅ Batch processing (multiple prompts)
- ✅ Custom sampling parameters
- ✅ Reproducible generation with seeds

**Model Routing**:
- Automatically loads requested model if not in memory
- Uses LRU cache (max 2 models by default)
- Unloads least recently used model when cache full

**Performance Features**:
- Async request handling with thread pool
- Parallel image processing for vision models
- Optimized Metal acceleration for MLX models

**Special Parameters**:
- `processing_mode`: "sequential", "parallel", or "conversation" (for batch)
- `return_individual`: Return separate responses for batch requests
- `include_timing`: Add performance metrics to response
        """,
        "response_description": "Chat completion with generated text and usage statistics"
    },
    
    "create_chat_multipart": {
        "summary": "Create Chat Completion with Raw Images",
        "description": """
Generate completions with raw image uploads (no base64 encoding required).

**Purpose**: Optimized endpoint for vision models that eliminates base64 overhead.

**Benefits**:
- ⚡ 57ms faster per image (no encoding/decoding)
- 📉 33% less bandwidth usage
- 🚀 Parallel image processing
- 💾 Optional image caching

**How it works**:
1. Upload images as multipart form data
2. Include messages as JSON string
3. Images replace `__RAW_IMAGE__` placeholders in messages

**Example message format**:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}}
  ]}
]
```

**Ideal for**:
- ComfyUI integration
- High-volume image processing
- Network-constrained environments
- Real-time vision applications
        """,
        "response_description": "Standard chat completion response"
    },
    
    "ollama_chat": {
        "summary": "Ollama Chat API",
        "description": """
Ollama-compatible chat endpoint that translates to OpenAI format internally.

**Purpose**: Drop-in compatibility for Ollama clients and tools.

**Translation Process**:
1. Receives Ollama-format request
2. Translates to OpenAI ChatRequest
3. Processes through same inference pipeline
4. Translates response back to Ollama format

**Supports**:
- ✅ Streaming with Ollama's format
- ✅ All Ollama chat parameters
- ✅ Image inputs (base64)
- ✅ Compatible with Ollama CLI/SDK

**Note**: Uses the same model routing and caching as OpenAI endpoints.
        """,
        "response_description": "Ollama-format chat response"
    },
    
    "performance_metrics": {
        "summary": "Performance Metrics",
        "description": """
Get detailed performance metrics from all model providers.

**Purpose**: Monitor inference performance and optimize deployments.

**Metrics Include**:
- Token generation rates (tokens/sec)
- Time to first token (TTFT)
- Model loading times
- Memory usage statistics
- Request processing times
- Cache hit rates

**Aggregations**:
- Per-model statistics
- Per-provider summaries
- Rolling averages
- Peak performance records

**Use Cases**:
- Performance monitoring dashboards
- Capacity planning
- Model comparison
- Troubleshooting slow inference
        """,
        "response_description": "Detailed performance metrics and summaries"
    }
}

# Example requests for documentation
EXAMPLE_REQUESTS = {
    "text_completion": {
        "summary": "Basic text completion",
        "value": {
            "model": "qwen2.5-coder-1.5b-instruct-4bit",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
    },
    
    "vision_completion": {
        "summary": "Vision model with base64 image",
        "value": {
            "model": "llava-1.5-7b-hf-4bit",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
                ]}
            ],
            "max_tokens": 256
        }
    },
    
    "streaming_request": {
        "summary": "Streaming response",
        "value": {
            "model": "qwen2.5-coder-1.5b-instruct-4bit",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True,
            "max_tokens": 1024
        }
    },
    
    "batch_processing": {
        "summary": "Batch processing multiple prompts",
        "value": {
            "model": "qwen2.5-coder-1.5b-instruct-4bit",
            "messages": [
                {"role": "user", "content": "Prompt 1"},
                {"role": "user", "content": "Prompt 2"},
                {"role": "user", "content": "Prompt 3"}
            ],
            "processing_mode": "parallel",
            "return_individual": True,
            "max_tokens": 256
        }
    }
}

# Example responses for documentation
EXAMPLE_RESPONSES = {
    "models_list": {
        "summary": "Available models response",
        "value": {
            "object": "list",
            "data": [
                {"id": "qwen2.5-coder-1.5b-instruct-4bit", "object": "model", "owned_by": "user"},
                {"id": "llava-1.5-7b-hf-4bit", "object": "model", "owned_by": "user"},
                {"id": "gemma-2-2b-it-GGUF", "object": "model", "owned_by": "user"}
            ]
        }
    },
    
    "completion_response": {
        "summary": "Standard completion response",
        "value": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "qwen2.5-coder-1.5b-instruct-4bit",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here's a Python function for Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 48,
                "total_tokens": 73
            }
        }
    },
    
    "performance_response": {
        "summary": "Performance metrics response",
        "value": {
            "metrics": {
                "mlx_provider": {
                    "requests_processed": 156,
                    "total_tokens_generated": 45678,
                    "average_tokens_per_second": 127.3,
                    "average_time_to_first_token": 0.234,
                    "models": {
                        "qwen2.5-coder-1.5b-instruct-4bit": {
                            "requests": 89,
                            "avg_tokens_per_second": 145.2,
                            "peak_tokens_per_second": 203.5
                        }
                    }
                },
                "llama_cpp_provider": {
                    "requests_processed": 67,
                    "average_tokens_per_second": 89.4
                }
            },
            "summary": {
                "total_requests": 223,
                "uptime_seconds": 3456,
                "cache_hit_rate": 0.73
            }
        }
    }
}

def get_custom_openapi_schema(app) -> Dict[str, Any]:
    """
    Generate enhanced OpenAPI schema with detailed descriptions and examples.
    """
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="HeylookLLM - High-Performance Local LLM Server",
        version="1.0.1",
        description="""
# HeylookLLM API Documentation

A unified, high-performance API server for local LLM inference with dual OpenAI and Ollama compatibility.

## 🚀 Key Features

- **Dual API Support**: OpenAI and Ollama compatible endpoints
- **Multi-Provider**: Supports MLX (Apple Silicon) and GGUF models
- **Vision Models**: Process images with vision-language models
- **Performance Optimized**: Metal acceleration, async processing, smart caching
- **Hot Swapping**: Change models without restarting the server
- **Batch Processing**: Process multiple prompts efficiently

## 🔧 Quick Start

1. **List available models**: `GET /v1/models`
2. **Generate text**: `POST /v1/chat/completions`
3. **Process images**: Use multipart endpoint for best performance

## 📊 Performance Tips

- Use `/v1/chat/completions/multipart` for vision models (57ms faster per image)
- Enable streaming for real-time output
- Monitor performance with `/v1/performance` endpoint
- Install performance extras: `pip install heylookllm[performance]`

## 🔗 Client Libraries

Compatible with:
- OpenAI Python SDK
- Ollama Python/JS clients
- Any OpenAI-compatible tool
- ComfyUI via custom nodes

## 📝 Authentication

No authentication required for local deployment. Add your own auth layer if exposing publicly.
        """,
        routes=app.routes,
        tags=[
            {
                "name": "OpenAI API",
                "description": "OpenAI-compatible endpoints for maximum compatibility"
            },
            {
                "name": "Ollama API",
                "description": "Ollama-compatible endpoints for drop-in replacement"
            },
            {
                "name": "Monitoring",
                "description": "Performance monitoring and server status"
            }
        ]
    )
    
    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8080", "description": "Default local server"},
        {"url": "http://localhost:11434", "description": "Ollama-compatible port"},
    ]
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "GitHub Repository",
        "url": "https://github.com/fredbliss/heylookitsanllm"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema