# src/heylook_llm/openapi_examples.py
"""
OpenAPI examples for API documentation
"""

chat_completion_examples = {
    "simple_chat": {
        "summary": "Simple chat completion",
        "description": "A basic chat completion request",
        "value": {
            "model": "qwen2.5-coder-1.5b-instruct-4bit",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
    },
    "streaming_chat": {
        "summary": "Streaming chat completion",
        "description": "Request with streaming enabled",
        "value": {
            "model": "qwen2.5-coder-1.5b-instruct-4bit",
            "messages": [
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
            ],
            "stream": True,
            "temperature": 0.5
        }
    },
    "vision_chat": {
        "summary": "Vision model chat",
        "description": "Chat with image input (requires vision-capable model)",
        "value": {
            "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }
    }
}

models_response_example = {
    "summary": "List of available models",
    "value": {
        "object": "list",
        "data": [
            {"id": "qwen2.5-coder-1.5b-instruct-4bit", "object": "model", "owned_by": "user"},
            {"id": "mlx-community/Qwen2-VL-2B-Instruct-4bit", "object": "model", "owned_by": "user"},
            {"id": "deepseek-coder-1.3b-instruct-q4", "object": "model", "owned_by": "user"}
        ]
    }
}

chat_completion_response_example = {
    "summary": "Successful chat completion",
    "value": {
        "id": "chatcmpl-6e3f8f74-a3c4-4b1e-9c72-7e8a9b1d2c3f",
        "object": "chat.completion",
        "created": 1704326400,
        "model": "qwen2.5-coder-1.5b-instruct-4bit",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 8,
            "total_tokens": 33
        }
    }
}

streaming_response_example = {
    "summary": "Streaming response format",
    "description": "Each chunk is sent as a Server-Sent Event",
    "value": """data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1704326400,"model":"qwen2.5-coder-1.5b-instruct-4bit","choices":[{"delta":{"content":"The"}}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1704326400,"model":"qwen2.5-coder-1.5b-instruct-4bit","choices":[{"delta":{"content":" capital"}}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1704326400,"model":"qwen2.5-coder-1.5b-instruct-4bit","choices":[{"delta":{"content":" of"}}]}

data: [DONE]"""
}

ollama_chat_example = {
    "summary": "Ollama chat request",
    "value": {
        "model": "qwen2.5-coder-1.5b-instruct-4bit",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False
    }
}

ollama_generate_example = {
    "summary": "Ollama generate request",
    "value": {
        "model": "qwen2.5-coder-1.5b-instruct-4bit",
        "prompt": "The meaning of life is",
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9
        }
    }
}

# Embeddings endpoint examples
embeddings_examples = {
    "single_text": {
        "summary": "Single text embedding",
        "description": "Generate embedding for a single text",
        "value": {
            "input": "The quick brown fox jumps over the lazy dog",
            "model": "dolphin-mistral"
        }
    },
    "batch_embeddings": {
        "summary": "Batch text embeddings",
        "description": "Generate embeddings for multiple texts",
        "value": {
            "input": [
                "Machine learning is fascinating",
                "Deep learning and neural networks",
                "Natural language processing"
            ],
            "model": "dolphin-mistral"
        }
    },
    "truncated_embeddings": {
        "summary": "Embeddings with dimension truncation",
        "description": "Generate embeddings truncated to specific dimensions",
        "value": {
            "input": "Text to embed with truncation",
            "model": "dolphin-mistral",
            "dimensions": 512
        }
    }
}

embeddings_response_example = {
    "single_response": {
        "summary": "Single embedding response",
        "description": "Response for a single text embedding",
        "value": {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.0234, -0.1567, 0.8901, "... (768 dimensions total)"],
                    "index": 0
                }
            ],
            "model": "dolphin-mistral",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
    },
    "batch_response": {
        "summary": "Batch embeddings response",
        "description": "Response for multiple text embeddings",
        "value": {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.0234, -0.1567, "..."],
                    "index": 0
                },
                {
                    "object": "embedding", 
                    "embedding": [0.1234, -0.2567, "..."],
                    "index": 1
                },
                {
                    "object": "embedding",
                    "embedding": [0.3234, -0.4567, "..."],
                    "index": 2
                }
            ],
            "model": "dolphin-mistral",
            "usage": {
                "prompt_tokens": 24,
                "total_tokens": 24
            }
        }
    }
}