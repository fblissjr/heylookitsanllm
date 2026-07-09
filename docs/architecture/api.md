# API Architecture

Last updated: 2026-07-09

This document provides complete documentation of the FastAPI server architecture, all endpoints, request/response formats, and implementation details.

## Overview

heylookitsanllm provides an OpenAI-compatible API with additional endpoints for batch processing, embeddings, hidden states extraction, and server administration.

## Recent Changes

### 2026-01-09: Server Architecture Simplification (DRY)

**Problem**: `server.py` had a `create_openai_only_app()` function that duplicated 55 lines of endpoint registration code from `api.py`. Adding new endpoints required changes in two places.

**Solution**: Removed `create_openai_only_app()` entirely. `server.py` now imports and uses `app` from `api.py` directly - single source of truth.

**Key Changes**:
- Removed: `create_openai_only_app()` function (55 lines of duplicate registration)
- Added: `get_api_endpoints(app)` helper for dynamic endpoint discovery at startup
- Added: `_get_api_endpoints()` in `api.py` for dynamic root endpoint response
- Benefit: Adding new endpoints now only requires adding the decorator in `api.py` - no changes needed in `server.py`

**Impact**: Simplified from 308 to 260 lines in `server.py`. Endpoints are now single-source-of-truth.

### 2026-01-09: Model Capabilities Discovery

**Added**: `/v1/models` endpoint now returns `provider` and `capabilities` fields for each model.

**Configuration**: Models can define capabilities in `models.toml`:
```toml
[[models]]
id = "Qwen3-4B"
provider = "mlx"
capabilities = ["chat", "hidden_states", "thinking"]
enabled = true
```

**Purpose**: Allow clients to discover what features each model supports (chat, vision, hidden_states, thinking, etc.)

### 2026-01-09: Structured Hidden States Endpoint

**Added**: `/v1/hidden_states/structured` endpoint for extracting hidden states with:
- Server-side chat template application
- Token boundary tracking (system, user, think, assistant sections)
- Support for pre-filled thinking/assistant content
- Debugging capabilities with `return_formatted_prompt`

**Use Cases**: Z-Image embeddings, token attribution research, ablation studies, template debugging

**Limitations**: MLX models only, Qwen3-style chat templates recommended

## Server Architecture

### FastAPI Application

**File**: `src/heylook_llm/api.py`

**Key Components**:
```python
def create_app(
    models_config: str = "models.toml"
) -> FastAPI:
    """
    Factory function to create FastAPI app.

    Creates router, registers endpoints based on api_type,
    sets up CORS, and configures OpenAPI docs.
    """
```

**Middleware**:
- CORS enabled for all origins
- Request logging at DEBUG level
- Error handling with proper HTTP status codes

**Startup**:
```python
@app.on_event("startup")
async def startup_event():
    # Load models.yaml
    # Initialize router
    # Log available models
```

**Shutdown**:
```python
@app.on_event("shutdown")
async def shutdown_event():
    # Unload models
    # Clean up resources
```

### Thread Safety

All endpoints use fine-grained locking:
- Router handles model-level locking
- No global locks blocking concurrent requests
- Streaming responses don't block other requests

### Error Handling

Consistent error responses across all endpoints:
```json
{
  "error": {
    "message": "Detailed error message",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

HTTP Status Codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Model not found
- `500` - Internal server error

## OpenAI API Endpoints

### List Models

**Endpoint**: `GET /v1/models`

**Purpose**: List all enabled models

**Request**: None

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-2.5-3b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "heylookitsanllm",
      "permission": [],
      "root": "qwen-2.5-3b",
      "parent": null,
      "provider": "mlx",
      "capabilities": ["chat", "hidden_states", "thinking"]
    }
  ]
}
```

**Fields**:
- `provider` (string, optional) - Provider type: `"mlx"` or `"mlx_embedding"`
- `capabilities` (array, optional) - Model capabilities (e.g., `["chat", "vision", "hidden_states", "thinking"]`)

These fields are populated from the model's configuration in models.toml and allow clients to discover what features each model supports.

**Implementation**:
```python
@app.get("/v1/models")
async def list_models():
    models = [
        format_model_info(config)
        for model_id, config in router.models_config.items()
        if config.enabled
    ]
    return {"object": "list", "data": models}
```

### Chat Completions

**Endpoint**: `POST /v1/chat/completions`

**Purpose**: Generate chat completions with streaming or non-streaming

**Request**:
```json
{
  "model": "qwen-2.5-3b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.05,
  "repetition_penalty": 1.1
}
```

**Vision Request** (with images):
```json
{
  "model": "qwen-vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."
          }
        }
      ]
    }
  ]
}
```

**Non-Streaming Response**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen-2.5-3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  }
}
```

**Response with Logprobs** (when `logprobs=true`):
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen-2.5-3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!"
      },
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "token_id": 9906,
            "logprob": -0.5,
            "bytes": [72, 101, 108, 108, 111],
            "top_logprobs": [
              {"token": "Hello", "token_id": 9906, "logprob": -0.5, "bytes": [72, 101, 108, 108, 111]},
              {"token": "Hi", "token_id": 13347, "logprob": -1.2, "bytes": [72, 105]}
            ]
          },
          {
            "token": "!",
            "token_id": 0,
            "logprob": -0.1,
            "bytes": [33],
            "top_logprobs": [
              {"token": "!", "token_id": 0, "logprob": -0.1, "bytes": [33]},
              {"token": ".", "token_id": 13, "logprob": -2.5, "bytes": [46]}
            ]
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12
  }
}
```

**Logprobs Field Details**:
- `content` - Array of token-level logprobs (one per generated token)
- `token` - The decoded text of the token
- `token_id` - The token ID (extension: not in OpenAI API)
- `logprob` - Log probability of the selected token
- `bytes` - UTF-8 byte representation of the token
- `top_logprobs` - Array of top-k most likely alternative tokens (if `top_logprobs` > 0)

**Streaming Response** (Server-Sent Events):
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen-2.5-3b","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen-2.5-3b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen-2.5-3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Streaming Response with Logprobs** (when `logprobs=true`):
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen-2.5-3b","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"logprobs":{"content":[{"token":"Hello","token_id":9906,"logprob":-0.5,"bytes":[72,101,108,108,111],"top_logprobs":[{"token":"Hello","token_id":9906,"logprob":-0.5,"bytes":[72,101,108,108,111]},{"token":"Hi","token_id":13347,"logprob":-1.2,"bytes":[72,105]}]}]},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen-2.5-3b","choices":[{"index":0,"delta":{"content":"!"},"logprobs":{"content":[{"token":"!","token_id":0,"logprob":-0.1,"bytes":[33],"top_logprobs":[{"token":"!","token_id":0,"logprob":-0.1,"bytes":[33]}]}]},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen-2.5-3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Note: In streaming mode with logprobs, each chunk includes the logprobs for the token(s) in that chunk's delta.

**Implementation**:
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Extract images from message content
    images = extract_images_from_messages(request.messages)

    # Build prompt from messages
    prompt = build_prompt(request.messages)

    # Generate
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, prompt, images),
            media_type="text/event-stream"
        )
    else:
        # Collect full response
        full_text = ""
        for chunk in router.generate(
            model_id=request.model,
            prompt=prompt,
            images=images,
            **generation_kwargs
        ):
            full_text += chunk

        return format_completion_response(full_text)
```

**Image Handling**:
- Base64 images: `data:image/jpeg;base64,...`
- URL images: `http://...` or `https://...`
- Multiple images per message supported
- Images processed in parallel

**Logprobs Limitation for Vision**:
- Logprobs are only available for text-only requests (MLX-LM generation path)
- When images are included, the MLX-VLM generation path is used which does NOT support logprobs
- If `logprobs=true` is set on a vision request (with images), the logprobs field will be omitted from the response
- Text-only requests to VLM models DO support logprobs

**Parameters**:
- `model` (required) - Model ID from models.yaml
- `messages` (required) - Array of chat messages
- `temperature` (optional, default 0.7) - Sampling temperature (0-2)
- `max_tokens` (optional, default 512) - Max tokens to generate
- `stream` (optional, default false) - Enable streaming
- `top_p` (optional, default 1.0) - Nucleus sampling threshold
- `top_k` (optional, default 0) - Top-k sampling
- `min_p` (optional, default 0.0) - Min-p sampling
- `repetition_penalty` (optional, default 1.0) - Repetition penalty (1.0 = disabled)
- `stop` (optional) - Stop sequences
- `presence_penalty` (optional) - Presence penalty
- `frequency_penalty` (optional) - Frequency penalty
- `logprobs` (optional, default false) - Include log probabilities in response
- `top_logprobs` (optional, 0-20) - Number of most likely alternative tokens to return at each position

### Chat Completions Multipart

**Endpoint**: `POST /v1/chat/completions/multipart`

**Purpose**: Upload raw images as multipart form data (faster than base64)

**Request** (multipart/form-data):
```
POST /v1/chat/completions/multipart
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="model"

qwen-vl
------WebKitFormBoundary
Content-Disposition: form-data; name="messages"

[{"role": "user", "content": "What's in these images?"}]
------WebKitFormBoundary
Content-Disposition: form-data; name="images"; filename="image1.jpg"
Content-Type: image/jpeg

[binary image data]
------WebKitFormBoundary
Content-Disposition: form-data; name="images"; filename="image2.jpg"
Content-Type: image/jpeg

[binary image data]
------WebKitFormBoundary--
```

**Response**: Same as `/v1/chat/completions`

**Performance Benefit**:
- No base64 encoding/decoding overhead
- ~57ms faster per image on average
- Recommended for multiple images

**Implementation**:
```python
@app.post("/v1/chat/completions/multipart")
async def chat_completions_multipart(
    model: str = Form(...),
    messages: str = Form(...),
    images: List[UploadFile] = File(None),
    temperature: float = Form(0.7),
    max_tokens: int = Form(512),
    stream: bool = Form(False)
):
    # Parse messages JSON
    messages_list = json.loads(messages)

    # Load images from uploaded files
    image_objects = []
    if images:
        for img_file in images:
            contents = await img_file.read()
            image = Image.open(io.BytesIO(contents))
            image_objects.append(image)

    # Generate (same as regular endpoint)
    ...
```

### Embeddings

**Endpoint**: `POST /v1/embeddings`

**Purpose**: Generate embeddings for text

**Request**:
```json
{
  "model": "bge-small",
  "input": "The quick brown fox"
}
```

**Request** (batch):
```json
{
  "model": "bge-small",
  "input": [
    "First sentence",
    "Second sentence",
    "Third sentence"
  ]
}
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "bge-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Implementation**:
```python
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    # Handle both string and list input
    texts = [request.input] if isinstance(request.input, str) else request.input

    # Generate embeddings
    embeddings = router.embed(
        model_id=request.model,
        texts=texts
    )

    # Format response
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i
            }
            for i, emb in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts)
        }
    }
```

**Parameters**:
- `model` (required) - Embedding model ID
- `input` (required) - String or array of strings
- `encoding_format` (optional) - "float" or "base64"

### Hidden States Extraction

**Added**: 2026-01-09

#### Basic Hidden States

**Endpoint**: `POST /v1/hidden_states`

**Purpose**: Extract raw hidden states from a specific layer for encoder use cases

**Request**:
```json
{
  "input": "The quick brown fox jumps over the lazy dog",
  "model": "Qwen3-4B",
  "layer": -2,
  "max_length": 512,
  "encoding_format": "base64"
}
```

**Response**:
```json
{
  "hidden_states": "SGVsbG8gV29ybGQ=...",
  "shape": [120, 2560],
  "model": "Qwen3-4B",
  "layer": -2,
  "dtype": "bfloat16",
  "encoding_format": "base64"
}
```

**Parameters**:
- `input` (required) - Text or array of texts (with chat template already applied)
- `model` (required) - Model ID
- `layer` (optional, default -2) - Layer to extract from
- `max_length` (optional, default 512) - Max sequence length
- `encoding_format` (optional, default "float") - Output format: "float" or "base64"
- `return_attention_mask` (optional, default false) - Include attention mask

**Limitations**:
- MLX models only (llama.cpp not supported)

#### Structured Hidden States

**Endpoint**: `POST /v1/hidden_states/structured`

**Purpose**: Extract hidden states with server-side chat template and token boundary tracking

**Request**:
```json
{
  "model": "Qwen3-4B",
  "user_prompt": "What is the capital of France?",
  "system_prompt": "You are a helpful geography assistant.",
  "thinking_content": "Let me recall... France's capital is Paris.",
  "enable_thinking": true,
  "layer": -2,
  "max_length": 512,
  "encoding_format": "base64",
  "return_token_boundaries": true,
  "return_formatted_prompt": false
}
```

**Response**:
```json
{
  "hidden_states": "SGVsbG8gV29ybGQ=...",
  "shape": [120, 2560],
  "model": "Qwen3-4B",
  "layer": -2,
  "dtype": "bfloat16",
  "encoding_format": "base64",
  "token_boundaries": {
    "system": {"start": 0, "end": 35},
    "user": {"start": 35, "end": 80},
    "think": {"start": 80, "end": 110},
    "assistant": {"start": 110, "end": 120}
  },
  "token_counts": {
    "system": 35,
    "user": 45,
    "think": 30,
    "assistant": 10,
    "total": 120
  }
}
```

**Parameters**:
- `model` (required) - Model ID
- `user_prompt` (required) - User message content
- `system_prompt` (optional) - System prompt content
- `thinking_content` (optional) - Pre-filled thinking block
- `assistant_content` (optional) - Pre-filled assistant response
- `enable_thinking` (optional, default true) - Enable thinking mode in template
- `layer` (optional, default -2) - Layer to extract from
- `max_length` (optional, default 512) - Max sequence length
- `encoding_format` (optional, default "float") - Output format: "float" or "base64"
- `return_token_boundaries` (optional, default false) - Include token boundaries
- `return_formatted_prompt` (optional, default false) - Include formatted prompt string

**Key Features**:
- Server applies chat template internally (Qwen3 format)
- Token boundary tracking for each section (system, user, think, assistant)
- Supports pre-filled thinking and assistant content
- Useful for Z-Image embeddings, ablation studies, template debugging

**Use Cases**:
- Z-Image text encoder embeddings
- Token attribution research
- Ablation studies on prompt sections
- Debugging chat template formatting

**Limitations**:
- MLX models only (llama.cpp not supported)
- Qwen3-style chat templates recommended for thinking support

**Implementation**:
```python
@app.post("/v1/hidden_states/structured")
async def extract_structured_hidden_states(
    request: Request,
    structured_request: dict = Body(...)
):
    from heylook_llm.hidden_states import (
        StructuredHiddenStatesRequest,
        create_structured_hidden_states,
    )

    req = StructuredHiddenStatesRequest(**structured_request)
    router = request.app.state.router_instance
    response = await create_structured_hidden_states(req, router)
    return response.model_dump(exclude_none=True)
```

### Batch Chat Completions

**Endpoint**: `POST /v1/batch/chat/completions`

**Purpose**: Process multiple prompts in parallel for 2-4x throughput

**Request**:
```json
{
  "model": "qwen-2.5-3b",
  "messages": [
    [{"role": "user", "content": "Write a haiku about coding"}],
    [{"role": "user", "content": "Write a limerick about AI"}],
    [{"role": "user", "content": "Write a sonnet about robots"}]
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "id": "batch-123",
  "object": "batch.completion",
  "created": 1700000000,
  "model": "qwen-2.5-3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Code flows like stream\nBugs hide in morning mist, wait\nDebugger finds truth"
      },
      "finish_reason": "stop"
    },
    {
      "index": 1,
      "message": {
        "role": "assistant",
        "content": "There once was an AI so bright..."
      },
      "finish_reason": "stop"
    },
    {
      "index": 2,
      "message": {
        "role": "assistant",
        "content": "When robots dream of electric sheep..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

**Implementation**:
```python
@app.post("/v1/batch/chat/completions")
async def batch_chat_completions(request: BatchChatCompletionRequest):
    # Build prompts from message arrays
    prompts = [
        build_prompt(messages)
        for messages in request.messages
    ]

    # Batch generate
    responses = router.batch_generate(
        model_id=request.model,
        prompts=prompts,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Format response
    return {
        "id": f"batch-{uuid.uuid4()}",
        "object": "batch.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }
            for i, text in enumerate(responses)
        ]
    }
```

**Limitations**:
- Text-only models (no vision in batch mode)
- All prompts use same max_tokens
- Non-streaming only
- MLX provider only (not llama.cpp)

**Performance**:
- 2 prompts: ~2x faster
- 4 prompts: ~3.5x faster
- 8 prompts: ~4x faster

### RLM Completions

**Endpoint**: `POST /v1/rlm/completions`

**Purpose**: Recursive Language Model inference. The model writes Python code to iteratively explore, slice, and transform a large context, calling itself recursively via `llm_query()` as needed.

**Reference**: Zhang, Kraska, Khattab -- arxiv 2512.24601v2

**Request**:
```json
{
  "model": "qwen-2.5-3b",
  "context": "Alice has 3 cats. Bob has 5 dogs. Carol has 2 fish.",
  "query": "How many animals total?",
  "system": "Be precise and show your work.",
  "max_iterations": 10,
  "max_tokens": 2048,
  "stream": false,
  "sandbox": true,
  "timeout": 30,
  "sub_model": null,
  "sub_max_tokens": null,
  "sub_temperature": null,
  "sub_top_p": null,
  "compaction": false,
  "compaction_threshold": 0.8,
  "max_context_tokens": 8192,
  "max_depth": 1,
  "max_errors": null,
  "max_timeout": null
}
```

**Response**:
```json
{
  "id": "rlm-a1b2c3d4e5f6",
  "object": "rlm.completion",
  "created": 1700000000,
  "model": "qwen-2.5-3b",
  "answer": "There are 10 animals total: 3 cats + 5 dogs + 2 fish = 10.",
  "finish_reason": "final",
  "usage": {
    "prompt_tokens": 250,
    "completion_tokens": 120,
    "total_tokens": 370
  },
  "rlm": {
    "iterations": 2,
    "sub_queries": 0,
    "compactions": 0,
    "child_traces": null,
    "trace": [
      {"iteration": 1, "code_len": 45, "stdout_len": 12, "stderr_len": 0, "action": null},
      {"iteration": 2, "code_len": 30, "stdout_len": 0, "stderr_len": 0, "action": "FINAL"}
    ]
  }
}
```

**Streaming** (SSE, `stream: true`):
```
event: rlm_start
data: {"id": "rlm-xxx", "model": "...", "context_length": 12345}

event: iteration_start
data: {"iteration": 1}

event: assistant_response
data: {"iteration": 1, "text": "Let me explore...\n```repl\nprint(len(context))\n```"}

event: code_output
data: {"iteration": 1, "stdout": "12345\n", "stderr": "", "code_len": 20}

event: compaction
data: {"iteration": 8, "compaction_count": 1}

event: rlm_complete
data: {"answer": "...", "finish_reason": "final", "usage": {...}, "rlm": {...}}
```

**Parameters**:
- `model` (required) - Model ID
- `context` (required) - The large input text loaded into REPL as `context` variable
- `query` (required) - The task/question to answer
- `system` (optional) - Additional system instructions
- `max_iterations` (optional, default 10, 1-50) - Max REPL loop iterations
- `max_tokens` (optional, default 2048) - Per-iteration generation limit
- `temperature`, `top_p` (optional) - Sampling parameters
- `stream` (optional, default false) - Stream iteration events via SSE
- `sandbox` (optional, default true) - Restrict builtins, block dunder attribute access
- `timeout` (optional, default 30, 1-120) - Per-execution timeout in seconds
- `max_output_chars` (optional, default 10000) - stdout/stderr truncation per iteration
- `sub_model` (optional) - Different model for `llm_query()` sub-calls
- `sub_max_tokens`, `sub_temperature`, `sub_top_p` (optional) - Override params for sub-calls
- `compaction` (optional, default false) - Summarize message history when context fills up
- `compaction_threshold` (optional, default 0.8, 0.5-0.95) - Compact at this fraction of `max_context_tokens`
- `max_context_tokens` (optional, default 8192) - Model context window size for compaction threshold
- `max_depth` (optional, default 1, 1-5) - Recursive depth. 1 = no recursion. 2+ enables `rlm_query()` child RLMs
- `max_errors` (optional, default null, 1-20) - Stop after N consecutive code execution errors
- `max_timeout` (optional, default null, >= 1.0) - Wall-clock timeout for the entire RLM loop in seconds. Checked per-iteration, not mid-execution

**Finish reasons**: `"final"` (FINAL/FINAL_VAR called), `"direct_response"` (no code block), `"max_iterations"` (loop exhausted), `"error_threshold"` (consecutive errors hit `max_errors`), `"timeout"` (wall-clock limit hit `max_timeout`)

**Sandbox**: safe builtins whitelist + AST-level dunder attribute blocking + thread-based timeout + stdout/stderr truncation

**Architecture**: Calls `provider.create_chat_completion()` directly (no HTTP round-trip). Model pinned for loop duration. Streaming uses thread pool executor with async queue bridge.

**Implementation**: `src/heylook_llm/rlm.py`

## Admin Endpoints

Admin endpoints live on three routers in `src/heylook_llm/admin_api.py`
(`admin_router`, `admin_ops_router`, `scan_import_router`). This section
covers the high-traffic ones; read `admin_api.py` for the full surface.

### Reload Configuration

**Endpoint**: `POST /v1/admin/reload`

**Purpose**: Hot-reload `models.toml` without restarting the server process.

**Request**: None

**Response**:
```json
{
  "status": "reloaded",
  "models_count": 5
}
```

Implemented on `admin_ops_router` in `admin_api.py`. No restart endpoint
exists -- use process manager (`launchctl`, `systemctl`) or Ctrl-C + restart
if you actually need to restart the server.

### Model Management (`/v1/admin/models/*`)

Full CRUD + scan-and-import surface in `admin_api.py`:
- `GET /v1/admin/models` -- list all configured models with status
- `POST /v1/admin/models` -- create or overwrite a model config
- `GET /v1/admin/models/{model_id}` -- read a single model
- `PATCH /v1/admin/models/{model_id}` -- partial update
- `DELETE /v1/admin/models/{model_id}` -- remove from `models.toml`
- `POST /v1/admin/models/{model_id}/enable` / `/disable` -- toggle without deleting
- `POST /v1/admin/models/{model_id}/load` / `/unload` -- control LRU residency
- `POST /v1/admin/models/scan` -- walk a directory / HF cache for importable models
- `POST /v1/admin/models/import` -- import selected models into `models.toml`
- `POST /v1/admin/models/validate` -- dry-run a config

### Conversation Storage (`/v1/conversations/*`)

Full CRUD + message append/edit/truncate for the SQLite-backed conversation
store in `conversation_api.py`. Consumed by the web frontends; shrug-prompter
and CLI clients typically don't touch it.

### Notebook Storage (`/v1/notebooks/*`)

Notebook CRUD in `notebook_api.py` (web-frontend text scratchpads, separate
from conversations).

### Preset Storage (`/v1/presets/*`)

**Added**: v1.34.22

CRUD for user-saved chat presets in `preset_api.py`, tag "Presets" (the
server's 8th API router). Backed by the `presets` table in the DuckDB
store (`db.py`).

- `GET /v1/presets` -- list all saved presets, ordered by name
- `POST /v1/presets` -- create a preset (`name`, `system_prompt`, `params`)
- `PUT /v1/presets/{id}` -- update a preset (only set fields are patched)
- `DELETE /v1/presets/{id}` -- delete a preset

Errors: `409` name collision (`PresetNameTaken`), `400` bad/empty fields,
`404` unknown id.

A preset is a UI-authored bundle (`{name, system_prompt, params}`) that
the v3 frontend expands CLIENT-side into explicit request fields when
applied -- deliberately distinct from the server-side TOML sampler preset
registry (`presets.py` / `ChatRequest.preset`, see [config.md](./config.md)).
No wire relationship between the two. Excluded from
`clear_all_data`/`POST /v1/data/clear` (config, not data). Wire contract:
[docs/frontend_v3_spec.md](../frontend_v3_spec.md) §4.

### Messages API (`/v1/messages`)

Anthropic Messages-inspired endpoint in `messages_api.py`. Wraps
`/v1/chat/completions` with typed content blocks.

### Data Management

- `POST /v1/data/clear` (in `api.py`) -- drops every conversation, message,
  and notebook from the SQLite store. Irreversible; guarded by a confirm
  step in the frontend.

### Cache Inspection

- `GET /v1/cache/list` (in `api.py`) -- enumerate prompt caches currently
  held in the router's radix tree, per loaded model.
- `POST /v1/cache/clear` (in `api.py`) -- invalidate caches for a specific
  model or all models.

### Performance Analytics

- `GET /v1/system/metrics` (in `api.py`) -- system + per-model resource
  metrics (RAM, CPU, memory per model, context used).
- `GET /v1/performance/profile/{time_range}` (in `api.py`) -- aggregated
  performance analytics over a `1h`/`6h`/`24h`/`7d` window. Backed by the
  in-memory `PerfCollector` ring buffer; see [observability guide](../observability_guide.md)
  for the disk-backed JSONL streams that complement it.

## Interactive Documentation

### Swagger UI

**URL**: `http://localhost:8080/docs`

**Features**:
- Interactive API explorer
- Try endpoints directly in browser
- Auto-generated from OpenAPI schema
- Request/response examples

### ReDoc

**URL**: `http://localhost:8080/redoc`

**Features**:
- Clean documentation layout
- Code samples in multiple languages
- Searchable
- Print-friendly

### OpenAPI Schema

**URL**: `http://localhost:8080/openapi.json`

**Purpose**: Machine-readable API specification for code generation

## Request Processing Flow

### Standard Flow (Non-Streaming)

```
1. Client sends POST /v1/chat/completions
   ↓
2. FastAPI validates request against ChatCompletionRequest schema
   ↓
3. Extract images from message content (if any)
   ↓
4. Build prompt from messages array
   ↓
5. Call router.generate(model_id, prompt, images, **kwargs)
   ↓
6. Router loads model (or retrieves from cache)
   ↓
7. Provider generates tokens
   ↓
8. Collect all tokens into full response
   ↓
9. Format as ChatCompletionResponse
   ↓
10. Return JSON to client
```

### Streaming Flow

```
1. Client sends POST /v1/chat/completions with stream=true
   ↓
2. FastAPI validates request
   ↓
3. Extract images and build prompt
   ↓
4. Return StreamingResponse immediately
   ↓
5. Generator function yields tokens as they're generated:
   ↓
6. For each token from router.generate():
   ↓
7. Format as SSE chunk
   ↓
8. Yield "data: {json}\n\n"
   ↓
9. Client receives and displays token immediately
   ↓
10. After last token, yield "data: [DONE]\n\n"
```

### Vision Request Flow

```
1. Client sends request with images in message content
   ↓
2. extract_images_from_messages() finds all image_url entries
   ↓
3. For each image:
   ↓
4. If base64: decode to PIL Image
   ↓
5. If URL: download and load to PIL Image
   ↓
6. Images processed in parallel (ThreadPoolExecutor)
   ↓
7. Build prompt (may include special image tokens like <image>)
   ↓
8. Call router.generate(model_id, prompt, images=[Image, Image, ...])
   ↓
9. Router checks model_type == "vision"
   ↓
10. VLM provider processes images + text together
   ↓
11. Stream or return response
```

### Batch Request Flow

```
1. Client sends POST /v1/batch/chat/completions with array of message arrays
   ↓
2. Extract N prompts from N message arrays
   ↓
3. Call router.batch_generate(model_id, prompts=[...])
   ↓
4. Router loads model
   ↓
5. MLXBatchProcessor tokenizes all prompts
   ↓
6. Pad all to same length
   ↓
7. Stack into batch tensor [N, max_len]
   ↓
8. Run model inference in parallel for all N prompts
   ↓
9. Decode N output sequences
   ↓
10. Return array of N responses
   ↓
11. Format as BatchChatCompletionResponse
```

## Error Handling

### Model Not Found

```python
if model_id not in router.models_config:
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' not found"
    )
```

Response:
```json
{
  "detail": "Model 'unknown-model' not found"
}
```

### Invalid Parameters

```python
# Pydantic validation catches invalid types
class ChatCompletionRequest(BaseModel):
    temperature: float = Field(ge=0, le=2)  # Must be 0-2
```

Response for invalid value:
```json
{
  "detail": [
    {
      "loc": ["body", "temperature"],
      "msg": "ensure this value is less than or equal to 2",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### Model Loading Errors

```python
try:
    provider = router.load_model(model_id)
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Failed to load model: {str(e)}"
    )
```

### Image Loading Errors

```python
try:
    image = load_image_from_url(url)
except Exception as e:
    raise HTTPException(
        status_code=400,
        detail=f"Failed to load image from {url}: {str(e)}"
    )
```

## Performance Optimizations

### Image Transfer

**Problem**: Base64 encoding adds ~33% overhead

**Solution**: Multipart endpoint (`/v1/chat/completions/multipart`)

**Benefit**: ~57ms faster per image

### Parallel Image Loading

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    images = list(executor.map(load_image, image_urls))
```

**Benefit**: Load multiple images concurrently

### Streaming Responses

**Problem**: Long generations block client waiting for full response

**Solution**: Server-Sent Events streaming

**Benefit**: Client sees tokens immediately, better UX

### LRU Model Cache

**Problem**: Loading models takes 2-30 seconds

**Solution**: Keep 2 most recently used models in memory

**Benefit**: Near-instant response for cached models

### Batch Processing

**Problem**: Sequential requests have overhead per request

**Solution**: Process multiple prompts in single forward pass

**Benefit**: 2-4x throughput improvement

## Security Considerations

### Input Validation

- All requests validated with Pydantic schemas
- Type checking enforced
- Range checks on numeric parameters (temperature, max_tokens)

### Image URL Validation

```python
def load_image_from_url(url: str) -> Image:
    # Only allow http/https
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL scheme")

    # Set timeout to prevent hanging
    response = requests.get(url, timeout=10)

    # Check content type
    if 'image' not in response.headers.get('content-type', ''):
        raise ValueError("URL does not point to an image")

    # Limit image size
    if len(response.content) > 10 * 1024 * 1024:  # 10MB
        raise ValueError("Image too large")

    return Image.open(io.BytesIO(response.content))
```

### File Upload Limits

```python
# FastAPI config
app = FastAPI(
    max_upload_size=50 * 1024 * 1024  # 50MB
)
```

### No Authentication by Default

This is a local deployment tool. For production:
- Add API key authentication
- Use HTTPS
- Implement rate limiting
- Add request logging

## Testing Endpoints

### Using curl

**List models**:
```bash
curl http://localhost:8080/v1/models
```

**Chat completion**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Streaming chat**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

**Chat with logprobs**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "logprobs": true,
    "top_logprobs": 5,
    "max_tokens": 20
  }'
```

**Vision request**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What'\''s in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }]
  }'
```

**Embeddings**:
```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-small",
    "input": "The quick brown fox"
  }'
```

**Batch completions**:
```bash
curl http://localhost:8080/v1/batch/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [
      [{"role": "user", "content": "Write a haiku"}],
      [{"role": "user", "content": "Write a limerick"}]
    ],
    "max_tokens": 256
  }'
```

### Using Python

**OpenAI SDK**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # Not used, but required by SDK
)

# Chat
response = client.chat.completions.create(
    model="qwen-2.5-3b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen-2.5-3b",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

# Embeddings
response = client.embeddings.create(
    model="bge-small",
    input="The quick brown fox"
)
print(response.data[0].embedding)
```

**Requests**:
```python
import requests

# Chat
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "qwen-2.5-3b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())

# Streaming
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "qwen-2.5-3b",
        "messages": [{"role": "user", "content": "Count to 10"}],
        "stream": True
    },
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode())
```

## Endpoint Summary Table

### Core Endpoints

| Endpoint | Method | Purpose | Streaming | Vision | Logprobs | Auth |
|----------|--------|---------|-----------|--------|----------|------|
| `/v1/models` | GET | List models | No | N/A | N/A | No |
| `/v1/chat/completions` | POST | Chat generation | Optional | Yes | Yes (text-only) | No |
| `/v1/chat/completions/multipart` | POST | Chat with raw images | Optional | Yes | Yes (text-only) | No |
| `/v1/embeddings` | POST | Generate embeddings | No | No | No | No |
| `/v1/hidden_states` | POST | Extract hidden states | No | No | No | No |
| `/v1/hidden_states/structured` | POST | Structured hidden states with token boundaries | No | No | No | No |
| `/v1/rlm/completions` | POST | RLM recursive inference | Optional (SSE) | No | No | No |
| `/v1/capabilities` | GET | Server capabilities | No | N/A | N/A | No |

### Batch Endpoints

| Endpoint | Method | Purpose | Notes |
|----------|--------|---------|-------|
| `/v1/batch/chat/completions` | POST | Batch text generation | Text-only MLX, no streaming |

### Admin Endpoints

See the [Admin Endpoints](#admin-endpoints) section above for the full list.
High-traffic:

| Endpoint | Method | Purpose | Notes |
|----------|--------|---------|-------|
| `/v1/admin/reload` | POST | Reload config | Hot-reload `models.toml` |
| `/v1/admin/models` | GET | List all model configs | Admin router |
| `/v1/admin/models/{id}` | GET/PATCH/DELETE | Per-model CRUD | Admin router |
| `/v1/admin/models/{id}/enable` / `/disable` | POST | Toggle without deleting | Admin router |
| `/v1/admin/models/{id}/load` / `/unload` | POST | Control LRU residency | Admin router |
| `/v1/admin/models/scan` | POST | Discover importable models | Scan/import router |
| `/v1/admin/models/import` | POST | Import scanned models | Scan/import router |

### Data & Storage Endpoints

| Endpoint | Method | Purpose | Notes |
|----------|--------|---------|-------|
| `POST /v1/data/clear` | POST | Drop all conversations + notebooks | Irreversible; presets NOT included |
| `/v1/conversations/*` | CRUD | Conversation storage | See `conversation_api.py` |
| `/v1/notebooks/*` | CRUD | Notebook storage | See `notebook_api.py` |
| `/v1/presets/*` | CRUD | User-saved chat presets (v3) | See `preset_api.py`; distinct from TOML sampler presets |
| `/v1/messages` | POST | Anthropic Messages API | Wraps `/v1/chat/completions` |

### Cache & Metrics Endpoints

| Endpoint | Method | Purpose | Notes |
|----------|--------|---------|-------|
| `/v1/cache/list` | GET | Enumerate prompt caches | Per loaded model |
| `/v1/cache/clear` | POST | Invalidate prompt caches | Per model or all |
| `/v1/system/metrics` | GET | System + per-model metrics | RAM, CPU, context used |
| `/v1/performance/profile/{range}` | GET | Aggregated perf analytics | 1h / 6h / 24h / 7d |

### Interpretability Endpoints (`/v1/jspace/*`)

Jacobian-lens ("j-space") read-out of a model's verbalizable workspace. See the
[j-space guide](../jspace_guide.md) for the full response shape and setup.

| Endpoint | Method | Purpose | Notes |
|----------|--------|---------|-------|
| `/v1/jspace/models` | GET | Models with a fitted lens | Lens files under `adapters/jspace/<model_id>/` |
| `/v1/jspace/analyze` | POST | Per-layer workspace + hallucination-risk | Lens-gated; pins the model + runs under the gen gate on a pinned MLX thread |

### Documentation Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |
| `/openapi.json` | GET | OpenAPI schema |

## Key Design Decisions

### Why OpenAI-Compatible API?

**Industry standard**: Widely supported by tools, libraries, and clients

**Broad compatibility**: Works with existing OpenAI SDKs and integrations

**Well-documented**: Extensive documentation and community support

### Why Server-Sent Events for Streaming?

- Native browser support
- Simple implementation
- No WebSocket complexity
- Works through proxies

### Why LRU Cache?

- Balance between memory usage and performance
- Most users switch between 1-2 models frequently
- `max_loaded_models` configurable in models.toml
- Easy to evict when needed

### Why Fine-Grained Locking?

- Allow concurrent loading of different models
- Prevent duplicate loads of same model
- Better performance than global lock
- More complex but worth it

### Why Multipart Endpoint?

- Base64 encoding is slow (~57ms overhead per image)
- Raw binary is faster
- Optional - base64 still works
- Better for batch image processing

## Future Enhancements

Single-user server, so classic "authentication / rate-limiting / multi-tenant"
items are not in scope. A focused list of things still worth doing:

- **Optional admin-endpoint token** (`HEYLOOK_ADMIN_TOKEN` env var gating
  `/v1/admin/*` + `/v1/data/clear`). Defense-in-depth for LAN deployments;
  tracked in the Slice 1.6 LAN hardening mini-slice.
- **Streaming batch** -- stream tokens from `/v1/batch/chat/completions`.
  Today the whole batch collects before responding.
- **Batch logprobs** -- logprobs currently require the single-request path.
- **Vision logprobs** -- blocked on upstream `mlx-vlm` exposing them.
- **Partial / resumable responses** -- resume generation from a checkpoint
  after a client disconnect.

Already shipped (don't re-add as planned):
- Request logging (in-memory `PerfCollector` + the three JSONL streams
  documented in [observability guide](../observability_guide.md))
- Model metrics (`/v1/system/metrics` + `/v1/performance/profile/*`)
- Model warmup (S1.4 `provider.warmup()` runs on every load)
- Batch vision (standalone client at `apps/batch-labeler/`)
- Logprobs (see [logprobs.md](../../internal/backend/logprobs.md))

## Related Documentation

- [router.md](./router.md) - Model routing and caching
- [config.md](./config.md) - Configuration system
- [overview.md](./overview.md) - Backend architecture overview
- [frontend_v3_spec.md](../frontend_v3_spec.md) §4 - the authoritative frontend/backend API contract
