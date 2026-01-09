# Frontend Developer Handoff Guide

A comprehensive guide for building a React + Vite + Tailwind frontend that leverages all heylookitsanllm API features.

**Target Audience:** Developers / Power Users
**Tech Stack:** React + Vite + Tailwind CSS
**API Compatibility:** OpenAI-compatible with extensions

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [API Endpoints Reference](#2-api-endpoints-reference)
3. [Sampler Parameters](#3-sampler-parameters)
4. [Feature Implementation Guides](#4-feature-implementation-guides)
5. [TypeScript Types](#5-typescript-types)
6. [React Patterns](#6-react-patterns)
7. [UX Recommendations](#7-ux-recommendations)
8. [Performance Optimizations](#8-performance-optimizations)
9. [Error Handling](#9-error-handling)
10. [Mobile/Responsive Design](#10-mobileresponsive-design)

---

## 1. Quick Start

### Server Connection

Default server URL: `http://localhost:8080`

```typescript
const BASE_URL = 'http://localhost:8080';

// Test connection
async function testConnection(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE_URL}/v1/models`);
    return response.ok;
  } catch {
    return false;
  }
}
```

### Capability Detection

Query server capabilities on startup to enable/disable features:

```typescript
async function getCapabilities() {
  const response = await fetch(`${BASE_URL}/v1/capabilities`);
  return response.json();
}

// Response includes:
// - server_version
// - optimizations (orjson, turbojpeg, xxhash availability)
// - features (streaming, vision_models, model_caching)
// - limits (max_tokens, max_images_per_request)
// - recommendations (use_multipart, optimal_batch_size)
```

### List Available Models

```typescript
async function getModels(): Promise<Model[]> {
  const response = await fetch(`${BASE_URL}/v1/models`);
  const data = await response.json();
  return data.data || [];
}

// Response now includes provider and capabilities:
// {
//   "object": "list",
//   "data": [
//     {
//       "id": "Qwen3-4B",
//       "object": "model",
//       "owned_by": "user",
//       "provider": "mlx",
//       "capabilities": ["chat", "hidden_states", "thinking"]
//     },
//     {
//       "id": "qwen-vl-chat",
//       "object": "model",
//       "owned_by": "user",
//       "provider": "mlx",
//       "capabilities": ["chat", "vision", "hidden_states"]
//     }
//   ]
// }
```

---

## 2. API Endpoints Reference

### Core Endpoints

| Endpoint | Method | Purpose | Streaming | Vision | Auth |
|----------|--------|---------|-----------|--------|------|
| `/v1/models` | GET | List available models | No | N/A | No |
| `/v1/chat/completions` | POST | Chat generation | Yes | Yes | No |
| `/v1/chat/completions/multipart` | POST | Fast vision upload | Yes | Yes | No |
| `/v1/embeddings` | POST | Generate embeddings | No | No | No |
| `/v1/batch/chat/completions` | POST | Batch text generation | No | No | No |
| `/v1/capabilities` | GET | Server features/limits | No | N/A | No |
| `/v1/audio/transcriptions` | POST | Speech-to-text (macOS) | No | No | No |
| `/v1/hidden_states` | POST | Extract hidden states | No | No | No |
| `/v1/hidden_states/structured` | POST | Structured hidden states with token boundaries | No | No | No |
| `/v1/admin/reload` | POST | Hot-reload model config | No | N/A | No |

### Interactive API Documentation

When the server is running:
- **Swagger UI:** `http://localhost:8080/docs`
- **ReDoc:** `http://localhost:8080/redoc`
- **OpenAPI Schema:** `http://localhost:8080/openapi.json`

---

## 3. Sampler Parameters

### All Tweakable Parameters

These parameters can be passed to `/v1/chat/completions`:

| Parameter | Type | Range | Default | Description | UI Control |
|-----------|------|-------|---------|-------------|------------|
| `temperature` | float | 0-2 | 0.7 | Sampling randomness | Slider |
| `max_tokens` | int | 1-8192+ | 512 | Max generation length | Number input |
| `top_p` | float | 0-1 | 0.9 | Nucleus sampling threshold | Slider |
| `top_k` | int | 0-100+ | 0 | Top-k sampling (0=disabled) | Number input |
| `min_p` | float | 0-1 | 0.0 | Min-p sampling threshold | Slider |
| `repetition_penalty` | float | 0.1-2.0 | 1.1 | Penalize repeated tokens | Slider |
| `repetition_context_size` | int | 1-100+ | 20 | Context window for penalty | Number input |
| `presence_penalty` | float | 0-2 | 0.0 | Discourage token reuse | Slider |
| `frequency_penalty` | float | 0-2 | 0.0 | Penalize by frequency | Slider |
| `seed` | int | any | random | Reproducible generation | Number input |
| `stop` | string[] | - | [] | Stop sequences | Text input |

### Boolean Options

| Parameter | Default | Description | UI Control |
|-----------|---------|-------------|------------|
| `stream` | false | Enable streaming response | Toggle |
| `logprobs` | false | Return token probabilities | Toggle |
| `enable_thinking` | false | Qwen3 thinking mode | Toggle |

### Logprobs Options

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `logprobs` | bool | - | false | Include log probabilities |
| `top_logprobs` | int | 0-20 | 0 | Number of top alternatives per token |

### Streaming Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream` | bool | false | Enable SSE streaming |
| `stream_options.include_usage` | bool | false | Include token counts in final chunk |

### Image Processing Options (Multipart Endpoint)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resize_max` | int | none | Max dimension (maintains aspect ratio) |
| `resize_width` | int | none | Target width |
| `resize_height` | int | none | Target height |
| `image_quality` | int | 85 | JPEG quality (1-100) |
| `preserve_alpha` | bool | false | Keep transparency (outputs PNG) |

### Recommended Default Configuration

```typescript
const DEFAULT_PARAMETERS = {
  // Core sampling
  temperature: 0.7,
  max_tokens: 2048,
  top_p: 0.9,
  top_k: 0,
  min_p: 0.0,
  repetition_penalty: 1.1,
  repetition_context_size: 20,

  // Penalties
  presence_penalty: 0.0,
  frequency_penalty: 0.0,

  // Processing
  stream: true,

  // Image (for vision models)
  image_quality: 85,
  preserve_alpha: false,
};
```

---

## 4. Feature Implementation Guides

### A. Streaming Chat (SSE)

Server-Sent Events provide real-time token delivery.

```typescript
async function streamChat(
  messages: Message[],
  model: string,
  parameters: ModelParameters,
  onToken: (token: string) => void,
  onComplete: (usage?: Usage) => void,
  onError: (error: Error) => void
): Promise<void> {
  try {
    const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        stream_options: { include_usage: true }, // Get token counts
        ...parameters
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('No response body for streaming');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();

          if (data === '[DONE]') {
            onComplete();
            return;
          }

          try {
            const parsed = JSON.parse(data);

            // Handle content
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) {
              onToken(content);
            }

            // Handle thinking (Qwen3)
            const thinking = parsed.choices?.[0]?.delta?.thinking;
            if (thinking) {
              // You may want a separate callback for thinking
              onToken(`[THINKING] ${thinking}`);
            }

            // Handle usage stats (final chunk)
            if (parsed.usage) {
              onComplete(parsed.usage);
              return;
            }
          } catch (e) {
            // Skip invalid JSON chunks
          }
        }
      }
    }
  } catch (error) {
    onError(error as Error);
  }
}
```

### B. Non-Streaming Chat

```typescript
async function chat(
  messages: Message[],
  model: string,
  parameters: ModelParameters
): Promise<ChatResponse> {
  const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      ...parameters
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
```

### C. Vision / Image Upload

#### Method 1: Base64 (Standard)

Works everywhere, ~33% overhead from encoding.

```typescript
async function chatWithImageBase64(
  prompt: string,
  imageFile: File,
  model: string,
  parameters: ModelParameters
): Promise<ChatResponse> {
  // Convert file to base64
  const base64 = await fileToBase64(imageFile);
  const mimeType = imageFile.type || 'image/jpeg';

  const messages = [{
    role: 'user',
    content: [
      { type: 'text', text: prompt },
      {
        type: 'image_url',
        image_url: {
          url: `data:${mimeType};base64,${base64}`
        }
      }
    ]
  }];

  return chat(messages, model, parameters);
}

// Helper function
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
```

#### Method 2: Multipart (Recommended)

57ms faster per image, 33% less bandwidth.

```typescript
async function chatWithImageMultipart(
  prompt: string,
  imageFiles: File[],
  model: string,
  parameters: ModelParameters
): Promise<ChatResponse> {
  const formData = new FormData();

  // Build messages with placeholders
  const content: MessageContent[] = [
    { type: 'text', text: prompt }
  ];

  imageFiles.forEach(() => {
    content.push({
      type: 'image_url',
      image_url: { url: '__RAW_IMAGE__' }
    });
  });

  const messages = [{ role: 'user', content }];

  // Add form fields
  formData.append('model', model);
  formData.append('messages', JSON.stringify(messages));

  // Add parameters
  Object.entries(parameters).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, String(value));
    }
  });

  // Add image files
  imageFiles.forEach((file, index) => {
    formData.append('images', file, `image-${index}.jpg`);
  });

  // Optional: resize on server
  formData.append('resize_max', '1024');
  formData.append('image_quality', '85');

  const response = await fetch(`${BASE_URL}/v1/chat/completions/multipart`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
```

### D. Batch Processing

Process multiple independent prompts in parallel for 2-4x throughput.

```typescript
async function batchChat(
  prompts: string[],
  model: string,
  parameters: ModelParameters
): Promise<BatchResponse> {
  // Each prompt becomes a separate message array
  const messages = prompts.map(prompt => [
    { role: 'user', content: prompt }
  ]);

  const response = await fetch(`${BASE_URL}/v1/batch/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      ...parameters
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Response structure:
// {
//   id: "batch-xxx",
//   object: "batch.completion",
//   choices: [
//     { index: 0, message: { role: "assistant", content: "Response 1" } },
//     { index: 1, message: { role: "assistant", content: "Response 2" } },
//     ...
//   ]
// }
```

**Limitations:**
- Text-only (no vision support in batch mode)
- Non-streaming only
- MLX provider only

### E. Thinking Mode (Qwen3)

Qwen3 models can output reasoning in `<think>...</think>` blocks before the response.

```typescript
// Enable thinking mode
const parameters = {
  enable_thinking: true,
  // Recommended settings for thinking mode:
  temperature: 0.6,    // Don't use 0 - causes repetition loops
  top_p: 0.95,
  top_k: 20,
  presence_penalty: 1.5,  // Reduce repetition in reasoning
};

// Non-streaming response
const response = await chat(messages, model, parameters);
const message = response.choices[0].message;

// Access thinking separately from content
const thinking = message.thinking;  // Reasoning process
const content = message.content;    // Final answer

// Streaming response
// Thinking comes in delta.thinking, content in delta.content
// They are streamed separately - thinking first, then content
```

### F. Embeddings

Generate vector embeddings for semantic search, RAG, etc.

```typescript
async function createEmbeddings(
  texts: string | string[],
  model: string
): Promise<EmbeddingResponse> {
  const input = Array.isArray(texts) ? texts : [texts];

  const response = await fetch(`${BASE_URL}/v1/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, input })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Response structure:
// {
//   object: "list",
//   data: [
//     { object: "embedding", embedding: [0.123, -0.456, ...], index: 0 },
//     { object: "embedding", embedding: [0.789, -0.012, ...], index: 1 },
//   ],
//   model: "embedding-model",
//   usage: { prompt_tokens: 10, total_tokens: 10 }
// }
```

### G. Log Probabilities

Get token-level probability data for analysis, confidence scoring, etc.

```typescript
// Request with logprobs
const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model,
    messages,
    logprobs: true,
    top_logprobs: 5,  // Get top 5 alternatives per token
    max_tokens: 100
  })
});

// Response includes logprobs per token:
// choices[0].logprobs.content = [
//   {
//     token: "Hello",
//     token_id: 9906,
//     logprob: -0.5,
//     bytes: [72, 101, 108, 108, 111],
//     top_logprobs: [
//       { token: "Hello", token_id: 9906, logprob: -0.5, bytes: [...] },
//       { token: "Hi", token_id: 13347, logprob: -1.2, bytes: [...] },
//       // ... more alternatives
//     ]
//   },
//   // ... more tokens
// ]
```

**Note:** Logprobs are only available for text-only requests. Vision requests do not support logprobs.

### H. Audio Transcription (macOS only)

Speech-to-text using CoreML.

```typescript
async function transcribeAudio(
  audioFile: File,
  model: string,
  language?: string
): Promise<{ text: string }> {
  const formData = new FormData();
  formData.append('file', audioFile);
  formData.append('model', model);
  if (language) {
    formData.append('language', language);
  }

  const response = await fetch(`${BASE_URL}/v1/audio/transcriptions`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Supported formats: mp3, wav, m4a, webm, flac, ogg, mp4, mpeg
```

### I. Structured Hidden States (MLX only)

Extract hidden states with server-side chat template application and token boundary tracking. This is useful for:
- Z-Image embeddings with precise template control
- Token attribution research
- Ablation studies on prompt sections
- Debugging chat template formatting

```typescript
interface StructuredHiddenStatesRequest {
  model: string;
  user_prompt: string;
  system_prompt?: string;
  thinking_content?: string;
  assistant_content?: string;
  enable_thinking?: boolean;
  layer?: number;
  max_length?: number;
  encoding_format?: 'float' | 'base64';
  return_token_boundaries?: boolean;
  return_formatted_prompt?: boolean;
}

interface TokenBoundary {
  start: number;
  end: number;
}

interface StructuredHiddenStatesResponse {
  hidden_states: number[][] | string;
  shape: [number, number];
  model: string;
  layer: number;
  dtype: string;
  encoding_format?: string;
  token_boundaries?: {
    system?: TokenBoundary;
    user?: TokenBoundary;
    think?: TokenBoundary;
    assistant?: TokenBoundary;
  };
  token_counts?: {
    system?: number;
    user?: number;
    think?: number;
    assistant?: number;
    total: number;
  };
  formatted_prompt?: string;
}

async function extractStructuredHiddenStates(
  request: StructuredHiddenStatesRequest
): Promise<StructuredHiddenStatesResponse> {
  const response = await fetch(`${BASE_URL}/v1/hidden_states/structured`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Example usage
const response = await extractStructuredHiddenStates({
  model: 'Qwen3-4B',
  user_prompt: 'What is the capital of France?',
  system_prompt: 'You are a helpful geography assistant.',
  enable_thinking: true,
  layer: -2,
  encoding_format: 'base64',
  return_token_boundaries: true
});

// Response includes:
// - hidden_states: base64-encoded tensor or float array
// - shape: [seq_len, hidden_dim] e.g., [120, 2560]
// - token_boundaries: { system: {start: 0, end: 35}, user: {start: 35, end: 80} }
// - token_counts: { system: 35, user: 45, total: 120 }
```

**Key Features:**
- Server applies Qwen3 chat template internally
- Returns token indices showing where each section starts/ends
- Supports pre-filled thinking and assistant content
- MLX models only (not supported for llama.cpp)
- Returns raw hidden states from specified layer (default: -2)

**Token Boundaries:**
The `token_boundaries` field shows where each prompt section starts and ends in the token sequence:
- `system`: System prompt tokens
- `user`: User prompt tokens
- `think`: Thinking block tokens (if thinking_content provided)
- `assistant`: Assistant content tokens (if assistant_content provided)

---

## 5. TypeScript Types

Copy-paste ready type definitions:

```typescript
// === Message Types ===

export type MessageRole = 'system' | 'user' | 'assistant';

export interface TextContent {
  type: 'text';
  text: string;
}

export interface ImageContent {
  type: 'image_url';
  image_url: {
    url: string;  // data:image/...;base64,... or __RAW_IMAGE__
  };
}

export type MessageContent = TextContent | ImageContent;

export interface Message {
  role: MessageRole;
  content: string | MessageContent[];
}

// === Model Types ===

export interface Model {
  id: string;
  object: 'model';
  created?: number;
  owned_by: string;
  provider?: 'mlx' | 'llama_cpp' | 'gguf' | 'coreml_stt' | 'mlx_stt';
  capabilities?: string[];  // e.g., ['chat', 'vision', 'hidden_states', 'thinking']
}

export interface ModelParameters {
  // Core sampling
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;

  // Repetition control
  repetition_penalty?: number;
  repetition_context_size?: number;
  presence_penalty?: number;
  frequency_penalty?: number;

  // Generation control
  seed?: number;
  stop?: string[];

  // Streaming
  stream?: boolean;
  stream_options?: {
    include_usage?: boolean;
  };

  // Logprobs
  logprobs?: boolean;
  top_logprobs?: number;

  // Thinking mode
  enable_thinking?: boolean;

  // Image processing (multipart only)
  resize_max?: number;
  resize_width?: number;
  resize_height?: number;
  image_quality?: number;
  preserve_alpha?: boolean;
}

// === Request Types ===

export interface ChatCompletionRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  seed?: number;
  stop?: string[];
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  logprobs?: boolean;
  top_logprobs?: number;
  enable_thinking?: boolean;
}

export interface EmbeddingRequest {
  model: string;
  input: string | string[];
  encoding_format?: 'float' | 'base64';
}

export interface BatchChatRequest {
  model: string;
  messages: Message[][];  // Array of conversation arrays
  temperature?: number;
  max_tokens?: number;
  // ... other parameters
}

// === Response Types ===

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface TokenLogprob {
  token: string;
  token_id: number;
  logprob: number;
  bytes: number[];
  top_logprobs?: TokenLogprob[];
}

export interface Logprobs {
  content: TokenLogprob[];
}

export interface ChatCompletionChoice {
  index: number;
  message: {
    role: 'assistant';
    content: string;
    thinking?: string;  // Qwen3 thinking mode
  };
  logprobs?: Logprobs;
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: Usage;
}

export interface StreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: [{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
      thinking?: string;
    };
    logprobs?: Logprobs;
    finish_reason: 'stop' | 'length' | null;
  }];
  usage?: Usage;  // Only in final chunk with stream_options.include_usage
}

export interface EmbeddingData {
  object: 'embedding';
  embedding: number[];
  index: number;
}

export interface EmbeddingResponse {
  object: 'list';
  data: EmbeddingData[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface BatchCompletionResponse {
  id: string;
  object: 'batch.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
}

// === Hidden States Types ===

export interface TokenBoundary {
  start: number;
  end: number;
}

export interface StructuredHiddenStatesRequest {
  model: string;
  user_prompt: string;
  system_prompt?: string;
  thinking_content?: string;
  assistant_content?: string;
  enable_thinking?: boolean;
  layer?: number;
  max_length?: number;
  encoding_format?: 'float' | 'base64';
  return_token_boundaries?: boolean;
  return_formatted_prompt?: boolean;
}

export interface StructuredHiddenStatesResponse {
  hidden_states: number[][] | string;
  shape: [number, number];
  model: string;
  layer: number;
  dtype: string;
  encoding_format?: string;
  token_boundaries?: {
    system?: TokenBoundary;
    user?: TokenBoundary;
    think?: TokenBoundary;
    assistant?: TokenBoundary;
  };
  token_counts?: {
    system?: number;
    user?: number;
    think?: number;
    assistant?: number;
    total: number;
  };
  formatted_prompt?: string;
}

// === Server Capabilities ===

export interface ServerCapabilities {
  server_version: string;
  optimizations: {
    json: {
      orjson_available: boolean;
      speedup: string;
    };
    image: {
      xxhash_available: boolean;
      turbojpeg_available: boolean;
      cachetools_available: boolean;
      hash_speedup: string;
      jpeg_speedup: string;
    };
  };
  metal?: {
    available: boolean;
    device_name: string;
    max_recommended_working_set_size: number;
  };
  endpoints: {
    fast_vision: {
      available: boolean;
      endpoint: string;
      description: string;
      benefits: {
        time_saved_per_image_ms: number;
        bandwidth_reduction_percent: number;
        supports_parallel_processing: boolean;
      };
    };
    batch_processing: {
      available: boolean;
      processing_modes: string[];
    };
  };
  features: {
    streaming: boolean;
    model_caching: {
      enabled: boolean;
      cache_size: number;
      eviction_policy: string;
    };
    vision_models: boolean;
    concurrent_requests: boolean;
    supported_image_formats: string[];
  };
  recommendations: {
    vision_models: {
      use_multipart: boolean;
      reason: string;
    };
    batch_size: {
      optimal: number;
      max: number;
    };
    image_format: {
      preferred: string;
      quality: number;
    };
  };
  limits: {
    max_tokens: number;
    max_images_per_request: number;
    max_request_size_mb: number;
    timeout_seconds: number;
  };
}
```

---

## 6. React Patterns

### Recommended State Structure

```typescript
interface ChatState {
  // Connection
  isConnected: boolean;
  capabilities: ServerCapabilities | null;

  // Models
  models: Model[];
  selectedModel: string | null;

  // Conversation
  messages: Message[];

  // Generation
  isStreaming: boolean;
  streamingContent: string;

  // Parameters
  parameters: ModelParameters;

  // UI
  error: string | null;
}
```

### API Service Class

```typescript
class HeylookAPI {
  private baseUrl: string;

  constructor(baseUrl = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
  }

  // Connection
  async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/models`);
      return response.ok;
    } catch {
      return false;
    }
  }

  async getCapabilities(): Promise<ServerCapabilities> {
    const response = await fetch(`${this.baseUrl}/v1/capabilities`);
    return response.json();
  }

  // Models
  async getModels(): Promise<Model[]> {
    const response = await fetch(`${this.baseUrl}/v1/models`);
    const data = await response.json();
    return data.data || [];
  }

  // Chat
  async chat(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: false })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async streamChat(
    request: ChatCompletionRequest,
    onToken: (token: string) => void,
    onThinking?: (thinking: string) => void,
    onComplete?: (usage?: Usage) => void,
    onError?: (error: Error) => void
  ): Promise<void> {
    // Implementation from Section 4.A
  }

  // Vision
  async chatWithImage(
    prompt: string,
    images: File[],
    model: string,
    parameters: ModelParameters,
    useMultipart = true
  ): Promise<ChatCompletionResponse> {
    if (useMultipart) {
      // Implementation from Section 4.C Method 2
    } else {
      // Implementation from Section 4.C Method 1
    }
  }

  // Embeddings
  async createEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    return response.json();
  }

  // Batch
  async batchChat(request: BatchChatRequest): Promise<BatchCompletionResponse> {
    const response = await fetch(`${this.baseUrl}/v1/batch/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    return response.json();
  }
}

export const api = new HeylookAPI();
```

### Image Utilities

```typescript
export const imageUtils = {
  // Convert File to base64
  fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  },

  // Resize image before upload
  resizeImage(
    file: File,
    maxWidth = 1024,
    maxHeight = 1024,
    quality = 0.85
  ): Promise<File> {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      const img = new Image();

      img.onload = () => {
        let { width, height } = img;

        // Calculate new dimensions
        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }

        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob(
          (blob) => {
            const resizedFile = new File([blob!], file.name, {
              type: file.type,
              lastModified: Date.now()
            });
            resolve(resizedFile);
          },
          file.type,
          quality
        );
      };

      img.src = URL.createObjectURL(file);
    });
  },

  // Get image dimensions
  getImageDimensions(file: File): Promise<{ width: number; height: number }> {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        resolve({ width: img.width, height: img.height });
      };
      img.src = URL.createObjectURL(file);
    });
  }
};
```

---

## 7. UX Recommendations

### Model Loading States

Models are loaded on-demand with LRU caching (max 2 models).

| State | Duration | UI Feedback |
|-------|----------|-------------|
| First request | 2-30s | Loading spinner + "Loading model..." |
| Cached model | Instant | No loading state needed |
| Model switch | 2-30s | Loading spinner + "Switching model..." |

```typescript
// Show loading state during first request to a model
const [isModelLoading, setIsModelLoading] = useState(false);

async function sendMessage() {
  setIsModelLoading(true);
  try {
    await api.chat(request);
  } finally {
    setIsModelLoading(false);
  }
}
```

### Parameter Controls

Organize parameters into sections:

**Essential (always visible):**
- Temperature (slider)
- Max tokens (number input)
- Stream toggle

**Sampling (collapsible):**
- Top P (slider)
- Top K (number input)
- Min P (slider)

**Repetition (collapsible):**
- Repetition penalty (slider)
- Presence penalty (slider)
- Frequency penalty (slider)

**Advanced (collapsible, default closed):**
- Seed (number input)
- Stop sequences (text input)
- Logprobs toggle
- Thinking mode toggle (Qwen3 only)

### Image Handling

1. **Preview before upload** - Show thumbnail and dimensions
2. **Resize indicator** - Show if image will be resized
3. **Progress feedback** - Show upload progress for multipart
4. **Drag and drop** - Support file drops onto chat area
5. **Paste support** - Handle clipboard images (Ctrl+V)

```typescript
// Paste handler example
function handlePaste(e: React.ClipboardEvent) {
  const items = e.clipboardData.items;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) {
        handleImageUpload(file);
      }
    }
  }
}
```

### Thinking Mode UI

When thinking mode is enabled:
1. Show thinking content in a collapsible/expandable section
2. Use distinct styling (e.g., italic, different background)
3. Allow users to hide thinking by default
4. Show "Thinking..." indicator during streaming

---

## 8. Performance Optimizations

### Summary Table

| Optimization | Benefit | Implementation |
|--------------|---------|----------------|
| Multipart images | 57ms faster/image | Use `/v1/chat/completions/multipart` |
| Batch processing | 2-4x throughput | Use `/v1/batch/chat/completions` |
| Client-side resize | Faster upload | Resize to 1024px before send |
| Connection reuse | Reduced latency | Single API service instance |
| Capability caching | Avoid repeated checks | Query once on startup |

### Multipart vs Base64

```typescript
// Check capabilities on startup
const caps = await api.getCapabilities();
const useMultipart = caps.recommendations.vision_models.use_multipart;

// Use appropriate method
if (useMultipart && images.length > 0) {
  return api.chatWithImageMultipart(prompt, images, model, params);
} else {
  return api.chatWithImageBase64(prompt, images, model, params);
}
```

### Batch Processing

Use batch endpoint when:
- Multiple independent prompts
- Don't need streaming
- Using text-only models

```typescript
// Instead of:
const results = await Promise.all(
  prompts.map(p => api.chat({ model, messages: [{ role: 'user', content: p }] }))
);

// Use batch:
const results = await api.batchChat({
  model,
  messages: prompts.map(p => [{ role: 'user', content: p }])
});
```

---

## 9. Error Handling

### HTTP Status Codes

| Status | Meaning | User Action |
|--------|---------|-------------|
| 200 | Success | - |
| 400 | Bad request | Check parameters |
| 404 | Model not found | Check model ID |
| 422 | Validation error | Check request format |
| 500 | Server error | Retry or check server logs |
| 503 | Model loading | Wait and retry |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

Or for validation errors:
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

### Retry Logic

```typescript
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  backoff = 1000
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      // Don't retry client errors (4xx)
      if (error instanceof Error && error.message.includes('HTTP 4')) {
        throw error;
      }

      // Wait before retry with exponential backoff
      await new Promise(r => setTimeout(r, backoff * Math.pow(2, attempt)));
    }
  }

  throw lastError;
}
```

---

## 10. Mobile/Responsive Design

### Layout Considerations

| Screen Size | Layout |
|-------------|--------|
| Desktop (>1024px) | Side-by-side chat + parameters |
| Tablet (768-1024px) | Chat with collapsible parameter drawer |
| Mobile (<768px) | Full-width chat, parameters in modal |

### Mobile-Specific Optimizations

1. **Reduced max_tokens default** - Save battery/bandwidth
2. **Aggressive image compression** - `quality: 70`, `resize_max: 768`
3. **Touch-friendly controls** - Larger tap targets, native sliders
4. **Keyboard handling** - Auto-scroll when keyboard opens

### Responsive Parameter Panel

```tsx
function ParameterPanel({ isOpen, onClose, parameters, onChange }) {
  const isMobile = useMediaQuery('(max-width: 768px)');

  if (isMobile) {
    return (
      <Modal isOpen={isOpen} onClose={onClose}>
        <ParameterControls parameters={parameters} onChange={onChange} />
      </Modal>
    );
  }

  return (
    <aside className={`parameter-sidebar ${isOpen ? 'open' : 'closed'}`}>
      <ParameterControls parameters={parameters} onChange={onChange} />
    </aside>
  );
}
```

### Image Upload on Mobile

```typescript
// Use native file picker with camera option
<input
  type="file"
  accept="image/*"
  capture="environment"  // Opens camera on mobile
  onChange={handleFileSelect}
/>
```

---

## Appendix: Quick Reference

### Endpoint Quick Reference

```bash
# List models
GET /v1/models

# Chat completion
POST /v1/chat/completions
Content-Type: application/json

# Chat with images (fast)
POST /v1/chat/completions/multipart
Content-Type: multipart/form-data

# Embeddings
POST /v1/embeddings

# Batch processing
POST /v1/batch/chat/completions

# Server capabilities
GET /v1/capabilities

# Audio transcription
POST /v1/audio/transcriptions

# Admin: reload config
POST /v1/admin/reload
```

### cURL Examples

```bash
# Simple chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'

# With logprobs
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "logprobs": true,
    "top_logprobs": 5
  }'
```

---

## Related Documentation

- **Swagger UI:** `http://localhost:8080/docs` - Interactive API explorer
- **ReDoc:** `http://localhost:8080/redoc` - Readable API documentation
- **OpenAPI Schema:** `http://localhost:8080/openapi.json` - For code generation
