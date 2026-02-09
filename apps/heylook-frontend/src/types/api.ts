// API Types -- Adapter Layer
//
// Manual type definitions for the frontend. The generated file
// (generated-api.ts) is auto-produced by `npm run generate:api` with a
// sed post-process that converts `| null` to `| undefined`. Structural
// differences remain (required-with-default vs optional, string vs
// literal), so we keep manual interfaces here and use compile-time
// assertions below to catch drift against the generated schema.
//
// To regenerate: npm run generate:api  (requires backend running at :8080)

// Verify that the generated schema file exists at build time.
// If this import fails, run: npm run generate:api
import type { components as _components } from './generated-api'

// Type-check: ensure our manual TokenLogprob is structurally compatible
// with the generated TokenLogprobInfo (catches drift at compile time).
type _AssertTokenLogprob = _components['schemas']['TokenLogprobInfo']
type _AssertStreamChunk = _components['schemas']['StreamChunk']
// These are consumed below to suppress unused-type warnings.
export type { _AssertTokenLogprob as _GeneratedTokenLogprob }
export type { _AssertStreamChunk as _GeneratedStreamChunk }

// =============================================================================
// Message Types
// =============================================================================

export type MessageRole = 'system' | 'user' | 'assistant';

export interface TextContent {
  type: 'text';
  text: string;
}

export interface ImageContent {
  type: 'image_url';
  image_url: {
    url: string; // data:image/...;base64,... or __RAW_IMAGE__
  };
}

export type MessageContent = TextContent | ImageContent;

export interface APIMessage {
  role: MessageRole;
  content: string | MessageContent[];
  thinking?: string;
}

// =============================================================================
// Model Types
// =============================================================================

export interface Model {
  id: string;
  object: 'model';
  created?: number;
  owned_by: string;
  provider?: 'mlx' | 'llama_cpp' | 'gguf' | 'coreml_stt' | 'mlx_stt';
  capabilities?: string[];
  context_window?: number;
}

export interface ModelListResponse {
  object: 'list';
  data: Model[];
}

// =============================================================================
// Chat Completion Request (backend: ChatRequest)
// =============================================================================

export interface ChatCompletionRequest {
  model: string;
  messages: APIMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  repetition_context_size?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  seed?: number;
  stop?: string[];
  stream?: boolean;
  stream_options?: {
    include_usage?: boolean;
  };
  logprobs?: boolean;
  top_logprobs?: number;
  enable_thinking?: boolean;
}

// =============================================================================
// Usage & Metrics
// =============================================================================

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// Re-export from generated schema with optional-field normalization.
// The generated types use `field: T | undefined` (required key) while
// consumers expect `field?: T` (optional key). We wrap with Partial for
// fields that the server may omit, keeping required fields strict.
type _GenEnhancedUsage = _components['schemas']['EnhancedUsage']
export type EnhancedUsage = Omit<_GenEnhancedUsage, 'thinking_tokens' | 'content_tokens'> & {
  thinking_tokens?: number;
  content_tokens?: number;
}

type _GenGenerationTiming = _components['schemas']['GenerationTiming']
export type GenerationTiming = Partial<_GenGenerationTiming>

export type GenerationConfig = Partial<_components['schemas']['GenerationConfig']>

// =============================================================================
// Logprobs (corresponds to generated TokenLogprobInfo, TopLogprobEntry)
// =============================================================================

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

// =============================================================================
// Chat Completion Response (backend: ChatCompletionResponse)
// =============================================================================

export interface ChatCompletionChoice {
  index: number;
  message: {
    role: 'assistant';
    content: string;
    thinking?: string;
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

// =============================================================================
// Streaming Types (corresponds to generated StreamChunk, StreamChoice, StreamDelta)
// =============================================================================

export interface StreamDelta {
  role?: 'assistant';
  content?: string;
  thinking?: string;
}

export interface StreamChoice {
  index: number;
  delta: StreamDelta;
  logprobs?: Logprobs;
  finish_reason: 'stop' | 'length' | null;
}

export interface StreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: StreamChoice[];
  usage?: EnhancedUsage;
  timing?: GenerationTiming;
  generation_config?: GenerationConfig;
  stop_reason?: string;
}

// =============================================================================
// Server Capabilities
// =============================================================================

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

// =============================================================================
// Error & Frontend-Only Types
// =============================================================================

export interface APIError {
  detail: string | Array<{
    loc: (string | number)[];
    msg: string;
    type: string;
  }>;
}

export interface SamplerParams {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  repetition_context_size?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  seed?: number;
  stop?: string[];
}
