// API Request/Response Types based on docs/FRONTEND_HANDOFF.md

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
}

export interface Model {
  id: string;
  object: 'model';
  created?: number;
  owned_by: string;
  provider?: 'mlx' | 'llama_cpp' | 'gguf' | 'coreml_stt' | 'mlx_stt';
  capabilities?: string[]; // 'chat', 'vision', 'hidden_states', 'thinking'
  context_window?: number; // Default context window from config
}

export interface ModelListResponse {
  object: 'list';
  data: Model[];
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
  usage?: Usage;
}

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

export interface APIError {
  detail: string | Array<{
    loc: (string | number)[];
    msg: string;
    type: string;
  }>;
}
