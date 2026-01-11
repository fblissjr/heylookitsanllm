// Internal Chat/Conversation Types

export interface GenerationConfig {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  enable_thinking?: boolean;
  max_tokens?: number;
}

export interface PerformanceMetrics {
  timeToFirstToken?: number;   // ms from request to first token
  tokensPerSecond?: number;    // completion tokens / generation time
  totalDuration?: number;      // ms total generation time
  promptTokens?: number;
  completionTokens?: number;
  cached?: boolean;            // prefix cache hit (when backend supports it)
  // Enhanced metrics from streaming
  thinkingTokens?: number;     // tokens used in thinking blocks
  contentTokens?: number;      // tokens in actual response content
  thinkingDuration?: number;   // ms spent in thinking phase
  stopReason?: string;         // why generation stopped (stop, length, etc)
  generationConfig?: GenerationConfig;  // sampler settings used
}

export interface Message {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string;
  thinking?: string;
  images?: string[]; // Base64 data URLs
  timestamp: number;
  tokenCount?: number;
  isEditing?: boolean;
  isRegenerating?: boolean;
  performance?: PerformanceMetrics;
  rawStream?: string[];  // Raw SSE events for debugging
  // Multi-model conversation support
  modelId?: string;            // Model that generated this message
}

export interface Conversation {
  id: string;
  title: string;
  defaultModelId: string;      // Default model for new messages (renamed from modelId)
  messages: Message[];
  systemPrompt?: string;
  createdAt: number;
  updatedAt: number;
}

export interface StreamingState {
  isStreaming: boolean;
  content: string;
  thinking: string;
  messageId: string | null;
  // Timing data captured during stream
  startTime?: number;
  firstTokenTime?: number;
  rawEvents?: string[];  // Raw SSE events for debugging
}

export type MessageActionType = 'copy' | 'edit' | 'delete' | 'regenerate';

export interface EditState {
  messageId: string | null;
  originalContent: string;
  editedContent: string;
}
