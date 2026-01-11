// Internal Chat/Conversation Types

export interface PerformanceMetrics {
  timeToFirstToken?: number;   // ms from request to first token
  tokensPerSecond?: number;    // completion tokens / generation time
  totalDuration?: number;      // ms total generation time
  promptTokens?: number;
  completionTokens?: number;
  cached?: boolean;            // prefix cache hit (when backend supports it)
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
}

export interface Conversation {
  id: string;
  title: string;
  modelId: string;
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
