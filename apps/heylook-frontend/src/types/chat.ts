// Internal Chat/Conversation Types

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
}

export type MessageActionType = 'copy' | 'edit' | 'delete' | 'regenerate';

export interface EditState {
  messageId: string | null;
  originalContent: string;
  editedContent: string;
}
