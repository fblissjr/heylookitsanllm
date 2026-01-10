// Model-related types

export interface ModelCapabilities {
  chat: boolean;
  vision: boolean;
  thinking: boolean;
  hidden_states: boolean;
  embeddings: boolean;
}

export interface LoadedModel {
  id: string;
  provider?: string;
  capabilities: ModelCapabilities;
  contextWindow: number;
  loadedAt?: number;
}

export interface ModelLoadConfig {
  modelId: string;
  contextWindow?: number;
  // Additional load-time settings can be added here
}

export type ModelStatus = 'unloaded' | 'loading' | 'loaded' | 'error';
