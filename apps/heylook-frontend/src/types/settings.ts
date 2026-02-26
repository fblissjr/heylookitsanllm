// Settings and Presets Types

export type PresetType = 'system_prompt' | 'jinja_template' | 'sampler';

export interface Preset {
  id: string;
  type: PresetType;
  name: string;
  description?: string;
  data: Record<string, unknown>;
  isBuiltIn: boolean;
  modelDefault?: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface SystemPromptPreset extends Preset {
  type: 'system_prompt';
  data: {
    prompt: string;
  };
}

export interface JinjaTemplatePreset extends Preset {
  type: 'jinja_template';
  data: {
    template: string;
  };
}

export interface SamplerPreset extends Preset {
  type: 'sampler';
  data: SamplerSettings;
}

export interface SamplerSettings {
  temperature: number;
  max_tokens: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  repetition_context_size: number;
  presence_penalty: number;
  frequency_penalty: number;
  seed?: number;
  stop?: string[];
  enable_thinking: boolean;
  streamTimeoutMs?: number;
  // Index signature for compatibility with Record<string, unknown>
  [key: string]: unknown;
}

export const DEFAULT_SAMPLER_SETTINGS: SamplerSettings = {
  temperature: 0.7,
  max_tokens: 2048,
  top_p: 0.9,
  top_k: 0,
  min_p: 0.0,
  repetition_penalty: 1.1,
  repetition_context_size: 20,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  enable_thinking: false,
  streamTimeoutMs: 30_000,
};
