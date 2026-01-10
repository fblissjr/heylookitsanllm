import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { SamplerSettings, Preset, PresetType } from '../types/settings'
import { DEFAULT_SAMPLER_SETTINGS } from '../types/settings'

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// Built-in presets
const BUILTIN_SYSTEM_PROMPTS: Preset[] = [
  {
    id: 'default',
    type: 'system_prompt',
    name: 'Default',
    description: 'A helpful AI assistant',
    data: { prompt: 'You are a helpful AI assistant.' },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: 'coding',
    type: 'system_prompt',
    name: 'Coding Assistant',
    description: 'Expert programmer with clear explanations',
    data: { prompt: 'You are an expert programmer. Provide clear, concise code examples with explanations. Use best practices and modern patterns.' },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: 'creative',
    type: 'system_prompt',
    name: 'Creative Writer',
    description: 'Creative and imaginative writing assistant',
    data: { prompt: 'You are a creative writing assistant. Help with storytelling, poetry, and imaginative content. Be expressive and engaging.' },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: 'analyst',
    type: 'system_prompt',
    name: 'Data Analyst',
    description: 'Precise data analysis and insights',
    data: { prompt: 'You are a data analyst. Provide precise, factual analysis. Use structured formats when presenting data. Be thorough and accurate.' },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
]

const BUILTIN_SAMPLER_PRESETS: Preset[] = [
  {
    id: 'balanced',
    type: 'sampler',
    name: 'Balanced',
    description: 'Good for general use',
    data: DEFAULT_SAMPLER_SETTINGS,
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: 'creative',
    type: 'sampler',
    name: 'Creative',
    description: 'Higher temperature for creative tasks',
    data: {
      ...DEFAULT_SAMPLER_SETTINGS,
      temperature: 1.2,
      top_p: 0.95,
      top_k: 50,
    },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: 'precise',
    type: 'sampler',
    name: 'Precise',
    description: 'Lower temperature for factual responses',
    data: {
      ...DEFAULT_SAMPLER_SETTINGS,
      temperature: 0.3,
      top_p: 0.85,
      top_k: 30,
    },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: 'deterministic',
    type: 'sampler',
    name: 'Deterministic',
    description: 'Reproducible outputs',
    data: {
      ...DEFAULT_SAMPLER_SETTINGS,
      temperature: 0.0,
      top_p: 1.0,
      top_k: 1,
    },
    isBuiltIn: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
]

interface SettingsState {
  // Current settings
  systemPrompt: string
  jinjaTemplate: string | null
  samplerSettings: SamplerSettings

  // Presets (user-created, built-ins are separate)
  userPresets: Preset[]

  // Active preset IDs
  activeSystemPromptPresetId: string | null
  activeSamplerPresetId: string | null

  // Actions - Settings
  setSystemPrompt: (prompt: string) => void
  setJinjaTemplate: (template: string | null) => void
  updateSamplerSettings: (settings: Partial<SamplerSettings>) => void
  resetSamplerToDefaults: () => void

  // Actions - Presets
  getAllPresets: (type: PresetType) => Preset[]
  savePreset: (type: PresetType, name: string, description?: string) => string
  loadPreset: (preset: Preset) => void
  duplicatePreset: (preset: Preset) => string
  deletePreset: (presetId: string) => void
  updatePreset: (presetId: string, updates: { name?: string; description?: string }) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      systemPrompt: BUILTIN_SYSTEM_PROMPTS[0].data.prompt as string,
      jinjaTemplate: null,
      samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS },
      userPresets: [],
      activeSystemPromptPresetId: 'default',
      activeSamplerPresetId: 'balanced',

      setSystemPrompt: (prompt) => {
        set({ systemPrompt: prompt, activeSystemPromptPresetId: null })
      },

      setJinjaTemplate: (template) => {
        set({ jinjaTemplate: template })
      },

      updateSamplerSettings: (settings) => {
        set(state => ({
          samplerSettings: { ...state.samplerSettings, ...settings },
          activeSamplerPresetId: null, // Clear preset when manually changing
        }))
      },

      resetSamplerToDefaults: () => {
        set({
          samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS },
          activeSamplerPresetId: 'balanced',
        })
      },

      getAllPresets: (type) => {
        const { userPresets } = get()
        const builtIns = type === 'system_prompt'
          ? BUILTIN_SYSTEM_PROMPTS
          : type === 'sampler'
            ? BUILTIN_SAMPLER_PRESETS
            : []
        return [...builtIns, ...userPresets.filter(p => p.type === type)]
      },

      savePreset: (type, name, description) => {
        const { systemPrompt, samplerSettings } = get()
        const id = generateId()
        const now = new Date().toISOString()

        let data: Record<string, unknown>
        if (type === 'system_prompt') {
          data = { prompt: systemPrompt }
        } else if (type === 'sampler') {
          data = { ...samplerSettings }
        } else {
          data = {}
        }

        const preset: Preset = {
          id,
          type,
          name,
          description,
          data,
          isBuiltIn: false,
          createdAt: now,
          updatedAt: now,
        }

        set(state => ({
          userPresets: [...state.userPresets, preset],
          ...(type === 'system_prompt' ? { activeSystemPromptPresetId: id } : {}),
          ...(type === 'sampler' ? { activeSamplerPresetId: id } : {}),
        }))

        return id
      },

      loadPreset: (preset) => {
        if (preset.type === 'system_prompt') {
          set({
            systemPrompt: preset.data.prompt as string,
            activeSystemPromptPresetId: preset.id,
          })
        } else if (preset.type === 'sampler') {
          set({
            samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS, ...preset.data as SamplerSettings },
            activeSamplerPresetId: preset.id,
          })
        }
      },

      duplicatePreset: (preset) => {
        const id = generateId()
        const now = new Date().toISOString()

        const newPreset: Preset = {
          ...preset,
          id,
          name: `${preset.name} (copy)`,
          isBuiltIn: false,
          createdAt: now,
          updatedAt: now,
        }

        set(state => ({
          userPresets: [...state.userPresets, newPreset],
        }))

        return id
      },

      deletePreset: (presetId) => {
        set(state => ({
          userPresets: state.userPresets.filter(p => p.id !== presetId),
          activeSystemPromptPresetId:
            state.activeSystemPromptPresetId === presetId ? null : state.activeSystemPromptPresetId,
          activeSamplerPresetId:
            state.activeSamplerPresetId === presetId ? null : state.activeSamplerPresetId,
        }))
      },

      updatePreset: (presetId, updates) => {
        set(state => ({
          userPresets: state.userPresets.map(p =>
            p.id === presetId
              ? { ...p, ...updates, updatedAt: new Date().toISOString() }
              : p
          ),
        }))
      },
    }),
    {
      name: 'heylook:settings',
      partialize: (state) => ({
        userPresets: state.userPresets,
        systemPrompt: state.systemPrompt,
        samplerSettings: state.samplerSettings,
        activeSystemPromptPresetId: state.activeSystemPromptPresetId,
        activeSamplerPresetId: state.activeSamplerPresetId,
      }),
    }
  )
)
