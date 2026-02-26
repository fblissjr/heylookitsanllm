import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { SamplerSettings, Preset, PresetType } from '../types/settings'
import { DEFAULT_SAMPLER_SETTINGS } from '../types/settings'
import { generateId } from '../lib/id'

// Built-in system prompt presets removed - default is empty to use model's jinja2 chat template
// Users can still create their own presets
const BUILTIN_SYSTEM_PROMPTS: Preset[] = []

// Built-in sampler presets removed - defaults should come from model config (models.toml)
// Users can still create their own presets
// TODO: Extend /v1/models endpoint to return sampler defaults from models.toml
const BUILTIN_SAMPLER_PRESETS: Preset[] = []

interface SettingsState {
  // Current settings
  systemPrompt: string
  jinjaTemplate: string | null
  samplerSettings: SamplerSettings
  streamTimeoutMs: number

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
  setStreamTimeoutMs: (ms: number) => void

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
      // Default to empty - model's jinja2 chat template handles system behavior
      systemPrompt: '',
      jinjaTemplate: null,
      samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS },
      streamTimeoutMs: 30_000,
      userPresets: [],
      // No active presets by default - using empty/default values
      activeSystemPromptPresetId: null,
      activeSamplerPresetId: null,

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
          activeSamplerPresetId: null,
        })
      },

      setStreamTimeoutMs: (ms) => set({ streamTimeoutMs: ms }),

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
        streamTimeoutMs: state.streamTimeoutMs,
        activeSystemPromptPresetId: state.activeSystemPromptPresetId,
        activeSamplerPresetId: state.activeSamplerPresetId,
      }),
    }
  )
)
