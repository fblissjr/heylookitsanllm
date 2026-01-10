import { describe, it, expect, beforeEach } from 'vitest'
import { useSettingsStore } from './settingsStore'
import { DEFAULT_SAMPLER_SETTINGS } from '../types/settings'

describe('settingsStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useSettingsStore.setState({
      systemPrompt: 'You are a helpful AI assistant.',
      jinjaTemplate: null,
      samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS },
      userPresets: [],
      activeSystemPromptPresetId: 'default',
      activeSamplerPresetId: 'balanced',
    })
  })

  describe('system prompt', () => {
    it('has default system prompt', () => {
      const { systemPrompt } = useSettingsStore.getState()
      expect(systemPrompt).toBe('You are a helpful AI assistant.')
    })

    it('updates system prompt and clears active preset', () => {
      const { setSystemPrompt } = useSettingsStore.getState()

      setSystemPrompt('Custom prompt')

      const state = useSettingsStore.getState()
      expect(state.systemPrompt).toBe('Custom prompt')
      expect(state.activeSystemPromptPresetId).toBeNull()
    })
  })

  describe('sampler settings', () => {
    it('has default sampler settings', () => {
      const { samplerSettings } = useSettingsStore.getState()
      expect(samplerSettings).toEqual(DEFAULT_SAMPLER_SETTINGS)
    })

    it('updates partial sampler settings', () => {
      const { updateSamplerSettings } = useSettingsStore.getState()

      updateSamplerSettings({ temperature: 0.5 })

      const { samplerSettings, activeSamplerPresetId } = useSettingsStore.getState()
      expect(samplerSettings.temperature).toBe(0.5)
      expect(samplerSettings.top_p).toBe(DEFAULT_SAMPLER_SETTINGS.top_p) // unchanged
      expect(activeSamplerPresetId).toBeNull() // cleared when manually changing
    })

    it('resets to defaults', () => {
      const { updateSamplerSettings, resetSamplerToDefaults } = useSettingsStore.getState()

      updateSamplerSettings({ temperature: 0.1, top_k: 100 })
      resetSamplerToDefaults()

      const { samplerSettings, activeSamplerPresetId } = useSettingsStore.getState()
      expect(samplerSettings).toEqual(DEFAULT_SAMPLER_SETTINGS)
      expect(activeSamplerPresetId).toBe('balanced')
    })
  })

  describe('presets', () => {
    it('returns built-in system prompts', () => {
      const { getAllPresets } = useSettingsStore.getState()
      const presets = getAllPresets('system_prompt')

      expect(presets.length).toBeGreaterThanOrEqual(4)
      expect(presets.some(p => p.id === 'default')).toBe(true)
      expect(presets.some(p => p.id === 'coding')).toBe(true)
      expect(presets.every(p => p.type === 'system_prompt')).toBe(true)
    })

    it('returns built-in sampler presets', () => {
      const { getAllPresets } = useSettingsStore.getState()
      const presets = getAllPresets('sampler')

      expect(presets.some(p => p.id === 'balanced')).toBe(true)
      expect(presets.some(p => p.id === 'creative')).toBe(true)
      expect(presets.some(p => p.id === 'precise')).toBe(true)
      expect(presets.some(p => p.id === 'deterministic')).toBe(true)
    })

    it('saves a new system prompt preset', () => {
      const { setSystemPrompt, savePreset, getAllPresets } = useSettingsStore.getState()

      setSystemPrompt('My custom system prompt')
      const presetId = savePreset('system_prompt', 'My Preset', 'Test description')

      const presets = getAllPresets('system_prompt')
      const saved = presets.find(p => p.id === presetId)

      expect(saved).toBeDefined()
      expect(saved?.name).toBe('My Preset')
      expect(saved?.description).toBe('Test description')
      expect(saved?.data.prompt).toBe('My custom system prompt')
      expect(saved?.isBuiltIn).toBe(false)
    })

    it('loads a preset', () => {
      const { getAllPresets, loadPreset } = useSettingsStore.getState()

      const creativePreset = getAllPresets('sampler').find(p => p.id === 'creative')
      expect(creativePreset).toBeDefined()

      loadPreset(creativePreset!)

      const { samplerSettings, activeSamplerPresetId } = useSettingsStore.getState()
      expect(samplerSettings.temperature).toBe(1.2)
      expect(activeSamplerPresetId).toBe('creative')
    })

    it('duplicates a preset', () => {
      const { getAllPresets, duplicatePreset } = useSettingsStore.getState()

      const codingPreset = getAllPresets('system_prompt').find(p => p.id === 'coding')
      const newId = duplicatePreset(codingPreset!)

      const presets = getAllPresets('system_prompt')
      const duplicated = presets.find(p => p.id === newId)

      expect(duplicated).toBeDefined()
      expect(duplicated?.name).toBe('Coding Assistant (copy)')
      expect(duplicated?.isBuiltIn).toBe(false)
    })

    it('deletes a user preset', () => {
      const { savePreset, deletePreset, getAllPresets } = useSettingsStore.getState()

      const presetId = savePreset('system_prompt', 'To Delete')
      expect(getAllPresets('system_prompt').some(p => p.id === presetId)).toBe(true)

      deletePreset(presetId)

      expect(getAllPresets('system_prompt').some(p => p.id === presetId)).toBe(false)
    })

    it('clears active preset ID when deleting active preset', () => {
      const { savePreset, deletePreset } = useSettingsStore.getState()

      const presetId = savePreset('system_prompt', 'Active Preset')
      expect(useSettingsStore.getState().activeSystemPromptPresetId).toBe(presetId)

      deletePreset(presetId)

      expect(useSettingsStore.getState().activeSystemPromptPresetId).toBeNull()
    })
  })
})
