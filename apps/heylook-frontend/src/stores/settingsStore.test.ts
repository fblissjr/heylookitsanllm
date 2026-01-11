import { describe, it, expect, beforeEach } from 'vitest'
import { useSettingsStore } from './settingsStore'
import { DEFAULT_SAMPLER_SETTINGS } from '../types/settings'

describe('settingsStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    // No built-in presets - default system prompt is empty
    useSettingsStore.setState({
      systemPrompt: '',
      jinjaTemplate: null,
      samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS },
      userPresets: [],
      activeSystemPromptPresetId: null,
      activeSamplerPresetId: null,
    })
  })

  describe('system prompt', () => {
    it('has empty default system prompt', () => {
      const { systemPrompt } = useSettingsStore.getState()
      expect(systemPrompt).toBe('')
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
      expect(activeSamplerPresetId).toBeNull() // No built-in presets
    })
  })

  describe('presets', () => {
    it('starts with no built-in system prompts (use model jinja2 template)', () => {
      const { getAllPresets } = useSettingsStore.getState()
      const presets = getAllPresets('system_prompt')

      // No built-in presets - users create their own
      expect(presets.length).toBe(0)
    })

    it('starts with no built-in sampler presets (use model defaults)', () => {
      const { getAllPresets } = useSettingsStore.getState()
      const presets = getAllPresets('sampler')

      // No built-in presets - sampler defaults should come from models.toml
      expect(presets.length).toBe(0)
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

    it('saves a new sampler preset', () => {
      const { updateSamplerSettings, savePreset, getAllPresets } = useSettingsStore.getState()

      updateSamplerSettings({ temperature: 1.5, top_p: 0.8 })
      const presetId = savePreset('sampler', 'Creative Preset', 'For creative writing')

      const presets = getAllPresets('sampler')
      const saved = presets.find(p => p.id === presetId)

      expect(saved).toBeDefined()
      expect(saved?.name).toBe('Creative Preset')
      expect(saved?.data.temperature).toBe(1.5)
      expect(saved?.data.top_p).toBe(0.8)
    })

    it('loads a user-created preset', () => {
      const { updateSamplerSettings, savePreset, getAllPresets, loadPreset, resetSamplerToDefaults } = useSettingsStore.getState()

      // Create and save a preset
      updateSamplerSettings({ temperature: 1.2 })
      const presetId = savePreset('sampler', 'Creative', 'Higher temperature')

      // Reset to defaults
      resetSamplerToDefaults()
      expect(useSettingsStore.getState().samplerSettings.temperature).toBe(DEFAULT_SAMPLER_SETTINGS.temperature)

      // Load the preset
      const createdPreset = getAllPresets('sampler').find(p => p.id === presetId)
      expect(createdPreset).toBeDefined()
      loadPreset(createdPreset!)

      const { samplerSettings, activeSamplerPresetId } = useSettingsStore.getState()
      expect(samplerSettings.temperature).toBe(1.2)
      expect(activeSamplerPresetId).toBe(presetId)
    })

    it('duplicates a preset', () => {
      const { setSystemPrompt, savePreset, getAllPresets, duplicatePreset } = useSettingsStore.getState()

      // Create a preset to duplicate
      setSystemPrompt('Original prompt')
      const originalId = savePreset('system_prompt', 'Original', 'Original description')

      const originalPreset = getAllPresets('system_prompt').find(p => p.id === originalId)
      const newId = duplicatePreset(originalPreset!)

      const presets = getAllPresets('system_prompt')
      const duplicated = presets.find(p => p.id === newId)

      expect(duplicated).toBeDefined()
      expect(duplicated?.name).toBe('Original (copy)')
      expect(duplicated?.isBuiltIn).toBe(false)
      expect(duplicated?.data.prompt).toBe('Original prompt')
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
