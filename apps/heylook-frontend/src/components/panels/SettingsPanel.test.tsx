import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SettingsPanel } from './SettingsPanel'
import { DEFAULT_SAMPLER_SETTINGS } from '../../types/settings'
import type { Preset } from '../../types/settings'

// Mock stores
const mockSetActivePanel = vi.fn()
const mockUpdateSamplerSettings = vi.fn()
const mockResetSamplerToDefaults = vi.fn()
const mockLoadPreset = vi.fn()
const mockSavePreset = vi.fn()
const mockDeletePreset = vi.fn()

const builtInPreset: Preset = {
  id: 'builtin-1',
  type: 'sampler',
  name: 'Balanced',
  data: { ...DEFAULT_SAMPLER_SETTINGS },
  isBuiltIn: true,
  createdAt: '2024-01-01',
  updatedAt: '2024-01-01',
}

const customPreset: Preset = {
  id: 'custom-1',
  type: 'sampler',
  name: 'My Preset',
  data: { ...DEFAULT_SAMPLER_SETTINGS, temperature: 1.5 },
  isBuiltIn: false,
  createdAt: '2024-01-15',
  updatedAt: '2024-01-15',
}

const defaultSettingsState = {
  samplerSettings: { ...DEFAULT_SAMPLER_SETTINGS },
  updateSamplerSettings: mockUpdateSamplerSettings,
  resetSamplerToDefaults: mockResetSamplerToDefaults,
  getAllPresets: vi.fn(() => [builtInPreset, customPreset]),
  loadPreset: mockLoadPreset,
  savePreset: mockSavePreset,
  deletePreset: mockDeletePreset,
  activeSamplerPresetId: null as string | null,
  streamTimeoutMs: 30_000,
  setStreamTimeoutMs: vi.fn(),
}

vi.mock('../../stores/settingsStore', () => ({
  useSettingsStore: vi.fn(() => defaultSettingsState),
}))

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    setActivePanel: mockSetActivePanel,
  })),
}))

const defaultModelState = {
  loadedModel: null as { capabilities?: { thinking?: boolean } } | null,
}

vi.mock('../../stores/modelStore', () => ({
  useModelStore: vi.fn((sel?: any) => typeof sel === 'function' ? sel(defaultModelState) : defaultModelState),
}))

// Mock SamplerControls to avoid rendering all sliders
vi.mock('../composed/SamplerControls', () => ({
  SamplerControls: ({ settings }: { settings: Record<string, unknown> }) => (
    <div data-testid="mock-sampler-controls">
      Temperature: {String(settings.temperature)}
    </div>
  ),
}))

import { useSettingsStore } from '../../stores/settingsStore'
import { useModelStore } from '../../stores/modelStore'

function renderPanel(overrides: Partial<typeof defaultSettingsState> = {}, modelOverrides: Partial<typeof defaultModelState> = {}) {
  const mergedSettings = { ...defaultSettingsState, ...overrides }
  const mergedModel = { ...defaultModelState, ...modelOverrides }
  vi.mocked(useSettingsStore).mockReturnValue(mergedSettings as ReturnType<typeof useSettingsStore>)
  vi.mocked(useModelStore).mockImplementation((sel?: any) => typeof sel === 'function' ? sel(mergedModel) : mergedModel)
  return render(<SettingsPanel />)
}

describe('SettingsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    defaultSettingsState.getAllPresets = vi.fn(() => [builtInPreset, customPreset])
    defaultSettingsState.activeSamplerPresetId = null
  })

  describe('rendering', () => {
    it('renders the heading', () => {
      renderPanel()
      expect(screen.getByText('Generation Settings')).toBeInTheDocument()
    })

    it('renders the description', () => {
      renderPanel()
      expect(screen.getByText('Control how the AI generates responses')).toBeInTheDocument()
    })

    it('renders SamplerControls', () => {
      renderPanel()
      expect(screen.getByTestId('mock-sampler-controls')).toBeInTheDocument()
    })

    it('renders Reset to Defaults button', () => {
      renderPanel()
      expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument()
    })
  })

  describe('close button', () => {
    it('calls setActivePanel(null) when close button is clicked', () => {
      renderPanel()
      // Close button is the only button with CloseIcon in the header
      const buttons = screen.getAllByRole('button')
      const closeButton = buttons.find(b => b.closest('.border-b'))
      fireEvent.click(closeButton!)
      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })
  })

  describe('presets', () => {
    it('renders preset buttons', () => {
      renderPanel()
      expect(screen.getByText('Balanced')).toBeInTheDocument()
      expect(screen.getByText('My Preset')).toBeInTheDocument()
    })

    it('calls loadPreset when a preset is clicked', () => {
      renderPanel()
      fireEvent.click(screen.getByText('Balanced'))
      expect(mockLoadPreset).toHaveBeenCalledWith(builtInPreset)
    })

    it('highlights active preset', () => {
      renderPanel({ activeSamplerPresetId: 'builtin-1' })
      const presetButton = screen.getByText('Balanced')
      expect(presetButton.className).toContain('bg-primary')
    })

    it('shows delete button only for custom presets', () => {
      renderPanel()
      const deleteButtons = screen.getAllByLabelText(/Delete preset/)
      expect(deleteButtons).toHaveLength(1)
      expect(deleteButtons[0]).toHaveAttribute('aria-label', 'Delete preset My Preset')
    })

    it('calls deletePreset when delete button is clicked', () => {
      renderPanel()
      const deleteButton = screen.getByLabelText('Delete preset My Preset')
      fireEvent.click(deleteButton)
      expect(mockDeletePreset).toHaveBeenCalledWith('custom-1')
    })
  })

  describe('save preset workflow', () => {
    it('shows "Save as Preset" button initially', () => {
      renderPanel()
      expect(screen.getByText('+ Save as Preset')).toBeInTheDocument()
    })

    it('shows input when "Save as Preset" is clicked', () => {
      renderPanel()
      fireEvent.click(screen.getByText('+ Save as Preset'))
      expect(screen.getByPlaceholderText('Preset name...')).toBeInTheDocument()
    })

    it('saves preset on Enter key', async () => {
      const user = userEvent.setup()
      renderPanel()
      fireEvent.click(screen.getByText('+ Save as Preset'))

      const input = screen.getByPlaceholderText('Preset name...')
      await user.type(input, 'New Preset')
      fireEvent.keyDown(input, { key: 'Enter' })

      expect(mockSavePreset).toHaveBeenCalledWith('sampler', 'New Preset')
    })

    it('cancels on Escape key', () => {
      renderPanel()
      fireEvent.click(screen.getByText('+ Save as Preset'))

      const input = screen.getByPlaceholderText('Preset name...')
      fireEvent.keyDown(input, { key: 'Escape' })

      // Input should be gone, "Save as Preset" button should be back
      expect(screen.queryByPlaceholderText('Preset name...')).not.toBeInTheDocument()
      expect(screen.getByText('+ Save as Preset')).toBeInTheDocument()
    })

    it('disables Save button when name is empty', () => {
      renderPanel()
      fireEvent.click(screen.getByText('+ Save as Preset'))

      const saveButton = screen.getByRole('button', { name: 'Save' })
      expect(saveButton).toBeDisabled()
    })

    it('does not save when name is only whitespace', () => {
      renderPanel()
      fireEvent.click(screen.getByText('+ Save as Preset'))

      const input = screen.getByPlaceholderText('Preset name...')
      fireEvent.change(input, { target: { value: '   ' } })
      fireEvent.keyDown(input, { key: 'Enter' })

      expect(mockSavePreset).not.toHaveBeenCalled()
    })

    it('shows Cancel button in creation mode', () => {
      renderPanel()
      fireEvent.click(screen.getByText('+ Save as Preset'))
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument()
    })
  })

  describe('thinking mode', () => {
    it('does not show thinking toggle when model has no thinking capability', () => {
      renderPanel()
      expect(screen.queryByText('Thinking Mode')).not.toBeInTheDocument()
    })

    it('shows thinking toggle when model supports thinking', () => {
      renderPanel({}, { loadedModel: { capabilities: { thinking: true } } })
      expect(screen.getByText('Thinking Mode')).toBeInTheDocument()
    })

    it('does not show thinking toggle when model is null', () => {
      renderPanel({}, { loadedModel: null })
      expect(screen.queryByText('Thinking Mode')).not.toBeInTheDocument()
    })
  })

  describe('reset to defaults', () => {
    it('calls resetSamplerToDefaults when clicked', () => {
      renderPanel()
      fireEvent.click(screen.getByRole('button', { name: 'Reset to Defaults' }))
      expect(mockResetSamplerToDefaults).toHaveBeenCalledTimes(1)
    })
  })
})
