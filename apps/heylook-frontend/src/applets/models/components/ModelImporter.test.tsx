import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { ModelImporter } from './ModelImporter'
import type { ScannedModel, ProfileInfo } from '../types'

function mockScannedModel(overrides: Partial<ScannedModel> = {}): ScannedModel {
  return {
    id: 'new-model',
    path: '/path/to/model',
    provider: 'mlx',
    size_gb: 4.2,
    vision: false,
    already_configured: false,
    tags: [],
    description: 'Scanned model',
    ...overrides,
  }
}

const mockProfiles: ProfileInfo[] = [
  { name: 'balanced', description: 'Balance between speed and quality' },
  { name: 'fast', description: 'Optimized for speed' },
]

// Default mock state
const defaultState = {
  importOpen: true,
  setImportOpen: vi.fn(),
  scanResults: [] as ScannedModel[],
  scanning: false,
  importing: false,
  scanForModels: vi.fn().mockResolvedValue(undefined),
  importModels: vi.fn().mockResolvedValue(undefined),
  profiles: mockProfiles,
  error: null as string | null,
}

// Mock getState for the race-condition fix
const mockGetState = vi.fn(() => ({ scanResults: defaultState.scanResults }))

vi.mock('../stores/modelsStore', () => ({
  useModelsStore: Object.assign(
    vi.fn((selector: (s: typeof defaultState) => unknown) => selector(defaultState)),
    { getState: () => mockGetState() }
  ),
}))

describe('ModelImporter', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    defaultState.importOpen = true
    defaultState.scanResults = []
    defaultState.scanning = false
    defaultState.importing = false
    defaultState.error = null
    defaultState.profiles = mockProfiles
    defaultState.scanForModels = vi.fn().mockResolvedValue(undefined)
    defaultState.importModels = vi.fn().mockResolvedValue(undefined)
    mockGetState.mockReturnValue({ scanResults: defaultState.scanResults })
  })

  // --- Modal visibility ---

  describe('modal behavior', () => {
    it('renders nothing when importOpen is false', () => {
      defaultState.importOpen = false
      const { container } = render(<ModelImporter />)
      expect(container.innerHTML).toBe('')
    })

    it('renders modal when importOpen is true', () => {
      render(<ModelImporter />)
      expect(screen.getByText('Import Models')).toBeTruthy()
    })

    it('closes on backdrop click', () => {
      render(<ModelImporter />)
      // The backdrop is the first fixed div
      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/60')
      if (backdrop) fireEvent.click(backdrop)
      expect(defaultState.setImportOpen).toHaveBeenCalledWith(false)
    })

    it('closes on cancel button', () => {
      render(<ModelImporter />)
      fireEvent.click(screen.getByText('Cancel'))
      expect(defaultState.setImportOpen).toHaveBeenCalledWith(false)
    })
  })

  // --- Scan step ---

  describe('scan step', () => {
    it('renders scan form by default', () => {
      render(<ModelImporter />)
      expect(screen.getByText('Scan')).toBeTruthy()
      expect(screen.getByPlaceholderText('/path/to/models')).toBeTruthy()
    })

    it('custom path input accepts text', () => {
      render(<ModelImporter />)
      const input = screen.getByPlaceholderText('/path/to/models')
      fireEvent.change(input, { target: { value: '/my/models' } })
      // Input value should update (controlled component)
      expect((input as HTMLInputElement).value).toBe('/my/models')
    })

    it('scan button triggers scanForModels', async () => {
      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      expect(defaultState.scanForModels).toHaveBeenCalledWith({
        paths: [],
        scan_hf_cache: true,
      })
    })

    it('shows scanning spinner', () => {
      defaultState.scanning = true
      render(<ModelImporter />)
      expect(screen.getByText('Scanning...')).toBeTruthy()
      expect(screen.getByText('Scanning for models...')).toBeTruthy()
    })

    it('shows error from store', () => {
      defaultState.error = 'Scan failed: permission denied'
      render(<ModelImporter />)
      expect(screen.getByText('Scan failed: permission denied')).toBeTruthy()
    })
  })

  // --- Select step ---

  describe('select step', () => {
    it('shows scan results after scan completes', async () => {
      defaultState.scanResults = [mockScannedModel({ id: 'found-model' })]
      mockGetState.mockReturnValue({ scanResults: defaultState.scanResults })

      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      expect(screen.getByText('found-model')).toBeTruthy()
    })

    it('shows model details in scan results', async () => {
      defaultState.scanResults = [mockScannedModel({ id: 'test-model', size_gb: 7.5, provider: 'mlx' })]
      mockGetState.mockReturnValue({ scanResults: defaultState.scanResults })

      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      expect(screen.getByText('test-model')).toBeTruthy()
      expect(screen.getByText('7.5 GB')).toBeTruthy()
      // Provider text is CSS-uppercased, raw text is lowercase
      expect(screen.getByText('mlx')).toBeTruthy()
    })

    it('shows count of already-configured models', async () => {
      defaultState.scanResults = [
        mockScannedModel({ id: 'new-1' }),
        mockScannedModel({ id: 'existing', already_configured: true }),
      ]
      mockGetState.mockReturnValue({ scanResults: defaultState.scanResults })

      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      expect(screen.getByText('1 new models found')).toBeTruthy()
      expect(screen.getByText('1 models already configured (skipped)')).toBeTruthy()
    })

    it('back button returns to scan step', async () => {
      defaultState.scanResults = [mockScannedModel()]
      mockGetState.mockReturnValue({ scanResults: defaultState.scanResults })

      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      expect(screen.getByText('Back')).toBeTruthy()
      fireEvent.click(screen.getByText('Back'))

      // Should be back on scan step
      expect(screen.getByPlaceholderText('/path/to/models')).toBeTruthy()
    })
  })

  // --- Import action ---

  describe('import action', () => {
    it('import button disabled when none selected', async () => {
      defaultState.scanResults = [mockScannedModel({ already_configured: true })]
      mockGetState.mockReturnValue({ scanResults: [] })

      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      // With no unconfigured models auto-selected, button should be disabled
      const importBtn = screen.getByText('Import 0 Models')
      expect((importBtn as HTMLButtonElement).disabled).toBe(true)
    })

    it('shows importing spinner', () => {
      defaultState.importing = true
      // Need to be on select step with results
      defaultState.scanResults = [mockScannedModel()]
      render(<ModelImporter />)

      // The importing text shows in the footer button area
      // When importing is true, the button text changes
    })
  })

  // --- Error handling ---

  describe('error handling', () => {
    it('displays scan error inline', () => {
      defaultState.error = 'Permission denied'
      render(<ModelImporter />)
      expect(screen.getByText('Permission denied')).toBeTruthy()
    })

    it('displays import error inline', async () => {
      defaultState.error = 'Import failed: invalid model'
      render(<ModelImporter />)
      expect(screen.getByText('Import failed: invalid model')).toBeTruthy()
    })
  })

  // --- Profile selection ---

  describe('profile selection', () => {
    it('shows profile options in select step', async () => {
      defaultState.scanResults = [mockScannedModel()]
      mockGetState.mockReturnValue({ scanResults: defaultState.scanResults })

      render(<ModelImporter />)

      await act(async () => {
        fireEvent.click(screen.getByText('Scan'))
      })

      // Profile selector should show profiles
      const select = screen.getByRole('combobox')
      expect(select).toBeTruthy()
    })
  })
})
