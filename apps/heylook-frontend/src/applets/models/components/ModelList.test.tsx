import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ModelList } from './ModelList'
import type { AdminModelConfig, ModelFilter } from '../types'

function mockModel(overrides: Partial<AdminModelConfig> = {}): AdminModelConfig {
  return {
    id: 'test-model',
    provider: 'mlx',
    description: 'Test model',
    tags: ['test'],
    enabled: true,
    capabilities: ['chat'],
    config: { model_path: '/path/to/model', vision: false },
    loaded: false,
    ...overrides,
  }
}

// Default mock state
const defaultState = {
  configs: [] as AdminModelConfig[],
  selectedId: null as string | null,
  setSelectedId: vi.fn(),
  searchQuery: '',
  setSearchQuery: vi.fn(),
  filters: { provider: [], status: [], capability: [], tag: [] } as ModelFilter,
  setFilters: vi.fn(),
  sortConfig: { field: 'name' as const, direction: 'asc' as const },
  setSortConfig: vi.fn(),
  setImportOpen: vi.fn(),
}

vi.mock('../stores/modelsStore', () => ({
  useModelsStore: vi.fn((selector: (s: typeof defaultState) => unknown) => selector(defaultState)),
}))

describe('ModelList', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    defaultState.configs = []
    defaultState.selectedId = null
    defaultState.searchQuery = ''
    defaultState.filters = { provider: [], status: [], capability: [], tag: [] }
  })

  // --- Rendering ---

  describe('rendering', () => {
    it('shows empty state when no models configured', () => {
      render(<ModelList />)
      expect(screen.getByText('No models configured')).toBeTruthy()
    })

    it('shows "no matches" when filters exclude all models', () => {
      defaultState.configs = [mockModel()]
      defaultState.filters = { provider: ['gguf'], status: [], capability: [], tag: [] }
      render(<ModelList />)
      expect(screen.getByText('No models match filters')).toBeTruthy()
    })

    it('renders models with status pills', () => {
      defaultState.configs = [
        mockModel({ id: 'loaded-model', loaded: true, enabled: true }),
        mockModel({ id: 'available-model', loaded: false, enabled: true }),
        mockModel({ id: 'disabled-model', enabled: false }),
      ]
      render(<ModelList />)

      expect(screen.getByText('loaded-model')).toBeTruthy()
      expect(screen.getByText('Loaded')).toBeTruthy()
      expect(screen.getByText('available-model')).toBeTruthy()
      expect(screen.getByText('Available')).toBeTruthy()
      expect(screen.getByText('disabled-model')).toBeTruthy()
      expect(screen.getByText('Disabled')).toBeTruthy()
    })

    it('highlights selected model', () => {
      defaultState.configs = [mockModel({ id: 'model-a' }), mockModel({ id: 'model-b' })]
      defaultState.selectedId = 'model-a'
      render(<ModelList />)

      const buttons = screen.getAllByRole('button')
      const modelABtn = buttons.find(b => b.textContent?.includes('model-a'))
      expect(modelABtn?.className).toContain('bg-primary/10')
    })

    it('shows footer count', () => {
      defaultState.configs = [mockModel({ id: 'a' }), mockModel({ id: 'b' })]
      render(<ModelList />)
      expect(screen.getByText('2 of 2 models')).toBeTruthy()
    })
  })

  // --- Search ---

  describe('search', () => {
    it('filters by model id', () => {
      defaultState.configs = [mockModel({ id: 'llama-3' }), mockModel({ id: 'qwen-2' })]
      defaultState.searchQuery = 'llama'
      render(<ModelList />)

      expect(screen.getByText('llama-3')).toBeTruthy()
      expect(screen.queryByText('qwen-2')).toBeNull()
    })

    it('filters by description', () => {
      defaultState.configs = [
        mockModel({ id: 'a', description: 'A fast model' }),
        mockModel({ id: 'b', description: 'A slow model' }),
      ]
      defaultState.searchQuery = 'fast'
      render(<ModelList />)

      expect(screen.getByText('a')).toBeTruthy()
      expect(screen.queryByText('b')).toBeNull()
    })

    it('filters by tags', () => {
      defaultState.configs = [
        mockModel({ id: 'a', tags: ['vision'] }),
        mockModel({ id: 'b', tags: ['text'] }),
      ]
      defaultState.searchQuery = 'vision'
      render(<ModelList />)

      expect(screen.getByText('a')).toBeTruthy()
      expect(screen.queryByText('b')).toBeNull()
    })
  })

  // --- Filters ---

  describe('filters', () => {
    it('filters by provider', () => {
      defaultState.configs = [
        mockModel({ id: 'mlx-model', provider: 'mlx' }),
        mockModel({ id: 'gguf-model', provider: 'gguf' }),
      ]
      defaultState.filters = { provider: ['mlx'], status: [], capability: [], tag: [] }
      render(<ModelList />)

      expect(screen.getByText('mlx-model')).toBeTruthy()
      expect(screen.queryByText('gguf-model')).toBeNull()
    })

    it('filters by status', () => {
      defaultState.configs = [
        mockModel({ id: 'model-active', loaded: true, enabled: true }),
        mockModel({ id: 'model-idle', loaded: false, enabled: true }),
      ]
      defaultState.filters = { provider: [], status: ['loaded'], capability: [], tag: [] }
      render(<ModelList />)

      expect(screen.getByText('model-active')).toBeTruthy()
      expect(screen.queryByText('model-idle')).toBeNull()
    })

    it('filters by capability', () => {
      defaultState.configs = [
        mockModel({ id: 'vision-model', capabilities: ['chat', 'vision'] }),
        mockModel({ id: 'text-model', capabilities: ['chat'] }),
      ]
      defaultState.filters = { provider: [], status: [], capability: ['vision'], tag: [] }
      render(<ModelList />)

      expect(screen.getByText('vision-model')).toBeTruthy()
      expect(screen.queryByText('text-model')).toBeNull()
    })

    it('combines search and filters', () => {
      defaultState.configs = [
        mockModel({ id: 'mlx-llama', provider: 'mlx' }),
        mockModel({ id: 'gguf-llama', provider: 'gguf' }),
        mockModel({ id: 'mlx-qwen', provider: 'mlx' }),
      ]
      defaultState.searchQuery = 'llama'
      defaultState.filters = { provider: ['mlx'], status: [], capability: [], tag: [] }
      render(<ModelList />)

      expect(screen.getByText('mlx-llama')).toBeTruthy()
      expect(screen.queryByText('gguf-llama')).toBeNull()
      expect(screen.queryByText('mlx-qwen')).toBeNull()
    })
  })

  // --- Interactions ---

  describe('interactions', () => {
    it('clicking a model selects it', () => {
      defaultState.configs = [mockModel({ id: 'model-a' })]
      render(<ModelList />)

      fireEvent.click(screen.getByText('model-a'))
      expect(defaultState.setSelectedId).toHaveBeenCalledWith('model-a')
    })

    it('import button opens modal', () => {
      render(<ModelList />)
      fireEvent.click(screen.getByText('Import'))
      expect(defaultState.setImportOpen).toHaveBeenCalledWith(true)
    })

    it('search input calls setSearchQuery', () => {
      render(<ModelList />)
      const input = screen.getByPlaceholderText('Search models...')
      fireEvent.change(input, { target: { value: 'test' } })
      expect(defaultState.setSearchQuery).toHaveBeenCalledWith('test')
    })
  })

  // --- Filter chips ---

  describe('filter chips', () => {
    it('hides chips when no filters active', () => {
      render(<ModelList />)
      // No chip elements should render
      expect(screen.queryByText('x')).toBeNull()
    })

    it('shows chips for active filters', () => {
      defaultState.filters = { provider: ['gguf'], status: ['disabled'], capability: [], tag: [] }
      render(<ModelList />)

      expect(screen.getByText('gguf')).toBeTruthy()
      expect(screen.getByText('disabled')).toBeTruthy()
    })

    it('removing a chip updates filters', () => {
      defaultState.filters = { provider: ['gguf', 'mlx_stt'], status: [], capability: [], tag: [] }
      render(<ModelList />)

      // Click the 'x' button next to 'gguf'
      const chip = screen.getByText('gguf')
      const removeBtn = chip.parentElement?.querySelector('button')
      if (removeBtn) fireEvent.click(removeBtn)

      expect(defaultState.setFilters).toHaveBeenCalledWith({ provider: ['mlx_stt'] })
    })
  })
})
