import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useModelsStore } from './modelsStore'
import type { AdminModelConfig, ScannedModel, ProfileInfo } from '../types'

// Mock the API client
const mockFetchAPI = vi.fn()
const mockPostAPI = vi.fn()

vi.mock('../../../api/client', () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
  postAPI: (...args: unknown[]) => mockPostAPI(...args),
}))

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

function mockScannedModel(overrides: Partial<ScannedModel> = {}): ScannedModel {
  return {
    id: 'scanned-model',
    path: '/path/to/scanned',
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

describe('modelsStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useModelsStore.setState({
      configs: [],
      selectedId: null,
      searchQuery: '',
      filters: { provider: [], status: [], capability: [], tag: [] },
      profiles: [],
      importOpen: false,
      scanResults: [],
      scanning: false,
      importing: false,
      loading: false,
      error: null,
      actionLoading: null,
      profilesLoaded: false,
    })
  })

  // --- Initial state ---

  describe('initial state', () => {
    it('starts with empty configs', () => {
      expect(useModelsStore.getState().configs).toEqual([])
    })

    it('starts with no selected model', () => {
      expect(useModelsStore.getState().selectedId).toBeNull()
    })

    it('starts with empty search query', () => {
      expect(useModelsStore.getState().searchQuery).toBe('')
    })

    it('starts with empty filters', () => {
      expect(useModelsStore.getState().filters).toEqual({ provider: [], status: [], capability: [], tag: [] })
    })

    it('starts with no loading or error', () => {
      const s = useModelsStore.getState()
      expect(s.loading).toBe(false)
      expect(s.error).toBeNull()
      expect(s.actionLoading).toBeNull()
      expect(s.profilesLoaded).toBe(false)
    })
  })

  // --- fetchConfigs ---

  describe('fetchConfigs', () => {
    it('fetches configs and sets them on success', async () => {
      const models = [mockModel(), mockModel({ id: 'model-2' })]
      mockFetchAPI.mockResolvedValue({ models })

      await useModelsStore.getState().fetchConfigs()

      expect(useModelsStore.getState().configs).toEqual(models)
      expect(useModelsStore.getState().loading).toBe(false)
      expect(useModelsStore.getState().error).toBeNull()
    })

    it('sets loading true during fetch', async () => {
      let resolvePromise: (v: unknown) => void
      mockFetchAPI.mockReturnValue(new Promise((r) => { resolvePromise = r }))

      const promise = useModelsStore.getState().fetchConfigs()
      expect(useModelsStore.getState().loading).toBe(true)

      resolvePromise!({ models: [] })
      await promise
      expect(useModelsStore.getState().loading).toBe(false)
    })

    it('sets error on failure', async () => {
      mockFetchAPI.mockRejectedValue(new Error('Network error'))

      await useModelsStore.getState().fetchConfigs()

      expect(useModelsStore.getState().error).toBe('Network error')
      expect(useModelsStore.getState().loading).toBe(false)
    })
  })

  // --- fetchProfiles ---

  describe('fetchProfiles', () => {
    it('fetches profiles and sets profilesLoaded', async () => {
      mockFetchAPI.mockResolvedValue({ profiles: mockProfiles })

      await useModelsStore.getState().fetchProfiles()

      expect(useModelsStore.getState().profiles).toEqual(mockProfiles)
      expect(useModelsStore.getState().profilesLoaded).toBe(true)
    })

    it('sets profilesLoaded even on failure', async () => {
      mockFetchAPI.mockRejectedValue(new Error('fail'))

      await useModelsStore.getState().fetchProfiles()

      expect(useModelsStore.getState().profilesLoaded).toBe(true)
      expect(useModelsStore.getState().profiles).toEqual([])
    })
  })

  // --- Simple setters ---

  describe('setters', () => {
    it('setSelectedId updates selection', () => {
      useModelsStore.getState().setSelectedId('model-1')
      expect(useModelsStore.getState().selectedId).toBe('model-1')
    })

    it('setSearchQuery updates query', () => {
      useModelsStore.getState().setSearchQuery('llama')
      expect(useModelsStore.getState().searchQuery).toBe('llama')
    })

    it('setFilters merges filters', () => {
      useModelsStore.getState().setFilters({ provider: ['mlx'] })
      expect(useModelsStore.getState().filters.provider).toEqual(['mlx'])
      expect(useModelsStore.getState().filters.status).toEqual([])
    })

    it('setImportOpen resets scan state', () => {
      useModelsStore.setState({ scanResults: [mockScannedModel()], scanning: true })
      useModelsStore.getState().setImportOpen(true)
      expect(useModelsStore.getState().importOpen).toBe(true)
      expect(useModelsStore.getState().scanResults).toEqual([])
      expect(useModelsStore.getState().scanning).toBe(false)
    })
  })

  // --- updateConfig ---

  describe('updateConfig', () => {
    it('updates a model in configs on success', async () => {
      const original = mockModel()
      useModelsStore.setState({ configs: [original] })

      const updated = mockModel({ description: 'Updated' })
      mockFetchAPI.mockResolvedValue({ model: updated, reload_required_fields: [] })

      const fields = await useModelsStore.getState().updateConfig('test-model', { description: 'Updated' })

      expect(fields).toEqual([])
      expect(useModelsStore.getState().configs[0].description).toBe('Updated')
    })

    it('returns reload_required_fields from API', async () => {
      useModelsStore.setState({ configs: [mockModel()] })
      mockFetchAPI.mockResolvedValue({ model: mockModel(), reload_required_fields: ['model_path'] })

      const fields = await useModelsStore.getState().updateConfig('test-model', { config: { model_path: '/new' } })
      expect(fields).toEqual(['model_path'])
    })

    it('sets error and re-throws on failure', async () => {
      useModelsStore.setState({ configs: [mockModel()] })
      mockFetchAPI.mockRejectedValue(new Error('Bad request'))

      await expect(useModelsStore.getState().updateConfig('test-model', {})).rejects.toThrow('Bad request')
      expect(useModelsStore.getState().error).toBe('Bad request')
    })
  })

  // --- removeConfig ---

  describe('removeConfig', () => {
    it('removes model from configs and clears selection if selected', async () => {
      useModelsStore.setState({ configs: [mockModel()], selectedId: 'test-model' })
      mockFetchAPI.mockResolvedValue({})

      await useModelsStore.getState().removeConfig('test-model')

      expect(useModelsStore.getState().configs).toEqual([])
      expect(useModelsStore.getState().selectedId).toBeNull()
    })

    it('does not clear selection if different model removed', async () => {
      useModelsStore.setState({ configs: [mockModel(), mockModel({ id: 'other' })], selectedId: 'other' })
      mockFetchAPI.mockResolvedValue({})

      await useModelsStore.getState().removeConfig('test-model')

      expect(useModelsStore.getState().selectedId).toBe('other')
    })

    it('sets error and re-throws on failure', async () => {
      useModelsStore.setState({ configs: [mockModel()] })
      mockFetchAPI.mockRejectedValue(new Error('Not found'))

      await expect(useModelsStore.getState().removeConfig('test-model')).rejects.toThrow('Not found')
      expect(useModelsStore.getState().error).toBe('Not found')
    })
  })

  // --- toggleEnabled ---

  describe('toggleEnabled', () => {
    it('toggles enabled state and sets actionLoading', async () => {
      const model = mockModel({ enabled: true })
      useModelsStore.setState({ configs: [model] })
      mockPostAPI.mockResolvedValue({ ...model, enabled: false })

      await useModelsStore.getState().toggleEnabled('test-model')

      expect(useModelsStore.getState().configs[0].enabled).toBe(false)
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })

    it('sets error on failure and clears actionLoading', async () => {
      useModelsStore.setState({ configs: [mockModel()] })
      mockPostAPI.mockRejectedValue(new Error('Toggle failed'))

      await useModelsStore.getState().toggleEnabled('test-model')

      expect(useModelsStore.getState().error).toBe('Toggle failed')
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })
  })

  // --- scanForModels ---

  describe('scanForModels', () => {
    it('sets scan results on success', async () => {
      const scanned = [mockScannedModel()]
      mockPostAPI.mockResolvedValue({ models: scanned })

      await useModelsStore.getState().scanForModels({ paths: [], scan_hf_cache: true })

      expect(useModelsStore.getState().scanResults).toEqual(scanned)
      expect(useModelsStore.getState().scanning).toBe(false)
    })

    it('sets scanning flag during scan', async () => {
      let resolvePromise: (v: unknown) => void
      mockPostAPI.mockReturnValue(new Promise((r) => { resolvePromise = r }))

      const promise = useModelsStore.getState().scanForModels({ paths: [], scan_hf_cache: true })
      expect(useModelsStore.getState().scanning).toBe(true)

      resolvePromise!({ models: [] })
      await promise
      expect(useModelsStore.getState().scanning).toBe(false)
    })

    it('sets error on failure', async () => {
      mockPostAPI.mockRejectedValue(new Error('Scan error'))

      await useModelsStore.getState().scanForModels({ paths: [], scan_hf_cache: true })

      expect(useModelsStore.getState().error).toBe('Scan error')
      expect(useModelsStore.getState().scanning).toBe(false)
    })
  })

  // --- importModels ---

  describe('importModels', () => {
    it('closes modal and refreshes configs on success', async () => {
      useModelsStore.setState({ importOpen: true, scanResults: [mockScannedModel()] })
      mockPostAPI.mockResolvedValue({})
      mockFetchAPI.mockResolvedValue({ models: [mockModel()] })

      await useModelsStore.getState().importModels({ models: [{ id: 'x' }] })

      expect(useModelsStore.getState().importOpen).toBe(false)
      expect(useModelsStore.getState().scanResults).toEqual([])
      expect(useModelsStore.getState().importing).toBe(false)
      // fetchConfigs was called to refresh
      expect(mockFetchAPI).toHaveBeenCalled()
    })

    it('sets error on failure without closing modal', async () => {
      useModelsStore.setState({ importOpen: true })
      mockPostAPI.mockRejectedValue(new Error('Import error'))

      await useModelsStore.getState().importModels({ models: [{ id: 'x' }] })

      expect(useModelsStore.getState().error).toBe('Import error')
      expect(useModelsStore.getState().importing).toBe(false)
      // Modal stays open on error
    })

    it('sets importing flag during import', async () => {
      let resolvePromise: (v: unknown) => void
      mockPostAPI.mockReturnValue(new Promise((r) => { resolvePromise = r }))

      const promise = useModelsStore.getState().importModels({ models: [{ id: 'x' }] })
      expect(useModelsStore.getState().importing).toBe(true)

      resolvePromise!({})
      mockFetchAPI.mockResolvedValue({ models: [] })
      await promise
      expect(useModelsStore.getState().importing).toBe(false)
    })
  })

  // --- loadModel / unloadModel ---

  describe('loadModel', () => {
    it('sets loaded true on success and clears actionLoading', async () => {
      useModelsStore.setState({ configs: [mockModel({ loaded: false })] })
      mockPostAPI.mockResolvedValue({})

      await useModelsStore.getState().loadModel('test-model')

      expect(useModelsStore.getState().configs[0].loaded).toBe(true)
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })

    it('sets error on failure and clears actionLoading', async () => {
      useModelsStore.setState({ configs: [mockModel()] })
      mockPostAPI.mockRejectedValue(new Error('Load failed'))

      await useModelsStore.getState().loadModel('test-model')

      expect(useModelsStore.getState().error).toBe('Load failed')
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })
  })

  describe('unloadModel', () => {
    it('sets loaded false on success and clears actionLoading', async () => {
      useModelsStore.setState({ configs: [mockModel({ loaded: true })] })
      mockPostAPI.mockResolvedValue({})

      await useModelsStore.getState().unloadModel('test-model')

      expect(useModelsStore.getState().configs[0].loaded).toBe(false)
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })

    it('sets error on failure and clears actionLoading', async () => {
      useModelsStore.setState({ configs: [mockModel({ loaded: true })] })
      mockPostAPI.mockRejectedValue(new Error('Unload failed'))

      await useModelsStore.getState().unloadModel('test-model')

      expect(useModelsStore.getState().error).toBe('Unload failed')
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })
  })

  // --- applyProfile ---

  describe('applyProfile', () => {
    it('refreshes configs on success and clears actionLoading', async () => {
      useModelsStore.setState({ configs: [mockModel()] })
      mockPostAPI.mockResolvedValue({})
      mockFetchAPI.mockResolvedValue({ models: [mockModel()] })

      await useModelsStore.getState().applyProfile(['test-model'], 'balanced')

      expect(mockPostAPI).toHaveBeenCalledWith('/v1/admin/models/bulk-profile', {
        model_ids: ['test-model'],
        profile: 'balanced',
      })
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })

    it('sets error on failure and clears actionLoading', async () => {
      mockPostAPI.mockRejectedValue(new Error('Profile error'))

      await useModelsStore.getState().applyProfile(['test-model'], 'bad-profile')

      expect(useModelsStore.getState().error).toBe('Profile error')
      expect(useModelsStore.getState().actionLoading).toBeNull()
    })
  })
})
