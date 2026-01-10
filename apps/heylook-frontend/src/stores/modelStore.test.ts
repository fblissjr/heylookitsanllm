import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { useModelStore, getModelCapabilities } from './modelStore'
import type { Model, ServerCapabilities } from '../types/api'
import type { LoadedModel } from '../types/models'

// Mock data
const mockModels: Model[] = [
  {
    id: 'test-model-1',
    object: 'model',
    owned_by: 'test',
    provider: 'mlx',
    capabilities: ['chat', 'vision'],
    context_window: 4096,
  },
  {
    id: 'test-model-2',
    object: 'model',
    owned_by: 'test',
    provider: 'llama_cpp',
    capabilities: ['chat', 'thinking', 'hidden_states'],
    context_window: 8192,
  },
  {
    id: 'test-model-3',
    object: 'model',
    owned_by: 'test',
    provider: 'mlx',
    capabilities: ['embeddings'],
  },
]

const mockCapabilities: ServerCapabilities = {
  server_version: '1.0.0',
  optimizations: {
    json: {
      orjson_available: true,
      speedup: '2x',
    },
    image: {
      xxhash_available: true,
      turbojpeg_available: false,
      cachetools_available: true,
      hash_speedup: '10x',
      jpeg_speedup: '0x',
    },
  },
  metal: {
    available: true,
    device_name: 'Apple M1',
    max_recommended_working_set_size: 16000000000,
  },
  endpoints: {
    fast_vision: {
      available: true,
      endpoint: '/v1/chat/completions/multipart',
      description: 'Fast vision endpoint',
      benefits: {
        time_saved_per_image_ms: 57,
        bandwidth_reduction_percent: 30,
        supports_parallel_processing: true,
      },
    },
    batch_processing: {
      available: true,
      processing_modes: ['parallel', 'sequential'],
    },
  },
  features: {
    streaming: true,
    model_caching: {
      enabled: true,
      cache_size: 2,
      eviction_policy: 'lru',
    },
    vision_models: true,
    concurrent_requests: true,
    supported_image_formats: ['png', 'jpeg', 'webp'],
  },
  recommendations: {
    vision_models: {
      use_multipart: true,
      reason: 'Faster processing',
    },
    batch_size: {
      optimal: 10,
      max: 50,
    },
    image_format: {
      preferred: 'jpeg',
      quality: 85,
    },
  },
  limits: {
    max_tokens: 4096,
    max_images_per_request: 10,
    max_request_size_mb: 20,
    timeout_seconds: 300,
  },
}

const mockLoadedModel: LoadedModel = {
  id: 'test-model-1',
  provider: 'mlx',
  capabilities: {
    chat: true,
    vision: true,
    thinking: false,
    hidden_states: false,
    embeddings: false,
  },
  contextWindow: 4096,
  loadedAt: Date.now(),
}

describe('modelStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useModelStore.setState({
      models: [],
      loadedModel: null,
      capabilities: null,
      modelStatus: 'unloaded',
      error: null,
    })
    // Clear all mocks
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('initial state', () => {
    it('has empty models array', () => {
      const { models } = useModelStore.getState()
      expect(models).toEqual([])
    })

    it('has null loadedModel', () => {
      const { loadedModel } = useModelStore.getState()
      expect(loadedModel).toBeNull()
    })

    it('has null capabilities', () => {
      const { capabilities } = useModelStore.getState()
      expect(capabilities).toBeNull()
    })

    it('has unloaded modelStatus', () => {
      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('unloaded')
    })

    it('has null error', () => {
      const { error } = useModelStore.getState()
      expect(error).toBeNull()
    })
  })

  describe('fetchModels', () => {
    it('fetches models successfully', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ data: mockModels }),
      })
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()
      await fetchModels()

      const { models, error } = useModelStore.getState()
      expect(models).toEqual(mockModels)
      expect(error).toBeNull()
      expect(mockFetch).toHaveBeenCalledWith('/v1/models')
    })

    it('handles empty models response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ data: [] }),
      })
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()
      await fetchModels()

      const { models, error } = useModelStore.getState()
      expect(models).toEqual([])
      expect(error).toBeNull()
    })

    it('handles missing data field in response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      })
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()
      await fetchModels()

      const { models, error } = useModelStore.getState()
      expect(models).toEqual([])
      expect(error).toBeNull()
    })

    it('sets error on HTTP error response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        statusText: 'Internal Server Error',
      })
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()

      await expect(fetchModels()).rejects.toThrow('Failed to fetch models: Internal Server Error')

      const { error } = useModelStore.getState()
      expect(error).toBe('Failed to fetch models: Internal Server Error')
    })

    it('sets error on network failure', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()

      await expect(fetchModels()).rejects.toThrow('Network error')

      const { error } = useModelStore.getState()
      expect(error).toBe('Network error')
    })

    it('sets generic error message for non-Error exceptions', async () => {
      const mockFetch = vi.fn().mockRejectedValue('string error')
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()

      await expect(fetchModels()).rejects.toBe('string error')

      const { error } = useModelStore.getState()
      expect(error).toBe('Failed to fetch models')
    })

    it('clears previous error on successful fetch', async () => {
      // Set an initial error state
      useModelStore.setState({ error: 'Previous error' })

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ data: mockModels }),
      })
      global.fetch = mockFetch

      const { fetchModels } = useModelStore.getState()
      await fetchModels()

      const { error } = useModelStore.getState()
      expect(error).toBeNull()
    })
  })

  describe('fetchCapabilities', () => {
    it('fetches capabilities successfully', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockCapabilities),
      })
      global.fetch = mockFetch

      const { fetchCapabilities } = useModelStore.getState()
      await fetchCapabilities()

      const { capabilities } = useModelStore.getState()
      expect(capabilities).toEqual(mockCapabilities)
      expect(mockFetch).toHaveBeenCalledWith('/v1/capabilities')
    })

    it('handles HTTP error silently (non-fatal)', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        statusText: 'Not Found',
      })
      global.fetch = mockFetch

      const { fetchCapabilities } = useModelStore.getState()
      await fetchCapabilities() // Should not throw

      const { capabilities } = useModelStore.getState()
      expect(capabilities).toBeNull()
      expect(consoleSpy).toHaveBeenCalled()
    })

    it('handles network failure silently (non-fatal)', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))
      global.fetch = mockFetch

      const { fetchCapabilities } = useModelStore.getState()
      await fetchCapabilities() // Should not throw

      const { capabilities } = useModelStore.getState()
      expect(capabilities).toBeNull()
      expect(consoleSpy).toHaveBeenCalled()
    })

    it('does not modify error state on failure', async () => {
      vi.spyOn(console, 'error').mockImplementation(() => {})
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))
      global.fetch = mockFetch

      const { fetchCapabilities } = useModelStore.getState()
      await fetchCapabilities()

      // Error state should remain null (not set by fetchCapabilities)
      const { error } = useModelStore.getState()
      expect(error).toBeNull()
    })
  })

  describe('setLoadedModel', () => {
    it('sets the loaded model', () => {
      const { setLoadedModel } = useModelStore.getState()

      setLoadedModel(mockLoadedModel)

      const { loadedModel, modelStatus } = useModelStore.getState()
      expect(loadedModel).toEqual(mockLoadedModel)
      expect(modelStatus).toBe('loaded')
    })

    it('clears the loaded model when set to null', () => {
      // First set a model
      useModelStore.setState({
        loadedModel: mockLoadedModel,
        modelStatus: 'loaded',
      })

      const { setLoadedModel } = useModelStore.getState()
      setLoadedModel(null)

      const { loadedModel, modelStatus } = useModelStore.getState()
      expect(loadedModel).toBeNull()
      expect(modelStatus).toBe('unloaded')
    })

    it('updates loaded model when replacing with different model', () => {
      // First set a model
      useModelStore.setState({
        loadedModel: mockLoadedModel,
        modelStatus: 'loaded',
      })

      const newModel: LoadedModel = {
        id: 'test-model-2',
        provider: 'llama_cpp',
        capabilities: {
          chat: true,
          vision: false,
          thinking: true,
          hidden_states: true,
          embeddings: false,
        },
        contextWindow: 8192,
      }

      const { setLoadedModel } = useModelStore.getState()
      setLoadedModel(newModel)

      const { loadedModel, modelStatus } = useModelStore.getState()
      expect(loadedModel).toEqual(newModel)
      expect(modelStatus).toBe('loaded')
    })
  })

  describe('setModelStatus', () => {
    it('sets status to loading', () => {
      const { setModelStatus } = useModelStore.getState()

      setModelStatus('loading')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('loading')
    })

    it('sets status to loaded', () => {
      const { setModelStatus } = useModelStore.getState()

      setModelStatus('loaded')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('loaded')
    })

    it('sets status to unloaded', () => {
      // First set to loaded
      useModelStore.setState({ modelStatus: 'loaded' })

      const { setModelStatus } = useModelStore.getState()
      setModelStatus('unloaded')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('unloaded')
    })

    it('sets status to error', () => {
      const { setModelStatus } = useModelStore.getState()

      setModelStatus('error')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('error')
    })

    it('transitions from loading to loaded', () => {
      useModelStore.setState({ modelStatus: 'loading' })

      const { setModelStatus } = useModelStore.getState()
      setModelStatus('loaded')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('loaded')
    })

    it('transitions from loading to error', () => {
      useModelStore.setState({ modelStatus: 'loading' })

      const { setModelStatus } = useModelStore.getState()
      setModelStatus('error')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('error')
    })

    it('transitions from error to loading', () => {
      useModelStore.setState({ modelStatus: 'error' })

      const { setModelStatus } = useModelStore.getState()
      setModelStatus('loading')

      const { modelStatus } = useModelStore.getState()
      expect(modelStatus).toBe('loading')
    })
  })

  describe('setError', () => {
    it('sets an error message', () => {
      const { setError } = useModelStore.getState()

      setError('Something went wrong')

      const { error } = useModelStore.getState()
      expect(error).toBe('Something went wrong')
    })

    it('clears error when set to null', () => {
      // First set an error
      useModelStore.setState({ error: 'Previous error' })

      const { setError } = useModelStore.getState()
      setError(null)

      const { error } = useModelStore.getState()
      expect(error).toBeNull()
    })

    it('replaces existing error message', () => {
      useModelStore.setState({ error: 'Old error' })

      const { setError } = useModelStore.getState()
      setError('New error')

      const { error } = useModelStore.getState()
      expect(error).toBe('New error')
    })

    it('accepts empty string as error', () => {
      const { setError } = useModelStore.getState()

      setError('')

      const { error } = useModelStore.getState()
      expect(error).toBe('')
    })
  })

  describe('getModelCapabilities helper', () => {
    it('parses model with all capabilities', () => {
      const model: Model = {
        id: 'full-model',
        object: 'model',
        owned_by: 'test',
        capabilities: ['chat', 'vision', 'thinking', 'hidden_states', 'embeddings'],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: true,
        vision: true,
        thinking: true,
        hidden_states: true,
        embeddings: true,
      })
    })

    it('parses model with chat and vision only', () => {
      const model: Model = {
        id: 'vision-model',
        object: 'model',
        owned_by: 'test',
        capabilities: ['chat', 'vision'],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: true,
        vision: true,
        thinking: false,
        hidden_states: false,
        embeddings: false,
      })
    })

    it('parses model with thinking capability', () => {
      const model: Model = {
        id: 'thinking-model',
        object: 'model',
        owned_by: 'test',
        capabilities: ['chat', 'thinking'],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: true,
        vision: false,
        thinking: true,
        hidden_states: false,
        embeddings: false,
      })
    })

    it('parses model with embeddings only', () => {
      const model: Model = {
        id: 'embedding-model',
        object: 'model',
        owned_by: 'test',
        capabilities: ['embeddings'],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: false,
        vision: false,
        thinking: false,
        hidden_states: false,
        embeddings: true,
      })
    })

    it('parses model with no capabilities', () => {
      const model: Model = {
        id: 'basic-model',
        object: 'model',
        owned_by: 'test',
        capabilities: [],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: false,
        vision: false,
        thinking: false,
        hidden_states: false,
        embeddings: false,
      })
    })

    it('parses model with undefined capabilities', () => {
      const model: Model = {
        id: 'basic-model',
        object: 'model',
        owned_by: 'test',
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: false,
        vision: false,
        thinking: false,
        hidden_states: false,
        embeddings: false,
      })
    })

    it('ignores unknown capabilities', () => {
      const model: Model = {
        id: 'extended-model',
        object: 'model',
        owned_by: 'test',
        capabilities: ['chat', 'unknown_capability', 'another_unknown'],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: true,
        vision: false,
        thinking: false,
        hidden_states: false,
        embeddings: false,
      })
    })

    it('handles duplicate capabilities gracefully', () => {
      const model: Model = {
        id: 'dupe-model',
        object: 'model',
        owned_by: 'test',
        capabilities: ['chat', 'chat', 'vision', 'vision'],
      }

      const caps = getModelCapabilities(model)

      expect(caps).toEqual({
        chat: true,
        vision: true,
        thinking: false,
        hidden_states: false,
        embeddings: false,
      })
    })
  })
})
