import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ModelSelector } from './ModelSelector'
import type { Model } from '../../types/api'
import type { LoadedModel, ModelCapabilities } from '../../types/models'

// Mock the stores
const mockSetLoadedModel = vi.fn()
const mockSetModelStatus = vi.fn()
const mockSetError = vi.fn()
const mockSetActivePanel = vi.fn()

const defaultModelCapabilities: ModelCapabilities = {
  chat: true,
  vision: false,
  thinking: false,
  hidden_states: false,
  embeddings: false,
}

const mockModels: Model[] = [
  {
    id: 'llama-3.2-1b',
    object: 'model' as const,
    owned_by: 'meta',
    capabilities: ['chat'],
    context_window: 8192,
  },
  {
    id: 'qwen-vl-4b',
    object: 'model' as const,
    owned_by: 'alibaba',
    capabilities: ['chat', 'vision'],
    context_window: 32768,
  },
  {
    id: 'qwen3-8b-thinking',
    object: 'model' as const,
    owned_by: 'alibaba',
    capabilities: ['chat', 'thinking'],
    context_window: 16384,
  },
]

const defaultModelStoreState = {
  models: mockModels,
  loadedModel: null as LoadedModel | null,
  setLoadedModel: mockSetLoadedModel,
  setModelStatus: mockSetModelStatus,
  setError: mockSetError,
}

vi.mock('../../stores/modelStore', () => ({
  useModelStore: vi.fn((sel?: any) => typeof sel === 'function' ? sel(defaultModelStoreState) : defaultModelStoreState),
  getModelCapabilities: vi.fn((model: Model) => {
    const caps = model.capabilities || []
    return {
      chat: caps.includes('chat'),
      vision: caps.includes('vision'),
      thinking: caps.includes('thinking'),
      hidden_states: caps.includes('hidden_states'),
      embeddings: caps.includes('embeddings'),
    }
  }),
}))

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    setActivePanel: mockSetActivePanel,
  })),
}))

// Import mocks after defining
import { useModelStore } from '../../stores/modelStore'

const setModelMock = (state: any) =>
  vi.mocked(useModelStore).mockImplementation((sel?: any) => typeof sel === 'function' ? sel(state) : state)

describe('ModelSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setModelMock(defaultModelStoreState)
  })

  describe('rendering', () => {
    it('renders the header with "Models" title', () => {
      render(<ModelSelector />)

      expect(screen.getByRole('heading', { name: 'Models' })).toBeInTheDocument()
    })

    it('renders close button in header', () => {
      render(<ModelSelector />)

      const closeButton = screen.getByRole('button', { name: '' })
      expect(closeButton).toBeInTheDocument()
    })

    it('renders list of available models', () => {
      render(<ModelSelector />)

      expect(screen.getByText('llama-3.2-1b')).toBeInTheDocument()
      expect(screen.getByText('qwen-vl-4b')).toBeInTheDocument()
      expect(screen.getByText('qwen3-8b-thinking')).toBeInTheDocument()
    })

    it('renders model owner information', () => {
      render(<ModelSelector />)

      expect(screen.getByText('meta')).toBeInTheDocument()
      expect(screen.getAllByText('alibaba')).toHaveLength(2)
    })
  })

  describe('empty state', () => {
    it('shows empty state when no models available', () => {
      setModelMock({
        ...defaultModelStoreState,
        models: [],
      })

      render(<ModelSelector />)

      expect(screen.getByText('No models available')).toBeInTheDocument()
      expect(screen.getByText('Make sure models.toml is configured')).toBeInTheDocument()
    })
  })

  describe('model selection', () => {
    it('selects a model when clicked', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      expect(screen.getByText('Load Configuration')).toBeInTheDocument()
    })

    it('shows context window slider when model is selected', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      expect(screen.getByText('Context Window')).toBeInTheDocument()
      expect(screen.getByRole('slider')).toBeInTheDocument()
    })

    it('sets context window to model default when selected', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      const valueDisplay = document.querySelector('.font-mono.text-primary')
      expect(valueDisplay).toBeInTheDocument()
      expect(valueDisplay).toHaveTextContent('8,192')
    })

    it('shows Load Model button when model is selected', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      expect(screen.getByRole('button', { name: /Load Model/i })).toBeInTheDocument()
    })
  })

  describe('context window slider', () => {
    it('allows changing context window value', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      const slider = screen.getByRole('slider')

      expect(slider).toHaveAttribute('min', '512')
      expect(slider).toHaveAttribute('max', '8192')
    })

    it('shows slider range labels', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('qwen-vl-4b').closest('button')
      await user.click(modelCard!)

      const rangeLabels = document.querySelector('.flex.justify-between.text-xs')
      expect(rangeLabels).toBeInTheDocument()
      expect(within(rangeLabels as HTMLElement).getByText('512')).toBeInTheDocument()
      expect(within(rangeLabels as HTMLElement).getByText('32,768')).toBeInTheDocument()
    })
  })

  describe('load model flow', () => {
    it('calls setModelStatus with loading when Load Model clicked', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      const loadButton = screen.getByRole('button', { name: /Load Model/i })
      await user.click(loadButton)

      expect(mockSetModelStatus).toHaveBeenCalledWith('loading')
    })

    it('calls setLoadedModel with correct data when loading', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      const loadButton = screen.getByRole('button', { name: /Load Model/i })
      await user.click(loadButton)

      expect(mockSetLoadedModel).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 'llama-3.2-1b',
          contextWindow: 8192,
          capabilities: expect.objectContaining({
            chat: true,
          }),
        })
      )
    })

    it('closes panel after loading model', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      const loadButton = screen.getByRole('button', { name: /Load Model/i })
      await user.click(loadButton)

      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })

    it('clears error before loading', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      const loadButton = screen.getByRole('button', { name: /Load Model/i })
      await user.click(loadButton)

      expect(mockSetError).toHaveBeenCalledWith(null)
    })
  })

  describe('unload model', () => {
    it('shows Unload Model button when model is loaded', () => {
      const loadedModel: LoadedModel = {
        id: 'llama-3.2-1b',
        contextWindow: 8192,
        capabilities: defaultModelCapabilities,
      }

      setModelMock({
        ...defaultModelStoreState,
        loadedModel,
      })

      render(<ModelSelector />)

      expect(screen.getByRole('button', { name: 'Unload Model' })).toBeInTheDocument()
    })

    it('shows current model indicator when loaded', () => {
      const loadedModel: LoadedModel = {
        id: 'llama-3.2-1b',
        contextWindow: 8192,
        capabilities: defaultModelCapabilities,
      }

      setModelMock({
        ...defaultModelStoreState,
        loadedModel,
      })

      render(<ModelSelector />)

      expect(screen.getByText('Current:')).toBeInTheDocument()
      const modelIdElements = screen.getAllByText('llama-3.2-1b')
      expect(modelIdElements.length).toBeGreaterThanOrEqual(1)
    })

    it('calls setLoadedModel with null when unload clicked', async () => {
      const user = userEvent.setup()
      const loadedModel: LoadedModel = {
        id: 'llama-3.2-1b',
        contextWindow: 8192,
        capabilities: defaultModelCapabilities,
      }

      setModelMock({
        ...defaultModelStoreState,
        loadedModel,
      })

      render(<ModelSelector />)

      const unloadButton = screen.getByRole('button', { name: 'Unload Model' })
      await user.click(unloadButton)

      expect(mockSetLoadedModel).toHaveBeenCalledWith(null)
    })

    it('shows "Loaded" badge on loaded model card', () => {
      const loadedModel: LoadedModel = {
        id: 'llama-3.2-1b',
        contextWindow: 8192,
        capabilities: defaultModelCapabilities,
      }

      setModelMock({
        ...defaultModelStoreState,
        loadedModel,
      })

      render(<ModelSelector />)

      expect(screen.getByText('Loaded')).toBeInTheDocument()
    })
  })

  describe('capability badges', () => {
    it('shows Chat badge for chat-capable models', () => {
      render(<ModelSelector />)

      const chatBadges = screen.getAllByText('Chat')
      expect(chatBadges.length).toBeGreaterThan(0)
    })

    it('shows Vision badge for vision-capable models', () => {
      render(<ModelSelector />)

      expect(screen.getByText('Vision')).toBeInTheDocument()
    })

    it('shows Thinking badge for thinking-capable models', () => {
      render(<ModelSelector />)

      expect(screen.getByText('Thinking')).toBeInTheDocument()
    })

    it('renders capability badge with correct icon abbreviation', () => {
      render(<ModelSelector />)

      const visionBadge = screen.getByText('Vision').closest('span')
      expect(visionBadge).toBeInTheDocument()
      expect(within(visionBadge!).getByText('V')).toBeInTheDocument()
    })

    it('renders multiple capabilities for multi-capability models', () => {
      render(<ModelSelector />)

      const qwenVLCard = screen.getByText('qwen-vl-4b').closest('button')
      expect(qwenVLCard).toBeInTheDocument()

      const chatBadge = within(qwenVLCard!).getByText('Chat')
      const visionBadge = within(qwenVLCard!).getByText('Vision')
      expect(chatBadge).toBeInTheDocument()
      expect(visionBadge).toBeInTheDocument()
    })
  })

  describe('close button', () => {
    it('calls setActivePanel with null when close button clicked', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const header = screen.getByRole('heading', { name: 'Models' }).closest('div')
      const closeButton = within(header!).getByRole('button')
      await user.click(closeButton)

      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })
  })

  describe('model card styling', () => {
    it('applies selected styling when model is selected', async () => {
      const user = userEvent.setup()
      render(<ModelSelector />)

      const modelCard = screen.getByText('llama-3.2-1b').closest('button')
      await user.click(modelCard!)

      expect(modelCard).toHaveClass('border-primary')
    })

    it('applies loaded styling when model is loaded', () => {
      const loadedModel: LoadedModel = {
        id: 'llama-3.2-1b',
        contextWindow: 8192,
        capabilities: defaultModelCapabilities,
      }

      setModelMock({
        ...defaultModelStoreState,
        loadedModel,
      })

      render(<ModelSelector />)

      const modelCards = screen.getAllByRole('button').filter(
        btn => btn.textContent?.includes('llama-3.2-1b') && btn.textContent?.includes('meta')
      )
      expect(modelCards.length).toBe(1)
      expect(modelCards[0]).toHaveClass('border-accent-green')
    })
  })

  describe('load configuration visibility', () => {
    it('hides Load Configuration when no model selected', () => {
      render(<ModelSelector />)

      expect(screen.queryByText('Load Configuration')).not.toBeInTheDocument()
    })

    it('hides Load Configuration when model is already loaded', () => {
      const loadedModel: LoadedModel = {
        id: 'llama-3.2-1b',
        contextWindow: 8192,
        capabilities: defaultModelCapabilities,
      }

      setModelMock({
        ...defaultModelStoreState,
        loadedModel,
      })

      render(<ModelSelector />)

      expect(screen.queryByText('Load Configuration')).not.toBeInTheDocument()
    })
  })
})
