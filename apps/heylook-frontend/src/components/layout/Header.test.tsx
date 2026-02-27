import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Header } from './Header'
import { ThemeProvider } from '../../contexts/ThemeContext'

// Wrapper component with ThemeProvider and Router
function renderHeader(initialRoute = '/chat') {
  return render(
    <MemoryRouter initialEntries={[initialRoute]}>
      <ThemeProvider>
        <Header />
      </ThemeProvider>
    </MemoryRouter>
  )
}

// Mock the stores
const mockToggleSidebar = vi.fn()
const mockTogglePanel = vi.fn()

const defaultModelState = {
  models: [
    { id: 'model-1', capabilities: ['chat'] },
    { id: 'model-2', capabilities: ['chat', 'vision'] },
  ],
  loadedModel: null,
  modelStatus: 'unloaded' as const,
}

const defaultUIState = {
  toggleSidebar: mockToggleSidebar,
  togglePanel: mockTogglePanel,
  activePanel: null as string | null,
}

vi.mock('../../stores/modelStore', () => ({
  useModelStore: vi.fn((sel?: any) => typeof sel === 'function' ? sel(defaultModelState) : defaultModelState),
}))

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => defaultUIState),
}))

// Import mocks after defining them
import { useModelStore } from '../../stores/modelStore'
import { useUIStore } from '../../stores/uiStore'

const setModelMock = (state: any) =>
  vi.mocked(useModelStore).mockImplementation((sel?: any) => typeof sel === 'function' ? sel(state) : state)

describe('Header', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset mocks to default state
    setModelMock(defaultModelState)
    vi.mocked(useUIStore).mockReturnValue(defaultUIState)
  })

  describe('rendering', () => {
    it('renders the header element', () => {
      renderHeader()

      const header = document.querySelector('header')
      expect(header).toBeInTheDocument()
    })

    it('renders the sidebar toggle button on chat route', () => {
      renderHeader('/chat')

      const sidebarButton = screen.getByLabelText('Toggle sidebar')
      expect(sidebarButton).toBeInTheDocument()
    })

    it('renders the model selector button', () => {
      renderHeader()

      // When no model is loaded, shows "Select Model"
      expect(screen.getByText('Select Model')).toBeInTheDocument()
    })

    it('renders advanced settings button', () => {
      renderHeader()

      const advancedButton = screen.getByLabelText('Advanced settings')
      expect(advancedButton).toBeInTheDocument()
    })

    it('renders sampler settings button', () => {
      renderHeader()

      const samplerButton = screen.getByLabelText('Sampler settings')
      expect(samplerButton).toBeInTheDocument()
    })
  })

  describe('route-aware hamburger', () => {
    it('shows sidebar toggle on /chat route', () => {
      renderHeader('/chat')

      expect(screen.getByLabelText('Toggle sidebar')).toBeInTheDocument()
    })

    it('hides sidebar toggle on non-chat routes', () => {
      renderHeader('/batch')

      expect(screen.queryByLabelText('Toggle sidebar')).not.toBeInTheDocument()
    })

    it('shows spacer instead of hamburger on non-chat routes', () => {
      renderHeader('/explore')

      // Should have a spacer div instead
      expect(screen.queryByLabelText('Toggle sidebar')).not.toBeInTheDocument()
      const spacer = document.querySelector('.min-w-\\[44px\\]')
      expect(spacer).toBeInTheDocument()
    })
  })

  describe('model display', () => {
    it('shows "Select Model" when no model is loaded', () => {
      renderHeader()

      expect(screen.getByText('Select Model')).toBeInTheDocument()
    })

    it('shows loaded model name when a model is loaded', () => {
      setModelMock({
        models: [{ id: 'my-model', capabilities: ['chat'] }],
        loadedModel: {
          id: 'my-model',
          capabilities: { chat: true, vision: false, thinking: false, hidden_states: false, embeddings: false },
          contextWindow: 4096,
        },
        modelStatus: 'loaded' as const,
      })

      renderHeader()

      expect(screen.getByText('my-model')).toBeInTheDocument()
    })

    it('shows dropdown indicator on model selector', () => {
      renderHeader()

      // The chevron down icon is rendered as an SVG inside the button
      const modelButton = screen.getByText('Select Model').closest('button')
      expect(modelButton).toBeInTheDocument()
      const svg = modelButton?.querySelector('svg')
      expect(svg).toBeInTheDocument()
    })
  })

  describe('status indicator', () => {
    it('shows gray indicator when model is unloaded', () => {
      setModelMock({
        models: [],
        loadedModel: null,
        modelStatus: 'unloaded' as const,
      })

      renderHeader()

      const indicator = document.querySelector('.bg-gray-500')
      expect(indicator).toBeInTheDocument()
    })

    it('shows green indicator when model is loaded', () => {
      setModelMock({
        models: [{ id: 'my-model', capabilities: ['chat'] }],
        loadedModel: {
          id: 'my-model',
          capabilities: { chat: true, vision: false, thinking: false, hidden_states: false, embeddings: false },
          contextWindow: 4096,
        },
        modelStatus: 'loaded' as const,
      })

      renderHeader()

      const indicator = document.querySelector('.bg-accent-green')
      expect(indicator).toBeInTheDocument()
    })

    it('shows pulsing amber indicator when model is loading', () => {
      setModelMock({
        models: [],
        loadedModel: null,
        modelStatus: 'loading' as const,
      })

      renderHeader()

      const indicator = document.querySelector('.bg-amber-500.animate-pulse')
      expect(indicator).toBeInTheDocument()
    })
  })

  describe('sidebar toggle', () => {
    it('calls toggleSidebar when sidebar button is clicked', async () => {
      const user = userEvent.setup()
      renderHeader('/chat')

      const sidebarButton = screen.getByLabelText('Toggle sidebar')
      await user.click(sidebarButton)

      expect(mockToggleSidebar).toHaveBeenCalledTimes(1)
    })
  })

  describe('panel toggles', () => {
    it('calls togglePanel with "models" when model selector is clicked', async () => {
      const user = userEvent.setup()
      renderHeader()

      const modelButton = screen.getByText('Select Model').closest('button')
      await user.click(modelButton!)

      expect(mockTogglePanel).toHaveBeenCalledWith('models')
    })

    it('calls togglePanel with "advanced" when advanced settings is clicked', async () => {
      const user = userEvent.setup()
      renderHeader()

      const advancedButton = screen.getByLabelText('Advanced settings')
      await user.click(advancedButton)

      expect(mockTogglePanel).toHaveBeenCalledWith('advanced')
    })

    it('calls togglePanel with "settings" when sampler settings is clicked', async () => {
      const user = userEvent.setup()
      renderHeader()

      const samplerButton = screen.getByLabelText('Sampler settings')
      await user.click(samplerButton)

      expect(mockTogglePanel).toHaveBeenCalledWith('settings')
    })
  })

  describe('active panel styling', () => {
    it('highlights advanced settings button when advanced panel is active', () => {
      vi.mocked(useUIStore).mockReturnValue({
        toggleSidebar: mockToggleSidebar,
        togglePanel: mockTogglePanel,
        activePanel: 'advanced',
      })

      renderHeader()

      const advancedButton = screen.getByLabelText('Advanced settings')
      expect(advancedButton).toHaveClass('bg-primary/20')
      expect(advancedButton).toHaveClass('text-primary')
    })

    it('highlights sampler settings button when settings panel is active', () => {
      vi.mocked(useUIStore).mockReturnValue({
        toggleSidebar: mockToggleSidebar,
        togglePanel: mockTogglePanel,
        activePanel: 'settings',
      })

      renderHeader()

      const samplerButton = screen.getByLabelText('Sampler settings')
      expect(samplerButton).toHaveClass('bg-primary/20')
      expect(samplerButton).toHaveClass('text-primary')
    })

    it('does not highlight buttons when no panel is active', () => {
      renderHeader()

      const advancedButton = screen.getByLabelText('Advanced settings')
      const samplerButton = screen.getByLabelText('Sampler settings')

      expect(advancedButton).not.toHaveClass('bg-primary/20')
      expect(samplerButton).not.toHaveClass('bg-primary/20')
    })
  })

  describe('button tooltips', () => {
    it('has correct title on advanced settings button', () => {
      renderHeader()

      const advancedButton = screen.getByLabelText('Advanced settings')
      expect(advancedButton).toHaveAttribute('title', 'System Prompt & Templates')
    })

    it('has correct title on sampler settings button', () => {
      renderHeader()

      const samplerButton = screen.getByLabelText('Sampler settings')
      expect(samplerButton).toHaveAttribute('title', 'Generation Parameters')
    })
  })

  describe('theme toggle', () => {
    it('renders theme toggle button with current theme in label', () => {
      renderHeader()

      const themeButton = screen.getByLabelText('Theme: Dark mode')
      expect(themeButton).toBeInTheDocument()
    })

    it('shows correct tooltip for dark mode', () => {
      renderHeader()

      const themeButton = screen.getByLabelText('Theme: Dark mode')
      expect(themeButton).toHaveAttribute('title', 'Dark mode')
    })

    it('cycles theme on click', async () => {
      const user = userEvent.setup()
      renderHeader()

      const themeButton = screen.getByLabelText('Theme: Dark mode')

      // Initial state is dark mode
      expect(themeButton).toHaveAttribute('title', 'Dark mode')

      // Click to change to light mode
      await user.click(themeButton)
      expect(themeButton).toHaveAttribute('title', 'Light mode')
      expect(themeButton).toHaveAttribute('aria-label', 'Theme: Light mode')

      // Click to change to auto mode
      await user.click(themeButton)
      expect(themeButton).toHaveAttribute('title', 'System theme')
      expect(themeButton).toHaveAttribute('aria-label', 'Theme: System theme')

      // Click to cycle back to dark mode
      await user.click(themeButton)
      expect(themeButton).toHaveAttribute('title', 'Dark mode')
      expect(themeButton).toHaveAttribute('aria-label', 'Theme: Dark mode')
    })
  })
})
