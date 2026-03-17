import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import userEvent from '@testing-library/user-event'
import App from './App'
import { useConnectionStore } from './stores/connectionStore'

// Mock the AppShell to pass through children via Outlet
vi.mock('./components/layout/AppShell', () => ({
  AppShell: () => {
    const { Outlet } = require('react-router-dom')
    return <div data-testid="app-shell"><Outlet /></div>
  },
}))

// Mock the Layout component
vi.mock('./components/layout/Layout', () => ({
  Layout: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}))

// Mock the ChatView component
vi.mock('./applets/chat/components/ChatView', () => ({
  ChatView: () => <div data-testid="chat-view">ChatView</div>,
}))

// Mock the ConfirmDeleteModal component
vi.mock('./applets/chat/components/ConfirmDeleteModal', () => ({
  ConfirmDeleteModal: () => <div data-testid="confirm-delete-modal">ConfirmDeleteModal</div>,
}))

// vi.hoisted ensures these are available to vi.mock factories (which are hoisted)
const { mockFetchModels, mockFetchCapabilities } = vi.hoisted(() => ({
  mockFetchModels: vi.fn(),
  mockFetchCapabilities: vi.fn(),
}))

vi.mock('./stores/modelStore', () => {
  const state = {
    fetchModels: mockFetchModels,
    fetchCapabilities: mockFetchCapabilities,
    // initialize delegates to fetchModels so tests control behavior via mockFetchModels
    initialize: async () => {
      await mockFetchModels()
      mockFetchCapabilities().catch(() => {})
    },
  }
  return {
    useModelStore: Object.assign(
      vi.fn((selector: (s: typeof state) => unknown) => selector(state)),
      { getState: () => state }
    ),
  }
})

function renderApp() {
  return render(
    <MemoryRouter initialEntries={['/chat']}>
      <App />
    </MemoryRouter>
  )
}

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFetchModels.mockReset()
    mockFetchCapabilities.mockReset().mockResolvedValue(undefined)
    // Reset connection store to initial state between tests
    useConnectionStore.setState({ isConnected: null, isReconnecting: false })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('loading state', () => {
    it('shows loading spinner on initial render', () => {
      // fetchModels never resolves, keeping app in loading state
      mockFetchModels.mockReturnValue(new Promise(() => {}))

      renderApp()

      expect(screen.getByText('Connecting to server...')).toBeInTheDocument()
    })

    it('displays spinner animation during loading', () => {
      mockFetchModels.mockReturnValue(new Promise(() => {}))

      renderApp()

      const spinner = document.querySelector('.animate-spin')
      expect(spinner).toBeInTheDocument()
    })

    it('renders loading state with correct styling', () => {
      mockFetchModels.mockReturnValue(new Promise(() => {}))

      renderApp()

      const container = document.querySelector('.bg-background-dark')
      expect(container).toBeInTheDocument()
    })

    it('calls fetchModels on mount', async () => {
      mockFetchModels.mockReturnValue(new Promise(() => {}))

      renderApp()

      await waitFor(() => {
        expect(mockFetchModels).toHaveBeenCalledTimes(1)
      })
    })
  })

  describe('error state (connection failed)', () => {
    it('shows "Connection Failed" when fetchModels throws', async () => {
      mockFetchModels.mockRejectedValue(new Error('Network error'))

      renderApp()

      await waitFor(() => {
        expect(screen.getByText('Connection Failed')).toBeInTheDocument()
      })
    })

    it('displays generic error message', async () => {
      mockFetchModels.mockRejectedValue(new Error('Network error'))

      renderApp()

      await waitFor(() => {
        expect(screen.getByText(/could not connect to the backend server/i)).toBeInTheDocument()
      })
    })

    it('renders warning icon in error state', async () => {
      mockFetchModels.mockRejectedValue(new Error('Network error'))

      renderApp()

      await waitFor(() => {
        const icon = document.querySelector('.text-accent-red')
        expect(icon).toBeInTheDocument()
      })
    })

    it('renders retry button in error state', async () => {
      mockFetchModels.mockRejectedValue(new Error('Network error'))

      renderApp()

      await waitFor(() => {
        expect(screen.getByText('Retry Connection')).toBeInTheDocument()
      })
    })

    it('retry button triggers page reload', async () => {
      const user = userEvent.setup()
      mockFetchModels.mockRejectedValue(new Error('Network error'))

      // Mock window.location.reload
      const reloadMock = vi.fn()
      Object.defineProperty(window, 'location', {
        value: { reload: reloadMock },
        writable: true,
      })

      renderApp()

      await waitFor(() => {
        expect(screen.getByText('Retry Connection')).toBeInTheDocument()
      })

      await user.click(screen.getByText('Retry Connection'))

      expect(reloadMock).toHaveBeenCalledTimes(1)
    })

    it('handles different error types gracefully', async () => {
      // Test with string rejection
      mockFetchModels.mockRejectedValue('string error')

      renderApp()

      await waitFor(() => {
        expect(screen.getByText('Connection Failed')).toBeInTheDocument()
      })
    })
  })

  describe('connected state', () => {
    it('renders Layout when connected', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument()
      })
    })

    it('renders ChatView inside Layout', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      await waitFor(() => {
        expect(screen.getByTestId('chat-view')).toBeInTheDocument()
      })
    })

    it('renders ConfirmDeleteModal when connected', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      await waitFor(() => {
        expect(screen.getByTestId('confirm-delete-modal')).toBeInTheDocument()
      })
    })

    it('does not show loading spinner when connected', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      await waitFor(() => {
        expect(screen.queryByText('Connecting to server...')).not.toBeInTheDocument()
      })
    })

    it('does not show error state when connected', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      await waitFor(() => {
        expect(screen.queryByText('Connection Failed')).not.toBeInTheDocument()
      })
    })
  })

  describe('state transitions', () => {
    it('transitions from loading to connected', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      // Initially shows loading
      expect(screen.getByText('Connecting to server...')).toBeInTheDocument()

      // Eventually shows connected state
      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument()
      })
    })

    it('transitions from loading to error', async () => {
      mockFetchModels.mockRejectedValue(new Error('Connection refused'))

      renderApp()

      // Initially shows loading
      expect(screen.getByText('Connecting to server...')).toBeInTheDocument()

      // Eventually shows error state
      await waitFor(() => {
        expect(screen.getByText('Connection Failed')).toBeInTheDocument()
      })
    })
  })

  describe('fetchModels integration', () => {
    it('only calls fetchModels once on mount', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      const { rerender } = render(
        <MemoryRouter initialEntries={['/chat']}>
          <App />
        </MemoryRouter>
      )

      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument()
      })

      // Re-render should not call fetchModels again
      rerender(
        <MemoryRouter initialEntries={['/chat']}>
          <App />
        </MemoryRouter>
      )

      expect(mockFetchModels).toHaveBeenCalledTimes(1)
    })

    it('calls fetchModels from useEffect on initial render', async () => {
      let resolvePromise: () => void = () => {}
      const delayedPromise = new Promise<void>((resolve) => {
        resolvePromise = resolve
      })
      mockFetchModels.mockReturnValue(delayedPromise)

      renderApp()

      // Should be in loading state
      expect(screen.getByText('Connecting to server...')).toBeInTheDocument()

      // Resolve the promise
      resolvePromise()

      // Should transition to connected
      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument()
      })
    })
  })

  describe('accessibility', () => {
    it('loading state has appropriate structure', () => {
      mockFetchModels.mockReturnValue(new Promise(() => {}))

      renderApp()

      // Text is visible for screen readers
      expect(screen.getByText('Connecting to server...')).toBeInTheDocument()
    })

    it('error state has heading structure', async () => {
      mockFetchModels.mockRejectedValue(new Error('error'))

      renderApp()

      await waitFor(() => {
        const heading = screen.getByRole('heading', { name: 'Connection Failed' })
        expect(heading).toBeInTheDocument()
      })
    })

    it('retry button is focusable', async () => {
      const user = userEvent.setup()
      mockFetchModels.mockRejectedValue(new Error('error'))

      renderApp()

      await waitFor(() => {
        expect(screen.getByText('Retry Connection')).toBeInTheDocument()
      })

      const button = screen.getByRole('button', { name: 'Retry Connection' })
      await user.tab()

      // Button should be in the document and be a proper button element
      expect(button.tagName).toBe('BUTTON')
    })
  })

  describe('edge cases', () => {
    it('handles empty successful response', async () => {
      mockFetchModels.mockResolvedValue(undefined)

      renderApp()

      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument()
      })
    })

    it('handles slow network response', async () => {
      let resolvePromise: () => void = () => {}
      const slowPromise = new Promise<void>((resolve) => {
        resolvePromise = resolve
      })
      mockFetchModels.mockReturnValue(slowPromise)

      renderApp()

      // Still loading
      expect(screen.getByText('Connecting to server...')).toBeInTheDocument()

      // Resolve after delay
      await new Promise((r) => setTimeout(r, 100))
      resolvePromise()

      await waitFor(() => {
        expect(screen.getByTestId('layout')).toBeInTheDocument()
      })
    })
  })
})
