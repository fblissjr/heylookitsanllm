import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Layout } from './Layout'

// Mock child components
vi.mock('../../applets/chat/components/Sidebar', () => ({
  Sidebar: () => <aside data-testid="mock-sidebar">Sidebar</aside>,
}))

// Mock store state
const mockToggleSidebar = vi.fn()

const defaultUIState = {
  isSidebarOpen: true,
  isMobile: false,
}

vi.mock('../../stores/uiStore', () => ({
  useUIStore: Object.assign(
    vi.fn((selector: (s: Record<string, unknown>) => unknown) => selector(defaultUIState)),
    { getState: () => ({ toggleSidebar: mockToggleSidebar }) }
  ),
}))

import { useUIStore } from '../../stores/uiStore'

describe('Layout', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useUIStore).mockImplementation(
      ((selector: (s: typeof defaultUIState) => unknown) => selector(defaultUIState)) as typeof useUIStore
    )
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering', () => {
    it('renders the layout container', () => {
      const { container } = render(<Layout><div>Content</div></Layout>)

      expect(container.firstElementChild).toBeInTheDocument()
    })

    it('renders children in main content area', () => {
      render(<Layout><div data-testid="test-content">Test Content</div></Layout>)

      expect(screen.getByTestId('test-content')).toBeInTheDocument()
      expect(screen.getByText('Test Content')).toBeInTheDocument()
    })
  })

  describe('desktop sidebar behavior', () => {
    it('shows sidebar when isSidebarOpen is true and not mobile', () => {
      vi.mocked(useUIStore).mockImplementation(
        ((selector: (s: typeof defaultUIState) => unknown) => selector({
          ...defaultUIState,
          isSidebarOpen: true,
          isMobile: false,
        })) as typeof useUIStore
      )

      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('mock-sidebar')).toBeInTheDocument()
    })

    it('hides sidebar when isSidebarOpen is false on desktop', () => {
      vi.mocked(useUIStore).mockImplementation(
        ((selector: (s: typeof defaultUIState) => unknown) => selector({
          ...defaultUIState,
          isSidebarOpen: false,
          isMobile: false,
        })) as typeof useUIStore
      )

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('mock-sidebar')).not.toBeInTheDocument()
    })
  })

  describe('mobile sidebar behavior', () => {
    it('shows mobile overlay with backdrop when sidebar open on mobile', () => {
      vi.mocked(useUIStore).mockImplementation(
        ((selector: (s: typeof defaultUIState) => unknown) => selector({
          ...defaultUIState,
          isSidebarOpen: true,
          isMobile: true,
        })) as typeof useUIStore
      )

      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('sidebar-backdrop')).toBeInTheDocument()
      expect(screen.getByTestId('mock-sidebar')).toBeInTheDocument()
    })

    it('does not show mobile overlay when sidebar is closed on mobile', () => {
      vi.mocked(useUIStore).mockImplementation(
        ((selector: (s: typeof defaultUIState) => unknown) => selector({
          ...defaultUIState,
          isSidebarOpen: false,
          isMobile: true,
        })) as typeof useUIStore
      )

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('sidebar-backdrop')).not.toBeInTheDocument()
    })

    it('calls toggleSidebar when clicking mobile backdrop', () => {
      vi.mocked(useUIStore).mockImplementation(
        ((selector: (s: typeof defaultUIState) => unknown) => selector({
          ...defaultUIState,
          isSidebarOpen: true,
          isMobile: true,
        })) as typeof useUIStore
      )

      render(<Layout><div>Content</div></Layout>)

      fireEvent.click(screen.getByTestId('sidebar-backdrop'))

      expect(mockToggleSidebar).toHaveBeenCalledTimes(1)
    })
  })

  describe('layout structure', () => {
    it('has a container element wrapping children', () => {
      const { container } = render(<Layout><div data-testid="child">Content</div></Layout>)

      // Layout wraps children in a flex container
      const root = container.firstElementChild
      expect(root).toBeInTheDocument()
      expect(screen.getByTestId('child')).toBeInTheDocument()
    })
  })
})
