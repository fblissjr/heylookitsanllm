import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Layout } from './Layout'

// Mock child components
vi.mock('./Header', () => ({
  Header: () => <header data-testid="mock-header">Header</header>,
}))

vi.mock('./Sidebar', () => ({
  Sidebar: () => <aside data-testid="mock-sidebar">Sidebar</aside>,
}))

vi.mock('../../features/models/components/ModelSelector', () => ({
  ModelSelector: () => <div data-testid="mock-model-selector">ModelSelector</div>,
}))

vi.mock('../panels/AdvancedPanel', () => ({
  AdvancedPanel: () => <div data-testid="mock-advanced-panel">AdvancedPanel</div>,
}))

vi.mock('../panels/SettingsPanel', () => ({
  SettingsPanel: () => <div data-testid="mock-settings-panel">SettingsPanel</div>,
}))

// Mock store state
const mockToggleSidebar = vi.fn()
const mockSetIsMobile = vi.fn()
const mockSetActivePanel = vi.fn()

const defaultUIState = {
  isSidebarOpen: true,
  isMobile: false,
  setIsMobile: mockSetIsMobile,
  activePanel: null as string | null,
  setActivePanel: mockSetActivePanel,
  toggleSidebar: mockToggleSidebar,
}

vi.mock('../../stores/uiStore', () => ({
  useUIStore: Object.assign(
    vi.fn(() => defaultUIState),
    { getState: () => ({ toggleSidebar: mockToggleSidebar, setIsMobile: mockSetIsMobile }) }
  ),
}))

import { useUIStore } from '../../stores/uiStore'

describe('Layout', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useUIStore).mockReturnValue(defaultUIState)
    // Reset window size
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 1024 })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering', () => {
    it('renders the layout container', () => {
      render(<Layout><div>Content</div></Layout>)

      const container = document.querySelector('.h-full')
      expect(container).toBeInTheDocument()
    })

    it('renders the Header component', () => {
      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('mock-header')).toBeInTheDocument()
    })

    it('renders children in main content area', () => {
      render(<Layout><div data-testid="test-content">Test Content</div></Layout>)

      expect(screen.getByTestId('test-content')).toBeInTheDocument()
      expect(screen.getByText('Test Content')).toBeInTheDocument()
    })
  })

  describe('desktop sidebar behavior', () => {
    it('shows sidebar when isSidebarOpen is true and not mobile', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isSidebarOpen: true,
        isMobile: false,
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('mock-sidebar')).toBeInTheDocument()
    })

    it('hides sidebar when isSidebarOpen is false on desktop', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isSidebarOpen: false,
        isMobile: false,
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('mock-sidebar')).not.toBeInTheDocument()
    })
  })

  describe('mobile sidebar behavior', () => {
    it('does not show sidebar in normal position on mobile even when open', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isSidebarOpen: true,
        isMobile: true,
      })

      render(<Layout><div>Content</div></Layout>)

      // Sidebar should be in overlay position, not in normal flex position
      // There should be a backdrop and a fixed positioned sidebar
      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).toBeInTheDocument()
    })

    it('shows mobile overlay with backdrop when sidebar open on mobile', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isSidebarOpen: true,
        isMobile: true,
      })

      render(<Layout><div>Content</div></Layout>)

      // Should have backdrop
      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).toBeInTheDocument()

      // Should have fixed positioned sidebar container
      const sidebarContainer = document.querySelector('.fixed.left-0.top-0.bottom-0.z-40.w-72')
      expect(sidebarContainer).toBeInTheDocument()
    })

    it('does not show mobile overlay when sidebar is closed on mobile', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isSidebarOpen: false,
        isMobile: true,
      })

      render(<Layout><div>Content</div></Layout>)

      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).not.toBeInTheDocument()
    })

    it('calls toggleSidebar when clicking mobile backdrop', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isSidebarOpen: true,
        isMobile: true,
      })

      render(<Layout><div>Content</div></Layout>)

      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).toBeInTheDocument()

      fireEvent.click(backdrop!)

      expect(mockToggleSidebar).toHaveBeenCalledTimes(1)
    })
  })

  describe('model selector panel', () => {
    it('does not show ModelSelector when activePanel is null', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: null,
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('mock-model-selector')).not.toBeInTheDocument()
    })

    it('shows ModelSelector when activePanel is "models"', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('mock-model-selector')).toBeInTheDocument()
    })

    it('does not show ModelSelector when activePanel is something else', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'settings',
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('mock-model-selector')).not.toBeInTheDocument()
    })

    it('shows model panel in aside element with correct width', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
        isMobile: false,
      })

      render(<Layout><div>Content</div></Layout>)

      const aside = document.querySelector('aside.w-80')
      expect(aside).toBeInTheDocument()
    })
  })

  describe('mobile model selector panel', () => {
    it('shows backdrop when model panel is open on mobile', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
        isMobile: true,
      })

      render(<Layout><div>Content</div></Layout>)

      // There should be a backdrop for the model panel
      const backdrops = document.querySelectorAll('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrops.length).toBeGreaterThan(0)
    })

    it('calls setActivePanel(null) when clicking model panel backdrop on mobile', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
        isMobile: true,
        isSidebarOpen: false, // Ensure sidebar is closed so only model panel backdrop exists
      })

      render(<Layout><div>Content</div></Layout>)

      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).toBeInTheDocument()

      fireEvent.click(backdrop!)

      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })

    it('model panel has fixed positioning on mobile', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
        isMobile: true,
      })

      render(<Layout><div>Content</div></Layout>)

      const aside = document.querySelector('aside.fixed.right-0.top-0.bottom-0.z-40')
      expect(aside).toBeInTheDocument()
    })

    it('model panel does not have fixed positioning on desktop', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
        isMobile: false,
      })

      render(<Layout><div>Content</div></Layout>)

      const aside = document.querySelector('aside.fixed')
      expect(aside).not.toBeInTheDocument()
    })
  })

  describe('resize listener', () => {
    it('calls setIsMobile on mount', () => {
      render(<Layout><div>Content</div></Layout>)

      expect(mockSetIsMobile).toHaveBeenCalled()
    })

    it('sets isMobile to true when window width is less than 768px', () => {
      Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 500 })

      render(<Layout><div>Content</div></Layout>)

      expect(mockSetIsMobile).toHaveBeenCalledWith(true)
    })

    it('sets isMobile to false when window width is 768px or more', () => {
      Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 1024 })

      render(<Layout><div>Content</div></Layout>)

      expect(mockSetIsMobile).toHaveBeenCalledWith(false)
    })

    it('responds to window resize events', () => {
      render(<Layout><div>Content</div></Layout>)

      // Clear initial call
      mockSetIsMobile.mockClear()

      // Simulate resize
      Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 500 })
      fireEvent.resize(window)

      expect(mockSetIsMobile).toHaveBeenCalledWith(true)
    })
  })

  describe('layout structure', () => {
    it('has proper flex layout structure', () => {
      render(<Layout><div>Content</div></Layout>)

      // Main container should have flex column
      const container = document.querySelector('.h-full.flex.flex-col')
      expect(container).toBeInTheDocument()

      // Content area should have flex
      const contentArea = document.querySelector('.flex-1.flex.overflow-hidden')
      expect(contentArea).toBeInTheDocument()
    })

    it('main content area has proper overflow handling', () => {
      render(<Layout><div>Content</div></Layout>)

      const main = document.querySelector('main.flex-1.flex.flex-col.min-w-0.overflow-hidden')
      expect(main).toBeInTheDocument()
    })
  })

  describe('advanced panel', () => {
    it('shows AdvancedPanel when activePanel is "advanced"', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'advanced',
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('mock-advanced-panel')).toBeInTheDocument()
    })

    it('does not show AdvancedPanel when activePanel is something else', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'models',
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('mock-advanced-panel')).not.toBeInTheDocument()
    })
  })

  describe('settings panel', () => {
    it('shows SettingsPanel when activePanel is "settings"', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'settings',
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.getByTestId('mock-settings-panel')).toBeInTheDocument()
    })

    it('does not show SettingsPanel when activePanel is something else', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activePanel: 'advanced',
      })

      render(<Layout><div>Content</div></Layout>)

      expect(screen.queryByTestId('mock-settings-panel')).not.toBeInTheDocument()
    })
  })
})
