import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { AppShell } from './AppShell'

// Mock child components
vi.mock('./AppNav', () => ({
  AppNav: () => <nav data-testid="mock-app-nav">AppNav</nav>,
  navItems: [],
}))

vi.mock('./MobileBottomNav', () => ({
  MobileBottomNav: () => <nav data-testid="mock-mobile-nav">MobileBottomNav</nav>,
}))

vi.mock('./Header', () => ({
  Header: () => <header data-testid="mock-header">Header</header>,
}))

vi.mock('./SystemStatusBar', () => ({
  SystemStatusBar: () => <div data-testid="mock-status-bar">StatusBar</div>,
}))

vi.mock('../composed/ModelSelector', () => ({
  ModelSelector: () => <div data-testid="mock-model-selector">ModelSelector</div>,
}))

vi.mock('../../applets/chat/components/AdvancedPanel', () => ({
  AdvancedPanel: () => <div data-testid="mock-advanced-panel">AdvancedPanel</div>,
}))

vi.mock('../panels/SettingsPanel', () => ({
  SettingsPanel: () => <div data-testid="mock-settings-panel">SettingsPanel</div>,
}))

// Mock store state
const mockSetIsMobile = vi.fn()
const mockSetActivePanel = vi.fn()

const defaultUIState = {
  isMobile: false,
  activePanel: null as string | null,
  setActivePanel: mockSetActivePanel,
}

vi.mock('../../stores/uiStore', () => ({
  useUIStore: Object.assign(
    vi.fn((selector: (s: Record<string, unknown>) => unknown) => selector(defaultUIState)),
    { getState: () => ({ setIsMobile: mockSetIsMobile }) }
  ),
}))

import { useUIStore } from '../../stores/uiStore'

function renderAppShell(state: Partial<typeof defaultUIState> = {}) {
  const mergedState = { ...defaultUIState, ...state }
  vi.mocked(useUIStore).mockImplementation(
    ((selector: (s: typeof defaultUIState) => unknown) => selector(mergedState)) as typeof useUIStore
  )
  return render(
    <MemoryRouter initialEntries={['/chat']}>
      <AppShell />
    </MemoryRouter>
  )
}

describe('AppShell', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 1024 })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering', () => {
    it('renders Header', () => {
      renderAppShell()
      expect(screen.getByTestId('mock-header')).toBeInTheDocument()
    })

    it('renders SystemStatusBar', () => {
      renderAppShell()
      expect(screen.getByTestId('mock-status-bar')).toBeInTheDocument()
    })

    it('renders AppNav on desktop', () => {
      renderAppShell({ isMobile: false })
      expect(screen.getByTestId('mock-app-nav')).toBeInTheDocument()
    })

    it('hides AppNav on mobile', () => {
      renderAppShell({ isMobile: true })
      expect(screen.queryByTestId('mock-app-nav')).not.toBeInTheDocument()
    })

    it('renders MobileBottomNav on mobile', () => {
      renderAppShell({ isMobile: true })
      expect(screen.getByTestId('mock-mobile-nav')).toBeInTheDocument()
    })

    it('hides MobileBottomNav on desktop', () => {
      renderAppShell({ isMobile: false })
      expect(screen.queryByTestId('mock-mobile-nav')).not.toBeInTheDocument()
    })
  })

  describe('right panels', () => {
    it('does not show any panel when activePanel is null', () => {
      renderAppShell({ activePanel: null })
      expect(screen.queryByTestId('mock-model-selector')).not.toBeInTheDocument()
      expect(screen.queryByTestId('mock-advanced-panel')).not.toBeInTheDocument()
      expect(screen.queryByTestId('mock-settings-panel')).not.toBeInTheDocument()
    })

    it('shows ModelSelector when activePanel is "models"', () => {
      renderAppShell({ activePanel: 'models' })
      expect(screen.getByTestId('mock-model-selector')).toBeInTheDocument()
    })

    it('shows AdvancedPanel when activePanel is "advanced"', () => {
      renderAppShell({ activePanel: 'advanced' })
      expect(screen.getByTestId('mock-advanced-panel')).toBeInTheDocument()
    })

    it('shows SettingsPanel when activePanel is "settings"', () => {
      renderAppShell({ activePanel: 'settings' })
      expect(screen.getByTestId('mock-settings-panel')).toBeInTheDocument()
    })

    it('shows aside with correct width', () => {
      renderAppShell({ activePanel: 'models', isMobile: false })
      const aside = document.querySelector('aside.w-80')
      expect(aside).toBeInTheDocument()
    })
  })

  describe('mobile panel behavior', () => {
    it('shows backdrop when panel open on mobile', () => {
      renderAppShell({ activePanel: 'models', isMobile: true })
      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).toBeInTheDocument()
    })

    it('calls setActivePanel(null) when clicking mobile backdrop', () => {
      renderAppShell({ activePanel: 'models', isMobile: true })
      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50.z-30')
      expect(backdrop).toBeInTheDocument()
      fireEvent.click(backdrop!)
      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })

    it('panel has fixed positioning on mobile', () => {
      renderAppShell({ activePanel: 'models', isMobile: true })
      const aside = document.querySelector('aside.fixed.right-0.top-0.bottom-0.z-40')
      expect(aside).toBeInTheDocument()
    })

    it('panel does not have fixed positioning on desktop', () => {
      renderAppShell({ activePanel: 'models', isMobile: false })
      const aside = document.querySelector('aside.fixed')
      expect(aside).not.toBeInTheDocument()
    })
  })

  describe('resize listener', () => {
    it('calls setIsMobile on mount', () => {
      renderAppShell()
      expect(mockSetIsMobile).toHaveBeenCalled()
    })

    it('sets isMobile to true when window width < 768px', () => {
      Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 500 })
      renderAppShell()
      expect(mockSetIsMobile).toHaveBeenCalledWith(true)
    })

    it('sets isMobile to false when window width >= 768px', () => {
      Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 1024 })
      renderAppShell()
      expect(mockSetIsMobile).toHaveBeenCalledWith(false)
    })

    it('responds to window resize events', () => {
      renderAppShell()
      mockSetIsMobile.mockClear()

      Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 500 })
      fireEvent.resize(window)

      expect(mockSetIsMobile).toHaveBeenCalledWith(true)
    })
  })

  describe('layout structure', () => {
    it('uses dvh for root height', () => {
      renderAppShell()
      const root = document.querySelector('.h-screen.flex.overflow-hidden')
      expect(root).toBeInTheDocument()
      expect(root).toHaveStyle({ height: '100dvh' })
    })
  })
})
