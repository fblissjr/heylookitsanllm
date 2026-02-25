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

    it('shows aside element when panel is active', () => {
      renderAppShell({ activePanel: 'models', isMobile: false })
      expect(screen.getByRole('complementary')).toBeInTheDocument()
    })
  })

  describe('mobile panel behavior', () => {
    it('shows backdrop when panel open on mobile', () => {
      renderAppShell({ activePanel: 'models', isMobile: true })
      expect(screen.getByTestId('panel-backdrop')).toBeInTheDocument()
    })

    it('calls setActivePanel(null) when clicking mobile backdrop', () => {
      renderAppShell({ activePanel: 'models', isMobile: true })
      fireEvent.click(screen.getByTestId('panel-backdrop'))
      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })

    it('does not show backdrop on desktop', () => {
      renderAppShell({ activePanel: 'models', isMobile: false })
      expect(screen.queryByTestId('panel-backdrop')).not.toBeInTheDocument()
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

  describe('Escape key dismissal', () => {
    it('calls setActivePanel(null) when Escape is pressed with active panel', () => {
      renderAppShell({ activePanel: 'models' })
      fireEvent.keyDown(document, { key: 'Escape' })
      expect(mockSetActivePanel).toHaveBeenCalledWith(null)
    })

    it('does not call setActivePanel when Escape is pressed without active panel', () => {
      renderAppShell({ activePanel: null })
      fireEvent.keyDown(document, { key: 'Escape' })
      expect(mockSetActivePanel).not.toHaveBeenCalled()
    })

    it('does not call setActivePanel for non-Escape keys', () => {
      renderAppShell({ activePanel: 'models' })
      fireEvent.keyDown(document, { key: 'Enter' })
      expect(mockSetActivePanel).not.toHaveBeenCalled()
    })
  })

  describe('aside aria-label', () => {
    it('labels aside "Model selector" when activePanel is models', () => {
      renderAppShell({ activePanel: 'models' })
      expect(screen.getByRole('complementary', { name: 'Model selector' })).toBeInTheDocument()
    })

    it('labels aside "System prompt" when activePanel is advanced', () => {
      renderAppShell({ activePanel: 'advanced' })
      expect(screen.getByRole('complementary', { name: 'System prompt' })).toBeInTheDocument()
    })

    it('labels aside "Generation settings" when activePanel is settings', () => {
      renderAppShell({ activePanel: 'settings' })
      expect(screen.getByRole('complementary', { name: 'Generation settings' })).toBeInTheDocument()
    })
  })

  describe('layout structure', () => {
    it('uses dvh for root height', () => {
      const { container } = renderAppShell()
      const root = container.firstElementChild
      expect(root).toHaveStyle({ height: '100dvh' })
    })
  })
})
