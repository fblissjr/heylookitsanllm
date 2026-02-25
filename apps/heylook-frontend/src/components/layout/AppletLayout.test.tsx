import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { AppletLayout } from './AppletLayout'

// Mock store state
const defaultUIState = {
  isMobile: false,
}

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn((selector: (s: Record<string, unknown>) => unknown) => selector(defaultUIState)),
}))

import { useUIStore } from '../../stores/uiStore'

function renderLayout(isMobile: boolean) {
  vi.mocked(useUIStore).mockImplementation(
    ((selector: (s: typeof defaultUIState) => unknown) => selector({ isMobile })) as typeof useUIStore
  )
  return render(
    <AppletLayout
      leftPanel={<div data-testid="left-panel">Left Panel Content</div>}
      leftPanelWidth="w-80"
    >
      <div data-testid="main-content">Main Content</div>
    </AppletLayout>
  )
}

describe('AppletLayout', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('desktop behavior', () => {
    it('renders left panel inline', () => {
      renderLayout(false)
      expect(screen.getByTestId('left-panel')).toBeInTheDocument()
      expect(screen.getByText('Left Panel Content')).toBeInTheDocument()
    })

    it('renders main content', () => {
      renderLayout(false)
      expect(screen.getByTestId('main-content')).toBeInTheDocument()
    })

    it('uses provided width class for left panel', () => {
      renderLayout(false)
      const panelWrapper = screen.getByTestId('left-panel').parentElement
      expect(panelWrapper).toHaveClass('w-80')
    })

    it('does not show mobile toggle button', () => {
      renderLayout(false)
      expect(screen.queryByLabelText('Open controls panel')).not.toBeInTheDocument()
    })

    it('renders side-by-side with both panels visible', () => {
      renderLayout(false)
      expect(screen.getByTestId('left-panel')).toBeInTheDocument()
      expect(screen.getByTestId('main-content')).toBeInTheDocument()
    })
  })

  describe('mobile behavior', () => {
    it('shows controls toggle button', () => {
      renderLayout(true)
      expect(screen.getByLabelText('Open controls panel')).toBeInTheDocument()
    })

    it('hides left panel by default', () => {
      renderLayout(true)
      expect(screen.queryByTestId('left-panel')).not.toBeInTheDocument()
    })

    it('renders main content', () => {
      renderLayout(true)
      expect(screen.getByTestId('main-content')).toBeInTheDocument()
    })

    it('shows left panel overlay when toggle clicked', () => {
      renderLayout(true)

      const toggle = screen.getByLabelText('Open controls panel')
      fireEvent.click(toggle)

      expect(screen.getByTestId('left-panel')).toBeInTheDocument()
    })

    it('shows backdrop when panel overlay is open', () => {
      renderLayout(true)

      const toggle = screen.getByLabelText('Open controls panel')
      fireEvent.click(toggle)

      expect(screen.getByTestId('applet-panel-backdrop')).toBeInTheDocument()
    })

    it('closes overlay when backdrop clicked', () => {
      renderLayout(true)

      // Open
      fireEvent.click(screen.getByLabelText('Open controls panel'))
      expect(screen.getByTestId('left-panel')).toBeInTheDocument()

      // Close via backdrop
      fireEvent.click(screen.getByTestId('applet-panel-backdrop'))
      expect(screen.queryByTestId('left-panel')).not.toBeInTheDocument()
    })

    it('closes overlay when close button clicked', () => {
      renderLayout(true)

      // Open
      fireEvent.click(screen.getByLabelText('Open controls panel'))
      expect(screen.getByTestId('left-panel')).toBeInTheDocument()

      // Close via X button
      fireEvent.click(screen.getByLabelText('Close controls panel'))
      expect(screen.queryByTestId('left-panel')).not.toBeInTheDocument()
    })

    it('shows left panel content in overlay when open', () => {
      renderLayout(true)

      fireEvent.click(screen.getByLabelText('Open controls panel'))

      expect(screen.getByTestId('left-panel')).toBeInTheDocument()
      expect(screen.getByText('Left Panel Content')).toBeInTheDocument()
    })
  })
})
