import { ReactNode, useState } from 'react'
import { useUIStore } from '../../stores/uiStore'

interface AppletLayoutProps {
  /** Content for the left panel (form, controls, etc.) */
  leftPanel: ReactNode
  /** Tailwind width class for the desktop left panel wrapper (default: 'w-72') */
  leftPanelWidth?: string
  /** Main content area */
  children: ReactNode
}

/**
 * Shared responsive wrapper for applets with a left panel + main content.
 * Desktop: inline left panel with configurable width.
 * Mobile: panel hidden behind toggle button, shown as overlay.
 */
export function AppletLayout({
  leftPanel,
  leftPanelWidth = 'w-72',
  children,
}: AppletLayoutProps) {
  const isMobile = useUIStore((s) => s.isMobile)
  const [panelOpen, setPanelOpen] = useState(false)

  if (!isMobile) {
    return (
      <div className="h-full flex overflow-hidden">
        <div className={`${leftPanelWidth} flex-shrink-0 border-r border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden`}>
          {leftPanel}
        </div>
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {children}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Mobile toggle bar */}
      <div className="shrink-0 px-3 py-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-surface-darker">
        <button
          onClick={() => setPanelOpen(true)}
          className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-primary transition-colors"
          aria-label="Open controls panel"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
          </svg>
          Controls
        </button>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {children}
      </div>

      {/* Mobile panel overlay */}
      {panelOpen && (
        <>
          <div
            className="fixed inset-0 bg-black/50 z-30"
            onClick={() => setPanelOpen(false)}
            data-testid="applet-panel-backdrop"
          />
          <div className="fixed left-0 top-0 bottom-mobile-nav z-40 w-80 bg-white dark:bg-surface-dark flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 dark:border-gray-700">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Controls</span>
              <button
                onClick={() => setPanelOpen(false)}
                className="p-1 rounded text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                aria-label="Close controls panel"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {leftPanel}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
