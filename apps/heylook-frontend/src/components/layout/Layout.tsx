import { ReactNode, useEffect } from 'react'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { SystemStatusBar } from './SystemStatusBar'
import { useUIStore } from '../../stores/uiStore'
import { ModelSelector } from '../../features/models/components/ModelSelector'
import { AdvancedPanel } from '../panels/AdvancedPanel'
import { SettingsPanel } from '../panels/SettingsPanel'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  const { isSidebarOpen, isMobile, activePanel, setActivePanel } = useUIStore()

  // Detect mobile viewport
  // Use getState() to avoid memory leak from unstable action references
  useEffect(() => {
    const checkMobile = () => {
      useUIStore.getState().setIsMobile(window.innerWidth < 768)
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-background-light dark:bg-background-dark">
      <Header />

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - hidden on mobile unless explicitly opened */}
        {!isMobile && isSidebarOpen && (
          <Sidebar />
        )}

        {/* Mobile sidebar overlay */}
        {isMobile && isSidebarOpen && (
          <>
            <div
              className="fixed inset-0 bg-black/50 z-30"
              onClick={() => useUIStore.getState().toggleSidebar()}
            />
            <div className="fixed left-0 top-0 bottom-0 z-40 w-72">
              <Sidebar />
            </div>
          </>
        )}

        {/* Main content */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <div className="flex-1 min-h-0">
            {children}
          </div>
          <SystemStatusBar />
        </main>

        {/* Right panels */}
        {activePanel && (
          <>
            {/* Backdrop for mobile */}
            {isMobile && (
              <div
                className="fixed inset-0 bg-black/50 z-30"
                onClick={() => setActivePanel(null)}
              />
            )}
            <aside className={`
              ${isMobile ? 'fixed right-0 top-0 bottom-0 z-40' : ''}
              w-80 bg-white dark:bg-surface-dark border-l border-gray-200 dark:border-gray-700
              flex flex-col overflow-hidden
            `}>
              {activePanel === 'models' && <ModelSelector />}
              {activePanel === 'advanced' && <AdvancedPanel />}
              {activePanel === 'settings' && <SettingsPanel />}
            </aside>
          </>
        )}
      </div>
    </div>
  )
}
