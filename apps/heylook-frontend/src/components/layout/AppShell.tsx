import { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { AppNav } from './AppNav'
import { MobileBottomNav } from './MobileBottomNav'
import { Header } from './Header'
import { SystemStatusBar } from './SystemStatusBar'
import { ModelSelector } from '../composed/ModelSelector'
import { AdvancedPanel } from '../../applets/chat/components/AdvancedPanel'
import { SettingsPanel } from '../panels/SettingsPanel'
import { useUIStore } from '../../stores/uiStore'

export function AppShell() {
  const isMobile = useUIStore((s) => s.isMobile)
  const activePanel = useUIStore((s) => s.activePanel)
  const setActivePanel = useUIStore((s) => s.setActivePanel)

  // Detect mobile viewport -- runs for all routes
  useEffect(() => {
    const checkMobile = () => {
      useUIStore.getState().setIsMobile(window.innerWidth < 768)
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  return (
    <div
      style={{ height: '100dvh' }}
      className="h-screen flex overflow-hidden bg-background-light dark:bg-background-dark"
    >
      {!isMobile && <AppNav />}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header />
        <div className="flex-1 flex overflow-hidden min-h-0">
          <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
            <Outlet />
            <SystemStatusBar />
          </main>

          {/* Right panels (shared across all routes) */}
          {activePanel && (
            <>
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
        {isMobile && <MobileBottomNav />}
      </div>
    </div>
  )
}
