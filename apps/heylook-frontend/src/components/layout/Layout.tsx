import { ReactNode } from 'react'
import { Sidebar } from '../../applets/chat/components/Sidebar'
import { useUIStore } from '../../stores/uiStore'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  const isSidebarOpen = useUIStore((s) => s.isSidebarOpen)
  const isMobile = useUIStore((s) => s.isMobile)

  return (
    <div className="h-full flex overflow-hidden">
      {/* Desktop sidebar */}
      {!isMobile && isSidebarOpen && <Sidebar />}

      {/* Mobile sidebar overlay */}
      {isMobile && isSidebarOpen && (
        <>
          <div
            className="fixed inset-0 bg-black/50 z-30"
            onClick={() => useUIStore.getState().toggleSidebar()}
          />
          <div className="fixed left-0 top-0 bottom-mobile-nav z-40 w-72">
            <Sidebar />
          </div>
        </>
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {children}
      </div>
    </div>
  )
}
