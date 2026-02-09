import { Outlet } from 'react-router-dom'
import { AppNav } from './AppNav'
import { useUIStore } from '../../stores/uiStore'

export function AppShell() {
  const isMobile = useUIStore((s) => s.isMobile)

  return (
    <div className="h-screen flex overflow-hidden bg-background-light dark:bg-background-dark">
      {!isMobile && <AppNav />}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Outlet />
      </div>
    </div>
  )
}
