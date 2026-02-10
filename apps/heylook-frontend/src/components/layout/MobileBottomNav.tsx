import { NavLink } from 'react-router-dom'
import { navItems } from './AppNav'
import clsx from 'clsx'

export function MobileBottomNav() {
  return (
    <nav className="shrink-0 bg-gray-900 border-t border-gray-800 flex items-center justify-around h-mobile-nav pb-safe">
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) => clsx(
            'flex flex-col items-center justify-center gap-0.5 px-2 py-1 min-w-0',
            isActive
              ? 'text-primary'
              : 'text-gray-500'
          )}
        >
          <Icon className="w-5 h-5" />
          <span className="text-[10px] font-medium truncate">{label}</span>
        </NavLink>
      ))}
    </nav>
  )
}
