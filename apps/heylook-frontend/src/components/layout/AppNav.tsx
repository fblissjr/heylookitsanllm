import { NavLink } from 'react-router-dom'
import { ChatBubbleIcon, LayersIcon } from '../icons'
import clsx from 'clsx'

interface NavItem {
  to: string
  label: string
  icon: typeof ChatBubbleIcon
}

const navItems: NavItem[] = [
  { to: '/chat', label: 'Chat', icon: ChatBubbleIcon },
  { to: '/batch', label: 'Batch', icon: LayersIcon },
]

export function AppNav() {
  return (
    <nav className="w-14 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col items-center py-3 gap-2">
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          title={label}
          className={({ isActive }) => clsx(
            'w-10 h-10 rounded-xl flex items-center justify-center transition-colors',
            isActive
              ? 'bg-primary text-white'
              : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
          )}
        >
          <Icon className="w-5 h-5" />
        </NavLink>
      ))}
    </nav>
  )
}
