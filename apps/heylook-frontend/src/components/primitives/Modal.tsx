import type { ReactNode } from 'react'

interface ModalProps {
  children: ReactNode
  maxWidth?: 'sm' | 'md' | 'lg'
}

const widthClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
} as const

export function Modal({ children, maxWidth = 'sm' }: ModalProps) {
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className={`bg-gray-800 rounded-2xl shadow-2xl w-full ${widthClasses[maxWidth]} overflow-hidden border border-gray-700 animate-in fade-in zoom-in-95 duration-200`}>
        {children}
      </div>
    </div>
  )
}
