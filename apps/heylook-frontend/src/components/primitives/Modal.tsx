import { useEffect, useRef, type ReactNode } from 'react'

interface ModalProps {
  children: ReactNode
  onClose?: () => void
  title?: string
  maxWidth?: 'sm' | 'md' | 'lg'
}

const widthClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
} as const

export function Modal({ children, onClose, title, maxWidth = 'sm' }: ModalProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const titleId = title ? 'modal-title' : undefined

  // Focus the panel on mount, dismiss on Escape
  useEffect(() => {
    panelRef.current?.focus()

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onClose) {
        onClose()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => { if (e.target === e.currentTarget && onClose) onClose() }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className={`bg-gray-800 rounded-2xl shadow-2xl w-full ${widthClasses[maxWidth]} overflow-hidden border border-gray-700 animate-in fade-in zoom-in-95 duration-200 outline-none`}
      >
        {children}
      </div>
    </div>
  )
}
