import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Modal } from './Modal'

describe('Modal', () => {
  describe('accessibility', () => {
    it('renders with role="dialog"', () => {
      render(<Modal><p>Content</p></Modal>)
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })

    it('has aria-modal="true"', () => {
      render(<Modal><p>Content</p></Modal>)
      expect(screen.getByRole('dialog')).toHaveAttribute('aria-modal', 'true')
    })

    it('sets aria-labelledby when title is provided', () => {
      render(<Modal title="My Title"><p>Content</p></Modal>)
      expect(screen.getByRole('dialog')).toHaveAttribute('aria-labelledby', 'modal-title')
    })

    it('does not set aria-labelledby when no title', () => {
      render(<Modal><p>Content</p></Modal>)
      expect(screen.getByRole('dialog')).not.toHaveAttribute('aria-labelledby')
    })

    it('has tabIndex={-1} for programmatic focus', () => {
      render(<Modal><p>Content</p></Modal>)
      expect(screen.getByRole('dialog')).toHaveAttribute('tabindex', '-1')
    })

    it('focuses the panel on mount', () => {
      render(<Modal><p>Content</p></Modal>)
      expect(document.activeElement).toBe(screen.getByRole('dialog'))
    })
  })

  describe('Escape key', () => {
    it('calls onClose when Escape is pressed', () => {
      const onClose = vi.fn()
      render(<Modal onClose={onClose}><p>Content</p></Modal>)

      fireEvent.keyDown(document, { key: 'Escape' })
      expect(onClose).toHaveBeenCalledTimes(1)
    })

    it('does not crash when Escape pressed without onClose', () => {
      render(<Modal><p>Content</p></Modal>)
      // Should not throw
      fireEvent.keyDown(document, { key: 'Escape' })
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })

    it('does not call onClose for other keys', () => {
      const onClose = vi.fn()
      render(<Modal onClose={onClose}><p>Content</p></Modal>)

      fireEvent.keyDown(document, { key: 'Enter' })
      expect(onClose).not.toHaveBeenCalled()
    })
  })

  describe('backdrop click', () => {
    it('calls onClose when backdrop is clicked', () => {
      const onClose = vi.fn()
      const { container } = render(<Modal onClose={onClose}><p>Content</p></Modal>)

      // The backdrop is the outermost fixed div
      const backdrop = container.firstElementChild as HTMLElement
      fireEvent.click(backdrop)
      expect(onClose).toHaveBeenCalledTimes(1)
    })

    it('does not call onClose when clicking inside the dialog', () => {
      const onClose = vi.fn()
      render(<Modal onClose={onClose}><p>Content</p></Modal>)

      const dialog = screen.getByRole('dialog')
      fireEvent.click(dialog)
      expect(onClose).not.toHaveBeenCalled()
    })

    it('does not crash on backdrop click without onClose', () => {
      const { container } = render(<Modal><p>Content</p></Modal>)
      const backdrop = container.firstElementChild as HTMLElement
      // Should not throw
      fireEvent.click(backdrop)
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })
  })

  describe('maxWidth', () => {
    it('defaults to max-w-sm', () => {
      render(<Modal><p>Content</p></Modal>)
      expect(screen.getByRole('dialog').className).toContain('max-w-sm')
    })

    it('applies max-w-md when maxWidth="md"', () => {
      render(<Modal maxWidth="md"><p>Content</p></Modal>)
      expect(screen.getByRole('dialog').className).toContain('max-w-md')
    })

    it('applies max-w-lg when maxWidth="lg"', () => {
      render(<Modal maxWidth="lg"><p>Content</p></Modal>)
      expect(screen.getByRole('dialog').className).toContain('max-w-lg')
    })
  })

  describe('content', () => {
    it('renders children', () => {
      render(<Modal><p>Test content</p></Modal>)
      expect(screen.getByText('Test content')).toBeInTheDocument()
    })
  })
})
