import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { EmptyState } from './EmptyState'

describe('EmptyState', () => {
  describe('no-model type', () => {
    it('renders "No Model Loaded" heading', () => {
      render(<EmptyState type="no-model" />)

      expect(screen.getByRole('heading', { name: 'No Model Loaded' })).toBeInTheDocument()
    })

    it('renders description about selecting a model', () => {
      render(<EmptyState type="no-model" />)

      expect(screen.getByText(/Select a model from the header to start chatting/)).toBeInTheDocument()
    })

    it('renders computer monitor icon', () => {
      render(<EmptyState type="no-model" />)

      // The icon container has specific dimensions and contains an SVG
      const svgElement = document.querySelector('svg')
      expect(svgElement).toBeInTheDocument()

      // Check for computer monitor path (contains "M5 17h14" and "M3 13h18")
      const path = svgElement?.querySelector('path')
      expect(path).toBeInTheDocument()
      const pathD = path?.getAttribute('d')
      expect(pathD).toContain('M3 13h18')
    })

    it('renders with proper styling structure', () => {
      const { container } = render(<EmptyState type="no-model" />)

      // The outermost div should have text-center and max-w-md classes
      const outerDiv = container.querySelector('.text-center')
      expect(outerDiv).toBeInTheDocument()
      expect(outerDiv).toHaveClass('max-w-md')
    })
  })

  describe('no-conversation type', () => {
    it('renders "Start a New Conversation" heading', () => {
      render(<EmptyState type="no-conversation" />)

      expect(screen.getByRole('heading', { name: 'Start a New Conversation' })).toBeInTheDocument()
    })

    it('renders description about starting a conversation', () => {
      render(<EmptyState type="no-conversation" />)

      expect(screen.getByText(/Click "New Chat" in the sidebar/)).toBeInTheDocument()
    })

    it('renders chat bubble icon', () => {
      render(<EmptyState type="no-conversation" />)

      const svgElement = document.querySelector('svg')
      expect(svgElement).toBeInTheDocument()

      // Check for chat bubble path (contains "M21 12c0 4.418")
      const path = svgElement?.querySelector('path')
      expect(path).toBeInTheDocument()
      const pathD = path?.getAttribute('d')
      expect(pathD).toContain('M21 12c0 4.418')
    })

    it('renders with proper styling structure', () => {
      const { container } = render(<EmptyState type="no-conversation" />)

      // The outermost div should have text-center and max-w-md classes
      const outerDiv = container.querySelector('.text-center')
      expect(outerDiv).toBeInTheDocument()
      expect(outerDiv).toHaveClass('max-w-md')
    })
  })

  describe('visual elements', () => {
    it('renders icon container with expected dimensions for no-model', () => {
      render(<EmptyState type="no-model" />)

      // Find the icon container by class
      const iconContainer = document.querySelector('.w-16.h-16')
      expect(iconContainer).toBeInTheDocument()
      expect(iconContainer).toHaveClass('rounded-full')
    })

    it('renders icon container with expected dimensions for no-conversation', () => {
      render(<EmptyState type="no-conversation" />)

      const iconContainer = document.querySelector('.w-16.h-16')
      expect(iconContainer).toBeInTheDocument()
      expect(iconContainer).toHaveClass('rounded-full')
    })

    it('heading has proper text styling', () => {
      render(<EmptyState type="no-model" />)

      const heading = screen.getByRole('heading')
      expect(heading).toHaveClass('text-lg')
      expect(heading).toHaveClass('font-medium')
    })

    it('description text has subdued styling', () => {
      render(<EmptyState type="no-conversation" />)

      const description = screen.getByText(/Click "New Chat" in the sidebar/)
      expect(description).toHaveClass('text-gray-500')
      expect(description).toHaveClass('text-sm')
    })
  })
})
