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

    it('renders an icon', () => {
      const { container } = render(<EmptyState type="no-model" />)

      expect(container.querySelector('svg')).toBeInTheDocument()
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

    it('renders an icon', () => {
      const { container } = render(<EmptyState type="no-conversation" />)

      expect(container.querySelector('svg')).toBeInTheDocument()
    })
  })

  describe('visual elements', () => {
    it('renders icon container for no-model', () => {
      const { container } = render(<EmptyState type="no-model" />)

      // Icon exists inside the component
      const svg = container.querySelector('svg')
      expect(svg).toBeInTheDocument()
      // SVG has a parent container element
      expect(svg?.parentElement).toBeInTheDocument()
    })

    it('renders icon container for no-conversation', () => {
      const { container } = render(<EmptyState type="no-conversation" />)

      const svg = container.querySelector('svg')
      expect(svg).toBeInTheDocument()
      expect(svg?.parentElement).toBeInTheDocument()
    })
  })
})
