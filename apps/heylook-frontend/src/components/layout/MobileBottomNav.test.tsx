import { describe, it, expect, vi } from 'vitest'
import { screen } from '@testing-library/react'
import { MobileBottomNav } from './MobileBottomNav'
import { renderWithProviders } from '../../test/render'

// Mock the navItems from AppNav
vi.mock('./AppNav', () => ({
  navItems: [
    { to: '/chat', label: 'Chat', icon: ({ className }: { className?: string }) => <span data-testid="icon-chat" className={className}>C</span> },
    { to: '/batch', label: 'Batch', icon: ({ className }: { className?: string }) => <span data-testid="icon-batch" className={className}>B</span> },
    { to: '/explore', label: 'Explore', icon: ({ className }: { className?: string }) => <span data-testid="icon-explore" className={className}>E</span> },
    { to: '/compare', label: 'Compare', icon: ({ className }: { className?: string }) => <span data-testid="icon-compare" className={className}>S</span> },
    { to: '/perf', label: 'Perf', icon: ({ className }: { className?: string }) => <span data-testid="icon-perf" className={className}>P</span> },
    { to: '/notebook', label: 'Note', icon: ({ className }: { className?: string }) => <span data-testid="icon-note" className={className}>N</span> },
  ],
}))

describe('MobileBottomNav', () => {
  it('renders all 6 nav items', () => {
    renderWithProviders(<MobileBottomNav />, { routerProps: { initialEntries: ['/chat'] } })

    expect(screen.getByText('Chat')).toBeInTheDocument()
    expect(screen.getByText('Batch')).toBeInTheDocument()
    expect(screen.getByText('Explore')).toBeInTheDocument()
    expect(screen.getByText('Compare')).toBeInTheDocument()
    expect(screen.getByText('Perf')).toBeInTheDocument()
    expect(screen.getByText('Note')).toBeInTheDocument()
  })

  it('renders icons for each nav item', () => {
    renderWithProviders(<MobileBottomNav />, { routerProps: { initialEntries: ['/chat'] } })

    expect(screen.getByTestId('icon-chat')).toBeInTheDocument()
    expect(screen.getByTestId('icon-batch')).toBeInTheDocument()
    expect(screen.getByTestId('icon-explore')).toBeInTheDocument()
  })

  it('renders as a nav element', () => {
    renderWithProviders(<MobileBottomNav />, { routerProps: { initialEntries: ['/chat'] } })

    const nav = document.querySelector('nav')
    expect(nav).toBeInTheDocument()
  })

  it('has proper mobile bottom bar styling', () => {
    renderWithProviders(<MobileBottomNav />, { routerProps: { initialEntries: ['/chat'] } })

    const nav = document.querySelector('nav.bg-gray-900.border-t.h-16')
    expect(nav).toBeInTheDocument()
  })

  it('highlights active route', () => {
    renderWithProviders(<MobileBottomNav />, { routerProps: { initialEntries: ['/chat'] } })

    const chatLink = screen.getByText('Chat').closest('a')
    expect(chatLink).toHaveClass('text-primary')

    const batchLink = screen.getByText('Batch').closest('a')
    expect(batchLink).toHaveClass('text-gray-500')
  })

  it('highlights different route when active', () => {
    renderWithProviders(<MobileBottomNav />, { routerProps: { initialEntries: ['/batch'] } })

    const batchLink = screen.getByText('Batch').closest('a')
    expect(batchLink).toHaveClass('text-primary')

    const chatLink = screen.getByText('Chat').closest('a')
    expect(chatLink).toHaveClass('text-gray-500')
  })
})
