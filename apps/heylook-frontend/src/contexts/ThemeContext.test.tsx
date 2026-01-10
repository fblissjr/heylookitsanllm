import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider, useTheme, type Theme } from './ThemeContext'

// Test component that uses the hook
function TestConsumer() {
  const { theme, resolvedTheme, setTheme } = useTheme()
  return (
    <div>
      <span data-testid="theme">{theme}</span>
      <span data-testid="resolved">{resolvedTheme}</span>
      <button onClick={() => setTheme('light')}>Set Light</button>
      <button onClick={() => setTheme('dark')}>Set Dark</button>
      <button onClick={() => setTheme('auto')}>Set Auto</button>
    </div>
  )
}

// Mock localStorage
const mockLocalStorage = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key]
    }),
    clear: vi.fn(() => {
      store = {}
    }),
    get store() {
      return store
    },
  }
})()

// Mock matchMedia with controllable matches property
let mockMediaQueryMatches = false
const mockMediaQueryListeners: Array<(e: MediaQueryListEvent) => void> = []

const createMockMediaQueryList = (query: string) => ({
  matches: mockMediaQueryMatches,
  media: query,
  onchange: null,
  addListener: vi.fn(),
  removeListener: vi.fn(),
  addEventListener: vi.fn((_, handler: (e: MediaQueryListEvent) => void) => {
    mockMediaQueryListeners.push(handler)
  }),
  removeEventListener: vi.fn((_, handler: (e: MediaQueryListEvent) => void) => {
    const index = mockMediaQueryListeners.indexOf(handler)
    if (index > -1) {
      mockMediaQueryListeners.splice(index, 1)
    }
  }),
  dispatchEvent: vi.fn(),
})

describe('ThemeContext', () => {
  beforeEach(() => {
    mockLocalStorage.clear()
    mockMediaQueryMatches = false
    mockMediaQueryListeners.length = 0

    Object.defineProperty(window, 'localStorage', {
      value: mockLocalStorage,
      writable: true,
    })

    Object.defineProperty(window, 'matchMedia', {
      value: vi.fn().mockImplementation(createMockMediaQueryList),
      writable: true,
    })

    // Reset documentElement classes
    document.documentElement.classList.remove('dark')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('ThemeProvider initialization', () => {
    it('defaults to dark theme when localStorage is empty', () => {
      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('theme')).toHaveTextContent('dark')
      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('reads theme from localStorage on initialization', () => {
      mockLocalStorage.setItem('heylook:theme', 'light')

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('theme')).toHaveTextContent('light')
      expect(screen.getByTestId('resolved')).toHaveTextContent('light')
    })

    it('reads dark theme from localStorage', () => {
      mockLocalStorage.setItem('heylook:theme', 'dark')

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('theme')).toHaveTextContent('dark')
      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('reads auto theme from localStorage', () => {
      mockLocalStorage.setItem('heylook:theme', 'auto')
      mockMediaQueryMatches = true // System prefers dark

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('theme')).toHaveTextContent('auto')
      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('ignores invalid localStorage values and defaults to dark', () => {
      mockLocalStorage.setItem('heylook:theme', 'invalid-theme')

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('theme')).toHaveTextContent('dark')
    })
  })

  describe('setTheme', () => {
    it('updates theme to light', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Light'))

      expect(screen.getByTestId('theme')).toHaveTextContent('light')
      expect(screen.getByTestId('resolved')).toHaveTextContent('light')
    })

    it('updates theme to dark', async () => {
      const user = userEvent.setup()
      mockLocalStorage.setItem('heylook:theme', 'light')

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Dark'))

      expect(screen.getByTestId('theme')).toHaveTextContent('dark')
      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('updates theme to auto', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Auto'))

      expect(screen.getByTestId('theme')).toHaveTextContent('auto')
    })

    it('persists theme to localStorage', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Light'))

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('heylook:theme', 'light')
    })
  })

  describe('resolvedTheme with auto mode', () => {
    it('resolves to light when system prefers light', async () => {
      const user = userEvent.setup()
      mockMediaQueryMatches = false // System prefers light

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Auto'))

      expect(screen.getByTestId('theme')).toHaveTextContent('auto')
      expect(screen.getByTestId('resolved')).toHaveTextContent('light')
    })

    it('resolves to dark when system prefers dark', async () => {
      const user = userEvent.setup()
      mockMediaQueryMatches = true // System prefers dark

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Auto'))

      expect(screen.getByTestId('theme')).toHaveTextContent('auto')
      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('responds to system theme changes in auto mode', async () => {
      const user = userEvent.setup()
      mockMediaQueryMatches = false // Start with light

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Auto'))
      expect(screen.getByTestId('resolved')).toHaveTextContent('light')

      // Simulate system theme change to dark
      act(() => {
        mockMediaQueryListeners.forEach((listener) => {
          listener({ matches: true } as MediaQueryListEvent)
        })
      })

      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('stops listening to system changes when switching from auto', async () => {
      const user = userEvent.setup()
      mockMediaQueryMatches = false

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      // Switch to auto mode
      await user.click(screen.getByText('Set Auto'))
      expect(mockMediaQueryListeners.length).toBe(1)

      // Switch to explicit dark mode
      await user.click(screen.getByText('Set Dark'))

      // Listener should be removed (cleanup runs)
      // Note: due to re-render, listener count may vary, but resolved should not change
      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')

      // System theme change should not affect resolved theme
      act(() => {
        mockMediaQueryListeners.forEach((listener) => {
          listener({ matches: false } as MediaQueryListEvent)
        })
      })

      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })
  })

  describe('document class management', () => {
    it('adds dark class when resolvedTheme is dark', () => {
      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(document.documentElement.classList.contains('dark')).toBe(true)
    })

    it('removes dark class when resolvedTheme is light', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Light'))

      expect(document.documentElement.classList.contains('dark')).toBe(false)
    })

    it('toggles dark class when theme changes', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      // Initial: dark
      expect(document.documentElement.classList.contains('dark')).toBe(true)

      // Switch to light
      await user.click(screen.getByText('Set Light'))
      expect(document.documentElement.classList.contains('dark')).toBe(false)

      // Switch back to dark
      await user.click(screen.getByText('Set Dark'))
      expect(document.documentElement.classList.contains('dark')).toBe(true)
    })
  })

  describe('useTheme hook', () => {
    it('throws error when used outside ThemeProvider', () => {
      // Suppress console.error for this test
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      expect(() => {
        render(<TestConsumer />)
      }).toThrow('useTheme must be used within a ThemeProvider')

      consoleSpy.mockRestore()
    })

    it('provides theme value', () => {
      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('theme')).toHaveTextContent('dark')
    })

    it('provides resolvedTheme value', () => {
      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      expect(screen.getByTestId('resolved')).toHaveTextContent('dark')
    })

    it('provides setTheme function', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Light'))
      expect(screen.getByTestId('theme')).toHaveTextContent('light')
    })
  })

  describe('Theme type', () => {
    it('accepts valid theme values', () => {
      const themes: Theme[] = ['light', 'dark', 'auto']
      themes.forEach((theme) => {
        mockLocalStorage.setItem('heylook:theme', theme)

        const { unmount } = render(
          <ThemeProvider>
            <TestConsumer />
          </ThemeProvider>
        )

        expect(screen.getByTestId('theme')).toHaveTextContent(theme)
        unmount()
      })
    })
  })

  describe('edge cases', () => {
    it('handles rapid theme changes', async () => {
      const user = userEvent.setup()

      render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Light'))
      await user.click(screen.getByText('Set Dark'))
      await user.click(screen.getByText('Set Auto'))
      await user.click(screen.getByText('Set Light'))

      expect(screen.getByTestId('theme')).toHaveTextContent('light')
      expect(screen.getByTestId('resolved')).toHaveTextContent('light')
    })

    it('maintains state across re-renders', async () => {
      const user = userEvent.setup()

      const { rerender } = render(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      await user.click(screen.getByText('Set Light'))

      rerender(
        <ThemeProvider>
          <TestConsumer />
        </ThemeProvider>
      )

      // Theme should persist (read from localStorage on re-mount)
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('heylook:theme', 'light')
    })
  })
})
