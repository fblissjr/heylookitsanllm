import { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { MemoryRouter, MemoryRouterProps } from 'react-router-dom'
import { ThemeProvider } from '../contexts/ThemeContext'

interface AppRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  routerProps?: MemoryRouterProps
}

/**
 * Render utility that wraps components in app-level providers.
 * Use this instead of raw `render()` when the component under test
 * needs routing context (useNavigate, Link, etc.) or theme context.
 */
export function renderWithProviders(
  ui: ReactElement,
  { routerProps, ...renderOptions }: AppRenderOptions = {},
) {
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <MemoryRouter {...routerProps}>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </MemoryRouter>
    )
  }

  return render(ui, { wrapper: Wrapper, ...renderOptions })
}
