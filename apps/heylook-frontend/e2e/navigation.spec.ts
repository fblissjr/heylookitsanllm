import { test, expect } from './fixtures'

/**
 * E2E tests for cross-applet navigation and routing.
 * Tests all 7 applet routes, lazy loading, state preservation, and unknown routes.
 */

const APPLET_ROUTES = [
  { path: '/chat', name: 'Chat' },
  { path: '/batch', name: 'Batch' },
  { path: '/explore', name: 'Token Explorer' },
  { path: '/compare', name: 'Model Comparison' },
  { path: '/perf', name: 'Performance' },
  { path: '/notebook', name: 'Notebook' },
  { path: '/models', name: 'Models' },
] as const

test.describe('Route Navigation', () => {
  for (const route of APPLET_ROUTES) {
    test(`navigates to ${route.name} (${route.path})`, async ({ backendPage: page }) => {
      await page.goto(route.path)

      // Each applet should render without crashing -- check that the main
      // content area has something visible (not just a blank page).
      const content = page.locator('#root')
      await expect(content).toBeVisible({ timeout: 10000 })

      // Page should not show an error state
      const errorBoundary = page.getByText(/something went wrong/i)
      const hasError = await errorBoundary.isVisible({ timeout: 1000 }).catch(() => false)
      expect(hasError).toBe(false)
    })
  }
})

test.describe('Unknown Routes', () => {
  test('unknown route redirects to /chat', async ({ backendPage: page }) => {
    await page.goto('/nonexistent-page')

    // Should redirect to /chat
    await page.waitForURL('**/chat', { timeout: 5000 })
    expect(page.url()).toContain('/chat')
  })

  test('root redirects to /chat', async ({ backendPage: page }) => {
    // Going to / should end up at /chat via the catch-all redirect
    await page.goto('/')
    await page.waitForURL('**/chat', { timeout: 5000 })
    expect(page.url()).toContain('/chat')
  })
})

test.describe('Lazy Loading', () => {
  for (const route of APPLET_ROUTES.filter(r => r.path !== '/chat')) {
    test(`${route.name} lazy-loads without error`, async ({ backendPage: page }) => {
      // Start from chat (eagerly loaded), then navigate to a lazy route
      await page.goto('/chat')
      await expect(page.locator('#root')).toBeVisible({ timeout: 5000 })

      await page.goto(route.path)

      // Should see the lazy fallback spinner or the loaded applet, never an error
      const root = page.locator('#root')
      await expect(root).toBeVisible({ timeout: 10000 })

      // Wait for lazy load to complete (spinner disappears, real content appears)
      // We just need to confirm no crash -- the spinner is fine
      const errorBoundary = page.getByText(/something went wrong|chunk.*failed/i)
      const hasError = await errorBoundary.isVisible({ timeout: 2000 }).catch(() => false)
      expect(hasError).toBe(false)
    })
  }
})

test.describe('State Preservation', () => {
  test('chat conversation survives navigation to another applet and back', async ({ conversationPage: page }) => {
    // conversationPage fixture has already sent a message and received a response

    // Navigate away to notebook
    await page.goto('/notebook')
    await expect(page.locator('#root')).toBeVisible({ timeout: 10000 })

    // Navigate back to chat
    await page.goto('/chat')

    // The conversation sidebar should still show the conversation
    const sidebar = page.locator('aside').first()
    await expect(sidebar).toBeVisible({ timeout: 10000 })

    // Should have at least one conversation entry
    const conversationButtons = sidebar.locator('button').filter({ hasText: /./i })
    const count = await conversationButtons.count()
    expect(count).toBeGreaterThanOrEqual(1)
  })
})
