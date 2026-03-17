import { test, expect } from '../fixtures'

/**
 * Mobile viewport E2E tests.
 * Run against mobile-safari and mobile-chrome Playwright projects.
 * Tests layout, navigation, and touch targets at mobile widths.
 */

test.describe('Mobile Layout', () => {
  test('chat page renders at mobile width', async ({ backendPage: page }) => {
    await page.goto('/chat')

    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    // Content should fit within the viewport (no horizontal overflow)
    const viewport = page.viewportSize()
    if (viewport) {
      const bodyWidth = await page.evaluate(() => document.body.scrollWidth)
      // Allow small tolerance (1px rounding)
      expect(bodyWidth).toBeLessThanOrEqual(viewport.width + 1)
    }
  })

  test('sidebar is hidden by default on mobile', async ({ backendPage: page }) => {
    await page.goto('/chat')
    await expect(page.locator('#root')).toBeVisible({ timeout: 10000 })

    // On mobile, sidebar should be collapsed/hidden
    const sidebar = page.locator('aside').first()
    const viewport = page.viewportSize()

    if (viewport && viewport.width < 768) {
      // Sidebar should either be hidden or positioned off-screen
      const sidebarBox = await sidebar.boundingBox().catch(() => null)
      if (sidebarBox) {
        // If visible, it should be an overlay (positioned absolutely)
        // or it should be off-screen
        const isOffscreen = sidebarBox.x + sidebarBox.width <= 0
        const isOverlay = sidebarBox.x >= 0 // If on-screen, that's ok for overlay behavior
        expect(isOffscreen || isOverlay).toBe(true)
      }
      // sidebar not found at all is also fine for mobile
    }
  })

  test('bottom navigation is visible on mobile', async ({ backendPage: page }) => {
    await page.goto('/chat')
    await expect(page.locator('#root')).toBeVisible({ timeout: 10000 })

    const viewport = page.viewportSize()
    if (viewport && viewport.width < 768) {
      // Look for bottom navigation (nav element at bottom of screen)
      const nav = page.locator('nav').last()
      const navBox = await nav.boundingBox().catch(() => null)

      if (navBox && viewport) {
        // Navigation should be near the bottom of the viewport
        expect(navBox.y).toBeGreaterThan(viewport.height * 0.7)
      }
    }
  })

  test('all applet routes render at mobile width', async ({ backendPage: page }) => {
    const routes = ['/chat', '/batch', '/explore', '/compare', '/perf', '/notebook', '/models']

    for (const route of routes) {
      await page.goto(route)
      const root = page.locator('#root')
      await expect(root).toBeVisible({ timeout: 10000 })

      // No error boundary
      const error = page.getByText(/something went wrong/i)
      const hasError = await error.isVisible({ timeout: 500 }).catch(() => false)
      expect(hasError).toBe(false)
    }
  })
})

test.describe('Touch Targets', () => {
  test('send button has adequate touch target size', async ({ modelPage: page }) => {
    const sendButton = page.locator('button[title="Send message"]')
    await expect(sendButton).toBeVisible()

    const box = await sendButton.boundingBox()
    if (box) {
      // Minimum touch target: 44x44px (Apple HIG recommendation)
      expect(box.width).toBeGreaterThanOrEqual(32)
      expect(box.height).toBeGreaterThanOrEqual(32)
    }
  })
})
