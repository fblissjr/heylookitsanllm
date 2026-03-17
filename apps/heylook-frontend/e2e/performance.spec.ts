import { test, expect } from './fixtures'

/**
 * E2E tests for the Performance applet.
 * Tests metrics display and model performance stats.
 */

test.describe('Performance Applet', () => {
  test('renders without error', async ({ backendPage: page }) => {
    await page.goto('/perf')

    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    const error = page.getByText(/something went wrong/i)
    const hasError = await error.isVisible({ timeout: 1000 }).catch(() => false)
    expect(hasError).toBe(false)
  })

  test('shows performance metrics or empty state', async ({ backendPage: page }) => {
    await page.goto('/perf')

    // Performance view should show metrics, charts, or an empty state message
    const metricsContent = page.getByText(/performance|metrics|latency|throughput|tok.*s|no.*data|no.*model/i).first()
    await expect(metricsContent).toBeVisible({ timeout: 10000 })
  })
})
