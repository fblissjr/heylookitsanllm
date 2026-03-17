import { test, expect } from './fixtures'

/**
 * E2E tests for the Model Comparison applet.
 * Tests side-by-side prompt entry and results display.
 */

test.describe('Model Comparison Applet', () => {
  test('renders without error', async ({ backendPage: page }) => {
    await page.goto('/compare')

    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    const error = page.getByText(/something went wrong/i)
    const hasError = await error.isVisible({ timeout: 1000 }).catch(() => false)
    expect(hasError).toBe(false)
  })

  test('shows comparison prompt area', async ({ backendPage: page }) => {
    await page.goto('/compare')

    // The comparison view should have at least one prompt input area
    const input = page.locator('textarea').first()
    const heading = page.getByText(/compar|side.*by.*side|prompt/i).first()

    const hasInput = await input.isVisible({ timeout: 5000 }).catch(() => false)
    const hasHeading = await heading.isVisible({ timeout: 1000 }).catch(() => false)

    expect(hasInput || hasHeading).toBe(true)
  })

  test('can enter a prompt', async ({ backendPage: page }) => {
    await page.goto('/compare')

    const input = page.locator('textarea').first()
    if (await input.isVisible({ timeout: 5000 }).catch(() => false)) {
      await input.fill('Test comparison prompt')
      await expect(input).toHaveValue('Test comparison prompt')
    }
  })
})
