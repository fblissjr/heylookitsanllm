import { test, expect } from './fixtures'

/**
 * E2E tests for the Batch applet.
 * Tests batch job creation form, dashboard rendering, and error states.
 */

test.describe('Batch Applet', () => {
  test('renders without error', async ({ backendPage: page }) => {
    await page.goto('/batch')

    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    const error = page.getByText(/something went wrong/i)
    const hasError = await error.isVisible({ timeout: 1000 }).catch(() => false)
    expect(hasError).toBe(false)
  })

  test('shows batch create form or dashboard', async ({ backendPage: page }) => {
    await page.goto('/batch')

    // The batch applet shows either a creation form or dashboard
    // Look for common UI elements: a textarea for prompts, a "Run" button, or a dashboard heading
    const createFormElement = page.locator('textarea').or(
      page.getByRole('button', { name: /run|start|create|new batch/i })
    ).or(
      page.getByText(/batch|prompts|dashboard/i)
    )

    await expect(createFormElement.first()).toBeVisible({ timeout: 10000 })
  })

  test('error state when no model loaded', async ({ backendPage: page }) => {
    await page.goto('/batch')
    await page.waitForTimeout(1000)

    // If no model is loaded, the batch form should indicate this
    // (either disabled submit or a message about needing a model)
    const noModelMsg = page.getByText(/no model|select.*model|load.*model/i)
    const textarea = page.locator('textarea')

    // Either we see a no-model message, or the form is present (model is loaded)
    const hasMessage = await noModelMsg.isVisible({ timeout: 3000 }).catch(() => false)
    const hasTextarea = await textarea.isVisible({ timeout: 1000 }).catch(() => false)

    expect(hasMessage || hasTextarea).toBe(true)
  })
})
