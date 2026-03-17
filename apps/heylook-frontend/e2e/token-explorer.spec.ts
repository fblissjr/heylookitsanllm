import { test, expect } from './fixtures'

/**
 * E2E tests for the Token Explorer applet.
 * Tests text input, tokenization display, and run history.
 */

test.describe('Token Explorer Applet', () => {
  test('renders without error', async ({ backendPage: page }) => {
    await page.goto('/explore')

    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    const error = page.getByText(/something went wrong/i)
    const hasError = await error.isVisible({ timeout: 1000 }).catch(() => false)
    expect(hasError).toBe(false)
  })

  test('shows explorer form with text input', async ({ backendPage: page }) => {
    await page.goto('/explore')

    // The explorer form has a textarea for entering text to tokenize
    const input = page.locator('textarea').first()
    await expect(input).toBeVisible({ timeout: 10000 })
  })

  test('can enter text in the explorer form', async ({ backendPage: page }) => {
    await page.goto('/explore')

    const input = page.locator('textarea').first()
    await expect(input).toBeVisible({ timeout: 10000 })

    await input.fill('Hello world, this is a test')
    await expect(input).toHaveValue('Hello world, this is a test')
  })

  test('has a run/explore button', async ({ backendPage: page }) => {
    await page.goto('/explore')

    // Look for a button to trigger tokenization
    const runButton = page.getByRole('button', { name: /run|explore|tokenize|generate/i }).first()
    const hasButton = await runButton.isVisible({ timeout: 5000 }).catch(() => false)

    // Should have some way to trigger the exploration
    expect(hasButton).toBe(true)
  })
})
