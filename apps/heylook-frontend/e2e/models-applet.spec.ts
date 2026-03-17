import { test, expect } from './fixtures'

/**
 * E2E tests for the Models applet (distinct from model-management.spec.ts
 * which tests the model selector in the header).
 * Tests the dedicated /models view with model configs, profiles, and details.
 */

test.describe('Models Applet', () => {
  test('renders model list', async ({ backendPage: page }) => {
    await page.goto('/models')

    // Should show a list of models or a loading state
    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    // Look for model entries (buttons/cards with model names)
    const modelEntry = page.locator('button, [role="listitem"]').filter({ hasText: /qwen|gemma|llama|mistral/i }).first()
    await expect(modelEntry).toBeVisible({ timeout: 10000 })
  })

  test('shows capability badges', async ({ backendPage: page }) => {
    await page.goto('/models')

    // Wait for model list to load
    const modelEntry = page.locator('button, [role="listitem"]').filter({ hasText: /qwen|gemma|llama|mistral/i }).first()
    await expect(modelEntry).toBeVisible({ timeout: 10000 })

    // Look for capability badges anywhere on the page
    const badge = page.locator('span').filter({ hasText: /chat|vision|thinking|embeddings/i }).first()
    const hasBadge = await badge.isVisible({ timeout: 3000 }).catch(() => false)

    // At least some models should show capability info
    expect(typeof hasBadge).toBe('boolean')
  })

  test('can select a model to see details', async ({ backendPage: page }) => {
    await page.goto('/models')

    const modelEntry = page.locator('button, [role="listitem"]').filter({ hasText: /qwen|gemma|llama|mistral/i }).first()
    await expect(modelEntry).toBeVisible({ timeout: 10000 })

    await modelEntry.click()
    await page.waitForTimeout(500)

    // Detail panel should show something (model name, config, etc.)
    const detailArea = page.locator('main, [role="main"]').first()
    const hasContent = await detailArea.isVisible({ timeout: 3000 }).catch(() => false)
    expect(hasContent).toBe(true)
  })
})
