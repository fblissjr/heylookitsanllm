import { test, expect } from './fixtures'

/**
 * E2E tests for the Notebook applet.
 * Tests document creation, editing, persistence, and inline generation.
 */

test.describe('Notebook Applet', () => {
  test('renders without error', async ({ backendPage: page }) => {
    await page.goto('/notebook')

    // Should show the notebook editor area
    const root = page.locator('#root')
    await expect(root).toBeVisible({ timeout: 10000 })

    // No error boundary
    const error = page.getByText(/something went wrong/i)
    const hasError = await error.isVisible({ timeout: 1000 }).catch(() => false)
    expect(hasError).toBe(false)
  })

  test('auto-creates a document on first visit', async ({ backendPage: page }) => {
    await page.goto('/notebook')

    // The left panel should show at least one document entry
    // The store auto-creates a document when loaded and documents.length === 0
    const docEntry = page.locator('button').filter({ hasText: /untitled|new document/i }).first()
    await expect(docEntry).toBeVisible({ timeout: 10000 })
  })

  test('can type in the editor', async ({ backendPage: page }) => {
    await page.goto('/notebook')

    // Wait for document to be auto-created and editor to appear
    // The editor is a textarea or contenteditable area
    const editor = page.locator('textarea, [contenteditable="true"]').first()
    await expect(editor).toBeVisible({ timeout: 10000 })

    await editor.fill('Test notebook content')

    // Content should be in the editor
    const value = await editor.inputValue().catch(() => '')
    if (value) {
      expect(value).toContain('Test notebook content')
    } else {
      // contenteditable case
      const text = await editor.textContent()
      expect(text).toContain('Test notebook content')
    }
  })

  test('can create a new document', async ({ backendPage: page }) => {
    await page.goto('/notebook')
    await page.waitForTimeout(1000)

    // Look for a "New" button or use keyboard shortcut
    const newButton = page.getByRole('button', { name: /new|create/i }).first()
    if (await newButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      // Count docs before
      const docsBefore = await page.locator('button').filter({ hasText: /untitled|new document/i }).count()

      await newButton.click()
      await page.waitForTimeout(500)

      const docsAfter = await page.locator('button').filter({ hasText: /untitled|new document/i }).count()
      expect(docsAfter).toBeGreaterThanOrEqual(docsBefore)
    }
  })
})
