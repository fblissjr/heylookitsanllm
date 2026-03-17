import { test, expect } from './fixtures'

/**
 * E2E tests for the chat interface.
 *
 * Note: These tests are designed to work even when the backend is not running.
 * The app will show a "Connection Failed" state which we handle gracefully.
 */

test.describe('Chat Page', () => {
  test('page loads successfully', async ({ page }) => {
    await page.goto('/')

    // The page should load - either showing loading, connection error, or the main UI
    // We check for any of these states to confirm the app is working
    const body = page.locator('body')
    await expect(body).toBeVisible()

    // The page title should be set (from index.html)
    await expect(page).toHaveTitle(/heylook/i)
  })

  test('shows loading or connection state on startup', async ({ page }) => {
    await page.goto('/')

    // App should show one of these states:
    // 1. "Connecting to server..." (loading)
    // 2. "Connection Failed" (backend not running)
    // 3. The main UI with "No Model Loaded" (backend connected)
    // 4. The chat interface with textarea (model loaded)
    const loadingText = page.getByText('Connecting to server...')
    const connectionFailedText = page.getByText('Connection Failed')
    const noModelLoaded = page.getByRole('heading', { name: /no model loaded/i })
    const chatInput = page.locator('textarea')

    // Wait for any of these states
    await expect(
      loadingText.or(connectionFailedText).or(noModelLoaded).or(chatInput)
    ).toBeVisible({ timeout: 10000 })
  })

  test('can click retry button when connection fails', async ({ page }) => {
    await page.goto('/')

    // Wait for connection attempt to complete
    await page.waitForTimeout(3000)

    const connectionFailedText = page.getByText('Connection Failed')

    // Only run this test if connection actually failed (backend not running)
    if (await connectionFailedText.isVisible()) {
      const retryButton = page.getByRole('button', { name: /retry/i })
      await expect(retryButton).toBeVisible()

      // Click retry - should trigger a page reload
      await retryButton.click()

      // Page should reload and show loading or error again
      const loadingOrError = page
        .getByText('Connecting to server...')
        .or(page.getByText('Connection Failed'))
      await expect(loadingOrError).toBeVisible({ timeout: 10000 })
    }
  })
})

test.describe('Chat Interface (requires backend)', () => {
  test('model selector is visible in header', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await expect(modelSelector).toBeVisible()
  })

  test('sidebar toggle works', async ({ backendPage: page }) => {
    const sidebarToggle = page.getByRole('button', { name: /toggle sidebar/i })
    await expect(sidebarToggle).toBeVisible()

    await sidebarToggle.click()
    await page.waitForTimeout(300)

    await sidebarToggle.click()
  })

  test('shows empty state when no model loaded', async ({ backendPage: page }) => {
    const emptyState = page.getByRole('heading', { name: /no model loaded/i })
    await expect(emptyState).toBeVisible({ timeout: 10000 })
  })
})

test.describe('Chat Input (requires model loaded)', () => {
  test('can type in chat input', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible({ timeout: 5000 })

    await chatInput.fill('Hello, this is a test message')
    await expect(chatInput).toHaveValue('Hello, this is a test message')
  })

  test('send button state changes with input', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible({ timeout: 5000 })

    // Clear input - send button should be disabled
    await chatInput.fill('')
    const sendButton = page.locator('button[title="Send message"]')
    await expect(sendButton).toBeDisabled()

    // Type text - send button should enable
    await chatInput.fill('Test message')
    await expect(sendButton).toBeEnabled()
  })
})
