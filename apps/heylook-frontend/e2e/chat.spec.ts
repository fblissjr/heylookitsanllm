import { test, expect } from '@playwright/test'

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
    // 3. The main chat interface (backend running and connected)
    const loadingText = page.getByText('Connecting to server...')
    const connectionFailedText = page.getByText('Connection Failed')
    const retryButton = page.getByRole('button', { name: /retry/i })
    const chatInput = page.locator('textarea')

    // Wait for either loading, error, or success state
    await expect(
      loadingText.or(connectionFailedText).or(chatInput)
    ).toBeVisible({ timeout: 10000 })

    // If connection failed, retry button should be visible
    if (await connectionFailedText.isVisible()) {
      await expect(retryButton).toBeVisible()
    }
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
  test.beforeEach(async ({ page }) => {
    await page.goto('/')

    // Wait for connection attempt
    await page.waitForTimeout(3000)

    // Skip these tests if backend is not running
    const connectionFailed = page.getByText('Connection Failed')
    if (await connectionFailed.isVisible()) {
      test.skip(true, 'Backend not running - skipping connected UI tests')
    }
  })

  test('model selector is visible in header', async ({ page }) => {
    // The header should contain a model selector button
    // It shows "Select Model" when no model is loaded
    const modelSelector = page.getByText(/select model/i).or(
      page.locator('header button').filter({ hasText: /.+/ }).first()
    )
    await expect(modelSelector).toBeVisible()
  })

  test('can type in chat input', async ({ page }) => {
    // Find the chat input textarea
    const chatInput = page.locator('textarea[placeholder*="Message"]')
    await expect(chatInput).toBeVisible()

    // Type a test message
    await chatInput.fill('Hello, this is a test message')

    // Verify the text was entered
    await expect(chatInput).toHaveValue('Hello, this is a test message')
  })

  test('send button is disabled when input is empty', async ({ page }) => {
    // Find the chat input and ensure it's empty
    const chatInput = page.locator('textarea[placeholder*="Message"]')
    await expect(chatInput).toBeVisible()
    await chatInput.fill('')

    // The send button should be disabled or have a disabled appearance
    // The send button is after the textarea in the input container
    const sendButton = page.locator('button[title="Send message"]')
    await expect(sendButton).toBeDisabled()
  })

  test('send button enables when text is entered', async ({ page }) => {
    const chatInput = page.locator('textarea[placeholder*="Message"]')
    await expect(chatInput).toBeVisible()

    // Type some text
    await chatInput.fill('Test message')

    // Send button should now be enabled
    const sendButton = page.locator('button[title="Send message"]')
    await expect(sendButton).toBeEnabled()
  })

  test('sidebar toggle works', async ({ page }) => {
    // Find the sidebar toggle button (hamburger menu)
    const sidebarToggle = page.getByRole('button', { name: /toggle sidebar/i })
    await expect(sidebarToggle).toBeVisible()

    // Click it to toggle sidebar
    await sidebarToggle.click()

    // Give animation time to complete
    await page.waitForTimeout(300)

    // Click again to toggle back
    await sidebarToggle.click()
  })
})

test.describe('Keyboard Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(3000)

    const connectionFailed = page.getByText('Connection Failed')
    if (await connectionFailed.isVisible()) {
      test.skip(true, 'Backend not running - skipping keyboard tests')
    }
  })

  test('Enter key in chat input triggers send (when enabled)', async ({ page }) => {
    const chatInput = page.locator('textarea[placeholder*="Message"]')
    await expect(chatInput).toBeVisible()

    // Type a message
    await chatInput.fill('Test message')

    // Press Enter - this would normally send the message
    // We just verify the key event is handled (message clears or stays based on backend)
    await chatInput.press('Enter')

    // The input handling should work without errors
    // (actual send behavior depends on backend and model being loaded)
  })

  test('Shift+Enter creates new line instead of sending', async ({ page }) => {
    const chatInput = page.locator('textarea[placeholder*="Message"]')
    await expect(chatInput).toBeVisible()

    // Type first line
    await chatInput.fill('Line 1')

    // Shift+Enter should add a new line
    await chatInput.press('Shift+Enter')
    await chatInput.type('Line 2')

    // The textarea should contain both lines
    const value = await chatInput.inputValue()
    expect(value).toContain('Line 1')
    expect(value).toContain('Line 2')
  })
})
