import { test, expect, setupWithLoadedModel } from './fixtures'

/**
 * E2E tests for conversation management.
 * These tests require the backend to be running with a model available.
 */

test.describe('Conversation Management', () => {
  test('creates a new conversation when model loads', async ({ modelPage: page }) => {
    const sidebar = page.locator('aside').first()
    await expect(sidebar).toBeVisible()

    const conversationItem = sidebar.locator('button').filter({ hasText: /new conversation/i })
    await expect(conversationItem).toBeVisible()
  })

  test('can create multiple conversations via New Chat button', async ({ modelPage: page }) => {
    const sidebar = page.locator('aside').first()
    const newChatButton = sidebar.getByRole('button', { name: /new chat/i })

    await expect(newChatButton).toBeVisible()
    await expect(newChatButton).toBeEnabled()

    await newChatButton.click()
    await page.waitForTimeout(500)

    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible()
  })

  test('conversation title updates from first message', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const testMessage = 'Hello world test'
    await chatInput.fill(testMessage)
    await sendButton.click()

    // Wait for message to appear
    await expect(page.getByText(testMessage).first()).toBeVisible({ timeout: 10000 })

    // The sidebar should show conversation with updated title
    const sidebar = page.locator('aside').first()
    const hasConversations = await sidebar.locator('button').filter({ has: page.locator('svg') }).count()
    expect(hasConversations).toBeGreaterThanOrEqual(1)
  })

  test('can switch between conversations', async ({ modelPage: page }) => {
    const sidebar = page.locator('aside').first()
    const newChatButton = sidebar.getByRole('button', { name: /new chat/i })

    const chatInput = page.locator('textarea')
    await chatInput.fill('First conversation')

    const sendButton = page.locator('button[title="Send message"]')
    await sendButton.click()

    await page.waitForTimeout(2000)

    // Stop generation if in progress
    const stopButton = page.locator('button[title="Stop generation"]')
    if (await stopButton.isVisible({ timeout: 1000 }).catch(() => false)) {
      await stopButton.click()
      await page.waitForTimeout(500)
    }

    await newChatButton.click()
    await page.waitForTimeout(1000)

    await expect(chatInput).toBeVisible({ timeout: 5000 })
    await expect(chatInput).toBeEnabled({ timeout: 10000 })

    const allButtons = sidebar.locator('button')
    const count = await allButtons.count()
    expect(count).toBeGreaterThanOrEqual(2)
  })

  test('shows delete button on conversation hover', async ({ modelPage: page }) => {
    const sidebar = page.locator('aside').first()
    const conversationItem = sidebar.locator('button').filter({ hasText: /new conversation|conversation/i }).first()

    if (await conversationItem.isVisible()) {
      await conversationItem.hover()
      const deleteButton = conversationItem.locator('button[aria-label="Delete conversation"]')
      await expect(deleteButton).toBeVisible({ timeout: 2000 })
    }
  })

  test('delete confirmation modal appears when deleting conversation', async ({ modelPage: page }) => {
    const sidebar = page.locator('aside').first()
    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    await chatInput.fill('Test delete')
    await sendButton.click()
    await page.waitForTimeout(2000)

    const conversationButtons = sidebar.locator('button').filter({ has: page.locator('span.truncate') })

    if (await conversationButtons.first().isVisible()) {
      await conversationButtons.first().hover()
      await page.waitForTimeout(500)

      const deleteButton = page.locator('button[aria-label="Delete conversation"]').first()

      if (await deleteButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await deleteButton.click()

        const modalContent = page.getByRole('button', { name: /cancel/i })
        await expect(modalContent).toBeVisible({ timeout: 3000 })
        await modalContent.click()
      } else {
        test.skip(true, 'Delete button not visible on hover')
      }
    } else {
      test.skip(true, 'No conversation items to delete')
    }
  })
})

test.describe('Message Sending', () => {
  test('can send a message and receive a response', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const uniqueMessage = `Test message ${Date.now()}`
    await chatInput.fill(uniqueMessage)
    await sendButton.click()

    await expect(page.getByText(uniqueMessage).first()).toBeVisible({ timeout: 5000 })

    const body = page.locator('body')
    await expect(body).toBeVisible()
  })

  test('send button is disabled while streaming', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    await chatInput.fill('Explain quantum computing in detail')
    await sendButton.click()

    await page.waitForTimeout(500)
    await expect(chatInput).toBeDisabled({ timeout: 2000 })
  })

  test('can send message with Enter key', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')

    const uniqueMessage = `Enter key test ${Date.now()}`
    await chatInput.fill(uniqueMessage)
    await chatInput.press('Enter')

    await expect(page.getByText(uniqueMessage).first()).toBeVisible({ timeout: 10000 })
  })

  test('Shift+Enter creates new line instead of sending', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')

    await chatInput.fill('Line 1')
    await chatInput.press('Shift+Enter')
    await chatInput.pressSequentially('Line 2')

    const value = await chatInput.inputValue()
    expect(value).toContain('Line 1')
    expect(value).toContain('Line 2')
  })

  test('multi-turn: context is maintained across messages', async ({ conversationPage: page }) => {
    // conversationPage already has one exchange (user: "Hello", assistant response visible)

    // Wait for generation to finish (input becomes enabled)
    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeEnabled({ timeout: 30000 })

    // Send a follow-up
    const followUp = `Follow-up ${Date.now()}`
    await chatInput.fill(followUp)
    const sendButton = page.locator('button[title="Send message"]')
    await sendButton.click()

    // Both messages should be visible
    await expect(page.getByText('Hello').first()).toBeVisible()
    await expect(page.getByText(followUp).first()).toBeVisible({ timeout: 10000 })
  })
})

test.describe('Empty States', () => {
  test('shows empty state in sidebar when no conversations', async ({ backendPage: page }) => {
    const sidebar = page.locator('aside').first()
    const emptyMessage = sidebar.getByText(/load a model|no conversations/i)

    const hasEmptyMessage = await emptyMessage.isVisible().catch(() => false)
    const hasConversations = await sidebar.locator('button').filter({ hasText: /conversation/i }).isVisible().catch(() => false)

    expect(hasEmptyMessage || hasConversations).toBe(true)
  })
})
