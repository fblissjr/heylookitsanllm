import { test, expect } from '../fixtures'

/**
 * Mobile persistence E2E tests.
 * Tests data saving on visibility change (simulating iOS tab backgrounding),
 * reconnection banner behavior, and rapid-send-then-background scenarios.
 */

test.describe('Tab Background Persistence', () => {
  test('data is saved when tab is backgrounded', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const testMessage = `Mobile persist ${Date.now()}`
    await chatInput.fill(testMessage)
    await sendButton.click()

    // Wait for message to appear
    await expect(page.getByText(testMessage).first()).toBeVisible({ timeout: 10000 })

    // Simulate tab backgrounding by dispatching visibilitychange
    // This triggers flushPendingSave() in chatStore
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { value: true, writable: true })
      document.dispatchEvent(new Event('visibilitychange'))
    })

    // Small wait for the flush to complete
    await page.waitForTimeout(500)

    // Verify data was written to IndexedDB
    const hasConversation = await page.evaluate(async (msg: string) => {
      return new Promise<boolean>((resolve) => {
        const request = indexedDB.open('heylook')
        request.onsuccess = () => {
          const db = request.result
          try {
            const tx = db.transaction('conversations', 'readonly')
            const store = tx.objectStore('conversations')
            const getAll = store.getAll()
            getAll.onsuccess = () => {
              const conversations = getAll.result
              const found = conversations.some((conv: { messages?: { content?: string }[] }) =>
                conv.messages?.some((m: { content?: string }) => m.content?.includes(msg))
              )
              db.close()
              resolve(found)
            }
            getAll.onerror = () => {
              db.close()
              resolve(false)
            }
          } catch {
            db.close()
            resolve(false)
          }
        }
        request.onerror = () => resolve(false)
      })
    }, testMessage)

    expect(hasConversation).toBe(true)
  })

  test('rapid send then background does not lose data', async ({ modelPage: page }) => {
    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const testMessage = `Rapid ${Date.now()}`
    await chatInput.fill(testMessage)
    await sendButton.click()

    // Immediately simulate backgrounding (don't wait for response)
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { value: true, writable: true })
      document.dispatchEvent(new Event('visibilitychange'))
    })

    await page.waitForTimeout(500)

    // Bring back to foreground
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { value: false, writable: true })
      document.dispatchEvent(new Event('visibilitychange'))
    })

    await page.waitForTimeout(1000)

    // The user message should still be in the UI
    await expect(page.getByText(testMessage).first()).toBeVisible({ timeout: 5000 })
  })
})

test.describe('Reconnection Banner', () => {
  test('reconnection module initializes without error', async ({ backendPage: page }) => {
    // The reconnection module is initialized in App.tsx on mount.
    // After navigating, it should not show a reconnection banner
    // (since the backend is reachable).
    const banner = page.getByText('Reconnecting to server...')
    const isBannerVisible = await banner.isVisible({ timeout: 2000 }).catch(() => false)

    // Backend is available, so banner should NOT be visible
    expect(isBannerVisible).toBe(false)
  })
})
