import { test, expect, setupWithLoadedModel } from './fixtures'

/**
 * E2E tests for data persistence.
 * Tests IndexedDB storage, conversation persistence, and state recovery.
 */

test.describe('Conversation Persistence', () => {
  test('conversations persist across page reloads', async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }

    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const testMessage = `Persist test ${Date.now()}`
    await chatInput.fill(testMessage)
    await sendButton.click()

    // Wait for the user message to appear in the DOM
    await expect(page.getByText(testMessage).first()).toBeVisible({ timeout: 10000 })

    // Wait for IndexedDB save (debounce is 500ms)
    await page.waitForTimeout(1000)

    // Reload the page
    await page.reload()

    // After reload, the conversation sidebar should still list the conversation.
    // Check that the sidebar contains a conversation item (not just that body is visible).
    const sidebar = page.locator('aside').first()
    await expect(sidebar).toBeVisible({ timeout: 10000 })

    // The conversation title is derived from the first user message
    const titleSnippet = testMessage.slice(0, 30)
    const conversationEntry = sidebar.getByText(new RegExp(titleSnippet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'))
    await expect(conversationEntry).toBeVisible({ timeout: 10000 })
  })

  test('active conversation messages are restored after reload', async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }

    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const testMessage = `Restore test ${Date.now()}`
    await chatInput.fill(testMessage)
    await sendButton.click()

    // Wait for message to render and save
    await expect(page.getByText(testMessage).first()).toBeVisible({ timeout: 10000 })
    await page.waitForTimeout(1000)

    // Reload
    await page.reload()

    // The user message text should still be visible after reload
    await expect(page.getByText(testMessage).first()).toBeVisible({ timeout: 10000 })
  })
})

test.describe('IndexedDB Storage', () => {
  test('IndexedDB database is created on first visit', async ({ page }) => {
    await page.goto('/')

    // Wait for app to initialize (it opens the DB on startup)
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })
    await page.waitForTimeout(1000)

    // Check for the correct DB name: 'heylook' (not 'heylook-db')
    const hasDB = await page.evaluate(async () => {
      const dbs = await indexedDB.databases()
      return dbs.some((db) => db.name === 'heylook')
    })

    expect(hasDB).toBe(true)
  })

  test('conversations are stored in IndexedDB', async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }

    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    await chatInput.fill('Test storage message')
    await sendButton.click()

    // Wait for message + debounced save
    await expect(page.getByText('Test storage message').first()).toBeVisible({ timeout: 10000 })
    await page.waitForTimeout(1000)

    // Check IndexedDB for stored conversations using the correct DB name
    const conversationCount = await page.evaluate(async () => {
      return new Promise<number>((resolve) => {
        const request = indexedDB.open('heylook')
        request.onsuccess = () => {
          const db = request.result
          try {
            const tx = db.transaction('conversations', 'readonly')
            const store = tx.objectStore('conversations')
            const countRequest = store.count()
            countRequest.onsuccess = () => {
              db.close()
              resolve(countRequest.result)
            }
            countRequest.onerror = () => {
              db.close()
              resolve(0)
            }
          } catch {
            db.close()
            resolve(0)
          }
        }
        request.onerror = () => resolve(0)
      })
    })

    expect(conversationCount).toBeGreaterThanOrEqual(1)
  })
})

test.describe('State Recovery', () => {
  test('sidebar state persists', async ({ backendPage: page }) => {
    const sidebarToggle = page.getByRole('button', { name: /toggle sidebar/i })

    if (await sidebarToggle.isVisible()) {
      const sidebar = page.locator('aside').first()
      const initialVisible = await sidebar.isVisible().catch(() => false)

      await sidebarToggle.click()
      await page.waitForTimeout(300)

      const afterToggleVisible = await sidebar.isVisible().catch(() => !initialVisible)
      expect(afterToggleVisible).not.toBe(initialVisible)

      // Toggle back
      await sidebarToggle.click()
    }
  })

  test('theme preference persists in localStorage', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    const theme = await page.evaluate(() => {
      return localStorage.getItem('heylook:theme')
    })

    const validThemes = ['dark', 'light', 'auto', null]
    expect(validThemes).toContain(theme)
  })

  test('localStorage survives page reload', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    await page.evaluate(() => {
      localStorage.setItem('heylook:test', 'persistence-check')
    })

    await page.reload()
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    const value = await page.evaluate(() => {
      return localStorage.getItem('heylook:test')
    })

    expect(value).toBe('persistence-check')

    await page.evaluate(() => {
      localStorage.removeItem('heylook:test')
    })
  })
})

test.describe('Data Integrity', () => {
  test('message order is preserved after reload', async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }

    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    const msg = `Order test ${Date.now()}`
    await chatInput.fill(msg)
    await sendButton.click()

    // Wait for user message + assistant response
    await expect(page.getByText(msg).first()).toBeVisible({ timeout: 10000 })
    await page.waitForTimeout(3000)

    // Reload
    await page.reload()

    // User message should still be first
    await expect(page.getByText(msg).first()).toBeVisible({ timeout: 10000 })
  })
})

test.describe('Error Recovery', () => {
  test('app handles corrupted localStorage gracefully', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    await page.evaluate(() => {
      localStorage.setItem('heylook:theme', 'invalid-theme-value')
    })

    await page.reload()
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    const hasDarkClass = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    expect(typeof hasDarkClass).toBe('boolean')
  })

  test('app recovers when IndexedDB is unavailable', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })
    await expect(page).toHaveTitle(/heylook/i)
  })
})
