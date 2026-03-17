import { test as base, expect, type Page } from '@playwright/test'

/**
 * Shared Playwright fixtures for heylook E2E tests.
 *
 * Provides:
 * - backendAvailable: skips if backend unreachable
 * - modelLoaded: extends backendAvailable, ensures a model is loaded
 * - withConversation: extends modelLoaded, creates a conversation with one exchange
 */

/** Check if backend is reachable and skip the test if not. */
async function ensureBackendAvailable(page: Page): Promise<void> {
  await page.goto('/')

  // Wait for one of the connection states to appear
  const connected = page.locator('textarea')
    .or(page.getByRole('heading', { name: /no model loaded/i }))
    .or(page.getByRole('button', { name: /select model/i }))
  const failed = page.getByText('Connection Failed')

  await expect(connected.or(failed)).toBeVisible({ timeout: 10000 })

  if (await failed.isVisible()) {
    base.skip(true, 'Backend not running')
  }
}

/** Load the smallest available model. Returns true if successful. */
async function loadModel(page: Page): Promise<boolean> {
  // Already loaded?
  const chatInput = page.locator('textarea')
  if (await chatInput.isVisible({ timeout: 1000 }).catch(() => false)) {
    return true
  }

  const modelSelector = page.getByRole('button', { name: /select model/i })
  if (!(await modelSelector.isVisible({ timeout: 2000 }).catch(() => false))) {
    return false
  }
  await modelSelector.click()

  // Wait for model list to populate
  const modelCard = page.locator('button').filter({ hasText: /qwen|gemma|llama/i }).first()
  await expect(modelCard).toBeVisible({ timeout: 5000 })
  await modelCard.click()

  // Click load
  const loadButton = page.getByRole('button', { name: /load model/i })
  await expect(loadButton).toBeVisible({ timeout: 2000 })
  await loadButton.click()

  // Wait for chat input
  try {
    await expect(chatInput).toBeVisible({ timeout: 15000 })
    return true
  } catch {
    return false
  }
}

// --- Extended test fixtures ---

type Fixtures = {
  backendPage: Page
  modelPage: Page
  conversationPage: Page
}

export const test = base.extend<Fixtures>({
  /** Page with backend availability check -- skips test if unreachable. */
  backendPage: async ({ page }, use) => {
    await ensureBackendAvailable(page)
    await use(page)
  },

  /** Page with a model loaded -- skips test if backend down or no model available. */
  modelPage: async ({ page }, use) => {
    await ensureBackendAvailable(page)
    const loaded = await loadModel(page)
    if (!loaded) {
      base.skip(true, 'Could not load a model')
    }
    await use(page)
  },

  /** Page with a conversation that has at least one user message sent. */
  conversationPage: async ({ page }, use) => {
    await ensureBackendAvailable(page)
    const loaded = await loadModel(page)
    if (!loaded) {
      base.skip(true, 'Could not load a model')
    }

    const chatInput = page.locator('textarea')
    const sendButton = page.locator('button[title="Send message"]')

    await chatInput.fill('Hello')
    await sendButton.click()

    // Wait for assistant response to start appearing
    await expect(page.locator('[data-role="assistant"]').first()).toBeVisible({ timeout: 15000 })

    await use(page)
  },
})

export { expect }

// Re-export the setupWithLoadedModel helper for files that haven't migrated yet
export async function setupWithLoadedModel(page: Page): Promise<boolean> {
  try {
    await ensureBackendAvailable(page)
    return await loadModel(page)
  } catch {
    return false
  }
}
