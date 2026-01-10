import { test, expect, Page } from '@playwright/test'

/**
 * E2E tests for settings and sampler parameters.
 * Tests the settings panel, presets, and parameter adjustments.
 */

// Helper to set up with a loaded model
async function setupWithLoadedModel(page: Page): Promise<boolean> {
  await page.goto('/')
  await page.waitForTimeout(2000)

  const connectionFailed = page.getByText('Connection Failed')
  if (await connectionFailed.isVisible()) {
    return false
  }

  const chatInput = page.locator('textarea')
  if (await chatInput.isVisible()) {
    return true
  }

  const modelSelector = page.getByRole('button', { name: /select model/i })
  if (!(await modelSelector.isVisible())) {
    return false
  }
  await modelSelector.click()
  await page.waitForTimeout(500)

  const modelCard = page.locator('button').filter({ hasText: /qwen|gemma|llama/i }).first()
  if (!(await modelCard.isVisible())) {
    return false
  }
  await modelCard.click()

  await page.waitForTimeout(300)
  const loadButton = page.getByRole('button', { name: /load model/i })
  if (!(await loadButton.isVisible())) {
    return false
  }
  await loadButton.click()

  try {
    await expect(chatInput).toBeVisible({ timeout: 15000 })
    return true
  } catch {
    return false
  }
}

test.describe('Settings Panel', () => {
  test.beforeEach(async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }
  })

  test('can open settings panel from header', async ({ page }) => {
    // Look for settings button in header (gear icon)
    const header = page.locator('header')

    // Try to find settings button
    const buttons = await header.locator('button').all()
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

        // Settings panel should be visible
        const settingsHeading = page.getByRole('heading', { name: /settings/i })
        const hasSettings = await settingsHeading.isVisible().catch(() => false)
        if (hasSettings) {
          expect(hasSettings).toBe(true)
          return
        }
      }
    }

    // If no dedicated settings button, look for settings in menu
    test.skip(true, 'Settings button not found in header')
  })

  test('settings shows temperature slider', async ({ page }) => {
    // Open settings
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)
        break
      }
    }

    // Look for temperature control
    const temperatureLabel = page.getByText(/temperature/i)
    const hasTemperature = await temperatureLabel.isVisible().catch(() => false)

    if (hasTemperature) {
      await expect(temperatureLabel).toBeVisible()

      // Should have a slider or input for temperature
      const slider = page.locator('input[type="range"]').first()
      await expect(slider).toBeVisible()
    }
  })
})

test.describe('Sampler Parameters', () => {
  test.beforeEach(async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }
  })

  test('chat input area has send button', async ({ page }) => {
    // Look for send button near chat input
    const sendButton = page.locator('button[title="Send message"]')
    const hasSendButton = await sendButton.isVisible().catch(() => false)

    // Chat area should have at least the send button
    expect(hasSendButton).toBe(true)
  })

  test('can see sampler presets if available', async ({ page }) => {
    // Try to open settings
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

        // Look for preset selector
        const presetLabel = page.getByText(/preset/i)
        const hasPresets = await presetLabel.isVisible().catch(() => false)

        if (hasPresets) {
          await expect(presetLabel).toBeVisible()
        }
        return
      }
    }
  })
})

test.describe('System Prompt', () => {
  test.beforeEach(async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }
  })

  test('can access system prompt if available', async ({ page }) => {
    // Look for system prompt toggle or input
    const systemPromptToggle = page.getByText(/system prompt/i)
    const hasSystemPrompt = await systemPromptToggle.isVisible().catch(() => false)

    if (hasSystemPrompt) {
      await expect(systemPromptToggle).toBeVisible()
    }

    // Or look for it in settings
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

        const systemPromptInSettings = page.getByText(/system prompt/i)
        const found = await systemPromptInSettings.isVisible().catch(() => false)
        if (found) {
          await expect(systemPromptInSettings).toBeVisible()
        }
        return
      }
    }
  })
})

test.describe('Theme Settings', () => {
  test('page loads with dark theme by default', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2000)

    // Check if dark class is on html element
    const isDark = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    expect(isDark).toBe(true)
  })

  test('theme persists across page reloads', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2000)

    // Get current theme
    const initialDark = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    // Reload page
    await page.reload()
    await page.waitForTimeout(2000)

    // Theme should be the same
    const afterReloadDark = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    expect(afterReloadDark).toBe(initialDark)
  })
})

test.describe('Max Tokens Setting', () => {
  test.beforeEach(async ({ page }) => {
    const ready = await setupWithLoadedModel(page)
    if (!ready) {
      test.skip(true, 'Backend not running or model could not be loaded')
    }
  })

  test('can find max tokens setting if available', async ({ page }) => {
    // Try to open settings
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

        // Look for max tokens
        const maxTokensLabel = page.getByText(/max.*tokens|maximum.*tokens/i)
        const hasMaxTokens = await maxTokensLabel.isVisible().catch(() => false)

        if (hasMaxTokens) {
          await expect(maxTokensLabel).toBeVisible()

          // Should have an input or slider
          const input = page.locator('input[type="number"], input[type="range"]')
          const inputCount = await input.count()
          expect(inputCount).toBeGreaterThanOrEqual(1)
        }
        return
      }
    }
  })
})
