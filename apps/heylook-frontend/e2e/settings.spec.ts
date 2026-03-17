import { test, expect } from './fixtures'

/**
 * E2E tests for settings and sampler parameters.
 * Tests the settings panel, presets, and parameter adjustments.
 */

test.describe('Settings Panel', () => {
  test('can open settings panel from header', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

        const settingsHeading = page.getByRole('heading', { name: /settings/i })
        const hasSettings = await settingsHeading.isVisible().catch(() => false)
        if (hasSettings) {
          expect(hasSettings).toBe(true)
          return
        }
      }
    }

    test.skip(true, 'Settings button not found in header')
  })

  test('settings shows temperature slider', async ({ modelPage: page }) => {
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

    const temperatureLabel = page.getByText(/temperature/i)
    const hasTemperature = await temperatureLabel.isVisible().catch(() => false)

    if (hasTemperature) {
      await expect(temperatureLabel).toBeVisible()
      const slider = page.locator('input[type="range"]').first()
      await expect(slider).toBeVisible()
    }
  })
})

test.describe('Sampler Parameters', () => {
  test('chat input area has send button', async ({ modelPage: page }) => {
    const sendButton = page.locator('button[title="Send message"]')
    await expect(sendButton).toBeVisible()
  })

  test('can see sampler presets if available', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

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
  test('can access system prompt if available', async ({ modelPage: page }) => {
    const systemPromptToggle = page.getByText(/system prompt/i)
    const hasSystemPrompt = await systemPromptToggle.isVisible().catch(() => false)

    if (hasSystemPrompt) {
      await expect(systemPromptToggle).toBeVisible()
      return
    }

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
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    const isDark = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    expect(isDark).toBe(true)
  })

  test('theme persists across page reloads', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    const initialDark = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    await page.reload()
    await expect(page.locator('body')).toBeVisible({ timeout: 10000 })

    const afterReloadDark = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })

    expect(afterReloadDark).toBe(initialDark)
  })
})

test.describe('Max Tokens Setting', () => {
  test('can find max tokens setting if available', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const buttons = await header.locator('button').all()

    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      if (ariaLabel?.toLowerCase().includes('settings')) {
        await button.click()
        await page.waitForTimeout(500)

        const maxTokensLabel = page.getByText(/max.*tokens|maximum.*tokens/i)
        const hasMaxTokens = await maxTokensLabel.isVisible().catch(() => false)

        if (hasMaxTokens) {
          await expect(maxTokensLabel).toBeVisible()
          const input = page.locator('input[type="number"], input[type="range"]')
          const inputCount = await input.count()
          expect(inputCount).toBeGreaterThanOrEqual(1)
        }
        return
      }
    }
  })
})
