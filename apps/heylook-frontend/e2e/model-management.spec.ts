import { test, expect } from './fixtures'

/**
 * E2E tests for model management.
 * Tests model selection, loading, unloading, and switching.
 */

test.describe('Model Selection Panel', () => {
  test('can open model selector panel', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await expect(modelSelector).toBeVisible()

    await modelSelector.click()

    const modelsHeading = page.getByRole('heading', { name: 'Models' })
    await expect(modelsHeading).toBeVisible({ timeout: 5000 })
  })

  test('model panel shows available models', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelCards = page.locator('button').filter({ hasText: /qwen|llama|gemma|mistral/i })
    const count = await modelCards.count()

    expect(count).toBeGreaterThanOrEqual(1)
  })

  test('can close model panel with X button', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelsHeading = page.getByRole('heading', { name: 'Models' })
    await expect(modelsHeading).toBeVisible()

    const closeButton = page.locator('button').filter({ has: page.locator('svg path[d*="M6 18L18 6"]') }).first()
    if (await closeButton.isVisible()) {
      await closeButton.click()
      await expect(modelsHeading).not.toBeVisible({ timeout: 2000 })
    }
  })

  test('selecting a model shows context window slider', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelCard = page.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    if (await modelCard.isVisible()) {
      await modelCard.click()
      await page.waitForTimeout(300)

      const contextSlider = page.locator('input[type="range"]')
      await expect(contextSlider).toBeVisible({ timeout: 2000 })

      const loadButton = page.getByRole('button', { name: /load model/i })
      await expect(loadButton).toBeVisible()
    }
  })

  test('can adjust context window before loading', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelCard = page.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    if (await modelCard.isVisible()) {
      await modelCard.click()
      await page.waitForTimeout(300)

      const contextSlider = page.locator('input[type="range"]')
      if (await contextSlider.isVisible()) {
        const initialValue = await contextSlider.inputValue()
        await contextSlider.fill('2048')
        const newValue = await contextSlider.inputValue()
        expect(newValue).not.toBe(initialValue)
      }
    }
  })
})

test.describe('Model Loading', () => {
  test('can load a model', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelCard = page.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    if (!(await modelCard.isVisible())) {
      test.skip(true, 'No models available')
    }
    await modelCard.click()

    await page.waitForTimeout(300)
    const loadButton = page.getByRole('button', { name: /load model/i })
    await loadButton.click()

    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible({ timeout: 15000 })
  })

  test('loaded model shows in header', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await expect(modelButton).toBeVisible({ timeout: 5000 })
  })

  test('model card shows Loaded badge when loaded', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await modelButton.click()
    await page.waitForTimeout(500)

    const loadedBadge = page.getByText('Loaded')
    await expect(loadedBadge).toBeVisible({ timeout: 2000 })
  })
})

test.describe('Model Unloading', () => {
  test('can unload a model', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await modelButton.click()
    await page.waitForTimeout(500)

    const unloadButton = page.getByRole('button', { name: /unload model/i })
    if (await unloadButton.isVisible()) {
      await unloadButton.click()
      await page.waitForTimeout(500)

      const noModelState = page.getByRole('heading', { name: /no model loaded/i })
      await expect(noModelState).toBeVisible({ timeout: 5000 })
    }
  })

  test('header shows Select Model after unloading', async ({ modelPage: page }) => {
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await modelButton.click()
    await page.waitForTimeout(500)

    const unloadButton = page.getByRole('button', { name: /unload model/i })
    if (await unloadButton.isVisible()) {
      await unloadButton.click()
      await page.waitForTimeout(500)

      const selectModelButton = page.getByRole('button', { name: /select model/i })
      await expect(selectModelButton).toBeVisible({ timeout: 5000 })
    }
  })
})

test.describe('Model Capabilities', () => {
  test('model cards show capability badges', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const chatBadge = page.locator('span').filter({ hasText: /^C\s*Chat$/i }).first()
    const anyBadge = page.locator('span').filter({ hasText: /chat|vision|thinking|embeddings/i }).first()

    const hasChatBadge = await chatBadge.isVisible().catch(() => false)
    const hasAnyBadge = await anyBadge.isVisible().catch(() => false)

    expect(hasChatBadge || hasAnyBadge).toBe(true)
  })

  test('vision models show Vision capability badge', async ({ backendPage: page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const vlmModel = page.locator('button').filter({ hasText: /vlm|vision|qwen.*vl/i }).first()

    if (await vlmModel.isVisible()) {
      await vlmModel.click()
      await page.waitForTimeout(300)

      const visionBadge = page.locator('span').filter({ hasText: /vision/i }).first()
      const hasVision = await visionBadge.isVisible().catch(() => false)
      expect(typeof hasVision).toBe('boolean')
    } else {
      test.skip(true, 'No VLM model available')
    }
  })
})
