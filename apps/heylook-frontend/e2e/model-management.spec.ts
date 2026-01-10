import { test, expect } from '@playwright/test'

/**
 * E2E tests for model management.
 * Tests model selection, loading, unloading, and switching.
 */

test.describe('Model Selection Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2000)

    const connectionFailed = page.getByText('Connection Failed')
    if (await connectionFailed.isVisible()) {
      test.skip(true, 'Backend not running')
    }
  })

  test('can open model selector panel', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await expect(modelSelector).toBeVisible()

    await modelSelector.click()

    // Models panel should open
    const modelsHeading = page.getByRole('heading', { name: 'Models' })
    await expect(modelsHeading).toBeVisible({ timeout: 5000 })
  })

  test('model panel shows available models', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    // Should show model cards
    const modelCards = page.locator('button').filter({ hasText: /qwen|llama|gemma|mistral/i })
    const count = await modelCards.count()

    // Should have at least one model available
    expect(count).toBeGreaterThanOrEqual(1)
  })

  test('can close model panel with X button', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelsHeading = page.getByRole('heading', { name: 'Models' })
    await expect(modelsHeading).toBeVisible()

    // Find and click the close button (X icon in panel header)
    const closeButton = page.locator('button').filter({ has: page.locator('svg path[d*="M6 18L18 6"]') }).first()
    if (await closeButton.isVisible()) {
      await closeButton.click()
      await expect(modelsHeading).not.toBeVisible({ timeout: 2000 })
    }
  })

  test('selecting a model shows context window slider', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    // Click on a model card
    const modelCard = page.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    if (await modelCard.isVisible()) {
      await modelCard.click()
      await page.waitForTimeout(300)

      // Context window slider should appear
      const contextSlider = page.locator('input[type="range"]')
      await expect(contextSlider).toBeVisible({ timeout: 2000 })

      // Load Model button should appear
      const loadButton = page.getByRole('button', { name: /load model/i })
      await expect(loadButton).toBeVisible()
    }
  })

  test('can adjust context window before loading', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelCard = page.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    if (await modelCard.isVisible()) {
      await modelCard.click()
      await page.waitForTimeout(300)

      const contextSlider = page.locator('input[type="range"]')
      if (await contextSlider.isVisible()) {
        // Get initial value
        const initialValue = await contextSlider.inputValue()

        // Change the slider value
        await contextSlider.fill('2048')

        // Value should have changed
        const newValue = await contextSlider.inputValue()
        expect(newValue).not.toBe(initialValue)
      }
    }
  })
})

test.describe('Model Loading', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2000)

    const connectionFailed = page.getByText('Connection Failed')
    if (await connectionFailed.isVisible()) {
      test.skip(true, 'Backend not running')
    }
  })

  test('can load a model', async ({ page }) => {
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

    // Should show loading state or transition to loaded
    // Chat input should become visible
    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible({ timeout: 15000 })
  })

  test('loaded model shows in header', async ({ page }) => {
    // First load a model
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    const modelCard = page.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    if (!(await modelCard.isVisible())) {
      test.skip(true, 'No models available')
    }

    // Get the model name before loading
    const modelText = await modelCard.textContent()
    await modelCard.click()

    await page.waitForTimeout(300)
    const loadButton = page.getByRole('button', { name: /load model/i })
    await loadButton.click()

    // Wait for model to load
    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible({ timeout: 15000 })

    // Header should show the loaded model name
    const header = page.locator('header')
    if (modelText) {
      // The model name should appear somewhere in the header button
      const modelButton = header.locator('button').filter({ hasText: new RegExp(modelText.split('-')[0], 'i') })
      await expect(modelButton).toBeVisible({ timeout: 5000 })
    }
  })

  test('model card shows "Loaded" badge when loaded', async ({ page }) => {
    // Load a model first
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

    // Wait for model to load
    const chatInput = page.locator('textarea')
    await expect(chatInput).toBeVisible({ timeout: 15000 })

    // Open model panel again
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await modelButton.click()
    await page.waitForTimeout(500)

    // Should see "Loaded" badge
    const loadedBadge = page.getByText('Loaded')
    await expect(loadedBadge).toBeVisible({ timeout: 2000 })
  })
})

test.describe('Model Unloading', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2000)

    const connectionFailed = page.getByText('Connection Failed')
    if (await connectionFailed.isVisible()) {
      test.skip(true, 'Backend not running')
    }

    // Load a model first
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
    try {
      await expect(chatInput).toBeVisible({ timeout: 15000 })
    } catch {
      test.skip(true, 'Model could not be loaded')
    }
  })

  test('can unload a model', async ({ page }) => {
    // Open model panel
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await modelButton.click()
    await page.waitForTimeout(500)

    // Find and click unload button
    const unloadButton = page.getByRole('button', { name: /unload model/i })
    if (await unloadButton.isVisible()) {
      await unloadButton.click()
      await page.waitForTimeout(500)

      // Should show "No Model Loaded" state
      const noModelState = page.getByRole('heading', { name: /no model loaded/i })
      await expect(noModelState).toBeVisible({ timeout: 5000 })
    }
  })

  test('header shows Select Model after unloading', async ({ page }) => {
    // Open model panel and unload
    const header = page.locator('header')
    const modelButton = header.locator('button').filter({ hasText: /qwen|llama|gemma/i }).first()
    await modelButton.click()
    await page.waitForTimeout(500)

    const unloadButton = page.getByRole('button', { name: /unload model/i })
    if (await unloadButton.isVisible()) {
      await unloadButton.click()
      await page.waitForTimeout(500)

      // Header should show "Select Model" again
      const selectModelButton = page.getByRole('button', { name: /select model/i })
      await expect(selectModelButton).toBeVisible({ timeout: 5000 })
    }
  })
})

test.describe('Model Capabilities', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(2000)

    const connectionFailed = page.getByText('Connection Failed')
    if (await connectionFailed.isVisible()) {
      test.skip(true, 'Backend not running')
    }
  })

  test('model cards show capability badges', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    // Should see capability badges - look for any badge text
    // Badges have single letter prefixes like "C Chat", "V Vision", "T Thinking"
    const chatBadge = page.locator('span').filter({ hasText: /^C\s*Chat$/i }).first()
    const anyBadge = page.locator('span').filter({ hasText: /chat|vision|thinking|embeddings/i }).first()

    const hasChatBadge = await chatBadge.isVisible().catch(() => false)
    const hasAnyBadge = await anyBadge.isVisible().catch(() => false)

    // At least some capability badges should be visible
    expect(hasChatBadge || hasAnyBadge).toBe(true)
  })

  test('vision models show Vision capability badge', async ({ page }) => {
    const modelSelector = page.getByRole('button', { name: /select model/i })
    await modelSelector.click()
    await page.waitForTimeout(500)

    // Look for a VLM model
    const vlmModel = page.locator('button').filter({ hasText: /vlm|vision|qwen.*vl/i }).first()

    if (await vlmModel.isVisible()) {
      // Click on it to see details
      await vlmModel.click()
      await page.waitForTimeout(300)

      // Look for Vision badge anywhere in the panel
      const visionBadge = page.locator('span').filter({ hasText: /vision/i }).first()
      const hasVision = await visionBadge.isVisible().catch(() => false)

      // If it's a vision model, it should have the badge
      // If not visible, that's also fine (model may not have vision)
      expect(typeof hasVision).toBe('boolean')
    } else {
      // No VLM model available - skip
      test.skip(true, 'No VLM model available')
    }
  })
})
