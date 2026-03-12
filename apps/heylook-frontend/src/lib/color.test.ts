import { describe, it, expect } from 'vitest'
import { probabilityToColor, probabilityToBarColor } from './color'

describe('probabilityToColor', () => {
  it('returns hsl string', () => {
    const color = probabilityToColor(0.5, false)
    expect(color).toMatch(/^hsl\(\d+/)
  })

  it('returns red-ish for low probability', () => {
    const color = probabilityToColor(0, false)
    expect(color).toContain('hsl(0,') // hue 0 = red
  })

  it('returns green-ish for high probability', () => {
    const color = probabilityToColor(1, false)
    expect(color).toContain('hsl(120,') // hue 120 = green
  })

  it('clamps values below 0', () => {
    const color = probabilityToColor(-0.5, false)
    expect(color).toContain('hsl(0,')
  })

  it('clamps values above 1', () => {
    const color = probabilityToColor(1.5, false)
    expect(color).toContain('hsl(120,')
  })

  it('adjusts lightness for dark mode', () => {
    const light = probabilityToColor(0.5, false)
    const dark = probabilityToColor(0.5, true)
    expect(light).not.toBe(dark)
  })
})

describe('probabilityToBarColor', () => {
  it('returns hsl string', () => {
    const color = probabilityToBarColor(0.5, false)
    expect(color).toMatch(/^hsl\(\d+/)
  })

  it('is brighter than chip color for same probability', () => {
    // Bar color has higher base lightness in dark mode
    const chip = probabilityToColor(0.5, true)
    const bar = probabilityToBarColor(0.5, true)
    // Extract lightness values
    const chipL = parseInt(chip.match(/(\d+)%\)$/)?.[1] ?? '0')
    const barL = parseInt(bar.match(/(\d+)%\)$/)?.[1] ?? '0')
    expect(barL).toBeGreaterThan(chipL)
  })
})
