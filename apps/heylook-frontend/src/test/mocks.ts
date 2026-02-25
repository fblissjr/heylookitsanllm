/**
 * Shared test mocks for browser APIs that need special handling.
 */

/**
 * Mock FileReader as a proper constructor class.
 * `vi.fn().mockImplementation(...)` produces objects that aren't `new`-able,
 * so we need a real class to satisfy `new FileReader()` in production code.
 *
 * Returns a restore function to call in afterEach or at end of test.
 *
 * @param result - The fake data URL result. Defaults to a minimal PNG data URL.
 */
export function mockFileReader(result = 'data:image/png;base64,abc123') {
  const original = globalThis.FileReader

  class FakeFileReader {
    result: string | null = result
    onload: (() => void) | null = null
    onerror: (() => void) | null = null
    readAsDataURL() {
      if (this.onload) this.onload()
    }
  }

  globalThis.FileReader = FakeFileReader as unknown as typeof FileReader
  return () => { globalThis.FileReader = original }
}
