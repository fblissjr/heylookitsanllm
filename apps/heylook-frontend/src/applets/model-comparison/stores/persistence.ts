import type { ComparisonPersistence } from '../types'

// No-op in-memory-only persistence. Swap to a real implementation
// when DuckDB integration is built.
export const sessionPersistence: ComparisonPersistence = {
  saveRun: async () => {},
  loadRuns: async () => [],
  deleteRun: async () => {},
}
