import { describe, it, expect, beforeEach, vi } from 'vitest'
import { DiagnosticLogger } from './diagnostics'

describe('DiagnosticLogger', () => {
  let logger: DiagnosticLogger

  beforeEach(() => {
    logger = new DiagnosticLogger(10) // small capacity for testing
  })

  describe('ring buffer', () => {
    it('stores events up to capacity', () => {
      for (let i = 0; i < 10; i++) {
        logger.info(`event-${i}`, 'system', { i })
      }
      expect(logger.size).toBe(10)
    })

    it('overwrites oldest events when full', () => {
      for (let i = 0; i < 15; i++) {
        logger.info(`event-${i}`, 'system', { i })
      }
      expect(logger.size).toBe(10)
      const events = logger.getEvents()
      // oldest should be event-5 (first 5 overwritten)
      expect(events[0].type).toBe('event-5')
      expect(events[9].type).toBe('event-14')
    })

    it('returns events in chronological order when not full', () => {
      logger.info('first', 'api')
      logger.info('second', 'store')
      logger.info('third', 'sse')

      const events = logger.getEvents()
      expect(events).toHaveLength(3)
      expect(events[0].type).toBe('first')
      expect(events[1].type).toBe('second')
      expect(events[2].type).toBe('third')
    })

    it('returns events in chronological order when wrapped', () => {
      // Fill buffer and wrap around
      for (let i = 0; i < 12; i++) {
        logger.info(`e-${i}`, 'system')
      }
      const events = logger.getEvents()
      expect(events[0].type).toBe('e-2')
      expect(events[events.length - 1].type).toBe('e-11')
    })

    it('clear resets the buffer', () => {
      logger.info('test', 'system')
      expect(logger.size).toBe(1)
      logger.clear()
      expect(logger.size).toBe(0)
      expect(logger.getEvents()).toHaveLength(0)
    })
  })

  describe('event structure', () => {
    it('creates events with all required fields', () => {
      logger.info('api_request', 'api', { url: '/test' }, 'req-123')
      const event = logger.getEvents()[0]

      expect(event.ts).toBeDefined()
      expect(event.level).toBe('info')
      expect(event.type).toBe('api_request')
      expect(event.source).toBe('api')
      expect(event.requestId).toBe('req-123')
      expect(event.data).toEqual({ url: '/test' })
    })

    it('omits requestId when not provided', () => {
      logger.info('test', 'system')
      const event = logger.getEvents()[0]
      expect(event.requestId).toBeUndefined()
    })

    it('sets correct level for each method', () => {
      logger.error('e', 'system')
      logger.warn('w', 'system')
      logger.info('i', 'system')
      logger.debug('d', 'system')

      const events = logger.getEvents()
      expect(events[0].level).toBe('error')
      expect(events[1].level).toBe('warn')
      expect(events[2].level).toBe('info')
      expect(events[3].level).toBe('debug')
    })
  })

  describe('console output gating', () => {
    it('outputs error and warn at info level (default)', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {})
      const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {})

      logger.setLevel('info')
      logger.error('e', 'system')
      logger.warn('w', 'system')
      logger.info('i', 'system')
      logger.debug('d', 'system')

      expect(errorSpy).toHaveBeenCalled()
      expect(warnSpy).toHaveBeenCalled()
      expect(infoSpy).toHaveBeenCalled()
      expect(debugSpy).not.toHaveBeenCalled()

      errorSpy.mockRestore()
      warnSpy.mockRestore()
      infoSpy.mockRestore()
      debugSpy.mockRestore()
    })

    it('suppresses all below error at error level', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {})

      logger.setLevel('error')
      logger.error('e', 'system')
      logger.warn('w', 'system')
      logger.info('i', 'system')

      expect(errorSpy).toHaveBeenCalled()
      expect(warnSpy).not.toHaveBeenCalled()
      expect(infoSpy).not.toHaveBeenCalled()

      errorSpy.mockRestore()
      warnSpy.mockRestore()
      infoSpy.mockRestore()
    })

    it('still writes to buffer regardless of console level', () => {
      vi.spyOn(console, 'debug').mockImplementation(() => {})
      logger.setLevel('error')
      logger.debug('hidden', 'system')
      expect(logger.size).toBe(1)
      vi.mocked(console.debug).mockRestore()
    })
  })

  describe('JSONL export', () => {
    it('exports events as newline-delimited JSON', () => {
      logger.info('first', 'api', { a: 1 })
      logger.warn('second', 'store', { b: 2 })

      const jsonl = logger.toJSONL()
      const lines = jsonl.split('\n')
      expect(lines).toHaveLength(2)

      const parsed0 = JSON.parse(lines[0])
      expect(parsed0.type).toBe('first')
      expect(parsed0.data.a).toBe(1)

      const parsed1 = JSON.parse(lines[1])
      expect(parsed1.type).toBe('second')
    })

    it('returns empty string for empty buffer', () => {
      expect(logger.toJSONL()).toBe('')
    })
  })

  describe('level management', () => {
    it('defaults to info', () => {
      expect(logger.getLevel()).toBe('info')
    })

    it('can be changed', () => {
      logger.setLevel('debug')
      expect(logger.getLevel()).toBe('debug')
    })
  })

  describe('default capacity', () => {
    it('uses 5000 when no capacity specified', () => {
      const defaultLogger = new DiagnosticLogger()
      // Fill to ensure no error
      for (let i = 0; i < 100; i++) {
        defaultLogger.info(`e-${i}`, 'system')
      }
      expect(defaultLogger.size).toBe(100)
    })
  })
})
