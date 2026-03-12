// Diagnostic logging with ring buffer for frontend debugging

export type LogLevel = 'error' | 'warn' | 'info' | 'debug'
export type EventSource = 'api' | 'store' | 'sse' | 'user' | 'system'

export interface DiagnosticEvent {
  ts: string
  level: LogLevel
  type: string
  source: EventSource
  requestId?: string
  data: Record<string, unknown>
}

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
}

const CONSOLE_METHODS: Record<LogLevel, 'error' | 'warn' | 'info' | 'debug'> = {
  error: 'error',
  warn: 'warn',
  info: 'info',
  debug: 'debug',
}

export class DiagnosticLogger {
  private buffer: (DiagnosticEvent | null)[]
  private head = 0
  private count = 0
  private capacity: number
  private consoleLevel: LogLevel = 'info'

  constructor(capacity = 5000) {
    this.capacity = capacity
    this.buffer = new Array(capacity).fill(null)
  }

  setLevel(level: LogLevel): void {
    this.consoleLevel = level
  }

  getLevel(): LogLevel {
    return this.consoleLevel
  }

  log(
    level: LogLevel,
    type: string,
    source: EventSource,
    data: Record<string, unknown> = {},
    requestId?: string,
  ): void {
    const event: DiagnosticEvent = {
      ts: new Date().toISOString(),
      level,
      type,
      source,
      ...(requestId ? { requestId } : {}),
      data,
    }

    // Always write to ring buffer
    this.buffer[this.head] = event
    this.head = (this.head + 1) % this.capacity
    if (this.count < this.capacity) this.count++

    // Console output gated by level
    if (LOG_LEVEL_PRIORITY[level] <= LOG_LEVEL_PRIORITY[this.consoleLevel]) {
      const method = CONSOLE_METHODS[level]
      const prefix = `[${source}:${type}]`
      if (requestId) {
        console[method](prefix, `req=${requestId}`, data)
      } else {
        console[method](prefix, data)
      }
    }
  }

  error(type: string, source: EventSource, data: Record<string, unknown> = {}, requestId?: string): void {
    this.log('error', type, source, data, requestId)
  }

  warn(type: string, source: EventSource, data: Record<string, unknown> = {}, requestId?: string): void {
    this.log('warn', type, source, data, requestId)
  }

  info(type: string, source: EventSource, data: Record<string, unknown> = {}, requestId?: string): void {
    this.log('info', type, source, data, requestId)
  }

  debug(type: string, source: EventSource, data: Record<string, unknown> = {}, requestId?: string): void {
    this.log('debug', type, source, data, requestId)
  }

  /** Get all events in chronological order. */
  getEvents(): DiagnosticEvent[] {
    const events: DiagnosticEvent[] = []
    if (this.count < this.capacity) {
      // Buffer not full yet -- events start at 0
      for (let i = 0; i < this.count; i++) {
        const evt = this.buffer[i]
        if (evt) events.push(evt)
      }
    } else {
      // Buffer full -- oldest event is at head
      for (let i = 0; i < this.capacity; i++) {
        const idx = (this.head + i) % this.capacity
        const evt = this.buffer[idx]
        if (evt) events.push(evt)
      }
    }
    return events
  }

  /** Export events as a JSONL string. */
  toJSONL(): string {
    return this.getEvents().map(e => JSON.stringify(e)).join('\n')
  }

  /** Trigger a browser download of the event log as JSONL. */
  downloadAsJSONL(filename = 'heylook-diagnostics.jsonl'): void {
    const blob = new Blob([this.toJSONL()], { type: 'application/x-ndjson' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  /** Number of events currently stored. */
  get size(): number {
    return this.count
  }

  /** Clear all events. */
  clear(): void {
    this.buffer = new Array(this.capacity).fill(null)
    this.head = 0
    this.count = 0
  }
}

// Singleton
export const logger = new DiagnosticLogger()

// Expose to devtools
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any
  w.__diagLogger = logger
  w.__setLogLevel = (level: LogLevel) => {
    logger.setLevel(level)
    console.info(`[diagnostics] Log level set to: ${level}`)
  }
}
