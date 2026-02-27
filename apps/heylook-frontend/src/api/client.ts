// API Client with error handling

import { generateId } from '../lib/id'
import { logger } from '../lib/diagnostics'

export class APIError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body?: unknown
  ) {
    super(`API Error ${status}: ${statusText}`)
    this.name = 'APIError'
  }
}

export async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const requestId = generateId('req')

  logger.debug('api_request', 'api', { endpoint, method: options?.method ?? 'GET' }, requestId)

  try {
    const response = await fetch(endpoint, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId,
        ...options?.headers,
      },
    })

    if (!response.ok) {
      let body: unknown
      try {
        body = await response.json()
      } catch {
        body = await response.text()
      }
      logger.warn('api_error', 'api', { endpoint, status: response.status, body }, requestId)
      throw new APIError(response.status, response.statusText, body)
    }

    logger.debug('api_response', 'api', { endpoint, status: response.status }, requestId)
    return response.json()
  } catch (error) {
    if (error instanceof APIError) throw error
    if (error instanceof TypeError && error.message.includes('fetch')) {
      logger.error('api_network_error', 'api', { endpoint }, requestId)
      throw new Error('Network error: Unable to connect to server')
    }
    throw error
  }
}

export async function postAPI<T>(
  endpoint: string,
  data: unknown,
  options?: RequestInit
): Promise<T> {
  return fetchAPI<T>(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
    ...options,
  })
}
