import { useCallback, useRef, useEffect } from 'react'

interface UseLongPressOptions {
  onLongPress: () => void
  onClick?: () => void
  delay?: number
}

export function useLongPress({ onLongPress, onClick, delay = 500 }: UseLongPressOptions) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isLongPressRef = useRef(false)
  const isMountedRef = useRef(true)

  // Track mounted state to prevent callback after unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      // Clean up any pending timer on unmount
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }
  }, [])

  const start = useCallback((e: React.TouchEvent | React.MouseEvent) => {
    // Prevent text selection on long press
    e.preventDefault()
    isLongPressRef.current = false
    timerRef.current = setTimeout(() => {
      // Only trigger if still mounted
      if (isMountedRef.current) {
        isLongPressRef.current = true
        onLongPress()
      }
    }, delay)
  }, [onLongPress, delay])

  const clear = useCallback((_e: React.TouchEvent | React.MouseEvent, shouldTriggerClick = true) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    if (shouldTriggerClick && !isLongPressRef.current && onClick && isMountedRef.current) {
      onClick()
    }
  }, [onClick])

  const cancel = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    isLongPressRef.current = false
  }, [])

  return {
    onMouseDown: start,
    onMouseUp: (e: React.MouseEvent) => clear(e),
    onMouseLeave: cancel,
    onTouchStart: start,
    onTouchEnd: (e: React.TouchEvent) => clear(e),
    onTouchCancel: cancel,
  }
}
