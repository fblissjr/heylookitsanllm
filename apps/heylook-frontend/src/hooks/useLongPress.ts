import { useCallback, useRef } from 'react'

interface UseLongPressOptions {
  onLongPress: () => void
  onClick?: () => void
  delay?: number
}

export function useLongPress({ onLongPress, onClick, delay = 500 }: UseLongPressOptions) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isLongPressRef = useRef(false)

  const start = useCallback((e: React.TouchEvent | React.MouseEvent) => {
    // Prevent text selection on long press
    e.preventDefault()
    isLongPressRef.current = false
    timerRef.current = setTimeout(() => {
      isLongPressRef.current = true
      onLongPress()
    }, delay)
  }, [onLongPress, delay])

  const clear = useCallback((_e: React.TouchEvent | React.MouseEvent, shouldTriggerClick = true) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    if (shouldTriggerClick && !isLongPressRef.current && onClick) {
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
