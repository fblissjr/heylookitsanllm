// Minimal event bus for cross-component coordination
const bus = new EventTarget()
bus.emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }))
bus.on = (name, fn) => { bus.addEventListener(name, fn); return () => bus.removeEventListener(name, fn) }
export default bus
