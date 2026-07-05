// createPage: the one implementation of the page lifecycle contract.
//
// Pages export `createPage({ setup(ctx), teardown?(ctx) })`. Per mount, ctx
// provides:
//   el         mount root
//   state      fresh object per mount (no module singletons)
//   signal     AbortSignal aborted on teardown -- pass to fetch/streamChat
//   alive      false once torn down; check after any await
//   guard(fn)  wraps a callback so it no-ops after teardown
//   throttle(fn)  throttleToFrame that auto-cancels on teardown
//   onTeardown(fn)  register extra cleanup (runs before teardown())

import { throttleToFrame } from './utils.js';

export function createPage(spec) {
  let current = null;

  return {
    async mount(el) {
      const controller = new AbortController();
      const cleanups = [];
      const ctx = {
        el,
        state: {},
        signal: controller.signal,
        alive: true,
        onTeardown(fn) { cleanups.push(fn); },
        guard(fn) {
          return (...args) => (ctx.alive ? fn(...args) : undefined);
        },
        throttle(fn) {
          const throttled = throttleToFrame(fn);
          cleanups.push(() => throttled.cancel());
          return throttled;
        },
      };
      current = { ctx, controller, cleanups };
      await spec.setup(ctx);
    },

    async unmount() {
      if (!current) return;
      const { ctx, controller, cleanups } = current;
      current = null;
      ctx.alive = false;
      controller.abort();
      for (const fn of cleanups.reverse()) {
        try { fn(); } catch (err) { console.error('teardown cleanup failed', err); }
      }
      try { await spec.teardown?.(ctx); } catch (err) { console.error('teardown failed', err); }
    },
  };
}
