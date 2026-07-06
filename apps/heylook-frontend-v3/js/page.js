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
//   linkedController()  AbortController chained to signal; abort() it on release
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
        // Per-operation AbortController chained to the page signal. The chain
        // listener is registered WITH the controller's own signal, so it
        // self-removes when the controller aborts -- callers must abort() the
        // controller when the operation ends (a no-op after normal
        // completion) or listeners accumulate on ctx.signal over a session.
        linkedController() {
          const controller = new AbortController();
          ctx.signal.addEventListener('abort', () => controller.abort(),
            { once: true, signal: controller.signal });
          return controller;
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
