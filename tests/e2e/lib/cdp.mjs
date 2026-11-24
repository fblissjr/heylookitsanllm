// A browser driver with no dependencies: system Chrome, headless, over the
// Chrome DevTools Protocol (CDP), using node's own child_process, fetch and
// WebSocket. It replaced puppeteer-core, which had been removed from this
// directory in v2.0.240 and left every browser suite unable to launch.
//
// The surface is two primitives and a few conveniences on top of them:
//
//   page.cdp(method, params)   one protocol call, the escape hatch for anything
//                              not wrapped here (Emulation.*, Network.*, ...)
//   page.js(fn, ...args)       run a function (or an expression string) IN THE
//                              PAGE and get its JSON value back; a returned
//                              promise is awaited
//
// Checks are written against those two. There is deliberately no selector API,
// no element handles and no request interception: a check reads and drives the
// page with page-side JavaScript, which is the same DOM the app uses, and a
// response is held or failed from inside the page (see sendAndHold in run.mjs).
// A wrapper that grew those back would be a second puppeteer to maintain.
//
// What this cannot see: WebKit. Chrome answers questions about the DOM, state,
// requests and CSS rules; it does not reproduce Safari's scroll anchoring, the
// iOS keyboard or focus zoom. `ios-sim.mjs` and a real phone cover those.
//
// Must run unsandboxed: it starts Chrome and talks to 127.0.0.1.

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { sleep } from './harness.mjs';

const DEFAULT_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// CDP's modifier bitfield, and the few keys the checks press. A key needs its
// virtual key code for Chrome to run the default action (Escape closing a
// dialog, Enter submitting), not only to dispatch the DOM event.
const MODIFIER = { alt: 1, ctrl: 2, meta: 4, shift: 8 };
const JS_TIMEOUT_MS = 60000;
const KEYS = {
  Escape: { code: 'Escape', windowsVirtualKeyCode: 27 },
  Enter: { code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' },
  Tab: { code: 'Tab', windowsVirtualKeyCode: 9 },
};

class Page {
  constructor(ws, targetId, origin) {
    this.ws = ws;
    this.targetId = targetId;
    this.origin = origin;
    this.seq = 0;
    this.pending = new Map();
    // Collected for the whole life of the page; a check asserts on them or
    // clears them. An uncaught exception and a console.error are both here:
    // the app reports a failed page mount through console.error.
    this.errors = [];
    // Every request the page made: { method, url, done, failed }. Read a slice
    // from a mark taken before the action under test, so earlier traffic cannot
    // answer.
    this.requests = [];
    ws.addEventListener('message', (m) => this.onMessage(JSON.parse(m.data)));
    // A Chrome that dies mid-run must fail the calls waiting on it. Without
    // this they stay pending for good and the run hangs, teardown included.
    ws.addEventListener('close', () => {
      for (const waiter of this.pending.values()) {
        waiter.reject(new Error(`${waiter.method}: the DevTools socket closed (did Chrome exit?)`));
      }
      this.pending.clear();
    });
  }

  onMessage(msg) {
    if (msg.id) {
      const waiter = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (!waiter) return;
      if (msg.error) waiter.reject(new Error(`${waiter.method}: ${msg.error.message}`));
      else waiter.resolve(msg.result);
      return;
    }
    const p = msg.params;
    if (msg.method === 'Runtime.exceptionThrown') {
      this.errors.push(p.exceptionDetails.exception?.description ?? p.exceptionDetails.text);
    } else if (msg.method === 'Runtime.consoleAPICalled' && p.type === 'error') {
      this.errors.push('console.error: ' + p.args.map((a) => a.value ?? a.description ?? '').join(' '));
    } else if (msg.method === 'Network.requestWillBeSent') {
      this.requests.push({ id: p.requestId, method: p.request.method, url: p.request.url, done: false, failed: false });
    } else if (msg.method === 'Network.loadingFinished' || msg.method === 'Network.loadingFailed') {
      // `done` is what lets a check wait for a response instead of for time.
      // `failed` keeps a cancelled or dropped request from reading as loaded.
      const req = this.requests.findLast((r) => r.id === p.requestId);
      if (req) {
        req.done = true;
        req.failed = msg.method === 'Network.loadingFailed';
      }
    }
  }

  cdp(method, params = {}) {
    const id = ++this.seq;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject, method }));
  }

  // Run in the page. A function is serialized with its arguments, so it must
  // not close over anything on the node side: pass what it needs as arguments
  // (JSON values only). `userGesture` lets a click made here open a file picker
  // or start playback the way a real one would.
  //
  // Bounded: a page-side promise that never settles would otherwise hold the
  // run for good. JS_TIMEOUT_MS is generous on purpose; a wait that needs
  // longer belongs in waitFor, which polls.
  async js(fnOrExpression, ...args) {
    return this.evaluate(JS_TIMEOUT_MS, fnOrExpression, args);
  }

  async evaluate(timeoutMs, fnOrExpression, args) {
    // JSON.stringify(undefined) is the VALUE undefined, which would leave a
    // hole in the argument list (`(fn)(, x)`); spell it out instead.
    const literal = (a) => (a === undefined ? 'undefined' : JSON.stringify(a));
    const expression = typeof fnOrExpression === 'function'
      ? `(${fnOrExpression})(${args.map(literal).join(', ')})`
      : String(fnOrExpression);
    let timer = null;
    const r = await Promise.race([
      this.cdp('Runtime.evaluate', {
        expression, awaitPromise: true, returnByValue: true, userGesture: true,
      }),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`a page-side call did not settle in ${timeoutMs}ms`)), timeoutMs);
      }),
    ]).finally(() => clearTimeout(timer));
    if (r.exceptionDetails) {
      throw new Error('in page: ' + (r.exceptionDetails.exception?.description ?? r.exceptionDetails.text));
    }
    return r.result.value;
  }

  // Poll a page-side condition. `what` is the sentence the failure prints, so
  // say what was being waited for, not how.
  async waitFor(fnOrExpression, what, { timeout = 20000, args = [] } = {}) {
    const start = Date.now();
    let last = null;
    while (Date.now() - start < timeout) {
      try {
        // each poll is bounded by what is left, so one hung poll cannot outlive the wait
        const left = Math.max(1000, timeout - (Date.now() - start));
        const v = await this.evaluate(left, fnOrExpression, args);
        if (v) return v;
      } catch (err) { last = err; }
      await sleep(100);
    }
    throw new Error(`timed out after ${timeout}ms waiting for ${what}${last ? ` (last error: ${last.message})` : ''}`);
  }

  // The old document keeps answering until the new one commits, and a check
  // that writes to the page in that window writes to a document about to be
  // thrown away. So a sentinel is planted on the old one, and the wait is for
  // a document that does not carry it.
  async goto(url) {
    await this.js(() => { window.__e2eLeaving = true; });
    const r = await this.cdp('Page.navigate', { url });
    if (r.errorText) throw new Error(`could not load ${url}: ${r.errorText}`);
    await this.waitFor(() => !window.__e2eLeaving && document.readyState !== 'loading', `${url} to load`);
  }

  async reload() {
    await this.js(() => { window.__e2eLeaving = true; });
    await this.cdp('Page.reload');
    await this.waitFor(() => !window.__e2eLeaving && document.readyState !== 'loading', 'the page to reload');
  }

  // A real key event, through the browser's input pipeline: the page sees a
  // trusted keydown, and Chrome runs the key's default action.
  async press(key, { alt = false, ctrl = false, meta = false, shift = false } = {}) {
    const modifiers = (alt && MODIFIER.alt) | (ctrl && MODIFIER.ctrl) | (meta && MODIFIER.meta) | (shift && MODIFIER.shift);
    const { text, ...spec } = KEYS[key] ?? { code: key, text: key.length === 1 ? key : undefined };
    const base = { key, ...spec, nativeVirtualKeyCode: spec.windowsVirtualKeyCode, modifiers };
    await this.cdp('Input.dispatchKeyEvent', { type: text ? 'keyDown' : 'rawKeyDown', ...base, ...(text ? { text } : {}) });
    await this.cdp('Input.dispatchKeyEvent', { type: 'keyUp', ...base });
  }

  // Viewport and input media. `touch` sets what a phone reports, so
  // `(hover: none)` and `(pointer: coarse)` rules apply: a desktop Chrome
  // otherwise reports a mouse at any width, and a phone-width check passes
  // against CSS the phone never uses.
  async emulate({ width, height, scale = 1, touch = false, colorScheme = null } = {}) {
    await this.cdp('Emulation.setDeviceMetricsOverride', {
      width, height, deviceScaleFactor: scale, mobile: touch,
    });
    await this.cdp('Emulation.setTouchEmulationEnabled', { enabled: touch });
    const features = [];
    if (touch) features.push({ name: 'hover', value: 'none' }, { name: 'pointer', value: 'coarse' });
    if (colorScheme) features.push({ name: 'prefers-color-scheme', value: colorScheme });
    await this.cdp('Emulation.setEmulatedMedia', { features });
  }

  async close() {
    try { this.ws.close(); } catch { /* already closed */ }
    try { await fetch(`${this.origin}/json/close/${this.targetId}`); } catch { /* browser already gone */ }
  }
}

export async function launchChrome({ width = 1280, height = 900 } = {}) {
  const profile = mkdtempSync(path.join(os.tmpdir(), 'heylook-e2e-chrome-'));
  const proc = spawn(process.env.E2E_CHROME || DEFAULT_CHROME, [
    ...(process.env.E2E_HEADFUL ? [] : ['--headless=new']),
    // Port 0: Chrome takes a free port and writes it to DevToolsActivePort in
    // the profile. A fixed port collides with any other run on the machine.
    '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--disable-gpu',
    `--window-size=${width},${height}`, 'about:blank',
  ], { stdio: 'ignore' });

  let exited = false;
  proc.on('exit', () => { exited = true; });
  const kill = () => {
    if (!exited) proc.kill('SIGKILL');
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
  };

  const portFile = path.join(profile, 'DevToolsActivePort');
  let port = null;
  for (let i = 0; i < 200 && !port && !exited; i++) {
    if (existsSync(portFile)) port = Number(readFileSync(portFile, 'utf8').split('\n')[0]) || null;
    if (!port) await sleep(100);
  }
  if (!port) {
    kill();
    throw new Error(exited
      ? 'Chrome exited before it published a DevTools port. Is this running sandboxed? It must not be.'
      : 'Chrome never published a DevTools port (no DevToolsActivePort in its profile).');
  }
  const origin = `http://127.0.0.1:${port}`;
  let version;
  try {
    version = (await (await fetch(`${origin}/json/version`)).json()).Browser;
  } catch (err) {
    kill(); // the caller has no handle yet, so nobody else can
    throw new Error(`Chrome published port ${port} but did not answer on it: ${err.message}`);
  }

  return {
    version,
    async newPage() {
      const target = await (await fetch(`${origin}/json/new?about:blank`, { method: 'PUT' })).json();
      const ws = new WebSocket(target.webSocketDebuggerUrl);
      await new Promise((resolve, reject) => {
        ws.addEventListener('open', resolve, { once: true });
        ws.addEventListener('error', () => reject(new Error('could not open the DevTools socket')), { once: true });
      });
      const page = new Page(ws, target.id, origin);
      await page.cdp('Runtime.enable');
      await page.cdp('Network.enable');
      await page.cdp('Page.enable');
      await page.emulate({ width, height });
      return page;
    },
    async close() { kill(); },
  };
}
