// iOS keyboard check: REAL Mobile Safari (WebKit) in the iOS Simulator, driven
// through Apple's own `safaridriver`, against an ALREADY-RUNNING heylook server.
//
//   bun run e2e:ios
//
// ============================================================================
// STATUS: first run 2026-09-06. 3/3 with 4 SKIPPED, and the skip is the finding.
//
// The plumbing works: it boots the simulator, drives real Mobile Safari, loads
// the page and measures. What the Simulator will NOT do under safaridriver is
// raise the software keyboard. Measured, in this order:
//   1. WebDriver click      -> document.activeElement stayed BODY, no keyboard.
//   2. ConnectHardwareKeyboard=false + reboot -> no change (reverted).
//   3. Element Send Keys    -> activeElement became TEXTAREA, so FOCUS lands --
//                              and visualViewport still never shrank.
// Focus works; the keyboard does not exist there. A real device is the
// remaining path, which is what IOS_REAL_DEVICE=1 is for.
//
// The four keyboard checks are therefore gated on the ENVIRONMENT, not on the
// outcome. Gating them on "the viewport did not shrink" would convert a real
// app regression into a silent skip, which is the one thing a skip must never
// do; gating on "this platform has no keyboard" cannot. On a real device they
// run for real and fail for real.
//
// What still runs here is real WebKit at iPhone size -- not redundant with the
// Chrome 390px checks, which emulate. What is NOT covered is the keyboard
// behaviour this file was written for; that needs a device.
// ============================================================================
//
// Why this exists and why Chrome cannot do it. When the software keyboard
// opens, iOS Safari keeps the LAYOUT viewport at full height and shrinks only
// the VISUAL viewport, then scrolls the page so the focused field shows.
// Chrome shrinks the layout viewport itself (the `interactive-widget` default).
// Everything in question here follows the layout viewport -- the fixed
// `#bottom-nav`, the `100dvh` app shell, the composer at its foot -- so a Chrome
// emulation answers a different question and can pass while the phone
// misbehaves. `render.mjs`'s mid-stream viewport shrink checks REFLOW under a
// short viewport; this checks what WebKit actually does with the keyboard.
//
// What it drives: the iOS Simulator (Xcode) running a real Mobile Safari, via
// the W3C WebDriver protocol that `/usr/bin/safaridriver` speaks. No npm
// dependency: raw fetch() against the driver's HTTP port. The simulator shares
// the Mac's network, so a server on 127.0.0.1 is reachable from inside it.
//
// One-time prerequisites (cannot be scripted from here):
//   1. `safaridriver --enable` on the Mac (admin, once).
//   2. In the booted simulator: Settings > Safari > Advanced > Remote
//      Automation = ON (once per simulator device). Without it the session
//      request fails; the error is surfaced with this instruction.
//   3. A running heylook server with a model that serves chat (scripts/
//      dev_server.sh, or the daily one -- this script only READS the page and
//      types into the composer; it never sends). IOS_SIM_BASE names it.
//
// What is asserted with the keyboard up (each a separate check so a partial
// answer still reads):
//   - the keyboard really opened: visualViewport.height shrank below
//     innerHeight (a WebDriver click may not raise the keyboard -- if it does
//     not, that is reported as the finding, not hidden behind a skip)
//   - the message field is fully inside the VISUAL viewport
//   - Send is fully inside the visual viewport
//   - the page does not scroll horizontally
//   - informational, never failing: whether #bottom-nav and the top bar are
//     still within the visual viewport, and how far the page was scrolled.
//     These are the numbers that decide whether hiding the nav on focus is
//     worth building (internal/log/log_2026-09-05.md).
//
// Screenshots (before + after keyboard) go to IOS_SIM_SHOTS, default a dir
// under the OS temp dir; the paths are printed.
//
// Config: IOS_SIM_BASE (default http://127.0.0.1:8000), IOS_SIM_DEVICE (default
// "iPhone 15 Pro" -- the newest available iOS runtime carrying that name wins;
// no 17 Pro exists in the simulator list, safe-area sizes differ by a few
// points), IOS_SIM_UDID (bypass the name lookup), IOS_SIM_PORT (safaridriver
// port, default 4445), IOS_SIM_SHOTS (screenshot dir), IOS_SIM_KEEP (set to
// leave the simulator booted afterwards).

import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { Suite, printSummary, assert, waitFor, sleep, skip } from './lib/harness.mjs';

const BASE = process.env.IOS_SIM_BASE || 'http://127.0.0.1:8000';
const DEVICE = process.env.IOS_SIM_DEVICE || 'iPhone 15 Pro';
// The Simulator under safaridriver focuses a field but never raises the
// software keyboard (measured 2026-09-06: Element Send Keys sets
// document.activeElement to TEXTAREA and visualViewport stays full height).
// So the keyboard checks are gated on the ENVIRONMENT, not on the outcome --
// gating on "the viewport did not shrink" would turn a real regression into
// a silent skip, which is the one thing a skip must never do. Point this at
// a real device (Web Inspector enabled, safaridriver over USB) to run them.
const REAL_DEVICE = process.env.IOS_REAL_DEVICE === '1';
const NO_KEYBOARD = 'the iOS Simulator does not raise a software keyboard under '
  + 'safaridriver (focus lands, visualViewport never shrinks). Set IOS_REAL_DEVICE=1 '
  + 'against a real iPhone to run this.';
const PORT = Number(process.env.IOS_SIM_PORT || 4445);
const SHOTS = process.env.IOS_SIM_SHOTS || path.join(os.tmpdir(), 'heylook-ios-sim');
const KEEP = Boolean(process.env.IOS_SIM_KEEP);

const COMPOSER = '.chat__composer textarea';

// --- simulator ---------------------------------------------------------------

function simctl(...args) {
  return execFileSync('xcrun', ['simctl', ...args], { encoding: 'utf8' });
}

// Pick the device: exact name match, newest iOS runtime. `simctl list -j`
// keys devices by runtime identifier (com.apple.CoreSimulator.SimRuntime.iOS-26-0).
function findDeviceUdid() {
  if (process.env.IOS_SIM_UDID) return process.env.IOS_SIM_UDID;
  const list = JSON.parse(simctl('list', 'devices', 'available', '-j'));
  const candidates = [];
  for (const [runtime, devices] of Object.entries(list.devices)) {
    const m = runtime.match(/iOS-(\d+)-(\d+)/);
    if (!m) continue;
    const version = Number(m[1]) * 1000 + Number(m[2]);
    for (const d of devices) {
      if (d.name === DEVICE && d.isAvailable !== false) candidates.push({ udid: d.udid, version, runtime });
    }
  }
  candidates.sort((a, b) => b.version - a.version);
  if (!candidates.length) {
    throw new Error(`no available simulator named "${DEVICE}" -- xcrun simctl list devices available; or set IOS_SIM_UDID`);
  }
  console.log(`[ios] device ${DEVICE} ${candidates[0].udid} (${candidates[0].runtime})`);
  return candidates[0].udid;
}

function bootSimulator(udid) {
  try {
    simctl('boot', udid);
  } catch (err) {
    // "Unable to boot device in current state: Booted" is the fine case.
    if (!/current state: Booted/.test(String(err.stderr || err.message))) throw err;
  }
  // -b blocks until the boot has finished (Springboard up), not just started.
  simctl('bootstatus', udid, '-b');
}

// --- safaridriver (W3C WebDriver over HTTP) ----------------------------------

async function startDriver() {
  const proc = spawn('safaridriver', ['-p', String(PORT)], { stdio: ['ignore', 'pipe', 'pipe'] });
  let err = '';
  proc.stderr.on('data', (d) => { err += d; });
  await waitFor(async () => {
    const r = await fetch(`http://127.0.0.1:${PORT}/status`).catch(() => null);
    return r && r.ok;
  }, { timeout: 15000, message: () => `safaridriver did not answer on :${PORT}${err ? `\n${err}` : ''}` });
  return proc;
}

class Driver {
  constructor(port) { this.base = `http://127.0.0.1:${port}`; this.sid = null; }

  async call(method, route, body) {
    const r = await fetch(`${this.base}${route}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) {
      const v = json.value || {};
      throw new Error(`${method} ${route} -> ${r.status} ${v.error || ''}: ${v.message || JSON.stringify(json)}`);
    }
    return json.value;
  }

  async newSession(udid) {
    try {
      const v = await this.call('POST', '/session', {
        capabilities: {
          alwaysMatch: {
            platformName: 'iOS',
            'safari:useSimulator': true,
            'safari:deviceUDID': udid,
          },
        },
      });
      this.sid = v.sessionId;
    } catch (err) {
      throw new Error(`${err.message}\n\n` +
        'If this names authorization or automation: run `safaridriver --enable` on the Mac (once), and in the\n' +
        'simulator turn on Settings > Safari > Advanced > Remote Automation (once per device), then retry.');
    }
  }

  s(route) { return `/session/${this.sid}${route}`; }
  navigate(url) { return this.call('POST', this.s('/url'), { url }); }
  exec(script, args = []) { return this.call('POST', this.s('/execute/sync'), { script, args }); }
  async find(css) {
    const v = await this.call('POST', this.s('/element'), { using: 'css selector', value: css });
    return Object.values(v)[0]; // { "element-6066-11e4-a52e-4f735466cecf": id }
  }
  click(el) { return this.call('POST', this.s(`/element/${el}/click`), {}); }
  type(el, text) { return this.call('POST', this.s(`/element/${el}/value`), { text }); }
  async screenshot(file) {
    const b64 = await this.call('GET', this.s('/screenshot'));
    fs.writeFileSync(file, Buffer.from(b64, 'base64'));
    return file;
  }
  async quit() { if (this.sid) await this.call('DELETE', this.s('')).catch(() => {}); }
}

// --- page measurements ---------------------------------------------------------
//
// Rects are relative to the LAYOUT viewport. The visual viewport sits inside
// it at (offsetLeft, offsetTop) with its own width/height, so "on screen with
// the keyboard up" means the rect lies within that box -- not within
// innerHeight, which iOS leaves untouched.
const MEASURE = `
  const r = (sel) => { const el = document.querySelector(sel); if (!el) return null;
    const b = el.getBoundingClientRect();
    return { top: Math.round(b.top), bottom: Math.round(b.bottom), left: Math.round(b.left), right: Math.round(b.right), h: Math.round(b.height), w: Math.round(b.width) }; };
  const vv = window.visualViewport;
  return {
    innerWidth, innerHeight,
    vv: vv ? { width: Math.round(vv.width), height: Math.round(vv.height), offsetTop: Math.round(vv.offsetTop), pageTop: Math.round(vv.pageTop), scale: vv.scale } : null,
    scrollY: Math.round(window.scrollY),
    docScrollWidth: document.documentElement.scrollWidth,
    textarea: r('.chat__composer textarea'),
    send: r('.chat__composer .btn--primary'),
    bar: r('.chat__bar'),
    nav: r('#bottom-nav'),
    focused: document.activeElement && document.activeElement.tagName,
  };
`;

function inVisual(rect, m) {
  if (!rect || !m.vv) return false;
  return rect.top >= m.vv.offsetTop && rect.bottom <= m.vv.offsetTop + m.vv.height;
}

function describe(label, m) {
  const vv = m.vv ? `${m.vv.width}x${m.vv.height} @top ${m.vv.offsetTop} (pageTop ${m.vv.pageTop})` : 'n/a';
  console.log(`[ios] ${label}: layout ${m.innerWidth}x${m.innerHeight}, visual ${vv}, scrollY ${m.scrollY}, focused ${m.focused}`);
  for (const k of ['bar', 'textarea', 'send', 'nav']) {
    const r = m[k];
    console.log(`[ios]   ${k.padEnd(8)} ${r ? `top ${r.top} bottom ${r.bottom} (${r.w}x${r.h})` : 'missing'}${r ? (inVisual(r, m) ? '  in visual viewport' : '  OUTSIDE visual viewport') : ''}`);
  }
}

// --- main ---------------------------------------------------------------------

async function main() {
  fs.mkdirSync(SHOTS, { recursive: true });
  const udid = findDeviceUdid();
  bootSimulator(udid);
  const driverProc = await startDriver();
  const d = new Driver(PORT);
  const suite = new Suite('ios-sim');
  let before = null;
  let after = null;

  try {
    await suite.check('a WebDriver session opens on the simulator', async () => {
      await d.newSession(udid);
    });
    if (!d.sid) throw new Error('no session -- nothing below can run');

    await suite.check('the chat page loads with its composer', async () => {
      await d.navigate(`${BASE}/#/chat`);
      await waitFor(() => d.exec(`return !!document.querySelector(${JSON.stringify(COMPOSER)});`),
        { timeout: 20000, message: 'composer never appeared' });
      before = await d.exec(MEASURE);
      describe('before keyboard', before);
      console.log(`[ios] screenshot ${await d.screenshot(path.join(SHOTS, 'before-keyboard.png'))}`);
      assert(before.textarea && before.send, 'composer field or Send missing');
    });

    await suite.check('tapping the field raises the keyboard (visual viewport shrinks)', async () => {
      if (!REAL_DEVICE) skip(NO_KEYBOARD);
      const el = await d.find(COMPOSER);
      await d.click(el);
      // Send keys BEFORE waiting, not after. A WebDriver click alone did not
      // raise the keyboard on the first run (2026-09-06); WebDriver's Element
      // Send Keys goes through the driver's own input path and focuses the
      // element itself, which is the remaining candidate for something iOS
      // treats as a real gesture. Typing here is not the assertion -- the
      // viewport wait below still is; this only tries to provoke the keyboard.
      await d.type(el, 'keyboard up');
      // Condition-wait on the thing the keyboard changes; a fixed sleep would
      // either race the animation or pad every run.
      after = await waitFor(async () => {
        const m = await d.exec(MEASURE);
        return m.vv && m.vv.height < before.innerHeight - 100 ? m : null;
      }, { timeout: 6000, message: async () => {
        const m = await d.exec(MEASURE);
        describe('no keyboard', m);
        return 'visual viewport never shrank: the keyboard did not open even after Element Send Keys -- the remaining candidate is a real device';
      } });
      await sleep(300);   // let the field's autoGrow settle before measuring
      after = await d.exec(MEASURE);
      describe('after keyboard', after);
      console.log(`[ios] screenshot ${await d.screenshot(path.join(SHOTS, 'after-keyboard.png'))}`);
    });

    await suite.check('the message field is fully inside the visual viewport with the keyboard up', () => {
      if (!REAL_DEVICE) skip(NO_KEYBOARD);
      assert(after, 'no keyboard-up measurement');
      assert(inVisual(after.textarea, after), `field ${JSON.stringify(after.textarea)} vs visual ${JSON.stringify(after.vv)}`);
    });

    await suite.check('Send is fully inside the visual viewport with the keyboard up', () => {
      if (!REAL_DEVICE) skip(NO_KEYBOARD);
      assert(after, 'no keyboard-up measurement');
      assert(inVisual(after.send, after), `Send ${JSON.stringify(after.send)} vs visual ${JSON.stringify(after.vv)}`);
    });

    await suite.check('no horizontal scroll with the keyboard up', () => {
      if (!REAL_DEVICE) skip(NO_KEYBOARD);
      assert(after, 'no keyboard-up measurement');
      assert(after.docScrollWidth <= after.innerWidth + 1, `scrollWidth ${after.docScrollWidth} > innerWidth ${after.innerWidth}`);
    });

    await suite.check('report: what the keyboard did to the nav and the top bar (never fails)', () => {
      if (!after) return;
      const navIn = inVisual(after.nav, after);
      const barIn = inVisual(after.bar, after);
      console.log(`[ios] bottom nav ${navIn ? 'still visible' : 'off the visual viewport'}; top bar ${barIn ? 'still visible' : 'scrolled away'}; page scrolled ${after.scrollY - before.scrollY}px`);
      console.log('[ios] decision input: if the nav is visible AND the field is inside the visual viewport, hiding the nav on focus buys vertical space; if the nav is already off-screen, it costs nothing and buys nothing.');
    });
  } finally {
    await d.quit();
    driverProc.kill('SIGTERM');
    if (!KEEP) { try { simctl('shutdown', udid); } catch { /* already down */ } }
  }

  const failed = printSummary([suite]);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(`[ios] harness error: ${err.stack || err.message}`);
  process.exit(1);
});
