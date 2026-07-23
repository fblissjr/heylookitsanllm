// Minimal test runner + assertions. No external framework -- puppeteer-core is
// the only dependency. A Suite runs checks sequentially, records pass/fail with
// timing, and never lets one thrown check abort the rest of the suite.

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function assert(cond, message) {
  if (!cond) throw new Error(message || 'assertion failed');
}

export function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message || 'assertEqual'}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

// Poll `fn` until it returns a truthy value or the timeout elapses. Returns the
// truthy value. `fn` may be async. Errors thrown by `fn` are swallowed until the
// deadline (so "element not there yet" reads don't abort the wait).
export async function waitFor(fn, { timeout = 15000, interval = 100, message } = {}) {
  const start = Date.now();
  let lastErr = null;
  while (Date.now() - start < timeout) {
    try {
      const v = await fn();
      if (v) return v;
    } catch (err) {
      lastErr = err;
    }
    await sleep(interval);
  }
  const suffix = lastErr ? ` (last error: ${lastErr.message})` : '';
  throw new Error(`${message || 'waitFor'} timed out after ${timeout}ms${suffix}`);
}

// Prove an ABSENCE: that nothing -- or nothing MORE -- happened.
//
// This is the one sanctioned exception to the suite-wide "never sleep for
// something observable" rule (README.md): "no second request follows" has no
// observable condition to wait on, so a bounded quiet window IS the technique
// rather than a shortcut. Keeping it in one helper stops that exception from
// being re-argued (and re-tuned) at every site.
//
// `watch` is a { urls, stop } from the suite's request watcher. `atLeast`
// condition-waits for the requests that SHOULD fire before the quiet window
// opens (never sleep for those); `expect` is the total permitted once it
// closes, defaulting to exactly the ones waited for.
export async function proveQuiet(watch, { atLeast = 0, quiet = 800, expect, message } = {}) {
  const label = message || 'request count';
  if (atLeast > 0) {
    await waitFor(async () => watch.urls.length >= atLeast,
      { message: `${label}: expected at least ${atLeast} request(s)` });
  }
  await sleep(quiet);
  watch.stop();
  const want = expect ?? atLeast;
  const seen = watch.urls.length;
  assert(seen === want,
    `${label}: expected ${want} after a ${quiet}ms quiet window, saw ${seen}`
    + (seen ? ` (${watch.urls.join(', ')})` : ''));
}

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

export class Suite {
  constructor(name) {
    this.name = name;
    this.results = [];
  }

  async check(name, fn) {
    const start = Date.now();
    try {
      await fn();
      const ms = Date.now() - start;
      this.results.push({ name, ok: true, ms });
      console.log(`  ${GREEN}✓${RESET} ${name} ${DIM}(${ms}ms)${RESET}`);
    } catch (err) {
      const ms = Date.now() - start;
      this.results.push({ name, ok: false, ms, error: err });
      console.log(`  ${RED}✗ ${name}${RESET} ${DIM}(${ms}ms)${RESET}`);
      console.log(`    ${RED}${err.message}${RESET}`);
    }
  }

  get passed() { return this.results.filter((r) => r.ok).length; }
  get failed() { return this.results.filter((r) => !r.ok).length; }
}

export function printSummary(suites) {
  let passed = 0;
  let failed = 0;
  console.log('\n────────────────────────────────────────');
  for (const s of suites) {
    passed += s.passed;
    failed += s.failed;
    const mark = s.failed === 0 ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`;
    console.log(`${mark}  ${s.name}: ${s.passed}/${s.passed + s.failed}`);
  }
  console.log('────────────────────────────────────────');
  const total = passed + failed;
  const color = failed === 0 ? GREEN : RED;
  console.log(`${color}${passed}/${total} checks passed${RESET}\n`);
  if (failed > 0) {
    console.log(`${RED}Failures:${RESET}`);
    for (const s of suites) {
      for (const r of s.results.filter((x) => !x.ok)) {
        console.log(`  ${RED}✗${RESET} [${s.name}] ${r.name}\n    ${r.error.message}`);
      }
    }
    console.log('');
  }
  return failed;
}
