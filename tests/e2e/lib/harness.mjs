// Minimal test runner + assertions. No framework and no dependencies. A Suite
// runs checks sequentially, records pass/fail with timing, and never lets one
// thrown check abort the rest of the suite.

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function assert(cond, message) {
  if (!cond) throw new Error(message || 'assertion failed');
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
  // `message` may be a function, evaluated only on failure, so a check can
  // report the state it actually found (one page query is cheap once, and
  // pointless on every passing run). Never let that reporting mask the
  // failure it is describing.
  let text = message;
  if (typeof message === 'function') {
    try { text = await message(); } catch (err) { text = `${message.name || 'message'}() threw: ${err.message}`; }
  }
  throw new Error(`${text || 'waitFor'} timed out after ${timeout}ms${suffix}`);
}

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

// A check that reaches a LEGAL early exit before its assertion (the stop
// landed after the thinking block closed, so the resume half never ran)
// reports SKIP, not PASS. A green tally used to hide exactly that half:
// the v1.79.64 mlx-vlm continuation crash sat behind a 48/48.
export class SkipCheck extends Error {}
export function skip(reason) { throw new SkipCheck(reason); }

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
      if (err instanceof SkipCheck) {
        this.results.push({ name, ok: true, skipped: true, ms });
        console.log(`  ${YELLOW}- ${name} SKIPPED${RESET} ${DIM}(${ms}ms)${RESET}\n    ${YELLOW}${err.message}${RESET}`);
        return;
      }
      this.results.push({ name, ok: false, ms, error: err });
      console.log(`  ${RED}✗ ${name}${RESET} ${DIM}(${ms}ms)${RESET}`);
      // A message-less error (a protocol error, an assert without text)
      // printed a blank line and left nothing to chase; show the top of the
      // stack instead.
      const detail = err.message?.trim()
        ? err.message
        : `${err.name ?? 'Error'} (no message)\n${(err.stack ?? '').split('\n').slice(0, 4).join('\n')}`;
      console.log(`    ${RED}${detail}${RESET}`);
    }
  }

  get passed() { return this.results.filter((r) => r.ok && !r.skipped).length; }
  get failed() { return this.results.filter((r) => !r.ok).length; }
  get skipped() { return this.results.filter((r) => r.skipped).length; }
}

export function printSummary(suites) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  console.log('\n────────────────────────────────────────');
  for (const s of suites) {
    passed += s.passed;
    failed += s.failed;
    skipped += s.skipped;
    const mark = s.failed === 0 ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`;
    const skipNote = s.skipped ? ` ${YELLOW}(${s.skipped} skipped)${RESET}` : '';
    console.log(`${mark}  ${s.name}: ${s.passed}/${s.passed + s.failed}${skipNote}`);
  }
  console.log('────────────────────────────────────────');
  const total = passed + failed;
  const color = failed === 0 ? GREEN : RED;
  const skipTotal = skipped ? ` ${YELLOW}(${skipped} skipped: did not reach their assertion)${RESET}` : '';
  console.log(`${color}${passed}/${total} checks passed${RESET}${skipTotal}\n`);
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
