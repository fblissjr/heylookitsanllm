// What does one streamed reply cost the page? An opt-in INSTRUMENT, never part
// of a suite: it asserts nothing about the app, it measures, and it writes a
// record with its conditions (docs/project/plan_dark_mode.md, Phase 5).
//
//   bun run e2e:cost -- --arm base=../../frontend --arm new=/path/to/other/frontend
//   bun run e2e:cost -- --arm a=../../frontend --arm b=../../frontend     # a control pair
//
// It drives the REAL chat page (phone emulation, CPU throttled) against the
// render suite's stub store, lets that suite's drip server feed one fixed
// reply, and reads Chrome's own cumulative counters either side of the
// stream: task, script, layout and style time, and how many layouts and style
// recalcs ran. Same family as scripts/perf_ab.py, and the same rules:
//
// - An arm is a frontend TREE (a directory). One fresh Chrome per arm run, and
//   rounds alternate the arm order (A B, then B A), so drift in the machine
//   lands on both.
// - The first stream in each process is warmup and is dropped.
// - Machine load is RECORDED (the load average either side of a run), not
//   assumed.
// - The verdict is deliberately crude: two arms whose ranges overlap are
//   NOISE. Two arms holding the same tree are a control pair, and a control
//   pair that does not come out as noise means the run is contaminated and
//   nothing else in the record deserves a verdict.
//
// What it is NOT: a measurement of the phone. Chrome under a CPU throttle
// ranks suspects (does this arm paint more, lay out more, run more script);
// WebKit on the device decides whether a ranked suspect matters there.
//
// Config: --reps N (measured streams per arm run, default 3), --rounds N
// (default 2), --throttle N (CPU slowdown, default 4), --out FILE (default
// internal/claude/perf/paint_cost_<stamp>.json, unversioned), E2E_CHROME,
// E2E_COLOR_SCHEME (default dark here: the phone at night is the case that
// asked the question).

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(__dirname, '..', '..');

// The workload. A reply long enough that the tail re-render, the scroll pin
// and paragraph layout all run many times, delivered in small deltas faster
// than the painter paints -- the shape of a local model answering at length.
const CHUNK_CHARS = 6;
const DELAY_MS = 8;
const REPLY_SECTIONS = 6;
const END_MARK = 'paint-cost-end-mark';

const METRICS = ['TaskDuration', 'ScriptDuration', 'LayoutDuration', 'RecalcStyleDuration',
  'LayoutCount', 'RecalcStyleCount'];

function replyText(streamDoc) {
  const prose = 'A long paragraph of ordinary prose, the kind a model writes when it is asked '
    + 'to explain something at length: clause after clause, each one adding a little, so that '
    + 'the line breaker has a real paragraph to work on every time the tail is laid out again. ';
  const sections = [];
  for (let i = 0; i < REPLY_SECTIONS; i++) sections.push(streamDoc, prose.repeat(3).trim());
  return `${sections.join('\n\n')}\n\n${END_MARK}`;
}

// --- worker: one arm, one fresh Chrome ----------------------------------------

async function worker() {
  const reps = Number(process.env.PAINT_COST_REPS);
  const rate = Number(process.env.PAINT_COST_THROTTLE);
  // `?lib` is the switch that stops render.mjs running its suite on import.
  const render = await import('./render.mjs?lib');
  const { launchBrowser } = await import('./lib/browser.mjs');
  const text = replyText(render.STREAM_DOC);
  const { server, base, setDrip } = await render.serveV3();
  const browser = await launchBrowser();
  const runs = [];
  try {
    for (let rep = 0; rep <= reps; rep++) {
      setDrip({ text, chunkChars: CHUNK_CHARS, delayMs: DELAY_MS, tailPauseMs: 150 });
      const { page, pageErrors } = await render.openChat(browser, base, { mobile: true, dripGenerate: true });
      const cdp = await page.createCDPSession();
      await cdp.send('Performance.enable');
      await cdp.send('Emulation.setCPUThrottlingRate', { rate });
      const read = async () => Object.fromEntries(
        (await cdp.send('Performance.getMetrics')).metrics.map((m) => [m.name, m.value]));
      const load0 = os.loadavg()[0];
      const before = await read();
      const t0 = performance.now();
      await render.sendAndWait(page);
      const wall = (performance.now() - t0) / 1000;
      const after = await read();
      // A stream that did not render is not a measurement. Fail loudly.
      const landed = await page.evaluate((mark) => {
        const rows = [...document.querySelectorAll('.message .message-content')];
        return rows.length > 0 && rows[rows.length - 1].textContent.includes(mark);
      }, END_MARK);
      if (!landed) throw new Error(`rep ${rep}: the reply's last line never reached the page`);
      if (pageErrors.length) throw new Error(`rep ${rep}: page errors: ${pageErrors.join(' | ')}`);
      const row = { rep, warmup: rep === 0, wall_s: wall, load_before: load0, load_after: os.loadavg()[0] };
      for (const m of METRICS) row[m] = after[m] - before[m];
      runs.push(row);
      await page.close();
    }
    console.log(`PAINT_COST_RESULT ${JSON.stringify({ chrome: await browser.version(), chars: text.length, runs })}`);
  } finally {
    await browser.close();
    server.close();
  }
}

// --- parent: arms, rounds, record, report -------------------------------------

// One hash for a frontend tree: two arms with the same hash are a control pair.
function treeHash(root) {
  const h = crypto.createHash('sha256');
  const walk = (dir) => {
    for (const name of fs.readdirSync(dir).sort()) {
      const p = path.join(dir, name);
      if (fs.statSync(p).isDirectory()) walk(p);
      else h.update(path.relative(root, p)).update('\0').update(fs.readFileSync(p));
    }
  };
  walk(root);
  return h.digest('hex').slice(0, 16);
}

function parseArgs(argv) {
  const opts = { arms: [], reps: 3, rounds: 2, throttle: 4, out: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--arm') {
      const [name, dir] = argv[++i].split(/=(.*)/s);
      const root = path.resolve(dir || '');
      if (!name || !dir || !fs.existsSync(path.join(root, 'index.html'))) {
        throw new Error(`--arm wants name=<frontend dir holding index.html>, got ${argv[i]}`);
      }
      opts.arms.push({ name, root });
    } else if (a === '--reps') opts.reps = Number(argv[++i]);
    else if (a === '--rounds') opts.rounds = Number(argv[++i]);
    else if (a === '--throttle') opts.throttle = Number(argv[++i]);
    else if (a === '--out') opts.out = path.resolve(argv[++i]);
    else throw new Error(`unknown argument ${a}`);
  }
  if (opts.arms.length < 2) throw new Error('give at least two --arm name=dir (the same dir twice is a control pair)');
  if (new Set(opts.arms.map((a) => a.name)).size !== opts.arms.length) throw new Error('arm names must differ');
  return opts;
}

const git = (...args) => spawnSync('git', ['-C', REPO, ...args], { encoding: 'utf8' }).stdout.trim();

function runArm(arm, opts, scheme) {
  const proc = spawnSync(process.execPath, [fileURLToPath(import.meta.url), 'worker'], {
    encoding: 'utf8', cwd: __dirname, maxBuffer: 64 * 1024 * 1024,
    env: { ...process.env, E2E_V3_ROOT: arm.root, E2E_COLOR_SCHEME: scheme,
      PAINT_COST_REPS: String(opts.reps), PAINT_COST_THROTTLE: String(opts.throttle) },
  });
  const line = (proc.stdout || '').split('\n').find((l) => l.startsWith('PAINT_COST_RESULT '));
  if (proc.status !== 0 || !line) {
    throw new Error(`arm ${arm.name} failed (exit ${proc.status}):\n${proc.stderr || proc.stdout}`);
  }
  return JSON.parse(line.slice('PAINT_COST_RESULT '.length));
}

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

function report(record) {
  const by = (name, metric) => record.runs.filter((r) => r.arm === name && !r.warmup).map((r) => r[metric]);
  const base = record.arms[0];
  const fmt = (x, metric) => (metric.endsWith('Count') ? String(Math.round(x)) : `${(x * 1000).toFixed(0)}ms`);
  let contaminated = false;
  const lines = [];
  for (const arm of record.arms.slice(1)) {
    const control = arm.tree === base.tree;
    lines.push(`\n${base.name} vs ${arm.name}${control ? '  (same tree: a control pair)' : ''}`);
    for (const metric of [...METRICS, 'wall_s']) {
      const a = by(base.name, metric), b = by(arm.name, metric);
      const overlap = Math.min(...a) <= Math.max(...b) && Math.min(...b) <= Math.max(...a);
      const verdict = overlap ? 'NOISE' : `DIFFERS x${(median(b) / median(a)).toFixed(2)}`;
      if (control && !overlap) contaminated = true;
      const range = (xs) => `${fmt(Math.min(...xs), metric)}..${fmt(Math.max(...xs), metric)}`;
      lines.push(`  ${metric.padEnd(20)} ${range(a).padEnd(18)} ${range(b).padEnd(18)} ${verdict}`);
    }
  }
  if (contaminated) {
    lines.push('\nCONTAMINATED: a control pair differs, so no line above is a verdict. Quiet the machine and run again.');
  }
  return { text: lines.join('\n'), contaminated };
}

async function run(argv) {
  const opts = parseArgs(argv);
  const scheme = process.env.E2E_COLOR_SCHEME || 'dark';
  const record = {
    instrument: 'paint_cost', date: new Date().toISOString(),
    commit: git('rev-parse', 'HEAD'), dirty: Boolean(git('status', '--porcelain', '--', 'frontend', 'tests/e2e')),
    scheme, throttle: opts.throttle, reps: opts.reps, rounds: opts.rounds,
    workload: { chunk_chars: CHUNK_CHARS, delay_ms: DELAY_MS, sections: REPLY_SECTIONS, phone_emulation: true },
    arms: opts.arms.map((a) => ({ name: a.name, root: path.relative(REPO, a.root) || '.', tree: treeHash(a.root) })),
    runs: [],
  };
  for (let round = 0; round < opts.rounds; round++) {
    const order = round % 2 ? [...opts.arms].reverse() : opts.arms;
    for (const arm of order) {
      process.stderr.write(`round ${round + 1}/${opts.rounds}  ${arm.name} ...\n`);
      const res = runArm(arm, opts, scheme);
      record.chrome = res.chrome;
      record.workload.chars = res.chars;
      for (const r of res.runs) record.runs.push({ arm: arm.name, round, ...r });
    }
  }
  const { text, contaminated } = report(record);
  record.contaminated = contaminated;
  const stamp = record.date.replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
  const out = opts.out || path.join(REPO, 'internal', 'claude', 'perf', `paint_cost_${stamp}.json`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, `${JSON.stringify(record, null, 1)}\n`);
  console.log(text);
  console.log(`\nrecord: ${path.relative(process.cwd(), out)}`);
  process.exit(contaminated ? 3 : 0);
}

const [mode, ...rest] = process.argv.slice(2);
const main = mode === 'worker' ? worker() : mode === 'run' ? run(rest)
  : Promise.reject(new Error('usage: node paint_cost.mjs run --arm name=dir --arm name=dir [--reps N] [--rounds N] [--throttle N] [--out FILE]'));
main.catch((err) => { console.error(err.message || err); process.exit(1); });
