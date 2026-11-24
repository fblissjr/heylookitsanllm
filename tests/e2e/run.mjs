// Browser checks for the frontend, against a real and ISOLATED server.
//
//   cd tests/e2e && node run.mjs          (or: bun run e2e)
//
// It starts its own heylookllm (lib/server.mjs: temp DB, access gate isolated,
// loopback only, never the daily server), drives headless system Chrome over
// the DevTools protocol (lib/cdp.mjs, no dependencies), and tears both down.
// Must run UNSANDBOXED: Metal, a local port and Chrome.
//
//   E2E_MODEL    model to load and drive   (default: Qwen3.5-0.8B-MLX-8bit)
//   E2E_PORT     server port                (default: 1264, never the daily 8000)
//   E2E_CHROME   Chrome binary              (default: the macOS app path)
//   E2E_HEADFUL  set to watch the run in a window
//
// How a check is written here:
//   - It drives and reads the page with page-side JavaScript (page.js), and
//     asks the SERVER what landed (api) rather than trusting what is painted.
//   - It waits for a condition (page.waitFor, loaded), never for time. The one
//     exception is proving that nothing more happened, which has no condition.
//   - A response is held or failed from inside the page (sendAndHold),
//     so a race is staged in order instead of hoped for.
//   - Chrome is not WebKit. These checks hold DOM, state, request and CSS-rule
//     behaviour; they say nothing about Safari's scrolling or the iOS keyboard.

import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Suite, printSummary, assert, skip, sleep } from './lib/harness.mjs';
import { startServer } from './lib/server.mjs';
import { launchChrome } from './lib/cdp.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const MODEL = process.env.E2E_MODEL || 'Qwen3.5-0.8B-MLX-8bit';
const PORT = Number(process.env.E2E_PORT || 1264);

// Short replies: the checks are about the page, not the model.
const PARAMS = { max_tokens: 24, enable_thinking: false };
const PNG_1PX = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

// --- page helpers: each is one thing a person does or sees ---------------------

const composerValue = (page) => page.js(() => document.querySelector('.chat__composer textarea').value);
const statusText = (page) => page.js(() => document.querySelector('.chat__status').textContent);

const typeComposer = (page, text) => page.js((t) => {
  const box = document.querySelector('.chat__composer textarea');
  box.value = t;
  box.dispatchEvent(new Event('input', { bubbles: true }));
}, text);

const clickSend = (page) => page.js(() => document.querySelector('.chat__composer .btn--primary').click());

// Click the first element matching `selector` whose trimmed text is `text`.
const clickText = (page, selector, text) => page.js((sel, t) => {
  const el = [...document.querySelectorAll(sel)].find((e) => e.textContent.trim() === t);
  if (!el) throw new Error(`no ${sel} reading "${t}"`);
  el.click();
}, selector, text);

// Go to a route and wait until its setup has RESOLVED (app.js marks #main), or
// until the router has painted a mount failure. A page builds its skeleton
// before its first await, so "the skeleton is there" says nothing about a
// setup that fails later, and the next navigation would cut that failure off.
async function route(page, name) {
  await page.js((n) => { location.hash = `#/${n}`; }, name);
  await page.waitFor((n) => document.getElementById('main').dataset.mounted === n
    || Boolean(document.querySelector('#main > .error-note')),
  `the ${name} page to finish mounting`, { args: [name] });
}

// Open a conversation and wait until its BODY has been adopted, not only until
// its row is highlighted: the row moves first and the rows arrive a request
// later, and a check that acts in between acts on the previous conversation.
async function selectConv(page, conv) {
  await page.waitFor((t) => [...document.querySelectorAll('.conv-item__title')].some((e) => e.textContent === t),
    `"${conv.title}" in the conversation list`, { args: [conv.title] });
  const bodyUrl = `/v1/conversations/${conv.id}`;
  const bodyRequests = (from) => page.requests.slice(from).filter((r) => r.method === 'GET' && r.url.endsWith(bodyUrl));
  const deadline = Date.now() + 20000;
  const already = await page.js((t) => document.querySelector('.conv-item--active .conv-item__title')?.textContent === t, conv.title);
  if (already) {
    // Re-clicking the open conversation does not refetch, so there is no new
    // request to wait for. But the row highlights before its body arrives, so
    // "highlighted" may still mean "loading": wait out any body request in flight.
    while (bodyRequests(0).some((r) => !r.done)) {
      assert(Date.now() < deadline, `the body of "${conv.title}" never finished loading`);
      await sleep(50);
    }
  } else {
    const mark = page.requests.length;
    await clickText(page, '.conv-item__title', conv.title);
    for (;;) {
      const req = bodyRequests(mark)[0];
      assert(!req?.failed, `the body request for "${conv.title}" failed or was cancelled`);
      if (req?.done) break;
      assert(Date.now() < deadline, `the body of "${conv.title}" never finished loading`);
      await sleep(50);
    }
  }
  // one turn of the page's own loop, so the adoption that follows the response has run
  await page.js(() => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0))));
}

// Hold the next message POST until the check decides its fate. Installed once
// per document; `release(fail)` either lets the real request go or rejects it
// the way a dropped connection does.
const installHold = (page) => page.js(() => {
  const real = window.fetch.bind(window);
  window.__hold = { armed: false, release: null };
  window.fetch = (url, init) => {
    if (window.__hold.armed && /\/messages$/.test(String(url)) && init?.method === 'POST') {
      window.__hold.armed = false;
      return new Promise((resolve, reject) => {
        window.__hold.release = (fail) => (fail
          ? reject(new TypeError('network down'))
          : real(url, init).then(resolve, reject));
      });
    }
    return real(url, init);
  };
});

async function sendAndHold(page, text) {
  await typeComposer(page, text);
  await page.js(() => { window.__hold.release = null; window.__hold.armed = true; });
  await clickSend(page);
  await page.waitFor(() => typeof window.__hold.release === 'function', 'the message POST to be held');
}

const release = (page, { fail }) => page.js((f) => {
  const go = window.__hold.release;
  window.__hold.release = null;
  go(f);
}, fail);

// A check that sends for real says so, in case an earlier one left the hold armed.
const disarmHold = (page) => page.js(() => {
  if (window.__hold) { window.__hold.armed = false; window.__hold.release = null; }
});

// --- the run ------------------------------------------------------------------

async function main() {
  const tmp = mkdtempSync(path.join(os.tmpdir(), 'heylook-e2e-'));
  const logPath = path.join(tmp, 'server.log');
  const suite = new Suite('browser');
  let server = null;
  let chrome = null;
  let fatal = null;

  try {
    console.log(`\nstarting an isolated server on :${PORT} with ${MODEL} (log: ${logPath})`);
    server = await startServer({
      port: PORT, dbPath: path.join(tmp, 'db.duckdb'), modelId: MODEL, repoRoot: ROOT, logPath,
    });
    const base = server.base;
    const api = async (method, urlPath, body) => {
      const res = await fetch(base + urlPath, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
      });
      const json = await res.json().catch(() => null);
      assert(res.ok, `${method} ${urlPath} -> ${res.status} ${JSON.stringify(json)}`);
      return json;
    };

    // Fixtures through the real API. Created oldest first; the list shows the
    // newest first, so A is the conversation the page opens on.
    const convC = await api('POST', '/v1/conversations', { title: 'C image', model_id: MODEL, params: PARAMS });
    await api('POST', `/v1/conversations/${convC.id}/messages`, {
      role: 'user',
      content: [
        { type: 'text', text: 'what is this' },
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: PNG_1PX } },
      ],
    });
    const convB = await api('POST', '/v1/conversations', { title: 'B check', model_id: MODEL, params: PARAMS });
    const convD = await api('POST', '/v1/conversations', {
      title: 'D long', model_id: MODEL, params: { ...PARAMS, max_tokens: 200 },
    });
    const convA = await api('POST', '/v1/conversations', { title: 'A check', model_id: MODEL, params: PARAMS });
    await api('POST', `/v1/conversations/${convA.id}/messages`, { role: 'user', content: 'first in A' });

    chrome = await launchChrome();
    console.log(`driving ${chrome.version}\n`);
    const page = await chrome.newPage();
    await page.goto(`${base}/#/chat`);
    await page.waitFor(() => document.getElementById('main').dataset.mounted === 'chat', 'the chat page to finish mounting');

    await suite.check('every route mounts with no page error', async () => {
      for (const name of ['notebook', 'models', 'perf', 'presets', 'embeddings', 'chat']) {
        await route(page, name);
        const failed = await page.js(() => document.querySelector('#main > .error-note')?.textContent ?? null);
        assert(!failed, `${name}: ${failed}`);
      }
      assert(page.errors.length === 0, `page errors: ${page.errors.join(' | ')}`);
    });

    // api.js split a call's arguments by the builder's arity, a default
    // parameter read as arity 0, and this tab was sent the events stream.
    await suite.check('Perf: the metrics tab requests the metrics stream', async () => {
      await route(page, 'perf');
      const mark = page.requests.length;
      await page.js(() => document.querySelector('button[data-stream="metrics"]').click());
      const logsRequests = () => page.requests.slice(mark).filter((r) => r.url.includes('/v1/admin/config/logs'));
      const deadline = Date.now() + 10000;
      while (!logsRequests().length && Date.now() < deadline) await sleep(50);
      const seen = logsRequests().map((r) => r.url.replace(base, ''));
      assert(seen.length > 0, 'the tab made no logs request');
      assert(seen.every((u) => u.includes('stream=metrics')), `requested ${seen.join(', ')}`);
    });

    // Every speed claim of the streaming renderer rests on ONE claim: the cut
    // MarkdownStream chooses is a place no markdown construct can span. That is
    // a property, not a list of examples, so it is checked as one: generated
    // documents grown a chunk at a time, the way a stream arrives, must render
    // exactly as the whole document does. Same technique and same reason as
    // the backend's TestParserInvariants. This is also the check to run after a
    // vendored marked or DOMPurify update (scripts/vendor_frontend.py).
    await suite.check('an incrementally grown document renders identically to a whole one', async () => {
      const bad = await page.js(async () => {
        const { MarkdownStream } = await import('/js/markdown-stream.js');
        const { renderMarkdown } = await import('/js/markdown.js');

        let seed = 20260826;
        const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
        const pick = (a) => a[Math.floor(rnd() * a.length)];
        const W = ('the model returns a value when parsing tokens across layers and caches '
          + 'inputs for later reuse in generation which is why latency matters here').split(' ');
        const words = (n) => Array.from({ length: n }, () => pick(W)).join(' ');

        // Every block shape that could plausibly straddle a split.
        const BLOCKS = [
          () => words(25) + '.',
          () => '## ' + words(4),
          () => '### ' + words(3),
          () => '- ' + words(5) + '\n- ' + words(4) + '\n- ' + words(6),
          () => '1. ' + words(5) + '\n2. ' + words(4),
          () => '- ' + words(5) + '\n\n- ' + words(4),                  // LOOSE list across a blank line
          () => '- ' + words(4) + '\n\n  continued paragraph in the item', // indented continuation
          () => '```js\nconst x = 1;\n\nfunction f() { return x; }\n```', // blank line INSIDE a fence
          () => '~~~\nplain ~~ fence body\n~~~',
          () => '````\n```\nnested ticks\n```\n````',                    // longer fence wins
          () => '```\nunclosed fence body\n' + words(6),                 // never closed
          () => '> ' + words(8) + '\n> ' + words(6),
          () => '> ' + words(5) + '\n\n> ' + words(5),                   // adjacent quotes
          () => '    indented code one\n    indented code two',
          () => '    code a\n\n    code b',                             // one code block across a blank line
          () => '| a | b |\n| --- | --- |\n| ' + words(2) + ' | ' + words(2) + ' |',
          () => '---',
          () => words(4) + '\n' + '='.repeat(5),                        // setext h1
          () => words(4) + '\n' + '-'.repeat(5),                        // setext h2
          () => words(8) + ' `inline` and **bold** and *em*.',
          () => '<div>literal html</div>',
          () => 'A line\nwith a soft break.',                           // breaks: true
          () => '[a link](https://example.com) mid-paragraph ' + words(6),
        ];
        // Reference and footnote definitions reach forward arbitrarily far, so
        // no split is safe once one appears: the fallback must keep these
        // CORRECT, which is the point of including them. The HTML blocks of
        // CommonMark types 1-5 end on a closing condition, not on a blank
        // line, so each renders as one token whole and as two if cut at its
        // inner blank line; a committed prefix is never revisited, so that
        // seam would be permanent.
        const UNSAFE = [
          () => '[ref]: https://example.com\n\nSee [the doc][ref] for more.',
          () => 'Text with a note[^1].\n\n[^1]: the footnote body.',
          () => '<pre>\nfoo\n\nbar\n</pre>',
          () => '<script>\nvar a = 1;\n\nvar b = 2;\n</script>',
          () => '<style>\n.a { color: red }\n\n.b { color: blue }\n</style>',
          () => '<textarea>\nline one\n\nline two\n</textarea>',
          () => '<!--\na comment\n\nstill the comment\n-->',
          () => '<?instruction\none\n\ntwo\n?>',
          () => '<!ELEMENT br\n\nEMPTY>',
          () => '<![CDATA[\nraw\n\nmore raw\n]]>',
        ];

        const CHUNKS = [1, 5, 23, 97];
        for (let d = 0; d < 60; d++) {
          const n = 4 + Math.floor(rnd() * 6);
          const parts = Array.from({ length: n }, () => pick(BLOCKS)());
          // Taken IN TURN, not at random: picked at random, two of these shapes
          // were never generated at all, and the rule that makes each of them
          // unsafe could be deleted with the check still green.
          if (d % 5 === 0) parts.splice(Math.floor(rnd() * parts.length), 0, UNSAFE[(d / 5) % UNSAFE.length]());
          const doc = parts.join('\n\n');

          const ref = document.createElement('div');
          ref.innerHTML = renderMarkdown(doc);
          const want = ref.innerHTML;

          for (const step of CHUNKS) {
            const el = document.createElement('div');
            const ms = new MarkdownStream(el);
            try {
              for (let i = step; i < doc.length; i += step) ms.render(doc.slice(0, i));
              ms.render(doc);
            } catch (err) {
              return { doc, step, threw: String(err && err.message) };
            }
            if (el.innerHTML !== want) return { doc, step, got: el.innerHTML, want };
          }
        }
        return null;
      });
      if (bad) {
        const detail = bad.threw
          ? `threw: ${bad.threw}`
          : `\n  got:  ${JSON.stringify(String(bad.got).slice(0, 500))}\n  want: ${JSON.stringify(String(bad.want).slice(0, 500))}`;
        assert(false,
          `growing this document ${bad.step} chars at a time diverged from a whole-document render.\n`
          + `  source: ${JSON.stringify(bad.doc.slice(0, 500))}\n  ${detail}`);
      }
    });

    // One giant block (no blank line anywhere) can never advance the boundary.
    // It must stay CORRECT: the paint rate is what bounds its cost.
    await suite.check('a document with no safe split still renders correctly', async () => {
      const ok = await page.js(async () => {
        const { MarkdownStream } = await import('/js/markdown-stream.js');
        const { renderMarkdown } = await import('/js/markdown.js');
        const doc = 'one enormous paragraph with no blank line anywhere in it '.repeat(40).trim();
        const el = document.createElement('div');
        const ms = new MarkdownStream(el);
        for (let i = 7; i < doc.length; i += 7) ms.render(doc.slice(0, i));
        ms.render(doc);
        const ref = document.createElement('div');
        ref.innerHTML = renderMarkdown(doc);
        return el.innerHTML === ref.innerHTML;
      });
      assert(ok, 'a split-free document did not survive incremental rendering');
    });

    // renderMarkdown is the only text -> HTML path. These three hold what it
    // must do with raw HTML and with URLs, and they are the only checks that
    // can see a vendored marked or DOMPurify update break either: the
    // incremental-vs-whole diff above is blind to it, because both of its arms
    // would break the same way.
    const render = (text) => page.js(async (t) => {
      const { renderMarkdown } = await import('/js/markdown.js');
      return renderMarkdown(t);
    }, text);

    // What a model writes is shown, not interpreted: Copy reads the stored
    // text, so a tag that renders (or silently disappears) disagrees with it.
    await suite.check('raw HTML in a reply is shown as text, and code and autolinks are untouched', async () => {
      const cases = [
        ['an unknown tag', 'Use the <d>tag</d> here.', '&lt;d&gt;tag&lt;/d&gt;'],
        ['an allowlisted tag', 'bold <b>works</b>', '&lt;b&gt;works&lt;/b&gt;'],
        ['angle-bracketed prose', 'math a <b and c> d', 'math a &lt;b and c&gt; d'],
        ['fenced code', '```\n<d>fenced</d>\n```', '<pre><code>&lt;d&gt;fenced&lt;/d&gt;'],
        ['inline code', 'inline `<d>` code', '<code>&lt;d&gt;</code>'],
        ['an autolink', '<https://example.com>', '<a href="https://example.com">'],
      ];
      for (const [name, src, want] of cases) {
        const html = await render(src);
        assert(html.includes(want), `${name} rendered as ${JSON.stringify(html)}`);
      }
    });

    // The claim is a property: no markdown spelling yields a live href or src
    // to a scheme outside the allowlist. marked emits `javascript:` hrefs
    // unfiltered, so a broken guard produces a page that renders and diffs
    // identically while being live XSS.
    //
    // ASSERT ON WHAT THE BROWSER RESOLVES, never on the HTML text. The exploit
    // emits `href="javascript&colon;..."`, which holds no literal colon, so a
    // regex for `javascript:` passes on the vulnerable output. Parsing the
    // attribute and reading `protocol` is the only oracle that agrees with
    // what a click would do.
    await suite.check('no markdown spelling renders a dangerous URL', async () => {
      const vectors = [
        ['inline link',     '[x](javascript:alert(1))'],
        ['mixed case',      '[x](JaVaScRiPt:alert(1))'],
        ['entity-encoded',  '[x](java&#115;cript:alert(1))'],
        // The colon itself as an entity: no literal colon at all, so a guard
        // that looks for one takes its "relative, therefore safe" return.
        ['entity colon',    '[x](javascript&colon;alert(1))'],
        ['numeric colon',   '[x](javascript&#58;alert(1))'],
        ['hex colon',       '[x](javascript&#x3a;alert(1))'],
        ['padded colon',    '[x](javascript&#0000058;alert(1))'],
        ['no-semicolon',    '[x](javascript&#58alert(1))'],
        ['entity newline',  '[x](javascript&NewLine;&colon;alert(1))'],
        ['entity tab',      '[x](jav&Tab;ascript&colon;alert(1))'],
        ['entity ref link', '[x][e]\n\n[e]: javascript&colon;alert(1)'],
        ['entity angle',    '[x](<javascript&colon;alert(1)>)'],
        ['entity img data', '![a](data&colon;text/html,hi)'],
        ['entity img svg',  '![a](data&colon;image/svg+xml,<svg onload=alert(1)>)'],
        ['percent-encoded', '[x](java%73cript:alert(1))'],
        ['leading space',   '[x]( javascript:alert(1))'],
        ['vbscript',        '[x](vbscript:alert(1))'],
        ['data html',       '[x](data:text/html,<script>alert(1)</script>)'],
        ['autolink',        '<javascript:alert(1)>'],
        ['bare autolink',   'go javascript:alert(1) now'],
        ['reference link',  '[x][r]\n\n[r]: javascript:alert(1)'],
        ['image',           '![x](javascript:alert(1))'],
        ['image data svg',  '![x](data:image/svg+xml,<svg onload=alert(1)>)'],
      ];
      const SAFE = new Set(['http:', 'https:', 'mailto:', 'tel:']);
      for (const [name, src] of vectors) {
        const protocols = await page.js(async (text) => {
          const { renderMarkdown } = await import('/js/markdown.js');
          const host = document.createElement('div');
          host.innerHTML = renderMarkdown(text);
          return [...host.querySelectorAll('[href],[src]')].map((el) => {
            try { return new URL(el.getAttribute('href') ?? el.getAttribute('src'), location.href).protocol; }
            catch { return 'unparseable:'; }
          });
        }, src);
        const bad = protocols.filter((p) => !SAFE.has(p));
        assert(bad.length === 0, `${name} produced a live ${bad.join(',')} URL (as the browser resolves it)`);
      }
    });

    // The SHAPE of the refusal, which is the only thing that can see the
    // guard in markdown.js itself. The protocol check above cannot: with the
    // guard deleted, DOMPurify still strips the href and the protocol comes
    // back clean. The two layers refuse differently: the guard drops the
    // anchor and keeps the label, DOMPurify keeps the anchor and strips the
    // attribute. "No anchor at all" is true only of the guard. The last table
    // is the other direction: a guard that refuses everything is also broken.
    await suite.check('a refused URL leaves no anchor, and ordinary links still render', async () => {
      const anchors = await page.js(async (text) => {
        const { renderMarkdown } = await import('/js/markdown.js');
        const host = document.createElement('div');
        host.innerHTML = renderMarkdown(text);
        return host.querySelectorAll('a').length;
      }, '[x](javascript:alert(1))');
      assert(anchors === 0, `a refused link left ${anchors} anchor(s): DOMPurify caught it, the guard did not`);
      const cases = [
        ['relative',    '[x](/a/b)',                   'href="/a/b"'],
        ['anchor',      '[x](#sec)',                   'href="#sec"'],
        ['https',       '[x](https://ok.test/p)',      'href="https://ok.test/p"'],
        ['mailto',      '[x](mailto:a@b.test)',        'href="mailto:a@b.test"'],
        ['tel',         '[x](tel:+15551234)',          'href="tel:+15551234"'],
        ['image',       '![x](https://ok.test/a.png)', 'src="https://ok.test/a.png"'],
        // relative URLs carrying an & and a later colon: a timestamp in a
        // query string was once enough to drop the link
        ['query+colon', '[x](/search?q=foo&t=1:30)',   'href="/search?q=foo&amp;t=1:30"'],
        ['frag+colon',  '[x](?a=1&b=2#sec:tion)',      'href="?a=1&amp;b=2#sec:tion"'],
      ];
      for (const [name, src, want] of cases) {
        const html = await render(src);
        assert(html.includes(want), `${name} stopped rendering (guard too strict?): ${JSON.stringify(html)}`);
      }
    });

    await route(page, 'chat');

    // The drawer's Escape listener runs before the page's, and closes the
    // drawer before the page asks "is a modal open". Unconsumed, one Escape
    // closed the drawer AND ran the page's own Escape action.
    await suite.check('Escape closes the drawer and does nothing else', async () => {
      await selectConv(page, convA);
      await page.js(() => document.querySelector('button[aria-label="Preview prompt"]').click());
      await page.waitFor(() => document.querySelector('.chat__prompt-preview').hidden === false, 'the prompt preview to open');
      await page.js(() => document.querySelector('.chat__settings-btn').click());
      await page.waitFor(() => Boolean(document.querySelector('.drawer--open')), 'the drawer to open');
      await page.press('Escape');
      await page.waitFor(() => !document.querySelector('.drawer--open'), 'the drawer to close');
      assert(await page.js(() => document.querySelector('.chat__prompt-preview').hidden === false),
        'the Escape that closed the drawer also closed the prompt preview');
    });

    await suite.check('the prompt preview closes when the conversation changes', async () => {
      await selectConv(page, convA);
      if (await page.js(() => document.querySelector('.chat__prompt-preview').hidden)) {
        await page.js(() => document.querySelector('button[aria-label="Preview prompt"]').click());
      }
      await page.waitFor(() => document.querySelector('.chat__prompt-preview').hidden === false, 'the prompt preview to open');
      await selectConv(page, convB);
      assert(await page.js(() => document.querySelector('.chat__prompt-preview').hidden === true),
        'the preview of A is still open in B');
    });

    await installHold(page);

    // A send's failure path used to restore its text and attachments into the
    // composer ON SCREEN, which after a switch is another conversation's.
    await suite.check('a send that fails after a switch returns to its own draft', async () => {
      await selectConv(page, convA);
      await sendAndHold(page, 'held text A');
      await selectConv(page, convB);
      await release(page, { fail: true });
      await page.waitFor(() => /Send failed in the conversation you left/.test(document.querySelector('.chat__status').textContent),
        'the line that says where the failed send went');
      assert((await composerValue(page)) === '', `B's composer holds ${JSON.stringify(await composerValue(page))}`);
      await selectConv(page, convA);
      assert((await composerValue(page)) === 'held text A', `A's composer holds ${JSON.stringify(await composerValue(page))}`);
    });

    await suite.check('a send that lands after a switch says no reply was requested', async () => {
      await selectConv(page, convA);
      await sendAndHold(page, 'saved text A');
      await selectConv(page, convB);
      await release(page, { fail: false });
      await page.waitFor(() => /no reply was requested/.test(document.querySelector('.chat__status').textContent),
        'the line that says no reply was requested');
      const a = await api('GET', `/v1/conversations/${convA.id}`);
      assert(a.messages.some((m) => m.content === 'saved text A'), 'the message did not reach conversation A');
    });

    await suite.check('a send streams a reply that persists', async () => {
      await selectConv(page, convB);
      await disarmHold(page);
      await typeComposer(page, 'Reply with the single word ok.');
      await clickSend(page);
      await page.waitFor(() => document.querySelectorAll('.message--assistant').length >= 1, 'an assistant row', { timeout: 90000 });
      await page.waitFor(() => !document.querySelector('.message--streaming'), 'the stream to finish', { timeout: 90000 });
      const b = await api('GET', `/v1/conversations/${convB.id}`);
      assert(b.messages.some((m) => m.role === 'assistant'), 'the store holds no assistant row');
      assert(page.errors.length === 0, `page errors: ${page.errors.join(' | ')}`);
    });

    // The dot read the open conversation's list row, and that row is only as
    // fresh as the last list fetch. So: refetch the list WHILE a reply streams
    // (what a phone does when the tab comes back), which returns the row as
    // generating, then let the reply finish. The dot must go out. Before the
    // fix it stayed on until the next resume, because nothing told the row.
    await suite.check('the nav dot goes out after a run, even when the list was refetched mid-run', async () => {
      await selectConv(page, convD);
      await disarmHold(page);
      await typeComposer(page, 'Count from 1 to 80, one number per line. Do not stop early.');
      await clickSend(page);
      await page.waitFor(() => Boolean(document.querySelector('.message--streaming')), 'the reply to start streaming', { timeout: 90000 });
      const mark = page.requests.length;
      await page.js(() => { document.dispatchEvent(new Event('visibilitychange')); });
      const listRefetched = () => page.requests.slice(mark)
        .some((r) => r.method === 'GET' && r.url.endsWith('/v1/conversations') && r.done && !r.failed);
      const deadline = Date.now() + 10000;
      while (!listRefetched() && Date.now() < deadline) await sleep(25);
      assert(listRefetched(), 'the resume did not refetch the conversation list');
      const during = await page.js(() => ({
        streaming: Boolean(document.querySelector('.message--streaming')),
        lit: Boolean(document.querySelector('.nav-item--busy')),
      }));
      const row = (await api('GET', '/v1/conversations')).conversations.find((c) => c.id === convD.id);
      if (!during.streaming || !row?.generating) {
        skip('the reply finished before the list was refetched mid-run, so the stale row was never staged');
      }
      assert(during.lit, 'the nav dot was not lit while a reply was streaming');
      await page.waitFor(() => !document.querySelector('.message--streaming'), 'the stream to finish', { timeout: 90000 });
      await page.waitFor(() => !document.querySelector('.nav-item--busy'), 'the nav dot to go out', { timeout: 5000 });
    });

    await suite.check('a double tap on New creates one conversation', async () => {
      const before = (await api('GET', '/v1/conversations')).conversations.length;
      await page.js(() => {
        const btn = [...document.querySelectorAll('.chat__convs-head button')].find((e) => e.textContent.trim() === 'New');
        btn.click();
        btn.click();
      });
      await page.waitFor((n) => document.querySelectorAll('.conv-item').length > n, 'the new conversation in the list', { args: [before] });
      await sleep(800); // absence: a second create has no condition to wait on
      const after = (await api('GET', '/v1/conversations')).conversations.length;
      assert(after === before + 1, `conversations went from ${before} to ${after}`);
    });

    // Kept media is matched by what it is, not by which object holds it. The
    // editor's entries can outlive the row objects they were built from (a
    // resync replaces every row object, and a later rebuild carries the old
    // entries across). Matched by identity, nothing was found, the save went
    // out as text only and the store's GC deleted the picture. The page cannot
    // be driven into that state without a second model, so the assembler is
    // run directly, on blocks that are equal to the stored ones and not them.
    await suite.check('kept media survives its row objects being replaced under the editor', async () => {
      const got = await page.js(async () => {
        const { editedContentBlocks } = await import('/js/pages/chat.js');
        const image = (id) => ({ type: 'image', source: { type: 'url', url: `/v1/media/${id}` } });
        const stored = () => [{ type: 'text', text: 'old' }, image('a'), image('b')];
        const old = stored();   // what the editor's entries were built from
        return {
          swapped: await editedContentBlocks(stored(), 'new', [{ kept: old[1] }, { kept: old[2] }], []),
          oneRemoved: await editedContentBlocks(stored(), 'new', [{ kept: old[2] }], []),
          twins: await editedContentBlocks([image('a'), image('a')], '', [{ kept: old[1] }], []),
          goneElsewhere: await editedContentBlocks([{ type: 'text', text: 'old' }, image('b')], 'new', [{ kept: old[1] }, { kept: old[2] }], []),
        };
      });
      const urls = (blocks) => blocks.filter((b) => b.type === 'image').map((b) => b.source.url).join(',');
      assert(urls(got.swapped) === '/v1/media/a,/v1/media/b', `both images should survive a row-object swap, got "${urls(got.swapped)}"`);
      assert(got.swapped[0].text === 'new', 'the edited text was not placed first');
      assert(urls(got.oneRemoved) === '/v1/media/b', `removing one should keep exactly the other, got "${urls(got.oneRemoved)}"`);
      assert(urls(got.twins) === '/v1/media/a', `one of two identical blocks should be kept once, got "${urls(got.twins)}"`);
      assert(urls(got.goneElsewhere) === '/v1/media/b', `a block the store no longer has must not come back, got "${urls(got.goneElsewhere)}"`);
    });

    // The same rule through the page: the ordinary round trip, where the row
    // objects are the editor's own. It cannot see the case above.
    await suite.check('saving an edited image message keeps its image', async () => {
      await selectConv(page, convC);
      await page.waitFor(() => Boolean(document.querySelector('.message--user')), 'the image row');
      await clickText(page, '.message__actions button', 'Edit');
      await page.waitFor(() => Boolean(document.querySelector('.message-edit textarea')), 'the editor');
      await page.js(() => {
        const box = document.querySelector('.message-edit textarea');
        box.value = 'what is this, edited';
        box.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await clickText(page, '.message-edit__buttons button', 'Save');
      await page.waitFor(() => !document.querySelector('.message-edit'), 'the editor to close');
      const row = (await api('GET', `/v1/conversations/${convC.id}`)).messages[0];
      const kinds = (row.content_blocks ?? []).map((b) => b.type);
      assert(kinds.includes('image'), `the image block is gone: ${JSON.stringify(kinds)}`);
      assert(JSON.stringify(row).includes('what is this, edited'), 'the edited text was not saved');
    });

    // A phone-shaped, touch-only viewport with the root font shrunk the way a
    // small iOS Text Size shrinks it. Holds two CSS rules, both of which only
    // WebKit acts on: fields stay at or above the size under which iOS zooms
    // on focus, and nothing in the chat scroll chain is left to scroll anchoring.
    await suite.check('touch: fields clear the zoom floor and the chat chain opts out of anchoring', async () => {
      await page.emulate({ width: 402, height: 874, scale: 3, touch: true });
      await page.reload();
      await page.waitFor(() => document.getElementById('main').dataset.mounted === 'chat', 'the chat page to finish mounting after the reload');
      await page.js(() => { document.documentElement.style.fontSize = '13px'; });
      const pre = await page.js(() => ({
        root: getComputedStyle(document.documentElement).fontSize,
        coarse: matchMedia('(pointer: coarse)').matches,
        hoverNone: matchMedia('(hover: none)').matches,
      }));
      assert(pre.root === '13px', `the root font size is ${pre.root}, not the shrunken size this check needs`);
      assert(pre.coarse && pre.hoverNone, 'the page does not report a touch device, so the floor rule is not in play');
      await selectConv(page, convA);
      await page.waitFor(() => Boolean(document.querySelector('.message__actions button')), 'a message row in A');
      await clickText(page, '.message__actions button', 'Edit');
      await page.waitFor(() => Boolean(document.querySelector('.message-edit textarea')), 'the editor');
      const got = await page.js(() => {
        const px = (sel) => parseFloat(getComputedStyle(document.querySelector(sel)).fontSize);
        const anchor = (el) => getComputedStyle(el).overflowAnchor;
        return {
          composer: px('.chat__composer textarea'),
          editor: px('.message-edit textarea'),
          anchors: [document.documentElement, document.getElementById('main'),
            document.querySelector('.chat__messages'), document.querySelector('.message-edit textarea'),
            document.querySelector('.chat__composer textarea')].map(anchor),
        };
      });
      assert(got.composer >= 16, `the composer is ${got.composer}px`);
      assert(got.editor >= 16, `the message editor is ${got.editor}px`);
      assert(got.anchors.every((a) => a === 'none'), `overflow-anchor reads ${got.anchors.join(', ')}`);
    });

    await suite.check('no page error across the whole run', async () => {
      assert(page.errors.length === 0, page.errors.join(' | '));
    });
  } catch (err) {
    fatal = err;
    console.error(`\nrun aborted: ${err.stack ?? err.message}`);
  } finally {
    try { await chrome?.close(); } catch { /* already gone */ }
    try { await server?.stop(); } catch { /* already gone */ }
  }

  const failed = printSummary([suite]);
  if (fatal || failed) {
    console.log(`server log kept at ${logPath}`);
    process.exit(1);
  }
  rmSync(tmp, { recursive: true, force: true });
}

main();
