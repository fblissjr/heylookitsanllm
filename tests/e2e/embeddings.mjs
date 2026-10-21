// Model-free browser checks for the embedding workbench, including phone layout.
import { serveV3 } from './render.mjs?lib';
import { launchBrowser } from './lib/browser.mjs';
import { assert } from './lib/harness.mjs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const { server, base } = await serveV3();
const browser = await launchBrowser();
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'heylook-embedding-ui-'));
let payload;
let pending, cancelled, hold = false;
const info = { modalities: ['text','image','audio','video'], dimensions: [128,768], native_dimensions: 768,
  max_input_tokens: 8192, prompts: { SearchQuery: 'query: ' } };
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.setViewport({ width, height: 900, isMobile: width < 500, hasTouch: width < 500 });
    await page.setRequestInterception(true);
    page.on('request', async (request) => {
      const url = new URL(request.url());
      if (!url.pathname.startsWith('/v1/')) return request.continue();
      let body = {};
      if (url.pathname === '/v1/models') body = { data: [{ id: 'embed', engine: { embedding: info } }, { id: 'chat', engine: {} }] };
      if (url.pathname === '/v1/embeddings') {
        payload = JSON.parse(request.postData());
        if (hold) { pending = request; return; }
        body = { model: 'embed', dimensions: payload.dimensions, elapsed_ms: 12, input_tokens: 9,
          data: payload.input.map((_, i) => ({ index: i, embedding: i === 2 ? [0,1] : [1,0] })) };
      }
      if (url.pathname.startsWith('/v1/requests/') && request.method() === 'DELETE') {
        cancelled = true;
        await pending.respond({ status: 409, contentType: 'application/json', body: JSON.stringify({ detail: 'Embedding request cancelled' }) });
      }
      await request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    });
    await page.goto(`${base}/#/embeddings`);
    await page.waitForFunction(() => document.querySelector('[role="status"]')?.textContent.startsWith('Ready.'));
    assert(await page.$eval('select[aria-label="Embedding model"]', e => e.options.length) === 1, 'chat model leaked into embedding selector');
    const filename = path.join(temp, 'scene.png');
    await fs.writeFile(filename, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a8l8AAAAASUVORK5CYII=', 'base64'));
    const audio = path.join(temp, 'sound.wav'), video = path.join(temp, 'clip.mp4');
    await fs.writeFile(audio, 'audio fixture'); await fs.writeFile(video, 'video fixture');
    await (await page.$('input[type=file]')).uploadFile(filename, audio, video);
    await page.select('select[aria-label="Task prompt"]', 'SearchQuery');
    await page.select('select[aria-label="Embedding dimensions"]', '128');
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Embed & compare').click());
    await page.waitForSelector('.embeddings__matrix');
    assert(payload.input.length === 3 && payload.input[0].content.map(b => b.type).join() === 'text,image,audio,video', 'joint multimodal composition lost');
    assert(payload.input[0].prompt_name === 'SearchQuery' && payload.dimensions === 128, 'task/dimension ignored');
    assert(await page.$eval('.embeddings__results ol', e => e.children[0].textContent) === 'Item 2: 1.000', 'ranking incorrect');
    const exported = await page.evaluate(async () => {
      let blob;
      URL.createObjectURL = value => { blob = value; return 'blob:embedding-test'; };
      HTMLAnchorElement.prototype.click = () => {};
      [...document.querySelectorAll('button')].find(e => e.textContent === 'Download vectors').click();
      return JSON.parse(await blob.text());
    });
    assert(exported.labels.length === 3 && exported.data.length === 3 && exported.dimensions === 128, 'export lost labels or vectors');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'horizontal page overflow');
    await page.screenshot({ path: `/tmp/heylook-embeddings-${width}.png`, fullPage: true });
    await page.$eval('textarea', e => { e.value = 'changed'; e.dispatchEvent(new Event('input', { bubbles: true })); });
    assert(await page.$('.embeddings__matrix') === null, 'stale result survives editing');
    hold = true; cancelled = false; pending = null;
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Embed & compare').click());
    await page.waitForFunction(() => document.querySelector('[role="status"]').textContent.startsWith('Embedding…'));
    // Let the request interception reach its pending state before clicking Stop.
    while (!pending) await new Promise(resolve => setTimeout(resolve, 10));
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Stop').click());
    await page.waitForFunction(() => !document.querySelector('fieldset').disabled);
    assert(cancelled && await page.$('.embeddings__matrix') === null, 'cancel did not reach the server or retained results');
    hold = false;
    assert(!errors.length, errors.join('\n'));
    await page.close();
    console.log(`PASS embedding composition, ranking, invalidation and layout at ${width}px`);
  }
} finally {
  await browser.close(); server.close(); await fs.rm(temp, { recursive: true, force: true });
}
