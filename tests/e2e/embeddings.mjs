// Model-free browser checks for the embedding workbench, including phone layout.
import { serveV3 } from './render.mjs?lib';
import { launchBrowser } from './lib/browser.mjs';
import { assert, skip } from './lib/harness.mjs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const { server, base } = await serveV3();
const browser = await launchBrowser();
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'heylook-embedding-ui-'));
let payload;
let pending, cancelled, hold = false;
const info = { modalities: ['text','image','audio','video'], dimensions: [128,768], native_dimensions: 768,
  max_input_tokens: 8192, prompts: { SearchQuery: 'query: ' },
  video_windows: { max_seconds: 32, sampling: ['content', 'fps'], default_sampling: 'content' },
  limits: { items: 16, blocks_per_item: 32, text_chars: 131072,
    media_bytes: { image: 1024, audio: 32 * 2 ** 20, video: 256 * 2 ** 20 },
    request_media_bytes: 256 * 2 ** 20 } };
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
          data: payload.input.map((item, i) => ({
            index: i,
            embedding: i === 2 ? [0,1] : [1,0],
            ...(item.content?.some(b => b.type === 'video' && b.windows) ? {
              windows: [{ start: 0, end: 32, frame_times: [0, 8, 16, 24], embedding: [0, 1] }, { start: 32, end: 64, frame_times: [32, 40, 48, 56], embedding: [1, 0] }],
            } : {}),
          })) };
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
    assert(await page.$eval('input[aria-label="Index media folder"]', e => e.webkitdirectory === true), 'finder folder input missing webkitdirectory');
    assert(await page.$eval('input[aria-label="Choose benchmark folder"]', e => e.webkitdirectory === true), 'lab folder input missing webkitdirectory');

    // Over-limit media check (image cap stubbed to 1024 bytes; scene.png is 68 bytes, oversized.png is 2048 bytes)
    const oversized = path.join(temp, 'oversized.png');
    await fs.writeFile(oversized, Buffer.alloc(2048, 1));
    await (await page.$('input[type=file]')).uploadFile(oversized);
    assert(await page.$eval('[role="status"]', e => e.textContent) === 'oversized.png exceeds the media size limit.', 'over-limit image not rejected');

    const filename = path.join(temp, 'scene.png');
    await fs.writeFile(filename, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a8l8AAAAASUVORK5CYII=', 'base64'));
    const audio = path.join(temp, 'sound.wav'), video = path.join(temp, `clip-${width}.mp4`);
    await fs.writeFile(audio, 'audio fixture'); await fs.writeFile(video, 'video fixture');
    await (await page.$('input[type=file]')).uploadFile(filename, audio, video);
    await page.waitForSelector('select[aria-label="Video windows"]');

    // Refusal when an item with video windows also contains other blocks
    await page.select('select[aria-label="Video windows"]', 'content');
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Embed & compare').click());
    assert(await page.$eval('[role="status"]', e => e.textContent) === 'Item 1: an item with video windows must contain only that video block.', 'multi-block video windows not refused');
    await page.select('select[aria-label="Video windows"]', '');

    // Make Item 2 a single-video item with windows='content' to verify Compare windowed payload + results
    const item2 = (await page.$$('.embeddings__item'))[1];
    await (await item2.$('input[type=file]')).uploadFile(video);
    await item2.$eval('button', b => {}); // ensure item2 exists
    await page.evaluate(() => {
      const card2 = document.querySelectorAll('.embeddings__item')[1];
      const rmBtn = [...card2.querySelectorAll('.embeddings__block')[0].querySelectorAll('button')].find(b => b.textContent === 'Remove block');
      rmBtn.click();
    });
    const winSelects = await page.$$('select[aria-label="Video windows"]');
    await winSelects[1].select('content');

    await page.select('select[aria-label="Task prompt"]', 'SearchQuery');
    await page.select('select[aria-label="Embedding dimensions"]', '128');
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Embed & compare').click());
    await page.waitForSelector('.embeddings__matrix');
    assert(payload.input.length === 3 && payload.input[0].content.map(b => b.type).join() === 'text,image,audio,video', 'joint multimodal composition lost');
    assert(payload.input[1].content.length === 1 && payload.input[1].content[0].windows?.sampling === 'content', 'Compare video windows payload missing');
    assert(payload.input[0].prompt_name === 'SearchQuery' && payload.dimensions === 128, 'task/dimension ignored');
    assert(await page.$eval('.embeddings__results ol', e => e.children[0].textContent) === 'Item 2: 1.000', 'ranking incorrect');
    assert(await page.$eval('.embeddings__windows-list li', e => e.textContent) === 'Item 2 [0:32–1:04]: 1.000', 'Compare video windows ranking incorrect');
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

    // Mode 2: Media Finder (in-app IndexedDB indexing + windowed video search)
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Media Finder').click());
    await (await page.$('input[aria-label="Index media files"]')).uploadFile(video);
    await page.waitForFunction(() => {
      const t = document.querySelector('[role="status"]').textContent;
      return t.includes('Indexed 1 file') || t.includes('already indexed');
    });
    const finderStatus = await page.$eval('[role="status"]', e => e.textContent);
    if (finderStatus.includes('already indexed')) {
      skip('media file already indexed in IndexedDB');
    } else {
      assert(payload.input[0].content[0].windows?.sampling === 'content', 'video windows sampling not sent in Media Finder');
    }
    await page.$eval('input[type="search"]', e => { e.value = 'collision'; });
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Search index').click());
    await page.waitForSelector('button.embeddings__card .embeddings__card-name');
    assert(await page.evaluate(() => [...document.querySelectorAll('button.embeddings__card .embeddings__card-name')].some(e => e.textContent.includes('[0:32–1:04]'))), 'video window span missing in Media Finder');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'horizontal overflow in Media Finder');

    // Mode 3: Embedding Lab (folder input + built-in 8-case sample set)
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Embedding Lab').click());
    await page.evaluate(() => {
      const inp = document.querySelector('input[aria-label="Choose benchmark folder"]');
      const dt = new DataTransfer();
      const setJson = JSON.stringify({
        name: 'folder-check',
        cases: [{ id: 'c1', query: 'hello', positives: ['world'], negatives: ['other'] }],
      });
      dt.items.add(new File([setJson], 'set.json', { type: 'application/json' }));
      inp.files = dt.files;
      inp.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForFunction(() => document.body.textContent.includes('folder-check: 1 cases, 2 documents'));
    // Raise image limit so the generated sample PNGs (a few hundred bytes) pass if needed
    info.limits.media_bytes.image = 32 * 2 ** 20;
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Load sample set (8 cases)').click());
    await page.waitForFunction(() => document.querySelectorAll('.embeddings__lab-config-item').length >= 1);
    await page.evaluate(() => [...document.querySelectorAll('button')].find(e => e.textContent === 'Run benchmark').click());
    await page.waitForFunction(() => document.querySelectorAll('.embeddings__lab-results .embeddings__matrix').length === 2);
    const firstSummaryCell = await page.$eval('.embeddings__lab-results .embeddings__matrix tbody td', e => e.textContent);
    assert(firstSummaryCell === '100%', `unexpected Lab Recall@1 cell: ${firstSummaryCell}`);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'horizontal overflow in Embedding Lab');
    assert(!errors.length, errors.join('\n'));
    await page.close();
    info.limits.media_bytes.image = 100;
    console.log(`PASS embedding composition, ranking, invalidation and layout at ${width}px`);
  }
} finally {
  await browser.close(); server.close(); await fs.rm(temp, { recursive: true, force: true });
}
