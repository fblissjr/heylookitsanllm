// Unit tests for frontend/js/preset-io.js (JSON/Markdown round-trip, diffing,
// bulk import planning, and STORE ZIP builder). Run with:
//   node --test tests/e2e/preset-io.test.mjs

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildZip,
  diffPreset,
  parsePresetFiles,
  parsePresetMarkdown,
  parsePresetsJson,
  planImport,
  serializePresetJson,
  serializePresetMarkdown,
  serializePresetsJson,
  uniqueMarkdownFilenames,
} from '../../frontend/js/preset-io.js';

test('single and bulk JSON round-trip', () => {
  const pirate = {
    name: 'Pirate',
    system_prompt: 'Talk like a pirate.\nAhoy!',
    params: { temperature: 1.2, top_k: 40, enable_thinking: true },
  };
  assert.deepEqual(parsePresetsJson(serializePresetJson(pirate)), [pirate]);

  const bare = { name: 'Deterministic', system_prompt: null, params: { temperature: 0, seed: 7 } };
  const bulk = serializePresetsJson([pirate, bare]);
  assert.deepEqual(parsePresetsJson(bulk), [pirate, bare]);
  assert.deepEqual(parsePresetsJson(JSON.stringify([pirate, bare])), [pirate, bare]);
});

test('markdown frontmatter round-trip including horizontal rules and special names', () => {
  const cases = [
    {
      name: 'Code Reviewer: Strict #1',
      system_prompt: 'Intro line\n\n---\n\nAfter horizontal rule.',
      params: { temperature: 0.3, top_k: 20, enable_thinking: true, reasoning_effort: 'medium' },
    },
    {
      name: 'Settings Only',
      system_prompt: null,
      params: { temperature: 0, seed: 42 },
    },
    {
      name: 'Prompt Only',
      system_prompt: 'Answer in haiku.',
      params: {},
    },
  ];
  for (const preset of cases) {
    const md = serializePresetMarkdown(preset);
    assert.deepEqual(parsePresetMarkdown(md), preset);
  }
});

test('parsePresetFiles supports multiple markdown files and rejects duplicate names', () => {
  const files = [
    { name: 'a.md', text: '---\nname: Alpha\nparams:\n  temperature: 0.4\n---\n\nHello A\n' },
    { name: 'b.md', text: 'Plain body without frontmatter.\n' },
  ];
  const parsed = parsePresetFiles(files);
  assert.deepEqual(parsed, [
    { name: 'Alpha', system_prompt: 'Hello A', params: { temperature: 0.4 } },
    { name: 'b', system_prompt: 'Plain body without frontmatter.', params: {} },
  ]);

  assert.throws(
    () => parsePresetFiles([
      { name: 'one.md', text: '---\nname: Dup\n---\n\nOne\n' },
      { name: 'two.md', text: '---\nname: Dup\n---\n\nTwo\n' },
    ]),
    /Duplicate preset name/,
  );
});

test('uniqueMarkdownFilenames deduplicates slug collisions', () => {
  const out = uniqueMarkdownFilenames([
    { name: 'Foo/Bar' },
    { name: 'Foo:Bar' },
    { name: 'Foo Bar' },
  ]);
  assert.deepEqual(out, ['Foo-Bar.md', 'Foo-Bar-2.md', 'Foo-Bar-3.md']);
});

test('diffPreset and planImport classify new, changed, and unchanged presets', () => {
  const existing = [
    { id: '1', name: 'Pirate', system_prompt: 'Old line\nSame line', params: { temperature: 0.7, top_k: 40 } },
    { id: '2', name: 'Exact', system_prompt: 'Keep', params: { temperature: 0.5 } },
  ];
  const incoming = [
    { name: 'Pirate', system_prompt: 'New line\nSame line', params: { temperature: 1.2, enable_thinking: true } },
    { name: 'Exact', system_prompt: 'Keep', params: { temperature: 0.5 } },
    { name: 'Fresh', system_prompt: 'Hi', params: {} },
  ];

  const plan = planImport(incoming, existing);
  assert.deepEqual(plan.newItems.map((i) => i.incoming.name), ['Fresh']);
  assert.deepEqual(plan.changedExistingItems.map((i) => i.incoming.name), ['Pirate']);
  assert.deepEqual(plan.unchangedExistingItems.map((i) => i.incoming.name), ['Exact']);

  const pirateDiff = diffPreset(existing[0], incoming[0]);
  assert.equal(pirateDiff.changed, true);
  assert.equal(pirateDiff.promptChanged, true);
  assert.deepEqual(
    pirateDiff.paramDiffs.map((d) => [d.key, d.kind]),
    [['enable_thinking', 'added'], ['temperature', 'changed'], ['top_k', 'removed']],
  );
  assert.ok(pirateDiff.promptDiff.some((l) => l.type === 'del' && l.text === 'Old line'));
  assert.ok(pirateDiff.promptDiff.some((l) => l.type === 'add' && l.text === 'New line'));
});

test('buildZip produces a valid ZIP header and stores every markdown file', () => {
  const bytes = buildZip([
    { name: 'Alpha.md', text: '---\nname: Alpha\n---\n\nHi\n' },
    { name: 'Beta.md', text: '---\nname: Beta\n---\n' },
  ]);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // Local file header magic PK\x03\x04 and EOCD magic PK\x05\x06 at tail
  assert.equal(view.getUint32(0, true), 0x04034b50);
  assert.equal(view.getUint32(bytes.length - 22, true), 0x06054b50);
  assert.equal(view.getUint16(bytes.length - 12, true), 2);
});
