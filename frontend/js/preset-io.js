// Portable preset import/export (JSON & Markdown), before/after diffing,
// and a tiny zero-dependency STORE ZIP builder for multi-file Markdown export
// on browsers without showDirectoryPicker.
//
// Format parity: mirrors src/heylook_llm/preset_io.py so files exported from
// the UI and from `uv run heylook presets` interchange losslessly.
// Pure ESM with no DOM globals at import time so Node unit tests import it directly.

const YAML_SPECIAL_START = new Set([...'?:,-[]{}#&*!|>\'"%@`']);
const YAML_BOOL_OR_NULL = new Set(['true', 'false', 'null', 'yes', 'no', 'on', 'off', '~']);
const INT_RE = /^-?\d+$/;
const FLOAT_RE = /^-?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$/;

export function normalizePrompt(value) {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'string') {
    throw new Error("Preset 'system_prompt' must be a string or null");
  }
  const cleaned = value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  return cleaned.trim() ? cleaned : null;
}

export function normalizeParams(value) {
  if (value === null || value === undefined) return {};
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error("Preset 'params' must be an object");
  }
  const out = {};
  for (const [k, v] of Object.entries(value)) {
    if (v !== null && v !== undefined) out[k] = v;
  }
  return out;
}

export function normalizePreset(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('Preset entry must be an object');
  }
  if (typeof raw.name !== 'string' || !raw.name.trim()) {
    throw new Error("Preset 'name' must be a non-empty string");
  }
  return {
    name: raw.name.trim(),
    system_prompt: normalizePrompt(raw.system_prompt ?? null),
    params: normalizeParams(raw.params ?? {}),
  };
}

export function slugifyPresetName(name) {
  const slug = String(name ?? '')
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^[-.]+|[-.]+$/g, '');
  return slug || 'preset';
}

export function uniqueMarkdownFilenames(presets) {
  const used = new Set();
  return presets.map((p) => {
    const base = slugifyPresetName(p.name);
    let candidate = `${base}.md`;
    let key = candidate.toLowerCase();
    if (used.has(key)) {
      let n = 2;
      while (used.has(`${base}-${n}.md`.toLowerCase())) n += 1;
      candidate = `${base}-${n}.md`;
      key = candidate.toLowerCase();
    }
    used.add(key);
    return candidate;
  });
}

// ---------------------------------------------------------------------------
// JSON serialization / parsing
// ---------------------------------------------------------------------------

export function serializePresetJson(preset) {
  const rec = normalizePreset(preset);
  return `${JSON.stringify(rec, null, 2)}\n`;
}

export function serializePresetsJson(presets) {
  const entries = presets.map((p) => normalizePreset(p));
  return `${JSON.stringify({ presets: entries }, null, 2)}\n`;
}

export function parsePresetsJson(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch (err) {
    throw new Error(`Invalid JSON: ${err.message}`);
  }
  let rawList;
  if (Array.isArray(data)) {
    rawList = data;
  } else if (data && typeof data === 'object') {
    if (Array.isArray(data.presets)) rawList = data.presets;
    else if ('name' in data) rawList = [data];
    else throw new Error("JSON must be a preset object with 'name', a list, or {\"presets\": [...]}");
  } else {
    throw new Error('JSON root must be an object or array');
  }
  return rawList.map((item) => normalizePreset(item));
}

// ---------------------------------------------------------------------------
// Markdown serialization / parsing (YAML-style frontmatter + prompt body)
// ---------------------------------------------------------------------------

function formatYamlScalar(value) {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'number') return String(value);
  const text = String(value);
  const needsQuote = (
    !text
    || text !== text.trim()
    || YAML_SPECIAL_START.has(text[0])
    || text.includes(': ')
    || text.endsWith(':')
    || text.includes(' #')
    || /[\n\r\t"\\]/.test(text)
    || YAML_BOOL_OR_NULL.has(text.toLowerCase())
    || INT_RE.test(text)
    || FLOAT_RE.test(text)
  );
  return needsQuote ? JSON.stringify(text) : text;
}

function parseYamlScalar(raw, { asString = false } = {}) {
  const s = raw.trim();
  if (!s) return asString ? '' : null;
  if (s.startsWith('"')) {
    try {
      const val = JSON.parse(s);
      return asString ? String(val) : val;
    } catch (err) {
      throw new Error(`Invalid quoted string in frontmatter: ${s} (${err.message})`);
    }
  }
  if (s.length >= 2 && s.startsWith("'") && s.endsWith("'")) {
    return s.slice(1, -1).replace(/''/g, "'");
  }
  if (asString) return s;
  const low = s.toLowerCase();
  if (low === 'true') return true;
  if (low === 'false') return false;
  if (low === 'null' || low === '~') return null;
  if (INT_RE.test(s)) return Number.parseInt(s, 10);
  if (FLOAT_RE.test(s)) return Number.parseFloat(s);
  return s;
}

export function serializePresetMarkdown(preset) {
  const rec = normalizePreset(preset);
  const lines = ['---', `name: ${formatYamlScalar(rec.name)}`];
  const paramKeys = Object.keys(rec.params).sort();
  if (paramKeys.length) {
    lines.push('params:');
    for (const key of paramKeys) {
      const val = rec.params[key];
      if (val !== null && val !== undefined) {
        lines.push(`  ${key}: ${formatYamlScalar(val)}`);
      }
    }
  }
  lines.push('---');
  let out = `${lines.join('\n')}\n`;
  if (rec.system_prompt) {
    out += `\n${rec.system_prompt}`;
    if (!rec.system_prompt.endsWith('\n')) out += '\n';
  }
  return out;
}

export function parsePresetMarkdown(text, fallbackName = null) {
  const cleaned = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = cleaned.split('\n');

  let name = null;
  const params = {};
  let bodyText = cleaned;

  if (lines.length && lines[0].trim() === '---') {
    let endIdx = -1;
    for (let i = 1; i < lines.length; i += 1) {
      if (lines[i].trim() === '---') {
        endIdx = i;
        break;
      }
    }
    if (endIdx < 0) {
      throw new Error("Unclosed frontmatter block (missing closing '---')");
    }

    let inParams = false;
    for (let i = 1; i < endIdx; i += 1) {
      const line = lines[i];
      if (!line.trim() || line.trimStart().startsWith('#')) continue;
      const indent = line.length - line.replace(/^ +/, '').length;
      if (indent === 0) {
        inParams = false;
        const colon = line.indexOf(':');
        if (colon < 0) throw new Error(`Malformed frontmatter line: ${line}`);
        const key = line.slice(0, colon).trim();
        const rest = line.slice(colon + 1).trim();
        if (key === 'name') {
          name = parseYamlScalar(rest, { asString: true });
        } else if (key === 'params') {
          if (!rest) {
            inParams = true;
          } else if (rest.startsWith('{')) {
            let parsed;
            try { parsed = JSON.parse(rest); }
            catch (err) { throw new Error(`Invalid inline params JSON: ${err.message}`); }
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
              throw new Error("Frontmatter 'params' must be an object");
            }
            Object.assign(params, parsed);
          } else {
            throw new Error(`Unexpected value for 'params': ${rest}`);
          }
        }
      } else if (inParams) {
        const colon = line.indexOf(':');
        if (colon < 0) throw new Error(`Malformed param line: ${line}`);
        const pKey = line.slice(0, colon).trim();
        const pVal = parseYamlScalar(line.slice(colon + 1));
        if (pVal !== null && pVal !== undefined) params[pKey] = pVal;
      } else {
        throw new Error(`Unexpected indented frontmatter line: ${line}`);
      }
    }

    const restLines = lines.slice(endIdx + 1);
    if (restLines.length && restLines[0] === '') restLines.shift();
    if (restLines.length && restLines[restLines.length - 1] === '') restLines.pop();
    bodyText = restLines.join('\n');
  } else {
    bodyText = cleaned.replace(/\n+$/, '');
  }

  const resolvedName = String(name || fallbackName || '').trim();
  if (!resolvedName) {
    throw new Error("Markdown preset is missing 'name' in frontmatter");
  }
  return normalizePreset({
    name: resolvedName,
    system_prompt: bodyText || null,
    params,
  });
}

function fileStem(filename) {
  const base = String(filename ?? '').split(/[\\/]/).pop() || '';
  return base.replace(/\.[^.]+$/, '');
}

export function parsePresetFiles(fileEntries) {
  const records = [];
  const seen = new Map();
  for (const entry of fileEntries) {
    const filename = entry.name || 'file';
    const lower = filename.toLowerCase();
    const text = entry.text ?? '';
    let batch;
    try {
      if (lower.endsWith('.json')) {
        batch = parsePresetsJson(text);
      } else if (lower.endsWith('.md') || lower.endsWith('.markdown')) {
        batch = [parsePresetMarkdown(text, fileStem(filename))];
      } else {
        const trimmed = text.replace(/^\uFEFF/, '').trimStart();
        if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
          batch = parsePresetsJson(text);
        } else {
          batch = [parsePresetMarkdown(text, fileStem(filename))];
        }
      }
    } catch (err) {
      throw new Error(`${filename}: ${err.message}`);
    }
    for (const rec of batch) {
      if (seen.has(rec.name)) {
        throw new Error(
          `Duplicate preset name "${rec.name}" in selected files (${seen.get(rec.name)} and ${filename})`,
        );
      }
      seen.set(rec.name, filename);
      records.push(rec);
    }
  }
  if (!records.length) {
    throw new Error('No presets found in selected file(s)');
  }
  return records;
}

// ---------------------------------------------------------------------------
// Diffing & import planning
// ---------------------------------------------------------------------------

function paramValuesEqual(a, b) {
  if (typeof a === 'boolean' || typeof b === 'boolean') return a === b;
  return a === b;
}

// Unified line diff with context lines. Uses exact LCS for normal prompts and
// falls back to a full replace block if the DP table would be huge.
export function unifiedLineDiff(oldText, newText, context = 3) {
  const a = oldText ? oldText.split('\n') : [];
  const b = newText ? newText.split('\n') : [];
  const m = a.length;
  const n = b.length;

  let ops;
  if (m === 0 && n === 0) {
    return [];
  }
  if (m === 0) {
    ops = b.map((line) => ({ type: 'add', text: line }));
  } else if (n === 0) {
    ops = a.map((line) => ({ type: 'del', text: line }));
  } else if (m * n > 400_000) {
    ops = [
      ...a.map((line) => ({ type: 'del', text: line })),
      ...b.map((line) => ({ type: 'add', text: line })),
    ];
  } else {
    const dp = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
    for (let i = m - 1; i >= 0; i -= 1) {
      for (let j = n - 1; j >= 0; j -= 1) {
        dp[i][j] = a[i] === b[j]
          ? dp[i + 1][j + 1] + 1
          : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    ops = [];
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
      if (a[i] === b[j]) {
        ops.push({ type: 'ctx', text: a[i], oldLine: i + 1, newLine: j + 1 });
        i += 1;
        j += 1;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        ops.push({ type: 'del', text: a[i], oldLine: i + 1 });
        i += 1;
      } else {
        ops.push({ type: 'add', text: b[j], newLine: j + 1 });
        j += 1;
      }
    }
    while (i < m) { ops.push({ type: 'del', text: a[i], oldLine: i + 1 }); i += 1; }
    while (j < n) { ops.push({ type: 'add', text: b[j], newLine: j + 1 }); j += 1; }
  }

  // Collapse unchanged context outside +/- `context` lines of a change.
  const keep = new Uint8Array(ops.length);
  for (let i = 0; i < ops.length; i += 1) {
    if (ops[i].type !== 'ctx') {
      const lo = Math.max(0, i - context);
      const hi = Math.min(ops.length - 1, i + context);
      for (let k = lo; k <= hi; k += 1) keep[k] = 1;
    }
  }

  const out = [];
  let inSkip = false;
  for (let i = 0; i < ops.length; i += 1) {
    if (keep[i]) {
      if (inSkip) {
        out.push({ type: 'hunk', text: '@@ ... @@' });
        inSkip = false;
      }
      out.push(ops[i]);
    } else if (out.length > 0) {
      inSkip = true;
    }
  }
  return out;
}

export function diffPreset(existing, incoming) {
  const oldParams = normalizeParams(existing.params ?? {});
  const newParams = normalizeParams(incoming.params ?? {});
  const keys = [...new Set([...Object.keys(oldParams), ...Object.keys(newParams)])].sort();

  const paramDiffs = [];
  for (const key of keys) {
    const hasOld = Object.prototype.hasOwnProperty.call(oldParams, key);
    const hasNew = Object.prototype.hasOwnProperty.call(newParams, key);
    if (!hasOld && hasNew) {
      paramDiffs.push({ key, kind: 'added', before: undefined, after: newParams[key] });
    } else if (hasOld && !hasNew) {
      paramDiffs.push({ key, kind: 'removed', before: oldParams[key], after: undefined });
    } else if (!paramValuesEqual(oldParams[key], newParams[key])) {
      paramDiffs.push({ key, kind: 'changed', before: oldParams[key], after: newParams[key] });
    }
  }

  const oldPrompt = normalizePrompt(existing.system_prompt ?? null);
  const newPrompt = normalizePrompt(incoming.system_prompt ?? null);
  const promptChanged = oldPrompt !== newPrompt;
  const promptDiff = promptChanged ? unifiedLineDiff(oldPrompt || '', newPrompt || '') : [];

  return {
    changed: paramDiffs.length > 0 || promptChanged,
    paramDiffs,
    promptChanged,
    promptDiff,
  };
}

export function planImport(incomingList, existingPresets) {
  const byName = new Map(existingPresets.map((p) => [p.name, p]));
  const items = incomingList.map((incoming) => {
    const existing = byName.get(incoming.name) ?? null;
    if (!existing) {
      return {
        incoming,
        existing: null,
        changed: true,
        paramDiffs: [],
        promptChanged: false,
        promptDiff: [],
      };
    }
    const diff = diffPreset(existing, incoming);
    return { incoming, existing, ...diff };
  });
  return {
    items,
    newItems: items.filter((it) => !it.existing),
    existingItems: items.filter((it) => Boolean(it.existing)),
    changedExistingItems: items.filter((it) => it.existing && it.changed),
    unchangedExistingItems: items.filter((it) => it.existing && !it.changed),
  };
}

// ---------------------------------------------------------------------------
// Minimal uncompressed (STORE) ZIP builder for multi-file Markdown export
// ---------------------------------------------------------------------------

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let c = i;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    c = CRC32_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

export function buildZip(files) {
  const encoder = new TextEncoder();
  const localParts = [];
  const centralParts = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const dataBytes = typeof file.text === 'string' ? encoder.encode(file.text) : file.bytes;
    const crc = crc32(dataBytes);

    const local = new Uint8Array(30 + nameBytes.length + dataBytes.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);  // local file header signature
    lv.setUint16(4, 20, true);          // version needed
    lv.setUint16(6, 0x0800, true);      // UTF-8 filename flag
    lv.setUint16(8, 0, true);           // method: STORE
    lv.setUint16(10, 0, true);          // mod time
    lv.setUint16(12, 0, true);          // mod date
    lv.setUint32(14, crc, true);
    lv.setUint32(18, dataBytes.length, true);
    lv.setUint32(22, dataBytes.length, true);
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true);
    local.set(nameBytes, 30);
    local.set(dataBytes, 30 + nameBytes.length);
    localParts.push(local);

    const central = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(central.buffer);
    cv.setUint32(0, 0x02014b50, true);  // central dir signature
    cv.setUint16(4, 20, true);          // version made by
    cv.setUint16(6, 20, true);          // version needed
    cv.setUint16(8, 0x0800, true);      // UTF-8 filename flag
    cv.setUint16(10, 0, true);          // method: STORE
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, dataBytes.length, true);
    cv.setUint32(24, dataBytes.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    central.set(nameBytes, 46);
    centralParts.push(central);

    offset += local.length;
  }

  const centralSize = centralParts.reduce((sum, p) => sum + p.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  ev.setUint16(20, 0, true);

  const total = offset + centralSize + eocd.length;
  const out = new Uint8Array(total);
  let cursor = 0;
  for (const part of [...localParts, ...centralParts, eocd]) {
    out.set(part, cursor);
    cursor += part.length;
  }
  return out;
}
