// The engine contract (plan W13; server side: src/heylook_llm/providers/contract.py).
// /v1/models and /v1/admin/models carry ONE `engine` object per model, the
// same shape whatever engine runs it. Every leaf is a Fact
// {value, provenance, source}; settings are {value, configured, auto,
// reason, provenance, effect}. Read values through readFact so no page
// grows its own notion of where a value lives, and render through
// renderEngine so a slot a later workstream fills (cache, thinking, image,
// steering) appears with no page-specific code.

import { createEl, formatTokens } from './utils.js';

export function readFact(fact) {
  return fact?.value ?? null;
}

// The context ceiling the model's files declare (gguf header / MLX config.json).
export function contextCeiling(row) {
  return readFact(row?.engine?.context?.length);
}

// What the resident process was sized to (gguf /props at ready); null when
// unloaded, and not applicable on MLX.
export function contextRunning(row) {
  return readFact(row?.engine?.context?.running);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Provenance -> one visual convention, used everywhere: derived plain,
// observed marked live, configured accented, unknown / not applicable muted
// with the reason spelled out. The words carry it too, never colour alone.
const PROVENANCE_LABEL = {
  derived: 'derived',
  configured: 'set',
  observed: 'live',
  observed_cached: 'cached',
  unknown: 'unknown',
  not_applicable: 'n/a',
};

// Group order for settings, by when a change takes effect
// (config.EFFECT_CLASSES); load decisions with no config field carry null.
const EFFECT_GROUPS = [
  [null, 'Load decisions'],
  ['requires_reload', 'Applies at the next load'],
  ['load_time_only', 'Fixed at load'],
  ['applies_live', 'Applies live'],
  ['per_request', 'Per-request defaults'],
  ['descriptive', 'Descriptive'],
];

const TOP_FACTS = new Set(['runtime', 'context', 'template', 'settings']);

function formatValue(value) {
  if (value === null || value === undefined) return '—';
  if (value === true) return 'on';
  if (value === false) return 'off';
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(', ') : '—';
  if (typeof value === 'object') return JSON.stringify(value);
  if (typeof value === 'string' && /^[0-9a-f]{64}$/.test(value)) return value.slice(0, 12);
  return String(value);
}

function provenanceTag(provenance) {
  return createEl('span', { class: `engine-prov engine-prov--${provenance || 'unknown'}` },
    [PROVENANCE_LABEL[provenance] ?? provenance ?? 'unknown']);
}

// One labelled value with its provenance, and its reason one tap away. A
// native <details> is the reveal: it works by touch and keyboard alike, which
// a title tooltip does not (DESIGN.md §7, no hover-only affordances).
function factRow(label, value, provenance, reason, extra = null) {
  const summary = createEl('summary', { class: 'engine-row__summary' }, [
    createEl('span', { class: 'engine-row__label' }, [label]),
    createEl('span', { class: 'engine-row__value' }, [formatValue(value)]),
    provenanceTag(provenance),
  ]);
  const body = [createEl('div', { class: 'engine-row__reason muted small' }, [reason || 'no reason given'])];
  if (extra) body.push(extra);
  return createEl('details', { class: `engine-row engine-row--${provenance || 'unknown'}` },
    [summary, ...body]);
}

function factSection(title, rows) {
  if (!rows.length) return null;
  return createEl('section', { class: 'engine-section' }, [
    createEl('h3', { class: 'engine-section__title small' }, [title]),
    ...rows,
  ]);
}

function isFact(node) {
  return node && typeof node === 'object' && 'provenance' in node && 'value' in node;
}

// A slot's facts, flattened one level: {length: Fact, running: Fact} ->
// "length", "running". Works for any later slot shaped the same way.
function slotRows(slot) {
  const rows = [];
  for (const [key, node] of Object.entries(slot)) {
    if (isFact(node)) rows.push(factRow(key.replace(/_/g, ' '), node.value, node.provenance, node.source));
    else if (node && typeof node === 'object') {
      for (const [sub, leaf] of Object.entries(node)) {
        if (isFact(leaf)) rows.push(factRow(`${key} ${sub}`.replace(/_/g, ' '), leaf.value, leaf.provenance, leaf.source));
      }
    } else if (node != null) {
      rows.push(factRow(key.replace(/_/g, ' '), node, 'observed', null));
    }
  }
  return rows;
}

// Speculative decoding in a word, or null when the model has no drafter (no
// noise on models that cannot draft). `in_force` is the running process's
// answer: true drafting, false found but not in use (the fit check dropped
// it, or the build could not load it), null not loaded yet.
function specState(engine) {
  const spec = engine?.speculative;
  const drafter = readFact(spec?.drafter);
  if (!drafter) return null;
  const inForce = readFact(spec?.in_force);
  const word = inForce === true ? 'on' : inForce === false ? 'not in use' : 'ready';
  const reason = [spec.drafter?.source ? `${drafter}: ${spec.drafter.source}` : drafter,
    readFact(spec.type) ? `type ${readFact(spec.type)}` : null,
    spec.in_force?.source].filter(Boolean).join('. ');
  return { word, provenance: spec.in_force?.provenance, reason };
}

// A one-line summary for a collapsed panel or a chip.
export function engineSummary(engine) {
  if (!engine) return '';
  const parts = [readFact(engine.runtime)];
  const ceiling = readFact(engine.context?.length);
  const running = readFact(engine.context?.running);
  if (running) parts.push(`ctx ${formatTokens(running)} of ${ceiling ? formatTokens(ceiling) : '?'}`);
  else if (ceiling) parts.push(`ctx ${formatTokens(ceiling)}`);
  const origin = readFact(engine.template?.origin);
  if (origin) parts.push(`template ${origin}`);
  const spec = specState(engine);
  if (spec) parts.push(`spec decode ${spec.word}`);
  return parts.filter(Boolean).join(' · ');
}

// The full panel: facts, every later slot that carries data, and every
// setting grouped by when a change takes effect.
//
// `fields` is the provider's /v1/admin/model-options field list, when known.
// A schema ui:"hidden" field means "no editor offers this", not "nobody sees
// it": it is omitted only when NOT configured, so a hand-set server_binary or
// port can never be invisible (owner decision, W13 remaining item 4).
export function renderEngine(engine, { fields = null } = {}) {
  if (!engine) {
    return createEl('div', { class: 'muted small' }, ['This server reports no engine description.']);
  }
  const hidden = new Set((fields || []).filter((f) => f.ui === 'hidden').map((f) => f.name));
  const sections = [];

  sections.push(factSection('Engine', [
    factRow('runtime', engine.runtime?.value, engine.runtime?.provenance, engine.runtime?.source),
    ...slotRows({ context: engine.context || {} }),
  ]));
  sections.push(factSection('Chat template', slotRows(engine.template || {})));

  // Later workstreams' slots (cache, thinking, image, steering): shown as
  // soon as the server fills one, with no code here naming it.
  for (const [key, slot] of Object.entries(engine)) {
    if (TOP_FACTS.has(key) || slot == null || typeof slot !== 'object') continue;
    sections.push(factSection(key.replace(/_/g, ' '), slotRows(slot)));
  }

  const settings = engine.settings || {};
  const byEffect = new Map();
  for (const [name, s] of Object.entries(settings)) {
    if (hidden.has(name) && s.provenance !== 'configured') continue;
    const key = s.effect ?? null;
    if (!byEffect.has(key)) byEffect.set(key, []);
    byEffect.get(key).push([name, s]);
  }
  const known = new Set(EFFECT_GROUPS.map(([k]) => k));
  const groups = [...EFFECT_GROUPS, ...[...byEffect.keys()].filter((k) => !known.has(k)).map((k) => [k, k])];
  for (const [effect, title] of groups) {
    const entries = byEffect.get(effect);
    if (!entries?.length) continue;
    entries.sort(([a], [b]) => a.localeCompare(b));
    sections.push(factSection(title, entries.map(([name, s]) => {
      const auto = s.provenance === 'configured'
        ? createEl('div', { class: 'engine-row__auto muted small' }, [`auto would be ${formatValue(s.auto)}`])
        : null;
      return factRow(name, s.value, s.provenance, s.reason, auto);
    })));
  }

  return createEl('div', { class: 'engine-panel__body' }, sections.filter(Boolean));
}

// The chat bar's compact view: what runs this model, its context, and the
// template in force, with the full panel one link away. No settings: the
// chat sampler panel owns the per-request ones, and everything else lives on
// the models page. The cache line is the one reuse fact each engine leads
// with: gguf's reuse class, MLX's text reuse. The spec decode line appears
// only for a model with a drafter.
export function renderEngineCompact(engine) {
  if (!engine) {
    return createEl('div', { class: 'muted small' }, ['This server reports no engine description.']);
  }
  const reuse = engine.cache?.reuse_class ?? engine.cache?.text_reuse;
  const rows = [
    factRow('runtime', engine.runtime?.value, engine.runtime?.provenance, engine.runtime?.source),
    ...slotRows({ context: engine.context || {} }),
    factRow('template', engine.template?.origin?.value, engine.template?.origin?.provenance,
      engine.template?.origin?.source),
  ];
  if (reuse) rows.push(factRow('cache reuse', reuse.value, reuse.provenance, reuse.source));
  const spec = specState(engine);
  if (spec) rows.push(factRow('spec decode', spec.word, spec.provenance, spec.reason));
  return createEl('div', { class: 'engine-panel__body engine-panel__body--compact' }, [
    ...rows,
    createEl('a', { href: '#/models', class: 'small' }, ['Every setting and why: Models page']),
  ]);
}
